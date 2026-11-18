//! mX on OpenGL and OpenGL ES: the bound tier (see `docs/BOUND_TIER.md`).
//!
//! The GPU has no bindless here. The heap lives on the CPU (indices, deferred reuse, the same
//! API as on the other backends); shaders are lowered so that each heap access indexed by an
//! immediate becomes a fixed binding, filled at draw time from the immediates
//! (`mx_shader::bound`). Root groups 0..=6 map to GL binding points as wgpu-hal maps them.
//!
//! There is one context per device, behind a lock and made current for each operation. Command
//! encoders record commands; `Queue::submit` replays them on the context, and a fence after
//! each submission tells `Device::poll` when it is done.
//!
//! Contexts come from EGL (`libEGL` loaded at run time): Linux (X11, Wayland, headless),
//! Android, and Windows through ANGLE's `libEGL.dll` over Direct3D 11.

mod bundle;
mod caps;
mod command;
mod context;
mod conv;
mod device;
mod heap;
mod pipeline;
mod query;
mod resource;
mod rt;
mod surface;

pub use bundle::{RenderBundle, RenderBundleEncoder};
pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use device::{Adapter, Device, DeviceDescriptor, Instance, Queue, SubmissionIndex};
pub use glow;
pub use pipeline::{ComputePipeline, RenderPipeline, ShaderModule, ShaderModuleDescriptor};
pub use query::QuerySet;
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, BufferView, IndirectDraws, Sampler, Texture, TextureView};
pub use rt::{Blas, Tlas};
pub use surface::{Surface, SurfaceTexture};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no OpenGL device mX can use (it needs EGL with OpenGL ES 3.0 or OpenGL 3.3)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: OpenGL could not compile the shader: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: OpenGL could not link the pipeline: {message}")]
    Pipeline { label: String, message: String },
    #[error("the GPU reported an error: {0}")]
    Device(String),
    /// The GPU is gone (removed, reset, hung): nothing made on this device works any more;
    /// the device has to be made again.
    #[error("the GPU was lost: {0}")]
    DeviceLost(String),
    #[error("surface: {0}")]
    Surface(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A GL object's creation failed (glow returns its error as a string).
pub(crate) fn gl_err(what: &str, e: String) -> Error {
    Error::Device(format!("{what}: {e}"))
}

/// Objects compare and hash by identity (the same object, not equal contents), as wgpu's do.
macro_rules! identity {
    ($($t:ty => |$x:ident| $e:expr;)*) => {$(
        impl PartialEq for $t {
            fn eq(&self, o: &Self) -> bool {
                let a = { let $x = self; $e };
                let b = { let $x = o; $e };
                a == b
            }
        }
        impl Eq for $t {}
        impl std::hash::Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
                let $x = self;
                ($e).hash(h)
            }
        }
    )*};
}

use std::sync::Arc;

identity! {
    Buffer => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Texture => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    TextureView => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Sampler => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    BindGroup => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Blas => |_x| 0usize;
    Device => |x| Arc::as_ptr(&x.shared) as *const u8 as usize;
}
