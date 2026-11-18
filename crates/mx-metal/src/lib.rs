//! mX on Metal 3.
//!
//! The bindless heap is two argument buffers of `MTLResourceID`s (textures, samplers); naga
//! compiles WGSL `binding_array`s into pointers to them. See `docs/ARCHITECTURE.md`.

#![cfg(target_vendor = "apple")]

mod bundle;
mod command;
mod conv;
mod device;
mod heap;
mod pipeline;
mod query;
mod resource;
mod rt;
mod surface;
mod zero;

pub use bundle::{RenderBundle, RenderBundleEncoder};
pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use device::{Adapter, Device, DeviceDescriptor, Instance, Queue, SubmissionIndex};
pub use pipeline::{ComputePipeline, RenderPipeline, ShaderModule, ShaderModuleDescriptor};
pub use query::QuerySet;
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, BufferView, IndirectDraws, Sampler, Texture, TextureView};
pub use rt::{Blas, Tlas};
pub use surface::{Surface, SurfaceTexture};

use std::ops::Deref;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no Metal 3 device (mX needs macOS 13 / iOS 16 and a Metal 3 GPU)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: Metal could not compile the shader: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: Metal could not build the pipeline: {message}")]
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

/// A Metal object shared between threads. Metal's devices, queues, resources and pipeline
/// states are documented thread-safe; objc2 leaves that to the user.
pub(crate) struct Mt<T>(pub T);

// SAFETY: only used for Metal objects Apple documents as safe to use from any thread.
unsafe impl<T> Send for Mt<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Mt<T> {}

impl<T> Deref for Mt<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Clone> Clone for Mt<T> {
    fn clone(&self) -> Self {
        Mt(self.0.clone())
    }
}

pub(crate) type Po<T> = objc2::rc::Retained<objc2::runtime::ProtocolObject<T>>;

pub(crate) fn ns(s: &str) -> objc2::rc::Retained<objc2_foundation::NSString> {
    objc2_foundation::NSString::from_str(s)
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
    Blas => |x| Arc::as_ptr(&x.0) as *const u8 as usize;
    Device => |x| Arc::as_ptr(&x.shared) as *const u8 as usize;
}
