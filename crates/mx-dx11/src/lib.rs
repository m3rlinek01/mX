//! mX on Direct3D 11: the bound tier (see `docs/BOUND_TIER.md`), for GPUs and drivers without
//! Direct3D 12 (feature level 11.0 and up, Windows 8 and newer: `ID3D11DeviceContext1`).
//!
//! The GPU has no bindless here. The heap lives on the CPU (indices, deferred reuse, the same
//! API as on the other backends): a texture view's index stands for its shader resource view,
//! a sampler's for its sampler state, a buffer view's for the buffer range. Shaders are lowered
//! so that each heap access indexed by an immediate becomes a register of its own, filled at
//! draw time from the immediates (`mx_shader::bound`), and written as HLSL shader model 5.0
//! with registers per stage (`mx_shader::hlsl11`), compiled with FXC (`d3dcompiler_47.dll`).
//!
//! The immediate context is the device's one queue, behind a lock. Command encoders record
//! commands; `Queue::submit` replays them on it, and an event query after each submission
//! tells `Device::poll` when it is done. Direct3D 11 tracks hazards itself.
//!
//! This backend has been compile-checked only (`x86_64-pc-windows-gnu`); it has never run.

#![cfg(windows)]

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

pub use bundle::{RenderBundle, RenderBundleEncoder};
pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use device::{Adapter, Device, DeviceDescriptor, Instance, Queue, SubmissionIndex};
pub use pipeline::{ComputePipeline, RenderPipeline, ShaderModule, ShaderModuleDescriptor};
pub use query::QuerySet;
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, BufferView, IndirectDraws, Sampler, Texture, TextureView};
pub use rt::{Blas, Tlas};
pub use surface::{Surface, SurfaceTexture};
pub use windows;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no Direct3D 11 device mX can use (it needs feature level 11.0 and Windows 8 or newer)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: FXC could not compile the shader: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: Direct3D 11 could not make the pipeline: {message}")]
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

/// A Direct3D call failed: out of memory, device removed, or anything else.
pub(crate) fn hr(what: &str, e: windows::core::Error) -> Error {
    use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET};
    let code = e.code();
    if code == windows::Win32::Foundation::E_OUTOFMEMORY {
        Error::OutOfMemory(what.into())
    } else if code == DXGI_ERROR_DEVICE_REMOVED || code == DXGI_ERROR_DEVICE_RESET || code == DXGI_ERROR_DEVICE_HUNG {
        Error::DeviceLost(format!("{what}: {e}"))
    } else {
        Error::Device(format!("{what}: {e}"))
    }
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
