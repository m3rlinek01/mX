//! mX on Vulkan (1.3, or 1.2 with dynamic rendering and synchronization2; MoltenVK works).
//!
//! The bindless heap is one descriptor set (set 0) of `UPDATE_AFTER_BIND | PARTIALLY_BOUND`
//! runtime arrays, one binding per typed array of the prelude; root groups 0..=6 are sets
//! 1..=7, allocated per draw from the pipeline's reflected bindings; immediates are push
//! constants. Every image lives in `GENERAL` layout (swapchain images go to `PRESENT_SRC` to
//! be shown), so the heap's descriptors are valid whatever a pass does, and passes and copies
//! are ordered by one memory barrier between them. See `docs/ARCHITECTURE.md`.

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

pub use ash;
pub use bundle::{RenderBundle, RenderBundleEncoder};
pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use device::{Adapter, Device, DeviceDescriptor, Instance, Queue, SubmissionIndex};
pub use pipeline::{ComputePipeline, RenderPipeline, ShaderModule, ShaderModuleDescriptor};
pub use query::QuerySet;
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, Sampler, Texture, TextureView};
pub use rt::{Blas, Tlas};
pub use surface::{Surface, SurfaceTexture};

use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no Vulkan device mX can use (it needs Vulkan 1.2 with descriptor indexing, timeline semaphores, dynamic rendering and synchronization2)")]
    NoDevice,
    #[error("out of GPU memory creating {0}")]
    OutOfMemory(String),
    #[error("{0} is not supported on this device")]
    Unsupported(String),
    #[error("the bindless heap is full: {0}")]
    HeapFull(&'static str),
    #[error(transparent)]
    Shader(#[from] mx_shader::ShaderError),
    #[error("{label}: Vulkan could not compile the shader: {message}")]
    Compile { label: String, message: String },
    #[error("{label}: Vulkan could not build the pipeline: {message}")]
    Pipeline { label: String, message: String },
    #[error("the GPU reported an error: {0}")]
    Device(String),
    #[error("surface: {0}")]
    Surface(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A Vulkan error from creating `what`.
pub(crate) fn vk_err(what: &str, e: vk::Result) -> Error {
    match e {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => Error::OutOfMemory(what.into()),
        vk::Result::ERROR_DEVICE_LOST => Error::Device(format!("device lost ({what})")),
        other => Error::Device(format!("{what}: {other}")),
    }
}

pub(crate) fn alloc_err(what: &str, e: gpu_allocator::AllocationError) -> Error {
    match e {
        gpu_allocator::AllocationError::OutOfMemory => Error::OutOfMemory(what.into()),
        other => Error::Device(format!("{what}: {other}")),
    }
}
