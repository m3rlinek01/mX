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
pub use resource::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferInitDescriptor, Sampler, Texture, TextureView};
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
