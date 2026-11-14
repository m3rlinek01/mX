//! mX: openOMSI's bindless graphics library.
//!
//! The API follows wgpu's names and shapes where the concept is the same. What differs:
//!
//! - **No layouts.** Pipelines have no pipeline or bind group layout; what a shader reads is
//!   reflected from it.
//! - **Bindless heap.** Every texture view made with `TEXTURE_BINDING` and every sampler has an
//!   index ([`TextureView::index`], [`Sampler::index`]). Shaders read them as
//!   `mx_tex2d[i]`, `mx_samplers[j]` (see [`PRELUDE`]), usually with the indices passed as
//!   immediates ([`RenderPass::set_immediates`], WGSL `var<immediate>`).
//! - **Root bind groups** (WGSL groups `0..=6`) stay for uniforms, storage buffers and code
//!   being ported from wgpu; group 7 is the heap.
//!
//! The platform's backend is chosen at compile time: Metal 3 on Apple platforms, Vulkan on
//! Linux and Android, Direct3D 12 on Windows, and Vulkan anywhere (macOS through MoltenVK)
//! with the `vulkan` feature (see `docs/ARCHITECTURE.md`).

pub use mx_shader::{HEAP_GROUP, MAX_ROOT_GROUP, PRELUDE};
pub use mx_types::*;

#[cfg(all(target_vendor = "apple", not(feature = "vulkan")))]
mod backend {
    pub use mx_metal::*;
}

// (with the `vulkan` feature on an Apple platform the Metal backend is still a dependency, unused)
#[cfg(all(target_vendor = "apple", feature = "vulkan"))]
use mx_metal as _;

// Vulkan: with the `vulkan` feature on any platform, and by default where there is no Metal
// or Direct3D 12.
#[cfg(any(feature = "vulkan", not(any(target_vendor = "apple", windows))))]
mod backend {
    pub use mx_vulkan::*;
}

// Direct3D 12: the Windows default (Vulkan there with the `vulkan` feature).
#[cfg(all(windows, not(feature = "vulkan")))]
mod backend {
    pub use mx_dx12::*;
}

pub use backend::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Blas, Buffer, BufferBinding, BufferInitDescriptor, CommandBuffer, CommandEncoder, ComputePass, ComputePipeline, Device,
    DeviceDescriptor, Error, Instance, QuerySet, Queue, RenderBundle, RenderBundleEncoder, RenderPass, RenderPipeline, Result, Sampler, ShaderModule, ShaderModuleDescriptor, SubmissionIndex, Surface,
    SurfaceTexture, Texture, TextureView, Tlas,
};

pub type RenderPipelineDescriptor<'a> = mx_types::RenderPipelineDescriptor<'a, ShaderModule>;
pub type ComputePipelineDescriptor<'a> = mx_types::ComputePipelineDescriptor<'a, ShaderModule>;
pub type VertexState<'a> = mx_types::VertexState<'a, ShaderModule>;
pub type FragmentState<'a> = mx_types::FragmentState<'a, ShaderModule>;
pub type RenderPassDescriptor<'a> = mx_types::RenderPassDescriptor<'a, TextureView, QuerySet>;
pub type ComputePassDescriptor<'a> = mx_types::ComputePassDescriptor<'a, QuerySet>;
pub type PassTimestampWrites<'a> = mx_types::PassTimestampWrites<'a, QuerySet>;
pub type RenderPassColorAttachment<'a> = mx_types::RenderPassColorAttachment<'a, TextureView>;
pub type RenderPassDepthStencilAttachment<'a> = mx_types::RenderPassDepthStencilAttachment<'a, TextureView>;
pub type TexelCopyTextureInfo<'a> = mx_types::TexelCopyTextureInfo<'a, Texture>;
pub type TexelCopyBufferInfo<'a> = mx_types::TexelCopyBufferInfo<'a, Buffer>;
pub type BlasBuildEntry<'a> = mx_types::BlasBuildEntry<'a, Blas, Buffer>;
pub type BlasTriangleGeometry<'a> = mx_types::BlasTriangleGeometry<'a, Buffer>;
pub type TlasInstance = mx_types::TlasInstance<Blas>;
