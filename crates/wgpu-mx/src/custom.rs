//! mX as a backend of wgpu itself, through wgpu's custom backend interfaces.
//!
//! With the `custom` feature, [`instance`] returns a real `wgpu::Instance` whose adapters,
//! devices and resources are mX's. A program that uses wgpu can then choose mX at run time
//! (openOMSI's "mX Graphics" setting) and keep wgpu's own backends otherwise.
//!
//! The behaviour is that of the rest of this crate: passes are recorded and replayed into mX
//! when they end, a bind group is built per set of dynamic offsets, and failed shaders and
//! pipelines are reported to error scopes. Ray queries are not offered: wgpu does not give a
//! custom backend the bottom-level structure of a `TlasInstance`.

use std::collections::HashMap;
use std::ops::Range;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;
use wgpu::custom::*;

use crate::conv;

/// A wgpu instance whose adapters are mX's.
pub fn instance() -> wgpu::Instance {
    wgpu::Instance::from_custom(MxInstance::new(wgpu::InstanceDescriptor::new_without_display_handle()))
}

/// The mX device under a wgpu device made by [`instance`], for interop.
pub fn device(d: &wgpu::Device) -> Option<&mx::Device> {
    d.as_custom::<MxDevice>().map(|d| &d.0.mx)
}

/// The mX queue under a wgpu queue made by [`instance`].
pub fn queue(q: &wgpu::Queue) -> Option<&mx::Queue> {
    q.as_custom::<MxQueue>().map(|q| &q.0.queue)
}

/// The mX texture under a wgpu texture made on an mX device.
pub fn texture(t: &wgpu::Texture) -> Option<&mx::Texture> {
    t.as_custom::<MxTexture>().map(|t| &t.0)
}

/// The mX buffer under a wgpu buffer made on an mX device.
pub fn buffer(b: &wgpu::Buffer) -> Option<&mx::Buffer> {
    b.as_custom::<MxBuffer>().map(|b| &b.0.mx)
}

/// Whether a wgpu device runs on mX.
pub fn is_mx(d: &wgpu::Device) -> bool {
    d.as_custom::<MxDevice>().is_some()
}

fn mx_backend() -> wgpu::Backend {
    if cfg!(target_vendor = "apple") {
        wgpu::Backend::Metal
    } else if cfg!(windows) {
        wgpu::Backend::Dx12
    } else {
        wgpu::Backend::Vulkan
    }
}

macro_rules! debug_name {
    ($($t:ident),*) => {$(
        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(concat!(stringify!($t), " (mX)"))
            }
        }
    )*};
}

debug_name!(
    MxInstance,
    MxAdapter,
    MxDevice,
    MxQueue,
    MxShaderModule,
    MxBindGroupLayout,
    MxBindGroup,
    MxTextureView,
    MxSampler,
    MxBuffer,
    MxTexture,
    MxQuerySet,
    MxPipelineLayout,
    MxRenderPipeline,
    MxComputePipeline,
    MxPipelineCache,
    MxCommandEncoder,
    MxComputePass,
    MxRenderPass,
    MxCommandBuffer,
    MxRenderBundleEncoder,
    MxRenderBundle,
    MxSurface,
    MxSurfaceOutput,
    MxStaging,
    MxMappedRange
);

// ---------------------------------------------------------------------------------------------
// Instance and adapter

struct MxInstance {
    inner: Arc<mx::Instance>,
}

impl InstanceInterface for MxInstance {
    fn new(_desc: wgpu::InstanceDescriptor) -> Self {
        MxInstance { inner: Arc::new(mx::Instance::new()) }
    }

    unsafe fn create_surface(&self, target: wgpu::SurfaceTargetUnsafe) -> Result<DispatchSurface, wgpu::CreateSurfaceError> {
        let wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle,
            raw_window_handle,
        } = target
        else {
            panic!("mX: only raw window handles can become surfaces");
        };
        let display = match raw_display_handle {
            Some(d) => d,
            #[cfg(target_vendor = "apple")]
            None => wgpu::rwh::RawDisplayHandle::AppKit(wgpu::rwh::AppKitDisplayHandle::new()),
            #[cfg(not(target_vendor = "apple"))]
            None => panic!("mX: a surface needs a display handle on this platform"),
        };
        // SAFETY: the caller keeps the window and display alive while the surface lives.
        let s = unsafe { self.inner.create_surface_unsafe(display, raw_window_handle) }.unwrap_or_else(|e| panic!("mX: creating a surface: {e}"));
        Ok(DispatchSurface::custom(MxSurface {
            inner: Arc::new(s),
            config: Mutex::new(None),
        }))
    }

    fn request_adapter(&self, _options: &wgpu::RequestAdapterOptions<'_, '_>) -> Pin<Box<dyn RequestAdapterFuture>> {
        let r = self.inner.default_adapter().ok().map(|a| DispatchAdapter::custom(MxAdapter(Arc::new(a))));
        Box::pin(std::future::ready(r.ok_or(wgpu::RequestAdapterError::NotFound {
            active_backends: wgpu::Backends::from(mx_backend()),
            requested_backends: wgpu::Backends::from(mx_backend()),
            supported_backends: wgpu::Backends::from(mx_backend()),
            no_fallback_backends: wgpu::Backends::empty(),
            no_adapter_backends: wgpu::Backends::from(mx_backend()),
            incompatible_surface_backends: wgpu::Backends::empty(),
        })))
    }

    fn poll_all_devices(&self, _force_wait: bool) -> bool {
        true
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        wgpu::WgslLanguageFeatures::empty()
    }

    fn enumerate_adapters(&self, backends: wgpu::Backends) -> Pin<Box<dyn EnumerateAdapterFuture>> {
        let list = if backends.contains(wgpu::Backends::from(mx_backend())) {
            self.inner.enumerate_adapters().into_iter().map(|a| DispatchAdapter::custom(MxAdapter(Arc::new(a)))).collect()
        } else {
            Vec::new()
        };
        Box::pin(std::future::ready(list))
    }
}

struct MxAdapter(Arc<mx::Adapter>);

impl MxAdapter {
    fn mx_features(&self) -> wgpu::Features {
        // (ray queries are left out: see the module documentation)
        let mut f = wgpu::Features::IMMEDIATES
            | wgpu::Features::DEPTH32FLOAT_STENCIL8
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::CLEAR_TEXTURE;
        if self.0.supports_bc() {
            f |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        if self.0.supports_timestamps() {
            f |= wgpu::Features::TIMESTAMP_QUERY;
        }
        f
    }

    fn info(&self) -> wgpu::AdapterInfo {
        let i = self.0.info();
        wgpu::AdapterInfo {
            name: i.name,
            vendor: i.vendor,
            device: i.device,
            device_type: match i.device_type {
                mx::DeviceType::DiscreteGpu => wgpu::DeviceType::DiscreteGpu,
                mx::DeviceType::IntegratedGpu => wgpu::DeviceType::IntegratedGpu,
                mx::DeviceType::VirtualGpu => wgpu::DeviceType::VirtualGpu,
                mx::DeviceType::Cpu => wgpu::DeviceType::Cpu,
                mx::DeviceType::Other => wgpu::DeviceType::Other,
            },
            device_pci_bus_id: String::new(),
            driver: format!("mX ({})", i.driver),
            driver_info: String::new(),
            backend: mx_backend(),
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            transient_saves_memory: cfg!(target_vendor = "apple"),
        }
    }
}

fn limits(l: &mx::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_1d: l.max_texture_dimension_2d,
        max_texture_dimension_2d: l.max_texture_dimension_2d,
        max_texture_dimension_3d: l.max_texture_dimension_3d,
        max_texture_array_layers: l.max_texture_array_layers,
        max_vertex_buffers: l.max_vertex_buffers,
        max_color_attachments: l.max_color_attachments,
        max_immediate_size: l.max_immediate_size,
        max_bind_groups: 7,
        max_storage_buffer_binding_size: 1 << 30,
        max_buffer_size: 1 << 32,
        max_uniform_buffers_per_shader_stage: 12,
        max_storage_buffers_per_shader_stage: 12,
        max_sampled_textures_per_shader_stage: 64,
        max_samplers_per_shader_stage: 16,
        ..wgpu::Limits::default()
    }
}

impl AdapterInterface for MxAdapter {
    fn request_device(&self, desc: &wgpu::DeviceDescriptor<'_>) -> Pin<Box<dyn RequestDeviceFuture>> {
        let missing = desc.required_features - self.mx_features();
        if !missing.is_empty() {
            log::error!("mX: the device asks for features mX does not have: {missing:?}");
        }
        let (d, q) = self
            .0
            .request_device(&mx::DeviceDescriptor {
                label: desc.label,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("mX: requesting a device: {e}"));
        let inner = Arc::new(DeviceInner {
            mx: d,
            queue: q,
            features: desc.required_features & self.mx_features(),
            limits: limits(&self.0.limits()),
            info: self.info(),
            scopes: Mutex::new(Vec::new()),
            uncaptured: Mutex::new(None),
            maps: Mutex::new(Vec::new()),
            done_callbacks: Mutex::new(Vec::new()),
            last_submission: Mutex::new(mx::SubmissionIndex(0)),
        });
        Box::pin(std::future::ready(Ok((DispatchDevice::custom(MxDevice(inner.clone())), DispatchQueue::custom(MxQueue(inner))))))
    }

    fn is_surface_supported(&self, _surface: &DispatchSurface) -> bool {
        true
    }

    fn features(&self) -> wgpu::Features {
        self.mx_features()
    }

    fn limits(&self) -> wgpu::Limits {
        limits(&self.0.limits())
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        self.info()
    }

    fn get_texture_format_features(&self, format: wgpu::TextureFormat) -> wgpu::TextureFormatFeatures {
        let mut f = format.guaranteed_format_features(self.mx_features());
        if f.flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
            f.flags |= wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2;
        }
        f
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        wgpu::PresentationTimestamp::INVALID_TIMESTAMP
    }

    fn cooperative_matrix_properties(&self) -> Vec<wgpu::wgt::CooperativeMatrixProperties> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Device and queue

type Callback = Box<dyn FnOnce() + Send>;

struct PendingMap {
    buffer: Arc<BufferInner>,
    until: mx::SubmissionIndex,
    callback: BufferMapCallback,
}

struct DeviceInner {
    mx: mx::Device,
    queue: mx::Queue,
    features: wgpu::Features,
    limits: wgpu::Limits,
    info: wgpu::AdapterInfo,
    scopes: Mutex<Vec<(wgpu::ErrorFilter, Option<wgpu::Error>)>>,
    uncaptured: Mutex<Option<Arc<dyn wgpu::UncapturedErrorHandler>>>,
    maps: Mutex<Vec<PendingMap>>,
    done_callbacks: Mutex<Vec<(mx::SubmissionIndex, Callback)>>,
    last_submission: Mutex<mx::SubmissionIndex>,
}

#[derive(Debug)]
struct Message(String);
impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Message {}

fn validation(description: String) -> wgpu::Error {
    wgpu::Error::Validation {
        source: Box::new(Message(description.clone())),
        description,
    }
}

impl DeviceInner {
    fn error(&self, e: wgpu::Error) {
        let filter = match e {
            wgpu::Error::OutOfMemory { .. } => wgpu::ErrorFilter::OutOfMemory,
            wgpu::Error::Validation { .. } => wgpu::ErrorFilter::Validation,
            wgpu::Error::Internal { .. } => wgpu::ErrorFilter::Internal,
        };
        {
            let mut scopes = self.scopes.lock();
            if let Some((_, slot)) = scopes.iter_mut().rev().find(|(f, _)| *f == filter) {
                if slot.is_none() {
                    *slot = Some(e);
                }
                return;
            }
        }
        let handler = self.uncaptured.lock().clone();
        match handler {
            Some(h) => h(e),
            None => log::error!("mX: uncaptured error: {e}"),
        }
    }

    /// Run the callbacks of finished work (maps, work done) without waiting.
    fn progress(&self) {
        let _ = self.mx.poll();
        let ready: Vec<PendingMap> = {
            let mut maps = self.maps.lock();
            let (ready, rest): (Vec<_>, Vec<_>) = maps.drain(..).partition(|m| self.mx.is_done(m.until));
            *maps = rest;
            ready
        };
        for m in ready {
            m.buffer.map.lock().mapped = true;
            (m.callback)(Ok(()));
        }
        let done: Vec<Callback> = {
            let mut cbs = self.done_callbacks.lock();
            let (ready, rest): (Vec<_>, Vec<_>) = cbs.drain(..).partition(|(i, _)| self.mx.is_done(*i));
            *cbs = rest;
            ready.into_iter().map(|(_, c)| c).collect()
        };
        for c in done {
            c();
        }
    }
}

struct MxDevice(Arc<DeviceInner>);
struct MxQueue(Arc<DeviceInner>);

fn cast<'a, T: 'static>(x: Option<&'a T>, what: &str) -> &'a T {
    x.unwrap_or_else(|| panic!("mX: a {what} from another backend"))
}

impl DeviceInterface for MxDevice {
    fn features(&self) -> wgpu::Features {
        self.0.features
    }

    fn limits(&self) -> wgpu::Limits {
        self.0.limits.clone()
    }

    fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.0.info.clone()
    }

    fn create_shader_module(&self, desc: wgpu::ShaderModuleDescriptor<'_>, _checks: wgpu::ShaderRuntimeChecks) -> DispatchShaderModule {
        let result = match &desc.source {
            wgpu::ShaderSource::Wgsl(source) => self.0.mx.create_shader_module(&mx::ShaderModuleDescriptor { label: desc.label, source }).map_err(|e| e.to_string()),
            _ => Err(format!("{}: mX takes WGSL shaders only", desc.label.unwrap_or("shader"))),
        };
        let (module, message) = match result {
            Ok(m) => (Some(m), None),
            Err(e) => {
                self.0.error(validation(e.clone()));
                (None, Some(e))
            }
        };
        DispatchShaderModule::custom(MxShaderModule { module, message })
    }

    unsafe fn create_shader_module_passthrough(&self, desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>) -> DispatchShaderModule {
        let e = format!("{}: mX does not take passthrough shaders", desc.label.unwrap_or("shader"));
        self.0.error(validation(e.clone()));
        DispatchShaderModule::custom(MxShaderModule { module: None, message: Some(e) })
    }

    fn create_bind_group_layout(&self, desc: &wgpu::BindGroupLayoutDescriptor<'_>) -> DispatchBindGroupLayout {
        DispatchBindGroupLayout::custom(MxBindGroupLayout(Arc::new(desc.entries.to_vec())))
    }

    fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> DispatchBindGroup {
        let layout = cast(desc.layout.as_custom::<MxBindGroupLayout>(), "bind group layout");
        let mut dynamic: Vec<u32> = layout
            .0
            .iter()
            .filter(|e| matches!(e.ty, wgpu::BindingType::Buffer { has_dynamic_offset: true, .. }))
            .map(|e| e.binding)
            .collect();
        dynamic.sort_unstable();
        let entries: Vec<(u32, Owned)> = desc
            .entries
            .iter()
            .map(|e| {
                let r = match &e.resource {
                    wgpu::BindingResource::Buffer(b) => Owned::Buffer(cast(b.buffer.as_custom::<MxBuffer>(), "buffer").0.clone(), b.offset, b.size.map(|s| s.get())),
                    wgpu::BindingResource::TextureView(v) => Owned::View(cast(v.as_custom::<MxTextureView>(), "texture view").0.clone()),
                    wgpu::BindingResource::Sampler(s) => Owned::Sampler(cast(s.as_custom::<MxSampler>(), "sampler").0.clone()),
                    _ => panic!("mX: binding arrays and acceleration structures are not offered (textures are reached through the heap)"),
                };
                (e.binding, r)
            })
            .collect();
        let inner = BindGroupInner {
            device: self.0.clone(),
            entries,
            dynamic,
            built: Mutex::new(HashMap::new()),
        };
        inner.with_offsets(&[]);
        DispatchBindGroup::custom(MxBindGroup(Arc::new(inner)))
    }

    fn create_pipeline_layout(&self, _desc: &wgpu::PipelineLayoutDescriptor<'_>) -> DispatchPipelineLayout {
        DispatchPipelineLayout::custom(MxPipelineLayout)
    }

    fn create_render_pipeline(&self, desc: &wgpu::RenderPipelineDescriptor<'_>) -> DispatchRenderPipeline {
        let p = self.render_pipeline(desc);
        if let Err(e) = &p {
            self.0.error(validation(e.clone()));
        }
        DispatchRenderPipeline::custom(MxRenderPipeline(Arc::new(p.ok())))
    }

    fn create_mesh_pipeline(&self, desc: &wgpu::MeshPipelineDescriptor<'_>) -> DispatchRenderPipeline {
        self.0.error(validation(format!("{}: mX has no mesh shaders", desc.label.unwrap_or("mesh pipeline"))));
        DispatchRenderPipeline::custom(MxRenderPipeline(Arc::new(None)))
    }

    fn create_compute_pipeline(&self, desc: &wgpu::ComputePipelineDescriptor<'_>) -> DispatchComputePipeline {
        let module = cast(desc.module.as_custom::<MxShaderModule>(), "shader module");
        let p = match &module.module {
            Some(m) => self
                .0
                .mx
                .create_compute_pipeline(&mx::ComputePipelineDescriptor {
                    label: desc.label,
                    module: m,
                    entry_point: desc.entry_point,
                    constants: desc.compilation_options.constants,
                })
                .map_err(|e| e.to_string()),
            None => Err(format!("{}: its shader module is invalid", desc.label.unwrap_or("compute pipeline"))),
        };
        if let Err(e) = &p {
            self.0.error(validation(e.clone()));
        }
        DispatchComputePipeline::custom(MxComputePipeline(Arc::new(p.ok())))
    }

    unsafe fn create_pipeline_cache(&self, _desc: &wgpu::PipelineCacheDescriptor<'_>) -> DispatchPipelineCache {
        DispatchPipelineCache::custom(MxPipelineCache)
    }

    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> DispatchBuffer {
        let b = self
            .0
            .mx
            .create_buffer(&mx::BufferDescriptor {
                label: desc.label,
                size: desc.size,
                usage: conv::buffer_usages(desc.usage),
            })
            .unwrap_or_else(|e| panic!("mX: {e}"));
        let mappable = desc.usage.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE);
        let map = MapState {
            mapped: desc.mapped_at_creation,
            staging: (desc.mapped_at_creation && !mappable).then(|| vec![0u8; desc.size as usize]),
        };
        DispatchBuffer::custom(MxBuffer(Arc::new(BufferInner {
            mx: b,
            size: desc.size,
            usage: desc.usage,
            map: Mutex::new(map),
            device: self.0.clone(),
        })))
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> DispatchTexture {
        let view_formats: Vec<mx::TextureFormat> = desc.view_formats.iter().map(|f| conv::format(*f)).collect();
        let t = self
            .0
            .mx
            .create_texture(&mx::TextureDescriptor {
                label: desc.label,
                size: conv::extent(desc.size),
                mip_level_count: desc.mip_level_count,
                sample_count: desc.sample_count,
                dimension: conv::texture_dimension(desc.dimension),
                format: conv::format(desc.format),
                usage: conv::texture_usages(desc.usage),
                view_formats: &view_formats,
            })
            .unwrap_or_else(|e| panic!("mX: {e}"));
        DispatchTexture::custom(MxTexture(t))
    }

    fn create_external_texture(&self, _desc: &wgpu::ExternalTextureDescriptor<'_>, _planes: &[&wgpu::TextureView]) -> DispatchExternalTexture {
        panic!("mX: external textures are not offered")
    }

    fn create_blas(&self, _desc: &wgpu::CreateBlasDescriptor<'_>, _sizes: wgpu::BlasGeometrySizeDescriptors) -> (Option<u64>, DispatchBlas) {
        panic!("mX: acceleration structures are not offered through wgpu's custom backends")
    }

    fn create_tlas(&self, _desc: &wgpu::CreateTlasDescriptor<'_>) -> DispatchTlas {
        panic!("mX: acceleration structures are not offered through wgpu's custom backends")
    }

    fn create_sampler(&self, desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        let s = self.0.mx.create_sampler(&conv::sampler(desc)).unwrap_or_else(|e| panic!("mX: {e}"));
        DispatchSampler::custom(MxSampler(s))
    }

    fn create_query_set(&self, desc: &wgpu::QuerySetDescriptor<'_>) -> DispatchQuerySet {
        let q = self
            .0
            .mx
            .create_query_set(&mx::QuerySetDescriptor {
                label: desc.label,
                ty: conv::query_type(desc.ty),
                count: desc.count,
            })
            .unwrap_or_else(|e| panic!("mX: {e}"));
        DispatchQuerySet::custom(MxQuerySet(Arc::new(q)))
    }

    fn create_command_encoder(&self, desc: &wgpu::CommandEncoderDescriptor<'_>) -> DispatchCommandEncoder {
        DispatchCommandEncoder::custom(MxCommandEncoder(Arc::new(Mutex::new(Some(self.0.mx.create_command_encoder(desc.label))))))
    }

    fn create_render_bundle_encoder(&self, _desc: &wgpu::RenderBundleEncoderDescriptor<'_>) -> DispatchRenderBundleEncoder {
        DispatchRenderBundleEncoder::custom(MxRenderBundleEncoder(Vec::new()))
    }

    fn set_device_lost_callback(&self, _callback: BoxDeviceLostCallback) {}

    fn on_uncaptured_error(&self, handler: Arc<dyn wgpu::UncapturedErrorHandler>) {
        *self.0.uncaptured.lock() = Some(handler);
    }

    fn push_error_scope(&self, filter: wgpu::ErrorFilter) -> u32 {
        let mut s = self.0.scopes.lock();
        s.push((filter, None));
        (s.len() - 1) as u32
    }

    fn pop_error_scope(&self, index: u32) -> Pin<Box<dyn PopErrorScopeFuture>> {
        let mut s = self.0.scopes.lock();
        let e = if (index as usize) < s.len() {
            s.drain(index as usize..).next().and_then(|(_, e)| e)
        } else {
            None
        };
        Box::pin(std::future::ready(e))
    }

    unsafe fn start_graphics_debugger_capture(&self) {}

    unsafe fn stop_graphics_debugger_capture(&self) {}

    fn poll(&self, poll_type: wgpu::wgt::PollType<u64>) -> Result<wgpu::PollStatus, wgpu::PollError> {
        let d = &self.0;
        match poll_type {
            wgpu::wgt::PollType::Poll => {
                d.progress();
                Ok(wgpu::PollStatus::Poll)
            }
            wgpu::wgt::PollType::Wait { submission_index, .. } => {
                let index = submission_index.map(mx::SubmissionIndex).unwrap_or(*d.last_submission.lock());
                d.mx.wait(index).map_err(|_| wgpu::PollError::Timeout)?;
                d.progress();
                Ok(if submission_index.is_some() {
                    wgpu::PollStatus::WaitSucceeded
                } else {
                    wgpu::PollStatus::QueueEmpty
                })
            }
        }
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        wgpu::InternalCounters::default()
    }

    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }

    fn destroy(&self) {}
}

impl MxDevice {
    fn render_pipeline(&self, desc: &wgpu::RenderPipelineDescriptor<'_>) -> Result<mx::RenderPipeline, String> {
        let label = desc.label.unwrap_or("render pipeline");
        let vm = cast(desc.vertex.module.as_custom::<MxShaderModule>(), "shader module")
            .module
            .as_ref()
            .ok_or_else(|| format!("{label}: its vertex shader module is invalid"))?;
        let attrs: Vec<Vec<mx::VertexAttribute>> = desc
            .vertex
            .buffers
            .iter()
            .map(|b| {
                b.attributes
                    .iter()
                    .map(|a| mx::VertexAttribute {
                        format: conv::vertex_format(a.format),
                        offset: a.offset,
                        shader_location: a.shader_location,
                    })
                    .collect()
            })
            .collect();
        let buffers: Vec<mx::VertexBufferLayout<'_>> = desc
            .vertex
            .buffers
            .iter()
            .zip(&attrs)
            .map(|(b, a)| mx::VertexBufferLayout {
                array_stride: b.array_stride,
                step_mode: conv::step_mode(b.step_mode),
                attributes: a,
            })
            .collect();
        let targets: Vec<Option<mx::ColorTargetState>> = desc.fragment.as_ref().map(|f| f.targets.iter().map(|t| t.as_ref().map(conv::target)).collect()).unwrap_or_default();
        let fragment = match &desc.fragment {
            Some(f) => Some(mx::FragmentState {
                module: cast(f.module.as_custom::<MxShaderModule>(), "shader module")
                    .module
                    .as_ref()
                    .ok_or_else(|| format!("{label}: its fragment shader module is invalid"))?,
                entry_point: f.entry_point,
                constants: f.compilation_options.constants,
                targets: &targets,
            }),
            None => None,
        };
        self.0
            .mx
            .create_render_pipeline(&mx::RenderPipelineDescriptor {
                label: desc.label,
                vertex: mx::VertexState {
                    module: vm,
                    entry_point: desc.vertex.entry_point,
                    constants: desc.vertex.compilation_options.constants,
                    buffers: &buffers,
                },
                primitive: conv::primitive(desc.primitive),
                depth_stencil: desc.depth_stencil.as_ref().map(conv::depth_stencil),
                multisample: conv::multisample(desc.multisample),
                fragment,
            })
            .map_err(|e| e.to_string())
    }
}

impl QueueInterface for MxQueue {
    fn write_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, data: &[u8]) {
        let b = cast(buffer.as_custom::<MxBuffer>(), "buffer");
        if let Err(e) = self.0.queue.write_buffer(&b.0.mx, offset, data) {
            self.0.error(wgpu::Error::OutOfMemory {
                source: Box::new(Message(e.to_string())),
            });
        }
    }

    fn create_staging_buffer(&self, size: wgpu::BufferSize) -> Option<DispatchQueueWriteBuffer> {
        Some(DispatchQueueWriteBuffer::custom(MxStaging(vec![0; size.get() as usize])))
    }

    fn validate_write_buffer(&self, _buffer: &DispatchBuffer, _offset: wgpu::BufferAddress, _size: wgpu::BufferSize) -> Option<()> {
        Some(())
    }

    fn write_staging_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, staging_buffer: &DispatchQueueWriteBuffer) {
        let s = cast(staging_buffer.as_custom::<MxStaging>(), "staging buffer");
        self.write_buffer(buffer, offset, &s.0);
    }

    fn write_texture(&self, texture: wgpu::TexelCopyTextureInfo<'_>, data: &[u8], data_layout: wgpu::TexelCopyBufferLayout, size: wgpu::Extent3d) {
        let t = cast(texture.texture.as_custom::<MxTexture>(), "texture");
        let dst = mx::TexelCopyTextureInfo {
            texture: &t.0,
            mip_level: texture.mip_level,
            origin: conv::origin(texture.origin),
            aspect: conv::aspect(texture.aspect),
        };
        if let Err(e) = self.0.queue.write_texture(dst, data, conv::layout(data_layout), conv::extent(size)) {
            self.0.error(wgpu::Error::OutOfMemory {
                source: Box::new(Message(e.to_string())),
            });
        }
    }

    fn submit(&self, command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>) -> u64 {
        let raw: Vec<mx::CommandBuffer> = command_buffers.filter_map(|c| c.as_custom::<MxCommandBuffer>().and_then(|c| c.0.lock().take())).collect();
        let i = self.0.queue.submit(raw);
        *self.0.last_submission.lock() = i;
        self.0.progress();
        i.0
    }

    fn get_timestamp_period(&self) -> f32 {
        self.0.queue.get_timestamp_period()
    }

    fn on_submitted_work_done(&self, callback: BoxSubmittedWorkDoneCallback) {
        let at = *self.0.last_submission.lock();
        self.0.done_callbacks.lock().push((at, callback));
    }

    fn compact_blas(&self, _blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        panic!("mX: acceleration structures are not offered through wgpu's custom backends")
    }
}

// ---------------------------------------------------------------------------------------------
// Resources

struct MxShaderModule {
    module: Option<mx::ShaderModule>,
    message: Option<String>,
}

impl ShaderModuleInterface for MxShaderModule {
    fn get_compilation_info(&self) -> Pin<Box<dyn ShaderCompilationInfoFuture>> {
        let messages = self
            .message
            .iter()
            .map(|m| wgpu::CompilationMessage {
                message: m.clone(),
                message_type: wgpu::CompilationMessageType::Error,
                location: None,
            })
            .collect();
        Box::pin(std::future::ready(wgpu::CompilationInfo { messages }))
    }
}

struct MxBindGroupLayout(Arc<Vec<wgpu::BindGroupLayoutEntry>>);
impl BindGroupLayoutInterface for MxBindGroupLayout {}

struct MxPipelineLayout;
impl PipelineLayoutInterface for MxPipelineLayout {}

struct MxPipelineCache;
impl PipelineCacheInterface for MxPipelineCache {
    fn get_data(&self) -> Option<Vec<u8>> {
        None
    }
}

enum Owned {
    Buffer(Arc<BufferInner>, u64, Option<u64>),
    View(mx::TextureView),
    Sampler(mx::Sampler),
}

struct BindGroupInner {
    device: Arc<DeviceInner>,
    entries: Vec<(u32, Owned)>,
    /// The bindings with dynamic offsets, in binding order (the order offsets come in).
    dynamic: Vec<u32>,
    built: Mutex<HashMap<Vec<u32>, mx::BindGroup>>,
}

impl BindGroupInner {
    fn build(&self, offsets: &[u32]) -> mx::BindGroup {
        let entries: Vec<mx::BindGroupEntry<'_>> = self
            .entries
            .iter()
            .map(|(binding, r)| mx::BindGroupEntry {
                binding: *binding,
                resource: match r {
                    Owned::Buffer(b, offset, size) => {
                        let extra = self.dynamic.iter().position(|d| d == binding).map_or(0, |i| offsets[i] as u64);
                        mx::BindingResource::Buffer(mx::BufferBinding {
                            buffer: &b.mx,
                            offset: offset + extra,
                            size: *size,
                        })
                    }
                    Owned::View(v) => mx::BindingResource::TextureView(v),
                    Owned::Sampler(s) => mx::BindingResource::Sampler(s),
                },
            })
            .collect();
        self.device.mx.create_bind_group(&mx::BindGroupDescriptor { label: None, entries: &entries })
    }

    fn with_offsets(&self, offsets: &[u32]) -> mx::BindGroup {
        let key = if offsets.is_empty() { vec![0; self.dynamic.len()] } else { offsets.to_vec() };
        let mut built = self.built.lock();
        if let Some(g) = built.get(&key) {
            return g.clone();
        }
        if built.len() > 4096 {
            built.clear();
        }
        let g = self.build(&key);
        built.insert(key, g.clone());
        g
    }
}

struct MxBindGroup(Arc<BindGroupInner>);
impl BindGroupInterface for MxBindGroup {}

struct MxTextureView(mx::TextureView);
impl TextureViewInterface for MxTextureView {}

struct MxSampler(mx::Sampler);
impl SamplerInterface for MxSampler {}

struct MxQuerySet(Arc<mx::QuerySet>);
impl QuerySetInterface for MxQuerySet {}

struct MapState {
    mapped: bool,
    /// The bytes of a buffer mapped at creation that the CPU cannot reach (written on unmap).
    staging: Option<Vec<u8>>,
}

struct BufferInner {
    mx: mx::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    map: Mutex<MapState>,
    device: Arc<DeviceInner>,
}

struct MxBuffer(Arc<BufferInner>);

impl BufferInterface for MxBuffer {
    fn map_async(&self, mode: wgpu::MapMode, _range: Range<wgpu::BufferAddress>, callback: BufferMapCallback) {
        let needed = if mode == wgpu::MapMode::Read {
            wgpu::BufferUsages::MAP_READ
        } else {
            wgpu::BufferUsages::MAP_WRITE
        };
        if !self.0.usage.contains(needed) {
            callback(Err(wgpu::BufferAsyncError));
            return;
        }
        let until = *self.0.device.last_submission.lock();
        self.0.device.maps.lock().push(PendingMap {
            buffer: self.0.clone(),
            until,
            callback,
        });
    }

    fn get_mapped_range(&self, sub_range: Range<wgpu::BufferAddress>) -> DispatchBufferMappedRange {
        let mut m = self.0.map.lock();
        assert!(m.mapped, "mX: get_mapped_range on a buffer that is not mapped");
        assert!(sub_range.end <= self.0.size, "mX: mapped range past the end of the buffer");
        let base = match &mut m.staging {
            Some(v) => NonNull::new(v.as_mut_ptr()).unwrap(),
            // SAFETY: the buffer is mapped, so no submitted work uses it.
            None => unsafe { self.0.mx.mapped_ptr() },
        };
        DispatchBufferMappedRange::custom(MxMappedRange {
            // SAFETY: inside the buffer (checked above).
            ptr: unsafe { base.add(sub_range.start as usize) },
            len: (sub_range.end - sub_range.start) as usize,
            _buffer: self.0.clone(),
        })
    }

    fn unmap(&self) {
        let staged = {
            let mut m = self.0.map.lock();
            m.mapped = false;
            m.staging.take()
        };
        if let Some(bytes) = staged {
            let _ = self.0.device.queue.write_buffer(&self.0.mx, 0, &bytes);
        }
    }

    fn destroy(&self) {}
}

struct MxMappedRange {
    ptr: NonNull<u8>,
    len: usize,
    _buffer: Arc<BufferInner>,
}
// SAFETY: a view of mapped bytes that nothing else writes while it lives.
unsafe impl Send for MxMappedRange {}
// SAFETY: as above.
unsafe impl Sync for MxMappedRange {}

impl BufferMappedRangeInterface for MxMappedRange {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn read_slice(&self) -> &[u8] {
        // SAFETY: mapped, in range, alive as long as the buffer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        // SAFETY: mapped for writing, in range, borrowed mutably through `self`.
        wgpu::WriteOnly::from_mut(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
    }
}

struct MxStaging(Vec<u8>);

impl QueueWriteBufferInterface for MxStaging {
    fn len(&self) -> usize {
        self.0.len()
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        wgpu::WriteOnly::from_mut(&mut self.0[..])
    }
}

struct MxTexture(mx::Texture);

impl TextureInterface for MxTexture {
    fn create_view(&self, desc: &wgpu::TextureViewDescriptor<'_>) -> DispatchTextureView {
        let v = self.0.create_view(&conv::view(desc)).unwrap_or_else(|e| panic!("mX: {e}"));
        DispatchTextureView::custom(MxTextureView(v))
    }

    fn destroy(&self) {}
}

struct MxRenderPipeline(Arc<Option<mx::RenderPipeline>>);
impl RenderPipelineInterface for MxRenderPipeline {
    fn get_bind_group_layout(&self, _index: u32) -> DispatchBindGroupLayout {
        DispatchBindGroupLayout::custom(MxBindGroupLayout(Arc::new(Vec::new())))
    }
}

struct MxComputePipeline(Arc<Option<mx::ComputePipeline>>);
impl ComputePipelineInterface for MxComputePipeline {
    fn get_bind_group_layout(&self, _index: u32) -> DispatchBindGroupLayout {
        DispatchBindGroupLayout::custom(MxBindGroupLayout(Arc::new(Vec::new())))
    }
}

// ---------------------------------------------------------------------------------------------
// Commands

type EncoderCell = Arc<Mutex<Option<mx::CommandEncoder>>>;

struct MxCommandEncoder(EncoderCell);

struct MxCommandBuffer(Mutex<Option<mx::CommandBuffer>>);
impl CommandBufferInterface for MxCommandBuffer {}

fn copy_texture<'a>(t: &wgpu::TexelCopyTextureInfo<'a>) -> mx::TexelCopyTextureInfo<'a> {
    mx::TexelCopyTextureInfo {
        texture: &cast(t.texture.as_custom::<MxTexture>(), "texture").0,
        mip_level: t.mip_level,
        origin: conv::origin(t.origin),
        aspect: conv::aspect(t.aspect),
    }
}

fn mx_buffer(b: &DispatchBuffer) -> Arc<BufferInner> {
    cast(b.as_custom::<MxBuffer>(), "buffer").0.clone()
}

impl MxCommandEncoder {
    fn with<R>(&self, f: impl FnOnce(&mut mx::CommandEncoder) -> R) -> R {
        f(self.0.lock().as_mut().expect("mX: the encoder has finished"))
    }
}

impl CommandEncoderInterface for MxCommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &DispatchBuffer,
        source_offset: wgpu::BufferAddress,
        destination: &DispatchBuffer,
        destination_offset: wgpu::BufferAddress,
        copy_size: Option<wgpu::BufferAddress>,
    ) {
        let (s, d) = (mx_buffer(source), mx_buffer(destination));
        let size = copy_size.unwrap_or(s.size - source_offset);
        self.with(|e| e.copy_buffer_to_buffer(&s.mx, source_offset, &d.mx, destination_offset, size));
    }

    fn copy_buffer_to_texture(&self, source: wgpu::TexelCopyBufferInfo<'_>, destination: wgpu::TexelCopyTextureInfo<'_>, copy_size: wgpu::Extent3d) {
        let b = cast(source.buffer.as_custom::<MxBuffer>(), "buffer");
        let src = mx::TexelCopyBufferInfo {
            buffer: &b.0.mx,
            layout: conv::layout(source.layout),
        };
        self.with(|e| e.copy_buffer_to_texture(src, copy_texture(&destination), conv::extent(copy_size)));
    }

    fn copy_texture_to_buffer(&self, source: wgpu::TexelCopyTextureInfo<'_>, destination: wgpu::TexelCopyBufferInfo<'_>, copy_size: wgpu::Extent3d) {
        let b = cast(destination.buffer.as_custom::<MxBuffer>(), "buffer");
        let dst = mx::TexelCopyBufferInfo {
            buffer: &b.0.mx,
            layout: conv::layout(destination.layout),
        };
        self.with(|e| e.copy_texture_to_buffer(copy_texture(&source), dst, conv::extent(copy_size)));
    }

    fn copy_texture_to_texture(&self, source: wgpu::TexelCopyTextureInfo<'_>, destination: wgpu::TexelCopyTextureInfo<'_>, copy_size: wgpu::Extent3d) {
        self.with(|e| e.copy_texture_to_texture(copy_texture(&source), copy_texture(&destination), conv::extent(copy_size)));
    }

    fn begin_compute_pass(&self, desc: &wgpu::ComputePassDescriptor<'_>) -> DispatchComputePass {
        DispatchComputePass::custom(MxComputePass {
            enc: Some(self.0.clone()),
            label: desc.label.map(str::to_owned),
            timestamps: desc.timestamp_writes.as_ref().map(|t| {
                (
                    cast(t.query_set.as_custom::<MxQuerySet>(), "query set").0.clone(),
                    t.beginning_of_pass_write_index,
                    t.end_of_pass_write_index,
                )
            }),
            cmds: Vec::new(),
        })
    }

    fn begin_render_pass(&self, desc: &wgpu::RenderPassDescriptor<'_>) -> DispatchRenderPass {
        let view = |v: &wgpu::TextureView| cast(v.as_custom::<MxTextureView>(), "texture view").0.clone();
        let colors = desc
            .color_attachments
            .iter()
            .map(|a| a.as_ref().map(|a| (view(a.view), a.resolve_target.map(view), conv::ops(a.ops, conv::color))))
            .collect();
        let depth = desc
            .depth_stencil_attachment
            .as_ref()
            .map(|d| (view(d.view), d.depth_ops.map(|o| conv::ops(o, |v| v)), d.stencil_ops.map(|o| conv::ops(o, |v| v))));
        DispatchRenderPass::custom(MxRenderPass {
            data: Some(PassData {
                enc: self.0.clone(),
                label: desc.label.map(str::to_owned),
                colors,
                depth,
                timestamps: desc.timestamp_writes.as_ref().map(|t| {
                    (
                        cast(t.query_set.as_custom::<MxQuerySet>(), "query set").0.clone(),
                        t.beginning_of_pass_write_index,
                        t.end_of_pass_write_index,
                    )
                }),
                occlusion: desc.occlusion_query_set.map(|q| cast(q.as_custom::<MxQuerySet>(), "query set").0.clone()),
                cmds: Vec::new(),
            }),
        })
    }

    fn finish(&mut self) -> DispatchCommandBuffer {
        let cb = self.0.lock().take().map(|e| e.finish());
        DispatchCommandBuffer::custom(MxCommandBuffer(Mutex::new(cb)))
    }

    fn clear_texture(&self, texture: &DispatchTexture, subresource_range: &wgpu::ImageSubresourceRange) {
        let t = cast(texture.as_custom::<MxTexture>(), "texture");
        let r = mx::ImageSubresourceRange {
            aspect: conv::aspect(subresource_range.aspect),
            base_mip_level: subresource_range.base_mip_level,
            mip_level_count: subresource_range.mip_level_count,
            base_array_layer: subresource_range.base_array_layer,
            array_layer_count: subresource_range.array_layer_count,
        };
        self.with(|e| e.clear_texture(&t.0, &r));
    }

    fn clear_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, size: Option<wgpu::BufferAddress>) {
        let b = mx_buffer(buffer);
        self.with(|e| e.clear_buffer(&b.mx, offset, size));
    }

    fn insert_debug_marker(&self, label: &str) {
        self.with(|e| e.insert_debug_marker(label));
    }

    fn push_debug_group(&self, label: &str) {
        self.with(|e| e.push_debug_group(label));
    }

    fn pop_debug_group(&self) {
        self.with(|e| e.pop_debug_group());
    }

    fn write_timestamp(&self, query_set: &DispatchQuerySet, query_index: u32) {
        let q = cast(query_set.as_custom::<MxQuerySet>(), "query set");
        self.with(|e| e.write_timestamp(&q.0, query_index));
    }

    fn resolve_query_set(&self, query_set: &DispatchQuerySet, first_query: u32, query_count: u32, destination: &DispatchBuffer, destination_offset: wgpu::BufferAddress) {
        let q = cast(query_set.as_custom::<MxQuerySet>(), "query set");
        let d = mx_buffer(destination);
        self.with(|e| e.resolve_query_set(&q.0, first_query..first_query + query_count, &d.mx, destination_offset));
    }

    fn mark_acceleration_structures_built<'a>(&self, _blas: &mut dyn Iterator<Item = &'a wgpu::Blas>, _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>) {}

    fn build_acceleration_structures<'a>(&self, _blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>, _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>) {
        panic!("mX: acceleration structures are not offered through wgpu's custom backends")
    }

    fn transition_resources<'a>(
        &mut self,
        _buffer_transitions: &mut dyn Iterator<Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>>,
        _texture_transitions: &mut dyn Iterator<Item = wgpu::wgt::TextureTransition<&'a DispatchTexture>>,
    ) {
    }
}

#[derive(Clone)]
enum Cmd {
    Pipeline(Arc<Option<mx::RenderPipeline>>),
    BindGroup(u32, Arc<BindGroupInner>, Vec<u32>),
    VertexBuffer(u32, Arc<BufferInner>, u64),
    IndexBuffer(Arc<BufferInner>, u64, wgpu::IndexFormat),
    Viewport([f32; 6]),
    Scissor([u32; 4]),
    StencilReference(u32),
    BlendConstant(wgpu::Color),
    Immediates(u32, Vec<u8>),
    Draw(Range<u32>, Range<u32>),
    DrawIndexed(Range<u32>, i32, Range<u32>),
    DrawIndirect(Arc<BufferInner>, u64),
    DrawIndexedIndirect(Arc<BufferInner>, u64),
    MultiDrawIndirect(Arc<BufferInner>, u64, u32),
    MultiDrawIndexedIndirect(Arc<BufferInner>, u64, u32),
    BeginOcclusion(u32),
    EndOcclusion,
    PushDebug(String),
    PopDebug,
    Marker(String),
    Bundle(Arc<Vec<Cmd>>),
}

/// Replay `cmds` into an mX render pass. Draws after an invalid pipeline are skipped.
fn replay(rp: &mut mx::RenderPass<'_>, cmds: &[Cmd], valid: &mut bool) {
    for c in cmds {
        match c {
            Cmd::Pipeline(p) => match p.as_ref() {
                Some(p) => {
                    rp.set_pipeline(p);
                    *valid = true;
                }
                None => *valid = false,
            },
            Cmd::BindGroup(i, g, o) => rp.set_bind_group(*i, &g.with_offsets(o)),
            Cmd::VertexBuffer(s, b, o) => rp.set_vertex_buffer(*s, &b.mx, *o),
            Cmd::IndexBuffer(b, o, f) => rp.set_index_buffer(&b.mx, *o, conv::index_format(*f)),
            Cmd::Viewport(v) => rp.set_viewport(v[0], v[1], v[2], v[3], v[4], v[5]),
            Cmd::Scissor(s) => rp.set_scissor_rect(s[0], s[1], s[2], s[3]),
            Cmd::StencilReference(r) => rp.set_stencil_reference(*r),
            Cmd::BlendConstant(c) => rp.set_blend_constant(conv::color(*c)),
            Cmd::Immediates(o, d) => rp.set_immediates(*o, d),
            Cmd::Draw(v, i) if *valid => rp.draw(v.clone(), i.clone()),
            Cmd::DrawIndexed(ix, b, i) if *valid => rp.draw_indexed(ix.clone(), *b, i.clone()),
            Cmd::DrawIndirect(b, o) if *valid => rp.draw_indirect(&b.mx, *o),
            Cmd::DrawIndexedIndirect(b, o) if *valid => rp.draw_indexed_indirect(&b.mx, *o),
            Cmd::MultiDrawIndirect(b, o, n) if *valid => rp.multi_draw_indirect(&b.mx, *o, *n),
            Cmd::MultiDrawIndexedIndirect(b, o, n) if *valid => rp.multi_draw_indexed_indirect(&b.mx, *o, *n),
            Cmd::Draw(..) | Cmd::DrawIndexed(..) | Cmd::DrawIndirect(..) | Cmd::DrawIndexedIndirect(..) | Cmd::MultiDrawIndirect(..) | Cmd::MultiDrawIndexedIndirect(..) => {}
            Cmd::BeginOcclusion(i) => rp.begin_occlusion_query(*i),
            Cmd::EndOcclusion => rp.end_occlusion_query(),
            Cmd::PushDebug(l) => rp.push_debug_group(l),
            Cmd::PopDebug => rp.pop_debug_group(),
            Cmd::Marker(l) => rp.insert_debug_marker(l),
            Cmd::Bundle(b) => {
                let mut v = false;
                replay(rp, b, &mut v);
                *valid = false;
            }
        }
    }
}

type Attachment = (mx::TextureView, Option<mx::TextureView>, mx::Operations<mx::Color>);
type DepthAttachment = (mx::TextureView, Option<mx::Operations<f32>>, Option<mx::Operations<u32>>);
type Timestamps = (Arc<mx::QuerySet>, Option<u32>, Option<u32>);

struct PassData {
    enc: EncoderCell,
    label: Option<String>,
    colors: Vec<Option<Attachment>>,
    depth: Option<DepthAttachment>,
    timestamps: Option<Timestamps>,
    occlusion: Option<Arc<mx::QuerySet>>,
    cmds: Vec<Cmd>,
}

impl PassData {
    fn run(self) {
        let mut guard = self.enc.lock();
        let Some(enc) = guard.as_mut() else { return };
        let colors: Vec<Option<mx::RenderPassColorAttachment<'_>>> = self
            .colors
            .iter()
            .map(|c| {
                c.as_ref().map(|(v, r, ops)| mx::RenderPassColorAttachment {
                    view: v,
                    resolve_target: r.as_ref(),
                    ops: *ops,
                })
            })
            .collect();
        let depth = self.depth.as_ref().map(|(v, d, s)| mx::RenderPassDepthStencilAttachment {
            view: v,
            depth_ops: *d,
            stencil_ops: *s,
        });
        let timestamps = self.timestamps.as_ref().map(|(q, b, e)| mx::PassTimestampWrites {
            query_set: &**q,
            beginning_of_pass_write_index: *b,
            end_of_pass_write_index: *e,
        });
        let mut rp = enc.begin_render_pass(&mx::RenderPassDescriptor {
            label: self.label.as_deref(),
            color_attachments: &colors,
            depth_stencil_attachment: depth,
            timestamp_writes: timestamps,
            occlusion_query_set: self.occlusion.as_deref(),
        });
        let mut valid = false;
        replay(&mut rp, &self.cmds, &mut valid);
    }
}

/// A render pass: recorded, then replayed into mX when it ends (dropped).
struct MxRenderPass {
    data: Option<PassData>,
}

impl Drop for MxRenderPass {
    fn drop(&mut self) {
        if let Some(d) = self.data.take() {
            d.run();
        }
    }
}

impl MxRenderPass {
    fn push(&mut self, c: Cmd) {
        self.data.as_mut().expect("mX: the pass has ended").cmds.push(c);
    }
}

fn pipeline_of(p: &DispatchRenderPipeline) -> Arc<Option<mx::RenderPipeline>> {
    cast(p.as_custom::<MxRenderPipeline>(), "render pipeline").0.clone()
}

fn group_of(g: &DispatchBindGroup) -> Arc<BindGroupInner> {
    cast(g.as_custom::<MxBindGroup>(), "bind group").0.clone()
}

impl RenderPassInterface for MxRenderPass {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline) {
        self.push(Cmd::Pipeline(pipeline_of(pipeline)));
    }

    fn set_bind_group(&mut self, index: u32, bind_group: Option<&DispatchBindGroup>, offsets: &[wgpu::DynamicOffset]) {
        if let Some(g) = bind_group {
            self.push(Cmd::BindGroup(index, group_of(g), offsets.to_vec()));
        }
    }

    fn set_index_buffer(&mut self, buffer: &DispatchBuffer, index_format: wgpu::IndexFormat, offset: wgpu::BufferAddress, _size: Option<wgpu::BufferSize>) {
        self.push(Cmd::IndexBuffer(mx_buffer(buffer), offset, index_format));
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, _size: Option<wgpu::BufferSize>) {
        self.push(Cmd::VertexBuffer(slot, mx_buffer(buffer), offset));
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.push(Cmd::Immediates(offset, data.to_vec()));
    }

    fn set_blend_constant(&mut self, color: wgpu::Color) {
        self.push(Cmd::BlendConstant(color));
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.push(Cmd::Scissor([x, y, width, height]));
    }

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        self.push(Cmd::Viewport([x, y, width, height, min_depth, max_depth]));
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        self.push(Cmd::StencilReference(reference));
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.push(Cmd::Draw(vertices, instances));
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }

    fn draw_mesh_tasks(&mut self, _x: u32, _y: u32, _z: u32) {}

    fn draw_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress) {
        self.push(Cmd::DrawIndirect(mx_buffer(indirect_buffer), indirect_offset));
    }

    fn draw_indexed_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress) {
        self.push(Cmd::DrawIndexedIndirect(mx_buffer(indirect_buffer), indirect_offset));
    }

    fn draw_mesh_tasks_indirect(&mut self, _indirect_buffer: &DispatchBuffer, _indirect_offset: wgpu::BufferAddress) {}

    fn multi_draw_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress, count: u32) {
        self.push(Cmd::MultiDrawIndirect(mx_buffer(indirect_buffer), indirect_offset, count));
    }

    fn multi_draw_indexed_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress, count: u32) {
        self.push(Cmd::MultiDrawIndexedIndirect(mx_buffer(indirect_buffer), indirect_offset, count));
    }

    fn multi_draw_indirect_count(&mut self, _b: &DispatchBuffer, _o: wgpu::BufferAddress, _cb: &DispatchBuffer, _co: wgpu::BufferAddress, _max: u32) {}

    fn multi_draw_mesh_tasks_indirect(&mut self, _b: &DispatchBuffer, _o: wgpu::BufferAddress, _count: u32) {}

    fn multi_draw_indexed_indirect_count(&mut self, _b: &DispatchBuffer, _o: wgpu::BufferAddress, _cb: &DispatchBuffer, _co: wgpu::BufferAddress, _max: u32) {}

    fn multi_draw_mesh_tasks_indirect_count(&mut self, _b: &DispatchBuffer, _o: wgpu::BufferAddress, _cb: &DispatchBuffer, _co: wgpu::BufferAddress, _max: u32) {}

    fn insert_debug_marker(&mut self, label: &str) {
        self.push(Cmd::Marker(label.into()));
    }

    fn push_debug_group(&mut self, group_label: &str) {
        self.push(Cmd::PushDebug(group_label.into()));
    }

    fn pop_debug_group(&mut self) {
        self.push(Cmd::PopDebug);
    }

    fn write_timestamp(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {}

    fn begin_occlusion_query(&mut self, query_index: u32) {
        self.push(Cmd::BeginOcclusion(query_index));
    }

    fn end_occlusion_query(&mut self) {
        self.push(Cmd::EndOcclusion);
    }

    fn begin_pipeline_statistics_query(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {}

    fn end_pipeline_statistics_query(&mut self) {}

    fn execute_bundles(&mut self, render_bundles: &mut dyn Iterator<Item = &DispatchRenderBundle>) {
        for b in render_bundles {
            let b = cast(b.as_custom::<MxRenderBundle>(), "render bundle").0.clone();
            self.push(Cmd::Bundle(b));
        }
    }
}

#[derive(Clone)]
enum CCmd {
    Pipeline(Arc<Option<mx::ComputePipeline>>),
    BindGroup(u32, Arc<BindGroupInner>, Vec<u32>),
    Immediates(u32, Vec<u8>),
    Dispatch(u32, u32, u32),
    DispatchIndirect(Arc<BufferInner>, u64),
    PushDebug(String),
    PopDebug,
    Marker(String),
}

/// A compute pass: recorded, then replayed into mX when it ends (dropped).
struct MxComputePass {
    enc: Option<EncoderCell>,
    label: Option<String>,
    timestamps: Option<Timestamps>,
    cmds: Vec<CCmd>,
}

impl Drop for MxComputePass {
    fn drop(&mut self) {
        let Some(cell) = self.enc.take() else { return };
        let mut guard = cell.lock();
        let Some(enc) = guard.as_mut() else { return };
        let timestamps = self.timestamps.as_ref().map(|(q, b, e)| mx::PassTimestampWrites {
            query_set: &**q,
            beginning_of_pass_write_index: *b,
            end_of_pass_write_index: *e,
        });
        let mut cp = enc.begin_compute_pass(&mx::ComputePassDescriptor {
            label: self.label.as_deref(),
            timestamp_writes: timestamps,
        });
        let mut valid = false;
        for c in &self.cmds {
            match c {
                CCmd::Pipeline(p) => match p.as_ref() {
                    Some(p) => {
                        cp.set_pipeline(p);
                        valid = true;
                    }
                    None => valid = false,
                },
                CCmd::BindGroup(i, g, o) => cp.set_bind_group(*i, &g.with_offsets(o)),
                CCmd::Immediates(o, d) => cp.set_immediates(*o, d),
                CCmd::Dispatch(x, y, z) if valid => cp.dispatch_workgroups(*x, *y, *z),
                CCmd::DispatchIndirect(b, o) if valid => cp.dispatch_workgroups_indirect(&b.mx, *o),
                CCmd::Dispatch(..) | CCmd::DispatchIndirect(..) => {}
                CCmd::PushDebug(l) => cp.push_debug_group(l),
                CCmd::PopDebug => cp.pop_debug_group(),
                CCmd::Marker(l) => cp.insert_debug_marker(l),
            }
        }
    }
}

impl ComputePassInterface for MxComputePass {
    fn set_pipeline(&mut self, pipeline: &DispatchComputePipeline) {
        let p = cast(pipeline.as_custom::<MxComputePipeline>(), "compute pipeline").0.clone();
        self.cmds.push(CCmd::Pipeline(p));
    }

    fn set_bind_group(&mut self, index: u32, bind_group: Option<&DispatchBindGroup>, offsets: &[wgpu::DynamicOffset]) {
        if let Some(g) = bind_group {
            self.cmds.push(CCmd::BindGroup(index, group_of(g), offsets.to_vec()));
        }
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.cmds.push(CCmd::Immediates(offset, data.to_vec()));
    }

    fn insert_debug_marker(&mut self, label: &str) {
        self.cmds.push(CCmd::Marker(label.into()));
    }

    fn push_debug_group(&mut self, group_label: &str) {
        self.cmds.push(CCmd::PushDebug(group_label.into()));
    }

    fn pop_debug_group(&mut self) {
        self.cmds.push(CCmd::PopDebug);
    }

    fn write_timestamp(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {}

    fn begin_pipeline_statistics_query(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {}

    fn end_pipeline_statistics_query(&mut self) {}

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.cmds.push(CCmd::Dispatch(x, y, z));
    }

    fn dispatch_workgroups_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress) {
        self.cmds.push(CCmd::DispatchIndirect(mx_buffer(indirect_buffer), indirect_offset));
    }
}

struct MxRenderBundleEncoder(Vec<Cmd>);

impl RenderBundleEncoderInterface for MxRenderBundleEncoder {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline) {
        self.0.push(Cmd::Pipeline(pipeline_of(pipeline)));
    }

    fn set_bind_group(&mut self, index: u32, bind_group: Option<&DispatchBindGroup>, offsets: &[wgpu::DynamicOffset]) {
        if let Some(g) = bind_group {
            self.0.push(Cmd::BindGroup(index, group_of(g), offsets.to_vec()));
        }
    }

    fn set_index_buffer(&mut self, buffer: &DispatchBuffer, index_format: wgpu::IndexFormat, offset: wgpu::BufferAddress, _size: Option<wgpu::BufferSize>) {
        self.0.push(Cmd::IndexBuffer(mx_buffer(buffer), offset, index_format));
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, _size: Option<wgpu::BufferSize>) {
        self.0.push(Cmd::VertexBuffer(slot, mx_buffer(buffer), offset));
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.0.push(Cmd::Immediates(offset, data.to_vec()));
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.0.push(Cmd::Draw(vertices, instances));
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.0.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }

    fn draw_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress) {
        self.0.push(Cmd::DrawIndirect(mx_buffer(indirect_buffer), indirect_offset));
    }

    fn draw_indexed_indirect(&mut self, indirect_buffer: &DispatchBuffer, indirect_offset: wgpu::BufferAddress) {
        self.0.push(Cmd::DrawIndexedIndirect(mx_buffer(indirect_buffer), indirect_offset));
    }

    fn finish(self, _desc: &wgpu::RenderBundleDescriptor<'_>) -> DispatchRenderBundle {
        DispatchRenderBundle::custom(MxRenderBundle(Arc::new(self.0)))
    }
}

struct MxRenderBundle(Arc<Vec<Cmd>>);
impl RenderBundleInterface for MxRenderBundle {}

// ---------------------------------------------------------------------------------------------
// Surfaces

struct MxSurface {
    inner: Arc<mx::Surface<'static>>,
    config: Mutex<Option<wgpu::SurfaceConfiguration>>,
}

impl SurfaceInterface for MxSurface {
    fn get_capabilities(&self, adapter: &DispatchAdapter) -> wgpu::SurfaceCapabilities {
        let a = cast(adapter.as_custom::<MxAdapter>(), "adapter");
        let c = self.inner.get_capabilities(&a.0);
        wgpu::SurfaceCapabilities {
            formats: c.formats.into_iter().map(conv::format_back).collect(),
            present_modes: c.present_modes.into_iter().map(conv::present_mode_back).collect(),
            alpha_modes: c.alpha_modes.into_iter().map(conv::alpha_mode_back).collect(),
            usages: conv::texture_usages_back(c.usages),
        }
    }

    fn configure(&self, device: &DispatchDevice, config: &wgpu::SurfaceConfiguration) {
        let d = cast(device.as_custom::<MxDevice>(), "device");
        self.inner.configure(
            &d.0.mx,
            &mx::SurfaceConfiguration {
                usage: conv::texture_usages(config.usage),
                format: conv::format(config.format),
                width: config.width,
                height: config.height,
                present_mode: conv::present_mode(config.present_mode),
                desired_maximum_frame_latency: config.desired_maximum_frame_latency,
                alpha_mode: conv::alpha_mode(config.alpha_mode),
                view_formats: config.view_formats.iter().map(|f| conv::format(*f)).collect(),
            },
        );
        *self.config.lock() = Some(config.clone());
    }

    fn get_current_texture(&self) -> (Option<DispatchTexture>, wgpu::SurfaceStatus, DispatchSurfaceOutputDetail) {
        let none = || DispatchSurfaceOutputDetail::custom(MxSurfaceOutput(Mutex::new(None)));
        let frame = |s: mx::SurfaceTexture| {
            (
                Some(DispatchTexture::custom(MxTexture(s.texture.clone()))),
                DispatchSurfaceOutputDetail::custom(MxSurfaceOutput(Mutex::new(Some(s)))),
            )
        };
        match self.inner.get_current_texture() {
            mx::CurrentSurfaceTexture::Success(s) => {
                let (t, d) = frame(s);
                (t, wgpu::SurfaceStatus::Good, d)
            }
            mx::CurrentSurfaceTexture::Suboptimal(s) => {
                let (t, d) = frame(s);
                (t, wgpu::SurfaceStatus::Suboptimal, d)
            }
            mx::CurrentSurfaceTexture::Timeout => (None, wgpu::SurfaceStatus::Timeout, none()),
            mx::CurrentSurfaceTexture::Occluded => (None, wgpu::SurfaceStatus::Occluded, none()),
            mx::CurrentSurfaceTexture::Outdated => (None, wgpu::SurfaceStatus::Outdated, none()),
            mx::CurrentSurfaceTexture::Lost => (None, wgpu::SurfaceStatus::Lost, none()),
        }
    }
}

struct MxSurfaceOutput(Mutex<Option<mx::SurfaceTexture>>);

impl SurfaceOutputDetailInterface for MxSurfaceOutput {
    fn present(&self) {
        if let Some(s) = self.0.lock().take() {
            s.present();
        }
    }

    fn texture_discard(&self) {
        self.0.lock().take();
    }
}
