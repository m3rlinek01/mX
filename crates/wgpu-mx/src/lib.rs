//! wgpu's API (version 29, as openOMSI uses it) implemented on mX.
//!
//! Code written for wgpu builds against this crate unchanged (depend on it under the name
//! `wgpu`) and draws through mX: no wgpu-core, no wgpu-hal. The plain data types are
//! wgpu-types' own; the objects are thin handles over mX's. What wgpu has and mX does not
//! (bind group layouts, pipeline layouts) is kept only as far as callers can observe it.
//!
//! - Render and compute passes are recorded and replayed into mX when they end, so they
//!   hold no borrow of their encoder (`forget_lifetime` is free) and can resolve dynamic
//!   offsets when they are replayed.
//! - Dynamic offsets: a bind group is rebuilt (and cached) for each set of offsets.
//! - Errors: a shader or pipeline that fails goes to the innermost error scope (or the
//!   uncaptured-error handler, or the log); the object comes back invalid and draws with it
//!   are skipped, as wgpu skips commands on invalid objects.
//! - `map_async` completes once the GPU has finished the work submitted before it, on the
//!   next `poll` or `submit`.

#![allow(clippy::new_without_default, clippy::too_many_arguments)]

mod conv;
#[cfg(feature = "custom")]
pub mod custom;
pub mod util;

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::ops::{Bound, Deref, Range, RangeBounds};
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;

pub use naga;
pub use raw_window_handle as rwh;
use wgpu_types as wgt;
pub use wgpu_types::*;

pub type Label<'a> = Option<&'a str>;
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;
pub type CommandEncoderDescriptor<'a> = wgt::CommandEncoderDescriptor<Label<'a>>;
pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;
pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;
pub type SamplerDescriptor<'a> = wgt::SamplerDescriptor<Label<'a>>;
pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, &'a [TextureFormat]>;
pub type TextureViewDescriptor<'a> = wgt::TextureViewDescriptor<Label<'a>>;
pub type SurfaceConfiguration = wgt::SurfaceConfiguration<Vec<TextureFormat>>;
pub type TexelCopyBufferInfo<'a> = wgt::TexelCopyBufferInfo<&'a Buffer>;
pub type TexelCopyTextureInfo<'a> = wgt::TexelCopyTextureInfo<&'a Texture>;
pub type RequestAdapterOptions<'a, 'b> = wgt::RequestAdapterOptions<&'a Surface<'b>>;
pub type PollType = wgt::PollType<SubmissionIndex>;
pub type CreateBlasDescriptor<'a> = wgt::CreateBlasDescriptor<Label<'a>>;
pub type CreateTlasDescriptor<'a> = wgt::CreateTlasDescriptor<Label<'a>>;
pub type ErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Objects compare and hash by identity, as wgpu's do.
/// Work to run once a submission is done (`map_async`, `on_submitted_work_done`).
type Callback = Box<dyn FnOnce() + Send>;

macro_rules! identity {
    ($($t:ident),*) => {$(
        impl PartialEq for $t {
            fn eq(&self, o: &Self) -> bool {
                Arc::ptr_eq(&self.inner, &o.inner)
            }
        }
        impl Eq for $t {}
        impl std::hash::Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
                (Arc::as_ptr(&self.inner) as *const u8 as usize).hash(h)
            }
        }
        impl PartialOrd for $t {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for $t {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                (Arc::as_ptr(&self.inner) as *const u8 as usize).cmp(&(Arc::as_ptr(&o.inner) as *const u8 as usize))
            }
        }
        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!(stringify!($t), "({:p})"), Arc::as_ptr(&self.inner))
            }
        }
    )*};
}

fn range_of<S: RangeBounds<u64>>(bounds: S, size: u64) -> Range<u64> {
    let start = match bounds.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match bounds.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => size,
    };
    start..end
}

// ---------------------------------------------------------------------------------------------
// Errors

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorFilter {
    OutOfMemory,
    Validation,
    Internal,
}

#[derive(Debug)]
pub enum Error {
    OutOfMemory { source: ErrorSource },
    Validation { source: ErrorSource, description: String },
    Internal { source: ErrorSource, description: String },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OutOfMemory { .. } => write!(f, "Out of Memory"),
            Error::Validation { description, .. } | Error::Internal { description, .. } => f.write_str(description),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
struct Message(String);
impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Message {}

fn validation(description: String) -> Error {
    Error::Validation {
        source: Box::new(Message(description.clone())),
        description,
    }
}

pub trait UncapturedErrorHandler: Fn(Error) + Send + Sync + 'static {}
impl<T: Fn(Error) + Send + Sync + 'static> UncapturedErrorHandler for T {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestDeviceError(String);
impl std::fmt::Display for RequestDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "requesting a device on mX: {}", self.0)
    }
}
impl std::error::Error for RequestDeviceError {}

#[derive(Clone, Debug)]
pub struct CreateSurfaceError(String);
impl std::fmt::Display for CreateSurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "creating a surface on mX: {}", self.0)
    }
}
impl std::error::Error for CreateSurfaceError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferAsyncError;
impl std::fmt::Display for BufferAsyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error occurred when trying to async map a buffer")
    }
}
impl std::error::Error for BufferAsyncError {}

// ---------------------------------------------------------------------------------------------
// Instance, adapter

#[derive(Clone)]
pub struct Instance {
    inner: Arc<mx::Instance>,
    backends: Backends,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Instance(mX)")
    }
}

impl Default for Instance {
    fn default() -> Self {
        Instance::new(InstanceDescriptor::new_without_display_handle())
    }
}

/// The backend mX runs on, as wgpu names it.
fn mx_backend() -> Backend {
    if cfg!(target_vendor = "apple") {
        Backend::Metal
    } else if cfg!(windows) {
        Backend::Dx12
    } else {
        Backend::Vulkan
    }
}

impl Instance {
    pub fn new(desc: InstanceDescriptor) -> Self {
        Instance {
            inner: Arc::new(mx::Instance::new()),
            backends: desc.backends,
        }
    }

    fn serves(&self, backends: Backends) -> bool {
        backends.contains(Backends::from(mx_backend())) && self.backends.contains(Backends::from(mx_backend()))
    }

    pub fn enumerate_adapters(&self, backends: Backends) -> impl Future<Output = Vec<Adapter>> + Send + use<> {
        let list = if self.serves(backends) {
            self.inner.enumerate_adapters().into_iter().map(Adapter::new).collect()
        } else {
            Vec::new()
        };
        std::future::ready(list)
    }

    pub fn request_adapter(&self, _options: &RequestAdapterOptions<'_, '_>) -> impl Future<Output = Result<Adapter, RequestAdapterError>> + Send + use<> {
        let all = Backends::all();
        let r = if self.serves(all) { self.inner.default_adapter().ok().map(Adapter::new) } else { None };
        std::future::ready(r.ok_or(RequestAdapterError::NotFound {
            active_backends: self.backends,
            requested_backends: self.backends,
            supported_backends: Backends::from(mx_backend()),
            no_fallback_backends: Backends::empty(),
            no_adapter_backends: Backends::from(mx_backend()),
            incompatible_surface_backends: Backends::empty(),
        }))
    }

    pub fn create_surface<'w>(&self, target: impl Into<SurfaceTarget<'w>>) -> Result<Surface<'w>, CreateSurfaceError> {
        let s = match target.into() {
            SurfaceTarget::DisplayAndWindow(h) => self.inner.create_surface(Both(h)),
            SurfaceTarget::Window(h) => self.inner.create_surface(WindowOnly(h)),
        }
        .map_err(|e| CreateSurfaceError(e.to_string()))?;
        Ok(Surface {
            inner: Arc::new(s),
            config: Mutex::new(None),
        })
    }

    pub fn poll_all(&self, _force_wait: bool) -> bool {
        true
    }
}

/// A boxed window and display handle, as mX takes them.
struct Both<'w>(Box<dyn DisplayAndWindowHandle + 'w>);
impl rwh::HasWindowHandle for Both<'_> {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        self.0.window_handle()
    }
}
impl rwh::HasDisplayHandle for Both<'_> {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        self.0.display_handle()
    }
}

/// A window handle passed without a display one: the display is the platform's only one.
struct WindowOnly<'w>(Box<dyn WindowHandle + 'w>);
impl rwh::HasWindowHandle for WindowOnly<'_> {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        self.0.window_handle()
    }
}
impl rwh::HasDisplayHandle for WindowOnly<'_> {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        #[cfg(target_vendor = "apple")]
        {
            Ok(rwh::DisplayHandle::appkit())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(rwh::HandleError::NotSupported)
        }
    }
}

pub trait WindowHandle: rwh::HasWindowHandle + Send + Sync {}
impl<T: rwh::HasWindowHandle + Send + Sync> WindowHandle for T {}
pub trait DisplayAndWindowHandle: WindowHandle + rwh::HasDisplayHandle {}
impl<T: WindowHandle + rwh::HasDisplayHandle> DisplayAndWindowHandle for T {}

pub enum SurfaceTarget<'w> {
    DisplayAndWindow(Box<dyn DisplayAndWindowHandle + 'w>),
    Window(Box<dyn WindowHandle + 'w>),
}

impl<'a, T: DisplayAndWindowHandle + 'a> From<T> for SurfaceTarget<'a> {
    fn from(t: T) -> Self {
        SurfaceTarget::DisplayAndWindow(Box::new(t))
    }
}

impl SurfaceTarget<'_> {
    pub fn from_window_without_display<'a>(window: impl WindowHandle + 'a) -> SurfaceTarget<'a> {
        SurfaceTarget::Window(Box::new(window))
    }
}

#[derive(Clone)]
pub struct Adapter {
    inner: Arc<mx::Adapter>,
}
identity!(Adapter);

impl Adapter {
    fn new(a: mx::Adapter) -> Self {
        Adapter { inner: Arc::new(a) }
    }

    pub fn features(&self) -> Features {
        // (adapter-specific format features: Metal takes 2× MSAA besides WebGPU's guaranteed 4×)
        let mut f = Features::IMMEDIATES | Features::DEPTH32FLOAT_STENCIL8 | Features::INDIRECT_FIRST_INSTANCE | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | Features::CLEAR_TEXTURE;
        if self.inner.supports_bc() {
            f |= Features::TEXTURE_COMPRESSION_BC;
        }
        if self.inner.supports_timestamps() {
            f |= Features::TIMESTAMP_QUERY;
        }
        if self.inner.supports_ray_query() {
            f |= Features::EXPERIMENTAL_RAY_QUERY;
        }
        f
    }

    pub fn limits(&self) -> Limits {
        limits(&self.inner.limits())
    }

    pub fn get_info(&self) -> AdapterInfo {
        let i = self.inner.info();
        AdapterInfo {
            name: i.name,
            vendor: i.vendor,
            device: i.device,
            device_type: match i.device_type {
                mx::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
                mx::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
                mx::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
                mx::DeviceType::Cpu => DeviceType::Cpu,
                mx::DeviceType::Other => DeviceType::Other,
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

    pub fn get_downlevel_capabilities(&self) -> DownlevelCapabilities {
        DownlevelCapabilities::default()
    }

    pub fn get_texture_format_features(&self, format: TextureFormat) -> TextureFormatFeatures {
        let mut f = format.guaranteed_format_features(self.features());
        if f.flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4) {
            f.flags |= TextureFormatFeatureFlags::MULTISAMPLE_X2;
        }
        f
    }

    pub fn is_surface_supported(&self, _surface: &Surface<'_>) -> bool {
        true
    }

    pub fn memory_info(&self) -> Option<AdapterMemoryInfo> {
        None
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> impl Future<Output = Result<(Device, Queue), RequestDeviceError>> + Send + use<> {
        let r = self.open(desc);
        std::future::ready(r)
    }

    fn open(&self, desc: &DeviceDescriptor<'_>) -> Result<(Device, Queue), RequestDeviceError> {
        let missing = desc.required_features - self.features();
        if !missing.is_empty() {
            return Err(RequestDeviceError(format!("features mX does not have: {missing:?}")));
        }
        let (d, q) = self
            .inner
            .request_device(&mx::DeviceDescriptor {
                label: desc.label,
                ..Default::default()
            })
            .map_err(|e| RequestDeviceError(e.to_string()))?;
        let inner = Arc::new(DeviceInner {
            mx: d,
            queue: q,
            features: desc.required_features,
            limits: self.limits(),
            info: self.get_info(),
            scopes: Mutex::new(Vec::new()),
            uncaptured: Mutex::new(None),
            maps: Mutex::new(Vec::new()),
            done_callbacks: Mutex::new(Vec::new()),
            last_submission: Mutex::new(mx::SubmissionIndex(0)),
        });
        Ok((Device { inner: inner.clone() }, Queue { inner }))
    }
}

fn limits(l: &mx::Limits) -> Limits {
    Limits {
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
        ..Limits::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Device, queue

struct PendingMap {
    buffer: Buffer,
    until: mx::SubmissionIndex,
    callback: MapCallback,
}

struct DeviceInner {
    mx: mx::Device,
    queue: mx::Queue,
    features: Features,
    limits: Limits,
    info: AdapterInfo,
    scopes: Mutex<Vec<(ErrorFilter, Option<Error>)>>,
    uncaptured: Mutex<Option<Arc<dyn UncapturedErrorHandler>>>,
    maps: Mutex<Vec<PendingMap>>,
    done_callbacks: Mutex<Vec<(mx::SubmissionIndex, Callback)>>,
    last_submission: Mutex<mx::SubmissionIndex>,
}

impl DeviceInner {
    fn error(&self, e: Error) {
        let filter = match e {
            Error::OutOfMemory { .. } => ErrorFilter::OutOfMemory,
            Error::Validation { .. } => ErrorFilter::Validation,
            Error::Internal { .. } => ErrorFilter::Internal,
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
            None => log::error!("wgpu-mx: uncaptured error: {e}"),
        }
    }

    /// Run the callbacks of finished work (maps, work-done) without waiting.
    fn progress(&self) {
        let _ = self.mx.poll();
        let ready: Vec<PendingMap> = {
            let mut maps = self.maps.lock();
            let (ready, rest): (Vec<_>, Vec<_>) = maps.drain(..).partition(|m| self.mx.is_done(m.until));
            *maps = rest;
            ready
        };
        for m in ready {
            m.buffer.inner.map.lock().mapped = true;
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

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}
identity!(Device);

#[derive(Clone)]
pub struct Queue {
    inner: Arc<DeviceInner>,
}
identity!(Queue);

#[derive(Clone, Debug)]
pub struct SubmissionIndex(u64);

/// Pops its error scope (with `pop`, or when dropped).
pub struct ErrorScopeGuard {
    device: Arc<DeviceInner>,
    popped: bool,
}

impl ErrorScopeGuard {
    pub fn pop(mut self) -> impl Future<Output = Option<Error>> + Send {
        self.popped = true;
        let e = self.device.scopes.lock().pop().and_then(|(_, e)| e);
        std::future::ready(e)
    }
}

impl Drop for ErrorScopeGuard {
    fn drop(&mut self) {
        if !self.popped {
            self.device.scopes.lock().pop();
        }
    }
}

impl Device {
    pub fn poll(&self, poll_type: PollType) -> Result<PollStatus, PollError> {
        let d = &self.inner;
        match poll_type {
            PollType::Poll => {
                d.progress();
                Ok(PollStatus::Poll)
            }
            PollType::Wait { submission_index, .. } => {
                let index = submission_index.as_ref().map(|s| mx::SubmissionIndex(s.0)).unwrap_or(*d.last_submission.lock());
                d.mx.wait(index).map_err(|_| PollError::Timeout)?;
                d.progress();
                Ok(if submission_index.is_some() { PollStatus::WaitSucceeded } else { PollStatus::QueueEmpty })
            }
        }
    }

    pub fn features(&self) -> Features {
        self.inner.features
    }

    pub fn limits(&self) -> Limits {
        self.inner.limits.clone()
    }

    pub fn adapter_info(&self) -> AdapterInfo {
        self.inner.info.clone()
    }

    pub fn push_error_scope(&self, filter: ErrorFilter) -> ErrorScopeGuard {
        self.inner.scopes.lock().push((filter, None));
        ErrorScopeGuard {
            device: self.inner.clone(),
            popped: false,
        }
    }

    pub fn on_uncaptured_error(&self, handler: Arc<dyn UncapturedErrorHandler>) {
        *self.inner.uncaptured.lock() = Some(handler);
    }

    pub fn set_device_lost_callback(&self, _callback: impl Fn(DeviceLostReason, String) + Send + 'static) {}

    pub fn destroy(&self) {}

    pub fn create_shader_module(&self, desc: ShaderModuleDescriptor<'_>) -> ShaderModule {
        let source = match &desc.source {
            ShaderSource::Wgsl(s) => s.as_ref(),
            ShaderSource::Dummy(_) => "",
        };
        let m = match self.inner.mx.create_shader_module(&mx::ShaderModuleDescriptor { label: desc.label, source }) {
            Ok(m) => Some(m),
            Err(e) => {
                self.inner.error(validation(e.to_string()));
                None
            }
        };
        ShaderModule { inner: Arc::new(m) }
    }

    /// # Safety
    /// As wgpu's: mX applies wgpu's checks either way.
    pub unsafe fn create_shader_module_trusted(&self, desc: ShaderModuleDescriptor<'_>, _checks: ShaderRuntimeChecks) -> ShaderModule {
        self.create_shader_module(desc)
    }

    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor<'_>) -> CommandEncoder {
        CommandEncoder {
            inner: Arc::new(Mutex::new(Some(self.inner.mx.create_command_encoder(desc.label)))),
            device: self.inner.clone(),
            maps: Vec::new(),
        }
    }

    pub fn create_render_bundle_encoder<'a>(&self, _desc: &RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder<'a> {
        RenderBundleEncoder { cmds: Vec::new(), _p: PhantomData }
    }

    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor<'_>) -> BindGroupLayout {
        BindGroupLayout {
            inner: Arc::new(desc.entries.to_vec()),
        }
    }

    pub fn create_pipeline_layout(&self, _desc: &PipelineLayoutDescriptor<'_>) -> PipelineLayout {
        PipelineLayout { inner: Arc::new(()) }
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let dynamic: Vec<u32> = {
            let mut v: Vec<u32> = desc
                .layout
                .inner
                .iter()
                .filter(|e| matches!(e.ty, BindingType::Buffer { has_dynamic_offset: true, .. }))
                .map(|e| e.binding)
                .collect();
            v.sort_unstable();
            v
        };
        let entries: Vec<(u32, Owned)> = desc
            .entries
            .iter()
            .map(|e| {
                let r = match &e.resource {
                    BindingResource::Buffer(b) => Owned::Buffer(b.buffer.clone(), b.offset, b.size.map(|s| s.get())),
                    BindingResource::TextureView(v) => Owned::View((*v).clone()),
                    BindingResource::Sampler(s) => Owned::Sampler((*s).clone()),
                    BindingResource::AccelerationStructure(t) => Owned::Tlas(t.inner.clone()),
                    _ => panic!("wgpu-mx: binding arrays are not supported (mX reaches textures through its heap)"),
                };
                (e.binding, r)
            })
            .collect();
        let inner = BindGroupInner {
            device: self.inner.clone(),
            entries,
            dynamic,
            built: Mutex::new(HashMap::new()),
        };
        let base = inner.build(&vec![0; inner.dynamic.len()]);
        inner.built.lock().insert(vec![0; inner.dynamic.len()], base);
        BindGroup { inner: Arc::new(inner) }
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> RenderPipeline {
        let p = self.render_pipeline(desc);
        if let Err(e) = &p {
            self.inner.error(validation(e.clone()));
        }
        RenderPipeline { inner: Arc::new(p.ok()) }
    }

    fn render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> Result<mx::RenderPipeline, String> {
        let label = desc.label.unwrap_or("render pipeline");
        let vm = desc.vertex.module.inner.as_ref().as_ref().ok_or_else(|| format!("{label}: its vertex shader module is invalid"))?;
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
                module: f.module.inner.as_ref().as_ref().ok_or_else(|| format!("{label}: its fragment shader module is invalid"))?,
                entry_point: f.entry_point,
                constants: f.compilation_options.constants,
                targets: &targets,
            }),
            None => None,
        };
        self.inner
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

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor<'_>) -> ComputePipeline {
        let p = match desc.module.inner.as_ref() {
            Some(m) => self
                .inner
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
            self.inner.error(validation(e.clone()));
        }
        ComputePipeline { inner: Arc::new(p.ok()) }
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Buffer {
        let usage = conv::buffer_usages(desc.usage);
        let b = self
            .inner
            .mx
            .create_buffer(&mx::BufferDescriptor {
                label: desc.label,
                size: desc.size,
                usage,
            })
            .unwrap_or_else(|e| panic!("wgpu-mx: {e}"));
        let mappable = desc.usage.intersects(BufferUsages::MAP_READ | BufferUsages::MAP_WRITE);
        let map = MapState {
            mapped: desc.mapped_at_creation,
            staging: (desc.mapped_at_creation && !mappable).then(|| vec![0u8; desc.size as usize]),
        };
        Buffer {
            inner: Arc::new(BufferInner {
                mx: b,
                size: desc.size,
                usage: desc.usage,
                map: Mutex::new(map),
                device: self.inner.clone(),
            }),
        }
    }

    pub fn create_texture(&self, desc: &TextureDescriptor<'_>) -> Texture {
        let view_formats: Vec<mx::TextureFormat> = desc.view_formats.iter().map(|f| conv::format(*f)).collect();
        let t = self
            .inner
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
            .unwrap_or_else(|e| panic!("wgpu-mx: {e}"));
        Texture::wrap(t, desc.size, desc.format, desc.dimension, desc.mip_level_count, desc.sample_count, desc.usage)
    }

    pub fn create_sampler(&self, desc: &SamplerDescriptor<'_>) -> Sampler {
        let s = self.inner.mx.create_sampler(&conv::sampler(desc)).unwrap_or_else(|e| panic!("wgpu-mx: {e}"));
        Sampler { inner: Arc::new(s) }
    }

    pub fn create_query_set(&self, desc: &QuerySetDescriptor<'_>) -> QuerySet {
        let q = self
            .inner
            .mx
            .create_query_set(&mx::QuerySetDescriptor {
                label: desc.label,
                ty: conv::query_type(desc.ty),
                count: desc.count,
            })
            .unwrap_or_else(|e| panic!("wgpu-mx: {e}"));
        QuerySet { inner: Arc::new(q) }
    }

    /// # Safety
    /// Nothing to uphold: mX keeps no pipeline cache of its own (Metal's is the OS's).
    pub unsafe fn create_pipeline_cache(&self, _desc: &PipelineCacheDescriptor<'_>) -> PipelineCache {
        PipelineCache { inner: Arc::new(()) }
    }

    pub fn generate_allocator_report(&self) -> Option<AllocatorReport> {
        None
    }

    pub fn create_blas(&self, desc: &CreateBlasDescriptor<'_>, sizes: BlasGeometrySizeDescriptors) -> Blas {
        self.need_ray_query();
        let BlasGeometrySizeDescriptors::Triangles { descriptors } = sizes;
        let sizes: Vec<mx::BlasTriangleGeometrySizeDescriptor> = descriptors.iter().map(conv::blas_size).collect();
        let d = mx::CreateBlasDescriptor {
            label: desc.label,
            flags: conv::as_flags(desc.flags),
            update_mode: conv::as_update_mode(desc.update_mode),
        };
        let raw = self.inner.mx.create_blas(&d, &sizes).unwrap_or_else(|e| panic!("wgpu-mx: create_blas: {e}"));
        Blas { inner: Arc::new(raw) }
    }

    pub fn create_tlas(&self, desc: &CreateTlasDescriptor<'_>) -> Tlas {
        self.need_ray_query();
        let d = mx::CreateTlasDescriptor {
            label: desc.label,
            max_instances: desc.max_instances,
            flags: conv::as_flags(desc.flags),
            update_mode: conv::as_update_mode(desc.update_mode),
        };
        let raw = self.inner.mx.create_tlas(&d).unwrap_or_else(|e| panic!("wgpu-mx: create_tlas: {e}"));
        Tlas {
            inner: Arc::new(Mutex::new(raw)),
            instances: vec![None; desc.max_instances as usize],
        }
    }

    fn need_ray_query(&self) {
        assert!(
            self.inner.features.contains(Features::EXPERIMENTAL_RAY_QUERY),
            "wgpu-mx: acceleration structures need the EXPERIMENTAL_RAY_QUERY feature"
        );
    }
}

/// The mX objects under wgpu's, for interop (MetalFX, OpenXR, capture: their `raw()`).
impl Device {
    pub fn mx(&self) -> &mx::Device {
        &self.inner.mx
    }
}

impl Queue {
    pub fn mx(&self) -> &mx::Queue {
        &self.inner.queue
    }
}

impl Queue {
    pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]) {
        if let Err(e) = self.inner.queue.write_buffer(&buffer.inner.mx, offset, data) {
            self.inner.error(Error::OutOfMemory {
                source: Box::new(Message(e.to_string())),
            });
        }
    }

    pub fn write_texture(&self, texture: TexelCopyTextureInfo<'_>, data: &[u8], data_layout: TexelCopyBufferLayout, size: Extent3d) {
        let dst = mx::TexelCopyTextureInfo {
            texture: &texture.texture.inner.mx,
            mip_level: texture.mip_level,
            origin: conv::origin(texture.origin),
            aspect: conv::aspect(texture.aspect),
        };
        if let Err(e) = self.inner.queue.write_texture(dst, data, conv::layout(data_layout), conv::extent(size)) {
            self.inner.error(Error::OutOfMemory {
                source: Box::new(Message(e.to_string())),
            });
        }
    }

    pub fn submit<I: IntoIterator<Item = CommandBuffer>>(&self, command_buffers: I) -> SubmissionIndex {
        let mut maps = Vec::new();
        let raw: Vec<mx::CommandBuffer> = command_buffers
            .into_iter()
            .filter_map(|mut c| {
                maps.append(&mut c.maps);
                c.inner.take()
            })
            .collect();
        let i = self.inner.queue.submit(raw);
        *self.inner.last_submission.lock() = i;
        self.inner.maps.lock().extend(maps.into_iter().map(|(buffer, callback)| PendingMap { buffer, until: i, callback }));
        self.inner.progress();
        SubmissionIndex(i.0)
    }

    pub fn get_timestamp_period(&self) -> f32 {
        self.inner.queue.get_timestamp_period()
    }

    pub fn on_submitted_work_done(&self, callback: impl FnOnce() + Send + 'static) {
        let at = *self.inner.last_submission.lock();
        self.inner.done_callbacks.lock().push((at, Box::new(callback)));
    }
}

// ---------------------------------------------------------------------------------------------
// Resources

struct MapState {
    mapped: bool,
    /// The bytes of a buffer mapped at creation that the CPU cannot reach (written on unmap).
    staging: Option<Vec<u8>>,
}

struct BufferInner {
    mx: mx::Buffer,
    size: u64,
    usage: BufferUsages,
    map: Mutex<MapState>,
    device: Arc<DeviceInner>,
}

#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}
identity!(Buffer);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapMode {
    Read,
    Write,
}

impl Buffer {
    pub fn mx(&self) -> &mx::Buffer {
        &self.inner.mx
    }

    pub fn size(&self) -> BufferAddress {
        self.inner.size
    }

    pub fn usage(&self) -> BufferUsages {
        self.inner.usage
    }

    pub fn destroy(&self) {}

    pub fn as_entire_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(self.as_entire_buffer_binding())
    }

    pub fn as_entire_buffer_binding(&self) -> BufferBinding<'_> {
        BufferBinding { buffer: self, offset: 0, size: None }
    }

    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice<'_> {
        let r = range_of(bounds, self.inner.size);
        BufferSlice {
            buffer: self,
            offset: r.start,
            size: BufferSize::new(r.end - r.start).expect("wgpu-mx: an empty buffer slice"),
        }
    }

    pub fn map_async<S: RangeBounds<BufferAddress>>(&self, mode: MapMode, _bounds: S, callback: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static) {
        let needed = if mode == MapMode::Read { BufferUsages::MAP_READ } else { BufferUsages::MAP_WRITE };
        if !self.inner.usage.contains(needed) {
            callback(Err(BufferAsyncError));
            return;
        }
        let until = *self.inner.device.last_submission.lock();
        self.inner.device.maps.lock().push(PendingMap {
            buffer: self.clone(),
            until,
            callback: Box::new(callback),
        });
    }

    fn bytes(&self, r: Range<u64>) -> (NonNull<u8>, usize) {
        let mut m = self.inner.map.lock();
        assert!(m.mapped, "wgpu-mx: get_mapped_range on a buffer that is not mapped");
        let len = (r.end - r.start) as usize;
        let base = match &mut m.staging {
            Some(v) => NonNull::new(v.as_mut_ptr()).unwrap(),
            // SAFETY: the buffer is mapped, so no submitted work uses it.
            None => unsafe { self.inner.mx.mapped_ptr() },
        };
        // SAFETY: in range of the buffer (checked by the caller's slice).
        (unsafe { base.add(r.start as usize) }, len)
    }

    pub fn get_mapped_range<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferView {
        let (ptr, len) = self.bytes(range_of(bounds, self.inner.size));
        BufferView { ptr, len, _buffer: self.clone() }
    }

    pub fn get_mapped_range_mut<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferViewMut {
        let (ptr, len) = self.bytes(range_of(bounds, self.inner.size));
        BufferViewMut { ptr, len, _buffer: self.clone() }
    }

    pub fn unmap(&self) {
        let staged = {
            let mut m = self.inner.map.lock();
            m.mapped = false;
            m.staging.take()
        };
        if let Some(bytes) = staged {
            let _ = self.inner.device.queue.write_buffer(&self.inner.mx, 0, &bytes);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: BufferAddress,
    size: BufferSize,
}

impl<'a> BufferSlice<'a> {
    pub fn map_async(&self, mode: MapMode, callback: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static) {
        self.buffer.map_async(mode, self.offset..self.offset + self.size.get(), callback)
    }
    pub fn get_mapped_range(&self) -> BufferView {
        self.buffer.get_mapped_range(self.offset..self.offset + self.size.get())
    }
    pub fn get_mapped_range_mut(&self) -> BufferViewMut {
        self.buffer.get_mapped_range_mut(self.offset..self.offset + self.size.get())
    }
    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }
    pub fn offset(&self) -> BufferAddress {
        self.offset
    }
    pub fn size(&self) -> BufferSize {
        self.size
    }
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice<'a> {
        let r = range_of(bounds, self.size.get());
        BufferSlice {
            buffer: self.buffer,
            offset: self.offset + r.start,
            size: BufferSize::new(r.end - r.start).expect("wgpu-mx: an empty buffer slice"),
        }
    }
}

impl<'a> From<BufferSlice<'a>> for BufferBinding<'a> {
    fn from(s: BufferSlice<'a>) -> Self {
        BufferBinding {
            buffer: s.buffer,
            offset: s.offset,
            size: Some(s.size),
        }
    }
}

impl<'a> From<BufferSlice<'a>> for BindingResource<'a> {
    fn from(s: BufferSlice<'a>) -> Self {
        BindingResource::Buffer(s.into())
    }
}

pub struct BufferView {
    ptr: NonNull<u8>,
    len: usize,
    _buffer: Buffer,
}
// SAFETY: a view of mapped bytes nothing else writes while it lives.
unsafe impl Send for BufferView {}
unsafe impl Sync for BufferView {}

impl Deref for BufferView {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: mapped, in range, alive as long as the buffer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl AsRef<[u8]> for BufferView {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

pub struct BufferViewMut {
    ptr: NonNull<u8>,
    len: usize,
    _buffer: Buffer,
}
// SAFETY: as `BufferView`.
unsafe impl Send for BufferViewMut {}
unsafe impl Sync for BufferViewMut {}

impl BufferViewMut {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn slice<'a, S: RangeBounds<usize>>(&'a mut self, bounds: S) -> WriteOnly<'a, [u8]> {
        // SAFETY: mapped for writing, in range, borrowed mutably through `self`.
        let all = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) };
        let start = match bounds.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };
        WriteOnly::from_mut(&mut all[start..end])
    }
    pub fn copy_from_slice(&mut self, src: &[u8]) {
        self.slice(..).copy_from_slice(src)
    }
}

struct TextureInner {
    mx: mx::Texture,
    size: Extent3d,
    format: TextureFormat,
    dimension: TextureDimension,
    mip_level_count: u32,
    sample_count: u32,
    usage: TextureUsages,
}

#[derive(Clone)]
pub struct Texture {
    inner: Arc<TextureInner>,
}
identity!(Texture);

impl Texture {
    pub fn mx(&self) -> &mx::Texture {
        &self.inner.mx
    }

    fn wrap(mx: mx::Texture, size: Extent3d, format: TextureFormat, dimension: TextureDimension, mip_level_count: u32, sample_count: u32, usage: TextureUsages) -> Texture {
        Texture {
            inner: Arc::new(TextureInner {
                mx,
                size,
                format,
                dimension,
                mip_level_count,
                sample_count,
                usage,
            }),
        }
    }

    pub fn create_view(&self, desc: &TextureViewDescriptor<'_>) -> TextureView {
        let v = self.inner.mx.create_view(&conv::view(desc)).unwrap_or_else(|e| panic!("wgpu-mx: {e}"));
        TextureView {
            inner: Arc::new(ViewInner { mx: v, texture: self.clone() }),
        }
    }

    pub fn destroy(&self) {}

    pub fn as_image_copy(&self) -> TexelCopyTextureInfo<'_> {
        TexelCopyTextureInfo {
            texture: self,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        }
    }

    pub fn size(&self) -> Extent3d {
        self.inner.size
    }
    pub fn width(&self) -> u32 {
        self.inner.size.width
    }
    pub fn height(&self) -> u32 {
        self.inner.size.height
    }
    pub fn depth_or_array_layers(&self) -> u32 {
        self.inner.size.depth_or_array_layers
    }
    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }
    pub fn sample_count(&self) -> u32 {
        self.inner.sample_count
    }
    pub fn dimension(&self) -> TextureDimension {
        self.inner.dimension
    }
    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }
    pub fn usage(&self) -> TextureUsages {
        self.inner.usage
    }
}

struct ViewInner {
    mx: mx::TextureView,
    texture: Texture,
}

#[derive(Clone)]
pub struct TextureView {
    inner: Arc<ViewInner>,
}
identity!(TextureView);

impl TextureView {
    pub fn mx(&self) -> &mx::TextureView {
        &self.inner.mx
    }

    pub fn texture(&self) -> &Texture {
        &self.inner.texture
    }
}

#[derive(Clone)]
pub struct Sampler {
    inner: Arc<mx::Sampler>,
}
identity!(Sampler);

#[derive(Clone)]
pub struct QuerySet {
    inner: Arc<mx::QuerySet>,
}
identity!(QuerySet);

#[derive(Clone)]
pub struct PipelineCache {
    inner: Arc<()>,
}
identity!(PipelineCache);

impl PipelineCache {
    pub fn get_data(&self) -> Option<Vec<u8>> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct PipelineCacheDescriptor<'a> {
    pub label: Label<'a>,
    pub data: Option<&'a [u8]>,
    pub fallback: bool,
}

#[derive(Clone)]
pub struct Blas {
    inner: Arc<mx::Blas>,
}
identity!(Blas);

/// A top-level structure: the instances are wgpu's (set with `tlas[i] = Some(..)`), copied
/// into mX's structure when it is built.
pub struct Tlas {
    inner: Arc<Mutex<mx::Tlas>>,
    instances: Vec<Option<TlasInstance>>,
}
identity!(Tlas);

impl Tlas {
    pub fn get_mut_slice(&mut self, range: Range<usize>) -> Option<&mut [Option<TlasInstance>]> {
        self.instances.get_mut(range)
    }

    pub fn get(&self, index: usize) -> Option<&TlasInstance> {
        self.instances.get(index)?.as_ref()
    }

    /// The mX top-level structure under it, for interop (locked: a build replaces its instances).
    pub fn mx(&self) -> parking_lot::MutexGuard<'_, mx::Tlas> {
        self.inner.lock()
    }
}

impl Tlas {
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::AccelerationStructure(self)
    }
}

impl std::ops::Index<usize> for Tlas {
    type Output = Option<TlasInstance>;
    fn index(&self, i: usize) -> &Self::Output {
        &self.instances[i]
    }
}

impl std::ops::IndexMut<usize> for Tlas {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.instances[i]
    }
}

#[derive(Clone, Debug)]
pub struct TlasInstance {
    pub transform: [f32; 12],
    pub custom_data: u32,
    pub mask: u8,
    blas: Blas,
}

impl TlasInstance {
    pub fn new(blas: &Blas, transform: [f32; 12], custom_data: u32, mask: u8) -> Self {
        TlasInstance {
            transform,
            custom_data,
            mask,
            blas: blas.clone(),
        }
    }
    pub fn blas(&self) -> &Blas {
        &self.blas
    }
}

#[derive(Clone, Debug)]
pub struct BlasTriangleGeometry<'a> {
    pub size: &'a BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: &'a Buffer,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub index_buffer: Option<&'a Buffer>,
    pub first_index: Option<u32>,
    pub transform_buffer: Option<&'a Buffer>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

#[derive(Clone, Debug)]
pub enum BlasGeometries<'a> {
    TriangleGeometries(Vec<BlasTriangleGeometry<'a>>),
}

#[derive(Clone, Debug)]
pub struct BlasBuildEntry<'a> {
    pub blas: &'a Blas,
    pub geometry: BlasGeometries<'a>,
}

// ---------------------------------------------------------------------------------------------
// Binding

#[derive(Clone)]
pub struct BindGroupLayout {
    inner: Arc<Vec<BindGroupLayoutEntry>>,
}
identity!(BindGroupLayout);

#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    pub label: Label<'a>,
    pub entries: &'a [BindGroupLayoutEntry],
}

#[derive(Clone)]
pub struct PipelineLayout {
    inner: Arc<()>,
}
identity!(PipelineLayout);

#[derive(Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a> {
    pub label: Label<'a>,
    pub bind_group_layouts: &'a [Option<&'a BindGroupLayout>],
    pub immediate_size: u32,
}

#[derive(Clone, Debug)]
pub struct BufferBinding<'a> {
    pub buffer: &'a Buffer,
    pub offset: BufferAddress,
    pub size: Option<BufferSize>,
}

#[derive(Clone, Debug)]
pub enum BindingResource<'a> {
    Buffer(BufferBinding<'a>),
    BufferArray(&'a [BufferBinding<'a>]),
    Sampler(&'a Sampler),
    SamplerArray(&'a [&'a Sampler]),
    TextureView(&'a TextureView),
    TextureViewArray(&'a [&'a TextureView]),
    AccelerationStructure(&'a Tlas),
}

#[derive(Clone, Debug)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

#[derive(Clone, Debug)]
pub struct BindGroupDescriptor<'a> {
    pub label: Label<'a>,
    pub layout: &'a BindGroupLayout,
    pub entries: &'a [BindGroupEntry<'a>],
}

enum Owned {
    Buffer(Buffer, u64, Option<u64>),
    View(TextureView),
    Sampler(Sampler),
    Tlas(Arc<Mutex<mx::Tlas>>),
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
        // (a bind group holds the part of a top-level structure that stays the same across
        // builds, so the lock is held only while it is made)
        let tlas: Vec<parking_lot::MutexGuard<'_, mx::Tlas>> = self
            .entries
            .iter()
            .filter_map(|(_, r)| match r {
                Owned::Tlas(t) => Some(t.lock()),
                _ => None,
            })
            .collect();
        let mut next_tlas = tlas.iter();
        let entries: Vec<mx::BindGroupEntry<'_>> = self
            .entries
            .iter()
            .map(|(binding, r)| mx::BindGroupEntry {
                binding: *binding,
                resource: match r {
                    Owned::Buffer(b, offset, size) => {
                        let extra = self.dynamic.iter().position(|d| d == binding).map_or(0, |i| offsets[i] as u64);
                        mx::BindingResource::Buffer(mx::BufferBinding {
                            buffer: &b.inner.mx,
                            offset: offset + extra,
                            size: *size,
                        })
                    }
                    Owned::View(v) => mx::BindingResource::TextureView(&v.inner.mx),
                    Owned::Sampler(s) => mx::BindingResource::Sampler(&s.inner),
                    Owned::Tlas(_) => mx::BindingResource::AccelerationStructure(next_tlas.next().expect("one guard per structure")),
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

#[derive(Clone)]
pub struct BindGroup {
    inner: Arc<BindGroupInner>,
}
identity!(BindGroup);

// ---------------------------------------------------------------------------------------------
// Shaders and pipelines

pub enum ShaderSource<'a> {
    Wgsl(std::borrow::Cow<'a, str>),
    #[doc(hidden)]
    Dummy(PhantomData<&'a ()>),
}

pub struct ShaderModuleDescriptor<'a> {
    pub label: Label<'a>,
    pub source: ShaderSource<'a>,
}

#[derive(Clone)]
pub struct ShaderModule {
    inner: Arc<Option<mx::ShaderModule>>,
}
identity!(ShaderModule);

#[derive(Clone, Debug)]
pub struct PipelineCompilationOptions<'a> {
    pub constants: &'a [(&'a str, f64)],
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for PipelineCompilationOptions<'_> {
    fn default() -> Self {
        PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VertexBufferLayout<'a> {
    pub array_stride: BufferAddress,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

#[derive(Clone, Debug)]
pub struct VertexState<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: PipelineCompilationOptions<'a>,
    pub buffers: &'a [VertexBufferLayout<'a>],
}

#[derive(Clone, Debug)]
pub struct FragmentState<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: PipelineCompilationOptions<'a>,
    pub targets: &'a [Option<ColorTargetState>],
}

#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a> {
    pub label: Label<'a>,
    pub layout: Option<&'a PipelineLayout>,
    pub vertex: VertexState<'a>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<FragmentState<'a>>,
    pub multiview_mask: Option<NonZeroU32>,
    pub cache: Option<&'a PipelineCache>,
}

#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a> {
    pub label: Label<'a>,
    pub layout: Option<&'a PipelineLayout>,
    pub module: &'a ShaderModule,
    pub entry_point: Option<&'a str>,
    pub compilation_options: PipelineCompilationOptions<'a>,
    pub cache: Option<&'a PipelineCache>,
}

#[derive(Clone)]
pub struct RenderPipeline {
    inner: Arc<Option<mx::RenderPipeline>>,
}
identity!(RenderPipeline);

impl RenderPipeline {
    /// A layout bind groups can be made with (mX takes any bind group a shader can read).
    pub fn get_bind_group_layout(&self, _index: u32) -> BindGroupLayout {
        BindGroupLayout { inner: Arc::new(Vec::new()) }
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    inner: Arc<Option<mx::ComputePipeline>>,
}
identity!(ComputePipeline);

impl ComputePipeline {
    pub fn get_bind_group_layout(&self, _index: u32) -> BindGroupLayout {
        BindGroupLayout { inner: Arc::new(Vec::new()) }
    }
}

// ---------------------------------------------------------------------------------------------
// Commands

type MapCallback = Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send>;

pub struct CommandBuffer {
    inner: Option<mx::CommandBuffer>,
    /// Buffers to map once this command buffer's submission has finished.
    maps: Vec<(Buffer, MapCallback)>,
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CommandBuffer")
    }
}

type EncoderCell = Arc<Mutex<Option<mx::CommandEncoder>>>;

pub struct CommandEncoder {
    inner: EncoderCell,
    device: Arc<DeviceInner>,
    maps: Vec<(Buffer, MapCallback)>,
}

impl std::fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CommandEncoder")
    }
}

fn copy_texture<'a>(t: &'a TexelCopyTextureInfo<'_>) -> mx::TexelCopyTextureInfo<'a> {
    mx::TexelCopyTextureInfo {
        texture: &t.texture.inner.mx,
        mip_level: t.mip_level,
        origin: conv::origin(t.origin),
        aspect: conv::aspect(t.aspect),
    }
}

impl CommandEncoder {
    fn with<R>(&self, f: impl FnOnce(&mut mx::CommandEncoder) -> R) -> R {
        f(self.inner.lock().as_mut().expect("wgpu-mx: the encoder has finished"))
    }

    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            inner: self.inner.lock().take().map(|e| e.finish()),
            maps: self.maps,
        }
    }

    /// Map `buffer` once the submission of this encoder's commands has finished.
    pub fn map_buffer_on_submit<S: RangeBounds<BufferAddress>>(&mut self, buffer: &Buffer, mode: MapMode, _bounds: S, callback: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static) {
        let needed = if mode == MapMode::Read { BufferUsages::MAP_READ } else { BufferUsages::MAP_WRITE };
        if !buffer.inner.usage.contains(needed) {
            callback(Err(BufferAsyncError));
            return;
        }
        self.maps.push((buffer.clone(), Box::new(callback)));
    }

    pub fn build_acceleration_structures<'a>(&mut self, blas: impl IntoIterator<Item = &'a BlasBuildEntry<'a>>, tlas: impl IntoIterator<Item = &'a Tlas>) {
        let blas: Vec<&BlasBuildEntry<'_>> = blas.into_iter().collect();
        // (mX's geometry refers to its own size descriptors: made first, borrowed below)
        let sizes: Vec<Vec<mx::BlasTriangleGeometrySizeDescriptor>> = blas
            .iter()
            .map(|e| {
                let BlasGeometries::TriangleGeometries(g) = &e.geometry;
                g.iter().map(|g| conv::blas_size(g.size)).collect()
            })
            .collect();
        let entries: Vec<mx::BlasBuildEntry<'_>> = blas
            .iter()
            .zip(&sizes)
            .map(|(e, sizes)| {
                let BlasGeometries::TriangleGeometries(g) = &e.geometry;
                mx::BlasBuildEntry {
                    blas: &*e.blas.inner,
                    geometry: g
                        .iter()
                        .zip(sizes)
                        .map(|(g, size)| {
                            assert!(g.transform_buffer.is_none(), "wgpu-mx: BLAS transform buffers are not supported");
                            mx::BlasTriangleGeometry {
                                size,
                                vertex_buffer: &g.vertex_buffer.inner.mx,
                                first_vertex: g.first_vertex,
                                vertex_stride: g.vertex_stride,
                                index_buffer: g.index_buffer.map(|b| &b.inner.mx),
                                first_index: g.first_index,
                            }
                        })
                        .collect(),
                }
            })
            .collect();
        // the instances set through wgpu's API go into mX's structures, then all is built
        let tlas: Vec<&Tlas> = tlas.into_iter().collect();
        let mut guards: Vec<parking_lot::MutexGuard<'_, mx::Tlas>> = tlas
            .iter()
            .map(|t| {
                let mut g = t.inner.lock();
                for (i, inst) in t.instances.iter().enumerate() {
                    g[i] = inst.as_ref().map(|inst| mx::TlasInstance::new((*inst.blas.inner).clone(), inst.transform, inst.custom_data, inst.mask));
                }
                g
            })
            .collect();
        let built: Vec<&mx::Tlas> = guards.iter_mut().map(|g| &**g).collect();
        self.with(|e| e.build_acceleration_structures(entries.iter(), built.iter().copied()));
    }

    pub fn begin_render_pass<'encoder>(&'encoder mut self, desc: &RenderPassDescriptor<'_>) -> RenderPass<'encoder> {
        let colors = desc
            .color_attachments
            .iter()
            .map(|a| a.as_ref().map(|a| (a.view.clone(), a.resolve_target.cloned(), conv::ops(a.ops, conv::color))))
            .collect();
        let depth = desc
            .depth_stencil_attachment
            .as_ref()
            .map(|d| (d.view.clone(), d.depth_ops.map(|o| conv::ops(o, |v| v)), d.stencil_ops.map(|o| conv::ops(o, |v| v))));
        RenderPass {
            data: Some(PassData {
                enc: self.inner.clone(),
                label: desc.label.map(str::to_owned),
                colors,
                depth,
                timestamps: desc
                    .timestamp_writes
                    .as_ref()
                    .map(|t| (t.query_set.clone(), t.beginning_of_pass_write_index, t.end_of_pass_write_index)),
                occlusion: desc.occlusion_query_set.cloned(),
                cmds: Vec::new(),
            }),
            _p: PhantomData,
        }
    }

    pub fn begin_compute_pass<'encoder>(&'encoder mut self, desc: &ComputePassDescriptor<'_>) -> ComputePass<'encoder> {
        ComputePass {
            enc: Some(self.inner.clone()),
            label: desc.label.map(str::to_owned),
            timestamps: desc
                .timestamp_writes
                .as_ref()
                .map(|t| (t.query_set.clone(), t.beginning_of_pass_write_index, t.end_of_pass_write_index)),
            cmds: Vec::new(),
            _p: PhantomData,
        }
    }

    pub fn copy_buffer_to_buffer(&mut self, source: &Buffer, source_offset: BufferAddress, destination: &Buffer, destination_offset: BufferAddress, copy_size: impl Into<Option<BufferAddress>>) {
        let size = copy_size.into().unwrap_or(source.size() - source_offset);
        self.with(|e| e.copy_buffer_to_buffer(&source.inner.mx, source_offset, &destination.inner.mx, destination_offset, size));
    }

    pub fn copy_buffer_to_texture(&mut self, source: TexelCopyBufferInfo<'_>, destination: TexelCopyTextureInfo<'_>, copy_size: Extent3d) {
        let src = mx::TexelCopyBufferInfo {
            buffer: &source.buffer.inner.mx,
            layout: conv::layout(source.layout),
        };
        self.with(|e| e.copy_buffer_to_texture(src, copy_texture(&destination), conv::extent(copy_size)));
    }

    pub fn copy_texture_to_buffer(&mut self, source: TexelCopyTextureInfo<'_>, destination: TexelCopyBufferInfo<'_>, copy_size: Extent3d) {
        let dst = mx::TexelCopyBufferInfo {
            buffer: &destination.buffer.inner.mx,
            layout: conv::layout(destination.layout),
        };
        self.with(|e| e.copy_texture_to_buffer(copy_texture(&source), dst, conv::extent(copy_size)));
    }

    pub fn copy_texture_to_texture(&mut self, source: TexelCopyTextureInfo<'_>, destination: TexelCopyTextureInfo<'_>, copy_size: Extent3d) {
        self.with(|e| e.copy_texture_to_texture(copy_texture(&source), copy_texture(&destination), conv::extent(copy_size)));
    }

    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: BufferAddress, size: Option<BufferAddress>) {
        self.with(|e| e.clear_buffer(&buffer.inner.mx, offset, size));
    }

    pub fn clear_texture(&mut self, texture: &Texture, subresource_range: &ImageSubresourceRange) {
        let r = mx::ImageSubresourceRange {
            aspect: conv::aspect(subresource_range.aspect),
            base_mip_level: subresource_range.base_mip_level,
            mip_level_count: subresource_range.mip_level_count,
            base_array_layer: subresource_range.base_array_layer,
            array_layer_count: subresource_range.array_layer_count,
        };
        self.with(|e| e.clear_texture(&texture.inner.mx, &r));
    }

    pub fn resolve_query_set(&mut self, query_set: &QuerySet, query_range: Range<u32>, destination: &Buffer, destination_offset: BufferAddress) {
        self.with(|e| e.resolve_query_set(&query_set.inner, query_range, &destination.inner.mx, destination_offset));
    }

    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        self.with(|e| e.write_timestamp(&query_set.inner, query_index));
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.with(|e| e.insert_debug_marker(label));
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.with(|e| e.push_debug_group(label));
    }
    pub fn pop_debug_group(&mut self) {
        self.with(|e| e.pop_debug_group());
    }

    pub fn transition_resources<'a>(&mut self, _buffer_transitions: impl Iterator<Item = BufferTransition<&'a Buffer>>, _texture_transitions: impl Iterator<Item = TextureTransition<&'a Texture>>) {}

    #[allow(dead_code)]
    fn device(&self) -> &DeviceInner {
        &self.device
    }
}

#[derive(Clone, Debug)]
pub struct RenderPassTimestampWrites<'a> {
    pub query_set: &'a QuerySet,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct RenderPassColorAttachment<'tex> {
    pub view: &'tex TextureView,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<&'tex TextureView>,
    pub ops: Operations<Color>,
}

#[derive(Clone, Debug)]
pub struct RenderPassDepthStencilAttachment<'tex> {
    pub view: &'tex TextureView,
    pub depth_ops: Option<Operations<f32>>,
    pub stencil_ops: Option<Operations<u32>>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderPassDescriptor<'a> {
    pub label: Label<'a>,
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a>>],
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
    pub timestamp_writes: Option<RenderPassTimestampWrites<'a>>,
    pub occlusion_query_set: Option<&'a QuerySet>,
    pub multiview_mask: Option<NonZeroU32>,
}

#[derive(Clone, Debug)]
pub struct ComputePassTimestampWrites<'a> {
    pub query_set: &'a QuerySet,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor<'a> {
    pub label: Label<'a>,
    pub timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
}

#[derive(Clone)]
enum Cmd {
    Pipeline(RenderPipeline),
    BindGroup(u32, BindGroup, Vec<u32>),
    VertexBuffer(u32, Buffer, u64),
    IndexBuffer(Buffer, u64, IndexFormat),
    Viewport([f32; 6]),
    Scissor([u32; 4]),
    StencilReference(u32),
    BlendConstant(Color),
    Immediates(u32, Vec<u8>),
    Draw(Range<u32>, Range<u32>),
    DrawIndexed(Range<u32>, i32, Range<u32>),
    DrawIndirect(Buffer, u64),
    DrawIndexedIndirect(Buffer, u64),
    MultiDrawIndirect(Buffer, u64, u32),
    MultiDrawIndexedIndirect(Buffer, u64, u32),
    BeginOcclusion(u32),
    EndOcclusion,
    PushDebug(String),
    PopDebug,
    Marker(String),
    Bundle(RenderBundle),
}

/// Replay `cmds` into an mX render pass. Draws after an invalid pipeline are skipped.
fn replay(rp: &mut mx::RenderPass<'_>, cmds: &[Cmd], valid: &mut bool) {
    for c in cmds {
        match c {
            Cmd::Pipeline(p) => match p.inner.as_ref() {
                Some(p) => {
                    rp.set_pipeline(p);
                    *valid = true;
                }
                None => *valid = false,
            },
            Cmd::BindGroup(i, g, o) => rp.set_bind_group(*i, &g.inner.with_offsets(o)),
            Cmd::VertexBuffer(s, b, o) => rp.set_vertex_buffer(*s, &b.inner.mx, *o),
            Cmd::IndexBuffer(b, o, f) => rp.set_index_buffer(&b.inner.mx, *o, conv::index_format(*f)),
            Cmd::Viewport(v) => rp.set_viewport(v[0], v[1], v[2], v[3], v[4], v[5]),
            Cmd::Scissor(s) => rp.set_scissor_rect(s[0], s[1], s[2], s[3]),
            Cmd::StencilReference(r) => rp.set_stencil_reference(*r),
            Cmd::BlendConstant(c) => rp.set_blend_constant(conv::color(*c)),
            Cmd::Immediates(o, d) => rp.set_immediates(*o, d),
            Cmd::Draw(v, i) if *valid => rp.draw(v.clone(), i.clone()),
            Cmd::DrawIndexed(ix, b, i) if *valid => rp.draw_indexed(ix.clone(), *b, i.clone()),
            Cmd::DrawIndirect(b, o) if *valid => rp.draw_indirect(&b.inner.mx, *o),
            Cmd::DrawIndexedIndirect(b, o) if *valid => rp.draw_indexed_indirect(&b.inner.mx, *o),
            Cmd::MultiDrawIndirect(b, o, n) if *valid => rp.multi_draw_indirect(&b.inner.mx, *o, *n),
            Cmd::MultiDrawIndexedIndirect(b, o, n) if *valid => rp.multi_draw_indexed_indirect(&b.inner.mx, *o, *n),
            Cmd::Draw(..) | Cmd::DrawIndexed(..) | Cmd::DrawIndirect(..) | Cmd::DrawIndexedIndirect(..) | Cmd::MultiDrawIndirect(..) | Cmd::MultiDrawIndexedIndirect(..) => {}
            Cmd::BeginOcclusion(i) => rp.begin_occlusion_query(*i),
            Cmd::EndOcclusion => rp.end_occlusion_query(),
            Cmd::PushDebug(l) => rp.push_debug_group(l),
            Cmd::PopDebug => rp.pop_debug_group(),
            Cmd::Marker(l) => rp.insert_debug_marker(l),
            Cmd::Bundle(b) => {
                let mut v = false;
                replay(rp, &b.inner, &mut v);
                *valid = false;
            }
        }
    }
}

type Attachment = (TextureView, Option<TextureView>, mx::Operations<mx::Color>);
type DepthAttachment = (TextureView, Option<mx::Operations<f32>>, Option<mx::Operations<u32>>);

struct PassData {
    enc: EncoderCell,
    label: Option<String>,
    colors: Vec<Option<Attachment>>,
    depth: Option<DepthAttachment>,
    timestamps: Option<(QuerySet, Option<u32>, Option<u32>)>,
    occlusion: Option<QuerySet>,
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
                    view: &v.inner.mx,
                    resolve_target: r.as_ref().map(|r| &r.inner.mx),
                    ops: *ops,
                })
            })
            .collect();
        let depth = self.depth.as_ref().map(|(v, d, s)| mx::RenderPassDepthStencilAttachment {
            view: &v.inner.mx,
            depth_ops: *d,
            stencil_ops: *s,
        });
        let timestamps = self.timestamps.as_ref().map(|(q, b, e)| mx::PassTimestampWrites {
            query_set: &*q.inner,
            beginning_of_pass_write_index: *b,
            end_of_pass_write_index: *e,
        });
        let mut rp = enc.begin_render_pass(&mx::RenderPassDescriptor {
            label: self.label.as_deref(),
            color_attachments: &colors,
            depth_stencil_attachment: depth,
            timestamp_writes: timestamps,
            occlusion_query_set: self.occlusion.as_ref().map(|q| &*q.inner),
        });
        let mut valid = false;
        replay(&mut rp, &self.cmds, &mut valid);
    }
}

/// A render pass: recorded, then replayed into mX when it ends (dropped).
pub struct RenderPass<'encoder> {
    data: Option<PassData>,
    _p: PhantomData<&'encoder mut ()>,
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        if let Some(d) = self.data.take() {
            d.run();
        }
    }
}

impl std::fmt::Debug for RenderPass<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RenderPass")
    }
}

impl RenderPass<'_> {
    fn push(&mut self, c: Cmd) {
        self.data.as_mut().unwrap().cmds.push(c);
    }

    pub fn forget_lifetime(mut self) -> RenderPass<'static> {
        RenderPass {
            data: self.data.take(),
            _p: PhantomData,
        }
    }

    pub fn set_bind_group<'a, BG: Into<Option<&'a BindGroup>>>(&mut self, index: u32, bind_group: BG, offsets: &[DynamicOffset]) {
        if let Some(g) = bind_group.into() {
            self.push(Cmd::BindGroup(index, g.clone(), offsets.to_vec()));
        }
    }
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.push(Cmd::Pipeline(pipeline.clone()));
    }
    pub fn set_blend_constant(&mut self, color: Color) {
        self.push(Cmd::BlendConstant(color));
    }
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'_>, index_format: IndexFormat) {
        self.push(Cmd::IndexBuffer(buffer_slice.buffer.clone(), buffer_slice.offset, index_format));
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'_>) {
        self.push(Cmd::VertexBuffer(slot, buffer_slice.buffer.clone(), buffer_slice.offset));
    }
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.push(Cmd::Scissor([x, y, width, height]));
    }
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.push(Cmd::Viewport([x, y, w, h, min_depth, max_depth]));
    }
    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.push(Cmd::StencilReference(reference));
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.push(Cmd::Marker(label.into()));
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.push(Cmd::PushDebug(label.into()));
    }
    pub fn pop_debug_group(&mut self) {
        self.push(Cmd::PopDebug);
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.push(Cmd::Draw(vertices, instances));
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }
    pub fn draw_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress) {
        self.push(Cmd::DrawIndirect(indirect_buffer.clone(), indirect_offset));
    }
    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress) {
        self.push(Cmd::DrawIndexedIndirect(indirect_buffer.clone(), indirect_offset));
    }
    pub fn multi_draw_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress, count: u32) {
        self.push(Cmd::MultiDrawIndirect(indirect_buffer.clone(), indirect_offset, count));
    }
    pub fn multi_draw_indexed_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress, count: u32) {
        self.push(Cmd::MultiDrawIndexedIndirect(indirect_buffer.clone(), indirect_offset, count));
    }
    pub fn execute_bundles<'a, I: IntoIterator<Item = &'a RenderBundle>>(&mut self, render_bundles: I) {
        for b in render_bundles {
            self.push(Cmd::Bundle(b.clone()));
        }
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.push(Cmd::Immediates(offset, data.to_vec()));
    }
    pub fn write_timestamp(&mut self, _query_set: &QuerySet, _query_index: u32) {}
    pub fn begin_occlusion_query(&mut self, query_index: u32) {
        self.push(Cmd::BeginOcclusion(query_index));
    }
    pub fn end_occlusion_query(&mut self) {
        self.push(Cmd::EndOcclusion);
    }
}

#[derive(Clone)]
enum CCmd {
    Pipeline(ComputePipeline),
    BindGroup(u32, BindGroup, Vec<u32>),
    Immediates(u32, Vec<u8>),
    Dispatch(u32, u32, u32),
    DispatchIndirect(Buffer, u64),
    PushDebug(String),
    PopDebug,
    Marker(String),
}

/// A compute pass: recorded, then replayed into mX when it ends (dropped).
pub struct ComputePass<'encoder> {
    enc: Option<EncoderCell>,
    label: Option<String>,
    timestamps: Option<(QuerySet, Option<u32>, Option<u32>)>,
    cmds: Vec<CCmd>,
    _p: PhantomData<&'encoder mut ()>,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        let Some(cell) = self.enc.take() else { return };
        let mut guard = cell.lock();
        let Some(enc) = guard.as_mut() else { return };
        let timestamps = self.timestamps.as_ref().map(|(q, b, e)| mx::PassTimestampWrites {
            query_set: &*q.inner,
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
                CCmd::Pipeline(p) => match p.inner.as_ref() {
                    Some(p) => {
                        cp.set_pipeline(p);
                        valid = true;
                    }
                    None => valid = false,
                },
                CCmd::BindGroup(i, g, o) => cp.set_bind_group(*i, &g.inner.with_offsets(o)),
                CCmd::Immediates(o, d) => cp.set_immediates(*o, d),
                CCmd::Dispatch(x, y, z) if valid => cp.dispatch_workgroups(*x, *y, *z),
                CCmd::DispatchIndirect(b, o) if valid => cp.dispatch_workgroups_indirect(&b.inner.mx, *o),
                CCmd::Dispatch(..) | CCmd::DispatchIndirect(..) => {}
                CCmd::PushDebug(l) => cp.push_debug_group(l),
                CCmd::PopDebug => cp.pop_debug_group(),
                CCmd::Marker(l) => cp.insert_debug_marker(l),
            }
        }
    }
}

impl std::fmt::Debug for ComputePass<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ComputePass")
    }
}

impl ComputePass<'_> {
    pub fn forget_lifetime(mut self) -> ComputePass<'static> {
        ComputePass {
            enc: self.enc.take(),
            label: self.label.take(),
            timestamps: self.timestamps.take(),
            cmds: std::mem::take(&mut self.cmds),
            _p: PhantomData,
        }
    }
    pub fn set_bind_group<'a, BG: Into<Option<&'a BindGroup>>>(&mut self, index: u32, bind_group: BG, offsets: &[DynamicOffset]) {
        if let Some(g) = bind_group.into() {
            self.cmds.push(CCmd::BindGroup(index, g.clone(), offsets.to_vec()));
        }
    }
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.cmds.push(CCmd::Pipeline(pipeline.clone()));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.cmds.push(CCmd::Immediates(offset, data.to_vec()));
    }
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.cmds.push(CCmd::Dispatch(x, y, z));
    }
    pub fn dispatch_workgroups_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress) {
        self.cmds.push(CCmd::DispatchIndirect(indirect_buffer.clone(), indirect_offset));
    }
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.cmds.push(CCmd::Marker(label.into()));
    }
    pub fn push_debug_group(&mut self, label: &str) {
        self.cmds.push(CCmd::PushDebug(label.into()));
    }
    pub fn pop_debug_group(&mut self) {
        self.cmds.push(CCmd::PopDebug);
    }
    pub fn write_timestamp(&mut self, _query_set: &QuerySet, _query_index: u32) {}
}

#[derive(Clone, Debug)]
pub struct RenderBundleEncoderDescriptor<'a> {
    pub label: Label<'a>,
    pub color_formats: &'a [Option<TextureFormat>],
    pub depth_stencil: Option<RenderBundleDepthStencil>,
    pub sample_count: u32,
    pub multiview: Option<NonZeroU32>,
}

pub struct RenderBundleEncoder<'a> {
    cmds: Vec<Cmd>,
    _p: PhantomData<&'a ()>,
}

impl std::fmt::Debug for RenderBundleEncoder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RenderBundleEncoder")
    }
}

#[derive(Clone)]
pub struct RenderBundle {
    inner: Arc<Vec<Cmd>>,
}
identity!(RenderBundle);

impl<'a> RenderBundleEncoder<'a> {
    pub fn finish(self, _desc: &RenderBundleDescriptor<'_>) -> RenderBundle {
        RenderBundle { inner: Arc::new(self.cmds) }
    }
    pub fn set_bind_group<'b, BG: Into<Option<&'b BindGroup>>>(&mut self, index: u32, bind_group: BG, offsets: &[DynamicOffset]) {
        if let Some(g) = bind_group.into() {
            self.cmds.push(Cmd::BindGroup(index, g.clone(), offsets.to_vec()));
        }
    }
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        self.cmds.push(Cmd::Pipeline(pipeline.clone()));
    }
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        self.cmds.push(Cmd::IndexBuffer(buffer_slice.buffer.clone(), buffer_slice.offset, index_format));
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        self.cmds.push(Cmd::VertexBuffer(slot, buffer_slice.buffer.clone(), buffer_slice.offset));
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.cmds.push(Cmd::Draw(vertices, instances));
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.cmds.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }
    pub fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        self.cmds.push(Cmd::DrawIndirect(indirect_buffer.clone(), indirect_offset));
    }
    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        self.cmds.push(Cmd::DrawIndexedIndirect(indirect_buffer.clone(), indirect_offset));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.cmds.push(Cmd::Immediates(offset, data.to_vec()));
    }
}

// ---------------------------------------------------------------------------------------------
// Surfaces

pub struct Surface<'w> {
    inner: Arc<mx::Surface<'w>>,
    config: Mutex<Option<SurfaceConfiguration>>,
}

impl std::fmt::Debug for Surface<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Surface")
    }
}

impl Surface<'_> {
    pub fn get_capabilities(&self, adapter: &Adapter) -> SurfaceCapabilities {
        let c = self.inner.get_capabilities(&adapter.inner);
        SurfaceCapabilities {
            formats: c.formats.into_iter().map(conv::format_back).collect(),
            present_modes: c.present_modes.into_iter().map(conv::present_mode_back).collect(),
            alpha_modes: c.alpha_modes.into_iter().map(conv::alpha_mode_back).collect(),
            usages: conv::texture_usages_back(c.usages),
        }
    }

    pub fn get_default_config(&self, adapter: &Adapter, width: u32, height: u32) -> Option<SurfaceConfiguration> {
        let c = self.get_capabilities(adapter);
        Some(SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: *c.formats.first()?,
            width,
            height,
            desired_maximum_frame_latency: 2,
            present_mode: *c.present_modes.first()?,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        })
    }

    pub fn configure(&self, device: &Device, config: &SurfaceConfiguration) {
        self.inner.configure(
            &device.inner.mx,
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

    pub fn get_configuration(&self) -> Option<SurfaceConfiguration> {
        self.config.lock().clone()
    }

    pub fn get_current_texture(&self) -> CurrentSurfaceTexture {
        let wrap = |s: mx::SurfaceTexture| {
            let c = self.config.lock().clone().expect("wgpu-mx: get_current_texture before configure");
            let texture = Texture::wrap(
                s.texture.clone(),
                Extent3d {
                    width: s.texture.width(),
                    height: s.texture.height(),
                    depth_or_array_layers: 1,
                },
                c.format,
                TextureDimension::D2,
                1,
                1,
                c.usage,
            );
            SurfaceTexture { texture, inner: Some(s) }
        };
        match self.inner.get_current_texture() {
            mx::CurrentSurfaceTexture::Success(s) => CurrentSurfaceTexture::Success(wrap(s)),
            mx::CurrentSurfaceTexture::Suboptimal(s) => CurrentSurfaceTexture::Suboptimal(wrap(s)),
            mx::CurrentSurfaceTexture::Timeout => CurrentSurfaceTexture::Timeout,
            mx::CurrentSurfaceTexture::Occluded => CurrentSurfaceTexture::Occluded,
            mx::CurrentSurfaceTexture::Outdated => CurrentSurfaceTexture::Outdated,
            mx::CurrentSurfaceTexture::Lost => CurrentSurfaceTexture::Lost,
        }
    }
}

pub struct SurfaceTexture {
    pub texture: Texture,
    inner: Option<mx::SurfaceTexture>,
}

impl std::fmt::Debug for SurfaceTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SurfaceTexture")
    }
}

impl SurfaceTexture {
    pub fn present(mut self) {
        if let Some(s) = self.inner.take() {
            s.present();
        }
    }
}

pub enum CurrentSurfaceTexture {
    Success(SurfaceTexture),
    Suboptimal(SurfaceTexture),
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

impl std::fmt::Debug for CurrentSurfaceTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CurrentSurfaceTexture::Success(_) => "Success",
            CurrentSurfaceTexture::Suboptimal(_) => "Suboptimal",
            CurrentSurfaceTexture::Timeout => "Timeout",
            CurrentSurfaceTexture::Occluded => "Occluded",
            CurrentSurfaceTexture::Outdated => "Outdated",
            CurrentSurfaceTexture::Lost => "Lost",
            CurrentSurfaceTexture::Validation => "Validation",
        })
    }
}

/// `[VertexAttribute]` from `location => format` pairs, offsets packed in order.
#[macro_export]
macro_rules! vertex_attr_array {
    ($($loc:expr => $fmt:ident),* $(,)?) => {
        $crate::vertex_attr_array!([] ; 0; $($loc => $fmt ,)*)
    };
    ([$($t:expr,)*] ; $off:expr ;) => { [$($t,)*] };
    ([$($t:expr,)*] ; $off:expr ; $loc:expr => $item:ident, $($ll:expr => $ii:ident ,)*) => {
        $crate::vertex_attr_array!(
            [$($t,)* $crate::VertexAttribute {
                format: $crate::VertexFormat :: $item,
                offset: $off,
                shader_location: $loc,
            },];
            $off + $crate::VertexFormat :: $item.size();
            $($ll => $ii ,)*
        )
    };
}
