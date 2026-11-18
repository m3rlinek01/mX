// The FidelityFX API structures, constants and identifiers below follow AMD's FidelityFX SDK
// headers (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, v2.4.0:
// `ffx_api.h`, `ffx_api_types.h`, `dx12/ffx_api_dx12.h`, `vk/ffx_api_vk.h`, `ffx_upscale.h`), which come under this
// licence:
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! AMD FSR 3.1 by way of AMD's FidelityFX API, DirectX 12 or Vulkan: upscaling (the same inputs
//! as DLSS, `crate::Inputs`) and frame generation ([`Generator`], with AMD's swapchain).

use crate::d3d12::{ListRing, wide};
use crate::vulkan::{VkGpu, VkRing};
use crate::{Constants as FrameConstants, Inputs, Method, Options};
use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

pub(crate) use crate::fidelityfx_frames::Generator;

/// The graphics API AMD's library drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gpu {
    Dx12,
    Vulkan,
}

impl Gpu {
    pub(crate) fn backend(self) -> mx::Backend {
        match self {
            Gpu::Dx12 => mx::Backend::Dx12,
            Gpu::Vulkan => mx::Backend::Vulkan,
        }
    }
}
use std::ffi::{CStr, c_void};
use std::sync::OnceLock;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW};
use windows::core::{Interface, PCSTR, PCWSTR};

type FfxContext = *mut c_void;

#[repr(C)]
pub(crate) struct Header {
    pub(crate) kind: u64,
    pub(crate) next: *const Header,
}

impl Header {
    pub(crate) const fn new(kind: u64) -> Header {
        Header { kind, next: std::ptr::null() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Dimensions {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Coords {
    pub(crate) x: f32,
    pub(crate) y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct ResourceDescription {
    pub(crate) kind: u32,
    pub(crate) format: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) mip_count: u32,
    pub(crate) flags: u32,
    pub(crate) usage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Resource {
    pub(crate) resource: *mut c_void,
    pub(crate) description: ResourceDescription,
    pub(crate) state: u32,
}

impl Resource {
    pub(crate) const NONE: Resource = Resource {
        resource: std::ptr::null_mut(),
        description: ResourceDescription {
            kind: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            mip_count: 0,
            flags: 0,
            usage: 0,
        },
        state: 0,
    };
}

#[repr(C)]
pub(crate) struct CreateBackendDx12 {
    pub(crate) header: Header,
    pub(crate) device: *mut c_void,
}

#[repr(C)]
pub(crate) struct CreateBackendVk {
    pub(crate) header: Header,
    pub(crate) device: vk::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
struct CreateVersion {
    header: Header,
    version: u32,
}

#[repr(C)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render_size: Dimensions,
    max_upscale_size: Dimensions,
    message: *const c_void,
}

#[repr(C)]
struct DispatchUpscale {
    header: Header,
    command_list: *mut c_void,
    color: Resource,
    depth: Resource,
    motion_vectors: Resource,
    exposure: Resource,
    reactive: Resource,
    transparency_and_composition: Resource,
    output: Resource,
    jitter_offset: Coords,
    motion_vector_scale: Coords,
    render_size: Dimensions,
    upscale_size: Dimensions,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_vertical: f32,
    view_space_to_meters: f32,
    flags: u32,
}

// (sizes the SDK's headers give on x64: a field out of place fails the build here)
const _: () = {
    assert!(std::mem::size_of::<Resource>() == 48);
    assert!(std::mem::size_of::<CreateVersion>() == 24);
    assert!(std::mem::size_of::<CreateBackendVk>() == 40);
    assert!(std::mem::size_of::<CreateUpscale>() == 48);
    assert!(std::mem::size_of::<DispatchUpscale>() == 432);
    assert!(std::mem::offset_of!(DispatchUpscale, output) == 312);
    assert!(std::mem::offset_of!(DispatchUpscale, flags) == 428);
};

pub(crate) const CREATE_BACKEND_DX12: u64 = 0x0000_0002;
pub(crate) const CREATE_BACKEND_VK: u64 = 0x0100_0003;
const CREATE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const CREATE_UPSCALE_VERSION: u64 = 0x0001_000b;
const UPSCALER_VERSION: u32 = (4 << 22) | (1 << 12) | 2;
const UPSCALE_DEPTH_INVERTED: u32 = 1 << 3;
const UPSCALE_NON_LINEAR_COLORSPACE: u32 = 1 << 8;
const UPSCALE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
const FORMAT_UNKNOWN: u32 = 0;
const FORMAT_R16G16_FLOAT: u32 = 18;
const FORMAT_R8G8B8A8_UNORM: u32 = 10;
const FORMAT_R8G8B8A8_SRGB: u32 = 12;
const FORMAT_B8G8R8A8_UNORM: u32 = 14;
const FORMAT_B8G8R8A8_SRGB: u32 = 15;
const FORMAT_R32_FLOAT: u32 = 28;
const FORMAT_R16G16B16A16_FLOAT: u32 = 4;
const FORMAT_R10G10B10A2_UNORM: u32 = 17;
const FORMAT_R16_FLOAT: u32 = 21;
// (FfxResourceStates: the states mX leaves textures in, see `rest_state`)
const STATE_UNORDERED_ACCESS: u32 = 1 << 0;
const STATE_PIXEL_COMPUTE_READ: u32 = (1 << 1) | (1 << 2);
const STATE_COMMON: u32 = 1 << 7;
const USAGE_UAV: u32 = 1 << 1;
const USAGE_DEPTHTARGET: u32 = 1 << 2;
const TYPE_TEXTURE2D: u32 = 2;

type FnCreateContext = unsafe extern "C" fn(*mut FfxContext, *const Header, *const c_void) -> u32;
type FnDestroyContext = unsafe extern "C" fn(*mut FfxContext, *const c_void) -> u32;
type FnDispatch = unsafe extern "C" fn(*mut FfxContext, *const Header) -> u32;
type FnConfigure = unsafe extern "C" fn(*mut FfxContext, *const Header) -> u32;
type FnQuery = unsafe extern "C" fn(*mut FfxContext, *mut Header) -> u32;

pub(crate) struct Api {
    _module: HMODULE,
    pub(crate) create_context: FnCreateContext,
    pub(crate) destroy_context: FnDestroyContext,
    pub(crate) dispatch: FnDispatch,
    pub(crate) configure: FnConfigure,
    pub(crate) query: FnQuery,
}

// (function pointers into a DLL loaded for the process's lifetime)
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API_DX12: OnceLock<Option<Api>> = OnceLock::new();
static API_VK: OnceLock<Option<Api>> = OnceLock::new();

fn loader_name(gpu: Gpu) -> &'static str {
    match gpu {
        Gpu::Dx12 => "amd_fidelityfx_loader_dx12.dll",
        Gpu::Vulkan => "amd_fidelityfx_loader_vk.dll",
    }
}

fn load_api(dir: &std::path::Path, gpu: Gpu) -> Result<Api> {
    let loader = dir.join(loader_name(gpu));
    let path = wide(&loader);
    let module =
        unsafe { LoadLibraryExW(PCWSTR(path.as_ptr()), None, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }.with_context(|| format!("opening {}", loader.display()))?;
    fn get<T: Copy>(module: HMODULE, name: &CStr) -> Result<T> {
        let f = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }.ok_or_else(|| anyhow!("{} is not exported by AMD's loader DLL", name.to_string_lossy()))?;
        Ok(unsafe { std::mem::transmute_copy::<_, T>(&f) })
    }
    Ok(Api {
        create_context: get(module, c"ffxCreateContext")?,
        destroy_context: get(module, c"ffxDestroyContext")?,
        dispatch: get(module, c"ffxDispatch")?,
        configure: get(module, c"ffxConfigure")?,
        query: get(module, c"ffxQuery")?,
        _module: module,
    })
}

/// AMD's loader for `gpu`, opened from `dir` the first time (once a process).
pub(crate) fn api_in(dir: &std::path::Path, gpu: Gpu) -> Result<&'static Api> {
    let cell = match gpu {
        Gpu::Dx12 => &API_DX12,
        Gpu::Vulkan => &API_VK,
    };
    cell.get_or_init(|| match load_api(dir, gpu) {
        Ok(a) => Some(a),
        Err(e) => {
            log::warn!("AMD FidelityFX is not available: {e:#}");
            None
        }
    })
    .as_ref()
    .ok_or_else(|| anyhow!("AMD's FidelityFX loader DLL ({}) could not be opened", loader_name(gpu)))
}

/// AMD's loader for `gpu`, opened already.
pub(crate) fn api(gpu: Gpu) -> Result<&'static Api> {
    match gpu {
        Gpu::Dx12 => &API_DX12,
        Gpu::Vulkan => &API_VK,
    }
    .get()
    .and_then(|a| a.as_ref())
    .ok_or_else(|| anyhow!("AMD's FidelityFX loader DLL ({}) is not open", loader_name(gpu)))
}

pub(crate) fn check(what: &str, r: u32) -> Result<()> {
    // (FfxApiReturnCodes; newer DLLs may return codes the headers do not list yet)
    const NAMES: [&str; 9] = [
        "ok",
        "error",
        "unknown description type",
        "runtime error",
        "no provider",
        "out of memory",
        "invalid parameter",
        "not supported by this provider",
        "this provider takes no newer description types",
    ];
    if r == 0 {
        Ok(())
    } else {
        Err(anyhow!("{what}: {} ({r})", NAMES.get(r as usize).copied().unwrap_or("unknown code")))
    }
}

fn ffx_format(f: DXGI_FORMAT) -> u32 {
    match f {
        DXGI_FORMAT_B8G8R8A8_UNORM => FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => FORMAT_B8G8R8A8_SRGB,
        DXGI_FORMAT_R8G8B8A8_UNORM => FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => FORMAT_R8G8B8A8_SRGB,
        DXGI_FORMAT_R16G16_FLOAT => FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R16_FLOAT => FORMAT_R16_FLOAT,
        DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32_TYPELESS => FORMAT_R32_FLOAT,
        _ => FORMAT_UNKNOWN,
    }
}

pub(crate) fn ffx_format_of(f: mx::TextureFormat) -> u32 {
    use mx::TextureFormat as Tf;
    match f {
        Tf::Bgra8Unorm => FORMAT_B8G8R8A8_UNORM,
        Tf::Bgra8UnormSrgb => FORMAT_B8G8R8A8_SRGB,
        Tf::Rgba8Unorm => FORMAT_R8G8B8A8_UNORM,
        Tf::Rgba8UnormSrgb => FORMAT_R8G8B8A8_SRGB,
        Tf::Rgba16Float => FORMAT_R16G16B16A16_FLOAT,
        Tf::Rgb10a2Unorm => FORMAT_R10G10B10A2_UNORM,
        Tf::Rg16Float => FORMAT_R16G16_FLOAT,
        Tf::R16Float => FORMAT_R16_FLOAT,
        Tf::Depth32Float => FORMAT_R32_FLOAT,
        _ => FORMAT_UNKNOWN,
    }
}

fn resource_vk(t: &mx::Texture, state: u32) -> Result<Resource> {
    let image = crate::vulkan::image(t)?;
    let depth = t.format().has_depth();
    let mut usage = if depth { USAGE_DEPTHTARGET } else { 0 };
    if t.usage().contains(mx::TextureUsages::STORAGE_BINDING) {
        usage |= USAGE_UAV;
    }
    let size = t.size();
    Ok(Resource {
        resource: vk::Handle::as_raw(image) as *mut c_void,
        description: ResourceDescription {
            kind: TYPE_TEXTURE2D,
            format: ffx_format_of(t.format()),
            width: size.width,
            height: size.height,
            depth: size.depth_or_array_layers,
            mip_count: t.mip_level_count(),
            flags: 0,
            usage,
        },
        state,
    })
}

/// The state mX leaves `t` in between its commands, as FidelityFX names it: on Direct3D 12 its
/// home state (`ALL_SHADER_RESOURCE` for a texture shaders read, else `COMMON`); on Vulkan
/// `GENERAL`, which FidelityFX's Vulkan backend takes `UNORDERED_ACCESS` for. FSR moves each
/// resource out of it and back.
pub(crate) fn rest_state(gpu: Gpu, t: &mx::Texture) -> u32 {
    match gpu {
        Gpu::Vulkan => STATE_UNORDERED_ACCESS,
        Gpu::Dx12 if t.usage().contains(mx::TextureUsages::TEXTURE_BINDING) => STATE_PIXEL_COMPUTE_READ,
        Gpu::Dx12 => STATE_COMMON,
    }
}

pub(crate) fn resource_on(gpu: Gpu, t: &mx::Texture) -> Result<Resource> {
    let state = rest_state(gpu, t);
    match gpu {
        Gpu::Dx12 => resource(t, state),
        Gpu::Vulkan => resource_vk(t, state),
    }
}

fn resource(t: &mx::Texture, state: u32) -> Result<Resource> {
    let raw = t.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 resource under an FSR 3 texture"))?.raw().clone();
    let desc = unsafe { raw.GetDesc() };
    let mut usage = if matches!(desc.Format, DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_D16_UNORM) {
        USAGE_DEPTHTARGET
    } else {
        0
    };
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS).0 != 0 {
        usage |= USAGE_UAV;
    }
    let resource = raw.as_raw();
    drop(raw);
    Ok(Resource {
        resource,
        description: ResourceDescription {
            kind: TYPE_TEXTURE2D,
            format: ffx_format(desc.Format),
            width: desc.Width as u32,
            height: desc.Height,
            depth: desc.DepthOrArraySize as u32,
            mip_count: desc.MipLevels as u32,
            flags: 0,
            usage,
        },
        state,
    })
}

/// AMD's context for one window size, and what it was created from (AMD may keep pointers into
/// it).
struct UpscaleContext {
    context: Box<FfxContext>,
    display: (u32, u32),
    _create: Box<(CreateUpscale, CreateVersion)>,
    _backend: BackendDesc,
}

pub(crate) enum BackendDesc {
    Dx12(Box<CreateBackendDx12>),
    Vulkan(Box<CreateBackendVk>),
}

#[allow(clippy::large_enum_variant)] // (one per upscaler, made once)
pub(crate) enum Backend {
    Dx12 { device: ID3D12Device, ring: ListRing },
    Vulkan { gpu: VkGpu, ring: VkRing },
}

impl Backend {
    /// Command lists (buffers) of ours on `device`'s queue.
    pub(crate) fn of(device: &mx::Device) -> Result<Backend> {
        Ok(if let Some(d) = device.as_dx12() {
            let raw = d.raw().clone();
            Backend::Dx12 {
                ring: ListRing::new(&raw).context("FidelityFX's command lists")?,
                device: raw,
            }
        } else if let Some(gpu) = VkGpu::of(device) {
            Backend::Vulkan {
                ring: VkRing::new(&gpu).context("FidelityFX's command buffers")?,
                gpu,
            }
        } else {
            bail!("not running on Direct3D 12 or Vulkan");
        })
    }

    /// The description of the backend, for `ffxCreateContext`.
    pub(crate) fn desc(&self) -> BackendDesc {
        match self {
            Backend::Dx12 { device, .. } => BackendDesc::Dx12(Box::new(CreateBackendDx12 {
                header: Header::new(CREATE_BACKEND_DX12),
                device: device.as_raw(),
            })),
            Backend::Vulkan { gpu, .. } => BackendDesc::Vulkan(Box::new(CreateBackendVk {
                header: Header::new(CREATE_BACKEND_VK),
                device: gpu.device.handle(),
                physical_device: gpu.physical,
                get_device_proc_addr: gpu.get_device_proc_addr,
            })),
        }
    }

    /// Record with `f` into a command list (buffer) of ours, and submit it on `queue`.
    pub(crate) fn record(&mut self, queue: &mx::Queue, f: impl FnOnce(*mut c_void) -> Result<()>) -> Result<()> {
        match self {
            Backend::Dx12 { ring, .. } => {
                let raw_queue = queue.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 queue under mX's"))?.raw().clone();
                let list = ring.begin()?;
                let recorded = f(list.as_raw());
                ring.submit(&raw_queue)?;
                recorded
            }
            Backend::Vulkan { ring, .. } => {
                let buffer = ring.begin()?;
                let recorded = f(vk::Handle::as_raw(buffer) as *mut c_void);
                crate::vulkan::with_queue(queue, |q| ring.submit(q))??;
                recorded
            }
        }
    }

    pub(crate) fn gpu(&self) -> Gpu {
        match self {
            Backend::Dx12 { .. } => Gpu::Dx12,
            Backend::Vulkan { .. } => Gpu::Vulkan,
        }
    }

    pub(crate) fn wait(&mut self) {
        match self {
            Backend::Dx12 { ring, .. } => ring.wait(),
            Backend::Vulkan { ring, .. } => ring.wait(),
        }
    }
}

pub struct Upscaler {
    backend: Backend,
    context: Option<UpscaleContext>,
    hdr: bool,
}

// (used from the render thread only)
unsafe impl Send for Upscaler {}

impl Upscaler {
    pub(crate) fn wait(&mut self) {
        self.backend.wait();
    }

    pub fn load(device: &mx::Device, options: &Options) -> Result<Upscaler> {
        let dir = &options.library_dir;
        let backend = Backend::of(device).context("FSR 3 upscaling")?;
        let gpu = backend.gpu();
        let missing = Method::Fsr3.missing(dir, gpu.backend());
        if !missing.is_empty() {
            bail!("{} not found in {} (they come with AMD's FidelityFX SDK, Kits/FidelityFX/signedbin)", missing.join(", "), dir.display());
        }
        api_in(dir, gpu)?;
        log::info!("FSR 3 upscaling: AMD FidelityFX for {gpu:?} loaded from {}", dir.display());
        Ok(Upscaler { backend, context: None, hdr: false })
    }

    fn context_for(&mut self, display: (u32, u32), hdr: bool) -> Result<*mut FfxContext> {
        let api = api(self.backend.gpu())?;
        if self.context.as_ref().is_some_and(|c| c.display != display) || self.hdr != hdr {
            self.backend.wait();
            if let Some(mut c) = self.context.take() {
                unsafe { (api.destroy_context)(&mut *c.context, std::ptr::null()) };
            }
        }
        if self.context.is_none() {
            let size = Dimensions { width: display.0, height: display.1 };
            let mut create = Box::new((
                CreateUpscale {
                    header: Header::new(CREATE_UPSCALE),
                    flags: UPSCALE_DEPTH_INVERTED | if hdr { UPSCALE_HIGH_DYNAMIC_RANGE } else { UPSCALE_NON_LINEAR_COLORSPACE },
                    max_render_size: size,
                    max_upscale_size: size,
                    message: std::ptr::null(),
                },
                CreateVersion {
                    header: Header::new(CREATE_UPSCALE_VERSION),
                    version: UPSCALER_VERSION,
                },
            ));
            let backend = self.backend.desc();
            create.1.header.next = match &backend {
                BackendDesc::Dx12(b) => &b.header,
                BackendDesc::Vulkan(b) => &b.header,
            };
            create.0.header.next = &create.1.header;
            let mut context: Box<FfxContext> = Box::new(std::ptr::null_mut());
            check("creating the FSR 3 upscaling context", unsafe {
                (api.create_context)(&mut *context, &create.0.header, std::ptr::null())
            })?;
            log::info!("FSR 3 upscaling: {}x{}", display.0, display.1);
            self.hdr = hdr;
            self.context = Some(UpscaleContext {
                context,
                display,
                _create: create,
                _backend: backend,
            });
        }
        Ok(&mut *self.context.as_mut().expect("made above").context)
    }

    pub fn upscale(&mut self, queue: &mx::Queue, inputs: &Inputs, c: &FrameConstants) -> Result<()> {
        let gpu = self.backend.gpu();
        let api = api(gpu)?;
        let context = self.context_for(inputs.output_size, c.hdr)?;
        let mut color = resource_on(gpu, inputs.color)?;
        color.description.format = match inputs.color.format().remove_srgb_suffix() {
            mx::TextureFormat::Bgra8Unorm => FORMAT_B8G8R8A8_UNORM,
            mx::TextureFormat::Rgba8Unorm => FORMAT_R8G8B8A8_UNORM,
            mx::TextureFormat::Rgba16Float => FORMAT_R16G16B16A16_FLOAT,
            mx::TextureFormat::Rgb10a2Unorm => FORMAT_R10G10B10A2_UNORM,
            f => bail!("FSR 3 upscaling does not take a picture in {f:?}"),
        };
        let need = |t: Option<&mx::Texture>, what: &str| -> Result<Resource> { resource_on(gpu, t.ok_or_else(|| anyhow!("FSR 3 needs the {what}"))?) };
        let depth = need(inputs.depth, "depth")?;
        let motion_vectors = need(inputs.motion, "motion vectors")?;
        let reactive = match inputs.mask {
            Some(m) => resource_on(gpu, m)?,
            None => Resource::NONE,
        };
        let output = resource_on(gpu, inputs.output)?;
        let dispatch = |command_list: *mut c_void| {
            let desc = DispatchUpscale {
                header: Header::new(DISPATCH_UPSCALE),
                command_list,
                color,
                depth,
                motion_vectors,
                exposure: Resource::NONE,
                reactive,
                transparency_and_composition: Resource::NONE,
                output,
                jitter_offset: Coords { x: c.jitter[0], y: c.jitter[1] },
                motion_vector_scale: Coords { x: 1.0, y: 1.0 },
                render_size: Dimensions {
                    width: inputs.render_size.0,
                    height: inputs.render_size.1,
                },
                upscale_size: Dimensions {
                    width: inputs.output_size.0,
                    height: inputs.output_size.1,
                },
                enable_sharpening: inputs.render_size != inputs.output_size,
                sharpness: 0.3,
                frame_time_delta: c.frame_ms,
                pre_exposure: c.pre_exposure,
                reset: c.reset,
                camera_near: c.near,
                camera_far: c.far,
                camera_fov_vertical: c.fov,
                view_space_to_meters: 1.0,
                // (not NON_LINEAR_COLOR_SRGB: with it FSR takes the input to linear and hands it
                // back linear, which is then shown as if it were sRGB, the whole picture too dark)
                flags: 0,
            };
            check("FSR 3 upscaling", unsafe { (api.dispatch)(context, &desc.header) })
        };
        self.backend.record(queue, dispatch)
    }
}

impl Drop for Upscaler {
    fn drop(&mut self) {
        self.backend.wait();
        let api = match self.backend.gpu() {
            Gpu::Dx12 => &API_DX12,
            Gpu::Vulkan => &API_VK,
        };
        if let (Some(api), Some(mut c)) = (api.get().and_then(|a| a.as_ref()), self.context.take()) {
            unsafe { (api.destroy_context)(&mut *c.context, std::ptr::null()) };
        }
    }
}
