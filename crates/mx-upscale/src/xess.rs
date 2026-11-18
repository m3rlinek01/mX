//! Intel XeSS Super Resolution (`libxess.dll`, Direct3D 12 or Vulkan): the same inputs as the
//! other upscalers (`crate::Inputs`).
//!
//! XeSS records into a command list of ours on mX's queue and moves no texture itself: around
//! its work this file moves the inputs to the states it reads them in (Direct3D 12:
//! `NON_PIXEL_SHADER_RESOURCE`, which mX's home state of a sampled texture includes; Vulkan:
//! `SHADER_READ_ONLY_OPTIMAL`) and the output to `UNORDERED_ACCESS` / `GENERAL`, and back to
//! where mX keeps them.
//!
//! On Vulkan the device must be made with XeSS's extensions and features (mutable descriptor
//! types among them); `xessVKGetRequiredDeviceExtensions` names them.

use crate::d3d12::{ListRing, wide};
use crate::vulkan::{VkGpu, VkRing};
use crate::xess_ffi::*;
use crate::{Constants as FrameConstants, Inputs, Method, Options, Quality};
use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use std::ffi::{CStr, c_void};
use std::sync::OnceLock;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW};
use windows::core::{Interface, PCSTR, PCWSTR};

struct Api {
    _module: HMODULE,
    get_version: PFN_xessGetVersion,
    get_optimal_input_resolution: PFN_xessGetOptimalInputResolution,
    destroy: PFN_xessDestroyContext,
    set_velocity_scale: PFN_xessSetVelocityScale,
    set_logging_callback: PFN_xessSetLoggingCallback,
    is_optimal_driver: PFN_xessIsOptimalDriver,
    d3d12_create: PFN_xessD3D12CreateContext,
    d3d12_init: PFN_xessD3D12Init,
    d3d12_execute: PFN_xessD3D12Execute,
    vk_create: Option<PFN_xessVKCreateContext>,
    vk_init: Option<PFN_xessVKInit>,
    vk_execute: Option<PFN_xessVKExecute>,
}

// (function pointers into a DLL loaded for the process's lifetime)
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

fn get<T: Copy>(module: HMODULE, name: &CStr) -> Result<T> {
    let f = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }.ok_or_else(|| anyhow!("{} is not exported by libxess.dll", name.to_string_lossy()))?;
    Ok(unsafe { std::mem::transmute_copy::<_, T>(&f) })
}

fn api(dir: &std::path::Path) -> Result<&'static Api> {
    API.get_or_init(|| {
        (|| -> Result<Api> {
            let path = wide(&dir.join(XESS_DLL_NAME));
            let module = unsafe { LoadLibraryExW(PCWSTR(path.as_ptr()), None, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }.context("opening libxess.dll")?;
            Ok(Api {
                get_version: get(module, c"xessGetVersion")?,
                get_optimal_input_resolution: get(module, c"xessGetOptimalInputResolution")?,
                destroy: get(module, c"xessDestroyContext")?,
                set_velocity_scale: get(module, c"xessSetVelocityScale")?,
                set_logging_callback: get(module, c"xessSetLoggingCallback")?,
                is_optimal_driver: get(module, c"xessIsOptimalDriver")?,
                d3d12_create: get(module, c"xessD3D12CreateContext")?,
                d3d12_init: get(module, c"xessD3D12Init")?,
                d3d12_execute: get(module, c"xessD3D12Execute")?,
                vk_create: get(module, c"xessVKCreateContext").ok(),
                vk_init: get(module, c"xessVKInit").ok(),
                vk_execute: get(module, c"xessVKExecute").ok(),
                _module: module,
            })
        })()
        .map_err(|e| format!("{e:#}"))
    })
    .as_ref()
    .map_err(|e| anyhow!("{e}"))
}

fn check(what: &str, r: xess_result_t) -> Result<()> {
    if r.0 >= 0 {
        if r.0 > 0 {
            log::warn!("XeSS: {what}: warning {}", r.0);
        }
        Ok(())
    } else {
        let name = match r.0 {
            -1 => "unsupported device",
            -2 => "unsupported driver",
            -3 => "uninitialised",
            -4 => "invalid argument",
            -5 => "out of device memory",
            -6 => "device error",
            -7 => "not implemented",
            -8 => "invalid context",
            -9 => "operation in progress",
            -10 => "unsupported",
            -11 => "cannot load a library",
            -12 => "wrong call order",
            _ => "unknown error",
        };
        bail!("XeSS: {what}: {name} ({})", r.0)
    }
}

unsafe extern "C" fn xess_log(message: *const std::ffi::c_char, level: xess_logging_level_t) {
    if message.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    match level.0 {
        3 => log::error!("XeSS: {}", text.trim_end()),
        2 => log::warn!("XeSS: {}", text.trim_end()),
        _ => log::debug!("XeSS: {}", text.trim_end()),
    }
}

fn quality_setting(q: Quality) -> xess_quality_settings_t {
    match q {
        Quality::Native => XESS_QUALITY_SETTING_AA,
        Quality::UltraQuality => XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS,
        Quality::Quality => XESS_QUALITY_SETTING_QUALITY,
        Quality::Balanced => XESS_QUALITY_SETTING_BALANCED,
        Quality::Performance => XESS_QUALITY_SETTING_PERFORMANCE,
        Quality::UltraPerformance => XESS_QUALITY_SETTING_ULTRA_PERFORMANCE,
    }
}

#[allow(clippy::large_enum_variant)] // (one per upscaler, made once)
enum Backend {
    Dx12 { ring: ListRing },
    Vulkan { gpu: VkGpu, ring: VkRing },
}

pub struct Upscaler {
    api: &'static Api,
    context: xess_context_handle_t,
    backend: Backend,
    /// What the context was initialised for: output size, quality, flags.
    init: Option<((u32, u32), Quality, u32)>,
    #[allow(clippy::type_complexity)] // (quality and output size, to render size)
    optimal: Vec<((Quality, (u32, u32)), (u32, u32))>,
}

// (XeSS's context, used from one thread at a time)
unsafe impl Send for Upscaler {}

impl Upscaler {
    pub fn load(device: &mx::Device, options: &Options) -> Result<Upscaler> {
        let dir = &options.library_dir;
        let backend_kind = if device.as_dx12().is_some() { mx::Backend::Dx12 } else { mx::Backend::Vulkan };
        let missing = Method::Xess.missing(dir, backend_kind);
        if !missing.is_empty() {
            bail!("{} not found in {} (it comes with Intel's XeSS SDK, bin)", missing.join(", "), dir.display());
        }
        let api = api(dir)?;
        let mut context: xess_context_handle_t = std::ptr::null_mut();
        let backend = if let Some(d) = device.as_dx12() {
            let raw = d.raw().clone();
            check("creating the context", unsafe { (api.d3d12_create)(raw.as_raw(), &mut context) })?;
            Backend::Dx12 {
                ring: ListRing::new(&raw).context("XeSS's command lists")?,
            }
        } else if let Some(gpu) = VkGpu::of(device) {
            let create = api.vk_create.ok_or_else(|| anyhow!("this libxess.dll has no Vulkan"))?;
            check("creating the Vulkan context", unsafe {
                create(
                    vk::Handle::as_raw(gpu.instance) as *mut c_void,
                    vk::Handle::as_raw(gpu.physical) as *mut c_void,
                    vk::Handle::as_raw(gpu.device.handle()) as *mut c_void,
                    &mut context,
                )
            })?;
            Backend::Vulkan {
                ring: VkRing::new(&gpu).context("XeSS's command buffers")?,
                gpu,
            }
        } else {
            bail!("not running on Direct3D 12 or Vulkan");
        };
        let _ = unsafe { (api.is_optimal_driver)(context) };
        let _ = unsafe { (api.set_logging_callback)(context, if options.verbose { XESS_LOGGING_LEVEL_DEBUG } else { XESS_LOGGING_LEVEL_WARNING }, Some(xess_log)) };
        let mut version = xess_version_t::default();
        let _ = unsafe { (api.get_version)(&mut version) };
        log::info!("XeSS {}.{}.{} loaded from {}", version.major, version.minor, version.patch, dir.display());
        Ok(Upscaler {
            api,
            context,
            backend,
            init: None,
            optimal: Vec::new(),
        })
    }

    pub(crate) fn wait(&mut self) {
        match &mut self.backend {
            Backend::Dx12 { ring } => ring.wait(),
            Backend::Vulkan { ring, .. } => ring.wait(),
        }
    }

    pub fn render_size(&mut self, quality: Quality, out: (u32, u32)) -> (u32, u32) {
        if let Some((_, s)) = self.optimal.iter().find(|(k, _)| *k == (quality, out)) {
            return *s;
        }
        let output = xess_2d_t { x: out.0, y: out.1 };
        let (mut optimal, mut min, mut max) = (xess_2d_t::default(), xess_2d_t::default(), xess_2d_t::default());
        let r = unsafe { (self.api.get_optimal_input_resolution)(self.context, &output, quality_setting(quality), &mut optimal, &mut min, &mut max) };
        let size = if r.0 >= 0 && optimal.x > 0 && optimal.y > 0 {
            (optimal.x.min(out.0), optimal.y.min(out.1))
        } else {
            crate::render_size(quality, out)
        };
        self.optimal.retain(|((_, w), _)| *w == out);
        self.optimal.push(((quality, out), size));
        size
    }

    fn init_for(&mut self, output: (u32, u32), quality: Quality, flags: u32) -> Result<()> {
        if self.init == Some((output, quality, flags)) {
            return Ok(());
        }
        // (a new context setting waits for the work recorded with the old one)
        self.wait();
        let res = xess_2d_t { x: output.0, y: output.1 };
        match &self.backend {
            Backend::Dx12 { .. } => {
                let p = xess_d3d12_init_params_t {
                    outputResolution: res,
                    qualitySetting: quality_setting(quality),
                    initFlags: flags,
                    creationNodeMask: 0,
                    visibleNodeMask: 0,
                    pTempBufferHeap: std::ptr::null_mut(),
                    bufferHeapOffset: 0,
                    pTempTextureHeap: std::ptr::null_mut(),
                    textureHeapOffset: 0,
                    pPipelineLibrary: std::ptr::null_mut(),
                };
                check("initialising", unsafe { (self.api.d3d12_init)(self.context, &p) })?;
            }
            Backend::Vulkan { .. } => {
                let init = self.api.vk_init.ok_or_else(|| anyhow!("this libxess.dll has no Vulkan"))?;
                let p = xess_vk_init_params_t {
                    outputResolution: res,
                    qualitySetting: quality_setting(quality),
                    initFlags: flags,
                    creationNodeMask: 0,
                    visibleNodeMask: 0,
                    tempBufferHeap: 0,
                    bufferHeapOffset: 0,
                    tempTextureHeap: 0,
                    textureHeapOffset: 0,
                    pipelineCache: 0,
                };
                check("initialising", unsafe { init(self.context, &p) })?;
            }
        }
        // (the motion vectors are where each pixel was, in pixels: XeSS's own convention)
        let _ = unsafe { (self.api.set_velocity_scale)(self.context, 1.0, 1.0) };
        log::info!("XeSS: {}x{} output, {quality:?}", output.0, output.1);
        self.init = Some((output, quality, flags));
        Ok(())
    }

    pub fn upscale(&mut self, queue: &mx::Queue, quality: Quality, inputs: &Inputs, c: &FrameConstants) -> Result<()> {
        let depth = inputs.depth.ok_or_else(|| anyhow!("XeSS needs the depth"))?;
        let motion = inputs.motion.ok_or_else(|| anyhow!("XeSS needs the motion vectors"))?;
        let mut flags = XESS_INIT_FLAG_INVERTED_DEPTH;
        if !c.hdr {
            flags |= XESS_INIT_FLAG_LDR_INPUT_COLOR;
        }
        if inputs.mask.is_some() {
            flags |= XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
        }
        self.init_for(inputs.output_size, quality, flags)?;
        let (rw, rh) = inputs.render_size;
        let api = self.api;
        let context = self.context;
        match &mut self.backend {
            Backend::Dx12 { ring } => {
                let raw = |t: &mx::Texture| -> Result<(ID3D12Resource, D3D12_RESOURCE_STATES)> {
                    let d = t.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 resource under an XeSS texture"))?;
                    Ok((d.raw().clone(), d.rest_state()))
                };
                let (color, color_state) = raw(inputs.color)?;
                let (depth, depth_state) = raw(depth)?;
                let (motion, motion_state) = raw(motion)?;
                let (output, output_state) = raw(inputs.output)?;
                let mask = inputs.mask.map(raw).transpose()?;
                let raw_queue = queue.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 queue under mX's"))?.raw().clone();
                let list = ring.begin()?;
                // (to the states XeSS reads and writes in, and back)
                let mut moves: Vec<(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> = Vec::new();
                let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                for (r, s) in [(&color, color_state), (&depth, depth_state), (&motion, motion_state)] {
                    if (s & read) != read {
                        moves.push((r, s, read));
                    }
                }
                if let Some((m, s)) = &mask
                    && (*s & read) != read
                {
                    moves.push((m, *s, read));
                }
                if output_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
                    moves.push((&output, output_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
                }
                let barrier = |r: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: before,
                            StateAfter: after,
                        }),
                    },
                };
                let to: Vec<D3D12_RESOURCE_BARRIER> = moves.iter().map(|(r, b, a)| barrier(r, *b, *a)).collect();
                let back: Vec<D3D12_RESOURCE_BARRIER> = moves.iter().map(|(r, b, a)| barrier(r, *a, *b)).collect();
                let exec = xess_d3d12_execute_params_t {
                    pColorTexture: color.as_raw(),
                    pVelocityTexture: motion.as_raw(),
                    pDepthTexture: depth.as_raw(),
                    pExposureScaleTexture: std::ptr::null_mut(),
                    pResponsivePixelMaskTexture: mask.as_ref().map_or(std::ptr::null_mut(), |(m, _)| m.as_raw()),
                    pOutputTexture: output.as_raw(),
                    jitterOffsetX: c.jitter[0],
                    jitterOffsetY: c.jitter[1],
                    exposureScale: c.pre_exposure,
                    resetHistory: c.reset as u32,
                    inputWidth: rw,
                    inputHeight: rh,
                    inputColorBase: xess_2d_t::default(),
                    inputMotionVectorBase: xess_2d_t::default(),
                    inputDepthBase: xess_2d_t::default(),
                    inputResponsiveMaskBase: xess_2d_t::default(),
                    reserved0: xess_2d_t::default(),
                    outputColorBase: xess_2d_t::default(),
                    pDescriptorHeap: std::ptr::null_mut(),
                    descriptorHeapOffset: 0,
                };
                let recorded = unsafe {
                    if !to.is_empty() {
                        list.ResourceBarrier(&to);
                    }
                    let r = (api.d3d12_execute)(context, list.as_raw(), &exec);
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                    r
                };
                for b in to.into_iter().chain(back) {
                    // (the barriers' references to the resources)
                    let t = std::mem::ManuallyDrop::into_inner(unsafe { b.Anonymous.Transition });
                    drop(std::mem::ManuallyDrop::into_inner(t.pResource));
                }
                drop(list);
                ring.submit(&raw_queue)?;
                check("upscaling", recorded)
            }
            Backend::Vulkan { gpu, ring } => {
                let execute = api.vk_execute.ok_or_else(|| anyhow!("this libxess.dll has no Vulkan"))?;
                let info = |t: &mx::Texture| -> Result<(xess_vk_image_view_info, vk::Image, vk::ImageAspectFlags)> {
                    let image = crate::vulkan::image(t)?;
                    let depth = t.format().has_depth();
                    let aspect = if depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
                    let view = t.create_view(&mx::TextureViewDescriptor {
                        aspect: if depth { mx::TextureAspect::DepthOnly } else { mx::TextureAspect::All },
                        ..Default::default()
                    })?;
                    let size = t.size();
                    let i = xess_vk_image_view_info {
                        imageView: vk::Handle::as_raw(crate::vulkan::view(&view)?),
                        image: vk::Handle::as_raw(image),
                        subresourceRange: VkImageSubresourceRange {
                            aspectMask: aspect.as_raw(),
                            baseMipLevel: 0,
                            levelCount: 1,
                            baseArrayLayer: 0,
                            layerCount: 1,
                        },
                        format: crate::vulkan::format(t.format()).as_raw(),
                        width: size.width,
                        height: size.height,
                    };
                    // (the view lives on in mX until its last use is done)
                    std::mem::forget(view);
                    Ok((i, image, aspect))
                };
                let (color, ci, ca) = info(inputs.color)?;
                let (dep, di, da) = info(depth)?;
                let (mot, mi, ma) = info(motion)?;
                let (out, _, _) = info(inputs.output)?;
                let mask = inputs.mask.map(info).transpose()?;
                let mut reads = vec![(ci, ca), (di, da), (mi, ma)];
                if let Some((_, i, a)) = &mask {
                    reads.push((*i, *a));
                }
                let exec = xess_vk_execute_params_t {
                    colorTexture: color,
                    velocityTexture: mot,
                    depthTexture: dep,
                    exposureScaleTexture: xess_vk_image_view_info::default(),
                    responsivePixelMaskTexture: mask.as_ref().map(|(m, _, _)| *m).unwrap_or_default(),
                    outputTexture: out,
                    jitterOffsetX: c.jitter[0],
                    jitterOffsetY: c.jitter[1],
                    exposureScale: c.pre_exposure,
                    resetHistory: c.reset as u32,
                    inputWidth: rw,
                    inputHeight: rh,
                    ..Default::default()
                };
                let buffer = ring.begin()?;
                let layout = |old, new| -> Vec<vk::ImageMemoryBarrier2<'static>> {
                    reads
                        .iter()
                        .map(|(image, aspect)| {
                            vk::ImageMemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                                .old_layout(old)
                                .new_layout(new)
                                .image(*image)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(*aspect)
                                        .level_count(vk::REMAINING_MIP_LEVELS)
                                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                                )
                        })
                        .collect()
                };
                let to = layout(vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                let back = layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::GENERAL);
                let recorded = unsafe {
                    gpu.device.cmd_pipeline_barrier2(buffer, &vk::DependencyInfo::default().image_memory_barriers(&to));
                    let r = execute(context, vk::Handle::as_raw(buffer) as *mut c_void, &exec);
                    gpu.device.cmd_pipeline_barrier2(buffer, &vk::DependencyInfo::default().image_memory_barriers(&back));
                    r
                };
                crate::vulkan::with_queue(queue, |q| ring.submit(q))??;
                check("upscaling", recorded)
            }
        }
    }
}

impl Drop for Upscaler {
    fn drop(&mut self) {
        self.wait();
        unsafe {
            let _ = (self.api.destroy)(self.context);
        }
    }
}
