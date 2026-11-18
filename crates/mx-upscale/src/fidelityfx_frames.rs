//! AMD FSR 3.1 frame generation (FidelityFX API, Direct3D 12 or Vulkan).
//!
//! AMD's swapchain takes the window's place (made by the surface's swapchain maker): the program
//! draws into its pictures and presents as before, and AMD's present makes and shows the picture
//! between the last frame and this one. Each frame, before its present, the frame generation
//! context is told about it (`ffxConfigure`) and given the frame's depth and motion vectors
//! (`ffxDispatch` of a "prepare", on a command list of ours on mX's queue).
//!
//! Call order (FidelityFX SDK 2.4.0, `ffx_fg.rs` at the end): the swapchain context is made by the
//! maker; the frame generation context from the first frame on (again when the window's size
//! changes); stopping disables it (which waits for AMD's presents), destroys the frame generation
//! context and then the swapchain context, and only then lets mX drop AMD's swapchain.

use crate::Options;
use crate::ffx_fg::*;
use crate::fidelityfx::{Api, Backend, BackendDesc, Coords, Dimensions, Gpu, Header, Resource, api_in, check, ffx_format_of, resource_on};
use crate::frames::{FrameGenMethod, FrameInputs, Show};
use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use parking_lot::Mutex;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use windows::Win32::Graphics::Dxgi::{IDXGISwapChain1, IDXGISwapChain4};
use windows::core::Interface;

type FfxContext = *mut c_void;

/// `ffxDispatch` of the loader in use, for the frame generation callback (which AMD's swapchain
/// calls with the context alone).
static DISPATCH: AtomicUsize = AtomicUsize::new(0);

/// AMD's swapchain asks for the frame generation work here, in its present: dispatched on the
/// frame generation context (`user`, a pointer to it).
unsafe extern "C" fn dispatch_generation(params: *mut ffxDispatchDescFrameGeneration, user: *mut c_void) -> u32 {
    let f = DISPATCH.load(Ordering::Acquire);
    if f == 0 || params.is_null() || user.is_null() {
        return FFX_API_RETURN_ERROR;
    }
    // SAFETY: `f` is the loader's `ffxDispatch`, set before any context was made; `user` the
    // generator's boxed context, alive while it is configured.
    unsafe {
        let dispatch: unsafe extern "C" fn(*mut FfxContext, *const Header) -> u32 = std::mem::transmute(f);
        dispatch(user.cast(), &(*params).header)
    }
}

/// Vulkan: mX's queue for AMD's own submissions (from its present thread), through mX's lock.
static VK_SUBMIT: Mutex<Option<(mx::Queue, ash::Device)>> = Mutex::new(None);

unsafe extern "C" fn submit_on_mx_queue(count: u32, submits: *const vk::SubmitInfo<'static>, fence: vk::Fence) -> vk::Result {
    let Some((queue, device)) = VK_SUBMIT.lock().clone() else {
        return vk::Result::ERROR_DEVICE_LOST;
    };
    let Some(q) = queue.as_vulkan() else {
        return vk::Result::ERROR_DEVICE_LOST;
    };
    q.with_raw(|raw| {
        // SAFETY: AMD's submissions, on mX's queue under its lock.
        let submits = unsafe { std::slice::from_raw_parts(submits, count as usize) };
        match unsafe { device.queue_submit(raw, submits, fence) } {
            Ok(()) => vk::Result::SUCCESS,
            Err(e) => e,
        }
    })
}

/// mX destroys a swapchain AMD made through this: AMD's own goes after its context
/// ([`Generator::stop`]).
unsafe extern "system" fn keep_swapchain(_: vk::Device, _: vk::SwapchainKHR, _: *const vk::AllocationCallbacks<'_>) {}

/// AMD's frame generation context, and what it was made from (AMD may keep pointers into it).
struct FgContext {
    context: Box<FfxContext>,
    display: (u32, u32),
    hudless_format: u32,
    _create: Box<(
        ffxCreateContextDescFrameGeneration,
        ffxCreateContextDescFrameGenerationVersion,
        ffxCreateContextDescFrameGenerationHudless,
    )>,
    _backend: BackendDesc,
}

/// The swapchain AMD made.
enum Swapchain {
    None,
    Dx12 {
        /// AMD's `IDXGISwapChain4` (mX holds the reference).
        raw: *mut c_void,
    },
    Vulkan {
        raw: vk::SwapchainKHR,
        device: vk::Device,
        fns: ffxQueryDescSwapchainReplacementFunctionsVK,
    },
}

struct State {
    gpu: Gpu,
    api: &'static Api,
    backend: Backend,
    swapchain_context: Option<Box<FfxContext>>,
    swapchain: Swapchain,
    /// The swapchain's format, as AMD names it.
    format: u32,
    fg: Option<FgContext>,
    frame: u64,
    show: bool,
    stopped: bool,
    generated: u64,
    /// Vulkan: the device's, for the swapchain AMD replaces.
    vk: Option<crate::vulkan::VkGpu>,
    vk_queue: vk::Queue,
}

// (AMD's objects, used under the lock)
unsafe impl Send for State {}

impl State {
    fn configure_off(&mut self) {
        let Some(fg) = self.fg.as_mut() else { return };
        let desc = configure_desc(&self.swapchain, &mut fg.context, false, Resource::NONE, 0, fg.display, self.frame);
        // SAFETY: a live context; AMD waits for its presents here.
        let _ = unsafe { (self.api.configure)(&mut *fg.context, &desc.header) };
    }

    fn destroy_contexts(&mut self) {
        self.configure_off();
        if let Some(mut fg) = self.fg.take() {
            // SAFETY: disabled above; destroyed once.
            unsafe { (self.api.destroy_context)(&mut *fg.context, std::ptr::null()) };
        }
        if let Some(mut sc) = self.swapchain_context.take() {
            if let Swapchain::Vulkan { .. } = self.swapchain {
                let wait = ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK {
                    header: Header::new(FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK),
                };
                // SAFETY: a live swapchain context.
                let _ = unsafe { (self.api.dispatch)(&mut *sc, &wait.header) };
            } else {
                let wait = ffxDispatchDescFrameGenerationSwapChainWaitForPresentsDX12 {
                    header: Header::new(FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WAIT_FOR_PRESENTS_DX12),
                };
                // SAFETY: as above.
                let _ = unsafe { (self.api.dispatch)(&mut *sc, &wait.header) };
            }
            // SAFETY: the frame generation context is gone; destroyed once.
            unsafe { (self.api.destroy_context)(&mut *sc, std::ptr::null()) };
        }
    }

    /// The frame generation context for this size (made, or made again).
    fn fg_for(&mut self, display: (u32, u32), hudless_format: u32) -> Result<&mut FgContext> {
        if self.fg.as_ref().is_some_and(|f| f.display != display || f.hudless_format != hudless_format) {
            self.configure_off();
            self.backend.wait();
            if let Some(mut fg) = self.fg.take() {
                // SAFETY: disabled above.
                unsafe { (self.api.destroy_context)(&mut *fg.context, std::ptr::null()) };
            }
        }
        if self.fg.is_none() {
            let size = Dimensions { width: display.0, height: display.1 };
            let mut create = Box::new((
                ffxCreateContextDescFrameGeneration {
                    header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION),
                    flags: FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED,
                    displaySize: size,
                    maxRenderSize: size,
                    backBufferFormat: self.format,
                },
                ffxCreateContextDescFrameGenerationVersion {
                    header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION),
                    version: FFX_FRAMEGENERATION_VERSION,
                },
                ffxCreateContextDescFrameGenerationHudless {
                    header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS),
                    hudlessBackBufferFormat: hudless_format,
                },
            ));
            let backend = self.backend.desc();
            create.2.header.next = match &backend {
                BackendDesc::Dx12(b) => &b.header,
                BackendDesc::Vulkan(b) => &b.header,
            };
            create.1.header.next = if hudless_format != self.format { &create.2.header } else { create.2.header.next };
            create.0.header.next = &create.1.header;
            let mut context: Box<FfxContext> = Box::new(std::ptr::null_mut());
            // SAFETY: the descriptions live in `create` / `backend`, kept with the context.
            check("creating the FSR 3 frame generation context", unsafe {
                (self.api.create_context)(&mut *context, &create.0.header, std::ptr::null())
            })?;
            log::info!("FSR 3 frame generation: {}x{}", display.0, display.1);
            self.fg = Some(FgContext {
                context,
                display,
                hudless_format,
                _create: create,
                _backend: backend,
            });
        }
        Ok(self.fg.as_mut().expect("made above"))
    }
}

fn configure_desc(swapchain: &Swapchain, context: &mut Box<FfxContext>, enabled: bool, hudless: Resource, flags: u32, display: (u32, u32), frame: u64) -> ffxConfigureDescFrameGeneration {
    let swap = match swapchain {
        Swapchain::Dx12 { raw } => *raw,
        Swapchain::Vulkan { raw, .. } => vk::Handle::as_raw(*raw) as *mut c_void,
        Swapchain::None => std::ptr::null_mut(),
    };
    ffxConfigureDescFrameGeneration {
        header: Header::new(FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION),
        swapChain: swap,
        presentCallback: None,
        presentCallbackUserContext: std::ptr::null_mut(),
        frameGenerationCallback: Some(dispatch_generation),
        frameGenerationCallbackUserContext: (&mut **context as *mut FfxContext).cast(),
        frameGenerationEnabled: enabled,
        allowAsyncWorkloads: false,
        HUDLessColor: hudless,
        flags,
        onlyPresentGenerated: false,
        generationRect: FfxApiRect2D {
            left: 0,
            top: 0,
            width: display.0 as i32,
            height: display.1 as i32,
        },
        frameID: frame,
    }
}

/// FSR 3 frame generation on one window.
pub struct Generator {
    state: Arc<Mutex<State>>,
}

impl Generator {
    pub fn new(device: &mx::Device, queue: &mx::Queue, surface: &mx::Surface<'_>, format: mx::TextureFormat, options: &Options) -> Result<Generator> {
        let dir = &options.library_dir;
        let backend = Backend::of(device).context("FSR 3 frame generation")?;
        let gpu = backend.gpu();
        let missing = FrameGenMethod::Fsr3.files(gpu.backend()).iter().filter(|f| !dir.join(f).is_file()).copied().collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!("{} not found in {} (they come with AMD's FidelityFX SDK, Kits/FidelityFX/signedbin)", missing.join(", "), dir.display());
        }
        let api = api_in(dir, gpu)?;
        DISPATCH.store(api.dispatch as usize, Ordering::Release);
        let (vk_gpu, vk_queue) = match gpu {
            Gpu::Vulkan => {
                let g = crate::vulkan::VkGpu::of(device).ok_or_else(|| anyhow!("no Vulkan device under mX's"))?;
                let raw = crate::vulkan::with_queue(queue, |q| q)?;
                *VK_SUBMIT.lock() = Some((queue.clone(), g.device.clone()));
                (Some(g), raw)
            }
            Gpu::Dx12 => (None, vk::Queue::null()),
        };
        let state = Arc::new(Mutex::new(State {
            gpu,
            api,
            backend,
            swapchain_context: None,
            swapchain: Swapchain::None,
            format: ffx_format_of(if gpu == Gpu::Dx12 { format.remove_srgb_suffix() } else { format }),
            fg: None,
            frame: 0,
            show: options.show != Show::Both,
            stopped: false,
            generated: 0,
            vk: vk_gpu,
            vk_queue,
        }));
        let installed = match gpu {
            Gpu::Dx12 => {
                let st = state.clone();
                surface.with_dx12(move |s| {
                    s.set_swapchain_maker(Some(Arc::new(move |req: &mx::mx_dx12::SwapchainRequest<'_>| make_dx12(&st, req))));
                })
            }
            Gpu::Vulkan => {
                let st = state.clone();
                surface.with_vulkan(move |s| {
                    s.set_swapchain_maker(Some(Arc::new(move |dev, info: &vk::SwapchainCreateInfoKHR<'_>| make_vulkan(&st, dev, info))));
                })
            }
        };
        if installed.is_none() {
            bail!("FSR 3 frame generation needs the window on the device's graphics API (configure the surface first)");
        }
        log::info!("FSR 3 frame generation: AMD FidelityFX for {gpu:?} loaded from {}", dir.display());
        Ok(Generator { state })
    }

    pub fn prepare(&self, queue: &mx::Queue, inputs: &FrameInputs) -> Result<()> {
        let mut guard = self.state.lock();
        let st = &mut *guard;
        if st.stopped || matches!(st.swapchain, Swapchain::None) {
            return Ok(());
        }
        let gpu = st.gpu;
        let result = (|| -> Result<()> {
            st.frame += 1;
            let frame = st.frame;
            let hudless = resource_on(gpu, &inputs.hudless)?;
            let flags = if st.show { FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW } else { 0 };
            let display = inputs.output_size;
            let hudless_format = hudless.description.format;
            let api = st.api;
            let swapchain = std::mem::replace(&mut st.swapchain, Swapchain::None);
            let fg = st.fg_for(display, hudless_format);
            let fg = match fg {
                Ok(f) => f,
                Err(e) => {
                    st.swapchain = swapchain;
                    return Err(e);
                }
            };
            let desc = configure_desc(&swapchain, &mut fg.context, true, hudless, flags, display, frame);
            let fg_context: *mut FfxContext = &mut *fg.context;
            st.swapchain = swapchain;
            // SAFETY: a live context; the description's resources are mX's, alive this frame.
            check("configuring FSR 3 frame generation", unsafe { (api.configure)(fg_context, &desc.header) })?;
            if let Some(sc) = st.swapchain_context.as_mut() {
                let ui = match &inputs.ui {
                    Some(t) => resource_on(gpu, t)?,
                    None => Resource::NONE,
                };
                let kind = if gpu == Gpu::Vulkan {
                    FFX_API_CONFIGURE_DESC_TYPE_FGSWAPCHAIN_REGISTERUIRESOURCE_VK
                } else {
                    FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_REGISTERUIRESOURCE_DX12
                };
                let register = ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12 {
                    header: Header::new(kind),
                    uiResource: ui,
                    flags: 0,
                };
                // SAFETY: the swapchain context (the DX12 and Vulkan descriptions are laid out alike).
                check("handing FSR 3 the interface", unsafe { (api.configure)(&mut **sc, &register.header) })?;
            }
            let c = &inputs.constants;
            let depth = resource_on(gpu, &inputs.depth)?;
            let motion = resource_on(gpu, &inputs.motion)?;
            let render = inputs.render_size;
            st.backend.record(queue, |command_list| {
                let prepare = ffxDispatchDescFrameGenerationPrepareV2 {
                    header: Header::new(FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2),
                    frameID: frame,
                    flags,
                    commandList: command_list,
                    renderSize: Dimensions { width: render.0, height: render.1 },
                    jitterOffset: Coords { x: c.jitter[0], y: c.jitter[1] },
                    motionVectorScale: Coords { x: 1.0, y: 1.0 },
                    frameTimeDelta: c.frame_ms,
                    reset: c.reset,
                    cameraNear: c.near,
                    cameraFar: c.far,
                    cameraFovAngleVertical: c.fov,
                    viewSpaceToMetersFactor: 1.0,
                    depth,
                    motionVectors: motion,
                    cameraPosition: c.camera_pos,
                    cameraUp: c.camera_up,
                    cameraRight: c.camera_right,
                    cameraForward: c.camera_fwd,
                };
                // SAFETY: a live context, recording into our command list.
                check("FSR 3 frame generation's prepare", unsafe { (api.dispatch)(fg_context, &prepare.header) })
            })?;
            st.generated += 1;
            Ok(())
        })();
        if let Err(e) = &result {
            log::error!("FSR 3 frame generation stopped working ({e:#}); frames are shown without it from now on");
            st.configure_off();
            st.stopped = true;
        }
        result
    }

    /// A frame without the pictures frame generation needs: shown alone.
    pub fn skip(&self) {
        let mut st = self.state.lock();
        st.frame += 1;
        st.configure_off();
    }

    pub fn working(&self) -> bool {
        !self.state.lock().stopped
    }

    pub fn generated(&self) -> u64 {
        self.state.lock().generated
    }

    pub fn wait(&self) {
        self.state.lock().backend.wait();
    }

    pub fn stop(self, surface: &mx::Surface<'_>) {
        let mut st = self.state.lock();
        st.backend.wait();
        st.destroy_contexts();
        let swapchain = std::mem::replace(&mut st.swapchain, Swapchain::None);
        drop(st);
        // (mX lets go of AMD's swapchain: on Direct3D 12 its last reference)
        let _ = surface.with_dx12(|s| s.set_swapchain_maker(None));
        let _ = surface.with_vulkan(|s| s.set_swapchain_maker(None));
        if let Swapchain::Vulkan { raw, device, fns } = swapchain
            && let Some(destroy) = fns.pOutDestroySwapchainFFXAPI
        {
            // SAFETY: its context is gone and mX dropped its side of it.
            unsafe { destroy(device, raw, std::ptr::null(), std::ptr::null_mut()) };
        }
        *VK_SUBMIT.lock() = None;
    }
}

/// The surface's swapchain, made by AMD (Direct3D 12).
fn make_dx12(state: &Arc<Mutex<State>>, req: &mx::mx_dx12::SwapchainRequest<'_>) -> windows::core::Result<IDXGISwapChain1> {
    let mut st = state.lock();
    // (one swapchain to a window: mX let go of the last one; its contexts go too)
    st.destroy_contexts();
    let mut raw: *mut c_void = std::ptr::null_mut();
    let mut desc = *req.desc;
    let version = ffxCreateContextDescFrameGenerationSwapChainVersionDX12 {
        header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12),
        version: FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION,
    };
    let mut create = ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 {
        header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12),
        swapchain: &mut raw,
        hwnd: req.hwnd.0,
        desc: (&mut desc as *mut windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_DESC1).cast(),
        fullscreenDesc: std::ptr::null_mut(),
        dxgiFactory: req.factory.as_raw(),
        gameQueue: req.queue.as_raw(),
    };
    create.header.next = &version.header;
    let mut context: Box<FfxContext> = Box::new(std::ptr::null_mut());
    // SAFETY: the descriptions outlive the call; the window is live (the surface's contract).
    let r = unsafe { (st.api.create_context)(&mut *context, &create.header, std::ptr::null()) };
    if r != 0 || raw.is_null() {
        log::error!("FSR 3 frame generation: AMD made no swapchain ({r}); the window's own is used");
        st.stopped = true;
        // SAFETY: a live window and queue.
        return unsafe { req.factory.CreateSwapChainForHwnd(req.queue, req.hwnd, req.desc, None, None) };
    }
    // SAFETY: AMD hands over one reference to its `IDXGISwapChain4`.
    let swap = unsafe { IDXGISwapChain4::from_raw(raw) };
    st.swapchain_context = Some(context);
    st.swapchain = Swapchain::Dx12 { raw };
    st.stopped = false;
    swap.cast()
}

/// The surface's swapchain, made by AMD in place of the driver's (Vulkan).
#[allow(clippy::field_reassign_with_default)] // (the create info copied field by field, as AMD's sample does)
fn make_vulkan(state: &Arc<Mutex<State>>, device: vk::Device, info: &vk::SwapchainCreateInfoKHR<'_>) -> std::result::Result<(vk::SwapchainKHR, mx::mx_vulkan::SwapchainFns), vk::Result> {
    let mut st = state.lock();
    let fns_of = |f: &ffxQueryDescSwapchainReplacementFunctionsVK| -> std::result::Result<mx::mx_vulkan::SwapchainFns, vk::Result> {
        Ok(mx::mx_vulkan::SwapchainFns {
            destroy: keep_swapchain,
            get_images: f.pOutGetSwapchainImagesKHR.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?,
            acquire: f.pOutAcquireNextImageKHR.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?,
            present: f.pOutQueuePresentKHR.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?,
        })
    };
    // a new size: AMD's swapchain made again in place
    if let Swapchain::Vulkan { raw, fns, .. } = &st.swapchain
        && info.old_swapchain == *raw
        && let Some(create) = fns.pOutCreateSwapchainFFXAPI
    {
        let mut sc = *raw;
        let mut again = *info;
        again.old_swapchain = sc;
        // SAFETY: AMD's own swapchain, made again with its function.
        let r = unsafe {
            create(
                device,
                (&again as *const vk::SwapchainCreateInfoKHR<'_>).cast(),
                std::ptr::null(),
                &mut sc,
                vk::Handle::as_raw(*raw) as *mut c_void,
            )
        };
        if r != vk::Result::SUCCESS {
            return Err(r);
        }
        let table = fns_of(fns)?;
        if let Swapchain::Vulkan { raw, .. } = &mut st.swapchain {
            *raw = sc;
        }
        return Ok((sc, table));
    }
    st.destroy_contexts();
    let gpu = st.vk.clone().ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    // the driver's swapchain first, which AMD replaces
    // SAFETY: the device's own function, by name.
    let driver_create: vk::PFN_vkCreateSwapchainKHR = unsafe {
        let f = (gpu.get_device_proc_addr)(device, c"vkCreateSwapchainKHR".as_ptr()).ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
        std::mem::transmute(f)
    };
    let mut first = *info;
    first.old_swapchain = vk::SwapchainKHR::null();
    let mut sc = vk::SwapchainKHR::null();
    // SAFETY: the surface's own description.
    unsafe { driver_create(device, &first, std::ptr::null(), &mut sc) }.result()?;
    let queue = VkQueueInfoFFXAPI {
        queue: st.vk_queue,
        familyIndex: gpu.family,
        submitFunc: Some(submit_on_mx_queue),
    };
    let version = ffxCreateContextDescFrameGenerationSwapChainVersionVK {
        header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_VK),
        version: FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION,
    };
    let mut create_info: vk::SwapchainCreateInfoKHR<'static> = vk::SwapchainCreateInfoKHR::default();
    create_info.flags = first.flags;
    create_info.surface = first.surface;
    create_info.min_image_count = first.min_image_count;
    create_info.image_format = first.image_format;
    create_info.image_color_space = first.image_color_space;
    create_info.image_extent = first.image_extent;
    create_info.image_array_layers = first.image_array_layers;
    create_info.image_usage = first.image_usage;
    create_info.image_sharing_mode = first.image_sharing_mode;
    create_info.pre_transform = first.pre_transform;
    create_info.composite_alpha = first.composite_alpha;
    create_info.present_mode = first.present_mode;
    create_info.clipped = first.clipped;
    let mut create = ffxCreateContextDescFrameGenerationSwapChainVK {
        header: Header::new(FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK),
        physicalDevice: gpu.physical,
        device,
        swapchain: &mut sc,
        allocator: std::ptr::null_mut(),
        createInfo: create_info,
        gameQueue: queue,
        asyncComputeQueue: queue,
        presentQueue: queue,
        imageAcquireQueue: queue,
    };
    create.header.next = &version.header;
    let mut context: Box<FfxContext> = Box::new(std::ptr::null_mut());
    // SAFETY: the descriptions outlive the call; AMD destroys the driver's swapchain in it.
    let r = unsafe { (st.api.create_context)(&mut *context, &create.header, std::ptr::null()) };
    if r != 0 {
        log::error!("FSR 3 frame generation: AMD made no swapchain ({r}); the window's own is used");
        st.stopped = true;
        // (each table field is the type the transmute makes)
        #[allow(clippy::missing_transmute_annotations)]
        let table = mx::mx_vulkan::SwapchainFns {
            // SAFETY: the device's own functions, by name.
            destroy: unsafe { std::mem::transmute((gpu.get_device_proc_addr)(device, c"vkDestroySwapchainKHR".as_ptr()).ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?) },
            get_images: unsafe { std::mem::transmute((gpu.get_device_proc_addr)(device, c"vkGetSwapchainImagesKHR".as_ptr()).ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?) },
            acquire: unsafe { std::mem::transmute((gpu.get_device_proc_addr)(device, c"vkAcquireNextImageKHR".as_ptr()).ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?) },
            present: unsafe { std::mem::transmute((gpu.get_device_proc_addr)(device, c"vkQueuePresentKHR".as_ptr()).ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?) },
        };
        return Ok((sc, table));
    }
    let mut query = ffxQueryDescSwapchainReplacementFunctionsVK {
        header: Header::new(FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK),
        pOutCreateSwapchainFFXAPI: None,
        pOutDestroySwapchainFFXAPI: None,
        pOutGetSwapchainImagesKHR: None,
        pOutAcquireNextImageKHR: None,
        pOutQueuePresentKHR: None,
        pOutSetHdrMetadataEXT: None,
        pOutGetLastPresentCountFFXAPI: None,
    };
    // SAFETY: the swapchain context just made.
    let r = unsafe { (st.api.query)(&mut *context, &mut query.header) };
    if r != 0 {
        log::error!("FSR 3 frame generation: AMD's swapchain gave no functions ({r})");
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let table = fns_of(&query)?;
    st.swapchain_context = Some(context);
    st.swapchain = Swapchain::Vulkan { raw: sc, device, fns: query };
    st.stopped = false;
    Ok((sc, table))
}

impl Drop for Generator {
    fn drop(&mut self) {
        let mut st = self.state.lock();
        st.backend.wait();
    }
}
