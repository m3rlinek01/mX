//! Intel XeSS Frame Generation (`libxess_fg.dll`, Direct3D 12), with Intel's latency reduction
//! XeLL (`libxell.dll`), which it needs.
//!
//! Intel's swapchain proxy takes the window's place (made by the surface's swapchain maker,
//! `xefgSwapChainD3D12InitFromSwapChainDesc`): the program draws into it and presents as
//! before. Each frame, before its present, the frame's depth, motion vectors and picture
//! before the interface are tagged on a command list of ours (valid only now: Intel copies
//! them), with the camera, under the id of the present that shows the frame.

use crate::Options;
use crate::d3d12::{ListRing, wide};
use crate::frames::{FrameGenMethod, FrameInputs};
use crate::xefg_ffi::*;
use anyhow::{Context, Result, anyhow, bail};
use std::ffi::{CStr, c_void};
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::{IDXGISwapChain1, IDXGISwapChain4};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW};
use windows::core::{Interface, PCSTR, PCWSTR};

struct Api {
    _fg: HMODULE,
    _xell: HMODULE,
    create: PFN_xefgSwapChainD3D12CreateContext,
    init_from_desc: PFN_xefgSwapChainD3D12InitFromSwapChainDesc,
    get_swapchain: PFN_xefgSwapChainD3D12GetSwapChainPtr,
    tag_resource: PFN_xefgSwapChainD3D12TagFrameResource,
    tag_constants: PFN_xefgSwapChainTagFrameConstants,
    set_enabled: PFN_xefgSwapChainSetEnabled,
    set_present_id: PFN_xefgSwapChainSetPresentId,
    set_latency_reduction: PFN_xefgSwapChainSetLatencyReduction,
    set_interpolated: Option<PFN_xefgSwapChainSetNumInterpolatedFrames>,
    destroy: PFN_xefgSwapChainDestroy,
    get_properties: PFN_xefgSwapChainGetProperties,
    xell_create: PFN_xellD3D12CreateContext,
    xell_destroy: PFN_xellDestroyContext,
    xell_sleep_mode: PFN_xellSetSleepMode,
    xell_sleep: PFN_xellSleep,
    xell_marker: PFN_xellAddMarkerData,
}

// (function pointers into DLLs loaded for the process's lifetime)
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

fn get<T: Copy>(module: HMODULE, name: &CStr) -> Result<T> {
    let f = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }.ok_or_else(|| anyhow!("{} is not exported by Intel's library", name.to_string_lossy()))?;
    Ok(unsafe { std::mem::transmute_copy::<_, T>(&f) })
}

fn open(dir: &std::path::Path, name: &str) -> Result<HMODULE> {
    let path = wide(&dir.join(name));
    unsafe { LoadLibraryExW(PCWSTR(path.as_ptr()), None, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }.with_context(|| format!("opening {name}"))
}

fn api(dir: &std::path::Path) -> Result<&'static Api> {
    API.get_or_init(|| {
        (|| -> Result<Api> {
            let fg = open(dir, XEFG_DLL_NAME)?;
            let xell = open(dir, XELL_DLL_NAME)?;
            Ok(Api {
                create: get(fg, c"xefgSwapChainD3D12CreateContext")?,
                init_from_desc: get(fg, c"xefgSwapChainD3D12InitFromSwapChainDesc")?,
                get_swapchain: get(fg, c"xefgSwapChainD3D12GetSwapChainPtr")?,
                tag_resource: get(fg, c"xefgSwapChainD3D12TagFrameResource")?,
                tag_constants: get(fg, c"xefgSwapChainTagFrameConstants")?,
                set_enabled: get(fg, c"xefgSwapChainSetEnabled")?,
                set_present_id: get(fg, c"xefgSwapChainSetPresentId")?,
                set_latency_reduction: get(fg, c"xefgSwapChainSetLatencyReduction")?,
                set_interpolated: get(fg, c"xefgSwapChainSetNumInterpolatedFrames").ok(),
                destroy: get(fg, c"xefgSwapChainDestroy")?,
                get_properties: get(fg, c"xefgSwapChainGetProperties")?,
                xell_create: get(xell, c"xellD3D12CreateContext")?,
                xell_destroy: get(xell, c"xellDestroyContext")?,
                xell_sleep_mode: get(xell, c"xellSetSleepMode")?,
                xell_sleep: get(xell, c"xellSleep")?,
                xell_marker: get(xell, c"xellAddMarkerData")?,
                _fg: fg,
                _xell: xell,
            })
        })()
        .map_err(|e| format!("{e:#}"))
    })
    .as_ref()
    .map_err(|e| anyhow!("{e}"))
}

fn check(what: &str, r: xefg_swapchain_result_t) -> Result<()> {
    if r.0 >= 0 {
        if r.0 > 0 {
            log::debug!("XeSS-FG: {what}: warning {}", r.0);
        }
        Ok(())
    } else {
        bail!("XeSS-FG: {what} failed ({})", r.0)
    }
}

struct State {
    api: &'static Api,
    device: ID3D12Device,
    context: xefg_swapchain_handle_t,
    xell: xell_context_handle_t,
    /// The context made the window's swapchain.
    initialised: bool,
    ring: ListRing,
    present: u32,
    stopped: bool,
    generated: u64,
}

// (Intel's handles, used under the lock)
unsafe impl Send for State {}

impl State {
    fn new_context(&mut self) -> Result<()> {
        if !self.context.is_null() {
            let _ = unsafe { (self.api.destroy)(self.context) };
            self.context = std::ptr::null_mut();
        }
        check("creating the context", unsafe { (self.api.create)(self.device.as_raw(), &mut self.context) })?;
        self.initialised = false;
        Ok(())
    }
}

/// XeSS-FG on one window.
pub struct Generator {
    state: Arc<Mutex<State>>,
}

impl Generator {
    pub fn new(device: &mx::Device, _queue: &mx::Queue, surface: &mx::Surface<'_>, options: &Options) -> Result<Generator> {
        let dir = &options.library_dir;
        let d = device.as_dx12().ok_or_else(|| anyhow!("XeSS Frame Generation runs on Direct3D 12 only"))?;
        let missing = FrameGenMethod::Xess.files(mx::Backend::Dx12).iter().filter(|f| !dir.join(f).is_file()).copied().collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!("{} not found in {} (they come with Intel's XeSS SDK, bin)", missing.join(", "), dir.display());
        }
        let api = api(dir)?;
        let raw = d.raw().clone();
        let mut xell: xell_context_handle_t = std::ptr::null_mut();
        let r = unsafe { (api.xell_create)(raw.as_raw(), &mut xell) };
        if r.0 < 0 {
            bail!("XeLL, which XeSS Frame Generation needs, does not run here ({})", r.0);
        }
        let sleep = xell_sleep_params_t {
            minimumIntervalUs: 0,
            flags: XELL_SLEEP_FLAG_LOW_LATENCY_MODE,
        };
        let _ = unsafe { (api.xell_sleep_mode)(xell, &sleep) };
        let mut state = State {
            api,
            device: raw.clone(),
            context: std::ptr::null_mut(),
            xell,
            initialised: false,
            ring: ListRing::new(&raw).context("XeSS-FG's command lists")?,
            present: 0,
            stopped: false,
            generated: 0,
        };
        if let Err(e) = state.new_context() {
            unsafe {
                let _ = (api.xell_destroy)(xell);
            }
            return Err(e.context("XeSS Frame Generation does not run on this card"));
        }
        let state = Arc::new(Mutex::new(state));
        let st = state.clone();
        surface
            .with_dx12(move |s| s.set_swapchain_maker(Some(Arc::new(move |req: &mx::mx_dx12::SwapchainRequest<'_>| make_dx12(&st, req)))))
            .ok_or_else(|| anyhow!("XeSS Frame Generation needs the window on Direct3D 12 (configure the surface first)"))?;
        log::info!("XeSS Frame Generation loaded from {}", dir.display());
        Ok(Generator { state })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn prepare(&self, queue: &mx::Queue, inputs: &FrameInputs) -> Result<()> {
        let mut guard = self.lock();
        let st = &mut *guard;
        if st.stopped || !st.initialised {
            return Ok(());
        }
        let result = (|| -> Result<()> {
            st.present = st.present.wrapping_add(1).max(1);
            let id = st.present;
            let api = st.api;
            unsafe {
                let _ = (api.xell_sleep)(st.xell, id);
                let _ = (api.xell_marker)(st.xell, id, XELL_SIMULATION_START);
                let _ = (api.xell_marker)(st.xell, id, XELL_SIMULATION_END);
                let _ = (api.xell_marker)(st.xell, id, XELL_RENDERSUBMIT_START);
            }
            let c = &inputs.constants;
            let flat = |m: [[f32; 4]; 4]| -> [f32; 16] { std::array::from_fn(|i| m[i / 4][i % 4]) };
            let constants = xefg_swapchain_frame_constant_data_t {
                viewMatrix: flat(c.world_to_view),
                projectionMatrix: flat(c.view_to_clip),
                jitterOffsetX: c.jitter[0],
                jitterOffsetY: c.jitter[1],
                motionVectorScaleX: 1.0,
                motionVectorScaleY: 1.0,
                resetHistory: c.reset as u32,
                frameRenderTime: c.frame_ms,
            };
            check("the frame's camera", unsafe { (api.tag_constants)(st.context, id, &constants) })?;
            let mut tags: Vec<(xefg_swapchain_resource_type_t, &mx::Texture, (u32, u32))> = vec![
                (XEFG_SWAPCHAIN_RES_DEPTH, &inputs.depth, inputs.render_size),
                (XEFG_SWAPCHAIN_RES_MOTION_VECTOR, &inputs.motion, inputs.render_size),
                (XEFG_SWAPCHAIN_RES_HUDLESS_COLOR, &inputs.hudless, inputs.output_size),
            ];
            if let Some(ui) = &inputs.ui {
                tags.push((XEFG_SWAPCHAIN_RES_UI, ui, inputs.output_size));
            }
            let raw_queue = queue.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 queue under mX's"))?.raw().clone();
            let list = st.ring.begin()?;
            let mut recorded = Ok(());
            for (kind, t, (w, h)) in tags {
                let d = t.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 resource under an XeSS-FG texture"))?;
                let data = xefg_swapchain_d3d12_resource_data_t {
                    r#type: kind,
                    validity: XEFG_SWAPCHAIN_RV_ONLY_NOW,
                    resourceBase: xefg_swapchain_2d_t { x: 0, y: 0 },
                    resourceSize: xefg_swapchain_2d_t { x: w, y: h },
                    pResource: d.raw().as_raw(),
                    incomingState: d.rest_state().0,
                };
                let r = unsafe { (api.tag_resource)(st.context, list.as_raw(), id, &data) };
                if recorded.is_ok() {
                    recorded = check("tagging the frame", r);
                }
            }
            drop(list);
            st.ring.submit(&raw_queue)?;
            recorded?;
            check("the present's id", unsafe { (api.set_present_id)(st.context, id) })?;
            unsafe {
                let _ = (api.xell_marker)(st.xell, id, XELL_RENDERSUBMIT_END);
            }
            Ok(())
        })();
        if let Err(e) = &result {
            log::error!("XeSS Frame Generation stopped working ({e:#}); frames are shown without it from now on");
            st.stopped = true;
            let _ = unsafe { (st.api.set_enabled)(st.context, 0) };
        }
        result
    }

    /// XeLL's markers around the window's present.
    pub fn presenting(&self, start: bool) {
        let mut st = self.lock();
        if st.present == 0 {
            return;
        }
        let marker = if start { XELL_PRESENT_START } else { XELL_PRESENT_END };
        unsafe {
            let _ = (st.api.xell_marker)(st.xell, st.present, marker);
        }
        if !start && st.initialised && !st.stopped {
            st.generated += 1;
        }
    }

    pub fn skip(&self) {
        let st = self.lock();
        if st.initialised {
            // (the next present shows the frame alone: Intel skips interpolation for it)
            let reset = xefg_swapchain_frame_constant_data_t {
                viewMatrix: [0.0; 16],
                projectionMatrix: [0.0; 16],
                jitterOffsetX: 0.0,
                jitterOffsetY: 0.0,
                motionVectorScaleX: 1.0,
                motionVectorScaleY: 1.0,
                resetHistory: 1,
                frameRenderTime: 0.0,
            };
            let _ = unsafe { (st.api.tag_constants)(st.context, st.present.wrapping_add(1), &reset) };
        }
    }

    pub fn working(&self) -> bool {
        !self.lock().stopped
    }

    pub fn generated(&self) -> u64 {
        self.lock().generated
    }

    pub fn wait(&self) {
        self.lock().ring.wait();
    }

    pub fn stop(self, surface: &mx::Surface<'_>) {
        {
            let mut st = self.lock();
            st.ring.wait();
            let _ = unsafe { (st.api.set_enabled)(st.context, 0) };
            st.stopped = true;
        }
        // (mX lets go of Intel's proxy first: the context goes once no one holds it)
        let _ = surface.with_dx12(|s| s.set_swapchain_maker(None));
        let mut st = self.lock();
        if !st.context.is_null() {
            let r = unsafe { (st.api.destroy)(st.context) };
            if r.0 < 0 {
                log::warn!("XeSS-FG: its context did not go ({})", r.0);
            }
            st.context = std::ptr::null_mut();
        }
        if !st.xell.is_null() {
            unsafe {
                let _ = (st.api.xell_destroy)(st.xell);
            }
            st.xell = std::ptr::null_mut();
        }
    }
}

/// The surface's swapchain, Intel's proxy (Direct3D 12).
fn make_dx12(state: &Arc<Mutex<State>>, req: &mx::mx_dx12::SwapchainRequest<'_>) -> windows::core::Result<IDXGISwapChain1> {
    let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
    let own = || unsafe { req.factory.CreateSwapChainForHwnd(req.queue, req.hwnd, req.desc, None, None) };
    // (a context makes one swapchain: a new one for the window's next)
    if st.initialised
        && let Err(e) = st.new_context()
    {
        log::error!("{e:#}; the window's own swapchain is used");
        st.stopped = true;
        return own();
    }
    let api = st.api;
    let mut props = xefg_swapchain_properties_t::default();
    let _ = unsafe { (api.get_properties)(st.context, &mut props) };
    let init = xefg_swapchain_d3d12_init_params_t {
        pApplicationSwapChain: std::ptr::null_mut(),
        initFlags: XEFG_SWAPCHAIN_INIT_FLAG_INVERTED_DEPTH,
        maxInterpolatedFrames: 1,
        creationNodeMask: 0,
        visibleNodeMask: 0,
        pTempBufferHeap: std::ptr::null_mut(),
        bufferHeapOffset: 0,
        pTempTextureHeap: std::ptr::null_mut(),
        textureHeapOffset: 0,
        pPipelineLibrary: std::ptr::null_mut(),
        uiMode: XEFG_SWAPCHAIN_UI_MODE_AUTO,
    };
    let r = unsafe {
        (api.init_from_desc)(
            st.context,
            req.hwnd.0,
            (req.desc as *const windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_DESC1).cast(),
            std::ptr::null(),
            req.queue.as_raw(),
            req.factory.as_raw(),
            &init,
        )
    };
    if r.0 < 0 {
        log::error!("XeSS Frame Generation: Intel made no swapchain ({}); the window's own is used", r.0);
        st.stopped = true;
        return own();
    }
    let _ = unsafe { (api.set_latency_reduction)(st.context, st.xell.cast()) };
    if let Some(n) = api.set_interpolated {
        let _ = unsafe { n(st.context, 1) };
    }
    let mut raw: *mut c_void = std::ptr::null_mut();
    let iid = IDXGISwapChain4::IID;
    let r = unsafe { (api.get_swapchain)(st.context, (&iid as *const windows::core::GUID).cast(), &mut raw) };
    if r.0 < 0 || raw.is_null() {
        log::error!("XeSS Frame Generation: no swapchain from Intel's context ({})", r.0);
        st.stopped = true;
        return own();
    }
    // SAFETY: an AddRef'd `IDXGISwapChain4`, ours.
    let swap = unsafe { IDXGISwapChain4::from_raw(raw) };
    let _ = unsafe { (api.set_enabled)(st.context, 1) };
    st.initialised = true;
    st.stopped = false;
    log::info!(
        "XeSS Frame Generation: Intel's swapchain is the window's (up to {} pictures between two)",
        props.maxSupportedInterpolations.max(1)
    );
    swap.cast()
}

impl Drop for Generator {
    fn drop(&mut self) {
        self.lock().ring.wait();
    }
}
