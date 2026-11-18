//! Window surfaces and swapchains.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ash::{khr, vk};
use mx_types as t;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::device::{InstanceShared, Recording, Shared};
use crate::{Adapter, CommandBuffer, Device, Error, Instance, Result, Texture, conv};

trait Window: HasWindowHandle + HasDisplayHandle {}
impl<W: HasWindowHandle + HasDisplayHandle> Window for W {}

/// An Objective-C object used from any thread (the layer is only handed to Vulkan, and its
/// view only read for the scale factor and occlusion).
#[cfg(target_vendor = "apple")]
struct Mt<T>(T);
// SAFETY: see above.
#[cfg(target_vendor = "apple")]
unsafe impl<T> Send for Mt<T> {}
// SAFETY: see above.
#[cfg(target_vendor = "apple")]
unsafe impl<T> Sync for Mt<T> {}

/// The `VkSurfaceKHR` (and on Apple platforms the `CAMetalLayer` under it). Swapchains hold
/// it, so it goes after them.
struct SurfaceRaw {
    raw: vk::SurfaceKHR,
    instance: Arc<InstanceShared>,
    #[cfg(target_vendor = "apple")]
    _layer: Mt<objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>>,
    #[cfg(target_os = "macos")]
    view: Mt<objc2::rc::Retained<objc2_app_kit::NSView>>,
}

impl Drop for SurfaceRaw {
    fn drop(&mut self) {
        if let Some(s) = &self.instance.surface {
            // SAFETY: every swapchain of the surface is gone (they hold this).
            unsafe { s.destroy_surface(self.raw, None) };
        }
    }
}

/// The swapchain functions a surface calls: the driver's, or those of a vendor's frame
/// generation that made the swapchain (`Surface::set_swapchain_maker`).
#[derive(Clone, Copy)]
pub struct SwapchainFns {
    pub destroy: vk::PFN_vkDestroySwapchainKHR,
    pub get_images: vk::PFN_vkGetSwapchainImagesKHR,
    pub acquire: vk::PFN_vkAcquireNextImageKHR,
    pub present: vk::PFN_vkQueuePresentKHR,
}

impl SwapchainFns {
    fn of(f: &ash::khr::swapchain::Device) -> SwapchainFns {
        let fp = f.fp();
        SwapchainFns {
            destroy: fp.destroy_swapchain_khr,
            get_images: fp.get_swapchain_images_khr,
            acquire: fp.acquire_next_image_khr,
            present: fp.queue_present_khr,
        }
    }
}

/// Makes a surface's swapchain in place of `vkCreateSwapchainKHR`, and gives the functions
/// to use it with.
pub type SwapchainMaker = Arc<dyn Fn(vk::Device, &vk::SwapchainCreateInfoKHR<'_>) -> std::result::Result<(vk::SwapchainKHR, SwapchainFns), vk::Result> + Send + Sync>;

struct Swapchain {
    raw: vk::SwapchainKHR,
    fns: SwapchainFns,
    _surface: Arc<SurfaceRaw>,
    shared: Arc<Shared>,
    images: Vec<vk::Image>,
    /// Semaphores the acquires signal, in turn, and the submission that last waited on each.
    acquire: Vec<vk::Semaphore>,
    acquire_used: Mutex<Vec<u64>>,
    next_acquire: AtomicUsize,
    /// One per image: signalled by the submission before its present.
    present: Vec<vk::Semaphore>,
    /// Submissions of the last frames presented, for the frame latency.
    presented: Mutex<VecDeque<u64>>,
    latency: usize,
    config: t::SurfaceConfiguration,
    outdated: AtomicBool,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let shared = &self.shared;
        // SAFETY: the queue (and with it the presentation) is idle; nothing uses the
        // swapchain's semaphores or images any more.
        unsafe {
            let q = shared.queue.lock();
            let _ = shared.raw.queue_wait_idle(*q);
            drop(q);
            let _ = shared.wait_idle();
            for s in self.acquire.iter().chain(&self.present) {
                shared.raw.destroy_semaphore(*s, None);
            }
            (self.fns.destroy)(shared.raw.handle(), self.raw, std::ptr::null());
        }
    }
}

struct State {
    /// (keeps the device open while the surface is configured for it)
    _device: Device,
    /// `None` while the window has no area.
    swapchain: Option<Arc<Swapchain>>,
}

/// Where a window's frames go.
pub struct Surface<'w> {
    raw: Arc<SurfaceRaw>,
    state: Mutex<Option<State>>,
    maker: Mutex<Option<SwapchainMaker>>,
    _window: Option<Box<dyn Window + Send + Sync + 'w>>,
    _lifetime: PhantomData<&'w ()>,
}

impl Instance {
    /// A surface for a window (a winit `Window`, `Arc<Window>` or `&Window`). On Apple
    /// platforms call on the main thread: a `CAMetalLayer` is put into the window's view.
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>> {
        let raw_window = window.window_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        let raw_display = window.display_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        // SAFETY: the handles are valid while `window` lives, and the surface keeps it.
        let s = unsafe { self.create_surface_unsafe(raw_display, raw_window)? };
        Ok(Surface {
            raw: s.raw.clone(),
            state: Mutex::new(None),
            maker: Mutex::new(None),
            _window: Some(Box::new(window)),
            _lifetime: PhantomData,
        })
    }

    /// A surface from raw handles.
    ///
    /// # Safety
    /// The window must outlive the surface (and on Apple platforms this must run on the main
    /// thread).
    pub unsafe fn create_surface_unsafe(&self, display: RawDisplayHandle, window: RawWindowHandle) -> Result<Surface<'static>> {
        let inst = self.shared.as_ref().ok_or(Error::NoDevice)?;
        if inst.surface.is_none() {
            return Err(Error::Surface("the Vulkan loader has no VK_KHR_surface".into()));
        }
        let missing = |what: &str| Error::Surface(format!("the Vulkan loader has no {what}"));
        let err = |e: vk::Result| Error::Surface(format!("creating the surface: {e}"));
        let _ = display;
        #[allow(unreachable_patterns)]
        let raw = match window {
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(h) => {
                use objc2_quartz_core::CAMetalLayer;
                if !inst.has(ash::ext::metal_surface::NAME) {
                    return Err(missing("VK_EXT_metal_surface"));
                }
                // SAFETY: the handle points to a live NSView (caller's contract).
                let view: objc2::rc::Retained<objc2_app_kit::NSView> = unsafe { objc2::rc::Retained::retain(h.ns_view.as_ptr().cast()) }.ok_or_else(|| Error::Surface("null NSView".into()))?;
                let layer = match view.layer().and_then(|l| l.downcast::<CAMetalLayer>().ok()) {
                    Some(l) => l,
                    None => {
                        let layer = CAMetalLayer::new();
                        // (a layer-hosting view: the view draws nothing itself)
                        view.setLayer(Some(&layer));
                        view.setWantsLayer(true);
                        layer
                    }
                };
                if let Some(w) = view.window() {
                    layer.setContentsScale(w.backingScaleFactor());
                }
                let f = ash::ext::metal_surface::Instance::new(&inst.entry, &inst.raw);
                let ptr: *const CAMetalLayer = &*layer;
                // SAFETY: a live layer.
                let raw = unsafe { f.create_metal_surface(&vk::MetalSurfaceCreateInfoEXT::default().layer(ptr.cast::<std::ffi::c_void>()), None) }.map_err(err)?;
                return Ok(Surface {
                    raw: Arc::new(SurfaceRaw {
                        raw,
                        instance: inst.clone(),
                        _layer: Mt(layer),
                        view: Mt(view),
                    }),
                    state: Mutex::new(None),
                    maker: Mutex::new(None),
                    _window: None,
                    _lifetime: PhantomData,
                });
            }
            #[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
            RawWindowHandle::UiKit(h) => {
                use objc2_quartz_core::CAMetalLayer;
                if !inst.has(ash::ext::metal_surface::NAME) {
                    return Err(missing("VK_EXT_metal_surface"));
                }
                // SAFETY: the handle points to a live UIView (caller's contract).
                let view: objc2::rc::Retained<objc2_ui_kit::UIView> = unsafe { objc2::rc::Retained::retain(h.ui_view.as_ptr().cast()) }.ok_or_else(|| Error::Surface("null UIView".into()))?;
                let main = view.layer();
                let layer = CAMetalLayer::new();
                layer.setFrame(main.bounds());
                layer.setContentsScale(main.contentsScale());
                main.addSublayer(&layer);
                let f = ash::ext::metal_surface::Instance::new(&inst.entry, &inst.raw);
                let ptr: *const CAMetalLayer = &*layer;
                // SAFETY: a live layer.
                let raw = unsafe { f.create_metal_surface(&vk::MetalSurfaceCreateInfoEXT::default().layer(ptr.cast::<std::ffi::c_void>()), None) }.map_err(err)?;
                return Ok(Surface {
                    raw: Arc::new(SurfaceRaw {
                        raw,
                        instance: inst.clone(),
                        _layer: Mt(layer),
                    }),
                    state: Mutex::new(None),
                    maker: Mutex::new(None),
                    _window: None,
                    _lifetime: PhantomData,
                });
            }
            RawWindowHandle::Win32(h) => {
                if !inst.has(khr::win32_surface::NAME) {
                    return Err(missing("VK_KHR_win32_surface"));
                }
                let f = khr::win32_surface::Instance::new(&inst.entry, &inst.raw);
                let info = vk::Win32SurfaceCreateInfoKHR::default().hinstance(h.hinstance.map_or(0, |h| h.get())).hwnd(h.hwnd.get());
                // SAFETY: live window handles (caller's contract).
                unsafe { f.create_win32_surface(&info, None) }.map_err(err)?
            }
            RawWindowHandle::Xlib(h) => {
                let RawDisplayHandle::Xlib(d) = display else {
                    return Err(Error::Surface("an Xlib window needs an Xlib display".into()));
                };
                if !inst.has(khr::xlib_surface::NAME) {
                    return Err(missing("VK_KHR_xlib_surface"));
                }
                let f = khr::xlib_surface::Instance::new(&inst.entry, &inst.raw);
                let dpy = d.display.map_or(std::ptr::null_mut(), |p| p.as_ptr());
                // SAFETY: live window handles (caller's contract).
                unsafe { f.create_xlib_surface(&vk::XlibSurfaceCreateInfoKHR::default().dpy(dpy.cast()).window(h.window as _), None) }.map_err(err)?
            }
            RawWindowHandle::Xcb(h) => {
                let RawDisplayHandle::Xcb(d) = display else {
                    return Err(Error::Surface("an XCB window needs an XCB connection".into()));
                };
                if !inst.has(khr::xcb_surface::NAME) {
                    return Err(missing("VK_KHR_xcb_surface"));
                }
                let f = khr::xcb_surface::Instance::new(&inst.entry, &inst.raw);
                let conn = d.connection.map_or(std::ptr::null_mut(), |p| p.as_ptr());
                // SAFETY: live window handles (caller's contract).
                unsafe { f.create_xcb_surface(&vk::XcbSurfaceCreateInfoKHR::default().connection(conn).window(h.window.get()), None) }.map_err(err)?
            }
            RawWindowHandle::Wayland(h) => {
                let RawDisplayHandle::Wayland(d) = display else {
                    return Err(Error::Surface("a Wayland surface needs a Wayland display".into()));
                };
                if !inst.has(khr::wayland_surface::NAME) {
                    return Err(missing("VK_KHR_wayland_surface"));
                }
                let f = khr::wayland_surface::Instance::new(&inst.entry, &inst.raw);
                let info = vk::WaylandSurfaceCreateInfoKHR::default().display(d.display.as_ptr()).surface(h.surface.as_ptr());
                // SAFETY: live window handles (caller's contract).
                unsafe { f.create_wayland_surface(&info, None) }.map_err(err)?
            }
            RawWindowHandle::AndroidNdk(h) => {
                if !inst.has(khr::android_surface::NAME) {
                    return Err(missing("VK_KHR_android_surface"));
                }
                let f = khr::android_surface::Instance::new(&inst.entry, &inst.raw);
                // SAFETY: a live native window (caller's contract).
                unsafe { f.create_android_surface(&vk::AndroidSurfaceCreateInfoKHR::default().window(h.a_native_window.as_ptr()), None) }.map_err(err)?
            }
            other => return Err(Error::Surface(format!("not a window Vulkan can draw into here: {other:?}"))),
        };
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Surface {
                raw: Arc::new(SurfaceRaw { raw, instance: inst.clone() }),
                state: Mutex::new(None),
                maker: Mutex::new(None),
                _window: None,
                _lifetime: PhantomData,
            })
        }
        #[cfg(target_vendor = "apple")]
        {
            // (Apple windows returned above; other handles do not exist there)
            let _ = raw;
            Err(Error::Surface("not a window Vulkan can draw into on this platform".into()))
        }
    }
}

fn present_mode(want: t::PresentMode, have: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    use vk::PresentModeKHR as P;
    let order: &[P] = match want {
        t::PresentMode::AutoVsync => &[P::FIFO_RELAXED, P::FIFO],
        t::PresentMode::AutoNoVsync => &[P::IMMEDIATE, P::MAILBOX, P::FIFO],
        t::PresentMode::Fifo => &[P::FIFO],
        t::PresentMode::FifoRelaxed => &[P::FIFO_RELAXED, P::FIFO],
        t::PresentMode::Immediate => &[P::IMMEDIATE, P::FIFO],
        t::PresentMode::Mailbox => &[P::MAILBOX, P::FIFO],
    };
    order.iter().copied().find(|m| have.contains(m)).unwrap_or(P::FIFO)
}

fn alpha_mode(want: t::CompositeAlphaMode, have: vk::CompositeAlphaFlagsKHR) -> vk::CompositeAlphaFlagsKHR {
    use vk::CompositeAlphaFlagsKHR as A;
    let first = match want {
        t::CompositeAlphaMode::Opaque | t::CompositeAlphaMode::Auto => A::OPAQUE,
        t::CompositeAlphaMode::PreMultiplied => A::PRE_MULTIPLIED,
        t::CompositeAlphaMode::PostMultiplied => A::POST_MULTIPLIED,
        t::CompositeAlphaMode::Inherit => A::INHERIT,
    };
    [first, A::OPAQUE, A::INHERIT, A::PRE_MULTIPLIED, A::POST_MULTIPLIED]
        .into_iter()
        .find(|a| have.contains(*a))
        .unwrap_or(first)
}

impl Surface<'_> {
    /// Whether `adapter`'s queue can present to this surface (a second graphics card may
    /// not reach the window's screen).
    pub fn is_supported_by(&self, adapter: &Adapter) -> bool {
        let inst = &self.raw.instance;
        let Some(s) = &inst.surface else { return false };
        // SAFETY: a plain query of a live surface, with the family the device is made on.
        unsafe { s.get_physical_device_surface_support(adapter.0.phys, adapter.0.caps.family, self.raw.raw) }.unwrap_or(false)
    }

    pub fn get_capabilities(&self, adapter: &Adapter) -> t::SurfaceCapabilities {
        let inst = &self.raw.instance;
        let Some(s) = &inst.surface else { return Default::default() };
        let phys = adapter.0.phys;
        // SAFETY: plain queries of a live surface.
        let (formats, modes, caps) = unsafe {
            (
                s.get_physical_device_surface_formats(phys, self.raw.raw).unwrap_or_default(),
                s.get_physical_device_surface_present_modes(phys, self.raw.raw).unwrap_or_default(),
                s.get_physical_device_surface_capabilities(phys, self.raw.raw).unwrap_or_default(),
            )
        };
        let mut list: Vec<t::TextureFormat> = Vec::new();
        for f in formats.iter().filter(|f| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR) {
            if let Some(m) = conv::surface_format(f.format)
                && !list.contains(&m)
            {
                list.push(m);
            }
        }
        // (the preferred format first: an sRGB one, as wgpu's examples pick)
        list.sort_by_key(|f| !f.is_srgb());
        let mut present_modes = vec![t::PresentMode::Fifo, t::PresentMode::AutoVsync, t::PresentMode::AutoNoVsync];
        for (v, m) in [
            (vk::PresentModeKHR::FIFO_RELAXED, t::PresentMode::FifoRelaxed),
            (vk::PresentModeKHR::IMMEDIATE, t::PresentMode::Immediate),
            (vk::PresentModeKHR::MAILBOX, t::PresentMode::Mailbox),
        ] {
            if modes.contains(&v) {
                present_modes.push(m);
            }
        }
        let a = caps.supported_composite_alpha;
        let alpha_modes = [
            (vk::CompositeAlphaFlagsKHR::OPAQUE, t::CompositeAlphaMode::Opaque),
            (vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED, t::CompositeAlphaMode::PreMultiplied),
            (vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED, t::CompositeAlphaMode::PostMultiplied),
            (vk::CompositeAlphaFlagsKHR::INHERIT, t::CompositeAlphaMode::Inherit),
        ]
        .into_iter()
        .filter(|(v, _)| a.contains(*v))
        .map(|(_, m)| m)
        .collect();
        let u = caps.supported_usage_flags;
        let mut usages = t::TextureUsages::empty();
        for (v, m) in [
            (vk::ImageUsageFlags::COLOR_ATTACHMENT, t::TextureUsages::RENDER_ATTACHMENT),
            (vk::ImageUsageFlags::TRANSFER_SRC, t::TextureUsages::COPY_SRC),
            (vk::ImageUsageFlags::TRANSFER_DST, t::TextureUsages::COPY_DST),
            (vk::ImageUsageFlags::SAMPLED, t::TextureUsages::TEXTURE_BINDING),
            (vk::ImageUsageFlags::STORAGE, t::TextureUsages::STORAGE_BINDING),
        ] {
            if u.contains(v) {
                usages |= m;
            }
        }
        t::SurfaceCapabilities {
            formats: list,
            present_modes,
            alpha_modes,
            usages,
        }
    }

    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration) {
        let shared = &device.shared;
        let inst = &self.raw.instance;
        let s = inst.surface.as_ref().expect("VK_KHR_surface");
        let f = shared.fns.swapchain.as_ref().expect("the device has no VK_KHR_swapchain");
        // SAFETY: plain queries of a live surface.
        let (supported, caps, modes) = unsafe {
            (
                s.get_physical_device_surface_support(shared.phys, shared.caps.family, self.raw.raw).unwrap_or(false),
                s.get_physical_device_surface_capabilities(shared.phys, self.raw.raw)
                    .expect("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"),
                s.get_physical_device_surface_present_modes(shared.phys, self.raw.raw).unwrap_or_default(),
            )
        };
        assert!(supported, "the device's queue cannot present to this surface");
        let mut state = self.state.lock();
        let old = state.take();
        let width = config.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width.max(caps.min_image_extent.width));
        let height = config.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height.max(caps.min_image_extent.height));
        if config.width == 0 || config.height == 0 || width == 0 || height == 0 {
            *state = Some(State {
                _device: device.clone(),
                swapchain: None,
            });
            return;
        }
        let mut count = (config.desired_maximum_frame_latency + 1).max(caps.min_image_count);
        if caps.max_image_count > 0 {
            count = count.min(caps.max_image_count);
        }
        let usage = conv::image_usage(config.usage, config.format) & caps.supported_usage_flags | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let old_raw = old.as_ref().and_then(|o| o.swapchain.as_ref()).map_or(vk::SwapchainKHR::null(), |s| s.raw);
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.raw.raw)
            .min_image_count(count)
            .image_format(conv::format(config.format, false))
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D { width, height })
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(if caps.supported_transforms.contains(vk::SurfaceTransformFlagsKHR::IDENTITY) {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                caps.current_transform
            })
            .composite_alpha(alpha_mode(config.alpha_mode, caps.supported_composite_alpha))
            .present_mode(present_mode(config.present_mode, &modes))
            .clipped(true)
            .old_swapchain(old_raw);
        // SAFETY: a live surface; the old swapchain is retired by this call and destroyed
        // below once idle.
        let maker = self.maker.lock().clone();
        let (raw, fns) = match maker {
            Some(make) => make(shared.raw.handle(), &info).expect("the surface's swapchain maker failed"),
            None => (unsafe { f.create_swapchain(&info, None) }.expect("vkCreateSwapchainKHR"), SwapchainFns::of(f)),
        };
        drop(old);
        // SAFETY: plain queries and creations.
        let (images, acquire, present) = unsafe {
            let images = {
                let dev = shared.raw.handle();
                let mut n = 0;
                (fns.get_images)(dev, raw, &mut n, std::ptr::null_mut()).result().expect("vkGetSwapchainImagesKHR");
                let mut images = vec![vk::Image::null(); n as usize];
                (fns.get_images)(dev, raw, &mut n, images.as_mut_ptr()).result().expect("vkGetSwapchainImagesKHR");
                images.truncate(n as usize);
                images
            };
            let sem = || shared.raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).expect("vkCreateSemaphore");
            let acquire: Vec<vk::Semaphore> = (0..images.len() + 1).map(|_| sem()).collect();
            let present: Vec<vk::Semaphore> = images.iter().map(|_| sem()).collect();
            (images, acquire, present)
        };
        let n = acquire.len();
        *state = Some(State {
            _device: device.clone(),
            swapchain: Some(Arc::new(Swapchain {
                raw,
                fns,
                _surface: self.raw.clone(),
                shared: shared.clone(),
                images,
                acquire,
                acquire_used: Mutex::new(vec![0; n]),
                next_acquire: AtomicUsize::new(0),
                present,
                presented: Mutex::new(VecDeque::new()),
                latency: config.desired_maximum_frame_latency.max(1) as usize,
                config: t::SurfaceConfiguration { width, height, ..config.clone() },
                outdated: AtomicBool::new(false),
            })),
        });
    }

    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture> {
        let guard = self.state.lock();
        let Some(state) = guard.as_ref() else {
            panic!("get_current_texture on a surface that was never configured");
        };
        let Some(sc) = state.swapchain.clone() else {
            return t::CurrentSurfaceTexture::Occluded;
        };
        drop(guard);
        #[cfg(target_os = "macos")]
        if let Some(w) = self.raw.view.0.window()
            && !w.occlusionState().contains(objc2_app_kit::NSWindowOcclusionState::Visible)
        {
            return t::CurrentSurfaceTexture::Occluded;
        }
        if sc.outdated.load(Ordering::Acquire) {
            return t::CurrentSurfaceTexture::Outdated;
        }
        let shared = &sc.shared;
        // the CPU runs at most `latency` frames ahead
        loop {
            let front = {
                let p = sc.presented.lock();
                if p.len() < sc.latency {
                    break;
                }
                p[0]
            };
            if shared.poll(Some(front)).is_err() {
                return t::CurrentSurfaceTexture::Lost;
            }
            sc.presented.lock().pop_front();
        }
        let slot = sc.next_acquire.fetch_add(1, Ordering::AcqRel) % sc.acquire.len();
        let used = sc.acquire_used.lock()[slot];
        if shared.poll(Some(used)).is_err() {
            return t::CurrentSurfaceTexture::Lost;
        }
        let sem = sc.acquire[slot];
        // SAFETY: the semaphore is unsignalled with no pending wait (its last waiter is done).
        let mut index = 0;
        let acquired = unsafe { (sc.fns.acquire)(shared.raw.handle(), sc.raw, 1_000_000_000, sem, vk::Fence::null(), &mut index) };
        let suboptimal = acquired == vk::Result::SUBOPTIMAL_KHR;
        match acquired {
            vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => {}
            vk::Result::TIMEOUT | vk::Result::NOT_READY => return t::CurrentSurfaceTexture::Timeout,
            vk::Result::ERROR_OUT_OF_DATE_KHR => return t::CurrentSurfaceTexture::Outdated,
            e => {
                log::error!("mX: vkAcquireNextImageKHR: {e}");
                return t::CurrentSurfaceTexture::Lost;
            }
        };
        // the next submission waits for the image; it starts in GENERAL with no contents
        let image = sc.images[index as usize];
        shared.pending_waits.lock().push(sem);
        sc.acquire_used.lock()[slot] = shared.release_after();
        shared.with_pending(|cmd, _| shared.transition(cmd, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL));
        let c = &sc.config;
        let desc = t::TextureDescriptor {
            label: Some("surface"),
            size: t::Extent3d {
                width: c.width,
                height: c.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: t::TextureDimension::D2,
            format: c.format,
            usage: c.usage,
            view_formats: &c.view_formats,
        };
        let texture = Texture::wrap(image, &desc, shared.clone(), Some(Box::new(sc.clone())));
        let st = SurfaceTexture { texture, swapchain: sc, index };
        if suboptimal {
            t::CurrentSurfaceTexture::Suboptimal(st)
        } else {
            t::CurrentSurfaceTexture::Success(st)
        }
    }

    /// Have `maker` make the swapchain from the next `configure` on (None: the driver's own
    /// again). The current swapchain is let go of: configure the surface again before the next
    /// frame.
    pub fn set_swapchain_maker(&self, maker: Option<SwapchainMaker>) {
        *self.maker.lock() = maker;
        if let Some(state) = self.state.lock().as_mut() {
            state.swapchain = None;
        }
    }

    /// The configuration of the current swapchain.
    pub fn config(&self) -> Option<t::SurfaceConfiguration> {
        self.state.lock().as_ref().and_then(|s| s.swapchain.as_ref()).map(|s| s.config.clone())
    }

    /// The current `VkSwapchainKHR`, for interop.
    pub fn raw_swapchain(&self) -> Option<vk::SwapchainKHR> {
        self.state.lock().as_ref().and_then(|s| s.swapchain.as_ref()).map(|s| s.raw)
    }

    /// The `VkSurfaceKHR`, for interop.
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw.raw
    }
}

/// A frame of a surface. Draw into `texture`, submit, then `present`.
pub struct SurfaceTexture {
    pub texture: Texture,
    swapchain: Arc<Swapchain>,
    index: u32,
}

impl SurfaceTexture {
    /// Show the frame once the work submitted so far is done.
    pub fn present(self) {
        let sc = &self.swapchain;
        let shared = &sc.shared;
        let image = sc.images[self.index as usize];
        let (pool, cmd) = shared.begin_cmd();
        shared.transition(cmd, image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::GENERAL, vk::ImageLayout::PRESENT_SRC_KHR);
        shared.end_cmd(cmd);
        let present_sem = sc.present[self.index as usize];
        let cb = CommandBuffer {
            raws: vec![cmd],
            rec: Recording {
                cmd_pools: vec![pool],
                keep: vec![Box::new(self.texture.clone())],
                ..Default::default()
            },
            shared: shared.clone(),
        };
        let submitted = shared.submit(vec![cb], Some(present_sem));
        let waits = [present_sem];
        let swapchains = [sc.raw];
        let indices = [self.index];
        let info = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices);
        let q = shared.queue.lock();
        // SAFETY: the image was acquired and is in PRESENT_SRC once the semaphore signals.
        match unsafe { (sc.fns.present)(*q, &info) } {
            vk::Result::SUCCESS => {}
            vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => sc.outdated.store(true, Ordering::Release),
            e => log::error!("mX: vkQueuePresentKHR: {e}"),
        }
        drop(q);
        sc.presented.lock().push_back(submitted.0);
    }
}
