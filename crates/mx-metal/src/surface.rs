//! Window surfaces: a `CAMetalLayer` in the window's view.

use std::marker::PhantomData;
use std::sync::Arc;

use mx_types as t;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::device::Shared;
use crate::{Adapter, Device, Error, Instance, Mt, Po, Result, Texture, conv};

trait Window: HasWindowHandle + HasDisplayHandle {}
impl<W: HasWindowHandle + HasDisplayHandle> Window for W {}

struct Config {
    shared: Arc<Shared>,
    config: t::SurfaceConfiguration,
}

/// Where a window's frames go.
pub struct Surface<'w> {
    layer: Mt<Retained<CAMetalLayer>>,
    /// The view the layer is in, for the window's scale factor and occlusion.
    #[cfg(target_os = "macos")]
    view: Mt<Retained<objc2_app_kit::NSView>>,
    config: Mutex<Option<Config>>,
    _window: Option<Box<dyn Window + Send + Sync + 'w>>,
    _lifetime: PhantomData<&'w ()>,
}

impl Instance {
    /// A surface for a window (a winit `Window`, `Arc<Window>` or `&Window`). Call on the main
    /// thread: the layer is put into the window's view.
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>> {
        let raw_window = window.window_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        let raw_display = window.display_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        // SAFETY: the handles are valid while `window` lives, and the surface keeps it.
        let mut s = unsafe { self.create_surface_unsafe(raw_display, raw_window)? };
        s._window = Some(Box::new(window));
        Ok(Surface {
            layer: s.layer.clone(),
            #[cfg(target_os = "macos")]
            view: s.view.clone(),
            config: Mutex::new(None),
            _window: s._window.take(),
            _lifetime: PhantomData,
        })
    }

    /// A surface from raw handles.
    ///
    /// # Safety
    /// The window must outlive the surface, and this must run on the main thread.
    pub unsafe fn create_surface_unsafe(&self, _display: RawDisplayHandle, window: RawWindowHandle) -> Result<Surface<'static>> {
        match window {
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(h) => {
                // SAFETY: the handle points to a live NSView (caller's contract).
                let view: Retained<objc2_app_kit::NSView> = unsafe { Retained::retain(h.ns_view.as_ptr().cast()) }.ok_or_else(|| Error::Surface("null NSView".into()))?;
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
                Ok(Surface {
                    layer: Mt(layer),
                    view: Mt(view),
                    config: Mutex::new(None),
                    _window: None,
                    _lifetime: PhantomData,
                })
            }
            #[cfg(not(target_os = "macos"))]
            RawWindowHandle::UiKit(h) => {
                // SAFETY: the handle points to a live UIView (caller's contract).
                let view: Retained<objc2_ui_kit::UIView> = unsafe { Retained::retain(h.ui_view.as_ptr().cast()) }.ok_or_else(|| Error::Surface("null UIView".into()))?;
                let main = view.layer();
                let layer = CAMetalLayer::new();
                layer.setFrame(main.bounds());
                layer.setContentsScale(main.contentsScale());
                main.addSublayer(&layer);
                Ok(Surface {
                    layer: Mt(layer),
                    config: Mutex::new(None),
                    _window: None,
                    _lifetime: PhantomData,
                })
            }
            other => Err(Error::Surface(format!("not a window Metal can draw into: {other:?}"))),
        }
    }
}

impl Surface<'_> {
    pub fn get_capabilities(&self, _adapter: &Adapter) -> t::SurfaceCapabilities {
        t::SurfaceCapabilities {
            formats: vec![
                t::TextureFormat::Bgra8UnormSrgb,
                t::TextureFormat::Bgra8Unorm,
                t::TextureFormat::Rgba16Float,
                t::TextureFormat::Rgb10a2Unorm,
            ],
            present_modes: vec![t::PresentMode::Fifo, t::PresentMode::Immediate, t::PresentMode::AutoVsync, t::PresentMode::AutoNoVsync],
            alpha_modes: vec![t::CompositeAlphaMode::Opaque, t::CompositeAlphaMode::PostMultiplied],
            usages: t::TextureUsages::RENDER_ATTACHMENT | t::TextureUsages::COPY_SRC | t::TextureUsages::COPY_DST | t::TextureUsages::TEXTURE_BINDING,
        }
    }

    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration) {
        let l = &self.layer;
        l.setDevice(Some(&device.shared.device));
        l.setPixelFormat(conv::pixel_format(config.format));
        l.setFramebufferOnly(config.usage == t::TextureUsages::RENDER_ATTACHMENT);
        l.setDrawableSize(CGSize {
            width: config.width.max(1) as f64,
            height: config.height.max(1) as f64,
        });
        // (latency 1 gives 2 drawables, latency 2 gives 3; Metal takes 2 or 3)
        l.setMaximumDrawableCount((config.desired_maximum_frame_latency + 1).clamp(2, 3) as usize);
        #[cfg(target_os = "macos")]
        l.setDisplaySyncEnabled(!matches!(config.present_mode, t::PresentMode::Immediate | t::PresentMode::AutoNoVsync | t::PresentMode::Mailbox));
        l.setOpaque(matches!(config.alpha_mode, t::CompositeAlphaMode::Opaque | t::CompositeAlphaMode::Auto));
        #[cfg(target_os = "macos")]
        if let Some(w) = self.view.window() {
            l.setContentsScale(w.backingScaleFactor());
        }
        *self.config.lock() = Some(Config {
            shared: device.shared.clone(),
            config: config.clone(),
        });
    }

    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture> {
        let guard = self.config.lock();
        let Some(cfg) = guard.as_ref() else {
            panic!("get_current_texture on a surface that was never configured");
        };
        #[cfg(target_os = "macos")]
        if let Some(w) = self.view.window()
            && !w.occlusionState().contains(objc2_app_kit::NSWindowOcclusionState::Visible)
        {
            return t::CurrentSurfaceTexture::Occluded;
        }
        let Some(drawable) = self.layer.nextDrawable() else {
            return t::CurrentSurfaceTexture::Timeout;
        };
        let raw = drawable.texture();
        let (w, h) = (raw.width() as u32, raw.height() as u32);
        let c = &cfg.config;
        let desc = t::TextureDescriptor {
            label: Some("surface"),
            size: t::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: t::TextureDimension::D2,
            format: c.format,
            usage: c.usage,
            view_formats: &c.view_formats,
        };
        let texture = Texture::from_raw_parts(Mt(raw), &desc, cfg.shared.clone());
        let st = SurfaceTexture {
            texture,
            drawable: Mt(drawable),
            shared: cfg.shared.clone(),
        };
        if (w, h) != (c.width.max(1), c.height.max(1)) {
            t::CurrentSurfaceTexture::Suboptimal(st)
        } else {
            t::CurrentSurfaceTexture::Success(st)
        }
    }

    /// The `CAMetalLayer`, for interop.
    pub fn raw(&self) -> &CAMetalLayer {
        &self.layer
    }
}

/// A frame of a surface. Draw into `texture`, submit, then `present`.
pub struct SurfaceTexture {
    pub texture: Texture,
    drawable: Mt<Po<dyn CAMetalDrawable>>,
    shared: Arc<Shared>,
}

impl SurfaceTexture {
    /// Show the frame once the work submitted so far is done.
    pub fn present(self) {
        let cmd = self.shared.new_command_buffer();
        cmd.setLabel(Some(&crate::ns("mX present")));
        cmd.presentDrawable(ProtocolObject::<dyn MTLDrawable>::from_ref(&*self.drawable.0));
        cmd.commit();
    }
}
