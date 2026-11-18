//! Windows: an EGL window surface. Frames are drawn into a texture of mX's and copied to the
//! window's default framebuffer when presented (flipped: mX draws with WebGPU's y down, as
//! wgpu-hal's GL backend does), then swapped.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use glow::HasContext;
use khronos_egl as egl;
use mx_types as t;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::{Adapter, Device, Error, Instance, Result, Texture};

trait Window {}
impl<T> Window for T {}

struct State {
    device: Device,
    surface: egl::Surface,
    config: t::SurfaceConfiguration,
}

/// Where a window's frames go.
pub struct Surface<'w> {
    window: *mut c_void,
    state: Mutex<Option<State>>,
    current: Arc<Mutex<Option<Texture>>>,
    _window: Option<Box<dyn Window + Send + Sync + 'w>>,
    _lifetime: PhantomData<&'w ()>,
}

// SAFETY: the native window handle is only passed to EGL, under the context's lock.
unsafe impl Send for Surface<'_> {}
unsafe impl Sync for Surface<'_> {}

impl Instance {
    /// A surface for a window (a winit `Window`, `Arc<Window>` or `&Window`).
    pub fn create_surface<'w>(&self, window: impl HasWindowHandle + HasDisplayHandle + Send + Sync + 'w) -> Result<Surface<'w>> {
        let raw_window = window.window_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        let raw_display = window.display_handle().map_err(|e| Error::Surface(e.to_string()))?.as_raw();
        // SAFETY: the handles are valid while `window` lives, and the surface keeps it.
        let s = unsafe { self.create_surface_unsafe(raw_display, raw_window)? };
        // SAFETY: only the lifetime changes; the surface owns the window now.
        let mut s = unsafe { std::mem::transmute::<Surface<'static>, Surface<'w>>(s) };
        s._window = Some(Box::new(window));
        Ok(s)
    }

    /// # Safety
    /// The handles are valid for as long as the surface is used.
    pub unsafe fn create_surface_unsafe(&self, _display: RawDisplayHandle, window: RawWindowHandle) -> Result<Surface<'static>> {
        let native: *mut c_void = match window {
            RawWindowHandle::Xlib(h) => h.window as *mut c_void,
            RawWindowHandle::Xcb(h) => h.window.get() as usize as *mut c_void,
            RawWindowHandle::Win32(h) => h.hwnd.get() as *mut c_void,
            RawWindowHandle::AndroidNdk(h) => h.a_native_window.as_ptr(),
            other => return Err(Error::Surface(format!("OpenGL windows of {other:?} are not supported yet (X11, Windows and Android are)"))),
        };
        Ok(Surface {
            window: native,
            state: Mutex::new(None),
            current: Arc::new(Mutex::new(None)),
            _window: None,
            _lifetime: PhantomData,
        })
    }
}

impl Surface<'_> {
    pub fn get_capabilities(&self, _adapter: &Adapter) -> t::SurfaceCapabilities {
        t::SurfaceCapabilities {
            formats: vec![t::TextureFormat::Rgba8UnormSrgb, t::TextureFormat::Rgba8Unorm],
            present_modes: vec![t::PresentMode::Fifo, t::PresentMode::Immediate],
            alpha_modes: vec![t::CompositeAlphaMode::Opaque],
            usages: t::TextureUsages::RENDER_ATTACHMENT | t::TextureUsages::COPY_SRC | t::TextureUsages::COPY_DST | t::TextureUsages::TEXTURE_BINDING,
        }
    }

    pub fn configure(&self, device: &Device, config: &t::SurfaceConfiguration) {
        let ctx = &device.shared.ctx;
        let mut state = self.state.lock();
        let surface = match state.take() {
            Some(s) if Arc::ptr_eq(&s.device.shared, &device.shared) => s.surface,
            Some(s) => {
                let _ = ctx.lib.egl.destroy_surface(ctx.display, s.surface);
                // SAFETY: the window handle is valid (the caller's contract).
                unsafe { ctx.lib.egl.create_window_surface(ctx.display, ctx.config, self.window as _, None) }.expect("eglCreateWindowSurface")
            }
            // SAFETY: as above.
            None => unsafe { ctx.lib.egl.create_window_surface(ctx.display, ctx.config, self.window as _, None) }.expect("eglCreateWindowSurface"),
        };
        {
            let mut gl = ctx.lock();
            if gl.set_surface(Some(surface)).is_ok() {
                let interval = if matches!(config.present_mode, t::PresentMode::Immediate | t::PresentMode::AutoNoVsync | t::PresentMode::Mailbox) {
                    0
                } else {
                    1
                };
                let _ = ctx.lib.egl.swap_interval(ctx.display, interval);
            }
            let _ = gl.set_surface(None);
        }
        *state = Some(State {
            device: device.clone(),
            surface,
            config: config.clone(),
        });
        *self.current.lock() = None;
    }

    pub fn get_current_texture(&self) -> t::CurrentSurfaceTexture<SurfaceTexture> {
        let state = self.state.lock();
        let Some(s) = state.as_ref() else {
            panic!("get_current_texture on a surface that was never configured");
        };
        if s.config.width == 0 || s.config.height == 0 {
            return t::CurrentSurfaceTexture::Occluded;
        }
        let mut current = self.current.lock();
        let tex = match current.as_ref() {
            Some(t) if t.width() == s.config.width && t.height() == s.config.height => t.clone(),
            _ => {
                let t = s
                    .device
                    .create_texture(&t::TextureDescriptor {
                        label: Some("mX surface"),
                        size: t::Extent3d {
                            width: s.config.width,
                            height: s.config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: t::TextureDimension::D2,
                        format: s.config.format,
                        usage: s.config.usage | t::TextureUsages::RENDER_ATTACHMENT | t::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    })
                    .expect("the surface's texture");
                *current = Some(t.clone());
                t
            }
        };
        t::CurrentSurfaceTexture::Success(SurfaceTexture {
            texture: tex,
            device: s.device.clone(),
            surface: s.surface,
        })
    }
}

impl Drop for Surface<'_> {
    fn drop(&mut self) {
        if let Some(s) = self.state.lock().take() {
            let ctx = &s.device.shared.ctx;
            let _ = ctx.lib.egl.destroy_surface(ctx.display, s.surface);
        }
    }
}

pub struct SurfaceTexture {
    pub texture: Texture,
    device: Device,
    surface: egl::Surface,
}

// SAFETY: the EGL surface handle is only used under the device's context lock.
unsafe impl Send for SurfaceTexture {}
unsafe impl Sync for SurfaceTexture {}

impl SurfaceTexture {
    /// Copy the frame to the window (after all work submitted so far) and show it.
    pub fn present(self) {
        let shared = &self.device.shared;
        let mut gl = shared.ctx.lock();
        if gl.set_surface(Some(self.surface)).is_err() {
            return;
        }
        let (w, h) = (self.texture.width() as i32, self.texture.height() as i32);
        // SAFETY: GL calls on the current context; the texture is alive.
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(shared.fbo_read));
            gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.texture.raw()), 0);
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.disable(glow::SCISSOR_TEST);
            gl.color_mask(true, true, true, true);
            // (flipped: the frame is stored with y down)
            gl.blit_framebuffer(0, h, w, 0, 0, 0, w, h, glow::COLOR_BUFFER_BIT, glow::NEAREST);
            gl.framebuffer_texture_2d(glow::READ_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, None, 0);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        }
        let ctx = gl.ctx();
        if let Err(e) = ctx.lib.egl.swap_buffers(ctx.display, self.surface) {
            log::warn!("mX: eglSwapBuffers: {e}");
        }
        let _ = gl.set_surface(None);
    }
}
