//! Frame generation: a picture between every two the program draws (or two or three, with DLSS
//! multi frame generation), made from the drawn ones, their depth and motion vectors.
//!
//! Two kinds:
//!
//! - **MetalFX** (macOS 26+): the program draws the window's picture into
//!   [`FrameGenerator::window_view`] instead of the window, and [`FrameGenerator::present`]
//!   shows the made picture, then the drawn one half a frame later, paced by a thread.
//! - **DLSS-G, FSR 3, XeSS-FG** (Windows): the vendor makes the window's swapchain
//!   (`mx_dx12::Surface::set_swapchain_maker`, `mx_vulkan::Surface::set_swapchain_maker`) and
//!   makes the pictures in its present. The program draws into the window as before, hands each
//!   frame's pictures over with [`FrameGenerator::prepare`] before the present, and configures
//!   the surface again after [`FrameGenerator::new`] and after [`FrameGenerator::stop`].

use crate::{Constants, Options, Upscaler};
use anyhow::Result;

/// Which vendor's frame generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameGenMethod {
    /// Apple MetalFX frame interpolation (macOS 26+, Metal).
    MetalFx,
    /// NVIDIA DLSS Frame Generation (Windows, D3D12 / Vulkan, GeForce RTX 40+): `generated`
    /// pictures between two drawn ones (1; 2 or 3 with multi frame generation, RTX 50+).
    Dlss { generated: u32 },
    /// AMD FSR 3.1 frame generation (Windows, D3D12 / Vulkan).
    Fsr3,
    /// Intel XeSS Frame Generation (Windows, D3D12).
    Xess,
}

impl FrameGenMethod {
    pub fn parse(s: &str) -> Option<FrameGenMethod> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "metalfx" => FrameGenMethod::MetalFx,
            "dlss" | "dlss2x" => FrameGenMethod::Dlss { generated: 1 },
            "dlss3x" => FrameGenMethod::Dlss { generated: 2 },
            "dlss4x" => FrameGenMethod::Dlss { generated: 3 },
            "fsr3" => FrameGenMethod::Fsr3,
            "xess" => FrameGenMethod::Xess,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FrameGenMethod::MetalFx => "metalfx",
            FrameGenMethod::Dlss { generated: 0 | 1 } => "dlss2x",
            FrameGenMethod::Dlss { generated: 2 } => "dlss3x",
            FrameGenMethod::Dlss { .. } => "dlss4x",
            FrameGenMethod::Fsr3 => "fsr3",
            FrameGenMethod::Xess => "xess",
        }
    }

    /// Whether the vendor makes the window's swapchain (all but MetalFX).
    pub fn owns_swapchain(self) -> bool {
        !matches!(self, FrameGenMethod::MetalFx)
    }

    /// The libraries it loads on `backend`, from [`Options::library_dir`].
    pub fn files(self, backend: mx::Backend) -> &'static [&'static str] {
        match (self, backend) {
            (FrameGenMethod::Dlss { .. }, _) => &["sl.interposer.dll", "sl.common.dll", "sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll", "nvngx_dlssg.dll"],
            (FrameGenMethod::Fsr3, mx::Backend::Vulkan) => &["amd_fidelityfx_loader_vk.dll", "amd_fidelityfx_framegeneration_vk.dll"],
            (FrameGenMethod::Fsr3, _) => &["amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_framegeneration_dx12.dll"],
            (FrameGenMethod::Xess, _) => &["libxess_fg.dll", "libxell.dll"],
            (FrameGenMethod::MetalFx, _) => &[],
        }
    }
}

/// Which pictures the window shows (to look at one kind alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Show {
    #[default]
    Both,
    Generated,
    Drawn,
}

/// One drawn frame, as frame generation takes it.
#[derive(Clone)]
pub struct FrameInputs {
    /// The window's picture before the interface (output size).
    pub hudless: mx::Texture,
    /// The last frame's (MetalFX; None: there is none of this size).
    pub last_hudless: Option<mx::Texture>,
    /// The upscaler's depth (reversed Z) and motion vectors, at `render_size`.
    pub depth: mx::Texture,
    pub motion: mx::Texture,
    /// The interface alone, colour and alpha, output size (DLSS-G, FSR 3, XeSS-FG; None: the
    /// vendor finds it from the difference of the finished and the hudless pictures).
    pub ui: Option<mx::Texture>,
    pub render_size: (u32, u32),
    pub output_size: (u32, u32),
    pub constants: Constants,
    #[cfg(target_os = "macos")]
    pub(crate) scaler: Option<crate::metalfx::RawScaler>,
}

impl FrameInputs {
    pub fn new(hudless: mx::Texture, depth: mx::Texture, motion: mx::Texture, render_size: (u32, u32), output_size: (u32, u32), constants: Constants) -> FrameInputs {
        FrameInputs {
            hudless,
            last_hudless: None,
            depth,
            motion,
            ui: None,
            render_size,
            output_size,
            constants,
            #[cfg(target_os = "macos")]
            scaler: None,
        }
    }

    /// The upscaler that made the frame: MetalFX's interpolator works with its temporal scaler.
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    pub fn with_upscaler(mut self, upscaler: Option<&Upscaler>) -> FrameInputs {
        #[cfg(target_os = "macos")]
        {
            self.scaler = upscaler.and_then(|u| u.metalfx_scaler());
        }
        #[cfg(not(target_os = "macos"))]
        let _ = upscaler;
        self
    }
}

enum Inner {
    #[cfg(target_os = "macos")]
    MetalFx(crate::metalfx_frames::GenerationWindow),
    #[cfg(windows)]
    Dlss(crate::streamline::Generator),
    #[cfg(windows)]
    Fsr3(crate::fidelityfx::Generator),
    #[cfg(windows)]
    Xess(crate::xefg::Generator),
}

/// One vendor's frame generation on a window.
pub struct FrameGenerator {
    method: FrameGenMethod,
    inner: Inner,
}

// SAFETY: what `&self` reaches is behind the generators' own locks (MetalFX's pacing, the
// Windows generators' state); Metal's objects are thread-safe.
unsafe impl Send for FrameGenerator {}
unsafe impl Sync for FrameGenerator {}

impl FrameGenerator {
    /// For `surface` (configured with `device`), whose pictures are `format` and `size`. For a
    /// vendor that makes the swapchain, configure the surface again before the next frame.
    #[cfg_attr(not(any(target_os = "macos", windows)), allow(unreachable_code))] // (no vendor runs there)
    pub fn new(device: &mx::Device, queue: &mx::Queue, surface: &mx::Surface<'_>, method: FrameGenMethod, format: mx::TextureFormat, size: (u32, u32), options: &Options) -> Result<FrameGenerator> {
        let _ = (device, queue, surface, format, size, options);
        let inner = match method {
            #[cfg(target_os = "macos")]
            FrameGenMethod::MetalFx => Inner::MetalFx(crate::metalfx_frames::GenerationWindow::new(device, queue, format, size, options.show)?),
            #[cfg(windows)]
            FrameGenMethod::Dlss { generated } => Inner::Dlss(crate::streamline::Generator::new(device, queue, surface, generated.clamp(1, 3), options)?),
            #[cfg(windows)]
            FrameGenMethod::Fsr3 => Inner::Fsr3(crate::fidelityfx::Generator::new(device, queue, surface, format, options)?),
            #[cfg(windows)]
            FrameGenMethod::Xess => Inner::Xess(crate::xefg::Generator::new(device, queue, surface, options)?),
            #[allow(unreachable_patterns)]
            other => anyhow::bail!("{} frame generation does not run on this platform", other.as_str()),
        };
        Ok(FrameGenerator { method, inner })
    }

    pub fn method(&self) -> FrameGenMethod {
        self.method
    }

    /// Where the program draws the window's picture (MetalFX); None: into the window, as without
    /// frame generation.
    pub fn window_view(&self) -> Option<mx::TextureView> {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => Some(w.view()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// The texture of [`FrameGenerator::window_view`].
    pub fn window_texture(&self) -> Option<mx::Texture> {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => Some(w.texture()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// The window changed its size.
    pub fn resize(&mut self, device: &mx::Device, queue: &mx::Queue, size: (u32, u32)) {
        match &mut self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => w.resize(device, queue, size),
            #[allow(unreachable_patterns)]
            _ => {
                let _ = (device, queue, size);
            }
        }
    }

    /// This frame's pictures, after they are drawn (submitted) and before the present.
    pub fn prepare(&self, queue: &mx::Queue, inputs: FrameInputs) -> Result<()> {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => {
                let _ = queue;
                w.handle().prepare(inputs);
                Ok(())
            }
            #[cfg(windows)]
            Inner::Dlss(g) => g.prepare(queue, &inputs),
            #[cfg(windows)]
            Inner::Fsr3(g) => g.prepare(queue, &inputs),
            #[cfg(windows)]
            Inner::Xess(g) => g.prepare(queue, &inputs),
            #[allow(unreachable_patterns)]
            _ => {
                let _ = (queue, inputs);
                Ok(())
            }
        }
    }

    /// A frame without pictures to go from (a menu, a loading screen): shown alone.
    pub fn skip(&self) {
        match &self.inner {
            #[cfg(windows)]
            Inner::Dlss(g) => g.skip(),
            #[cfg(windows)]
            Inner::Fsr3(g) => g.skip(),
            #[cfg(windows)]
            Inner::Xess(g) => g.skip(),
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    /// Show `frame`: for MetalFX, the made picture and then the drawn one (from
    /// [`FrameGenerator::window_view`]); for the others, `frame` itself, and the vendor's
    /// swapchain adds the made ones.
    pub fn present(&self, surface: &mx::Surface<'_>, device: &mx::Device, queue: &mx::Queue, frame: mx::SurfaceTexture) {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => w.present(surface, device, queue, frame),
            #[cfg(windows)]
            Inner::Dlss(g) => {
                g.presenting(true);
                frame.present();
                g.presenting(false);
            }
            #[cfg(windows)]
            Inner::Xess(g) => {
                g.presenting(true);
                frame.present();
                g.presenting(false);
            }
            #[allow(unreachable_patterns)]
            _ => {
                let _ = (surface, device, queue);
                frame.present()
            }
        }
    }

    /// False once it gave up (the window is then drawn without it).
    pub fn working(&self) -> bool {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => w.working(),
            #[cfg(windows)]
            Inner::Dlss(g) => g.working(),
            #[cfg(windows)]
            Inner::Fsr3(g) => g.working(),
            #[cfg(windows)]
            Inner::Xess(g) => g.working(),
            #[allow(unreachable_patterns)]
            _ => unreachable!("no frame generation on this platform"),
        }
    }

    /// The pictures made so far (where the vendor tells).
    pub fn generated(&self) -> u64 {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => w.handle().generated(),
            #[cfg(windows)]
            Inner::Dlss(g) => g.generated(),
            #[cfg(windows)]
            Inner::Fsr3(g) => g.generated(),
            #[cfg(windows)]
            Inner::Xess(g) => g.generated(),
            #[allow(unreachable_patterns)]
            _ => unreachable!("no frame generation on this platform"),
        }
    }

    /// The GPU done with the pictures handed over (before they are let go of).
    pub fn wait(&self) {
        match &self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(w) => w.handle().wait(),
            #[cfg(windows)]
            Inner::Dlss(g) => g.wait(),
            #[cfg(windows)]
            Inner::Fsr3(g) => g.wait(),
            #[cfg(windows)]
            Inner::Xess(g) => g.wait(),
            #[allow(unreachable_patterns)]
            _ => unreachable!("no frame generation on this platform"),
        }
    }

    /// Turn it off; the window's own swapchain comes back (configure the surface again).
    pub fn stop(self, surface: &mx::Surface<'_>) {
        self.wait();
        match self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFx(_) => {
                let _ = surface;
            }
            #[cfg(windows)]
            Inner::Dlss(g) => g.stop(surface),
            #[cfg(windows)]
            Inner::Fsr3(g) => g.stop(surface),
            #[cfg(windows)]
            Inner::Xess(g) => g.stop(surface),
            #[allow(unreachable_patterns)]
            _ => {
                let _ = surface;
            }
        }
    }
}

/// The constants' camera as frame generation wants it: seconds since the last frame.
#[cfg(target_os = "macos")]
pub(crate) fn seconds(c: &Constants) -> f32 {
    c.frame_ms / 1000.0
}
