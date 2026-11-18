//! mX's upscalers and frame generators: the vendors' own, driven on mX's devices, queues,
//! textures and surfaces.
//!
//! | | Upscaling | Frame generation |
//! |---|---|---|
//! | Apple MetalFX (macOS 13+, Metal) | spatial, temporal | interpolation (macOS 26+) |
//! | NVIDIA DLSS via Streamline (Windows, D3D12 / Vulkan) | Super Resolution, Ray Reconstruction | DLSS-G (D3D12 / Vulkan) |
//! | AMD FSR 3.1 via the FidelityFX API (Windows, D3D12 / Vulkan) | upscaling | frame generation (D3D12 / Vulkan) |
//! | Intel XeSS (Windows, D3D12 / Vulkan) | Super Resolution | XeSS-FG (D3D12) |
//!
//! No vendor's code is linked: MetalFX is opened by name (it comes with macOS 13), the Windows
//! libraries are loaded from [`Options::library_dir`]. Without them [`Upscaler::new`] says why,
//! and changes nothing.
//!
//! Every upscaler takes the same [`Inputs`] and [`Constants`] and records its work on mX's queue
//! right after the work submitted so far (a queue runs its submissions in order). Textures stay
//! in the state mX leaves them in between its commands (Direct3D 12: their home state,
//! `mx_dx12::Texture::rest_state`; Vulkan: `GENERAL`), and the vendors are told so.
//!
//! The upscalers that work on one frame only (Catmull-Rom, FSR 1) are shaders, not vendors':
//! they belong with the renderer.

use std::path::PathBuf;

pub use anyhow::{Error, Result};

#[cfg(windows)]
mod d3d12;
#[cfg(windows)]
mod ffx_fg;
#[cfg(windows)]
mod fidelityfx;
#[cfg(windows)]
mod fidelityfx_frames;
#[cfg(target_os = "macos")]
mod metalfx;
#[cfg(target_os = "macos")]
mod metalfx_frames;
/// Apple's frame interpolator itself, below [`FrameGenerator`]: hand it frames
/// ([`MetalFxInterpolator::prepare`]) and make the pictures between them
/// ([`MetalFxInterpolator::generate`]) where the program presents itself.
#[cfg(target_os = "macos")]
pub use metalfx_frames::{FrameGeneration as MetalFxInterpolator, window_picture};
#[cfg(target_os = "macos")]
mod metalfx_spatial;
#[cfg(windows)]
mod sl_ext;
#[cfg(windows)]
mod streamline;
#[cfg(windows)]
mod vulkan;
#[cfg(windows)]
mod xefg;
#[cfg(windows)]
mod xefg_ffi;
#[cfg(windows)]
mod xess;
#[cfg(windows)]
mod xess_ffi;

mod frames;
pub use frames::*;

/// Which vendor's upscaler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    /// Apple MetalFX's spatial scaler: one frame, no motion vectors.
    MetalFxSpatial,
    /// Apple MetalFX's temporal scaler.
    MetalFxTemporal,
    /// NVIDIA DLSS Super Resolution (DLAA at full size).
    Dlss,
    /// NVIDIA DLSS Ray Reconstruction: DLSS with its own denoiser, for ray-traced frames; it
    /// takes [`Inputs::guides`].
    DlssRayReconstruction,
    /// AMD FSR 3.1 upscaling.
    Fsr3,
    /// Intel XeSS Super Resolution.
    Xess,
}

impl Method {
    pub fn parse(s: &str) -> Option<Method> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "metalfx_spatial" => Method::MetalFxSpatial,
            "metalfx" | "metalfx_temporal" => Method::MetalFxTemporal,
            "dlss" => Method::Dlss,
            "dlss_rr" | "dlss_ray_reconstruction" => Method::DlssRayReconstruction,
            "fsr3" => Method::Fsr3,
            "xess" => Method::Xess,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Method::MetalFxSpatial => "metalfx_spatial",
            Method::MetalFxTemporal => "metalfx",
            Method::Dlss => "dlss",
            Method::DlssRayReconstruction => "dlss_rr",
            Method::Fsr3 => "fsr3",
            Method::Xess => "xess",
        }
    }

    /// Whether it carries pictures over from frame to frame (motion vectors, depth, jitter).
    pub fn temporal(self) -> bool {
        !matches!(self, Method::MetalFxSpatial)
    }

    /// The graphics APIs it runs on.
    pub fn backends(self) -> &'static [mx::Backend] {
        match self {
            Method::MetalFxSpatial | Method::MetalFxTemporal => &[mx::Backend::Metal],
            Method::Dlss | Method::DlssRayReconstruction | Method::Fsr3 | Method::Xess => &[mx::Backend::Dx12, mx::Backend::Vulkan],
        }
    }

    /// The libraries it loads on `backend`, from [`Options::library_dir`] (none: it comes with
    /// the system).
    pub fn files(self, backend: mx::Backend) -> &'static [&'static str] {
        match (self, backend) {
            (Method::Dlss, _) => &["sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "nvngx_dlss.dll"],
            (Method::DlssRayReconstruction, _) => &["sl.interposer.dll", "sl.common.dll", "sl.dlss_d.dll", "nvngx_dlssd.dll"],
            (Method::Fsr3, mx::Backend::Vulkan) => &["amd_fidelityfx_loader_vk.dll", "amd_fidelityfx_upscaler_vk.dll"],
            (Method::Fsr3, _) => &["amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll"],
            (Method::Xess, _) => &["libxess.dll"],
            (Method::MetalFxSpatial | Method::MetalFxTemporal, _) => &[],
        }
    }

    /// Of [`Method::files`], those missing from `dir`.
    pub fn missing(self, dir: &std::path::Path, backend: mx::Backend) -> Vec<&'static str> {
        self.files(backend).iter().copied().filter(|f| !dir.join(f).is_file()).collect()
    }
}

/// How much smaller than the output the picture is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Quality {
    /// Full size (DLAA, FSR 3 native AA, XeSS native AA).
    Native,
    /// 1.3x a side (XeSS's "ultra quality plus"; the others take it as quality).
    UltraQuality,
    #[default]
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl Quality {
    pub fn parse(s: &str) -> Quality {
        match s.trim().to_ascii_lowercase().replace([' ', '-'], "_").as_str() {
            "native" | "dlaa" | "native_aa" => Quality::Native,
            "ultra_quality" => Quality::UltraQuality,
            "balanced" => Quality::Balanced,
            "performance" => Quality::Performance,
            "ultra_performance" => Quality::UltraPerformance,
            _ => Quality::Quality,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Quality::Native => "native",
            Quality::UltraQuality => "ultra_quality",
            Quality::Quality => "quality",
            Quality::Balanced => "balanced",
            Quality::Performance => "performance",
            Quality::UltraPerformance => "ultra_performance",
        }
    }

    /// A side of the drawn picture over the output's.
    pub fn scale(self) -> f32 {
        match self {
            Quality::Native => 1.0,
            Quality::UltraQuality => 0.77,
            Quality::Quality => 0.667,
            Quality::Balanced => 0.58,
            Quality::Performance => 0.5,
            Quality::UltraPerformance => 0.333,
        }
    }
}

/// The size to draw at for `quality` and an output of `(w, h)`, by [`Quality::scale`].
pub fn render_size(quality: Quality, (w, h): (u32, u32)) -> (u32, u32) {
    let s = quality.scale();
    (((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1))
}

/// The size MetalFX draws at, and whether it is larger than the quality asked for: Apple's
/// scaler goes up to `max_scale` times a side (`supportedInputContentMaxScaleForDevice`), which
/// is less than ultra performance's three on most Macs.
pub fn metalfx_render_size(quality: Quality, out: (u32, u32), max_scale: f32) -> ((u32, u32), bool) {
    let asked = render_size(quality, out);
    let least = |side: u32| ((side as f32 / max_scale.max(1.0)).ceil() as u32).clamp(1, side.max(1));
    let size = (asked.0.max(least(out.0)), asked.1.max(least(out.1)));
    (size, size != asked)
}

/// Where the vendors' libraries are, and how they are run.
#[derive(Debug, Clone)]
pub struct Options {
    /// The folder of the Windows libraries (`sl.*.dll`, `amd_fidelityfx_*.dll`, `libxess*.dll`).
    /// Default: the program's own.
    pub library_dir: PathBuf,
    /// Which pictures frame generation shows (to look at one kind alone; FSR 3 and DLSS-G tint
    /// the made ones instead).
    pub show: Show,
    /// DLSS: the preset (Streamline's numbers, 0: NVIDIA's default for the quality).
    pub dlss_preset: u32,
    /// Ask the vendors to log what they do (to `log`).
    pub verbose: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            library_dir: std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_else(|| ".".into()),
            show: Show::Both,
            dlss_preset: 0,
            verbose: false,
        }
    }
}

/// The textures of one frame. They are mX's, in mX's resting states.
pub struct Inputs<'a> {
    /// The drawn picture, `render_size` (or larger, its corner used). For the temporal
    /// upscalers, jittered by [`Constants::jitter`].
    pub color: &'a mx::Texture,
    /// Its depth, reversed Z (1 near, 0 far). Temporal upscalers only.
    pub depth: Option<&'a mx::Texture>,
    /// Where each pixel was a frame ago, in pixels of the drawn picture, y down
    /// (`previous = current + motion`). Temporal upscalers only.
    pub motion: Option<&'a mx::Texture>,
    /// How little of the history each pixel should keep, 0..1 (glass, particles): FSR 3's
    /// reactive mask, MetalFX's reactive mask, DLSS's bias, XeSS's responsive pixel mask.
    pub mask: Option<&'a mx::Texture>,
    /// Where the result goes, `output_size`; it needs `STORAGE_BINDING` (and
    /// `RENDER_ATTACHMENT` for MetalFX).
    pub output: &'a mx::Texture,
    pub render_size: (u32, u32),
    pub output_size: (u32, u32),
    /// Ray Reconstruction's guides.
    pub guides: Option<Guides<'a>>,
}

/// What DLSS Ray Reconstruction denoises with, at the drawn size.
pub struct Guides<'a> {
    pub diffuse_albedo: &'a mx::Texture,
    pub specular_albedo: &'a mx::Texture,
    /// World-space normals in xyz, roughness in w.
    pub normal_roughness: &'a mx::Texture,
    /// The distance to what a specular ray hit (optional).
    pub specular_hit_distance: Option<&'a mx::Texture>,
}

/// This frame's camera as the upscalers want it. The matrices are unjittered and stored column
/// by column (as glam's `to_cols_array_2d`), which is how Streamline (row vectors) reads its rows.
#[derive(Debug, Clone, Copy)]
pub struct Constants {
    pub view_to_clip: [[f32; 4]; 4],
    pub clip_to_view: [[f32; 4]; 4],
    pub clip_to_prev_clip: [[f32; 4]; 4],
    pub prev_clip_to_clip: [[f32; 4]; 4],
    pub world_to_view: [[f32; 4]; 4],
    pub view_to_world: [[f32; 4]; 4],
    /// The jitter the picture was drawn with, in pixels, y down.
    pub jitter: [f32; 2],
    pub camera_pos: [f32; 3],
    pub camera_up: [f32; 3],
    pub camera_right: [f32; 3],
    pub camera_fwd: [f32; 3],
    pub near: f32,
    pub far: f32,
    /// Vertical, in radians.
    pub fov: f32,
    pub aspect: f32,
    /// The camera jumped: forget the history.
    pub reset: bool,
    /// Since the last frame.
    pub frame_ms: f32,
    /// The picture's exposure (1: the picture is the finished, tone-mapped one).
    pub pre_exposure: f32,
    /// The picture is HDR (linear, not tone mapped).
    pub hdr: bool,
}

impl Default for Constants {
    fn default() -> Constants {
        const I: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        Constants {
            view_to_clip: I,
            clip_to_view: I,
            clip_to_prev_clip: I,
            prev_clip_to_clip: I,
            world_to_view: I,
            view_to_world: I,
            jitter: [0.0; 2],
            camera_pos: [0.0; 3],
            camera_up: [0.0, 1.0, 0.0],
            camera_right: [1.0, 0.0, 0.0],
            camera_fwd: [0.0, 0.0, 1.0],
            near: 0.1,
            far: 1000.0,
            fov: 1.0,
            aspect: 1.0,
            reset: false,
            frame_ms: 16.6,
            pre_exposure: 1.0,
            hdr: false,
        }
    }
}

enum Inner {
    #[cfg(target_os = "macos")]
    MetalFxTemporal(metalfx::Upscaler),
    #[cfg(target_os = "macos")]
    MetalFxSpatial(metalfx_spatial::Upscaler),
    #[cfg(windows)]
    Dlss(streamline::Runtime),
    #[cfg(windows)]
    Fsr3(fidelityfx::Upscaler),
    #[cfg(windows)]
    Xess(xess::Upscaler),
}

/// One vendor's upscaler on one mX device.
pub struct Upscaler {
    method: Method,
    inner: Inner,
}

// SAFETY: the vendors' objects are used through `&mut self` alone (the Metal objects are
// thread-safe; the Windows contexts are only called from the thread that has the upscaler).
unsafe impl Send for Upscaler {}
unsafe impl Sync for Upscaler {}

impl Upscaler {
    /// Load `method`'s library and check that it runs on `device`.
    #[cfg_attr(not(any(target_os = "macos", windows)), allow(unreachable_code))] // (no vendor runs there)
    pub fn new(device: &mx::Device, method: Method, options: &Options) -> Result<Upscaler> {
        let _ = options;
        let inner = match method {
            #[cfg(target_os = "macos")]
            Method::MetalFxTemporal => Inner::MetalFxTemporal(metalfx::Upscaler::load(device)?),
            #[cfg(target_os = "macos")]
            Method::MetalFxSpatial => Inner::MetalFxSpatial(metalfx_spatial::Upscaler::load(device)?),
            #[cfg(windows)]
            Method::Dlss => Inner::Dlss(streamline::Runtime::load(device, options, false)?),
            #[cfg(windows)]
            Method::DlssRayReconstruction => Inner::Dlss(streamline::Runtime::load(device, options, true)?),
            #[cfg(windows)]
            Method::Fsr3 => Inner::Fsr3(fidelityfx::Upscaler::load(device, options)?),
            #[cfg(windows)]
            Method::Xess => Inner::Xess(xess::Upscaler::load(device, options)?),
            #[allow(unreachable_patterns)]
            other => {
                let _ = device;
                anyhow::bail!("{} does not run on this platform", other.as_str())
            }
        };
        Ok(Upscaler { method, inner })
    }

    pub fn method(&self) -> Method {
        self.method
    }

    /// The size to draw at for `quality` and an output of `output`: the vendor's own where it
    /// tells one.
    pub fn render_size(&mut self, quality: Quality, output: (u32, u32)) -> (u32, u32) {
        match &mut self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFxTemporal(u) => u.render_size(quality, output),
            #[cfg(target_os = "macos")]
            Inner::MetalFxSpatial(_) => render_size(quality, output),
            #[cfg(windows)]
            Inner::Dlss(u) => u.render_size(quality, output),
            #[cfg(windows)]
            Inner::Fsr3(_) => render_size(quality, output),
            #[cfg(windows)]
            Inner::Xess(u) => u.render_size(quality, output),
            #[allow(unreachable_patterns)]
            _ => {
                let _ = (quality, output);
                unreachable!()
            }
        }
    }

    /// Upscale `inputs.color` into `inputs.output`, on `queue` after the work submitted so far.
    pub fn upscale(&mut self, queue: &mx::Queue, quality: Quality, inputs: &Inputs, constants: &Constants) -> Result<()> {
        if self.method.temporal() && (inputs.depth.is_none() || inputs.motion.is_none()) {
            anyhow::bail!("{} needs the depth and the motion vectors", self.method.as_str());
        }
        if self.method == Method::DlssRayReconstruction && inputs.guides.is_none() {
            anyhow::bail!("DLSS Ray Reconstruction needs its guides (albedos, normals and roughness)");
        }
        match &mut self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFxTemporal(u) => u.upscale(queue, inputs, constants),
            #[cfg(target_os = "macos")]
            Inner::MetalFxSpatial(u) => u.upscale(queue, inputs),
            #[cfg(windows)]
            Inner::Dlss(u) => u.evaluate(queue, quality, inputs, constants),
            #[cfg(windows)]
            Inner::Fsr3(u) => u.upscale(queue, inputs, constants),
            #[cfg(windows)]
            Inner::Xess(u) => u.upscale(queue, quality, inputs, constants),
            #[allow(unreachable_patterns)]
            _ => {
                let _ = (queue, quality, constants);
                unreachable!()
            }
        }
    }

    /// Wait for the GPU to finish the last upscale (before textures it used are let go of).
    pub fn wait(&mut self) {
        match &mut self.inner {
            #[cfg(target_os = "macos")]
            Inner::MetalFxTemporal(u) => u.wait(),
            #[cfg(target_os = "macos")]
            Inner::MetalFxSpatial(u) => u.wait(),
            #[cfg(windows)]
            Inner::Dlss(u) => u.wait(),
            #[cfg(windows)]
            Inner::Fsr3(u) => u.wait(),
            #[cfg(windows)]
            Inner::Xess(u) => u.wait(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Apple's temporal scaler, for MetalFX frame interpolation's descriptor.
    #[cfg(target_os = "macos")]
    pub(crate) fn metalfx_scaler(&self) -> Option<metalfx::RawScaler> {
        match &self.inner {
            Inner::MetalFxTemporal(u) => u.scaler(),
            _ => None,
        }
    }

    /// The device extensions DLSS needs of a Vulkan device (`mx::DeviceDescriptor::vulkan_extensions`).
    /// They come from Streamline, which is loaded for it.
    pub fn dlss_vulkan_extensions(options: &Options) -> Result<Vec<&'static std::ffi::CStr>> {
        #[cfg(windows)]
        return streamline::vulkan_extensions(options);
        #[cfg(not(windows))]
        {
            let _ = options;
            anyhow::bail!("DLSS is Windows-only")
        }
    }

    /// Let go of Streamline loaded for [`Upscaler::dlss_vulkan_extensions`] when the device was not
    /// made after all.
    pub fn dlss_vulkan_abandon() {
        #[cfg(windows)]
        streamline::vulkan_abandon();
    }
}
