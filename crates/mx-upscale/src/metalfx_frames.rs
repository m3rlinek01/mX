//! Frame generation with Apple's MetalFX frame interpolator (macOS 26 and newer, Apple silicon):
//! between every two frames the game draws, the window shows one MetalFX makes from them, from
//! the upscaler's depth and motion vectors.
//!
//! While it is on, the game draws the window's picture into a buffer of the window's size instead
//! of the window itself ([`GenerationWindow`], as `SrgbEncode` does on ANGLE). At the present,
//! MetalFX makes the picture between the last frame and this one, the window shows that first,
//! and this frame half a frame time later. The interface lies over both: MetalFX is handed the
//! picture before the HUD and the finished one ("composited UI" in Apple's words), and leaves
//! what differs where it is.
//!
//! As the scaler (`crate::metalfx`), the framework is opened by name, and the work goes into
//! command buffers of mX's own Metal queue.

use crate::frames::{FrameInputs as GenerationInputs, Show, seconds};
use crate::metalfx::{Texture, class, texture};
use anyhow::{Result, anyhow, bail};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{Message, msg_send};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLTexture, MTLTextureUsage};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The drawn frame follows the generated one after this share of a frame time (Apple's
/// present helper waits 31/64 of it: a little under half, so that a frame that comes early
/// is not held back).
const HALF: f32 = 31.0 / 64.0;
/// No frame is held back longer than this (s): below 10 fps there is nothing to pace.
const LONGEST_WAIT: f32 = 0.05;

/// How long the drawn frame waits behind the generated one, for frames this far apart (s).
pub(crate) fn half_interval(frame_seconds: f32) -> f32 {
    if frame_seconds.is_finite() { (frame_seconds * HALF).clamp(0.0, LONGEST_WAIT) } else { 0.0 }
}

/// What the interpolator was made for: a change of any of it makes a new one.
#[derive(PartialEq, Clone, Copy)]
struct Made {
    render: (u32, u32),
    output: (u32, u32),
    /// Of the picture, the depth, the motion vectors, the finished picture and the result.
    formats: [MTLPixelFormat; 5],
    /// The scaler it was told of (its address; 0: none).
    scaler: usize,
}

struct Shared {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// The frame the renderer upscaled last, until the window's present takes it.
    prepared: Option<GenerationInputs>,
    interpolator: Option<(Retained<AnyObject>, Made)>,
    last: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// A frame went by without a generated one: MetalFX's own history is out of date.
    gap: bool,
    generated: u64,
    stopped: bool,
}

// (used from the render thread only; the lock is for the two owners, renderer and window)
unsafe impl Send for Shared {}

/// Frame generation's handle: the renderer hands it each upscaled frame, the window's present
/// takes it from there.
#[derive(Clone)]
pub struct FrameGeneration(Arc<Mutex<Shared>>);

impl FrameGeneration {
    pub fn load(device: &mx::Device) -> Result<FrameGeneration> {
        let raw = device.as_metal().ok_or_else(|| anyhow!("the game is not running on Metal"))?.raw().retain();
        let class = class(c"MTLFXFrameInterpolatorDescriptor").ok_or_else(|| anyhow!("this macOS has no MetalFX frame interpolation (it comes with macOS 26 and newer)"))?;
        let supported: bool = unsafe { msg_send![class, supportsDevice: &*raw] };
        if !supported {
            bail!("this Mac's graphics chip cannot run MetalFX's frame interpolator");
        }
        Ok(FrameGeneration(Arc::new(Mutex::new(Shared {
            device: raw,
            prepared: None,
            interpolator: None,
            last: None,
            gap: true,
            generated: 0,
            stopped: false,
        }))))
    }

    fn shared(&self) -> std::sync::MutexGuard<'_, Shared> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// False once it gave up: the window then draws without it.
    pub fn working(&self) -> bool {
        !self.shared().stopped
    }

    /// The in-between frames made so far.
    pub fn generated(&self) -> u64 {
        self.shared().generated
    }

    /// An upscaled frame, for the window's present of it.
    pub fn prepare(&self, inputs: GenerationInputs) {
        self.shared().prepared = Some(inputs);
    }

    /// The frame handed over last, once.
    pub fn take(&self) -> Option<GenerationInputs> {
        self.shared().prepared.take()
    }

    /// MetalFX's GPU work done: before the textures it reads are let go of.
    pub fn wait(&self) {
        let mut shared = self.shared();
        shared.prepared = None;
        shared.gap = true;
        if let Some(buffer) = shared.last.take() {
            buffer.waitUntilCompleted();
        }
    }

    fn stop(&self, why: &anyhow::Error) {
        let mut shared = self.shared();
        if !shared.stopped {
            log::error!("frame generation stopped working ({why:#}); the window is drawn without it from now on");
        }
        shared.stopped = true;
        shared.prepared = None;
    }

    /// The picture between the last frame and this one into `output`: `finished` is this
    /// frame as the window shows it, with the interface. False: there is nothing to go from
    /// (the first frame, a cut), and no picture was made.
    pub fn generate(&self, queue: &mx::Queue, inputs: &GenerationInputs, finished: &mx::Texture, output: &mx::Texture) -> Result<bool> {
        let mut shared = self.shared();
        let c = &inputs.constants;
        let Some(last) = inputs.last_hudless.as_ref().filter(|_| !c.reset) else {
            shared.gap = true;
            return Ok(false);
        };
        autoreleasepool(|_| {
            let picture = texture(&inputs.hudless, "picture before the HUD")?;
            let last = texture(last, "last picture")?;
            let depth = texture(&inputs.depth, "depth")?;
            let motion = texture(&inputs.motion, "motion vectors")?;
            let finished = texture(finished, "finished picture")?;
            let output = texture(output, "result")?;
            let scaler = inputs.scaler.as_ref().map_or(0, |s| Retained::as_ptr(s) as usize);
            let made = Made {
                render: inputs.render_size,
                output: inputs.output_size,
                formats: [picture.pixelFormat(), depth.pixelFormat(), motion.pixelFormat(), finished.pixelFormat(), output.pixelFormat()],
                scaler,
            };
            if shared.interpolator.as_ref().is_some_and(|(_, m)| *m != made) {
                shared.interpolator = None;
            }
            if shared.interpolator.is_none() {
                let raw = interpolator(&shared.device, &made, inputs.scaler.as_deref(), [&picture, &depth, &motion, &finished, &output])?;
                shared.interpolator = Some((raw, made));
                shared.gap = true;
            }
            let buffer = queue
                .as_metal()
                .ok_or_else(|| anyhow!("no Metal queue under mX's"))?
                .raw()
                .commandBuffer()
                .ok_or_else(|| anyhow!("Metal gave no command buffer for MetalFX"))?;
            let none: Option<&ProtocolObject<dyn MTLTexture>> = None;
            let reset = shared.gap;
            unsafe {
                let i = &*shared.interpolator.as_ref().expect("made above").0;
                let _: () = msg_send![i, setColorTexture: &*picture];
                let _: () = msg_send![i, setPrevColorTexture: &*last];
                let _: () = msg_send![i, setDepthTexture: &*depth];
                let _: () = msg_send![i, setMotionTexture: &*motion];
                let _: () = msg_send![i, setUITexture: &*finished];
                let _: () = msg_send![i, setIsUITextureComposited: true];
                let _: () = msg_send![i, setOutputTexture: &*output];
                // (as the scaler takes them: where each pixel was a frame ago, in pixels of
                // the small picture with y down, and the jitter the picture was drawn with)
                let _: () = msg_send![i, setMotionVectorScaleX: 1.0f32];
                let _: () = msg_send![i, setMotionVectorScaleY: 1.0f32];
                let _: () = msg_send![i, setJitterOffsetX: c.jitter[0]];
                let _: () = msg_send![i, setJitterOffsetY: c.jitter[1]];
                let _: () = msg_send![i, setDepthReversed: true];
                let _: () = msg_send![i, setDeltaTime: seconds(c)];
                let _: () = msg_send![i, setFieldOfView: c.fov.to_degrees()];
                let _: () = msg_send![i, setNearPlane: c.near];
                let _: () = msg_send![i, setFarPlane: c.far];
                let _: () = msg_send![i, setAspectRatio: c.aspect];
                let _: () = msg_send![i, setShouldResetHistory: reset];
                let _: () = msg_send![i, encodeToCommandBuffer: &*buffer];
                // (the command buffer holds the textures from here)
                let _: () = msg_send![i, setColorTexture: none];
                let _: () = msg_send![i, setPrevColorTexture: none];
                let _: () = msg_send![i, setDepthTexture: none];
                let _: () = msg_send![i, setMotionTexture: none];
                let _: () = msg_send![i, setUITexture: none];
                let _: () = msg_send![i, setOutputTexture: none];
            }
            buffer.commit();
            shared.last = Some(buffer);
            shared.gap = false;
            shared.generated += 1;
            Ok(true)
        })
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        if let Some(buffer) = self.last.take() {
            buffer.waitUntilCompleted();
        }
    }
}

/// Apple's interpolator for these sizes and formats, its textures' usages checked.
fn interpolator(device: &ProtocolObject<dyn MTLDevice>, made: &Made, scaler: Option<&AnyObject>, textures: [&Texture; 5]) -> Result<Retained<AnyObject>> {
    let class = class(c"MTLFXFrameInterpolatorDescriptor").ok_or_else(|| anyhow!("MetalFX's frame interpolation is gone"))?;
    let desc: Retained<AnyObject> = unsafe { msg_send![class, new] };
    let raw: Option<Retained<AnyObject>> = unsafe {
        let _: () = msg_send![&*desc, setColorTextureFormat: made.formats[0]];
        let _: () = msg_send![&*desc, setDepthTextureFormat: made.formats[1]];
        let _: () = msg_send![&*desc, setMotionTextureFormat: made.formats[2]];
        let _: () = msg_send![&*desc, setUITextureFormat: made.formats[3]];
        let _: () = msg_send![&*desc, setOutputTextureFormat: made.formats[4]];
        let _: () = msg_send![&*desc, setInputWidth: made.render.0 as usize];
        let _: () = msg_send![&*desc, setInputHeight: made.render.1 as usize];
        let _: () = msg_send![&*desc, setOutputWidth: made.output.0 as usize];
        let _: () = msg_send![&*desc, setOutputHeight: made.output.1 as usize];
        // (Apple: "for higher combined performance, provide the upscaler object")
        if let Some(scaler) = scaler {
            let _: () = msg_send![&*desc, setScaler: scaler];
        }
        msg_send![&*desc, newFrameInterpolatorWithDevice: device]
    };
    let raw = raw.ok_or_else(|| {
        anyhow!(
            "MetalFX made no frame interpolator for {}x{} to {}x{} ({:?})",
            made.render.0,
            made.render.1,
            made.output.0,
            made.output.1,
            made.formats
        )
    })?;
    let needs: [(&str, MTLTextureUsage); 5] = unsafe {
        [
            ("picture", msg_send![&*raw, colorTextureUsage]),
            ("depth", msg_send![&*raw, depthTextureUsage]),
            ("motion vectors", msg_send![&*raw, motionTextureUsage]),
            ("finished picture", msg_send![&*raw, uiTextureUsage]),
            ("result", msg_send![&*raw, outputTextureUsage]),
        ]
    };
    for ((what, need), have) in needs.into_iter().zip(textures) {
        if have.usage().0 & need.0 != need.0 {
            bail!("MetalFX's frame interpolator cannot use the {what} texture (usage {:#x}, it needs {:#x})", have.usage().0, need.0);
        }
    }
    log::info!(
        "frame generation: Apple MetalFX for {}x{}, the scene drawn at {}x{}{}",
        made.output.0,
        made.output.1,
        made.render.0,
        made.render.1,
        if scaler.is_some() { "" } else { ", without the scaler's help" }
    );
    Ok(raw)
}

/// An Objective-C object handed to the pacing thread (Metal's command buffers and events may be
/// used from any thread).
struct Sent<T>(T);
unsafe impl<T> Send for Sent<T> {}

/// The drawn frame's timing, as Apple's present helper does it: a thread waits until the GPU is
/// done with the generated frame, then lets the drawn one go a little under half a frame time
/// later (the time between two generated frames, smoothed). The drawn frame's present waits for
/// that on a queue of its own, so that the next frame's work on mX's queue is not held up.
/// A present for the pacer to send: the command buffer and the event value it waits for.
type Paced = Sent<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, u64)>;

struct Pacer {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    event: Retained<AnyObject>,
    value: u64,
    send: std::sync::mpsc::Sender<Paced>,
}

impl Pacer {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Pacer> {
        let queue = device.newCommandQueue().ok_or_else(|| anyhow!("Metal gave no queue for frame generation's presents"))?;
        let event: Option<Retained<AnyObject>> = unsafe { msg_send![device, newSharedEvent] };
        let event = event.ok_or_else(|| anyhow!("Metal gave no event for frame generation's pacing"))?;
        let (send, receive) = std::sync::mpsc::channel::<Paced>();
        let signal = Sent(event.clone());
        std::thread::Builder::new()
            .name("frame generation pacing".into())
            .spawn(move || {
                let signal = signal;
                let (mut last, mut interval): (Option<Instant>, f32) = (None, 0.0);
                for Sent((done, value)) in receive {
                    done.waitUntilCompleted();
                    let now = Instant::now();
                    if let Some(last) = last {
                        let seconds = (now - last).as_secs_f32().min(0.25);
                        interval = if interval > 0.0 { interval * 0.8 + seconds * 0.2 } else { seconds };
                    }
                    last = Some(now);
                    std::thread::sleep(std::time::Duration::from_secs_f32(half_interval(interval)));
                    unsafe {
                        let _: () = msg_send![&*signal.0, setSignaledValue: value];
                    }
                }
            })
            .map_err(|e| anyhow!("no thread for frame generation's pacing: {e}"))?;
        Ok(Pacer { queue, event, value: 0, send })
    }
}

/// The window's side of frame generation: the buffer the game draws the window's picture into,
/// MetalFX's picture beside it, and the present of the two.
pub struct GenerationWindow {
    handle: FrameGeneration,
    /// The window's own format (the renderer's).
    format: mx::TextureFormat,
    size: (u32, u32),
    drawn: mx::Texture,
    drawn_view: mx::TextureView,
    drawn_group: mx::BindGroup,
    made: mx::Texture,
    made_group: mx::BindGroup,
    copy: mx::RenderPipeline,
    pacer: Mutex<Pacer>,
    show: Show,
}

const COPY: &str = "
@group(0) @binding(0) var picture: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(picture, vec2<i32>(at.xy), 0);
}
";

/// A picture of the window's size that MetalFX reads or writes as the window shows it: a plain
/// UNORM resource (the sRGB-encoded numbers, as the scaler's), drawn and read through its sRGB
/// view. Cleared here: mX fills a texture it has not drawn into with zeros before its first
/// read, which would wipe what MetalFX wrote into it meanwhile.
pub fn window_picture(device: &mx::Device, queue: &mx::Queue, label: &str, format: mx::TextureFormat, (width, height): (u32, u32)) -> (mx::Texture, mx::TextureView) {
    let texture = device
        .create_texture(&mx::TextureDescriptor {
            label: Some(label),
            size: mx::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: mx::TextureDimension::D2,
            format: format.remove_srgb_suffix(),
            usage: mx::TextureUsages::RENDER_ATTACHMENT | mx::TextureUsages::TEXTURE_BINDING | mx::TextureUsages::COPY_SRC | mx::TextureUsages::COPY_DST,
            view_formats: &[format],
        })
        .expect("mX: a texture");
    let view = texture
        .create_view(&mx::TextureViewDescriptor {
            format: Some(format),
            ..Default::default()
        })
        .expect("mX: a texture view");
    let mut encoder = device.create_command_encoder(Some(label));
    drop(encoder.begin_render_pass(&mx::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(mx::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: mx::Operations {
                load: mx::LoadOp::Clear(mx::Color::BLACK),
                store: mx::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    }));
    queue.submit(Some(encoder.finish()));
    (texture, view)
}

fn picture(device: &mx::Device, queue: &mx::Queue, label: &str, format: mx::TextureFormat, size: (u32, u32)) -> (mx::Texture, mx::TextureView, mx::BindGroup) {
    let (texture, view) = window_picture(device, queue, label, format, size);
    let group = device.create_bind_group(&mx::BindGroupDescriptor {
        label: Some(label),
        entries: &[mx::BindGroupEntry {
            binding: 0,
            resource: mx::BindingResource::TextureView(&view),
        }],
    });
    (texture, view, group)
}

impl GenerationWindow {
    /// For a window of this format and size, on the renderer's device.
    pub(crate) fn new(device: &mx::Device, queue: &mx::Queue, format: mx::TextureFormat, size: (u32, u32), show: Show) -> Result<GenerationWindow> {
        let handle = FrameGeneration::load(device)?;
        let module = device
            .create_shader_module(&mx::ShaderModuleDescriptor {
                label: Some("frame generation copy"),
                source: COPY,
            })
            .expect("mX: a shader");
        let copy = device
            .create_render_pipeline(&mx::RenderPipelineDescriptor {
                label: Some("frame generation copy"),
                vertex: mx::VertexState {
                    constants: &[],
                    module: &module,
                    entry_point: Some("vs"),
                    buffers: &[],
                },
                fragment: Some(mx::FragmentState {
                    constants: &[],
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
            })
            .expect("mX: a render pipeline");
        let (drawn, drawn_view, drawn_group) = picture(device, queue, "frame generation: the drawn frame", format, size);
        let (made, _, made_group) = picture(device, queue, "frame generation: the generated frame", format, size);
        let pacer = Pacer::new(&handle.shared().device)?;
        Ok(GenerationWindow {
            handle,
            format,
            size,
            drawn,
            drawn_view,
            drawn_group,
            made,
            made_group,
            copy,
            pacer: Mutex::new(pacer),
            show,
        })
    }

    pub(crate) fn handle(&self) -> FrameGeneration {
        self.handle.clone()
    }

    pub(crate) fn working(&self) -> bool {
        self.handle.working()
    }

    /// The window changed its size: the pictures are made anew, MetalFX's work on the old ones done.
    pub(crate) fn resize(&mut self, device: &mx::Device, queue: &mx::Queue, size: (u32, u32)) {
        if self.size == size {
            return;
        }
        self.handle.wait();
        self.size = size;
        (self.drawn, self.drawn_view, self.drawn_group) = picture(device, queue, "frame generation: the drawn frame", self.format, size);
        let (made, _, made_group) = picture(device, queue, "frame generation: the generated frame", self.format, size);
        (self.made, self.made_group) = (made, made_group);
    }

    /// What the game draws the window's picture into.
    pub(crate) fn view(&self) -> mx::TextureView {
        self.drawn_view.clone()
    }

    /// The same, to be copied from.
    pub(crate) fn texture(&self) -> mx::Texture {
        self.drawn.clone()
    }

    fn copy(&self, device: &mx::Device, queue: &mx::Queue, picture: &mx::BindGroup, into: &mx::TextureView) {
        let mut encoder = device.create_command_encoder(Some("frame generation copy"));
        {
            let mut pass = encoder.begin_render_pass(&mx::RenderPassDescriptor {
                label: Some("frame generation copy"),
                color_attachments: &[Some(mx::RenderPassColorAttachment {
                    view: into,
                    resolve_target: None,
                    ops: mx::Operations {
                        load: mx::LoadOp::Clear(mx::Color::BLACK),
                        store: mx::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.copy);
            pass.set_bind_group(0, picture);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Shows the frame drawn into [`Self::view`]: MetalFX's picture between the last frame and
    /// this one in `frame` (the window's picture the game acquired), then this frame in the
    /// window's next picture, half a frame time later. Without an upscaled frame to go from
    /// (a menu, the loading screen, a cut), this frame alone.
    pub(crate) fn present(&self, surface: &mx::Surface<'_>, device: &mx::Device, queue: &mx::Queue, frame: mx::SurfaceTexture) {
        let into = frame.texture.create_view(&Default::default()).expect("mX: a texture view");
        let prepared = self.handle.take().filter(|p| p.output_size == self.size);
        let generated = match prepared.as_ref().filter(|_| self.show != Show::Drawn && self.handle.working()) {
            Some(inputs) => self.handle.generate(queue, inputs, &self.drawn, &self.made).unwrap_or_else(|e| {
                self.handle.stop(&e);
                false
            }),
            None => {
                self.handle.shared().gap = true;
                false
            }
        };
        if !generated {
            self.copy(device, queue, &self.drawn_group, &into);
            frame.present();
            return;
        }
        self.copy(device, queue, &self.made_group, &into);
        frame.present();
        if self.show == Show::Generated {
            return;
        }
        if let Err(e) = self.present_drawn(surface, device, queue) {
            self.handle.stop(&e);
        }
    }

    /// The drawn frame into the window's next picture, shown a little under half a frame time
    /// after the generated one ([`Pacer`]). mX's own present cannot wait for that, so this
    /// picture is taken from the window's layer and presented by hand, as wgpu does it.
    fn present_drawn(&self, surface: &mx::Surface<'_>, device: &mx::Device, queue: &mx::Queue) -> Result<()> {
        autoreleasepool(|_| {
            let drawable: Option<Retained<AnyObject>> = surface
                .with_metal(|s| unsafe { msg_send![s.raw(), nextDrawable] })
                .ok_or_else(|| anyhow!("no Metal layer under the window"))?;
            // (none: the window went away under the frame; the next frame shows it)
            let Some(drawable) = drawable else {
                return Ok(());
            };
            let raw: Texture = unsafe { msg_send![&*drawable, texture] };
            let (width, height) = (raw.width() as u32, raw.height() as u32);
            if (width, height) != self.size {
                return Ok(());
            }
            let metal = device.as_metal().ok_or_else(|| anyhow!("the game is not running on Metal"))?;
            // SAFETY: the drawable's texture, of this device, described as it is
            let texture = mx::Texture::Metal(unsafe {
                metal.texture_from_raw(
                    raw,
                    &mx::TextureDescriptor {
                        label: Some("frame generation: the window's second picture"),
                        size: mx::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: mx::TextureDimension::D2,
                        format: self.format,
                        usage: mx::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    },
                )
            });
            self.copy(device, queue, &self.drawn_group, &texture.create_view(&Default::default()).expect("mX: a texture view"));
            // (behind the generated frame and both copies on mX's queue: its end is the pacing's mark)
            let mark = queue
                .as_metal()
                .ok_or_else(|| anyhow!("no Metal queue under mX's"))?
                .raw()
                .commandBuffer()
                .ok_or_else(|| anyhow!("Metal gave no command buffer for the pacing"))?;
            mark.commit();
            let mut pacer = self.pacer.lock().unwrap_or_else(|e| e.into_inner());
            pacer.value += 1;
            let value = pacer.value;
            pacer.send.send(Sent((mark, value))).map_err(|_| anyhow!("frame generation's pacing thread is gone"))?;
            let buffer = pacer.queue.commandBuffer().ok_or_else(|| anyhow!("Metal gave no command buffer for the present"))?;
            unsafe {
                let _: () = msg_send![&*buffer, encodeWaitForEvent: &*pacer.event, value: value];
                let _: () = msg_send![&*buffer, presentDrawable: &*drawable];
            }
            buffer.commit();
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::half_interval;

    /// Frame generation shows the drawn frame a little under half a frame time after the generated
    /// one, and never holds it back long: below 10 fps there is nothing left to pace.
    #[test]
    fn the_drawn_frame_follows_the_generated_one_at_half_a_frame() {
        assert!((half_interval(1.0 / 30.0) - 31.0 / 64.0 / 30.0).abs() < 1e-6);
        assert!(half_interval(1.0 / 30.0) < 1.0 / 60.0);
        assert_eq!(half_interval(0.5), 0.05);
        assert_eq!(half_interval(0.0), 0.0);
        assert_eq!(half_interval(f32::NAN), 0.0);
    }
}
