//! Apple's MetalFX spatial scaler (macOS 13 and newer): one frame brought up to the output, no
//! depth, motion vectors or jitter. Opened by name as the temporal scaler (`metalfx.rs`).

use crate::Inputs;
use crate::metalfx::{Texture, class, texture};
use anyhow::{Result, anyhow, bail};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{Message, msg_send};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLTexture};

/// `MTLFXSpatialScalerColorProcessingMode`: the picture is sRGB-encoded (perceptual), linear, or
/// HDR (linear, beyond 1).
const COLOR_PERCEPTUAL: isize = 0;
const COLOR_LINEAR: isize = 1;

struct Scaler {
    raw: Retained<AnyObject>,
    render: (u32, u32),
    output: (u32, u32),
    formats: [MTLPixelFormat; 2],
}

pub struct Upscaler {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    scaler: Option<Scaler>,
    last: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

// (used from one thread at a time)
unsafe impl Send for Upscaler {}

impl Upscaler {
    pub fn load(device: &mx::Device) -> Result<Upscaler> {
        let raw = device.as_metal().ok_or_else(|| anyhow!("not running on Metal"))?.raw().retain();
        let class = class(c"MTLFXSpatialScalerDescriptor").ok_or_else(|| anyhow!("this macOS has no MetalFX (it comes with macOS 13 and newer)"))?;
        let supported: bool = unsafe { msg_send![class, supportsDevice: &*raw] };
        if !supported {
            bail!("this Mac's graphics chip cannot run MetalFX's spatial scaler");
        }
        log::info!("MetalFX upscaling: Apple's spatial scaler");
        Ok(Upscaler {
            device: raw,
            scaler: None,
            last: None,
        })
    }

    pub(crate) fn wait(&mut self) {
        if let Some(b) = self.last.take() {
            b.waitUntilCompleted();
        }
    }

    fn scaler_for(&mut self, color: &Texture, output: &Texture, render: (u32, u32), out: (u32, u32)) -> Result<&Scaler> {
        let formats = [color.pixelFormat(), output.pixelFormat()];
        if self.scaler.as_ref().is_some_and(|s| s.render != render || s.output != out || s.formats != formats) {
            self.scaler = None;
        }
        if self.scaler.is_none() {
            let class = class(c"MTLFXSpatialScalerDescriptor").ok_or_else(|| anyhow!("MetalFX is gone"))?;
            let desc: Retained<AnyObject> = unsafe { msg_send![class, new] };
            // (an sRGB format holds linear numbers once read; a UNORM one the encoded picture)
            let linear = matches!(
                formats[0],
                MTLPixelFormat::RGBA8Unorm_sRGB | MTLPixelFormat::BGRA8Unorm_sRGB | MTLPixelFormat::RGBA16Float | MTLPixelFormat::RGBA32Float
            );
            let made: Option<Retained<AnyObject>> = unsafe {
                let _: () = msg_send![&*desc, setColorTextureFormat: formats[0]];
                let _: () = msg_send![&*desc, setOutputTextureFormat: formats[1]];
                let _: () = msg_send![&*desc, setInputWidth: render.0 as usize];
                let _: () = msg_send![&*desc, setInputHeight: render.1 as usize];
                let _: () = msg_send![&*desc, setOutputWidth: out.0 as usize];
                let _: () = msg_send![&*desc, setOutputHeight: out.1 as usize];
                let _: () = msg_send![&*desc, setColorProcessingMode: if linear { COLOR_LINEAR } else { COLOR_PERCEPTUAL }];
                msg_send![&*desc, newSpatialScalerWithDevice: &*self.device]
            };
            let raw = made.ok_or_else(|| anyhow!("MetalFX made no spatial scaler for {}x{} to {}x{} ({formats:?})", render.0, render.1, out.0, out.1))?;
            let needs: [(&str, objc2_metal::MTLTextureUsage, &Texture); 2] =
                unsafe { [("picture", msg_send![&*raw, colorTextureUsage], color), ("result", msg_send![&*raw, outputTextureUsage], output)] };
            for (what, need, have) in needs {
                if have.usage().0 & need.0 != need.0 {
                    bail!("MetalFX cannot use the {what} texture (usage {:#x}, it needs {:#x})", have.usage().0, need.0);
                }
            }
            log::info!("MetalFX spatial upscaling: {}x{} to {}x{}", render.0, render.1, out.0, out.1);
            self.scaler = Some(Scaler { raw, render, output: out, formats });
        }
        Ok(self.scaler.as_ref().expect("made above"))
    }

    pub fn upscale(&mut self, queue: &mx::Queue, inputs: &Inputs) -> Result<()> {
        autoreleasepool(|_| {
            let color = texture(inputs.color, "picture")?;
            let output = texture(inputs.output, "result")?;
            let buffer = queue
                .as_metal()
                .ok_or_else(|| anyhow!("no Metal queue under mX's"))?
                .raw()
                .commandBuffer()
                .ok_or_else(|| anyhow!("Metal gave no command buffer for MetalFX"))?;
            let scaler = self.scaler_for(&color, &output, inputs.render_size, inputs.output_size)?;
            let none: Option<&ProtocolObject<dyn MTLTexture>> = None;
            unsafe {
                let s = &*scaler.raw;
                let _: () = msg_send![s, setColorTexture: &*color];
                let _: () = msg_send![s, setOutputTexture: &*output];
                let _: () = msg_send![s, setInputContentWidth: inputs.render_size.0 as usize];
                let _: () = msg_send![s, setInputContentHeight: inputs.render_size.1 as usize];
                let _: () = msg_send![s, encodeToCommandBuffer: &*buffer];
                let _: () = msg_send![s, setColorTexture: none];
                let _: () = msg_send![s, setOutputTexture: none];
            }
            buffer.commit();
            self.last = Some(buffer);
            Ok(())
        })
    }
}

impl Drop for Upscaler {
    fn drop(&mut self) {
        self.wait();
    }
}
