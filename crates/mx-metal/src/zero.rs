//! Zeroing textures: new ones before their first use (wgpu's guarantee that a texture reads
//! as zeros until written) and `CommandEncoder::clear_texture`.

use std::ops::Range;

use mx_types as t;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::device::Shared;
use crate::resource::copy_buffer_texture;
use crate::{Mt, Po, Texture};

/// Which aspects of a depth-stencil texture to zero.
#[derive(Clone, Copy)]
pub(crate) struct Aspects {
    pub depth: bool,
    pub stencil: bool,
}

impl Aspects {
    pub fn of(format: t::TextureFormat, aspect: t::TextureAspect) -> Aspects {
        Aspects {
            depth: format.has_depth() && aspect != t::TextureAspect::StencilOnly,
            stencil: format.has_stencil() && aspect != t::TextureAspect::DepthOnly,
        }
    }
}

impl Shared {
    /// A buffer of at least `len` zero bytes, for copies into textures that cannot be
    /// render targets (compressed formats, textures made without `RENDER_ATTACHMENT`).
    /// Never written: Metal hands out a new buffer's bytes as zeros.
    pub(crate) fn zeros(&self, len: usize) -> Mt<Po<dyn MTLBuffer>> {
        let mut z = self.zeros.lock();
        if let Some(b) = z.as_ref()
            && b.length() >= len
        {
            return b.clone();
        }
        let b = Mt(self
            .device
            .newBufferWithLength_options(len.max(4096), MTLResourceOptions::StorageModeShared)
            .expect("Metal returned no buffer for zeros"));
        b.setLabel(Some(&crate::ns("mX zeros")));
        *z = Some(b.clone());
        b
    }
}

/// Zero `layers` of mip `level` of `tex` in `cmd` (no encoder may be open on it).
pub(crate) fn zero(cmd: &ProtocolObject<dyn MTLCommandBuffer>, shared: &Shared, tex: &Texture, level: u32, layers: Range<u32>, aspects: Aspects) {
    let format = tex.format();
    let raw = &tex.0.raw;
    let mip = tex.size().mip_level_size(level, tex.dimension());
    let is_3d = tex.dimension() == t::TextureDimension::D3;
    if raw.usage().contains(MTLTextureUsage::RenderTarget) {
        // a render pass that only clears: what Metal does fastest, and the only way into a
        // multisampled texture
        let slices: Range<u32> = if is_3d { 0..mip.depth_or_array_layers } else { layers };
        for s in slices {
            let d = MTLRenderPassDescriptor::new();
            let place = |a: &MTLRenderPassAttachmentDescriptor| {
                a.setTexture(Some(raw));
                a.setLevel(level as usize);
                if is_3d {
                    a.setDepthPlane(s as usize);
                } else {
                    a.setSlice(s as usize);
                }
                a.setLoadAction(MTLLoadAction::Clear);
                a.setStoreAction(MTLStoreAction::Store);
            };
            if format.has_depth() || format.has_stencil() {
                if aspects.depth {
                    let da = d.depthAttachment();
                    place(&da);
                    da.setClearDepth(0.0);
                }
                if aspects.stencil {
                    let sa = d.stencilAttachment();
                    place(&sa);
                    sa.setClearStencil(0);
                }
            } else {
                // SAFETY: attachment 0 exists on every descriptor.
                let ca = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
                place(&ca);
                ca.setClearColor(MTLClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 0.0,
                });
            }
            let enc = cmd.renderCommandEncoderWithDescriptor(&d).expect("Metal returned no render encoder");
            enc.setLabel(Some(&crate::ns("mX zero")));
            enc.endEncoding();
        }
        return;
    }
    // a copy from a buffer of zeros, whole blocks
    let (bw, bh) = format.block_dimensions();
    let block = format.block_copy_size().unwrap_or(4);
    let size = t::Extent3d {
        width: mip.width.div_ceil(bw) * bw,
        height: mip.height.div_ceil(bh) * bh,
        depth_or_array_layers: if is_3d { mip.depth_or_array_layers } else { layers.len() as u32 },
    };
    let bytes_per_row = size.width / bw * block;
    let rows = size.height / bh;
    let len = bytes_per_row as usize * rows as usize * size.depth_or_array_layers as usize;
    let zeros = shared.zeros(len);
    let blit = cmd.blitCommandEncoder().expect("Metal returned no blit encoder");
    blit.setLabel(Some(&crate::ns("mX zero")));
    let layout = t::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(bytes_per_row),
        rows_per_image: Some(rows),
    };
    let mut aspect_list = Vec::new();
    match (format.has_depth() && format.has_stencil(), aspects.depth, aspects.stencil) {
        (true, d, s) => {
            if d {
                aspect_list.push(t::TextureAspect::DepthOnly);
            }
            if s {
                aspect_list.push(t::TextureAspect::StencilOnly);
            }
        }
        _ => aspect_list.push(t::TextureAspect::All),
    }
    for aspect in aspect_list {
        let at = t::TexelCopyTextureInfo {
            texture: tex,
            mip_level: level,
            origin: t::Origin3d {
                x: 0,
                y: 0,
                z: if is_3d { 0 } else { layers.start },
            },
            aspect,
        };
        copy_buffer_texture(&blit, &zeros, 0, tex, &at, layout, size, true);
    }
    blit.endEncoding();
}

impl Shared {
    /// Zero what nothing has written of the textures made since the last submission, in a
    /// command buffer of its own (committed before the queue's writes and the submission).
    pub(crate) fn zero_new_textures(&self, index: u64) -> Option<Mt<Po<dyn MTLCommandBuffer>>> {
        let fresh = std::mem::take(&mut *self.fresh.lock());
        let mut cmd = None;
        for weak in fresh {
            let Some(inner) = weak.upgrade() else { continue };
            let Some(left) = inner.unwritten.lock().take() else { continue };
            if left.is_empty() {
                continue;
            }
            let tex = Texture(inner);
            let cmd = cmd.get_or_insert_with(|| {
                let c = self.new_command_buffer();
                c.setLabel(Some(&crate::ns("mX zero new textures")));
                c
            });
            let aspects = Aspects::of(tex.format(), t::TextureAspect::All);
            for (level, layers) in left.runs() {
                zero(cmd, self, &tex, level, layers, aspects);
            }
            self.heap.lock().written(tex.owner_id(), index);
        }
        cmd
    }
}
