//! Zeroing textures: new ones before their first use (wgpu's guarantee that a texture reads
//! as zeros until written) and `CommandEncoder::clear_texture`.

use std::ops::Range;

use ash::vk;
use mx_types as t;

use crate::Texture;
use crate::device::{Garbage, Recording, Shared};
use crate::resource::buffer_image_region;

/// Zero `layers` of mip `level` of `tex` (in `GENERAL` layout) in `cmd`. Staging memory goes
/// to `rec`.
pub(crate) fn zero(shared: &Shared, cmd: vk::CommandBuffer, rec: &mut Recording, tex: &Texture, level: u32, layers: Range<u32>, aspect: t::TextureAspect) {
    let format = tex.format();
    let aspect_mask = crate::conv::aspects(format, aspect);
    if aspect_mask.is_empty() {
        return;
    }
    let range = [vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: level,
        level_count: 1,
        base_array_layer: layers.start,
        layer_count: layers.len() as u32,
    }];
    let raw = tex.0.raw;
    if format.has_depth() || format.has_stencil() {
        let value = vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 };
        // SAFETY: the image is in GENERAL with TRANSFER_DST (every mX image has it).
        unsafe { shared.raw.cmd_clear_depth_stencil_image(cmd, raw, vk::ImageLayout::GENERAL, &value, &range) };
    } else if !format.is_compressed() {
        // (all bits zero: the same for float, unorm, uint and sint formats)
        let value = vk::ClearColorValue { uint32: [0; 4] };
        // SAFETY: as above.
        unsafe { shared.raw.cmd_clear_color_image(cmd, raw, vk::ImageLayout::GENERAL, &value, &range) };
    } else {
        // compressed: no clear exists, a copy of zeros, whole blocks
        let is_3d = tex.dimension() == t::TextureDimension::D3;
        let mip = tex.size().mip_level_size(level, tex.dimension());
        let (bw, bh) = format.block_dimensions();
        let block = format.block_copy_size().unwrap_or(16);
        let size = t::Extent3d {
            width: mip.width.div_ceil(bw) * bw,
            height: mip.height.div_ceil(bh) * bh,
            depth_or_array_layers: if is_3d { mip.depth_or_array_layers } else { layers.len() as u32 },
        };
        let bytes_per_row = size.width / bw * block;
        let rows = size.height / bh;
        let len = bytes_per_row as usize * rows as usize * size.depth_or_array_layers as usize;
        let Ok((staging, alloc)) = shared.staging_buffer(&vec![0u8; len], vk::BufferUsageFlags::TRANSFER_SRC) else {
            log::error!("mX: no memory to zero a compressed texture");
            return;
        };
        let at = t::TexelCopyTextureInfo {
            texture: tex,
            mip_level: level,
            origin: t::Origin3d {
                x: 0,
                y: 0,
                z: if is_3d { 0 } else { layers.start },
            },
            aspect: t::TextureAspect::All,
        };
        let layout = t::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(rows),
        };
        let region = [buffer_image_region(&at, layout, 0, size)];
        // SAFETY: the region is inside the image and the staging buffer.
        unsafe { shared.raw.cmd_copy_buffer_to_image(cmd, staging, raw, vk::ImageLayout::GENERAL, &region) };
        rec.temp.push(Garbage::Buffer(staging, alloc));
    }
}

impl Shared {
    /// Bring the textures made since the last submission to `GENERAL` and zero what nothing
    /// has written of them, in a command buffer of its own (submitted before the queue's
    /// writes and the submission).
    pub(crate) fn zero_new_textures(&self) -> Option<(vk::CommandBuffer, Recording)> {
        let fresh = std::mem::take(&mut *self.fresh.lock());
        let mut out: Option<(vk::CommandBuffer, Recording)> = None;
        for weak in fresh {
            let Some(inner) = weak.upgrade() else { continue };
            let left = inner.unwritten.lock().take();
            let tex = Texture(inner);
            let (cmd, rec) = out.get_or_insert_with(|| {
                let (pool, cmd) = self.begin_cmd();
                self.name(cmd, "mX new textures");
                (
                    cmd,
                    Recording {
                        cmd_pools: vec![pool],
                        ..Default::default()
                    },
                )
            });
            // every image lives in GENERAL
            let aspect = crate::conv::aspects(tex.format(), t::TextureAspect::All);
            self.transition(*cmd, tex.0.raw, aspect, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            if let Some(left) = left {
                for (level, layers) in left.runs() {
                    zero(self, *cmd, rec, &tex, level, layers, t::TextureAspect::All);
                }
            }
            rec.keep.push(Box::new(tex));
        }
        if let Some((cmd, _)) = &out {
            self.end_cmd(*cmd);
        }
        out
    }
}
