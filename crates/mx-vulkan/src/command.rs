//! Command encoders and passes.
//!
//! Ordering: one full memory barrier before every pass and copy that follows other work
//! (and between dispatches of a compute pass), so each sees what came before, as in wgpu.
//! Images stay in `GENERAL`: no layout changes inside a command buffer.

use std::collections::HashMap;
use std::ffi::CString;
use std::ops::Range;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use mx_shader::{MAX_ROOT_GROUP, RootKind};
use mx_types as t;

use crate::device::{Recording, Shared};
use crate::pipeline::{Group, PipelineInner};
use crate::resource::{Bound, buffer_image_region};
use crate::{BindGroup, Buffer, ComputePipeline, Device, QuerySet, RenderPipeline, Texture, TextureView, conv};

const GROUPS: usize = MAX_ROOT_GROUP as usize + 1;
const MAX_IMMEDIATES: usize = 4096;

/// Recorded commands, ready for [`crate::Queue::submit`].
pub struct CommandBuffer {
    /// Run in this order; each after the first goes in a queue submission of its own (an
    /// encoder may split its work, see `CommandEncoder::split`).
    pub(crate) raws: Vec<vk::CommandBuffer>,
    pub(crate) rec: Recording,
    pub(crate) shared: Arc<Shared>,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        // (never submitted: give the pools back now)
        let rec = std::mem::take(&mut self.rec);
        self.shared.recycle(rec);
    }
}

pub struct CommandEncoder {
    pub(crate) raw: vk::CommandBuffer,
    /// Command buffers this encoder already ended (see `split`).
    done: Vec<vk::CommandBuffer>,
    pub(crate) shared: Arc<Shared>,
    pub(crate) rec: Recording,
    desc_pool: Option<vk::DescriptorPool>,
    /// Root descriptor sets made in this encoder, by (set layout, bind group).
    sets: HashMap<(u64, usize), vk::DescriptorSet>,
    /// Occlusion query sets reset in this encoder.
    reset_queries: Vec<usize>,
    /// Work was recorded since the last barrier.
    dirty: bool,
}

impl Device {
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder {
        let (pool, raw) = self.shared.begin_cmd();
        if let Some(l) = label {
            self.shared.name(raw, l);
        }
        CommandEncoder {
            raw,
            done: Vec::new(),
            shared: self.shared.clone(),
            rec: Recording {
                cmd_pools: vec![pool],
                ..Default::default()
            },
            desc_pool: None,
            sets: HashMap::new(),
            reset_queries: Vec::new(),
            dirty: false,
        }
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        let rec = std::mem::take(&mut self.rec);
        self.shared.recycle(rec);
    }
}

fn new_desc_pool(shared: &Shared) -> vk::DescriptorPool {
    let mut sizes = vec![
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 512,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 512,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 512,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 128,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLER,
            descriptor_count: 256,
        },
    ];
    if shared.caps.ray_query {
        sizes.push(vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 32,
        });
    }
    // SAFETY: plain creation.
    unsafe { shared.raw.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(256).pool_sizes(&sizes), None) }.expect("out of memory for a descriptor pool")
}

impl CommandEncoder {
    /// The `VkCommandBuffer` the encoder records into, for interop (DLSS / FSR record
    /// here). Passes must be ended before; mX orders the next command after it.
    pub fn raw(&mut self) -> vk::CommandBuffer {
        self.dirty = true;
        self.raw
    }

    pub fn finish(mut self) -> CommandBuffer {
        self.shared.end_cmd(self.raw);
        let mut raws = std::mem::take(&mut self.done);
        raws.push(self.raw);
        CommandBuffer {
            raws,
            rec: std::mem::take(&mut self.rec),
            shared: self.shared.clone(),
        }
    }

    /// Go on in a new command buffer, submitted separately after this one. MoltenVK makes
    /// timestamp results available for copies only to later queue submissions (a resolve in
    /// the same one copied stale values), so a resolve of timestamps starts a new one there.
    pub(crate) fn split(&mut self) {
        self.shared.end_cmd(self.raw);
        self.done.push(self.raw);
        let (pool, raw) = self.shared.begin_cmd();
        self.rec.cmd_pools.push(pool);
        self.raw = raw;
        self.dirty = false;
    }

    /// Before a command: wait for what came before.
    pub(crate) fn before(&mut self) {
        if self.dirty {
            self.shared.full_barrier(self.raw);
            self.dirty = false;
        }
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub(crate) fn keep<T: Clone + Send + Sync + 'static>(&mut self, x: &T) {
        self.rec.keep.push(Box::new(x.clone()));
    }

    fn device(&self) -> &ash::Device {
        &self.shared.raw
    }

    /// Fill `size` bytes (to the end with `None`) from `offset` with zeros.
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        let size = size.unwrap_or(buffer.size() - offset);
        assert!(offset + size <= buffer.size(), "clear_buffer past the end of the buffer");
        if size == 0 {
            return;
        }
        self.before();
        // SAFETY: the range was checked above.
        unsafe { self.device().cmd_fill_buffer(self.raw, buffer.0.raw, offset, size, 0) };
        self.keep(buffer);
        self.dirty = true;
    }

    /// Zero the mips and layers of `range` (colour to 0, depth to 0.0, stencil to 0).
    pub fn clear_texture(&mut self, texture: &Texture, range: &t::ImageSubresourceRange) {
        let layers = t::Unwritten::layer_count(texture.dimension(), texture.size());
        let (mips, layers) = range.resolve(texture.mip_level_count(), layers);
        if mips.is_empty() || layers.is_empty() {
            return;
        }
        self.before();
        let shared = self.shared.clone();
        for level in mips {
            crate::zero::zero(&shared, self.raw, &mut self.rec, texture, level, layers.clone(), range.aspect);
        }
        self.keep(texture);
        self.dirty = true;
    }

    pub fn push_debug_group(&mut self, label: &str) {
        if let (Some(d), Ok(name)) = (&self.shared.fns.debug, CString::new(label)) {
            // SAFETY: recording.
            unsafe { d.cmd_begin_debug_utils_label(self.raw, &vk::DebugUtilsLabelEXT::default().label_name(&name)) };
        }
    }

    pub fn pop_debug_group(&mut self) {
        if let Some(d) = &self.shared.fns.debug {
            // SAFETY: recording.
            unsafe { d.cmd_end_debug_utils_label(self.raw) };
        }
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        if let (Some(d), Ok(name)) = (&self.shared.fns.debug, CString::new(label)) {
            // SAFETY: recording.
            unsafe { d.cmd_insert_debug_utils_label(self.raw, &vk::DebugUtilsLabelEXT::default().label_name(&name)) };
        }
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        assert!(src_offset + size <= src.size() && dst_offset + size <= dst.size(), "copy_buffer_to_buffer out of range");
        if size == 0 {
            return;
        }
        self.before();
        let region = [vk::BufferCopy { src_offset, dst_offset, size }];
        // SAFETY: both ranges checked above.
        unsafe { self.device().cmd_copy_buffer(self.raw, src.0.raw, dst.0.raw, &region) };
        self.keep(src);
        self.keep(dst);
        self.dirty = true;
    }

    pub fn copy_buffer_to_texture(&mut self, src: t::TexelCopyBufferInfo<'_, Buffer>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        src.layout.assert_encoder_pitch();
        self.before();
        let region = [buffer_image_region(&dst, src.layout, src.layout.offset, size)];
        // SAFETY: the region is the caller's to keep inside both (checked by the validation layer).
        unsafe { self.device().cmd_copy_buffer_to_image(self.raw, src.buffer.0.raw, dst.texture.0.raw, vk::ImageLayout::GENERAL, &region) };
        self.keep(src.buffer);
        self.keep(dst.texture);
        self.dirty = true;
    }

    pub fn copy_texture_to_buffer(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyBufferInfo<'_, Buffer>, size: t::Extent3d) {
        dst.layout.assert_encoder_pitch();
        self.before();
        let region = [buffer_image_region(&src, dst.layout, dst.layout.offset, size)];
        // SAFETY: as above.
        unsafe { self.device().cmd_copy_image_to_buffer(self.raw, src.texture.0.raw, vk::ImageLayout::GENERAL, dst.buffer.0.raw, &region) };
        self.keep(src.texture);
        self.keep(dst.buffer);
        self.dirty = true;
    }

    pub fn copy_texture_to_texture(&mut self, src: t::TexelCopyTextureInfo<'_, Texture>, dst: t::TexelCopyTextureInfo<'_, Texture>, size: t::Extent3d) {
        self.before();
        let layers = |at: &t::TexelCopyTextureInfo<'_, Texture>| {
            let is_3d = at.texture.dimension() == t::TextureDimension::D3;
            let mut aspect = conv::aspects(at.texture.format(), at.aspect);
            if aspect == vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL && at.aspect != t::TextureAspect::All {
                aspect = vk::ImageAspectFlags::DEPTH;
            }
            (
                vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: at.mip_level,
                    base_array_layer: if is_3d { 0 } else { at.origin.z },
                    layer_count: if is_3d { 1 } else { size.depth_or_array_layers },
                },
                conv::origin(at.origin, is_3d),
                is_3d,
            )
        };
        let (s, so, s3) = layers(&src);
        let (d, d_o, _) = layers(&dst);
        let region = [vk::ImageCopy {
            src_subresource: s,
            src_offset: so,
            dst_subresource: d,
            dst_offset: d_o,
            extent: {
                let a = crate::resource::clamp_to_mip(src.texture, src.mip_level, src.origin, size);
                let b = crate::resource::clamp_to_mip(dst.texture, dst.mip_level, dst.origin, size);
                conv::extent(
                    t::Extent3d {
                        width: a.width.min(b.width),
                        height: a.height.min(b.height),
                        depth_or_array_layers: a.depth_or_array_layers.min(b.depth_or_array_layers),
                    },
                    s3,
                )
            },
        }];
        // SAFETY: regions inside both textures (the caller's contract).
        unsafe {
            self.device()
                .cmd_copy_image(self.raw, src.texture.0.raw, vk::ImageLayout::GENERAL, dst.texture.0.raw, vk::ImageLayout::GENERAL, &region)
        };
        self.keep(src.texture);
        self.keep(dst.texture);
        self.dirty = true;
    }

    /// Fill a texture's mip chain from level 0 (each level a linear blit of the one above).
    pub fn generate_mipmaps(&mut self, texture: &Texture) {
        let f = texture.format();
        let filter = match f.sample_type() {
            t::TextureSampleType::Float { filterable: true } if !f.has_depth() => vk::Filter::LINEAR,
            _ => vk::Filter::NEAREST,
        };
        let aspect = conv::aspects(f, t::TextureAspect::All);
        let is_3d = texture.dimension() == t::TextureDimension::D3;
        let layers = texture.layers();
        for level in 1..texture.mip_level_count() {
            self.before();
            let (a, b) = (texture.size().mip_level_size(level - 1, texture.dimension()), texture.size().mip_level_size(level, texture.dimension()));
            let corner = |e: t::Extent3d| vk::Offset3D {
                x: e.width as i32,
                y: e.height as i32,
                z: if is_3d { e.depth_or_array_layers as i32 } else { 1 },
            };
            let sub = |mip_level| vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level,
                base_array_layer: 0,
                layer_count: layers,
            };
            let region = [vk::ImageBlit {
                src_subresource: sub(level - 1),
                src_offsets: [vk::Offset3D::default(), corner(a)],
                dst_subresource: sub(level),
                dst_offsets: [vk::Offset3D::default(), corner(b)],
            }];
            // SAFETY: two different levels of the same image, both in GENERAL.
            unsafe {
                self.device()
                    .cmd_blit_image(self.raw, texture.0.raw, vk::ImageLayout::GENERAL, texture.0.raw, vk::ImageLayout::GENERAL, &region, filter)
            };
            self.dirty = true;
        }
        self.keep(texture);
    }

    /// Reset an occlusion query set once per encoder, before the first pass counting into it.
    fn reset_occlusion(&mut self, q: &QuerySet) {
        let key = Arc::as_ptr(&q.0) as usize;
        if !self.reset_queries.contains(&key) {
            self.reset_queries.push(key);
            // SAFETY: outside a render pass; the range is the set.
            unsafe { self.device().cmd_reset_query_pool(self.raw, q.0.raw, 0, q.0.count) };
            self.keep(q);
        }
    }

    /// Reset timestamp query `index` (outside passes, before it is written).
    pub(crate) fn reset_timestamp(&mut self, q: &QuerySet, index: u32) {
        assert!(index < q.count(), "timestamp index {index} past the query set");
        assert!(q.ty() == t::QueryType::Timestamp, "timestamps written to an occlusion query set");
        // SAFETY: outside a render pass; the index is inside the set.
        unsafe { self.device().cmd_reset_query_pool(self.raw, q.0.raw, index, 1) };
    }

    /// Write timestamp query `index` (reset before), at the bottom of the pipe as wgpu-hal.
    pub(crate) fn timestamp(&mut self, q: &QuerySet, index: u32) {
        let stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
        // SAFETY: the index is inside the set (checked at reset).
        unsafe {
            match &self.shared.fns.sync2 {
                Some(f) => f.cmd_write_timestamp2(self.raw, stage, q.0.raw, index),
                None => self.device().cmd_write_timestamp2(self.raw, stage, q.0.raw, index),
            }
        }
        self.keep(q);
    }

    /// A descriptor set from this encoder's pools.
    fn alloc_set(&mut self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let layouts = [layout];
        loop {
            let pool = match self.desc_pool {
                Some(p) => p,
                None => {
                    let p = self.shared.take_desc_pool().unwrap_or_else(|| new_desc_pool(&self.shared));
                    self.rec.desc_pools.push(p);
                    self.desc_pool = Some(p);
                    p
                }
            };
            // SAFETY: the pool is this encoder's alone.
            match unsafe {
                self.shared
                    .raw
                    .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))
            } {
                Ok(s) => return s[0],
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => self.desc_pool = None,
                Err(e) => panic!("vkAllocateDescriptorSets: {e}"),
            }
        }
    }

    /// The descriptor set of root group `group` with `bg`'s resources.
    fn root_set(&mut self, group: u32, g: &Group, bg: &BindGroup, label: &str) -> vk::DescriptorSet {
        let key = (g.layout.as_raw(), Arc::as_ptr(&bg.0) as usize);
        if let Some(s) = self.sets.get(&key) {
            return *s;
        }
        enum Info {
            Buffer(vk::DescriptorBufferInfo),
            Image(vk::DescriptorImageInfo),
            Accel(vk::AccelerationStructureKHR),
        }
        let mut infos = Vec::with_capacity(g.bindings.len());
        for &(binding, kind, ty) in &g.bindings {
            let Some(r) = bg.get(binding) else {
                panic!("{label}: draw with nothing bound at @group({group}) @binding({binding})");
            };
            let wrong = || -> ! { panic!("{label}: the resource at @group({group}) @binding({binding}) is not the kind the shader declares") };
            let info = match (kind, r) {
                (RootKind::UniformBuffer | RootKind::StorageBuffer { .. }, Bound::Buffer { buffer, offset, size }) => Info::Buffer(vk::DescriptorBufferInfo {
                    buffer: buffer.0.raw,
                    offset: *offset,
                    range: *size,
                }),
                (RootKind::Texture | RootKind::StorageTexture, Bound::Texture(v)) => Info::Image(vk::DescriptorImageInfo::default().image_view(v.0.raw).image_layout(vk::ImageLayout::GENERAL)),
                (RootKind::Sampler, Bound::Sampler(s)) => Info::Image(vk::DescriptorImageInfo::default().sampler(s.0.raw)),
                (RootKind::AccelerationStructure, Bound::Tlas(tl)) => Info::Accel(tl.raw),
                _ => wrong(),
            };
            infos.push((binding, ty, info));
        }
        let set = self.alloc_set(g.layout);
        let accels: Vec<[vk::AccelerationStructureKHR; 1]> = infos.iter().map(|(_, _, i)| [if let Info::Accel(a) = i { *a } else { vk::AccelerationStructureKHR::null() }]).collect();
        let mut accel_writes: Vec<vk::WriteDescriptorSetAccelerationStructureKHR> =
            accels.iter().map(|a| vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(a)).collect();
        let mut writes = Vec::with_capacity(infos.len());
        for ((binding, ty, info), aw) in infos.iter().zip(accel_writes.iter_mut()) {
            let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty);
            writes.push(match info {
                Info::Buffer(b) => w.buffer_info(std::slice::from_ref(b)),
                Info::Image(i) => w.image_info(std::slice::from_ref(i)),
                Info::Accel(_) => w.descriptor_count(1).push_next(aw),
            });
        }
        // SAFETY: a new set of this encoder; the infos outlive the call.
        unsafe { self.shared.raw.update_descriptor_sets(&writes, &[]) };
        self.sets.insert(key, set);
        self.keep(bg);
        set
    }

    pub fn begin_render_pass(&mut self, desc: &t::RenderPassDescriptor<'_, TextureView, QuerySet>) -> RenderPass<'_> {
        self.before();
        if let Some(q) = desc.occlusion_query_set {
            assert!(q.ty() == t::QueryType::Occlusion, "occlusion queries in a timestamp query set");
            self.reset_occlusion(q);
        }
        // (the pass's timestamps are written inside the rendering scope, so that MoltenVK
        // samples them at the render encoder's boundaries, as the Metal backend does; written
        // just outside, MoltenVK resolved the end one before sampling it)
        let mut end_timestamp = None;
        if let Some(ts) = &desc.timestamp_writes {
            for i in [ts.beginning_of_pass_write_index, ts.end_of_pass_write_index].into_iter().flatten() {
                self.reset_timestamp(ts.query_set, i);
            }
            end_timestamp = ts.end_of_pass_write_index.map(|i| (ts.query_set.clone(), i));
        }
        let mut extent = None;
        let mut colors = Vec::new();
        for a in desc.color_attachments {
            let Some(a) = a else {
                colors.push(vk::RenderingAttachmentInfo::default());
                continue;
            };
            extent.get_or_insert(a.view.extent());
            let mut info = vk::RenderingAttachmentInfo::default()
                .image_view(a.view.0.raw)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(conv::load_op(&a.ops.load))
                .store_op(conv::store_op(a.ops.store));
            if let t::LoadOp::Clear(c) = a.ops.load {
                info = info.clear_value(vk::ClearValue {
                    color: conv::clear_color(c, a.view.format()),
                });
            }
            if let Some(r) = a.resolve_target {
                let mode = match a.view.format().sample_type() {
                    t::TextureSampleType::Uint | t::TextureSampleType::Sint => vk::ResolveModeFlags::SAMPLE_ZERO,
                    _ => vk::ResolveModeFlags::AVERAGE,
                };
                info = info.resolve_mode(mode).resolve_image_view(r.0.raw).resolve_image_layout(vk::ImageLayout::GENERAL);
                self.keep(r);
            }
            self.keep(a.view);
            colors.push(info);
        }
        let mut depth = None;
        let mut stencil = None;
        if let Some(ds) = &desc.depth_stencil_attachment {
            extent.get_or_insert(ds.view.extent());
            let format = ds.view.texture().format();
            let att = |load: vk::AttachmentLoadOp, store: vk::AttachmentStoreOp, clear: vk::ClearValue| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(ds.view.0.raw)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(load)
                    .store_op(store)
                    .clear_value(clear)
            };
            if format.has_depth() {
                depth = Some(match &ds.depth_ops {
                    Some(ops) => {
                        let v = if let t::LoadOp::Clear(v) = ops.load { v } else { 0.0 };
                        att(
                            conv::load_op(&ops.load),
                            conv::store_op(ops.store),
                            vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue { depth: v, stencil: 0 },
                            },
                        )
                    }
                    None => att(vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE, vk::ClearValue::default()),
                });
            }
            if format.has_stencil() {
                stencil = Some(match &ds.stencil_ops {
                    Some(ops) => {
                        let v = if let t::LoadOp::Clear(v) = ops.load { v } else { 0 };
                        att(
                            conv::load_op(&ops.load),
                            conv::store_op(ops.store),
                            vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: v },
                            },
                        )
                    }
                    None => att(vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE, vk::ClearValue::default()),
                });
            }
            self.keep(ds.view);
        }
        let extent = extent.expect("a render pass needs at least one attachment");
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
        };
        let mut info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors);
        if let Some(d) = &depth {
            info = info.depth_attachment(d);
        }
        if let Some(s) = &stencil {
            info = info.stencil_attachment(s);
        }
        let cmd = self.raw;
        let shared = self.shared.clone();
        // SAFETY: the attachments are live views in GENERAL layout; the arrays outlive the call.
        unsafe {
            match &shared.fns.dynamic_rendering {
                Some(f) => f.cmd_begin_rendering(cmd, &info),
                None => shared.raw.cmd_begin_rendering(cmd, &info),
            }
        }
        if let Some(l) = desc.label {
            self.push_debug_group(l);
        }
        if let Some(ts) = &desc.timestamp_writes
            && let Some(i) = ts.beginning_of_pass_write_index
        {
            self.timestamp(ts.query_set, i);
        }
        let mut pass = RenderPass {
            enc: self,
            state: PassState::default(),
            pipeline: None,
            index: false,
            occlusion: desc.occlusion_query_set.cloned(),
            query_active: None,
            end_timestamp,
            labelled: desc.label.is_some(),
            label: desc.label.unwrap_or("render pass").to_owned(),
        };
        // WebGPU's defaults: the whole target, no blend constant, stencil reference 0
        pass.set_viewport(0.0, 0.0, extent.width as f32, extent.height as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, extent.width, extent.height);
        pass.set_blend_constant(t::Color::TRANSPARENT);
        pass.set_stencil_reference(0);
        pass
    }

    pub fn begin_compute_pass(&mut self, desc: &t::ComputePassDescriptor<'_, QuerySet>) -> ComputePass<'_> {
        self.before();
        let mut end_timestamp = None;
        if let Some(ts) = &desc.timestamp_writes {
            for i in [ts.beginning_of_pass_write_index, ts.end_of_pass_write_index].into_iter().flatten() {
                self.reset_timestamp(ts.query_set, i);
            }
            if let Some(i) = ts.beginning_of_pass_write_index {
                self.timestamp(ts.query_set, i);
            }
            end_timestamp = ts.end_of_pass_write_index.map(|i| (ts.query_set.clone(), i));
        }
        if let Some(l) = desc.label {
            self.push_debug_group(l);
        }
        ComputePass {
            enc: self,
            state: PassState::default(),
            pipeline: None,
            end_timestamp,
            labelled: desc.label.is_some(),
            dispatched: false,
        }
    }
}

/// Bindings set on a pass and not yet handed to the command buffer.
struct PassState {
    groups: [Option<BindGroup>; GROUPS],
    roots_dirty: bool,
    immediates: Box<[u8; MAX_IMMEDIATES]>,
    immediates_dirty: bool,
}

impl Default for PassState {
    fn default() -> Self {
        PassState {
            groups: Default::default(),
            roots_dirty: true,
            immediates: Box::new([0; MAX_IMMEDIATES]),
            immediates_dirty: true,
        }
    }
}

impl PassState {
    fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        assert!(group <= MAX_ROOT_GROUP, "root bind groups are 0..={MAX_ROOT_GROUP}; group {} is the heap", mx_shader::HEAP_GROUP);
        self.groups[group as usize] = Some(bind_group.clone());
        self.roots_dirty = true;
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        let o = offset as usize;
        self.immediates[o..o + data.len()].copy_from_slice(data);
        self.immediates_dirty = true;
    }
}

/// Bind a pipeline and the heap (set 0).
fn bind_pipeline(enc: &mut CommandEncoder, p: &PipelineInner) {
    let heap = enc.shared.heap.lock().set;
    // SAFETY: live pipeline and layout; set 0 is the heap layout of every pipeline layout.
    unsafe {
        enc.shared.raw.cmd_bind_pipeline(enc.raw, p.bind_point, p.raw);
        enc.shared.raw.cmd_bind_descriptor_sets(enc.raw, p.bind_point, p.layout, 0, &[heap], &[]);
    }
}

/// Bind the root groups and immediates the pipeline reads.
fn flush(enc: &mut CommandEncoder, p: &PipelineInner, state: &mut PassState, label: &str) {
    if state.roots_dirty {
        for (g, group) in p.groups.iter().enumerate() {
            let Some(group) = group else { continue };
            let Some(bg) = state.groups[g].clone() else {
                panic!("{label}: draw with nothing bound at @group({g}) @binding({})", group.bindings[0].0);
            };
            let set = enc.root_set(g as u32, group, &bg, label);
            // SAFETY: a set of the layout the pipeline layout has at `g + 1`.
            unsafe { enc.shared.raw.cmd_bind_descriptor_sets(enc.raw, p.bind_point, p.layout, g as u32 + 1, &[set], &[]) };
        }
        state.roots_dirty = false;
    }
    if state.immediates_dirty && p.immediate_size > 0 {
        let n = (p.immediate_size.div_ceil(4) * 4) as usize;
        // SAFETY: the range is the pipeline layout's push-constant range.
        unsafe { enc.shared.raw.cmd_push_constants(enc.raw, p.layout, p.push_stages, 0, &state.immediates[..n]) };
        state.immediates_dirty = false;
    }
}

/// A render pass. Ends when dropped.
pub struct RenderPass<'e> {
    enc: &'e mut CommandEncoder,
    state: PassState,
    pipeline: Option<RenderPipeline>,
    /// An index buffer is set.
    index: bool,
    occlusion: Option<QuerySet>,
    query_active: Option<u32>,
    end_timestamp: Option<(QuerySet, u32)>,
    labelled: bool,
    label: String,
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        if self.query_active.is_some() {
            self.end_occlusion_query();
        }
        if let Some((q, i)) = self.end_timestamp.take() {
            self.enc.timestamp(&q, i);
        }
        let shared = &self.enc.shared;
        // SAFETY: rendering was begun by `begin_render_pass`.
        unsafe {
            match &shared.fns.dynamic_rendering {
                Some(f) => f.cmd_end_rendering(self.enc.raw),
                None => shared.raw.cmd_end_rendering(self.enc.raw),
            }
        }
        if self.labelled {
            self.enc.pop_debug_group();
        }
        self.enc.dirty = true;
    }
}

impl RenderPass<'_> {
    fn cmd(&self) -> (vk::CommandBuffer, &ash::Device) {
        (self.enc.raw, &self.enc.shared.raw)
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        bind_pipeline(self.enc, &pipeline.0);
        self.enc.keep(pipeline);
        self.pipeline = Some(pipeline.clone());
        self.state.roots_dirty = true;
        self.state.immediates_dirty = true;
    }

    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        self.state.set_bind_group(group, bind_group);
    }

    /// Set `data` at byte `offset` of the shader's `var<immediate>` block.
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.state.set_immediates(offset, data);
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64) {
        let (cmd, d) = self.cmd();
        // SAFETY: a live buffer.
        unsafe { d.cmd_bind_vertex_buffers(cmd, slot, &[buffer.0.raw], &[offset]) };
        self.enc.keep(buffer);
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64, format: t::IndexFormat) {
        let (cmd, d) = self.cmd();
        // SAFETY: a live buffer.
        unsafe { d.cmd_bind_index_buffer(cmd, buffer.0.raw, offset, conv::index_type(format)) };
        self.enc.keep(buffer);
        self.index = true;
    }

    /// As in wgpu: y down from the top of the target (Vulkan's viewport is flipped for it).
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        let (cmd, d) = self.cmd();
        let vp = [vk::Viewport {
            x,
            y: y + h,
            width: w,
            height: -h,
            min_depth,
            max_depth,
        }];
        // SAFETY: recording inside the pass.
        unsafe { d.cmd_set_viewport(cmd, 0, &vp) };
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let (cmd, d) = self.cmd();
        let r = [vk::Rect2D {
            offset: vk::Offset2D { x: x as i32, y: y as i32 },
            extent: vk::Extent2D { width: w, height: h },
        }];
        // SAFETY: recording inside the pass.
        unsafe { d.cmd_set_scissor(cmd, 0, &r) };
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        let (cmd, d) = self.cmd();
        // SAFETY: recording inside the pass.
        unsafe { d.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::FRONT_AND_BACK, reference) };
    }

    pub fn set_blend_constant(&mut self, c: t::Color) {
        let (cmd, d) = self.cmd();
        // SAFETY: recording inside the pass.
        unsafe { d.cmd_set_blend_constants(cmd, &[c.r as f32, c.g as f32, c.b as f32, c.a as f32]) };
    }

    pub(crate) fn reset_state(&mut self) {
        self.pipeline = None;
        self.index = false;
        self.state.groups = Default::default();
        self.state.roots_dirty = true;
        self.state.immediates_dirty = true;
    }

    fn flush(&mut self) {
        let pipeline = self.pipeline.clone().expect("draw without a pipeline");
        flush(self.enc, &pipeline.0, &mut self.state, &self.label);
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.flush();
        if vertices.is_empty() || instances.is_empty() {
            return;
        }
        let (cmd, d) = self.cmd();
        // SAFETY: a complete pipeline and bindings were set by `flush`.
        unsafe { d.cmd_draw(cmd, vertices.len() as u32, instances.len() as u32, vertices.start, instances.start) };
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.flush();
        assert!(self.index, "draw_indexed without an index buffer");
        if indices.is_empty() || instances.is_empty() {
            return;
        }
        let (cmd, d) = self.cmd();
        // SAFETY: as in `draw`; the index range is the caller's to keep inside the buffer.
        unsafe { d.cmd_draw_indexed(cmd, indices.len() as u32, instances.len() as u32, indices.start, base_vertex, instances.start) };
    }

    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        self.multi_draw_indirect(args, offset, 1);
    }

    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        self.multi_draw_indexed_indirect(args, offset, 1);
    }

    pub fn multi_draw_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        self.flush();
        let stride = std::mem::size_of::<t::DrawIndirectArgs>() as u64;
        let (cmd, d) = self.cmd();
        // SAFETY: `t::DrawIndirectArgs` is VkDrawIndirectCommand.
        unsafe {
            if count <= 1 || self.enc.shared.caps.multi_draw_indirect {
                d.cmd_draw_indirect(cmd, args.0.raw, offset, count, stride as u32);
            } else {
                for i in 0..count as u64 {
                    d.cmd_draw_indirect(cmd, args.0.raw, offset + i * stride, 1, stride as u32);
                }
            }
        }
        self.enc.keep(args);
    }

    pub fn multi_draw_indexed_indirect(&mut self, args: &Buffer, offset: u64, count: u32) {
        self.flush();
        assert!(self.index, "draw_indexed_indirect without an index buffer");
        let stride = std::mem::size_of::<t::DrawIndexedIndirectArgs>() as u64;
        let (cmd, d) = self.cmd();
        // SAFETY: `t::DrawIndexedIndirectArgs` is VkDrawIndexedIndirectCommand.
        unsafe {
            if count <= 1 || self.enc.shared.caps.multi_draw_indirect {
                d.cmd_draw_indexed_indirect(cmd, args.0.raw, offset, count, stride as u32);
            } else {
                for i in 0..count as u64 {
                    d.cmd_draw_indexed_indirect(cmd, args.0.raw, offset + i * stride, 1, stride as u32);
                }
            }
        }
        self.enc.keep(args);
    }

    /// Count the samples of the following draws into query `index` of the pass's
    /// occlusion query set.
    pub fn begin_occlusion_query(&mut self, index: u32) {
        let q = self.occlusion.as_ref().expect("begin_occlusion_query in a pass without an occlusion query set");
        assert!(index < q.count(), "occlusion query {index} past the set");
        let flags = if self.enc.shared.caps.precise_occlusion {
            vk::QueryControlFlags::PRECISE
        } else {
            vk::QueryControlFlags::empty()
        };
        let (cmd, d) = self.cmd();
        // SAFETY: the set was reset before the pass.
        unsafe { d.cmd_begin_query(cmd, q.0.raw, index, flags) };
        self.query_active = Some(index);
    }

    pub fn end_occlusion_query(&mut self) {
        let index = self.query_active.take().expect("end_occlusion_query without begin_occlusion_query");
        let pool = self.occlusion.as_ref().expect("occlusion query set").0.raw;
        let (cmd, d) = self.cmd();
        // SAFETY: the query was begun in this pass.
        unsafe { d.cmd_end_query(cmd, pool, index) };
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.enc.push_debug_group(label);
    }

    pub fn pop_debug_group(&mut self) {
        self.enc.pop_debug_group();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.enc.insert_debug_marker(label);
    }

    /// The `VkCommandBuffer` inside dynamic rendering, for interop.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.enc.raw
    }
}

/// A compute pass. Ends when dropped.
pub struct ComputePass<'e> {
    enc: &'e mut CommandEncoder,
    state: PassState,
    pipeline: Option<ComputePipeline>,
    end_timestamp: Option<(QuerySet, u32)>,
    labelled: bool,
    /// A dispatch was recorded: the next one waits for it (as wgpu orders dispatches).
    dispatched: bool,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        if self.labelled {
            self.enc.pop_debug_group();
        }
        if let Some((q, i)) = self.end_timestamp.take() {
            self.enc.timestamp(&q, i);
        }
        self.enc.dirty = true;
    }
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        bind_pipeline(self.enc, &pipeline.0);
        self.enc.keep(pipeline);
        self.pipeline = Some(pipeline.clone());
        self.state.roots_dirty = true;
        self.state.immediates_dirty = true;
    }

    pub fn set_bind_group(&mut self, group: u32, bind_group: &BindGroup) {
        self.state.set_bind_group(group, bind_group);
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.state.set_immediates(offset, data);
    }

    fn flush(&mut self) {
        let pipeline = self.pipeline.clone().expect("dispatch without a pipeline");
        if self.dispatched {
            self.enc.shared.full_barrier(self.enc.raw);
        }
        flush(self.enc, &pipeline.0, &mut self.state, &pipeline.0.label);
        self.dispatched = true;
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.flush();
        if x == 0 || y == 0 || z == 0 {
            return;
        }
        // SAFETY: a complete pipeline and bindings were set by `flush`.
        unsafe { self.enc.shared.raw.cmd_dispatch(self.enc.raw, x, y, z) };
    }

    pub fn dispatch_workgroups_indirect(&mut self, args: &Buffer, offset: u64) {
        self.flush();
        // SAFETY: `t::DispatchIndirectArgs` is VkDispatchIndirectCommand.
        unsafe { self.enc.shared.raw.cmd_dispatch_indirect(self.enc.raw, args.0.raw, offset) };
        self.enc.keep(args);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.enc.push_debug_group(label);
    }

    pub fn pop_debug_group(&mut self) {
        self.enc.pop_debug_group();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.enc.insert_debug_marker(label);
    }

    /// The `VkCommandBuffer`, for interop.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.enc.raw
    }
}
