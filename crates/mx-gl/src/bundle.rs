//! Render bundles: commands recorded once and replayed into passes.

use std::ops::Range;
use std::sync::Arc;

use mx_types as t;

use crate::command::Cmd;
use crate::{BindGroup, Buffer, Device, RenderPipeline};

pub(crate) struct BundleInner {
    pub cmds: Vec<Cmd>,
}

#[derive(Clone)]
pub struct RenderBundle(pub(crate) Arc<BundleInner>);

pub struct RenderBundleEncoder {
    cmds: Vec<Cmd>,
}

impl Device {
    pub fn create_render_bundle_encoder(&self, _desc: &t::RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder {
        RenderBundleEncoder { cmds: Vec::new() }
    }
}

impl RenderBundleEncoder {
    pub fn finish(self, _label: Option<&str>) -> RenderBundle {
        RenderBundle(Arc::new(BundleInner { cmds: self.cmds }))
    }
    pub fn set_pipeline(&mut self, p: &RenderPipeline) {
        self.cmds.push(Cmd::SetRenderPipeline(p.clone()));
    }
    pub fn set_bind_group(&mut self, group: u32, g: &BindGroup) {
        self.cmds.push(Cmd::SetBindGroup(group, g.clone()));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.cmds.push(Cmd::SetImmediates(offset, data.to_vec()));
    }
    pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
        self.cmds.push(Cmd::SetBufferOffset(group, binding, offset));
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, b: &Buffer, offset: u64) {
        self.cmds.push(Cmd::SetVertexBuffer(slot, b.clone(), offset));
    }
    pub fn set_index_buffer(&mut self, b: &Buffer, offset: u64, format: t::IndexFormat) {
        self.cmds.push(Cmd::SetIndexBuffer(b.clone(), offset, format));
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.cmds.push(Cmd::Draw(vertices, instances));
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.cmds.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }
    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        self.cmds.push(Cmd::DrawIndirect(args.clone(), offset, false));
    }
    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        self.cmds.push(Cmd::DrawIndirect(args.clone(), offset, true));
    }
}
