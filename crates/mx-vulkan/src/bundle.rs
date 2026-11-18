//! Render bundles: draws recorded once on the CPU and replayed into render passes.

use std::ops::Range;
use std::sync::Arc;

use mx_types as t;

use crate::{BindGroup, Buffer, Device, RenderPass, RenderPipeline};

#[derive(Clone)]
pub(crate) enum Cmd {
    Pipeline(RenderPipeline),
    BindGroup(u32, BindGroup),
    BufferOffset(u32, u32, u64),
    Immediates(u32, Box<[u8]>),
    VertexBuffer(u32, Buffer, u64),
    IndexBuffer(Buffer, u64, t::IndexFormat),
    Draw(Range<u32>, Range<u32>),
    DrawIndexed(Range<u32>, i32, Range<u32>),
    DrawIndirect(Buffer, u64),
    DrawIndexedIndirect(Buffer, u64),
}

/// Records a [`RenderBundle`]: the drawing half of a render pass.
pub struct RenderBundleEncoder {
    cmds: Vec<Cmd>,
    _desc: t::RenderBundleDepthStencil,
}

#[derive(Clone)]
pub struct RenderBundle(pub(crate) Arc<Vec<Cmd>>);

impl Device {
    pub fn create_render_bundle_encoder(&self, desc: &t::RenderBundleEncoderDescriptor<'_>) -> RenderBundleEncoder {
        let ds = desc.depth_stencil.unwrap_or(t::RenderBundleDepthStencil {
            format: t::TextureFormat::Depth32Float,
            depth_read_only: true,
            stencil_read_only: true,
        });
        RenderBundleEncoder { cmds: Vec::new(), _desc: ds }
    }
}

impl RenderBundleEncoder {
    pub fn finish(self, _label: Option<&str>) -> RenderBundle {
        RenderBundle(Arc::new(self.cmds))
    }
    pub fn set_pipeline(&mut self, p: &RenderPipeline) {
        self.cmds.push(Cmd::Pipeline(p.clone()));
    }
    pub fn set_bind_group(&mut self, group: u32, g: &BindGroup) {
        self.cmds.push(Cmd::BindGroup(group, g.clone()));
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.cmds.push(Cmd::Immediates(offset, data.into()));
    }
    pub fn set_buffer_offset(&mut self, group: u32, binding: u32, offset: u64) {
        self.cmds.push(Cmd::BufferOffset(group, binding, offset));
    }
    pub fn set_vertex_buffer(&mut self, slot: u32, b: &Buffer, offset: u64) {
        self.cmds.push(Cmd::VertexBuffer(slot, b.clone(), offset));
    }
    pub fn set_index_buffer(&mut self, b: &Buffer, offset: u64, format: t::IndexFormat) {
        self.cmds.push(Cmd::IndexBuffer(b.clone(), offset, format));
    }
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.cmds.push(Cmd::Draw(vertices, instances));
    }
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.cmds.push(Cmd::DrawIndexed(indices, base_vertex, instances));
    }
    pub fn draw_indirect(&mut self, args: &Buffer, offset: u64) {
        self.cmds.push(Cmd::DrawIndirect(args.clone(), offset));
    }
    pub fn draw_indexed_indirect(&mut self, args: &Buffer, offset: u64) {
        self.cmds.push(Cmd::DrawIndexedIndirect(args.clone(), offset));
    }
}

impl RenderPass<'_> {
    /// Replay bundles. Each starts with no pipeline or bindings, and the pass has none
    /// afterwards (as in WebGPU).
    pub fn execute_bundles<'b>(&mut self, bundles: impl IntoIterator<Item = &'b RenderBundle>) {
        for b in bundles {
            self.reset_state();
            for c in b.0.iter() {
                match c {
                    Cmd::Pipeline(p) => self.set_pipeline(p),
                    Cmd::BindGroup(g, bg) => self.set_bind_group(*g, bg),
                    Cmd::BufferOffset(g, b, o) => self.set_buffer_offset(*g, *b, *o),
                    Cmd::Immediates(o, d) => self.set_immediates(*o, d),
                    Cmd::VertexBuffer(s, buf, o) => self.set_vertex_buffer(*s, buf, *o),
                    Cmd::IndexBuffer(buf, o, f) => self.set_index_buffer(buf, *o, *f),
                    Cmd::Draw(v, i) => self.draw(v.clone(), i.clone()),
                    Cmd::DrawIndexed(ix, bv, i) => self.draw_indexed(ix.clone(), *bv, i.clone()),
                    Cmd::DrawIndirect(buf, o) => self.draw_indirect(buf, *o),
                    Cmd::DrawIndexedIndirect(buf, o) => self.draw_indexed_indirect(buf, *o),
                }
            }
        }
        self.reset_state();
    }
}
