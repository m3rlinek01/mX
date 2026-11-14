//! wgpu's `util` module, the parts code uses.

use std::ops::Range;

use crate::*;

/// What render passes and render bundle encoders both do.
pub trait RenderEncoder<'a> {
    fn set_bind_group(&mut self, index: u32, bind_group: Option<&'a BindGroup>, offsets: &[DynamicOffset]);
    fn set_pipeline(&mut self, pipeline: &'a RenderPipeline);
    fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat);
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>);
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress);
    fn draw_indexed_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress);
    fn set_immediates(&mut self, offset: u32, data: &[u8]);
}

impl<'a> RenderEncoder<'a> for RenderPass<'_> {
    fn set_bind_group(&mut self, index: u32, bind_group: Option<&'a BindGroup>, offsets: &[DynamicOffset]) {
        RenderPass::set_bind_group(self, index, bind_group, offsets)
    }
    fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        RenderPass::set_pipeline(self, pipeline)
    }
    fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        RenderPass::set_index_buffer(self, buffer_slice, index_format)
    }
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        RenderPass::set_vertex_buffer(self, slot, buffer_slice)
    }
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        RenderPass::draw(self, vertices, instances)
    }
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        RenderPass::draw_indexed(self, indices, base_vertex, instances)
    }
    fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        RenderPass::draw_indirect(self, indirect_buffer, indirect_offset)
    }
    fn draw_indexed_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        RenderPass::draw_indexed_indirect(self, indirect_buffer, indirect_offset)
    }
    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        RenderPass::set_immediates(self, offset, data)
    }
}

impl<'a> RenderEncoder<'a> for RenderBundleEncoder<'a> {
    fn set_bind_group(&mut self, index: u32, bind_group: Option<&'a BindGroup>, offsets: &[DynamicOffset]) {
        RenderBundleEncoder::set_bind_group(self, index, bind_group, offsets)
    }
    fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        RenderBundleEncoder::set_pipeline(self, pipeline)
    }
    fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        RenderBundleEncoder::set_index_buffer(self, buffer_slice, index_format)
    }
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        RenderBundleEncoder::set_vertex_buffer(self, slot, buffer_slice)
    }
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        RenderBundleEncoder::draw(self, vertices, instances)
    }
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        RenderBundleEncoder::draw_indexed(self, indices, base_vertex, instances)
    }
    fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        RenderBundleEncoder::draw_indirect(self, indirect_buffer, indirect_offset)
    }
    fn draw_indexed_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        RenderBundleEncoder::draw_indexed_indirect(self, indirect_buffer, indirect_offset)
    }
    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        RenderBundleEncoder::set_immediates(self, offset, data)
    }
}

#[derive(Clone, Debug)]
pub struct BufferInitDescriptor<'a> {
    pub label: Label<'a>,
    pub contents: &'a [u8],
    pub usage: BufferUsages,
}

pub trait DeviceExt {
    fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Buffer;
}

impl DeviceExt for Device {
    fn create_buffer_init(&self, desc: &BufferInitDescriptor<'_>) -> Buffer {
        let size = (desc.contents.len() as u64).next_multiple_of(COPY_BUFFER_ALIGNMENT).max(COPY_BUFFER_ALIGNMENT);
        let b = self.create_buffer(&BufferDescriptor {
            label: desc.label,
            size,
            usage: desc.usage,
            mapped_at_creation: true,
        });
        b.get_mapped_range_mut(..).slice(..desc.contents.len()).copy_from_slice(desc.contents);
        b.unmap();
        b
    }
}
