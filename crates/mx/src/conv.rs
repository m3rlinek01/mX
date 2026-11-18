//! Descriptors that hold mX objects, as the backend's (each object picked for that backend).

use crate::*;

pub(crate) fn vertex<'a, M: 'a>(v: &t::VertexState<'a, ShaderModule>) -> t::VertexState<'a, M>
where
    ShaderModule: Pick<M>,
{
    t::VertexState {
        module: v.module.pick(),
        entry_point: v.entry_point,
        constants: v.constants,
        buffers: v.buffers,
    }
}

pub(crate) fn render_pipeline<'a, M: 'a>(d: &t::RenderPipelineDescriptor<'a, ShaderModule>) -> t::RenderPipelineDescriptor<'a, M>
where
    ShaderModule: Pick<M>,
{
    t::RenderPipelineDescriptor {
        label: d.label,
        vertex: vertex(&d.vertex),
        primitive: d.primitive,
        depth_stencil: d.depth_stencil,
        multisample: d.multisample,
        fragment: d.fragment.as_ref().map(|f| t::FragmentState {
            module: f.module.pick(),
            entry_point: f.entry_point,
            constants: f.constants,
            targets: f.targets,
        }),
    }
}

pub(crate) fn compute_pipeline<'a, M: 'a>(d: &t::ComputePipelineDescriptor<'a, ShaderModule>) -> t::ComputePipelineDescriptor<'a, M>
where
    ShaderModule: Pick<M>,
{
    t::ComputePipelineDescriptor {
        label: d.label,
        module: d.module.pick(),
        entry_point: d.entry_point,
        constants: d.constants,
    }
}

pub(crate) fn texture_info<'a, T: 'a>(i: &t::TexelCopyTextureInfo<'a, Texture>) -> t::TexelCopyTextureInfo<'a, T>
where
    Texture: Pick<T>,
{
    t::TexelCopyTextureInfo {
        texture: i.texture.pick(),
        mip_level: i.mip_level,
        origin: i.origin,
        aspect: i.aspect,
    }
}

pub(crate) fn buffer_info<'a, B: 'a>(i: &t::TexelCopyBufferInfo<'a, Buffer>) -> t::TexelCopyBufferInfo<'a, B>
where
    Buffer: Pick<B>,
{
    t::TexelCopyBufferInfo {
        buffer: i.buffer.pick(),
        layout: i.layout,
    }
}

pub(crate) fn timestamps<'a, Q: 'a>(w: &Option<t::PassTimestampWrites<'a, QuerySet>>) -> Option<t::PassTimestampWrites<'a, Q>>
where
    QuerySet: Pick<Q>,
{
    w.as_ref().map(|w| t::PassTimestampWrites {
        query_set: w.query_set.pick(),
        beginning_of_pass_write_index: w.beginning_of_pass_write_index,
        end_of_pass_write_index: w.end_of_pass_write_index,
    })
}

pub(crate) fn colors<'a, V: 'a>(c: &[Option<t::RenderPassColorAttachment<'a, TextureView>>]) -> Vec<Option<t::RenderPassColorAttachment<'a, V>>>
where
    TextureView: Pick<V>,
{
    c.iter()
        .map(|c| {
            c.as_ref().map(|c| t::RenderPassColorAttachment {
                view: c.view.pick(),
                resolve_target: c.resolve_target.map(|r| r.pick()),
                ops: c.ops,
            })
        })
        .collect()
}

pub(crate) fn render_pass<'a, V: 'a, Q: 'a>(d: &t::RenderPassDescriptor<'a, TextureView, QuerySet>, colors: &'a [Option<t::RenderPassColorAttachment<'a, V>>]) -> t::RenderPassDescriptor<'a, V, Q>
where
    TextureView: Pick<V>,
    QuerySet: Pick<Q>,
{
    t::RenderPassDescriptor {
        label: d.label,
        color_attachments: colors,
        depth_stencil_attachment: d.depth_stencil_attachment.as_ref().map(|a| t::RenderPassDepthStencilAttachment {
            view: a.view.pick(),
            depth_ops: a.depth_ops,
            stencil_ops: a.stencil_ops,
        }),
        timestamp_writes: timestamps(&d.timestamp_writes),
        occlusion_query_set: d.occlusion_query_set.map(|q| q.pick()),
    }
}

pub(crate) fn compute_pass<'a, Q: 'a>(d: &t::ComputePassDescriptor<'a, QuerySet>) -> t::ComputePassDescriptor<'a, Q>
where
    QuerySet: Pick<Q>,
{
    t::ComputePassDescriptor {
        label: d.label,
        timestamp_writes: timestamps(&d.timestamp_writes),
    }
}

pub(crate) fn blas_entry<'a, A: 'a, B: 'a>(e: &t::BlasBuildEntry<'a, Blas, Buffer>) -> t::BlasBuildEntry<'a, A, B>
where
    Blas: Pick<A>,
    Buffer: Pick<B>,
{
    t::BlasBuildEntry {
        blas: e.blas.pick(),
        geometry: e
            .geometry
            .iter()
            .map(|g| t::BlasTriangleGeometry {
                size: g.size,
                vertex_buffer: g.vertex_buffer.pick(),
                first_vertex: g.first_vertex,
                vertex_stride: g.vertex_stride,
                index_buffer: g.index_buffer.map(|b| b.pick()),
                first_index: g.first_index,
            })
            .collect(),
    }
}

pub(crate) fn index<'a, B: 'a>(i: Option<(&'a Buffer, u64, t::IndexFormat)>) -> Option<(&'a B, u64, t::IndexFormat)>
where
    Buffer: Pick<B>,
{
    i.map(|(b, o, f)| (b.pick(), o, f))
}
