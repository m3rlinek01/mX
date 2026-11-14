//! wgpu's plain data (wgpu-types) to mX's.

use mx_types as m;
use wgpu_types as w;

macro_rules! same_enum {
    ($fn:ident: $t:ident { $($v:ident),* $(,)? }) => {
        pub fn $fn(x: w::$t) -> m::$t {
            #[allow(unreachable_patterns)]
            match x {
                $(w::$t::$v => m::$t::$v,)*
                other => panic!("mX has no {other:?}"),
            }
        }
    };
}

macro_rules! same_flags {
    ($fn:ident: $t:ident { $($f:ident),* $(,)? }) => {
        pub fn $fn(x: w::$t) -> m::$t {
            let mut o = m::$t::empty();
            $(if x.contains(w::$t::$f) { o |= m::$t::$f; })*
            o
        }
    };
}

same_enum!(index_format: IndexFormat { Uint16, Uint32 });
same_enum!(texture_dimension: TextureDimension { D1, D2, D3 });
same_enum!(view_dimension: TextureViewDimension { D1, D2, D2Array, Cube, CubeArray, D3 });
same_enum!(aspect: TextureAspect { All, StencilOnly, DepthOnly });
same_enum!(address_mode: AddressMode { ClampToEdge, Repeat, MirrorRepeat, ClampToBorder });
same_enum!(filter_mode: FilterMode { Nearest, Linear });
same_enum!(mipmap_filter: MipmapFilterMode { Nearest, Linear });
same_enum!(compare: CompareFunction { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always });
same_enum!(border_color: SamplerBorderColor { TransparentBlack, OpaqueBlack, OpaqueWhite });
same_enum!(step_mode: VertexStepMode { Vertex, Instance });
same_enum!(topology: PrimitiveTopology { PointList, LineList, LineStrip, TriangleList, TriangleStrip });
same_enum!(front_face: FrontFace { Ccw, Cw });
same_enum!(face: Face { Front, Back });
same_enum!(stencil_op: StencilOperation { Keep, Zero, Replace, Invert, IncrementClamp, DecrementClamp, IncrementWrap, DecrementWrap });
same_enum!(blend_factor: BlendFactor { Zero, One, Src, OneMinusSrc, SrcAlpha, OneMinusSrcAlpha, Dst, OneMinusDst, DstAlpha, OneMinusDstAlpha, SrcAlphaSaturated, Constant, OneMinusConstant });
same_enum!(blend_op: BlendOperation { Add, Subtract, ReverseSubtract, Min, Max });
same_enum!(present_mode: PresentMode { AutoVsync, AutoNoVsync, Fifo, FifoRelaxed, Immediate, Mailbox });
same_enum!(alpha_mode: CompositeAlphaMode { Auto, Opaque, PreMultiplied, PostMultiplied, Inherit });
same_enum!(vertex_format: VertexFormat {
    Uint8, Uint8x2, Uint8x4, Unorm8x4, Snorm8x4, Sint8, Sint8x2, Sint8x4, Unorm8, Unorm8x2, Snorm8, Snorm8x2, Unorm8x4Bgra, Uint16, Sint16, Sint16x2,
    Sint16x4, Unorm16, Snorm16, Float16, Unorm10_10_10_2, Uint16x2, Uint16x4, Unorm16x2, Unorm16x4, Snorm16x2, Snorm16x4, Float16x2, Float16x4, Float32,
    Float32x2, Float32x3, Float32x4, Uint32, Uint32x2, Uint32x3, Uint32x4, Sint32, Sint32x2, Sint32x3, Sint32x4,
});
same_enum!(format: TextureFormat {
    R8Unorm, R8Snorm, R8Uint, R16Float, R16Uint, R32Float, R32Uint, R32Sint, Rg8Unorm, Rg16Float, Rg32Float, Rgba8Unorm, Rgba8UnormSrgb, Rgba8Snorm,
    Rgba8Uint, Bgra8Unorm, Bgra8UnormSrgb, Rgb10a2Unorm, Rg11b10Ufloat, Rgba16Float, Rgba16Uint, Rgba32Float, Rgba32Uint, Depth16Unorm, Depth32Float,
    Depth24PlusStencil8, Depth32FloatStencil8, Bc1RgbaUnorm, Bc1RgbaUnormSrgb, Bc2RgbaUnorm, Bc2RgbaUnormSrgb, Bc3RgbaUnorm, Bc3RgbaUnormSrgb,
    Bc4RUnorm, Bc5RgUnorm, Bc6hRgbUfloat, Bc7RgbaUnorm, Bc7RgbaUnormSrgb,
});

pub fn format_back(x: m::TextureFormat) -> w::TextureFormat {
    use m::TextureFormat as M;
    use w::TextureFormat as W;
    match x {
        M::Bgra8UnormSrgb => W::Bgra8UnormSrgb,
        M::Bgra8Unorm => W::Bgra8Unorm,
        M::Rgba8UnormSrgb => W::Rgba8UnormSrgb,
        M::Rgba8Unorm => W::Rgba8Unorm,
        M::Rgba16Float => W::Rgba16Float,
        M::Rgb10a2Unorm => W::Rgb10a2Unorm,
        other => panic!("{other:?} as a surface format"),
    }
}

pub fn present_mode_back(x: m::PresentMode) -> w::PresentMode {
    use m::PresentMode as M;
    use w::PresentMode as W;
    match x {
        M::AutoVsync => W::AutoVsync,
        M::AutoNoVsync => W::AutoNoVsync,
        M::Fifo => W::Fifo,
        M::FifoRelaxed => W::FifoRelaxed,
        M::Immediate => W::Immediate,
        M::Mailbox => W::Mailbox,
    }
}

pub fn alpha_mode_back(x: m::CompositeAlphaMode) -> w::CompositeAlphaMode {
    use m::CompositeAlphaMode as M;
    use w::CompositeAlphaMode as W;
    match x {
        M::Auto => W::Auto,
        M::Opaque => W::Opaque,
        M::PreMultiplied => W::PreMultiplied,
        M::PostMultiplied => W::PostMultiplied,
        M::Inherit => W::Inherit,
    }
}

same_flags!(buffer_usages: BufferUsages { MAP_READ, MAP_WRITE, COPY_SRC, COPY_DST, INDEX, VERTEX, UNIFORM, STORAGE, INDIRECT, QUERY_RESOLVE, BLAS_INPUT });
same_flags!(texture_usages: TextureUsages { COPY_SRC, COPY_DST, TEXTURE_BINDING, STORAGE_BINDING, RENDER_ATTACHMENT });
same_flags!(color_writes: ColorWrites { RED, GREEN, BLUE, ALPHA });

pub fn texture_usages_back(x: m::TextureUsages) -> w::TextureUsages {
    let mut o = w::TextureUsages::empty();
    for (a, b) in [
        (m::TextureUsages::COPY_SRC, w::TextureUsages::COPY_SRC),
        (m::TextureUsages::COPY_DST, w::TextureUsages::COPY_DST),
        (m::TextureUsages::TEXTURE_BINDING, w::TextureUsages::TEXTURE_BINDING),
        (m::TextureUsages::STORAGE_BINDING, w::TextureUsages::STORAGE_BINDING),
        (m::TextureUsages::RENDER_ATTACHMENT, w::TextureUsages::RENDER_ATTACHMENT),
    ] {
        if x.contains(a) {
            o |= b;
        }
    }
    o
}

pub fn extent(x: w::Extent3d) -> m::Extent3d {
    m::Extent3d {
        width: x.width,
        height: x.height,
        depth_or_array_layers: x.depth_or_array_layers,
    }
}

pub fn origin(x: w::Origin3d) -> m::Origin3d {
    m::Origin3d { x: x.x, y: x.y, z: x.z }
}

pub fn color(x: w::Color) -> m::Color {
    m::Color { r: x.r, g: x.g, b: x.b, a: x.a }
}

pub fn layout(x: w::TexelCopyBufferLayout) -> m::TexelCopyBufferLayout {
    m::TexelCopyBufferLayout {
        offset: x.offset,
        bytes_per_row: x.bytes_per_row,
        rows_per_image: x.rows_per_image,
    }
}

pub fn ops<V: Copy, U>(x: w::Operations<V>, v: impl Fn(V) -> U) -> m::Operations<U> {
    m::Operations {
        load: match x.load {
            w::LoadOp::Clear(c) => m::LoadOp::Clear(v(c)),
            w::LoadOp::Load => m::LoadOp::Load,
            w::LoadOp::DontCare(_) => m::LoadOp::DontCare,
        },
        store: match x.store {
            w::StoreOp::Store => m::StoreOp::Store,
            w::StoreOp::Discard => m::StoreOp::Discard,
        },
    }
}

pub fn blend(x: w::BlendState) -> m::BlendState {
    let c = |c: w::BlendComponent| m::BlendComponent {
        src_factor: blend_factor(c.src_factor),
        dst_factor: blend_factor(c.dst_factor),
        operation: blend_op(c.operation),
    };
    m::BlendState { color: c(x.color), alpha: c(x.alpha) }
}

pub fn target(x: &w::ColorTargetState) -> m::ColorTargetState {
    m::ColorTargetState {
        format: format(x.format),
        blend: x.blend.map(blend),
        write_mask: color_writes(x.write_mask),
    }
}

pub fn primitive(x: w::PrimitiveState) -> m::PrimitiveState {
    m::PrimitiveState {
        topology: topology(x.topology),
        strip_index_format: x.strip_index_format.map(index_format),
        front_face: front_face(x.front_face),
        cull_mode: x.cull_mode.map(face),
        unclipped_depth: x.unclipped_depth,
    }
}

fn stencil_face(x: w::StencilFaceState) -> m::StencilFaceState {
    m::StencilFaceState {
        compare: compare(x.compare),
        fail_op: stencil_op(x.fail_op),
        depth_fail_op: stencil_op(x.depth_fail_op),
        pass_op: stencil_op(x.pass_op),
    }
}

pub fn depth_stencil(x: &w::DepthStencilState) -> m::DepthStencilState {
    m::DepthStencilState {
        format: format(x.format),
        depth_write_enabled: x.depth_write_enabled.unwrap_or(false),
        depth_compare: compare(x.depth_compare.unwrap_or(w::CompareFunction::Always)),
        stencil: m::StencilState {
            front: stencil_face(x.stencil.front),
            back: stencil_face(x.stencil.back),
            read_mask: x.stencil.read_mask,
            write_mask: x.stencil.write_mask,
        },
        bias: m::DepthBiasState {
            constant: x.bias.constant,
            slope_scale: x.bias.slope_scale,
            clamp: x.bias.clamp,
        },
    }
}

pub fn multisample(x: w::MultisampleState) -> m::MultisampleState {
    m::MultisampleState {
        count: x.count,
        mask: x.mask,
        alpha_to_coverage_enabled: x.alpha_to_coverage_enabled,
    }
}

pub fn sampler(x: &w::SamplerDescriptor<Option<&str>>) -> m::SamplerDescriptor<'static> {
    m::SamplerDescriptor {
        label: None,
        address_mode_u: address_mode(x.address_mode_u),
        address_mode_v: address_mode(x.address_mode_v),
        address_mode_w: address_mode(x.address_mode_w),
        mag_filter: filter_mode(x.mag_filter),
        min_filter: filter_mode(x.min_filter),
        mipmap_filter: mipmap_filter(x.mipmap_filter),
        lod_min_clamp: x.lod_min_clamp,
        lod_max_clamp: x.lod_max_clamp,
        compare: x.compare.map(compare),
        anisotropy_clamp: x.anisotropy_clamp,
        border_color: x.border_color.map(border_color),
    }
}

pub fn view(x: &w::TextureViewDescriptor<Option<&str>>) -> m::TextureViewDescriptor<'static> {
    m::TextureViewDescriptor {
        label: None,
        format: x.format.map(format),
        dimension: x.dimension.map(view_dimension),
        aspect: aspect(x.aspect),
        base_mip_level: x.base_mip_level,
        mip_level_count: x.mip_level_count,
        base_array_layer: x.base_array_layer,
        array_layer_count: x.array_layer_count,
    }
}

pub fn query_type(x: w::QueryType) -> m::QueryType {
    match x {
        w::QueryType::Occlusion => m::QueryType::Occlusion,
        w::QueryType::Timestamp => m::QueryType::Timestamp,
        other => panic!("mX has no {other:?} queries"),
    }
}

pub fn blas_size(d: &w::BlasTriangleGeometrySizeDescriptor) -> m::BlasTriangleGeometrySizeDescriptor {
    m::BlasTriangleGeometrySizeDescriptor {
        vertex_format: vertex_format(d.vertex_format),
        vertex_count: d.vertex_count,
        index_format: d.index_format.map(index_format),
        index_count: d.index_count,
        flags: m::AccelerationStructureGeometryFlags::from_bits_truncate(d.flags.bits() as u32),
    }
}

/// (the flags mX has keep their bits; vertex return and transforms have no counterpart)
pub fn as_flags(f: w::AccelerationStructureFlags) -> m::AccelerationStructureFlags {
    m::AccelerationStructureFlags::from_bits_truncate(f.bits() as u32)
}

pub fn as_update_mode(u: w::AccelerationStructureUpdateMode) -> m::AccelerationStructureUpdateMode {
    match u {
        w::AccelerationStructureUpdateMode::Build => m::AccelerationStructureUpdateMode::Build,
        w::AccelerationStructureUpdateMode::PreferUpdate => m::AccelerationStructureUpdateMode::PreferUpdate,
    }
}
