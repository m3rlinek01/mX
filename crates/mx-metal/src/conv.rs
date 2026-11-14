//! mX types to Metal types.

use mx_types as t;
use objc2_metal::*;

pub fn pixel_format(f: t::TextureFormat) -> MTLPixelFormat {
    use MTLPixelFormat as M;
    use t::TextureFormat as F;
    match f {
        F::R8Unorm => M::R8Unorm,
        F::R8Snorm => M::R8Snorm,
        F::R8Uint => M::R8Uint,
        F::R16Float => M::R16Float,
        F::R16Uint => M::R16Uint,
        F::R32Float => M::R32Float,
        F::R32Uint => M::R32Uint,
        F::R32Sint => M::R32Sint,
        F::Rg8Unorm => M::RG8Unorm,
        F::Rg16Float => M::RG16Float,
        F::Rg32Float => M::RG32Float,
        F::Rgba8Unorm => M::RGBA8Unorm,
        F::Rgba8UnormSrgb => M::RGBA8Unorm_sRGB,
        F::Rgba8Snorm => M::RGBA8Snorm,
        F::Rgba8Uint => M::RGBA8Uint,
        F::Bgra8Unorm => M::BGRA8Unorm,
        F::Bgra8UnormSrgb => M::BGRA8Unorm_sRGB,
        F::Rgb10a2Unorm => M::RGB10A2Unorm,
        F::Rg11b10Ufloat => M::RG11B10Float,
        F::Rgba16Float => M::RGBA16Float,
        F::Rgba16Uint => M::RGBA16Uint,
        F::Rgba32Float => M::RGBA32Float,
        F::Rgba32Uint => M::RGBA32Uint,
        F::Depth16Unorm => M::Depth16Unorm,
        F::Depth32Float => M::Depth32Float,
        // (as wgpu on Apple GPUs, which have no 24-bit depth)
        F::Depth24PlusStencil8 | F::Depth32FloatStencil8 => M::Depth32Float_Stencil8,
        F::Bc1RgbaUnorm => M::BC1_RGBA,
        F::Bc1RgbaUnormSrgb => M::BC1_RGBA_sRGB,
        F::Bc2RgbaUnorm => M::BC2_RGBA,
        F::Bc2RgbaUnormSrgb => M::BC2_RGBA_sRGB,
        F::Bc3RgbaUnorm => M::BC3_RGBA,
        F::Bc3RgbaUnormSrgb => M::BC3_RGBA_sRGB,
        F::Bc4RUnorm => M::BC4_RUnorm,
        F::Bc5RgUnorm => M::BC5_RGUnorm,
        F::Bc6hRgbUfloat => M::BC6H_RGBUfloat,
        F::Bc7RgbaUnorm => M::BC7_RGBAUnorm,
        F::Bc7RgbaUnormSrgb => M::BC7_RGBAUnorm_sRGB,
    }
}

/// The format of a view of one aspect of a depth-stencil texture.
pub fn aspect_format(f: t::TextureFormat, aspect: t::TextureAspect) -> MTLPixelFormat {
    match (f, aspect) {
        (t::TextureFormat::Depth24PlusStencil8 | t::TextureFormat::Depth32FloatStencil8, t::TextureAspect::StencilOnly) => MTLPixelFormat::X32_Stencil8,
        (t::TextureFormat::Depth24PlusStencil8 | t::TextureFormat::Depth32FloatStencil8, t::TextureAspect::DepthOnly) => MTLPixelFormat::Depth32Float,
        _ => pixel_format(f),
    }
}

pub fn texture_type(dim: t::TextureViewDimension, samples: u32) -> MTLTextureType {
    use t::TextureViewDimension as D;
    match dim {
        D::D1 => MTLTextureType::Type1D,
        D::D2 if samples > 1 => MTLTextureType::Type2DMultisample,
        D::D2 => MTLTextureType::Type2D,
        D::D2Array => MTLTextureType::Type2DArray,
        D::Cube => MTLTextureType::TypeCube,
        D::CubeArray => MTLTextureType::TypeCubeArray,
        D::D3 => MTLTextureType::Type3D,
    }
}

/// The view dimension a texture gets when none is asked for (wgpu's rule).
pub fn default_view_dimension(desc_dim: t::TextureDimension, layers: u32) -> t::TextureViewDimension {
    match desc_dim {
        t::TextureDimension::D1 => t::TextureViewDimension::D1,
        t::TextureDimension::D2 if layers > 1 => t::TextureViewDimension::D2Array,
        t::TextureDimension::D2 => t::TextureViewDimension::D2,
        t::TextureDimension::D3 => t::TextureViewDimension::D3,
    }
}

pub fn texture_usage(u: t::TextureUsages, format: t::TextureFormat, view_formats: bool) -> MTLTextureUsage {
    let mut m = MTLTextureUsage::Unknown;
    if u.intersects(t::TextureUsages::TEXTURE_BINDING) {
        m |= MTLTextureUsage::ShaderRead;
    }
    if u.contains(t::TextureUsages::STORAGE_BINDING) {
        m |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
    }
    if u.contains(t::TextureUsages::RENDER_ATTACHMENT) {
        m |= MTLTextureUsage::RenderTarget;
    }
    // (views in another format, or of one aspect of depth-stencil)
    if view_formats || format.has_stencil() {
        m |= MTLTextureUsage::PixelFormatView;
    }
    m
}

pub fn compare(f: t::CompareFunction) -> MTLCompareFunction {
    use t::CompareFunction as C;
    match f {
        C::Never => MTLCompareFunction::Never,
        C::Less => MTLCompareFunction::Less,
        C::Equal => MTLCompareFunction::Equal,
        C::LessEqual => MTLCompareFunction::LessEqual,
        C::Greater => MTLCompareFunction::Greater,
        C::NotEqual => MTLCompareFunction::NotEqual,
        C::GreaterEqual => MTLCompareFunction::GreaterEqual,
        C::Always => MTLCompareFunction::Always,
    }
}

pub fn filter(f: t::FilterMode) -> MTLSamplerMinMagFilter {
    match f {
        t::FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        t::FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

pub fn mip_filter(f: t::MipmapFilterMode) -> MTLSamplerMipFilter {
    match f {
        t::MipmapFilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        t::MipmapFilterMode::Linear => MTLSamplerMipFilter::Linear,
    }
}

pub fn address(a: t::AddressMode) -> MTLSamplerAddressMode {
    match a {
        t::AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        t::AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        t::AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        t::AddressMode::ClampToBorder => MTLSamplerAddressMode::ClampToBorderColor,
    }
}

pub fn border(c: t::SamplerBorderColor) -> MTLSamplerBorderColor {
    match c {
        t::SamplerBorderColor::TransparentBlack => MTLSamplerBorderColor::TransparentBlack,
        t::SamplerBorderColor::OpaqueBlack => MTLSamplerBorderColor::OpaqueBlack,
        t::SamplerBorderColor::OpaqueWhite => MTLSamplerBorderColor::OpaqueWhite,
    }
}

pub fn primitive_type(p: t::PrimitiveTopology) -> MTLPrimitiveType {
    use t::PrimitiveTopology as P;
    match p {
        P::PointList => MTLPrimitiveType::Point,
        P::LineList => MTLPrimitiveType::Line,
        P::LineStrip => MTLPrimitiveType::LineStrip,
        P::TriangleList => MTLPrimitiveType::Triangle,
        P::TriangleStrip => MTLPrimitiveType::TriangleStrip,
    }
}

pub fn topology_class(p: t::PrimitiveTopology) -> MTLPrimitiveTopologyClass {
    use t::PrimitiveTopology as P;
    match p {
        P::PointList => MTLPrimitiveTopologyClass::Point,
        P::LineList | P::LineStrip => MTLPrimitiveTopologyClass::Line,
        P::TriangleList | P::TriangleStrip => MTLPrimitiveTopologyClass::Triangle,
    }
}

pub fn write_mask(w: t::ColorWrites) -> MTLColorWriteMask {
    let mut m = MTLColorWriteMask::None;
    if w.contains(t::ColorWrites::RED) {
        m |= MTLColorWriteMask::Red;
    }
    if w.contains(t::ColorWrites::GREEN) {
        m |= MTLColorWriteMask::Green;
    }
    if w.contains(t::ColorWrites::BLUE) {
        m |= MTLColorWriteMask::Blue;
    }
    if w.contains(t::ColorWrites::ALPHA) {
        m |= MTLColorWriteMask::Alpha;
    }
    m
}

pub fn blend_factor(f: t::BlendFactor) -> MTLBlendFactor {
    use t::BlendFactor as B;
    match f {
        B::Zero => MTLBlendFactor::Zero,
        B::One => MTLBlendFactor::One,
        B::Src => MTLBlendFactor::SourceColor,
        B::OneMinusSrc => MTLBlendFactor::OneMinusSourceColor,
        B::SrcAlpha => MTLBlendFactor::SourceAlpha,
        B::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        B::Dst => MTLBlendFactor::DestinationColor,
        B::OneMinusDst => MTLBlendFactor::OneMinusDestinationColor,
        B::DstAlpha => MTLBlendFactor::DestinationAlpha,
        B::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        B::SrcAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
        B::Constant => MTLBlendFactor::BlendColor,
        B::OneMinusConstant => MTLBlendFactor::OneMinusBlendColor,
    }
}

pub fn blend_op(o: t::BlendOperation) -> MTLBlendOperation {
    use t::BlendOperation as O;
    match o {
        O::Add => MTLBlendOperation::Add,
        O::Subtract => MTLBlendOperation::Subtract,
        O::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        O::Min => MTLBlendOperation::Min,
        O::Max => MTLBlendOperation::Max,
    }
}

/// The format naga's vertex pulling reads (the names are wgpu's, and mX's).
pub fn naga_vertex_format(f: t::VertexFormat) -> mx_shader::msl::VertexFormat {
    use mx_shader::msl::VertexFormat as N;
    use t::VertexFormat as V;
    match f {
        V::Uint8 => N::Uint8,
        V::Uint8x2 => N::Uint8x2,
        V::Uint8x4 => N::Uint8x4,
        V::Unorm8x4 => N::Unorm8x4,
        V::Snorm8x4 => N::Snorm8x4,
        V::Sint8 => N::Sint8,
        V::Sint8x2 => N::Sint8x2,
        V::Sint8x4 => N::Sint8x4,
        V::Unorm8 => N::Unorm8,
        V::Unorm8x2 => N::Unorm8x2,
        V::Snorm8 => N::Snorm8,
        V::Snorm8x2 => N::Snorm8x2,
        V::Unorm8x4Bgra => N::Unorm8x4Bgra,
        V::Uint16 => N::Uint16,
        V::Sint16 => N::Sint16,
        V::Sint16x2 => N::Sint16x2,
        V::Sint16x4 => N::Sint16x4,
        V::Unorm16 => N::Unorm16,
        V::Snorm16 => N::Snorm16,
        V::Float16 => N::Float16,
        V::Unorm10_10_10_2 => N::Unorm10_10_10_2,
        V::Uint16x2 => N::Uint16x2,
        V::Uint16x4 => N::Uint16x4,
        V::Unorm16x2 => N::Unorm16x2,
        V::Unorm16x4 => N::Unorm16x4,
        V::Snorm16x2 => N::Snorm16x2,
        V::Snorm16x4 => N::Snorm16x4,
        V::Float16x2 => N::Float16x2,
        V::Float16x4 => N::Float16x4,
        V::Float32 => N::Float32,
        V::Float32x2 => N::Float32x2,
        V::Float32x3 => N::Float32x3,
        V::Float32x4 => N::Float32x4,
        V::Uint32 => N::Uint32,
        V::Uint32x2 => N::Uint32x2,
        V::Uint32x3 => N::Uint32x3,
        V::Uint32x4 => N::Uint32x4,
        V::Sint32 => N::Sint32,
        V::Sint32x2 => N::Sint32x2,
        V::Sint32x3 => N::Sint32x3,
        V::Sint32x4 => N::Sint32x4,
    }
}

pub fn index_type(f: t::IndexFormat) -> MTLIndexType {
    match f {
        t::IndexFormat::Uint16 => MTLIndexType::UInt16,
        t::IndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}

pub fn stencil_op(o: t::StencilOperation) -> MTLStencilOperation {
    use t::StencilOperation as S;
    match o {
        S::Keep => MTLStencilOperation::Keep,
        S::Zero => MTLStencilOperation::Zero,
        S::Replace => MTLStencilOperation::Replace,
        S::Invert => MTLStencilOperation::Invert,
        S::IncrementClamp => MTLStencilOperation::IncrementClamp,
        S::DecrementClamp => MTLStencilOperation::DecrementClamp,
        S::IncrementWrap => MTLStencilOperation::IncrementWrap,
        S::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

pub fn winding(f: t::FrontFace) -> MTLWinding {
    match f {
        t::FrontFace::Ccw => MTLWinding::CounterClockwise,
        t::FrontFace::Cw => MTLWinding::Clockwise,
    }
}

pub fn cull(c: Option<t::Face>) -> MTLCullMode {
    match c {
        None => MTLCullMode::None,
        Some(t::Face::Front) => MTLCullMode::Front,
        Some(t::Face::Back) => MTLCullMode::Back,
    }
}

pub fn load_action<V>(l: &t::LoadOp<V>) -> MTLLoadAction {
    match l {
        t::LoadOp::Clear(_) => MTLLoadAction::Clear,
        t::LoadOp::Load => MTLLoadAction::Load,
        t::LoadOp::DontCare => MTLLoadAction::DontCare,
    }
}

pub fn store_action(store: t::StoreOp, resolve: bool) -> MTLStoreAction {
    match (store, resolve) {
        (t::StoreOp::Store, false) => MTLStoreAction::Store,
        (t::StoreOp::Discard, false) => MTLStoreAction::DontCare,
        (t::StoreOp::Store, true) => MTLStoreAction::StoreAndMultisampleResolve,
        (t::StoreOp::Discard, true) => MTLStoreAction::MultisampleResolve,
    }
}

pub fn origin(o: t::Origin3d) -> MTLOrigin {
    MTLOrigin {
        x: o.x as usize,
        y: o.y as usize,
        z: o.z as usize,
    }
}

pub fn size(e: t::Extent3d) -> MTLSize {
    MTLSize {
        width: e.width as usize,
        height: e.height as usize,
        depth: e.depth_or_array_layers as usize,
    }
}
