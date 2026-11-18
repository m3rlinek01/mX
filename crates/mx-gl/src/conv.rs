//! mX's types in GL's terms (the formats and states wgpu-hal's GL backend uses).

use mx_types as t;

/// How a texture format is made and uploaded in GL.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FormatDesc {
    pub internal: u32,
    /// The external format and type of uploads (`glTexSubImage*`), or the compressed
    /// internal format again for compressed formats.
    pub external: u32,
    pub data_type: u32,
}

pub(crate) fn format(f: t::TextureFormat) -> FormatDesc {
    use t::TextureFormat as F;
    let d = |internal, external, data_type| FormatDesc { internal, external, data_type };
    match f {
        F::R8Unorm => d(glow::R8, glow::RED, glow::UNSIGNED_BYTE),
        F::R8Snorm => d(glow::R8_SNORM, glow::RED, glow::BYTE),
        F::R8Uint => d(glow::R8UI, glow::RED_INTEGER, glow::UNSIGNED_BYTE),
        F::R16Float => d(glow::R16F, glow::RED, glow::HALF_FLOAT),
        F::R16Uint => d(glow::R16UI, glow::RED_INTEGER, glow::UNSIGNED_SHORT),
        F::R32Float => d(glow::R32F, glow::RED, glow::FLOAT),
        F::R32Uint => d(glow::R32UI, glow::RED_INTEGER, glow::UNSIGNED_INT),
        F::R32Sint => d(glow::R32I, glow::RED_INTEGER, glow::INT),
        F::Rg8Unorm => d(glow::RG8, glow::RG, glow::UNSIGNED_BYTE),
        F::Rg16Float => d(glow::RG16F, glow::RG, glow::HALF_FLOAT),
        F::Rg32Float => d(glow::RG32F, glow::RG, glow::FLOAT),
        F::Rgba8Unorm => d(glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE),
        F::Rgba8UnormSrgb => d(glow::SRGB8_ALPHA8, glow::RGBA, glow::UNSIGNED_BYTE),
        F::Rgba8Snorm => d(glow::RGBA8_SNORM, glow::RGBA, glow::BYTE),
        F::Rgba8Uint => d(glow::RGBA8UI, glow::RGBA_INTEGER, glow::UNSIGNED_BYTE),
        // (GL has no BGRA internal formats: stored as RGBA, uploads swizzled by `BGRA` where
        // the driver takes it; as wgpu-hal)
        F::Bgra8Unorm => d(glow::RGBA8, glow::BGRA, glow::UNSIGNED_BYTE),
        F::Bgra8UnormSrgb => d(glow::SRGB8_ALPHA8, glow::BGRA, glow::UNSIGNED_BYTE),
        F::Rgb10a2Unorm => d(glow::RGB10_A2, glow::RGBA, glow::UNSIGNED_INT_2_10_10_10_REV),
        F::Rg11b10Ufloat => d(glow::R11F_G11F_B10F, glow::RGB, glow::UNSIGNED_INT_10F_11F_11F_REV),
        F::Rgba16Float => d(glow::RGBA16F, glow::RGBA, glow::HALF_FLOAT),
        F::Rgba16Uint => d(glow::RGBA16UI, glow::RGBA_INTEGER, glow::UNSIGNED_SHORT),
        F::Rgba32Float => d(glow::RGBA32F, glow::RGBA, glow::FLOAT),
        F::Rgba32Uint => d(glow::RGBA32UI, glow::RGBA_INTEGER, glow::UNSIGNED_INT),
        F::Depth16Unorm => d(glow::DEPTH_COMPONENT16, glow::DEPTH_COMPONENT, glow::UNSIGNED_SHORT),
        F::Depth32Float => d(glow::DEPTH_COMPONENT32F, glow::DEPTH_COMPONENT, glow::FLOAT),
        F::Depth24PlusStencil8 => d(glow::DEPTH24_STENCIL8, glow::DEPTH_STENCIL, glow::UNSIGNED_INT_24_8),
        F::Depth32FloatStencil8 => d(glow::DEPTH32F_STENCIL8, glow::DEPTH_STENCIL, glow::FLOAT_32_UNSIGNED_INT_24_8_REV),
        F::Bc1RgbaUnorm => compressed(glow::COMPRESSED_RGBA_S3TC_DXT1_EXT),
        F::Bc1RgbaUnormSrgb => compressed(glow::COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT),
        F::Bc2RgbaUnorm => compressed(glow::COMPRESSED_RGBA_S3TC_DXT3_EXT),
        F::Bc2RgbaUnormSrgb => compressed(glow::COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT),
        F::Bc3RgbaUnorm => compressed(glow::COMPRESSED_RGBA_S3TC_DXT5_EXT),
        F::Bc3RgbaUnormSrgb => compressed(glow::COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT),
        F::Bc4RUnorm => compressed(glow::COMPRESSED_RED_RGTC1),
        F::Bc5RgUnorm => compressed(glow::COMPRESSED_RG_RGTC2),
        F::Bc6hRgbUfloat => compressed(glow::COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT),
        F::Bc7RgbaUnorm => compressed(glow::COMPRESSED_RGBA_BPTC_UNORM),
        F::Bc7RgbaUnormSrgb => compressed(glow::COMPRESSED_SRGB_ALPHA_BPTC_UNORM),
    }
}

fn compressed(f: u32) -> FormatDesc {
    FormatDesc {
        internal: f,
        external: f,
        data_type: 0,
    }
}

/// The GL texture target of a texture.
pub(crate) fn texture_target(dim: t::TextureDimension, layers: u32, samples: u32, cube_compatible: bool) -> u32 {
    match dim {
        t::TextureDimension::D1 | t::TextureDimension::D2 if samples > 1 => glow::TEXTURE_2D_MULTISAMPLE,
        t::TextureDimension::D1 | t::TextureDimension::D2 if cube_compatible && layers == 6 => glow::TEXTURE_CUBE_MAP,
        t::TextureDimension::D1 | t::TextureDimension::D2 if layers > 1 => glow::TEXTURE_2D_ARRAY,
        t::TextureDimension::D1 | t::TextureDimension::D2 => glow::TEXTURE_2D,
        t::TextureDimension::D3 => glow::TEXTURE_3D,
    }
}

pub(crate) fn compare(c: t::CompareFunction) -> u32 {
    use t::CompareFunction as C;
    match c {
        C::Never => glow::NEVER,
        C::Less => glow::LESS,
        C::Equal => glow::EQUAL,
        C::LessEqual => glow::LEQUAL,
        C::Greater => glow::GREATER,
        C::NotEqual => glow::NOTEQUAL,
        C::GreaterEqual => glow::GEQUAL,
        C::Always => glow::ALWAYS,
    }
}

pub(crate) fn stencil_op(o: t::StencilOperation) -> u32 {
    use t::StencilOperation as S;
    match o {
        S::Keep => glow::KEEP,
        S::Zero => glow::ZERO,
        S::Replace => glow::REPLACE,
        S::Invert => glow::INVERT,
        S::IncrementClamp => glow::INCR,
        S::DecrementClamp => glow::DECR,
        S::IncrementWrap => glow::INCR_WRAP,
        S::DecrementWrap => glow::DECR_WRAP,
    }
}

pub(crate) fn blend_factor(f: t::BlendFactor) -> u32 {
    use t::BlendFactor as B;
    match f {
        B::Zero => glow::ZERO,
        B::One => glow::ONE,
        B::Src => glow::SRC_COLOR,
        B::OneMinusSrc => glow::ONE_MINUS_SRC_COLOR,
        B::SrcAlpha => glow::SRC_ALPHA,
        B::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
        B::Dst => glow::DST_COLOR,
        B::OneMinusDst => glow::ONE_MINUS_DST_COLOR,
        B::DstAlpha => glow::DST_ALPHA,
        B::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
        B::SrcAlphaSaturated => glow::SRC_ALPHA_SATURATE,
        B::Constant => glow::CONSTANT_COLOR,
        B::OneMinusConstant => glow::ONE_MINUS_CONSTANT_COLOR,
    }
}

pub(crate) fn blend_op(o: t::BlendOperation) -> u32 {
    use t::BlendOperation as O;
    match o {
        O::Add => glow::FUNC_ADD,
        O::Subtract => glow::FUNC_SUBTRACT,
        O::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
        O::Min => glow::MIN,
        O::Max => glow::MAX,
    }
}

pub(crate) fn address(a: t::AddressMode) -> u32 {
    match a {
        t::AddressMode::ClampToEdge => glow::CLAMP_TO_EDGE,
        t::AddressMode::Repeat => glow::REPEAT,
        t::AddressMode::MirrorRepeat => glow::MIRRORED_REPEAT,
        t::AddressMode::ClampToBorder => glow::CLAMP_TO_BORDER,
    }
}

pub(crate) fn min_filter(min: t::FilterMode, mip: t::MipmapFilterMode) -> u32 {
    use t::{FilterMode as F, MipmapFilterMode as M};
    match (min, mip) {
        (F::Nearest, M::Nearest) => glow::NEAREST_MIPMAP_NEAREST,
        (F::Nearest, M::Linear) => glow::NEAREST_MIPMAP_LINEAR,
        (F::Linear, M::Nearest) => glow::LINEAR_MIPMAP_NEAREST,
        (F::Linear, M::Linear) => glow::LINEAR_MIPMAP_LINEAR,
    }
}

pub(crate) fn filter(f: t::FilterMode) -> u32 {
    match f {
        t::FilterMode::Nearest => glow::NEAREST,
        t::FilterMode::Linear => glow::LINEAR,
    }
}

pub(crate) fn topology(p: t::PrimitiveTopology) -> u32 {
    match p {
        t::PrimitiveTopology::PointList => glow::POINTS,
        t::PrimitiveTopology::LineList => glow::LINES,
        t::PrimitiveTopology::LineStrip => glow::LINE_STRIP,
        t::PrimitiveTopology::TriangleList => glow::TRIANGLES,
        t::PrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}

pub(crate) fn index_type(f: t::IndexFormat) -> (u32, u64) {
    match f {
        t::IndexFormat::Uint16 => (glow::UNSIGNED_SHORT, 2),
        t::IndexFormat::Uint32 => (glow::UNSIGNED_INT, 4),
    }
}

/// How a vertex attribute is read: components, type, normalised, integer (`glVertexAttribIPointer`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttribDesc {
    pub size: i32,
    pub ty: u32,
    pub normalized: bool,
    pub integer: bool,
}

pub(crate) fn vertex_format(f: t::VertexFormat) -> AttribDesc {
    use t::VertexFormat as V;
    let a = |size, ty, normalized, integer| AttribDesc { size, ty, normalized, integer };
    match f {
        V::Uint8 => a(1, glow::UNSIGNED_BYTE, false, true),
        V::Uint8x2 => a(2, glow::UNSIGNED_BYTE, false, true),
        V::Uint8x4 => a(4, glow::UNSIGNED_BYTE, false, true),
        V::Sint8 => a(1, glow::BYTE, false, true),
        V::Sint8x2 => a(2, glow::BYTE, false, true),
        V::Sint8x4 => a(4, glow::BYTE, false, true),
        V::Unorm8 => a(1, glow::UNSIGNED_BYTE, true, false),
        V::Unorm8x2 => a(2, glow::UNSIGNED_BYTE, true, false),
        V::Unorm8x4 => a(4, glow::UNSIGNED_BYTE, true, false),
        V::Snorm8 => a(1, glow::BYTE, true, false),
        V::Snorm8x2 => a(2, glow::BYTE, true, false),
        V::Snorm8x4 => a(4, glow::BYTE, true, false),
        // (GL's `GL_BGRA` size swizzles the components, desktop GL only; ES reads RGBA)
        V::Unorm8x4Bgra => a(glow::BGRA as i32, glow::UNSIGNED_BYTE, true, false),
        V::Uint16 => a(1, glow::UNSIGNED_SHORT, false, true),
        V::Uint16x2 => a(2, glow::UNSIGNED_SHORT, false, true),
        V::Uint16x4 => a(4, glow::UNSIGNED_SHORT, false, true),
        V::Sint16 => a(1, glow::SHORT, false, true),
        V::Sint16x2 => a(2, glow::SHORT, false, true),
        V::Sint16x4 => a(4, glow::SHORT, false, true),
        V::Unorm16 => a(1, glow::UNSIGNED_SHORT, true, false),
        V::Unorm16x2 => a(2, glow::UNSIGNED_SHORT, true, false),
        V::Unorm16x4 => a(4, glow::UNSIGNED_SHORT, true, false),
        V::Snorm16 => a(1, glow::SHORT, true, false),
        V::Snorm16x2 => a(2, glow::SHORT, true, false),
        V::Snorm16x4 => a(4, glow::SHORT, true, false),
        V::Float16 => a(1, glow::HALF_FLOAT, false, false),
        V::Float16x2 => a(2, glow::HALF_FLOAT, false, false),
        V::Float16x4 => a(4, glow::HALF_FLOAT, false, false),
        V::Unorm10_10_10_2 => a(4, glow::UNSIGNED_INT_2_10_10_10_REV, true, false),
        V::Float32 => a(1, glow::FLOAT, false, false),
        V::Float32x2 => a(2, glow::FLOAT, false, false),
        V::Float32x3 => a(3, glow::FLOAT, false, false),
        V::Float32x4 => a(4, glow::FLOAT, false, false),
        V::Uint32 => a(1, glow::UNSIGNED_INT, false, true),
        V::Uint32x2 => a(2, glow::UNSIGNED_INT, false, true),
        V::Uint32x3 => a(3, glow::UNSIGNED_INT, false, true),
        V::Uint32x4 => a(4, glow::UNSIGNED_INT, false, true),
        V::Sint32 => a(1, glow::INT, false, true),
        V::Sint32x2 => a(2, glow::INT, false, true),
        V::Sint32x3 => a(3, glow::INT, false, true),
        V::Sint32x4 => a(4, glow::INT, false, true),
    }
}
