//! mX types to Direct3D 11 and DXGI types (the formats are DXGI's, as on Direct3D 12).

use mx_types as t;
use windows::Win32::Graphics::Direct3D as d3d;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub fn format(f: t::TextureFormat) -> DXGI_FORMAT {
    use t::TextureFormat as F;
    match f {
        F::R8Unorm => DXGI_FORMAT_R8_UNORM,
        F::R8Snorm => DXGI_FORMAT_R8_SNORM,
        F::R8Uint => DXGI_FORMAT_R8_UINT,
        F::R16Float => DXGI_FORMAT_R16_FLOAT,
        F::R16Uint => DXGI_FORMAT_R16_UINT,
        F::R32Float => DXGI_FORMAT_R32_FLOAT,
        F::R32Uint => DXGI_FORMAT_R32_UINT,
        F::R32Sint => DXGI_FORMAT_R32_SINT,
        F::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        F::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        F::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        F::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        F::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        F::Rgba8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
        F::Rgba8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
        F::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        F::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        F::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        F::Rg11b10Ufloat => DXGI_FORMAT_R11G11B10_FLOAT,
        F::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        F::Rgba16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
        F::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        F::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        F::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        F::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        F::Depth24PlusStencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        F::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        F::Bc1RgbaUnorm => DXGI_FORMAT_BC1_UNORM,
        F::Bc1RgbaUnormSrgb => DXGI_FORMAT_BC1_UNORM_SRGB,
        F::Bc2RgbaUnorm => DXGI_FORMAT_BC2_UNORM,
        F::Bc2RgbaUnormSrgb => DXGI_FORMAT_BC2_UNORM_SRGB,
        F::Bc3RgbaUnorm => DXGI_FORMAT_BC3_UNORM,
        F::Bc3RgbaUnormSrgb => DXGI_FORMAT_BC3_UNORM_SRGB,
        F::Bc4RUnorm => DXGI_FORMAT_BC4_UNORM,
        F::Bc5RgUnorm => DXGI_FORMAT_BC5_UNORM,
        F::Bc6hRgbUfloat => DXGI_FORMAT_BC6H_UF16,
        F::Bc7RgbaUnorm => DXGI_FORMAT_BC7_UNORM,
        F::Bc7RgbaUnormSrgb => DXGI_FORMAT_BC7_UNORM_SRGB,
    }
}

/// The format without sRGB (swapchain buffers, which flip-model swapchains keep linear).
pub fn format_nosrgb(f: t::TextureFormat) -> DXGI_FORMAT {
    format(f.remove_srgb_suffix())
}

/// The format a texture's resource is created with: typeless when views may reinterpret it,
/// or when a depth format is also read by shaders (as wgpu-hal without format casting).
pub fn resource_format(f: t::TextureFormat, usage: t::TextureUsages, has_view_formats: bool) -> DXGI_FORMAT {
    use t::TextureFormat as F;
    if has_view_formats {
        match f {
            F::Rgba8Unorm | F::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_TYPELESS,
            F::Bgra8Unorm | F::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_TYPELESS,
            F::Bc1RgbaUnorm | F::Bc1RgbaUnormSrgb => DXGI_FORMAT_BC1_TYPELESS,
            F::Bc2RgbaUnorm | F::Bc2RgbaUnormSrgb => DXGI_FORMAT_BC2_TYPELESS,
            F::Bc3RgbaUnorm | F::Bc3RgbaUnormSrgb => DXGI_FORMAT_BC3_TYPELESS,
            F::Bc7RgbaUnorm | F::Bc7RgbaUnormSrgb => DXGI_FORMAT_BC7_TYPELESS,
            other => format(other),
        }
    } else if f.has_depth() && usage.intersects(t::TextureUsages::TEXTURE_BINDING | t::TextureUsages::STORAGE_BINDING) {
        match f {
            F::Depth16Unorm => DXGI_FORMAT_R16_TYPELESS,
            F::Depth32Float => DXGI_FORMAT_R32_TYPELESS,
            F::Depth32FloatStencil8 => DXGI_FORMAT_R32G8X24_TYPELESS,
            _ => DXGI_FORMAT_R24G8_TYPELESS,
        }
    } else {
        format(f)
    }
}

/// The format of a shader view of `aspect` (`None`: not viewable, a both-aspects view of a
/// depth-stencil texture).
pub fn srv_format(f: t::TextureFormat, aspect: t::TextureAspect) -> Option<DXGI_FORMAT> {
    use t::TextureAspect as A;
    use t::TextureFormat as F;
    Some(match (f, aspect) {
        (F::Depth16Unorm, A::All | A::DepthOnly) => DXGI_FORMAT_R16_UNORM,
        (F::Depth32Float, A::All | A::DepthOnly) => DXGI_FORMAT_R32_FLOAT,
        (F::Depth32FloatStencil8, A::DepthOnly) => DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        (F::Depth24PlusStencil8, A::DepthOnly) => DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
        (F::Depth32FloatStencil8, A::StencilOnly) => DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
        (F::Depth24PlusStencil8, A::StencilOnly) => DXGI_FORMAT_X24_TYPELESS_G8_UINT,
        (F::Depth24PlusStencil8 | F::Depth32FloatStencil8, A::All) => return None,
        (other, _) => format(other),
    })
}

/// The format of a buffer footprint for copies of one aspect.
pub fn copy_format(f: t::TextureFormat, aspect: t::TextureAspect) -> DXGI_FORMAT {
    use t::TextureAspect as A;
    use t::TextureFormat as F;
    match (f, aspect) {
        (F::Depth16Unorm, _) => DXGI_FORMAT_R16_UNORM,
        (F::Depth32Float, _) | (F::Depth32FloatStencil8, A::DepthOnly) => DXGI_FORMAT_R32_FLOAT,
        (F::Depth24PlusStencil8 | F::Depth32FloatStencil8, A::StencilOnly) => DXGI_FORMAT_R8_UINT,
        (F::Depth24PlusStencil8 | F::Depth32FloatStencil8, _) => panic!("copies of a depth-stencil texture take one aspect (DepthOnly or StencilOnly)"),
        (other, _) => format(other),
    }
}

pub fn vertex_format(f: t::VertexFormat) -> DXGI_FORMAT {
    use t::VertexFormat as V;
    match f {
        V::Uint8 => DXGI_FORMAT_R8_UINT,
        V::Sint8 => DXGI_FORMAT_R8_SINT,
        V::Sint8x2 => DXGI_FORMAT_R8G8_SINT,
        V::Sint8x4 => DXGI_FORMAT_R8G8B8A8_SINT,
        V::Unorm8 => DXGI_FORMAT_R8_UNORM,
        V::Unorm8x2 => DXGI_FORMAT_R8G8_UNORM,
        V::Snorm8 => DXGI_FORMAT_R8_SNORM,
        V::Snorm8x2 => DXGI_FORMAT_R8G8_SNORM,
        V::Unorm8x4Bgra => DXGI_FORMAT_B8G8R8A8_UNORM,
        V::Uint16 => DXGI_FORMAT_R16_UINT,
        V::Sint16 => DXGI_FORMAT_R16_SINT,
        V::Sint16x2 => DXGI_FORMAT_R16G16_SINT,
        V::Sint16x4 => DXGI_FORMAT_R16G16B16A16_SINT,
        V::Unorm16 => DXGI_FORMAT_R16_UNORM,
        V::Snorm16 => DXGI_FORMAT_R16_SNORM,
        V::Float16 => DXGI_FORMAT_R16_FLOAT,
        V::Unorm10_10_10_2 => DXGI_FORMAT_R10G10B10A2_UNORM,
        V::Uint8x2 => DXGI_FORMAT_R8G8_UINT,
        V::Uint8x4 => DXGI_FORMAT_R8G8B8A8_UINT,
        V::Unorm8x4 => DXGI_FORMAT_R8G8B8A8_UNORM,
        V::Snorm8x4 => DXGI_FORMAT_R8G8B8A8_SNORM,
        V::Uint16x2 => DXGI_FORMAT_R16G16_UINT,
        V::Uint16x4 => DXGI_FORMAT_R16G16B16A16_UINT,
        V::Unorm16x2 => DXGI_FORMAT_R16G16_UNORM,
        V::Unorm16x4 => DXGI_FORMAT_R16G16B16A16_UNORM,
        V::Snorm16x2 => DXGI_FORMAT_R16G16_SNORM,
        V::Snorm16x4 => DXGI_FORMAT_R16G16B16A16_SNORM,
        V::Float16x2 => DXGI_FORMAT_R16G16_FLOAT,
        V::Float16x4 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        V::Float32 => DXGI_FORMAT_R32_FLOAT,
        V::Float32x2 => DXGI_FORMAT_R32G32_FLOAT,
        V::Float32x3 => DXGI_FORMAT_R32G32B32_FLOAT,
        V::Float32x4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        V::Uint32 => DXGI_FORMAT_R32_UINT,
        V::Uint32x2 => DXGI_FORMAT_R32G32_UINT,
        V::Uint32x3 => DXGI_FORMAT_R32G32B32_UINT,
        V::Uint32x4 => DXGI_FORMAT_R32G32B32A32_UINT,
        V::Sint32 => DXGI_FORMAT_R32_SINT,
        V::Sint32x2 => DXGI_FORMAT_R32G32_SINT,
        V::Sint32x3 => DXGI_FORMAT_R32G32B32_SINT,
        V::Sint32x4 => DXGI_FORMAT_R32G32B32A32_SINT,
    }
}

pub fn index_format(f: t::IndexFormat) -> DXGI_FORMAT {
    match f {
        t::IndexFormat::Uint16 => DXGI_FORMAT_R16_UINT,
        t::IndexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
    }
}

pub fn compare(f: t::CompareFunction) -> D3D11_COMPARISON_FUNC {
    use t::CompareFunction as C;
    match f {
        C::Never => D3D11_COMPARISON_NEVER,
        C::Less => D3D11_COMPARISON_LESS,
        C::LessEqual => D3D11_COMPARISON_LESS_EQUAL,
        C::Equal => D3D11_COMPARISON_EQUAL,
        C::GreaterEqual => D3D11_COMPARISON_GREATER_EQUAL,
        C::Greater => D3D11_COMPARISON_GREATER,
        C::NotEqual => D3D11_COMPARISON_NOT_EQUAL,
        C::Always => D3D11_COMPARISON_ALWAYS,
    }
}

pub fn address(a: t::AddressMode) -> D3D11_TEXTURE_ADDRESS_MODE {
    match a {
        t::AddressMode::Repeat => D3D11_TEXTURE_ADDRESS_WRAP,
        t::AddressMode::MirrorRepeat => D3D11_TEXTURE_ADDRESS_MIRROR,
        t::AddressMode::ClampToEdge => D3D11_TEXTURE_ADDRESS_CLAMP,
        t::AddressMode::ClampToBorder => D3D11_TEXTURE_ADDRESS_BORDER,
    }
}

fn filter_type(f: t::FilterMode) -> i32 {
    match f {
        t::FilterMode::Nearest => D3D11_FILTER_TYPE_POINT.0,
        t::FilterMode::Linear => D3D11_FILTER_TYPE_LINEAR.0,
    }
}

/// wgpu-hal's sampler filter encoding.
pub fn filter(desc: &t::SamplerDescriptor<'_>) -> D3D11_FILTER {
    let reduction = if desc.compare.is_some() {
        D3D11_FILTER_REDUCTION_TYPE_COMPARISON.0
    } else {
        D3D11_FILTER_REDUCTION_TYPE_STANDARD.0
    };
    let mip = match desc.mipmap_filter {
        t::MipmapFilterMode::Nearest => D3D11_FILTER_TYPE_POINT.0,
        t::MipmapFilterMode::Linear => D3D11_FILTER_TYPE_LINEAR.0,
    };
    let mut f = (filter_type(desc.min_filter) << D3D11_MIN_FILTER_SHIFT)
        | (filter_type(desc.mag_filter) << D3D11_MAG_FILTER_SHIFT)
        | (mip << D3D11_MIP_FILTER_SHIFT)
        | (reduction << D3D11_FILTER_REDUCTION_TYPE_SHIFT);
    if desc.anisotropy_clamp != 1 {
        f |= D3D11_FILTER_ANISOTROPIC.0;
    }
    D3D11_FILTER(f)
}

pub fn border(c: Option<t::SamplerBorderColor>) -> [f32; 4] {
    match c {
        Some(t::SamplerBorderColor::TransparentBlack) | None => [0.0; 4],
        Some(t::SamplerBorderColor::OpaqueBlack) => [0.0, 0.0, 0.0, 1.0],
        Some(t::SamplerBorderColor::OpaqueWhite) => [1.0; 4],
    }
}

pub fn topology(p: t::PrimitiveTopology) -> d3d::D3D_PRIMITIVE_TOPOLOGY {
    use t::PrimitiveTopology as P;
    match p {
        P::PointList => d3d::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        P::LineList => d3d::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        P::LineStrip => d3d::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        P::TriangleList => d3d::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        P::TriangleStrip => d3d::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}

fn blend_factor(f: t::BlendFactor, alpha: bool) -> D3D11_BLEND {
    use t::BlendFactor as B;
    match f {
        B::Zero => D3D11_BLEND_ZERO,
        B::One => D3D11_BLEND_ONE,
        B::Src if alpha => D3D11_BLEND_SRC_ALPHA,
        B::Src => D3D11_BLEND_SRC_COLOR,
        B::OneMinusSrc if alpha => D3D11_BLEND_INV_SRC_ALPHA,
        B::OneMinusSrc => D3D11_BLEND_INV_SRC_COLOR,
        B::Dst if alpha => D3D11_BLEND_DEST_ALPHA,
        B::Dst => D3D11_BLEND_DEST_COLOR,
        B::OneMinusDst if alpha => D3D11_BLEND_INV_DEST_ALPHA,
        B::OneMinusDst => D3D11_BLEND_INV_DEST_COLOR,
        B::SrcAlpha => D3D11_BLEND_SRC_ALPHA,
        B::OneMinusSrcAlpha => D3D11_BLEND_INV_SRC_ALPHA,
        B::DstAlpha => D3D11_BLEND_DEST_ALPHA,
        B::OneMinusDstAlpha => D3D11_BLEND_INV_DEST_ALPHA,
        B::Constant => D3D11_BLEND_BLEND_FACTOR,
        B::OneMinusConstant => D3D11_BLEND_INV_BLEND_FACTOR,
        B::SrcAlphaSaturated => D3D11_BLEND_SRC_ALPHA_SAT,
    }
}

fn blend_op(o: t::BlendOperation) -> D3D11_BLEND_OP {
    match o {
        t::BlendOperation::Add => D3D11_BLEND_OP_ADD,
        t::BlendOperation::Subtract => D3D11_BLEND_OP_SUBTRACT,
        t::BlendOperation::ReverseSubtract => D3D11_BLEND_OP_REV_SUBTRACT,
        t::BlendOperation::Min => D3D11_BLEND_OP_MIN,
        t::BlendOperation::Max => D3D11_BLEND_OP_MAX,
    }
}

pub fn render_targets(targets: &[Option<t::ColorTargetState>]) -> [D3D11_RENDER_TARGET_BLEND_DESC; 8] {
    let dummy = D3D11_RENDER_TARGET_BLEND_DESC {
        BlendEnable: false.into(),
        SrcBlend: D3D11_BLEND_ZERO,
        DestBlend: D3D11_BLEND_ZERO,
        BlendOp: D3D11_BLEND_OP_ADD,
        SrcBlendAlpha: D3D11_BLEND_ZERO,
        DestBlendAlpha: D3D11_BLEND_ZERO,
        BlendOpAlpha: D3D11_BLEND_OP_ADD,
        RenderTargetWriteMask: 0,
    };
    let mut raw = [dummy; 8];
    for (r, ct) in raw.iter_mut().zip(targets) {
        let Some(ct) = ct else { continue };
        r.RenderTargetWriteMask = ct.write_mask.bits() as u8;
        if let Some(b) = ct.blend {
            r.BlendEnable = true.into();
            r.BlendOp = blend_op(b.color.operation);
            r.SrcBlend = blend_factor(b.color.src_factor, false);
            r.DestBlend = blend_factor(b.color.dst_factor, false);
            r.BlendOpAlpha = blend_op(b.alpha.operation);
            r.SrcBlendAlpha = blend_factor(b.alpha.src_factor, true);
            r.DestBlendAlpha = blend_factor(b.alpha.dst_factor, true);
        }
    }
    raw
}

fn stencil_op(o: t::StencilOperation) -> D3D11_STENCIL_OP {
    use t::StencilOperation as S;
    match o {
        S::Keep => D3D11_STENCIL_OP_KEEP,
        S::Zero => D3D11_STENCIL_OP_ZERO,
        S::Replace => D3D11_STENCIL_OP_REPLACE,
        S::IncrementClamp => D3D11_STENCIL_OP_INCR_SAT,
        S::IncrementWrap => D3D11_STENCIL_OP_INCR,
        S::DecrementClamp => D3D11_STENCIL_OP_DECR_SAT,
        S::DecrementWrap => D3D11_STENCIL_OP_DECR,
        S::Invert => D3D11_STENCIL_OP_INVERT,
    }
}

fn stencil_face(f: &t::StencilFaceState) -> D3D11_DEPTH_STENCILOP_DESC {
    D3D11_DEPTH_STENCILOP_DESC {
        StencilFailOp: stencil_op(f.fail_op),
        StencilDepthFailOp: stencil_op(f.depth_fail_op),
        StencilPassOp: stencil_op(f.pass_op),
        StencilFunc: compare(f.compare),
    }
}

/// As wgpu-hal: depth enabled unless it is a no-op (always, no writes); stencil enabled unless
/// both faces ignore it.
pub fn depth_stencil(ds: &t::DepthStencilState) -> D3D11_DEPTH_STENCIL_DESC {
    let depth_enabled = ds.depth_compare != t::CompareFunction::Always || ds.depth_write_enabled;
    let s = &ds.stencil;
    let stencil_enabled = (s.front != t::StencilFaceState::IGNORE || s.back != t::StencilFaceState::IGNORE) && (s.read_mask != 0 || s.write_mask != 0);
    D3D11_DEPTH_STENCIL_DESC {
        DepthEnable: depth_enabled.into(),
        DepthWriteMask: if ds.depth_write_enabled { D3D11_DEPTH_WRITE_MASK_ALL } else { D3D11_DEPTH_WRITE_MASK_ZERO },
        DepthFunc: compare(ds.depth_compare),
        StencilEnable: stencil_enabled.into(),
        StencilReadMask: s.read_mask as u8,
        StencilWriteMask: s.write_mask as u8,
        FrontFace: stencil_face(&s.front),
        BackFace: stencil_face(&s.back),
    }
}
