//! mX types to Vulkan (as wgpu-hal maps them).

use ash::vk;
use mx_types as t;

/// The Vulkan format of `f`. `d24s8`: the device renders to `D24_UNORM_S8_UINT` (else
/// `Depth24PlusStencil8` is `D32_SFLOAT_S8_UINT`, as in wgpu).
pub fn format(f: t::TextureFormat, d24s8: bool) -> vk::Format {
    use t::TextureFormat as T;
    use vk::Format as F;
    match f {
        T::R8Unorm => F::R8_UNORM,
        T::R8Snorm => F::R8_SNORM,
        T::R8Uint => F::R8_UINT,
        T::R16Float => F::R16_SFLOAT,
        T::R16Uint => F::R16_UINT,
        T::R32Float => F::R32_SFLOAT,
        T::R32Uint => F::R32_UINT,
        T::R32Sint => F::R32_SINT,
        T::Rg8Unorm => F::R8G8_UNORM,
        T::Rg16Float => F::R16G16_SFLOAT,
        T::Rg32Float => F::R32G32_SFLOAT,
        T::Rgba8Unorm => F::R8G8B8A8_UNORM,
        T::Rgba8UnormSrgb => F::R8G8B8A8_SRGB,
        T::Rgba8Snorm => F::R8G8B8A8_SNORM,
        T::Rgba8Uint => F::R8G8B8A8_UINT,
        T::Bgra8Unorm => F::B8G8R8A8_UNORM,
        T::Bgra8UnormSrgb => F::B8G8R8A8_SRGB,
        T::Rgb10a2Unorm => F::A2B10G10R10_UNORM_PACK32,
        T::Rg11b10Ufloat => F::B10G11R11_UFLOAT_PACK32,
        T::Rgba16Float => F::R16G16B16A16_SFLOAT,
        T::Rgba16Uint => F::R16G16B16A16_UINT,
        T::Rgba32Float => F::R32G32B32A32_SFLOAT,
        T::Rgba32Uint => F::R32G32B32A32_UINT,
        T::Depth16Unorm => F::D16_UNORM,
        T::Depth32Float => F::D32_SFLOAT,
        T::Depth24PlusStencil8 => {
            if d24s8 {
                F::D24_UNORM_S8_UINT
            } else {
                F::D32_SFLOAT_S8_UINT
            }
        }
        T::Depth32FloatStencil8 => F::D32_SFLOAT_S8_UINT,
        T::Bc1RgbaUnorm => F::BC1_RGBA_UNORM_BLOCK,
        T::Bc1RgbaUnormSrgb => F::BC1_RGBA_SRGB_BLOCK,
        T::Bc2RgbaUnorm => F::BC2_UNORM_BLOCK,
        T::Bc2RgbaUnormSrgb => F::BC2_SRGB_BLOCK,
        T::Bc3RgbaUnorm => F::BC3_UNORM_BLOCK,
        T::Bc3RgbaUnormSrgb => F::BC3_SRGB_BLOCK,
        T::Bc4RUnorm => F::BC4_UNORM_BLOCK,
        T::Bc5RgUnorm => F::BC5_UNORM_BLOCK,
        T::Bc6hRgbUfloat => F::BC6H_UFLOAT_BLOCK,
        T::Bc7RgbaUnorm => F::BC7_UNORM_BLOCK,
        T::Bc7RgbaUnormSrgb => F::BC7_SRGB_BLOCK,
    }
}

/// The mX format of a surface format (those mX knows).
pub fn surface_format(f: vk::Format) -> Option<t::TextureFormat> {
    use t::TextureFormat as T;
    use vk::Format as F;
    Some(match f {
        F::B8G8R8A8_SRGB => T::Bgra8UnormSrgb,
        F::B8G8R8A8_UNORM => T::Bgra8Unorm,
        F::R8G8B8A8_SRGB => T::Rgba8UnormSrgb,
        F::R8G8B8A8_UNORM => T::Rgba8Unorm,
        F::R16G16B16A16_SFLOAT => T::Rgba16Float,
        F::A2B10G10R10_UNORM_PACK32 => T::Rgb10a2Unorm,
        _ => return None,
    })
}

/// The aspects of a format, narrowed to `aspect`.
pub fn aspects(f: t::TextureFormat, aspect: t::TextureAspect) -> vk::ImageAspectFlags {
    let all = if f.has_depth() {
        let mut a = vk::ImageAspectFlags::DEPTH;
        if f.has_stencil() {
            a |= vk::ImageAspectFlags::STENCIL;
        }
        a
    } else {
        vk::ImageAspectFlags::COLOR
    };
    match aspect {
        t::TextureAspect::All => all,
        t::TextureAspect::DepthOnly => all & vk::ImageAspectFlags::DEPTH,
        t::TextureAspect::StencilOnly => all & vk::ImageAspectFlags::STENCIL,
    }
}

pub fn view_type(d: t::TextureViewDimension) -> vk::ImageViewType {
    match d {
        t::TextureViewDimension::D1 => vk::ImageViewType::TYPE_1D,
        t::TextureViewDimension::D2 => vk::ImageViewType::TYPE_2D,
        t::TextureViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
        t::TextureViewDimension::Cube => vk::ImageViewType::CUBE,
        t::TextureViewDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        t::TextureViewDimension::D3 => vk::ImageViewType::TYPE_3D,
    }
}

pub fn default_view_dimension(dim: t::TextureDimension, layers: u32) -> t::TextureViewDimension {
    match dim {
        t::TextureDimension::D1 => t::TextureViewDimension::D1,
        t::TextureDimension::D2 if layers > 1 => t::TextureViewDimension::D2Array,
        t::TextureDimension::D2 => t::TextureViewDimension::D2,
        t::TextureDimension::D3 => t::TextureViewDimension::D3,
    }
}

pub fn image_usage(u: t::TextureUsages, f: t::TextureFormat) -> vk::ImageUsageFlags {
    let mut v = vk::ImageUsageFlags::empty();
    if u.contains(t::TextureUsages::COPY_SRC) {
        v |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if u.contains(t::TextureUsages::COPY_DST) {
        v |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if u.contains(t::TextureUsages::TEXTURE_BINDING) {
        v |= vk::ImageUsageFlags::SAMPLED;
    }
    if u.contains(t::TextureUsages::STORAGE_BINDING) {
        v |= vk::ImageUsageFlags::STORAGE;
    }
    if u.contains(t::TextureUsages::RENDER_ATTACHMENT) {
        v |= if f.has_depth() {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
    }
    v
}

pub fn samples(n: u32) -> vk::SampleCountFlags {
    vk::SampleCountFlags::from_raw(n.max(1))
}

pub fn compare(f: t::CompareFunction) -> vk::CompareOp {
    use t::CompareFunction as C;
    match f {
        C::Never => vk::CompareOp::NEVER,
        C::Less => vk::CompareOp::LESS,
        C::Equal => vk::CompareOp::EQUAL,
        C::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        C::Greater => vk::CompareOp::GREATER,
        C::NotEqual => vk::CompareOp::NOT_EQUAL,
        C::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        C::Always => vk::CompareOp::ALWAYS,
    }
}

pub fn filter(f: t::FilterMode) -> vk::Filter {
    match f {
        t::FilterMode::Nearest => vk::Filter::NEAREST,
        t::FilterMode::Linear => vk::Filter::LINEAR,
    }
}

pub fn mip_filter(f: t::MipmapFilterMode) -> vk::SamplerMipmapMode {
    match f {
        t::MipmapFilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        t::MipmapFilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

pub fn address(a: t::AddressMode) -> vk::SamplerAddressMode {
    match a {
        t::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        t::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        t::AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        t::AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

pub fn border(c: t::SamplerBorderColor) -> vk::BorderColor {
    match c {
        t::SamplerBorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        t::SamplerBorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        t::SamplerBorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
    }
}

pub fn topology(p: t::PrimitiveTopology) -> vk::PrimitiveTopology {
    match p {
        t::PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        t::PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        t::PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        t::PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        t::PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

pub fn front_face(f: t::FrontFace) -> vk::FrontFace {
    match f {
        t::FrontFace::Cw => vk::FrontFace::CLOCKWISE,
        t::FrontFace::Ccw => vk::FrontFace::COUNTER_CLOCKWISE,
    }
}

pub fn cull(c: Option<t::Face>) -> vk::CullModeFlags {
    match c {
        None => vk::CullModeFlags::NONE,
        Some(t::Face::Front) => vk::CullModeFlags::FRONT,
        Some(t::Face::Back) => vk::CullModeFlags::BACK,
    }
}

pub fn blend_factor(f: t::BlendFactor) -> vk::BlendFactor {
    use t::BlendFactor as B;
    match f {
        B::Zero => vk::BlendFactor::ZERO,
        B::One => vk::BlendFactor::ONE,
        B::Src => vk::BlendFactor::SRC_COLOR,
        B::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        B::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        B::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        B::Dst => vk::BlendFactor::DST_COLOR,
        B::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        B::DstAlpha => vk::BlendFactor::DST_ALPHA,
        B::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        B::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        B::Constant => vk::BlendFactor::CONSTANT_COLOR,
        B::OneMinusConstant => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    }
}

pub fn blend_op(o: t::BlendOperation) -> vk::BlendOp {
    match o {
        t::BlendOperation::Add => vk::BlendOp::ADD,
        t::BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        t::BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        t::BlendOperation::Min => vk::BlendOp::MIN,
        t::BlendOperation::Max => vk::BlendOp::MAX,
    }
}

pub fn write_mask(w: t::ColorWrites) -> vk::ColorComponentFlags {
    let mut m = vk::ColorComponentFlags::empty();
    if w.contains(t::ColorWrites::RED) {
        m |= vk::ColorComponentFlags::R;
    }
    if w.contains(t::ColorWrites::GREEN) {
        m |= vk::ColorComponentFlags::G;
    }
    if w.contains(t::ColorWrites::BLUE) {
        m |= vk::ColorComponentFlags::B;
    }
    if w.contains(t::ColorWrites::ALPHA) {
        m |= vk::ColorComponentFlags::A;
    }
    m
}

pub fn vertex_format(f: t::VertexFormat) -> vk::Format {
    use t::VertexFormat as V;
    use vk::Format as F;
    match f {
        V::Uint8 => F::R8_UINT,
        V::Sint8 => F::R8_SINT,
        V::Sint8x2 => F::R8G8_SINT,
        V::Sint8x4 => F::R8G8B8A8_SINT,
        V::Unorm8 => F::R8_UNORM,
        V::Unorm8x2 => F::R8G8_UNORM,
        V::Snorm8 => F::R8_SNORM,
        V::Snorm8x2 => F::R8G8_SNORM,
        V::Unorm8x4Bgra => F::B8G8R8A8_UNORM,
        V::Uint16 => F::R16_UINT,
        V::Sint16 => F::R16_SINT,
        V::Sint16x2 => F::R16G16_SINT,
        V::Sint16x4 => F::R16G16B16A16_SINT,
        V::Unorm16 => F::R16_UNORM,
        V::Snorm16 => F::R16_SNORM,
        V::Float16 => F::R16_SFLOAT,
        V::Unorm10_10_10_2 => F::A2B10G10R10_UNORM_PACK32,
        V::Uint8x2 => F::R8G8_UINT,
        V::Uint8x4 => F::R8G8B8A8_UINT,
        V::Unorm8x4 => F::R8G8B8A8_UNORM,
        V::Snorm8x4 => F::R8G8B8A8_SNORM,
        V::Uint16x2 => F::R16G16_UINT,
        V::Uint16x4 => F::R16G16B16A16_UINT,
        V::Unorm16x2 => F::R16G16_UNORM,
        V::Unorm16x4 => F::R16G16B16A16_UNORM,
        V::Snorm16x2 => F::R16G16_SNORM,
        V::Snorm16x4 => F::R16G16B16A16_SNORM,
        V::Float16x2 => F::R16G16_SFLOAT,
        V::Float16x4 => F::R16G16B16A16_SFLOAT,
        V::Float32 => F::R32_SFLOAT,
        V::Float32x2 => F::R32G32_SFLOAT,
        V::Float32x3 => F::R32G32B32_SFLOAT,
        V::Float32x4 => F::R32G32B32A32_SFLOAT,
        V::Uint32 => F::R32_UINT,
        V::Uint32x2 => F::R32G32_UINT,
        V::Uint32x3 => F::R32G32B32_UINT,
        V::Uint32x4 => F::R32G32B32A32_UINT,
        V::Sint32 => F::R32_SINT,
        V::Sint32x2 => F::R32G32_SINT,
        V::Sint32x3 => F::R32G32B32_SINT,
        V::Sint32x4 => F::R32G32B32A32_SINT,
    }
}

pub fn index_type(f: t::IndexFormat) -> vk::IndexType {
    match f {
        t::IndexFormat::Uint16 => vk::IndexType::UINT16,
        t::IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

pub fn stencil_op(o: t::StencilOperation) -> vk::StencilOp {
    use t::StencilOperation as S;
    match o {
        S::Keep => vk::StencilOp::KEEP,
        S::Zero => vk::StencilOp::ZERO,
        S::Replace => vk::StencilOp::REPLACE,
        S::Invert => vk::StencilOp::INVERT,
        S::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        S::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        S::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        S::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

pub fn load_op<V>(l: &t::LoadOp<V>) -> vk::AttachmentLoadOp {
    match l {
        t::LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
        t::LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        t::LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

pub fn store_op(s: t::StoreOp) -> vk::AttachmentStoreOp {
    match s {
        t::StoreOp::Store => vk::AttachmentStoreOp::STORE,
        t::StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
    }
}

pub fn clear_color(c: t::Color, f: t::TextureFormat) -> vk::ClearColorValue {
    match f.sample_type() {
        t::TextureSampleType::Uint => vk::ClearColorValue {
            uint32: [c.r as u32, c.g as u32, c.b as u32, c.a as u32],
        },
        t::TextureSampleType::Sint => vk::ClearColorValue {
            int32: [c.r as i32, c.g as i32, c.b as i32, c.a as i32],
        },
        _ => vk::ClearColorValue {
            float32: [c.r as f32, c.g as f32, c.b as f32, c.a as f32],
        },
    }
}

pub fn origin(o: t::Origin3d, is_3d: bool) -> vk::Offset3D {
    vk::Offset3D {
        x: o.x as i32,
        y: o.y as i32,
        z: if is_3d { o.z as i32 } else { 0 },
    }
}

pub fn extent(e: t::Extent3d, is_3d: bool) -> vk::Extent3D {
    vk::Extent3D {
        width: e.width,
        height: e.height,
        depth: if is_3d { e.depth_or_array_layers } else { 1 },
    }
}

pub fn buffer_usage(u: t::BufferUsages, device_address: bool) -> vk::BufferUsageFlags {
    // (copies and fills are always allowed: `create_buffer_init`, `clear_buffer`)
    let mut v = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    if u.contains(t::BufferUsages::INDEX) {
        v |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if u.contains(t::BufferUsages::VERTEX) {
        v |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if u.contains(t::BufferUsages::UNIFORM) {
        v |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if u.contains(t::BufferUsages::STORAGE) {
        v |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if u.contains(t::BufferUsages::INDIRECT) {
        v |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if u.contains(t::BufferUsages::BLAS_INPUT) && device_address {
        v |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    v
}
