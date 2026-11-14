//! Texture and vertex formats.

/// A texel format. The set is the one openOMSI uses plus its close relatives; a backend
/// that cannot make a format says so when the texture is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    R8Snorm,
    R8Uint,
    R16Float,
    R16Uint,
    R32Float,
    R32Uint,
    R32Sint,
    Rg8Unorm,
    Rg16Float,
    Rg32Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    Rgba16Float,
    Rgba16Uint,
    Rgba32Float,
    Rgba32Uint,
    Depth16Unorm,
    Depth32Float,
    Depth24PlusStencil8,
    Depth32FloatStencil8,
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc5RgUnorm,
    Bc6hRgbUfloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
}

/// What a shader gets when it reads a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Depth,
    Uint,
    Sint,
}

impl TextureFormat {
    /// Width and height of one block (1×1 for uncompressed formats).
    pub const fn block_dimensions(self) -> (u32, u32) {
        if self.is_compressed() { (4, 4) } else { (1, 1) }
    }

    /// Bytes per block (per texel for uncompressed formats). `None` for depth-stencil
    /// formats, whose layout is the driver's.
    pub const fn block_copy_size(self) -> Option<u32> {
        use TextureFormat::*;
        Some(match self {
            R8Unorm | R8Snorm | R8Uint => 1,
            R16Float | R16Uint | Rg8Unorm | Depth16Unorm => 2,
            R32Float | R32Uint | R32Sint | Rg16Float | Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Bgra8Unorm | Bgra8UnormSrgb | Rgb10a2Unorm | Rg11b10Ufloat | Depth32Float => 4,
            Rg32Float | Rgba16Float | Rgba16Uint => 8,
            Rgba32Float | Rgba32Uint => 16,
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc4RUnorm => 8,
            Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc5RgUnorm | Bc6hRgbUfloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => 16,
            Depth24PlusStencil8 | Depth32FloatStencil8 => return None,
        })
    }

    pub const fn is_compressed(self) -> bool {
        use TextureFormat::*;
        matches!(
            self,
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc5RgUnorm | Bc6hRgbUfloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb
        )
    }

    pub const fn is_srgb(self) -> bool {
        use TextureFormat::*;
        matches!(self, Rgba8UnormSrgb | Bgra8UnormSrgb | Bc1RgbaUnormSrgb | Bc2RgbaUnormSrgb | Bc3RgbaUnormSrgb | Bc7RgbaUnormSrgb)
    }

    pub const fn has_depth(self) -> bool {
        use TextureFormat::*;
        matches!(self, Depth16Unorm | Depth32Float | Depth24PlusStencil8 | Depth32FloatStencil8)
    }

    pub const fn has_stencil(self) -> bool {
        matches!(self, TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8)
    }

    pub const fn sample_type(self) -> TextureSampleType {
        use TextureFormat::*;
        match self {
            R8Uint | R16Uint | R32Uint | Rgba8Uint | Rgba16Uint | Rgba32Uint => TextureSampleType::Uint,
            R32Sint => TextureSampleType::Sint,
            Depth16Unorm | Depth32Float | Depth24PlusStencil8 | Depth32FloatStencil8 => TextureSampleType::Depth,
            R32Float | Rg32Float | Rgba32Float => TextureSampleType::Float { filterable: false },
            _ => TextureSampleType::Float { filterable: true },
        }
    }

    /// The same format with sRGB decoding removed (or itself).
    pub const fn remove_srgb_suffix(self) -> TextureFormat {
        use TextureFormat::*;
        match self {
            Rgba8UnormSrgb => Rgba8Unorm,
            Bgra8UnormSrgb => Bgra8Unorm,
            Bc1RgbaUnormSrgb => Bc1RgbaUnorm,
            Bc2RgbaUnormSrgb => Bc2RgbaUnorm,
            Bc3RgbaUnormSrgb => Bc3RgbaUnorm,
            Bc7RgbaUnormSrgb => Bc7RgbaUnorm,
            f => f,
        }
    }

    /// Bytes in one row of `width` texels (whole blocks for compressed formats).
    pub fn bytes_per_row(self, width: u32) -> Option<u32> {
        let (bw, _) = self.block_dimensions();
        Some(width.div_ceil(bw) * self.block_copy_size()?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Uint8,
    Uint8x2,
    Uint8x4,
    Unorm8x4,
    Snorm8x4,
    Sint8,
    Sint8x2,
    Sint8x4,
    Unorm8,
    Unorm8x2,
    Snorm8,
    Snorm8x2,
    /// Four `u8`s in B, G, R, A order, read as RGBA.
    Unorm8x4Bgra,
    Uint16,
    Sint16,
    Sint16x2,
    Sint16x4,
    Unorm16,
    Snorm16,
    Float16,
    /// RGB 10 bits each, A 2 bits, in one little-endian `u32` (R lowest).
    Unorm10_10_10_2,
    Uint16x2,
    Uint16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

impl VertexFormat {
    pub const fn size(self) -> u64 {
        use VertexFormat::*;
        match self {
            Uint8 | Sint8 | Unorm8 | Snorm8 => 1,
            Uint8x2 | Sint8x2 | Unorm8x2 | Snorm8x2 | Uint16 | Sint16 | Unorm16 | Snorm16 | Float16 => 2,
            Uint8x4 | Unorm8x4 | Snorm8x4 | Sint8x4 | Unorm8x4Bgra | Uint16x2 | Sint16x2 | Unorm16x2 | Snorm16x2 | Float16x2 | Unorm10_10_10_2 | Float32 | Uint32 | Sint32 => 4,
            Uint16x4 | Sint16x4 | Unorm16x4 | Snorm16x4 | Float16x4 | Float32x2 | Uint32x2 | Sint32x2 => 8,
            Float32x3 | Uint32x3 | Sint32x3 => 12,
            Float32x4 | Uint32x4 | Sint32x4 => 16,
        }
    }
}

/// `[VertexAttribute]` from `location => format` pairs, offsets packed in order, as
/// `wgpu::vertex_attr_array!`.
#[macro_export]
macro_rules! vertex_attr_array {
    ($($loc:expr => $fmt:ident),* $(,)?) => {{
        let mut offset = 0u64;
        [$({
            let a = $crate::VertexAttribute { format: $crate::VertexFormat::$fmt, offset, shader_location: $loc };
            offset += $crate::VertexFormat::$fmt.size();
            let _ = offset;
            a
        }),*]
    }};
}
