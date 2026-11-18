//! Intel XeSS Super Resolution (XeSS-SR) — raw FFI transcription, D3D12 + Vulkan.
//!
//! Source: https://github.com/intel/xess  — release tag **v3.0.2** (published 2026-07-24)
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xess/xess.h
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xess/xess_d3d12.h
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xess/xess_vk.h
//! Call-order notes also use doc/xess_sr_developer_guide_english.md at the same tag.
//!
//! DLL: every function below is exported by `libxess.dll` (import lib `lib/libxess.lib`).
//! D3D12 **and** Vulkan backends live in the same `libxess.dll` (SR guide, "Deployment":
//! "For D3D12 and Vulkan: `libxess.dll`"). D3D11 is a separate `libxess_dx11.dll` that exports
//! the same generic names (xessGetVersion, ...); the guide forbids mixing the two DLLs for one
//! context. Not transcribed here: xess_d3d11.h, xess_debug.h, xess_d3d12_debug.h, xess_vk_debug.h.
//!
//! Licence text exactly as it appears at the top of xess.h / xess_d3d12.h
//! ("Copyright (C) 2021") and xess_vk.h ("Copyright (C) 2023"):
//!
//! ```text
//!  * Copyright (C) 2021 Intel Corporation
//!  *
//!  * This software and the related documents are Intel copyrighted materials, and
//!  * your use of them is governed by the express license under which they were
//!  * provided to you ("License"). Unless the License provides otherwise, you may
//!  * not use, modify, copy, publish, distribute, disclose or transmit this
//!  * software or the related documents without Intel's prior written permission.
//!  *
//!  * This software and the related documents are provided as is, with no express
//!  * or implied warranties, other than those that are expressly stated in the
//!  * License.
//! ```
//!
//! Transcription conventions:
//! * Every C struct is wrapped in `#pragma pack(push, 8)` in the headers. Pack(8) on x64 never
//!   reduces any field's natural alignment (max is 8), so plain `#[repr(C)]` is identical.
//!   Layout is asserted at compile time below (64-bit only).
//! * C enums are 4 bytes (headers static-assert this). They are transcribed as
//!   `#[repr(transparent)]` newtypes over the C underlying int, NOT Rust enums, so an
//!   unknown value returned by a newer DLL is never UB.
//! * C `bool` (xessD3D12BuildPipelines / xessVKBuildPipelines / xessForceLegacyScaleFactors)
//!   is 1 byte => Rust `bool`.
//! * No explicit calling convention in the headers => cdecl => `extern "C"` (x64: the one ABI).
//! * D3D12 COM interfaces are opaque here (`*mut c_void`-like). Pass `windows::...::as_raw()`.
//! * Vulkan: dispatchable handles (VkInstance, VkPhysicalDevice, VkDevice, VkCommandBuffer) are
//!   pointers; non-dispatchable (VkImage, VkImageView, VkDeviceMemory, VkPipelineCache) are u64
//!   on all targets (same representation as `ash::vk::*` handles).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

use core::ffi::{c_char, c_void};

// ---------------------------------------------------------------------------------------------
// Opaque foreign types
// ---------------------------------------------------------------------------------------------

/// `typedef struct _xess_context_handle_t* xess_context_handle_t;`  (xess.h:47)
#[repr(C)]
pub struct _xess_context_handle_t {
    _private: [u8; 0],
}
pub type xess_context_handle_t = *mut _xess_context_handle_t;

// D3D12 COM interfaces (opaque; only ever passed by pointer).
pub type ID3D12Device = c_void;
pub type ID3D12Resource = c_void;
pub type ID3D12DescriptorHeap = c_void;
pub type ID3D12Heap = c_void;
pub type ID3D12PipelineLibrary = c_void;
pub type ID3D12GraphicsCommandList = c_void;

// Vulkan handles.
pub type VkInstance = *mut c_void; // dispatchable
pub type VkPhysicalDevice = *mut c_void; // dispatchable
pub type VkDevice = *mut c_void; // dispatchable
pub type VkCommandBuffer = *mut c_void; // dispatchable
pub type VkImage = u64; // non-dispatchable
pub type VkImageView = u64; // non-dispatchable
pub type VkDeviceMemory = u64; // non-dispatchable
pub type VkPipelineCache = u64; // non-dispatchable
pub type VkFormat = i32; // C enum
pub type VkImageAspectFlags = u32;

/// Vulkan core `VkImageSubresourceRange` (20 bytes, align 4) — embedded by value in
/// `xess_vk_image_view_info`. Bit-identical to `ash::vk::ImageSubresourceRange`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VkImageSubresourceRange {
    pub aspectMask: VkImageAspectFlags,
    pub baseMipLevel: u32,
    pub levelCount: u32,
    pub baseArrayLayer: u32,
    pub layerCount: u32,
}

// ---------------------------------------------------------------------------------------------
// xess.h
// ---------------------------------------------------------------------------------------------

/// xess.h:55-70 `xess_version_t`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct xess_version_t {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

/// xess.h:77-81 `xess_2d_t`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct xess_2d_t {
    pub x: u32,
    pub y: u32,
}

/// xess.h:87 `typedef xess_2d_t xess_coord_t;`
pub type xess_coord_t = xess_2d_t;

/// xess.h:92-101 `xess_quality_settings_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xess_quality_settings_t(pub i32);
impl xess_quality_settings_t {
    pub const ULTRA_PERFORMANCE: Self = Self(100); // 3.0x
    pub const PERFORMANCE: Self = Self(101); // 2.3x
    pub const BALANCED: Self = Self(102); // 2.0x
    pub const QUALITY: Self = Self(103); // 1.7x
    pub const ULTRA_QUALITY: Self = Self(104); // 1.5x
    pub const ULTRA_QUALITY_PLUS: Self = Self(105); // 1.3x
    /// "Native Anti-Aliasing" (NAA), 1.0x. C name: XESS_QUALITY_SETTING_AA.
    pub const AA: Self = Self(106);
}
pub const XESS_QUALITY_SETTING_ULTRA_PERFORMANCE: xess_quality_settings_t = xess_quality_settings_t(100);
pub const XESS_QUALITY_SETTING_PERFORMANCE: xess_quality_settings_t = xess_quality_settings_t(101);
pub const XESS_QUALITY_SETTING_BALANCED: xess_quality_settings_t = xess_quality_settings_t(102);
pub const XESS_QUALITY_SETTING_QUALITY: xess_quality_settings_t = xess_quality_settings_t(103);
pub const XESS_QUALITY_SETTING_ULTRA_QUALITY: xess_quality_settings_t = xess_quality_settings_t(104);
pub const XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS: xess_quality_settings_t = xess_quality_settings_t(105);
pub const XESS_QUALITY_SETTING_AA: xess_quality_settings_t = xess_quality_settings_t(106);

/// xess.h:106-127 `xess_init_flags_t` — bit flags; the init-param fields that carry them are
/// plain `uint32_t initFlags`, so these are u32 constants.
pub type xess_init_flags_t = u32;
pub const XESS_INIT_FLAG_NONE: u32 = 0;
/// Use motion vectors at target resolution.
pub const XESS_INIT_FLAG_HIGH_RES_MV: u32 = 1 << 0;
/// Use inverted (increased precision) depth encoding.
pub const XESS_INIT_FLAG_INVERTED_DEPTH: u32 = 1 << 1;
/// Use exposure texture to scale input color.
pub const XESS_INIT_FLAG_EXPOSURE_SCALE_TEXTURE: u32 = 1 << 2;
/// Use responsive pixel mask texture.
pub const XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK: u32 = 1 << 3;
/// Use velocity in NDC.
pub const XESS_INIT_FLAG_USE_NDC_VELOCITY: u32 = 1 << 4;
/// Use external descriptor heap (D3D12 only; not supported on Vulkan).
pub const XESS_INIT_FLAG_EXTERNAL_DESCRIPTOR_HEAP: u32 = 1 << 5;
/// Disable tonemapping for input and output.
pub const XESS_INIT_FLAG_LDR_INPUT_COLOR: u32 = 1 << 6;
/// Remove jitter from input velocity.
pub const XESS_INIT_FLAG_JITTERED_MV: u32 = 1 << 7;
/// Enable automatic exposure calculation.
pub const XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE: u32 = 1 << 8;

/// xess.h:133-141 `xess_properties_t`  (u32 at 0, 4 bytes padding, u64 at 8, u64 at 16)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct xess_properties_t {
    /// Required amount of descriptors for XeSS.
    pub requiredDescriptorCount: u32,
    /// Heap size required for temporary buffer storage.
    pub tempBufferHeapSize: u64,
    /// Heap size required for temporary texture storage.
    pub tempTextureHeapSize: u64,
}

/// xess.h:147-182 `xess_result_t`. Positive = warning, 0 = success, negative = error.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[must_use]
pub struct xess_result_t(pub i32);
impl xess_result_t {
    pub const WARNING_NONEXISTING_FOLDER: Self = Self(1);
    pub const WARNING_OLD_DRIVER: Self = Self(2);
    pub const SUCCESS: Self = Self(0);
    pub const ERROR_UNSUPPORTED_DEVICE: Self = Self(-1);
    pub const ERROR_UNSUPPORTED_DRIVER: Self = Self(-2);
    pub const ERROR_UNINITIALIZED: Self = Self(-3);
    pub const ERROR_INVALID_ARGUMENT: Self = Self(-4);
    pub const ERROR_DEVICE_OUT_OF_MEMORY: Self = Self(-5);
    pub const ERROR_DEVICE: Self = Self(-6);
    pub const ERROR_NOT_IMPLEMENTED: Self = Self(-7);
    pub const ERROR_INVALID_CONTEXT: Self = Self(-8);
    pub const ERROR_OPERATION_IN_PROGRESS: Self = Self(-9);
    pub const ERROR_UNSUPPORTED: Self = Self(-10);
    pub const ERROR_CANT_LOAD_LIBRARY: Self = Self(-11);
    pub const ERROR_WRONG_CALL_ORDER: Self = Self(-12);
    pub const ERROR_UNKNOWN: Self = Self(-1000);

    #[inline]
    pub fn is_error(self) -> bool {
        self.0 < 0
    }
}
pub const XESS_RESULT_WARNING_NONEXISTING_FOLDER: xess_result_t = xess_result_t(1);
pub const XESS_RESULT_WARNING_OLD_DRIVER: xess_result_t = xess_result_t(2);
pub const XESS_RESULT_SUCCESS: xess_result_t = xess_result_t(0);
pub const XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: xess_result_t = xess_result_t(-1);
pub const XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: xess_result_t = xess_result_t(-2);
pub const XESS_RESULT_ERROR_UNINITIALIZED: xess_result_t = xess_result_t(-3);
pub const XESS_RESULT_ERROR_INVALID_ARGUMENT: xess_result_t = xess_result_t(-4);
pub const XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: xess_result_t = xess_result_t(-5);
pub const XESS_RESULT_ERROR_DEVICE: xess_result_t = xess_result_t(-6);
pub const XESS_RESULT_ERROR_NOT_IMPLEMENTED: xess_result_t = xess_result_t(-7);
pub const XESS_RESULT_ERROR_INVALID_CONTEXT: xess_result_t = xess_result_t(-8);
pub const XESS_RESULT_ERROR_OPERATION_IN_PROGRESS: xess_result_t = xess_result_t(-9);
pub const XESS_RESULT_ERROR_UNSUPPORTED: xess_result_t = xess_result_t(-10);
pub const XESS_RESULT_ERROR_CANT_LOAD_LIBRARY: xess_result_t = xess_result_t(-11);
pub const XESS_RESULT_ERROR_WRONG_CALL_ORDER: xess_result_t = xess_result_t(-12);
pub const XESS_RESULT_ERROR_UNKNOWN: xess_result_t = xess_result_t(-1000);

/// xess.h:187-193 `xess_logging_level_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xess_logging_level_t(pub i32);
pub const XESS_LOGGING_LEVEL_DEBUG: xess_logging_level_t = xess_logging_level_t(0);
pub const XESS_LOGGING_LEVEL_INFO: xess_logging_level_t = xess_logging_level_t(1);
pub const XESS_LOGGING_LEVEL_WARNING: xess_logging_level_t = xess_logging_level_t(2);
pub const XESS_LOGGING_LEVEL_ERROR: xess_logging_level_t = xess_logging_level_t(3);

/// xess.h:200 `typedef void (*xess_app_log_callback_t)(const char *message, xess_logging_level_t loggingLevel);`
/// May be called from other threads, concurrently. `message` is null-terminated UTF-8, valid only
/// during the call. NOTE: no user-data pointer (unlike XeSS-FG's callback).
pub type xess_app_log_callback_t = Option<unsafe extern "C" fn(message: *const c_char, loggingLevel: xess_logging_level_t)>;

// ---------------------------------------------------------------------------------------------
// xess_d3d12.h
// ---------------------------------------------------------------------------------------------

/// xess_d3d12.h:31-77 `xess_d3d12_execute_params_t`  (size 136 on x64)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xess_d3d12_execute_params_t {
    /// Input color. NON_PIXEL_SHADER_RESOURCE state.
    pub pColorTexture: *mut ID3D12Resource, // 0
    /// Input motion vectors. NON_PIXEL_SHADER_RESOURCE state.
    pub pVelocityTexture: *mut ID3D12Resource, // 8
    /// Optional depth; required if HIGH_RES_MV NOT set. NON_PIXEL_SHADER_RESOURCE state.
    pub pDepthTexture: *mut ID3D12Resource, // 16
    /// Optional 1x1 exposure; required with EXPOSURE_SCALE_TEXTURE. NON_PIXEL_SHADER_RESOURCE.
    pub pExposureScaleTexture: *mut ID3D12Resource, // 24
    /// Optional responsive mask; required with RESPONSIVE_PIXEL_MASK. NON_PIXEL_SHADER_RESOURCE.
    pub pResponsivePixelMaskTexture: *mut ID3D12Resource, // 32
    /// Output at target resolution. UNORDERED_ACCESS state.
    pub pOutputTexture: *mut ID3D12Resource, // 40
    /// Jitter X in [-0.5, 0.5].
    pub jitterOffsetX: f32, // 48
    /// Jitter Y in [-0.5, 0.5].
    pub jitterOffsetY: f32, // 52
    /// Optional input color scaling. Default 1.
    pub exposureScale: f32, // 56
    /// Resets history accumulation this frame (0/1).
    pub resetHistory: u32, // 60
    pub inputWidth: u32,                       // 64
    pub inputHeight: u32,                      // 68
    pub inputColorBase: xess_coord_t,          // 72
    pub inputMotionVectorBase: xess_coord_t,   // 80
    pub inputDepthBase: xess_coord_t,          // 88
    pub inputResponsiveMaskBase: xess_coord_t, // 96
    /// Reserved; zero it.
    pub reserved0: xess_coord_t, // 104
    pub outputColorBase: xess_coord_t,         // 112
    /// Optional external descriptor heap (only with EXTERNAL_DESCRIPTOR_HEAP).
    pub pDescriptorHeap: *mut ID3D12DescriptorHeap, // 120
    /// Offset in the external descriptor heap, in BYTES.
    pub descriptorHeapOffset: u32, // 128  (+4 tail padding => 136)
}

/// xess_d3d12.h:84-112 `xess_d3d12_init_params_t`  (size 64 on x64)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xess_d3d12_init_params_t {
    pub outputResolution: xess_2d_t,             // 0
    pub qualitySetting: xess_quality_settings_t, // 8
    /// OR of XESS_INIT_FLAG_*.
    pub initFlags: u32, // 12
    pub creationNodeMask: u32,                   // 16
    pub visibleNodeMask: u32,                    // 20
    /// Optional external buffer heap (D3D12_HEAP_TYPE_DEFAULT) or null.
    pub pTempBufferHeap: *mut ID3D12Heap, // 24
    pub bufferHeapOffset: u64,                   // 32
    /// Optional external texture heap (D3D12_HEAP_TYPE_DEFAULT) or null.
    pub pTempTextureHeap: *mut ID3D12Heap, // 40
    pub textureHeapOffset: u64,                  // 48
    /// Optional pipeline library for pipeline caching, or null.
    pub pPipelineLibrary: *mut ID3D12PipelineLibrary, // 56
}

// ---------------------------------------------------------------------------------------------
// xess_vk.h
// ---------------------------------------------------------------------------------------------

/// xess_vk.h:29-37 `xess_vk_image_view_info`  (size 48)
/// All fields are required (Vulkan cannot query them back). Only 2D views. For depth / depth-
/// stencil images the view & subresourceRange must cover only VK_IMAGE_ASPECT_DEPTH_BIT.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct xess_vk_image_view_info {
    pub imageView: VkImageView,                    // 0
    pub image: VkImage,                            // 8
    pub subresourceRange: VkImageSubresourceRange, // 16 (20 bytes)
    pub format: VkFormat,                          // 36
    /// C `unsigned int`.
    pub width: u32,       // 40
    pub height: u32,                               // 44
}

/// xess_vk.h:45-87 `xess_vk_execute_params_t`  (size 360). No descriptor heap fields (VK has none).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct xess_vk_execute_params_t {
    /// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    pub colorTexture: xess_vk_image_view_info, // 0
    /// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    pub velocityTexture: xess_vk_image_view_info, // 48
    /// Optional (required w/o HIGH_RES_MV). SHADER_READ_ONLY_OPTIMAL.
    pub depthTexture: xess_vk_image_view_info, // 96
    /// Optional (EXPOSURE_SCALE_TEXTURE). SHADER_READ_ONLY_OPTIMAL.
    pub exposureScaleTexture: xess_vk_image_view_info, // 144
    /// Optional (RESPONSIVE_PIXEL_MASK). SHADER_READ_ONLY_OPTIMAL.
    pub responsivePixelMaskTexture: xess_vk_image_view_info, // 192
    /// VK_IMAGE_LAYOUT_GENERAL.
    pub outputTexture: xess_vk_image_view_info, // 240
    pub jitterOffsetX: f32,                    // 288
    pub jitterOffsetY: f32,                    // 292
    pub exposureScale: f32,                    // 296
    pub resetHistory: u32,                     // 300
    pub inputWidth: u32,                       // 304
    pub inputHeight: u32,                      // 308
    pub inputColorBase: xess_coord_t,          // 312
    pub inputMotionVectorBase: xess_coord_t,   // 320
    pub inputDepthBase: xess_coord_t,          // 328
    pub inputResponsiveMaskBase: xess_coord_t, // 336
    pub reserved0: xess_coord_t,               // 344
    pub outputColorBase: xess_coord_t,         // 352
}

/// xess_vk.h:94-124 `xess_vk_init_params_t`  (size 64)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xess_vk_init_params_t {
    pub outputResolution: xess_2d_t,             // 0
    pub qualitySetting: xess_quality_settings_t, // 8
    pub initFlags: u32,                          // 12
    pub creationNodeMask: u32,                   // 16
    pub visibleNodeMask: u32,                    // 20
    /// Optional external buffer memory (DEVICE_LOCAL) or VK_NULL_HANDLE (0).
    pub tempBufferHeap: VkDeviceMemory, // 24
    pub bufferHeapOffset: u64,                   // 32
    /// Optional external texture memory (DEVICE_LOCAL) or VK_NULL_HANDLE (0).
    pub tempTextureHeap: VkDeviceMemory, // 40
    pub textureHeapOffset: u64,                  // 48
    /// Optional pipeline cache or VK_NULL_HANDLE.
    pub pipelineCache: VkPipelineCache, // 56
}

// ---------------------------------------------------------------------------------------------
// Compile-time layout checks (x64 / any 64-bit target)
// ---------------------------------------------------------------------------------------------
#[cfg(target_pointer_width = "64")]
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<xess_version_t>() == 8);
    assert!(size_of::<xess_2d_t>() == 8);
    assert!(size_of::<xess_properties_t>() == 24);
    assert!(offset_of!(xess_properties_t, tempBufferHeapSize) == 8);
    assert!(size_of::<xess_d3d12_execute_params_t>() == 136);
    assert!(offset_of!(xess_d3d12_execute_params_t, jitterOffsetX) == 48);
    assert!(offset_of!(xess_d3d12_execute_params_t, inputColorBase) == 72);
    assert!(offset_of!(xess_d3d12_execute_params_t, pDescriptorHeap) == 120);
    assert!(offset_of!(xess_d3d12_execute_params_t, descriptorHeapOffset) == 128);
    assert!(size_of::<xess_d3d12_init_params_t>() == 64);
    assert!(offset_of!(xess_d3d12_init_params_t, pTempBufferHeap) == 24);
    assert!(offset_of!(xess_d3d12_init_params_t, pPipelineLibrary) == 56);
    assert!(size_of::<VkImageSubresourceRange>() == 20);
    assert!(size_of::<xess_vk_image_view_info>() == 48);
    assert!(offset_of!(xess_vk_image_view_info, format) == 36);
    assert!(size_of::<xess_vk_execute_params_t>() == 360);
    assert!(offset_of!(xess_vk_execute_params_t, jitterOffsetX) == 288);
    assert!(offset_of!(xess_vk_execute_params_t, outputColorBase) == 352);
    assert!(size_of::<xess_vk_init_params_t>() == 64);
    assert!(size_of::<xess_quality_settings_t>() == 4);
    assert!(size_of::<xess_result_t>() == 4);
    assert!(size_of::<xess_logging_level_t>() == 4);
};

// ---------------------------------------------------------------------------------------------
// Exported functions — all from libxess.dll. Type aliases for GetProcAddress / libloading.
// Export names are the undecorated C names given in each doc line.
// ---------------------------------------------------------------------------------------------

pub const XESS_DLL_NAME: &str = "libxess.dll";

// ---- xess.h (backend-agnostic) ----

/// "xessGetVersion" — xess.h:213. Callable without a context.
pub type PFN_xessGetVersion = unsafe extern "C" fn(pVersion: *mut xess_version_t) -> xess_result_t;

/// "xessGetIntelXeFXVersion" — xess.h:223. 0.0.0 on non-Intel.
pub type PFN_xessGetIntelXeFXVersion = unsafe extern "C" fn(hContext: xess_context_handle_t, pVersion: *mut xess_version_t) -> xess_result_t;

/// "xessGetProperties" — xess.h:233. Heap sizes / descriptor count for an output resolution.
pub type PFN_xessGetProperties = unsafe extern "C" fn(hContext: xess_context_handle_t, pOutputResolution: *const xess_2d_t, pBindingProperties: *mut xess_properties_t) -> xess_result_t;

/// "xessGetInputResolution" — xess.h:248. DEPRECATED since 1.2; use GetOptimalInputResolution.
pub type PFN_xessGetInputResolution =
    unsafe extern "C" fn(hContext: xess_context_handle_t, pOutputResolution: *const xess_2d_t, qualitySettings: xess_quality_settings_t, pInputResolution: *mut xess_2d_t) -> xess_result_t;

/// "xessGetOptimalInputResolution" — xess.h:269.
pub type PFN_xessGetOptimalInputResolution = unsafe extern "C" fn(
    hContext: xess_context_handle_t,
    pOutputResolution: *const xess_2d_t,
    qualitySettings: xess_quality_settings_t,
    pInputResolutionOptimal: *mut xess_2d_t,
    pInputResolutionMin: *mut xess_2d_t,
    pInputResolutionMax: *mut xess_2d_t,
) -> xess_result_t;

/// "xessGetJitterScale" — xess.h:280
pub type PFN_xessGetJitterScale = unsafe extern "C" fn(hContext: xess_context_handle_t, pX: *mut f32, pY: *mut f32) -> xess_result_t;

/// "xessGetVelocityScale" — xess.h:289
pub type PFN_xessGetVelocityScale = unsafe extern "C" fn(hContext: xess_context_handle_t, pX: *mut f32, pY: *mut f32) -> xess_result_t;

/// "xessDestroyContext" — xess.h:297. Pending command lists must have completed first.
pub type PFN_xessDestroyContext = unsafe extern "C" fn(hContext: xess_context_handle_t) -> xess_result_t;

/// "xessSetJitterScale" — xess.h:307
pub type PFN_xessSetJitterScale = unsafe extern "C" fn(hContext: xess_context_handle_t, x: f32, y: f32) -> xess_result_t;

/// "xessSetVelocityScale" — xess.h:317
pub type PFN_xessSetVelocityScale = unsafe extern "C" fn(hContext: xess_context_handle_t, x: f32, y: f32) -> xess_result_t;

/// "xessSetExposureMultiplier" — xess.h:328. Applied on top of passed or auto exposure.
pub type PFN_xessSetExposureMultiplier = unsafe extern "C" fn(hContext: xess_context_handle_t, scale: f32) -> xess_result_t;

/// "xessGetExposureMultiplier" — xess.h:337
pub type PFN_xessGetExposureMultiplier = unsafe extern "C" fn(hContext: xess_context_handle_t, pScale: *mut f32) -> xess_result_t;

/// "xessSetMaxResponsiveMaskValue" — xess.h:349. value in [0,1].
pub type PFN_xessSetMaxResponsiveMaskValue = unsafe extern "C" fn(hContext: xess_context_handle_t, value: f32) -> xess_result_t;

/// "xessGetMaxResponsiveMaskValue" — xess.h:358
pub type PFN_xessGetMaxResponsiveMaskValue = unsafe extern "C" fn(hContext: xess_context_handle_t, pValue: *mut f32) -> xess_result_t;

/// "xessSetLoggingCallback" — xess.h:368
pub type PFN_xessSetLoggingCallback = unsafe extern "C" fn(hContext: xess_context_handle_t, loggingLevel: xess_logging_level_t, loggingCallback: xess_app_log_callback_t) -> xess_result_t;

/// "xessIsOptimalDriver" — xess.h:380. SUCCESS or WARNING_OLD_DRIVER.
pub type PFN_xessIsOptimalDriver = unsafe extern "C" fn(hContext: xess_context_handle_t) -> xess_result_t;

/// "xessForceLegacyScaleFactors" — xess.h:401. `force` is C `bool` (1 byte).
pub type PFN_xessForceLegacyScaleFactors = unsafe extern "C" fn(hContext: xess_context_handle_t, force: bool) -> xess_result_t;

/// "xessGetPipelineBuildStatus" — xess.h:417. Only between xess*BuildPipelines and xess*Init.
pub type PFN_xessGetPipelineBuildStatus = unsafe extern "C" fn(hContext: xess_context_handle_t) -> xess_result_t;

// ---- xess_d3d12.h ----

/// "xessD3D12CreateContext" — xess_d3d12.h:124
pub type PFN_xessD3D12CreateContext = unsafe extern "C" fn(pDevice: *mut ID3D12Device, phContext: *mut xess_context_handle_t) -> xess_result_t;

/// "xessD3D12BuildPipelines" — xess_d3d12.h:143. `blocking` is C `bool`. initFlags must equal
/// the flags later passed to xessD3D12Init; same pipeline library must be passed to Init.
pub type PFN_xessD3D12BuildPipelines = unsafe extern "C" fn(hContext: xess_context_handle_t, pPipelineLibrary: *mut ID3D12PipelineLibrary, blocking: bool, initFlags: u32) -> xess_result_t;

/// "xessD3D12Init" — xess_d3d12.h:161. Blocking (allocs + JIT).
pub type PFN_xessD3D12Init = unsafe extern "C" fn(hContext: xess_context_handle_t, pInitParams: *const xess_d3d12_init_params_t) -> xess_result_t;

/// "xessD3D12GetInitParams" — xess_d3d12.h:173
pub type PFN_xessD3D12GetInitParams = unsafe extern "C" fn(hContext: xess_context_handle_t, pInitParams: *mut xess_d3d12_init_params_t) -> xess_result_t;

/// "xessD3D12Execute" — xess_d3d12.h:184. Records into the command list; no GPU work itself.
pub type PFN_xessD3D12Execute = unsafe extern "C" fn(hContext: xess_context_handle_t, pCommandList: *mut ID3D12GraphicsCommandList, pExecParams: *const xess_d3d12_execute_params_t) -> xess_result_t;

// ---- xess_vk.h ----

/// "xessVKGetRequiredInstanceExtensions" — xess_vk.h:142. Array owned by the DLL.
/// `minVkApiVersion` -> VkApplicationInfo.apiVersion must be >= this.
pub type PFN_xessVKGetRequiredInstanceExtensions = unsafe extern "C" fn(instanceExtensionsCount: *mut u32, instanceExtensions: *mut *const *const c_char, minVkApiVersion: *mut u32) -> xess_result_t;

/// "xessVKGetRequiredDeviceExtensions" — xess_vk.h:157. Array owned by the DLL.
pub type PFN_xessVKGetRequiredDeviceExtensions =
    unsafe extern "C" fn(instance: VkInstance, physicalDevice: VkPhysicalDevice, deviceExtensionsCount: *mut u32, deviceExtensions: *mut *const *const c_char) -> xess_result_t;

/// "xessVKGetRequiredDeviceFeatures" — xess_vk.h:182. `features` is `void**`: in = head of
/// your pNext chain (or null), out = patched chain to put in VkDeviceCreateInfo.pNext. You must
/// use VkPhysicalDeviceFeatures2 in the chain; VkDeviceCreateInfo.pEnabledFeatures must be null.
/// Structures added by XeSS are owned by the DLL.
pub type PFN_xessVKGetRequiredDeviceFeatures = unsafe extern "C" fn(instance: VkInstance, physicalDevice: VkPhysicalDevice, features: *mut *mut c_void) -> xess_result_t;

/// "xessVKCreateContext" — xess_vk.h:193
pub type PFN_xessVKCreateContext = unsafe extern "C" fn(instance: VkInstance, physicalDevice: VkPhysicalDevice, device: VkDevice, phContext: *mut xess_context_handle_t) -> xess_result_t;

/// "xessVKBuildPipelines" — xess_vk.h:212. `blocking` is C `bool`.
pub type PFN_xessVKBuildPipelines = unsafe extern "C" fn(hContext: xess_context_handle_t, pipelineCache: VkPipelineCache, blocking: bool, initFlags: u32) -> xess_result_t;

/// "xessVKInit" — xess_vk.h:230
pub type PFN_xessVKInit = unsafe extern "C" fn(hContext: xess_context_handle_t, pInitParams: *const xess_vk_init_params_t) -> xess_result_t;

/// "xessVKGetInitParams" — xess_vk.h:242
pub type PFN_xessVKGetInitParams = unsafe extern "C" fn(hContext: xess_context_handle_t, pInitParams: *mut xess_vk_init_params_t) -> xess_result_t;

/// "xessVKExecute" — xess_vk.h:253. Records into the command buffer.
pub type PFN_xessVKExecute = unsafe extern "C" fn(hContext: xess_context_handle_t, commandBuffer: VkCommandBuffer, pExecParams: *const xess_vk_execute_params_t) -> xess_result_t;

// =============================================================================================
// CALL SEQUENCE (from headers + doc/xess_sr_developer_guide_english.md @ v3.0.2)
// =============================================================================================
//
// --- Vulkan only: BEFORE creating the instance/device ---
//  1. xessVKGetRequiredInstanceExtensions(&n, &names, &minApi)            (xess_vk.h:142)
//     -> enable those in vkCreateInstance, VkApplicationInfo.apiVersion >= minApi.
//  2. after picking the physical device:
//     xessVKGetRequiredDeviceExtensions(inst, phys, &n, &names)           (xess_vk.h:157)
//     xessVKGetRequiredDeviceFeatures(inst, phys, &pNextChain)            (xess_vk.h:182)
//     -> vkCreateDevice with those extensions and the patched pNext chain
//        (VkPhysicalDeviceFeatures2 in the chain, pEnabledFeatures = NULL).
//     Requirements per guide: Vulkan 1.1, shaderStorageImageWriteWithoutFormat,
//     VK_EXT_mutable_descriptor_type.mutableDescriptorType (+ optional int8/int-dot-product).
//
// --- Context + init (both APIs) ---
//  3. xessD3D12CreateContext(device, &ctx)  /  xessVKCreateContext(inst, phys, dev, &ctx)
//     UNSUPPORTED_DEVICE / UNSUPPORTED_DRIVER => no XeSS on this GPU.
//  4. xessIsOptimalDriver(ctx) — WARNING_OLD_DRIVER is non-fatal (on VK the Khronos validation
//     layer can trigger it spuriously).
//  5. xessSetLoggingCallback(ctx, level, cb)  (cb may run on any thread, concurrently)
//  6. optional, early at load: xessD3D12BuildPipelines(ctx, lib, blocking=false, initFlags)
//     / xessVKBuildPipelines; poll xessGetPipelineBuildStatus. Same flags + same pipeline
//     library/cache must go to Init.
//  7. xessGetOptimalInputResolution(ctx, &outRes, quality, &opt, &min, &max)
//     -> render resolution = opt (do NOT hardcode ratios; 1.3+ changed them). Mip bias and jitter
//     sequence length must be derived from the actual ratio (guide: "Update from XeSS 1.2").
//  8. optional external heaps: xessGetProperties(ctx, &outRes, &props) -> allocate
//     D3D12_HEAP_TYPE_DEFAULT heaps (buffers and textures in SEPARATE heaps, HEAP_TIER_1).
//  9. xessD3D12Init(ctx, &init) / xessVKInit(ctx, &init) — blocking. Re-init on output-res or
//     flag change, only after all XeSS command lists are complete. Quality change is cheap.
// 10. optional: xessSetJitterScale / xessSetVelocityScale (e.g. NDC -> pixels),
//     xessSetExposureMultiplier.
//
// --- Per frame ---
//  * Camera jitter (Jx,Jy) in [-0.5,0.5] input pixels, applied to projection as
//      P[2][0] += Jx*2/inputWidth;  P[2][1] -= Jy*2/inputHeight;
//    and the SAME (Jx,Jy) passed in execute params jitterOffsetX/Y.
//  * Motion vectors: current -> previous frame, in pixels (or NDC with USE_NDC_VELOCITY),
//    no camera-jitter component (unless JITTERED_MV set). Low-res by default; with HIGH_RES_MV
//    they must be target-res and dilated, and depth becomes optional.
//  * Resource states at the moment the recorded commands execute:
//      D3D12: all inputs D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
//             output D3D12_RESOURCE_STATE_UNORDERED_ACCESS.
//      VK:    inputs VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, output VK_IMAGE_LAYOUT_GENERAL;
//             accessed at VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT with SHADER_READ (inputs) and
//             SHADER_READ|SHADER_WRITE (output).
//    XeSS inserts NO barriers on inputs/outputs — the app does all transitions.
//  * xessD3D12Execute(ctx, gfxCmdList, &exec) / xessVKExecute(ctx, cmdBuf, &exec) records only;
//    the app submits. Resources must stay alive until GPU completion.
//    D3D12 with EXTERNAL_DESCRIPTOR_HEAP: exec.pDescriptorHeap + descriptorHeapOffset (bytes),
//    requiredDescriptorCount free descriptors (CBV_SRV_UAV, shader visible).
//  * exec.resetHistory = 1 on camera cuts. inputWidth/inputHeight = actual render size each frame.
//  * Output alpha is written as 1.0. Output placed at outputColorBase (default 0,0).
//
// --- Teardown ---
//  * Wait GPU idle (all command lists with XeSS work finished), then xessDestroyContext(ctx).
//    Release external heaps after that. For VK, destroy the context before vkDestroyDevice.
//
// --- Struct citation index (v3.0.2) ---
//  xess_version_t                 xess.h:55-70
//  xess_2d_t / xess_coord_t       xess.h:77-81 / :87
//  xess_quality_settings_t        xess.h:92-101
//  xess_init_flags_t              xess.h:106-127
//  xess_properties_t              xess.h:133-141
//  xess_result_t                  xess.h:147-182
//  xess_logging_level_t           xess.h:187-193
//  xess_app_log_callback_t        xess.h:200
//  xess_d3d12_execute_params_t    xess_d3d12.h:31-77
//  xess_d3d12_init_params_t       xess_d3d12.h:84-112
//  xess_vk_image_view_info        xess_vk.h:29-37
//  xess_vk_execute_params_t       xess_vk.h:45-87
//  xess_vk_init_params_t          xess_vk.h:94-124
