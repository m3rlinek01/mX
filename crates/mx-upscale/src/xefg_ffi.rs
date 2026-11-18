//! Intel XeSS Frame Generation (XeSS-FG, `xefg_swapchain` proxy swap chain API, D3D12) and
//! Intel Xe Low Latency (XeLL) — raw FFI transcription.
//!
//! Source: https://github.com/intel/xess  — release tag **v3.0.2** (published 2026-07-24)
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xess_fg/xefg_swapchain.h
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xess_fg/xefg_swapchain_d3d12.h
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xell/xell.h
//!   https://raw.githubusercontent.com/intel/xess/v3.0.2/inc/xell/xell_d3d12.h
//! Call-order notes use doc/xess_fg_developer_guide_english.md (guide v1.3) and
//! doc/xell_developer_guide_english.md (guide v1.3) at the same tag.
//! Not transcribed: inc/xess_fg/xefg_swapchain_debug.h.
//!
//! DLLs:
//!   * xefg* functions  -> `libxess_fg.dll` (import lib `lib/libxess_fg.lib`)
//!   * xell* functions  -> `libxell.dll`    (import lib `lib/libxell.lib`)
//!
//! XeLL IS REQUIRED by XeSS-FG: proxy swap-chain init fails without a XeLL context
//! (FG guide, "Connect with XeLL"), FG is disabled unless XeLL low-latency mode is on, and the
//! guide requires XeLL >= 1.3.0.
//!
//! Licence text exactly as it appears at the top of all four headers:
//!
//! ```text
//!  * Copyright (C) 2024 Intel Corporation
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
//! Transcription conventions (same as xess.rs):
//! * All structs are `#pragma pack(push, 8)`; identical to `#[repr(C)]` on x64. Asserted below.
//! * C enums (4 bytes, static-asserted in the headers) => `#[repr(transparent)]` i32 newtypes.
//! * No explicit calling convention => `extern "C"`.
//! * COM / Win32 types are opaque pointers; pass `windows` crate `.as_raw()` values.
//!   `D3D12_RESOURCE_STATES` and `DXGI_FORMAT` are C enums => i32 (same as the windows crate
//!   newtypes `D3D12_RESOURCE_STATES(i32)` / `DXGI_FORMAT(i32)`).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]
// (the C headers' names)
#![allow(clippy::upper_case_acronyms)]

use core::ffi::{c_char, c_void};

// ---------------------------------------------------------------------------------------------
// Opaque foreign types
// ---------------------------------------------------------------------------------------------

/// xefg_swapchain.h:62 `typedef struct _xefg_swapchain_handle_t* xefg_swapchain_handle_t;`
#[repr(C)]
pub struct _xefg_swapchain_handle_t {
    _private: [u8; 0],
}
pub type xefg_swapchain_handle_t = *mut _xefg_swapchain_handle_t;

/// xell.h:146 `typedef struct _xell_context_handle_t* xell_context_handle_t;`
#[repr(C)]
pub struct _xell_context_handle_t {
    _private: [u8; 0],
}
pub type xell_context_handle_t = *mut _xell_context_handle_t;

pub type ID3D12Device = c_void;
pub type ID3D12Resource = c_void;
pub type ID3D12Heap = c_void;
pub type ID3D12PipelineLibrary = c_void;
pub type ID3D12DescriptorHeap = c_void;
pub type ID3D12CommandQueue = c_void;
/// NOTE: TagFrameResource takes the base `ID3D12CommandList*` (not GraphicsCommandList).
/// Passing an ID3D12GraphicsCommandList pointer is fine (single-inheritance COM, same vtable base).
pub type ID3D12CommandList = c_void;
pub type IDXGISwapChain = c_void;
pub type IDXGIFactory2 = c_void;
/// Opaque here: use `windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_DESC1` and pass its address.
pub type DXGI_SWAP_CHAIN_DESC1 = c_void;
/// Opaque here: `windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FULLSCREEN_DESC`. May be null.
pub type DXGI_SWAP_CHAIN_FULLSCREEN_DESC = c_void;
pub type HWND = *mut c_void;
/// C enum `D3D12_RESOURCE_STATES`.
pub type D3D12_RESOURCE_STATES = i32;
/// C enum `DXGI_FORMAT`.
pub type DXGI_FORMAT = i32;
pub const DXGI_FORMAT_UNKNOWN: DXGI_FORMAT = 0;

/// Win32 GUID; `REFIID` == `const GUID*`. Bit-identical to `windows::core::GUID`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GUID {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}
pub type REFIID = *const GUID;

// =============================================================================================
// xefg_swapchain.h
// =============================================================================================

/// xefg_swapchain.h:56 `#define XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES UINT32_MAX`
pub const XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES: u32 = u32::MAX;

/// xefg_swapchain.h:70-77 `xefg_swapchain_resource_validity_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xefg_swapchain_resource_validity_t(pub i32);
/// Resource is valid until the next present (FG may defer/avoid a copy). Use incoming state
/// D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.
pub const XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT: xefg_swapchain_resource_validity_t = xefg_swapchain_resource_validity_t(0);
/// Resource only valid now; FG records a copy into the supplied command list. Use incoming state
/// D3D12_RESOURCE_STATE_COPY_SOURCE.
pub const XEFG_SWAPCHAIN_RV_ONLY_NOW: xefg_swapchain_resource_validity_t = xefg_swapchain_resource_validity_t(1);
pub const XEFG_SWAPCHAIN_RV_COUNT: xefg_swapchain_resource_validity_t = xefg_swapchain_resource_validity_t(2);

/// xefg_swapchain.h:82-98 `xefg_swapchain_init_flags_t` — OR'ed into `uint32_t initFlags`.
/// NOTE: bit values differ from XeSS-SR's XESS_INIT_FLAG_* — do not reuse SR flags.
pub type xefg_swapchain_init_flags_t = u32;
pub const XEFG_SWAPCHAIN_INIT_FLAG_NONE: u32 = 0;
pub const XEFG_SWAPCHAIN_INIT_FLAG_INVERTED_DEPTH: u32 = 1 << 0;
pub const XEFG_SWAPCHAIN_INIT_FLAG_EXTERNAL_DESCRIPTOR_HEAP: u32 = 1 << 1;
/// Deprecated. Setting this flag has no effect.
pub const XEFG_SWAPCHAIN_INIT_FLAG_HIGH_RES_MV: u32 = 1 << 2;
pub const XEFG_SWAPCHAIN_INIT_FLAG_USE_NDC_VELOCITY: u32 = 1 << 3;
pub const XEFG_SWAPCHAIN_INIT_FLAG_JITTERED_MV: u32 = 1 << 4;
/// UI texture RGB is NOT premultiplied by alpha.
pub const XEFG_SWAPCHAIN_INIT_FLAG_UITEXTURE_NOT_PREMUL_ALPHA: u32 = 1 << 5;

/// xefg_swapchain.h:103-111 `xefg_swapchain_resource_type_t`  (note order: HUDless is 0)
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xefg_swapchain_resource_type_t(pub i32);
pub const XEFG_SWAPCHAIN_RES_HUDLESS_COLOR: xefg_swapchain_resource_type_t = xefg_swapchain_resource_type_t(0);
pub const XEFG_SWAPCHAIN_RES_DEPTH: xefg_swapchain_resource_type_t = xefg_swapchain_resource_type_t(1);
pub const XEFG_SWAPCHAIN_RES_MOTION_VECTOR: xefg_swapchain_resource_type_t = xefg_swapchain_resource_type_t(2);
pub const XEFG_SWAPCHAIN_RES_UI: xefg_swapchain_resource_type_t = xefg_swapchain_resource_type_t(3);
/// Only resourceBase/resourceSize are read (letterbox interpolation area); rest ignored.
pub const XEFG_SWAPCHAIN_RES_BACKBUFFER: xefg_swapchain_resource_type_t = xefg_swapchain_resource_type_t(4);
pub const XEFG_SWAPCHAIN_RES_COUNT: xefg_swapchain_resource_type_t = xefg_swapchain_resource_type_t(5);

/// xefg_swapchain.h:119-134 `xefg_swapchain_version_t`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct xefg_swapchain_version_t {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

/// xefg_swapchain.h:141-145 `xefg_swapchain_2d_t`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct xefg_swapchain_2d_t {
    pub x: u32,
    pub y: u32,
}

/// xefg_swapchain.h:152-170 `xefg_swapchain_frame_constant_data_t`  (size 152)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xefg_swapchain_frame_constant_data_t {
    /// float4x4, ROW-MAJOR, without jitter.
    pub viewMatrix: [f32; 16], // 0
    /// float4x4, ROW-MAJOR, without jitter.
    pub projectionMatrix: [f32; 16], // 64
    /// Jitter X in [-0.5, 0.5].
    pub jitterOffsetX: f32, // 128
    /// Jitter Y in [-0.5, 0.5].
    pub jitterOffsetY: f32, // 132
    pub motionVectorScaleX: f32, // 136
    pub motionVectorScaleY: f32, // 140
    /// Non-zero => skip interpolation this frame (history kept for next).
    pub resetHistory: u32, // 144
    /// Frame time in ms, 0 if unknown (to be deprecated).
    pub frameRenderTime: f32, // 148
}

/// xefg_swapchain.h:177-189 `xefg_swapchain_properties_t`  (size 40)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct xefg_swapchain_properties_t {
    pub requiredDescriptorCount: u32,    // 0 (+4 pad)
    pub tempBufferHeapSize: u64,         // 8
    pub tempTextureHeapSize: u64,        // 16
    pub constantBufferSize: u64,         // 24
    pub maxSupportedInterpolations: u32, // 32 (+4 tail pad)
}

/// xefg_swapchain.h:195-255 `xefg_swapchain_result_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[must_use]
pub struct xefg_swapchain_result_t(pub i32);
impl xefg_swapchain_result_t {
    #[inline]
    pub fn is_error(self) -> bool {
        self.0 < 0
    }
}
pub const XEFG_SWAPCHAIN_RESULT_WARNING_OLD_DRIVER: xefg_swapchain_result_t = xefg_swapchain_result_t(2);
/// "Frame ID should be over 0."
pub const XEFG_SWAPCHAIN_RESULT_WARNING_TOO_FEW_FRAMES: xefg_swapchain_result_t = xefg_swapchain_result_t(3);
pub const XEFG_SWAPCHAIN_RESULT_WARNING_FRAMES_ID_MISMATCH: xefg_swapchain_result_t = xefg_swapchain_result_t(4);
pub const XEFG_SWAPCHAIN_RESULT_WARNING_MISSING_PRESENT_STATUS: xefg_swapchain_result_t = xefg_swapchain_result_t(5);
pub const XEFG_SWAPCHAIN_RESULT_WARNING_RESOURCE_SIZES_MISMATCH: xefg_swapchain_result_t = xefg_swapchain_result_t(6);
pub const XEFG_SWAPCHAIN_RESULT_SUCCESS: xefg_swapchain_result_t = xefg_swapchain_result_t(0);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_UNSUPPORTED_DEVICE: xefg_swapchain_result_t = xefg_swapchain_result_t(-1);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_UNSUPPORTED_DRIVER: xefg_swapchain_result_t = xefg_swapchain_result_t(-2);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_UNINITIALIZED: xefg_swapchain_result_t = xefg_swapchain_result_t(-3);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_INVALID_ARGUMENT: xefg_swapchain_result_t = xefg_swapchain_result_t(-4);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: xefg_swapchain_result_t = xefg_swapchain_result_t(-5);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_DEVICE: xefg_swapchain_result_t = xefg_swapchain_result_t(-6);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_NOT_IMPLEMENTED: xefg_swapchain_result_t = xefg_swapchain_result_t(-7);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_INVALID_CONTEXT: xefg_swapchain_result_t = xefg_swapchain_result_t(-8);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_OPERATION_IN_PROGRESS: xefg_swapchain_result_t = xefg_swapchain_result_t(-9);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_UNSUPPORTED: xefg_swapchain_result_t = xefg_swapchain_result_t(-10);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_CANT_LOAD_LIBRARY: xefg_swapchain_result_t = xefg_swapchain_result_t(-11);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_MISMATCH_INPUT_RESOURCES: xefg_swapchain_result_t = xefg_swapchain_result_t(-12);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_INCORRECT_OUTPUT_RESOURCES: xefg_swapchain_result_t = xefg_swapchain_result_t(-13);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_INCORRECT_INPUT_RESOURCES: xefg_swapchain_result_t = xefg_swapchain_result_t(-14);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_LATENCY_REDUCTION_UNSUPPORTED: xefg_swapchain_result_t = xefg_swapchain_result_t(-15);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_LATENCY_REDUCTION_FUNCTION_MISSING: xefg_swapchain_result_t = xefg_swapchain_result_t(-16);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_HRESULT_FAILURE: xefg_swapchain_result_t = xefg_swapchain_result_t(-17);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_DXGI_INVALID_CALL: xefg_swapchain_result_t = xefg_swapchain_result_t(-18);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_POINTER_STILL_IN_USE: xefg_swapchain_result_t = xefg_swapchain_result_t(-19);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_INVALID_DESCRIPTOR_HEAP: xefg_swapchain_result_t = xefg_swapchain_result_t(-20);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_WRONG_CALL_ORDER: xefg_swapchain_result_t = xefg_swapchain_result_t(-21);
pub const XEFG_SWAPCHAIN_RESULT_ERROR_UNKNOWN: xefg_swapchain_result_t = xefg_swapchain_result_t(-1000);

/// xefg_swapchain.h:260-266 `xefg_swapchain_logging_level_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xefg_swapchain_logging_level_t(pub i32);
pub const XEFG_SWAPCHAIN_LOGGING_LEVEL_DEBUG: xefg_swapchain_logging_level_t = xefg_swapchain_logging_level_t(0);
pub const XEFG_SWAPCHAIN_LOGGING_LEVEL_INFO: xefg_swapchain_logging_level_t = xefg_swapchain_logging_level_t(1);
pub const XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING: xefg_swapchain_logging_level_t = xefg_swapchain_logging_level_t(2);
pub const XEFG_SWAPCHAIN_LOGGING_LEVEL_ERROR: xefg_swapchain_logging_level_t = xefg_swapchain_logging_level_t(3);

/// xefg_swapchain.h:280-294 `xefg_swapchain_ui_mode_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xefg_swapchain_ui_mode_t(pub i32);
/// Auto-select from provided inputs (HUD-less color, UI texture).
pub const XEFG_SWAPCHAIN_UI_MODE_AUTO: xefg_swapchain_ui_mode_t = xefg_swapchain_ui_mode_t(0);
/// Interpolate back buffer, no UI composition.
pub const XEFG_SWAPCHAIN_UI_MODE_NONE: xefg_swapchain_ui_mode_t = xefg_swapchain_ui_mode_t(1);
pub const XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_UITEXTURE: xefg_swapchain_ui_mode_t = xefg_swapchain_ui_mode_t(2);
pub const XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE: xefg_swapchain_ui_mode_t = xefg_swapchain_ui_mode_t(3);
pub const XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_HUDLESS: xefg_swapchain_ui_mode_t = xefg_swapchain_ui_mode_t(4);
pub const XEFG_SWAPCHAIN_UI_MODE_BACKBUFFER_HUDLESS_UITEXTURE: xefg_swapchain_ui_mode_t = xefg_swapchain_ui_mode_t(5);

/// xefg_swapchain.h:300-308 `xefg_swapchain_present_status_t`  (size 12)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xefg_swapchain_present_status_t {
    /// Frames sent to present during the last call (rendered + generated).
    pub framesPresented: u32,
    pub frameGenResult: xefg_swapchain_result_t,
    pub isFrameGenEnabled: u32,
}

/// xefg_swapchain.h:317
/// `typedef void (*xefg_swapchain_app_log_callback_t)(const char* message, xefg_swapchain_logging_level_t loggingLevel, void* userData);`
/// Any thread. `userData` must stay valid until xefgSwapChainDestroy returns.
pub type xefg_swapchain_app_log_callback_t = Option<unsafe extern "C" fn(message: *const c_char, loggingLevel: xefg_swapchain_logging_level_t, userData: *mut c_void)>;

/// xefg_swapchain.h:492-498 anonymous enum `xefg_swapchain_ui_composition_state_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xefg_swapchain_ui_composition_state_t(pub i32);
/// Default: UI mode set at init has no effect.
pub const XEFG_SWAPCHAIN_UI_COMPOSITION_STATE_DISABLED: xefg_swapchain_ui_composition_state_t = xefg_swapchain_ui_composition_state_t(0);
pub const XEFG_SWAPCHAIN_UI_COMPOSITION_STATE_ENABLED: xefg_swapchain_ui_composition_state_t = xefg_swapchain_ui_composition_state_t(1);

// =============================================================================================
// xefg_swapchain_d3d12.h
// =============================================================================================

/// xefg_swapchain_d3d12.h:34-81 `xefg_swapchain_d3d12_init_params_t`  (size 72)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xefg_swapchain_d3d12_init_params_t {
    /// InitFromSwapChain: app swap chain, refcount must be exactly 1, ownership transferred
    /// (it is RELEASED by FG). InitFromSwapChainDesc: must be NULL.
    pub pApplicationSwapChain: *mut IDXGISwapChain, // 0
    /// OR of XEFG_SWAPCHAIN_INIT_FLAG_*.
    pub initFlags: u32, // 8
    /// 1..=max supported, or XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES.
    pub maxInterpolatedFrames: u32, // 12
    pub creationNodeMask: u32, // 16
    pub visibleNodeMask: u32,  // 20
    /// Optional, D3D12_HEAP_TYPE_DEFAULT, or null.
    pub pTempBufferHeap: *mut ID3D12Heap, // 24
    pub bufferHeapOffset: u64, // 32
    /// Optional, D3D12_HEAP_TYPE_DEFAULT, or null.
    pub pTempTextureHeap: *mut ID3D12Heap, // 40
    pub textureHeapOffset: u64, // 48
    /// Optional pipeline library or null.
    pub pPipelineLibrary: *mut ID3D12PipelineLibrary, // 56
    /// Used only when UI composition is enabled (xefgSwapChainSetUiCompositionState).
    pub uiMode: xefg_swapchain_ui_mode_t, // 64 (+4 tail pad => 72)
}

/// xefg_swapchain_d3d12.h:89-103 `xefg_swapchain_d3d12_resource_data_t`  (size 40)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xefg_swapchain_d3d12_resource_data_t {
    pub r#type: xefg_swapchain_resource_type_t,       // 0   (C field name: `type`)
    pub validity: xefg_swapchain_resource_validity_t, // 4
    /// Offset of the valid region.
    pub resourceBase: xefg_swapchain_2d_t, // 8
    /// Width/height of the valid region.
    pub resourceSize: xefg_swapchain_2d_t, // 16
    pub pResource: *mut ID3D12Resource,               // 24
    /// State the resource is in at that point of the tagging command list; FG adds the
    /// transitions from/to it.
    pub incomingState: D3D12_RESOURCE_STATES, // 32 (+4 tail pad => 40)
}

// =============================================================================================
// xell.h / xell_d3d12.h
// =============================================================================================

/// xell.h:43-65 `xell_result_t`  (note: no -5, -9)
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[must_use]
pub struct xell_result_t(pub i32);
pub const XELL_RESULT_SUCCESS: xell_result_t = xell_result_t(0);
pub const XELL_RESULT_ERROR_UNSUPPORTED_DEVICE: xell_result_t = xell_result_t(-1);
pub const XELL_RESULT_ERROR_UNSUPPORTED_DRIVER: xell_result_t = xell_result_t(-2);
pub const XELL_RESULT_ERROR_UNINITIALIZED: xell_result_t = xell_result_t(-3);
pub const XELL_RESULT_ERROR_INVALID_ARGUMENT: xell_result_t = xell_result_t(-4);
pub const XELL_RESULT_ERROR_DEVICE: xell_result_t = xell_result_t(-6);
pub const XELL_RESULT_ERROR_NOT_IMPLEMENTED: xell_result_t = xell_result_t(-7);
pub const XELL_RESULT_ERROR_INVALID_CONTEXT: xell_result_t = xell_result_t(-8);
pub const XELL_RESULT_ERROR_UNSUPPORTED: xell_result_t = xell_result_t(-10);
pub const XELL_RESULT_ERROR_UNKNOWN: xell_result_t = xell_result_t(-1000);

/// xell.h:72-83 `xell_latency_marker_type_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xell_latency_marker_type_t(pub i32);
pub const XELL_SIMULATION_START: xell_latency_marker_type_t = xell_latency_marker_type_t(0); // required
pub const XELL_SIMULATION_END: xell_latency_marker_type_t = xell_latency_marker_type_t(1); // required
pub const XELL_RENDERSUBMIT_START: xell_latency_marker_type_t = xell_latency_marker_type_t(2); // required
pub const XELL_RENDERSUBMIT_END: xell_latency_marker_type_t = xell_latency_marker_type_t(3); // required
pub const XELL_PRESENT_START: xell_latency_marker_type_t = xell_latency_marker_type_t(4); // required
pub const XELL_PRESENT_END: xell_latency_marker_type_t = xell_latency_marker_type_t(5); // required
pub const XELL_INPUT_SAMPLE: xell_latency_marker_type_t = xell_latency_marker_type_t(6);
pub const XELL_MARKER_COUNT: xell_latency_marker_type_t = xell_latency_marker_type_t(7);

/// xell.h:86-99 `xell_sleep_params_t`  (size 8)
/// C declares three bit-fields in one `uint32_t` storage unit:
///   `uint32_t bLowLatencyMode : 1; uint32_t bLowLatencyBoost : 1; uint32_t reserved : 30;`
/// MSVC allocates bit-fields LSB-first, so: bit0 = bLowLatencyMode, bit1 = bLowLatencyBoost.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct xell_sleep_params_t {
    /// Frame cap in microseconds; 0 = no cap (e.g. 33333 for 30 FPS).
    pub minimumIntervalUs: u32,
    /// Bit-field word; use XELL_SLEEP_FLAG_* below.
    pub flags: u32,
}
/// `bLowLatencyMode : 1` — enables latency reduction.
pub const XELL_SLEEP_FLAG_LOW_LATENCY_MODE: u32 = 1 << 0;
/// `bLowLatencyBoost : 1` — "Boost is not supported as of now."
pub const XELL_SLEEP_FLAG_LOW_LATENCY_BOOST: u32 = 1 << 1;

/// xell.h:105-111 `xell_logging_level_t`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct xell_logging_level_t(pub i32);
pub const XELL_LOGGING_LEVEL_DEBUG: xell_logging_level_t = xell_logging_level_t(0);
pub const XELL_LOGGING_LEVEL_INFO: xell_logging_level_t = xell_logging_level_t(1);
pub const XELL_LOGGING_LEVEL_WARNING: xell_logging_level_t = xell_logging_level_t(2);
pub const XELL_LOGGING_LEVEL_ERROR: xell_logging_level_t = xell_logging_level_t(3);

/// xell.h:118 `typedef void (*xell_app_log_callback_t)(const char* message, xell_logging_level_t loggingLevel);`
/// (no user data)
pub type xell_app_log_callback_t = Option<unsafe extern "C" fn(message: *const c_char, loggingLevel: xell_logging_level_t)>;

/// xell.h:126-143 `xell_version_t`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct xell_version_t {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

/// xell.h:211-227 `xell_frame_report_t`  (size 96). Timestamps in ns.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct xell_frame_report_t {
    pub m_frame_id: u32,     // 0 (+4 pad)
    pub m_sim_start_ts: u64, // 8
    pub m_sim_end_ts: u64,
    pub m_render_submit_start_ts: u64,
    pub m_render_submit_end_ts: u64,
    pub m_present_start_ts: u64,
    pub m_present_end_ts: u64,
    pub reserved1: u64,
    pub reserved2: u64,
    pub reserved3: u64,
    pub reserved4: u64,
    pub reserved5: u64,
}

// ---------------------------------------------------------------------------------------------
// Compile-time layout checks (64-bit)
// ---------------------------------------------------------------------------------------------
#[cfg(target_pointer_width = "64")]
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<xefg_swapchain_version_t>() == 8);
    assert!(size_of::<xefg_swapchain_2d_t>() == 8);
    assert!(size_of::<xefg_swapchain_frame_constant_data_t>() == 152);
    assert!(offset_of!(xefg_swapchain_frame_constant_data_t, jitterOffsetX) == 128);
    assert!(offset_of!(xefg_swapchain_frame_constant_data_t, frameRenderTime) == 148);
    assert!(size_of::<xefg_swapchain_properties_t>() == 40);
    assert!(offset_of!(xefg_swapchain_properties_t, tempBufferHeapSize) == 8);
    assert!(offset_of!(xefg_swapchain_properties_t, maxSupportedInterpolations) == 32);
    assert!(size_of::<xefg_swapchain_present_status_t>() == 12);
    assert!(size_of::<xefg_swapchain_d3d12_init_params_t>() == 72);
    assert!(offset_of!(xefg_swapchain_d3d12_init_params_t, initFlags) == 8);
    assert!(offset_of!(xefg_swapchain_d3d12_init_params_t, pTempBufferHeap) == 24);
    assert!(offset_of!(xefg_swapchain_d3d12_init_params_t, pPipelineLibrary) == 56);
    assert!(offset_of!(xefg_swapchain_d3d12_init_params_t, uiMode) == 64);
    assert!(size_of::<xefg_swapchain_d3d12_resource_data_t>() == 40);
    assert!(offset_of!(xefg_swapchain_d3d12_resource_data_t, resourceBase) == 8);
    assert!(offset_of!(xefg_swapchain_d3d12_resource_data_t, pResource) == 24);
    assert!(offset_of!(xefg_swapchain_d3d12_resource_data_t, incomingState) == 32);
    assert!(size_of::<xell_sleep_params_t>() == 8);
    assert!(size_of::<xell_version_t>() == 8);
    assert!(size_of::<xell_frame_report_t>() == 96);
    assert!(size_of::<GUID>() == 16);
};

// =============================================================================================
// Exported functions
// =============================================================================================

pub const XEFG_DLL_NAME: &str = "libxess_fg.dll";
pub const XELL_DLL_NAME: &str = "libxell.dll";

// ---- libxess_fg.dll : xefg_swapchain.h ----

/// "xefgSwapChainGetVersion" — xefg_swapchain.h:334
pub type PFN_xefgSwapChainGetVersion = unsafe extern "C" fn(pVersion: *mut xefg_swapchain_version_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainGetProperties" — xefg_swapchain.h:351. Before proxy init only
/// maxSupportedInterpolations is filled (rest zero); use xefgSwapChainD3D12GetProperties then.
/// NOT thread-safe.
pub type PFN_xefgSwapChainGetProperties = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pProperties: *mut xefg_swapchain_properties_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainTagFrameConstants" — xefg_swapchain.h:365
pub type PFN_xefgSwapChainTagFrameConstants =
    unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, presentId: u32, pConstants: *const xefg_swapchain_frame_constant_data_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetEnabled" — xefg_swapchain.h:377. enable: non-zero = on.
pub type PFN_xefgSwapChainSetEnabled = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, enable: u32) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetPresentId" — xefg_swapchain.h:390. ID of the frame the NEXT Present shows.
pub type PFN_xefgSwapChainSetPresentId = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, presentId: u32) -> xefg_swapchain_result_t;

/// "xefgSwapChainGetLastPresentStatus" — xefg_swapchain.h:401
pub type PFN_xefgSwapChainGetLastPresentStatus = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pPresentStatus: *mut xefg_swapchain_present_status_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetLoggingCallback" — xefg_swapchain.h:412
pub type PFN_xefgSwapChainSetLoggingCallback = unsafe extern "C" fn(
    hSwapChain: xefg_swapchain_handle_t,
    loggingLevel: xefg_swapchain_logging_level_t,
    loggingCallback: xefg_swapchain_app_log_callback_t,
    userData: *mut c_void,
) -> xefg_swapchain_result_t;

/// "xefgSwapChainDestroy" — xefg_swapchain.h:421. Fails with POINTER_STILL_IN_USE if proxy
/// swap-chain refs are outstanding. Releases the native swap chain. NOT thread-safe.
pub type PFN_xefgSwapChainDestroy = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetLatencyReduction" — xefg_swapchain.h:431. hXeLLContext is the
/// `xell_context_handle_t` passed as `void*`.
pub type PFN_xefgSwapChainSetLatencyReduction = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, hXeLLContext: *mut c_void) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetSceneChangeThreshold" — xefg_swapchain.h:446. [0,1], default 0.7.
pub type PFN_xefgSwapChainSetSceneChangeThreshold = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, threshold: f32) -> xefg_swapchain_result_t;

/// "xefgSwapChainGetPipelineBuildStatus" — xefg_swapchain.h:460
pub type PFN_xefgSwapChainGetPipelineBuildStatus = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetNumInterpolatedFrames" — xefg_swapchain.h:486. 1..=maxInterpolatedFrames.
pub type PFN_xefgSwapChainSetNumInterpolatedFrames = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, numInterpolatedFrames: u32) -> xefg_swapchain_result_t;

/// "xefgSwapChainSetUiCompositionState" — xefg_swapchain.h:511
pub type PFN_xefgSwapChainSetUiCompositionState = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, state: xefg_swapchain_ui_composition_state_t) -> xefg_swapchain_result_t;

// ---- libxess_fg.dll : xefg_swapchain_d3d12.h ----

/// "xefgSwapChainD3D12CreateContext" — xefg_swapchain_d3d12.h:120. Also the support check.
/// NOT thread-safe.
pub type PFN_xefgSwapChainD3D12CreateContext = unsafe extern "C" fn(pDevice: *mut ID3D12Device, phSwapChain: *mut xefg_swapchain_handle_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12BuildPipelines" — xefg_swapchain_d3d12.h:139.
/// NOTE: `blocking` is `uint8_t` here (XeSS-SR uses C `bool`) — both 1 byte.
pub type PFN_xefgSwapChainD3D12BuildPipelines =
    unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pPipelineLibrary: *mut ID3D12PipelineLibrary, blocking: u8, initFlags: u32) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12GetProperties" — xefg_swapchain_d3d12.h:168. Before init: pass the
/// intended initParams + back-buffer size/format. After init: (null, 0, 0, DXGI_FORMAT_UNKNOWN).
pub type PFN_xefgSwapChainD3D12GetProperties = unsafe extern "C" fn(
    context: xefg_swapchain_handle_t,
    initParams: *const xefg_swapchain_d3d12_init_params_t,
    backBufferWidth: u32,
    backBufferHeight: u32,
    backBufferFormat: DXGI_FORMAT,
    properties: *mut xefg_swapchain_properties_t,
) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12InitFromSwapChain" — xefg_swapchain_d3d12.h:190. NOT thread-safe.
/// Fails in exclusive fullscreen.
pub type PFN_xefgSwapChainD3D12InitFromSwapChain =
    unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pCmdQueue: *mut ID3D12CommandQueue, pInitParams: *const xefg_swapchain_d3d12_init_params_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12InitFromSwapChainDesc" — xefg_swapchain_d3d12.h:216. NOT thread-safe.
/// The window must have no existing swap chain. pInitParams->pApplicationSwapChain must be NULL.
/// pFullscreenDesc may be NULL.
pub type PFN_xefgSwapChainD3D12InitFromSwapChainDesc = unsafe extern "C" fn(
    hSwapChain: xefg_swapchain_handle_t,
    hWnd: HWND,
    pSwapChainDesc: *const DXGI_SWAP_CHAIN_DESC1,
    pFullscreenDesc: *const DXGI_SWAP_CHAIN_FULLSCREEN_DESC,
    pCmdQueue: *mut ID3D12CommandQueue,
    pDxgiFactory: *mut IDXGIFactory2,
    pInitParams: *const xefg_swapchain_d3d12_init_params_t,
) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12GetSwapChainPtr" — xefg_swapchain_d3d12.h:232. QueryInterface-style:
/// returns an AddRef'd proxy (IDXGISwapChain4 implemented) in *ppSwapChain.
pub type PFN_xefgSwapChainD3D12GetSwapChainPtr = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, riid: REFIID, ppSwapChain: *mut *mut c_void) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12TagFrameResource" — xefg_swapchain_d3d12.h:255.
/// pCmdList is optional (may be NULL) for RV_UNTIL_NEXT_PRESENT; required for RV_ONLY_NOW.
pub type PFN_xefgSwapChainD3D12TagFrameResource =
    unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pCmdList: *mut ID3D12CommandList, presentId: u32, pResData: *const xefg_swapchain_d3d12_resource_data_t) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12SetDescriptorHeap" — xefg_swapchain_d3d12.h:271. CBV_SRV_UAV, shader
/// visible; offset in BYTES. Required before Present when EXTERNAL_DESCRIPTOR_HEAP is set.
pub type PFN_xefgSwapChainD3D12SetDescriptorHeap =
    unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pDescriptorHeap: *mut ID3D12DescriptorHeap, descriptorHeapOffsetInBytes: u32) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12UpdateExternalHeapOnResize" — xefg_swapchain_d3d12.h:294. Takes effect at
/// the next ResizeBuffers/ResizeBuffers1. Offsets must be 0 when the heap is null.
pub type PFN_xefgSwapChainD3D12UpdateExternalHeapOnResize = unsafe extern "C" fn(
    hSwapChain: xefg_swapchain_handle_t,
    tempBufferHeap: *mut ID3D12Heap,
    tempBufferHeapOffset: u64,
    tempTextureHeap: *mut ID3D12Heap,
    tempTextureHeapOffset: u64,
) -> xefg_swapchain_result_t;

/// "xefgSwapChainD3D12GetInitializationParameters" — xefg_swapchain_d3d12.h:319.
/// Returned pApplicationSwapChain is NULL; maxInterpolatedFrames is the resolved value.
pub type PFN_xefgSwapChainD3D12GetInitializationParameters = unsafe extern "C" fn(hSwapChain: xefg_swapchain_handle_t, pParams: *mut xefg_swapchain_d3d12_init_params_t) -> xefg_swapchain_result_t;

// ---- libxell.dll : xell.h / xell_d3d12.h ----

/// "xellD3D12CreateContext" — xell_d3d12.h:28
pub type PFN_xellD3D12CreateContext = unsafe extern "C" fn(device: *mut ID3D12Device, out_context: *mut xell_context_handle_t) -> xell_result_t;

/// "xellDestroyContext" — xell.h:153
pub type PFN_xellDestroyContext = unsafe extern "C" fn(context: xell_context_handle_t) -> xell_result_t;

/// "xellSetSleepMode" — xell.h:161. Call once per settings change.
pub type PFN_xellSetSleepMode = unsafe extern "C" fn(context: xell_context_handle_t, param: *const xell_sleep_params_t) -> xell_result_t;

/// "xellGetSleepMode" — xell.h:169
pub type PFN_xellGetSleepMode = unsafe extern "C" fn(context: xell_context_handle_t, param: *mut xell_sleep_params_t) -> xell_result_t;

/// "xellSleep" — xell.h:177. First XeLL call for a new frame_id; once per frame.
pub type PFN_xellSleep = unsafe extern "C" fn(context: xell_context_handle_t, frame_id: u32) -> xell_result_t;

/// "xellAddMarkerData" — xell.h:186
pub type PFN_xellAddMarkerData = unsafe extern "C" fn(context: xell_context_handle_t, frame_id: u32, marker: xell_latency_marker_type_t) -> xell_result_t;

/// "xellGetVersion" — xell.h:193
pub type PFN_xellGetVersion = unsafe extern "C" fn(pVersion: *mut xell_version_t) -> xell_result_t;

/// "xellSetLoggingCallback" — xell.h:203
pub type PFN_xellSetLoggingCallback = unsafe extern "C" fn(hContext: xell_context_handle_t, loggingLevel: xell_logging_level_t, loggingCallback: xell_app_log_callback_t) -> xell_result_t;

/// "xellGetFramesReports" — xell.h:236. `outdata` must point to an array of 64 reports
/// ("Last 64 frames reports").
pub type PFN_xellGetFramesReports = unsafe extern "C" fn(context: xell_context_handle_t, outdata: *mut xell_frame_report_t) -> xell_result_t;

// =============================================================================================
// CALL SEQUENCE (headers + doc/xess_fg_developer_guide_english.md + doc/xell_developer_guide_english.md, v3.0.2)
// =============================================================================================
//
// --- Init ---
//  1. D3D12 device + DIRECT command queue (this queue is the one passed to FG init; FG keeps a
//     reference and runs interpolation on it; it must outlive the FG context).
//  2. XeLL:   xellD3D12CreateContext(device, &xell)          (xell_d3d12.h:28)
//             xellSetLoggingCallback(xell, level, cb)         (xell.h:203)
//             xellSetSleepMode(xell, &{minimumIntervalUs: 0 or cap, flags: LOW_LATENCY_MODE})
//     XeLL low-latency MUST be enabled before FG is enabled (may be turned off while FG is off).
//     On non-Intel GPUs XeLL only reduces latency while XeSS-FG is enabled.
//  3. FG:     xefgSwapChainD3D12CreateContext(device, &fg)    (support check; 10-50 ms)
//             xefgSwapChainSetLoggingCallback(fg, level, cb, userData)
//             xefgSwapChainSetLatencyReduction(fg, xell as *mut c_void)   <- before proxy init,
//                                                   proxy init FAILS without XeLL.
//     optional: xefgSwapChainD3D12BuildPipelines(fg, lib, 0 /*non-blocking*/, initFlags) early;
//               poll xefgSwapChainGetPipelineBuildStatus; same flags+library at init.
//  4. xefgSwapChainD3D12GetProperties(fg, &initParams, bbW, bbH, bbFormat, &props)
//     -> requiredDescriptorCount / heap sizes (only needed for external heap / descriptor heap).
//  5. Proxy swap chain, either:
//       a) xefgSwapChainD3D12InitFromSwapChainDesc(fg, hwnd, &desc1, &fsDescOrNull, queue,
//          dxgiFactory2, &initParams{pApplicationSwapChain: NULL, maxInterpolatedFrames:
//          XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES, uiMode: AUTO, ...})
//          (window must not already own a swap chain; preferred for us — no refcount games), or
//       b) xefgSwapChainD3D12InitFromSwapChain(fg, queue, &initParams{pApplicationSwapChain:
//          existing swap chain with refcount EXACTLY 1, ownership transferred; it gets released
//          and recreated; fails in exclusive fullscreen}).
//     Only ONE proxy swap chain may exist at a time. Exclusive fullscreen is unsupported.
//  6. xefgSwapChainD3D12GetSwapChainPtr(fg, &IID_IDXGISwapChain4, &ptr) — from now on ALL
//     swap-chain calls (GetBuffer, Present, ResizeBuffers, ...) go through this proxy.
//  7. xefgSwapChainSetEnabled(fg, 1)  (FG is disabled by default). Turn off/reduce motion blur.
//     UI composition is off by default (UI gets interpolated); see SetUiCompositionState.
//
// --- Per frame (frameId: u32, same value for XeLL frame_id and FG presentId) ---
//   Header warning XEFG_SWAPCHAIN_RESULT_WARNING_TOO_FEW_FRAMES says "Frame ID should be over 0",
//   so start at 1 (the guide says the initial value does not matter; >0 is the safe choice).
//
//   frameId += 1
//   xellSleep(xell, frameId)                             // first XeLL call for this id; before input
//   xellAddMarkerData(xell, frameId, XELL_SIMULATION_START)
//     ... sample input, simulate ...
//   xellAddMarkerData(xell, frameId, XELL_SIMULATION_END)
//   xellAddMarkerData(xell, frameId, XELL_RENDERSUBMIT_START)
//     ... record scene (jittered), XeSS-SR execute, post, UI into back buffer ...
//     Tag inputs for THIS frameId (any thread, but must complete before Present):
//       xefgSwapChainD3D12TagFrameResource(fg, cmdList, frameId, &{type: MOTION_VECTOR, ...}) REQUIRED
//       xefgSwapChainD3D12TagFrameResource(fg, cmdList, frameId, &{type: DEPTH, ...})         REQUIRED
//       optional HUDLESS_COLOR / UI (depends on UI mode), BACKBUFFER (letterbox area only).
//         - RV_UNTIL_NEXT_PRESENT: incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
//           cmdList may be NULL, resource must not be overwritten until after Present(frameId).
//         - RV_ONLY_NOW: incomingState = D3D12_RESOURCE_STATE_COPY_SOURCE, cmdList REQUIRED; FG
//           records a copy (+ barriers from/to incomingState) into it; resource reusable once that
//           list is submitted (guide example assumes later writes go to the same queue).
//         - incomingState = the state the resource is in at that point in cmdList; FG adds the
//           transitions from/to it, so the resource is back in incomingState afterwards.
//         - MV and depth: same resourceSize (=> low-res SR inputs). HUD-less, UI, back buffer: same
//           resourceSize; HUD-less/UI must match back-buffer format.
//         - MV: current->previous, pixels (or NDC with USE_NDC_VELOCITY), no jitter component
//           unless JITTERED_MV. Depth: any format, INVERTED_DEPTH for reverse-Z.
//       xefgSwapChainTagFrameConstants(fg, frameId, &{view, proj (row-major, UNjittered),
//           jitterOffsetX/Y, motionVectorScaleX/Y, resetHistory, frameRenderTime})
//       if EXTERNAL_DESCRIPTOR_HEAP: xefgSwapChainD3D12SetDescriptorHeap(fg, heap, offsetBytes)
//     ExecuteCommandLists(...)   // ALL tagging command lists must be submitted before Present
//   xellAddMarkerData(xell, frameId, XELL_RENDERSUBMIT_END)
//   xellAddMarkerData(xell, frameId, XELL_PRESENT_START)
//   xefgSwapChainSetPresentId(fg, frameId)              // REQUIRED, before Present
//   proxy->Present(sync, flags)                          // FG submits interpolation work here
//   xellAddMarkerData(xell, frameId, XELL_PRESENT_END)
//   optional: xefgSwapChainGetLastPresentStatus(fg, &st) (framesPresented for displayed FPS)
//
//   Toggle: xefgSwapChainSetEnabled(fg, 0/1) any time (disable in menus/pause; re-enable needs
//   one extra frame of history). Camera cut: resetHistory=1 in frame constants instead.
//   Leaving exclusive fullscreen auto-disables FG; call SetEnabled(1) again.
//
// --- Resize / window-state change ---
//   Release all back-buffer refs, wait for (or reset unsubmitted) tagging command lists (they may
//   reference proxy-owned resources), then proxy->ResizeBuffers(...). Queues passed to
//   ResizeBuffers1 are ignored. External heaps: xefgSwapChainD3D12GetProperties(fg, NULL, newW,
//   newH, DXGI_FORMAT_UNKNOWN, &p) then xefgSwapChainD3D12UpdateExternalHeapOnResize before
//   ResizeBuffers.
//
// --- Teardown (order matters) ---
//   1. GPU idle; no tagging / SetEnabled calls in flight.
//   2. Release EVERY reference to the proxy swap chain (incl. back buffers obtained from it).
//   3. xefgSwapChainDestroy(fg)       — releases native swap chain; returns
//      ERROR_POINTER_STILL_IN_USE if proxy refs remain. XeLL must still be alive here.
//   4. xellDestroyContext(xell)       — XeLL strictly AFTER FG.
//   5. Command queue / device after that.
//
// --- Threading ---
//   Not thread-safe: xefgSwapChainD3D12CreateContext, InitFromSwapChain, InitFromSwapChainDesc,
//   xefgSwapChainGetProperties, xefgSwapChainDestroy (and SetNumInterpolatedFrames /
//   GetInitializationParameters vs. init/destroy). Everything else may be called from any thread.
//
// --- Struct citation index (v3.0.2) ---
//  XEFG_SWAPCHAIN_USE_MAX_SUPPORTED_INTERPOLATED_FRAMES  xefg_swapchain.h:56
//  xefg_swapchain_resource_validity_t     xefg_swapchain.h:70-77
//  xefg_swapchain_init_flags_t            xefg_swapchain.h:82-98
//  xefg_swapchain_resource_type_t         xefg_swapchain.h:103-111
//  xefg_swapchain_version_t               xefg_swapchain.h:119-134
//  xefg_swapchain_2d_t                    xefg_swapchain.h:141-145
//  xefg_swapchain_frame_constant_data_t   xefg_swapchain.h:152-170
//  xefg_swapchain_properties_t            xefg_swapchain.h:177-189
//  xefg_swapchain_result_t                xefg_swapchain.h:195-255
//  xefg_swapchain_logging_level_t         xefg_swapchain.h:260-266
//  xefg_swapchain_ui_mode_t               xefg_swapchain.h:280-294
//  xefg_swapchain_present_status_t        xefg_swapchain.h:300-308
//  xefg_swapchain_app_log_callback_t      xefg_swapchain.h:317
//  xefg_swapchain_ui_composition_state_t  xefg_swapchain.h:492-498
//  xefg_swapchain_d3d12_init_params_t     xefg_swapchain_d3d12.h:34-81
//  xefg_swapchain_d3d12_resource_data_t   xefg_swapchain_d3d12.h:89-103
//  xell_result_t                          xell.h:43-65
//  xell_latency_marker_type_t             xell.h:72-83
//  xell_sleep_params_t                    xell.h:86-99
//  xell_logging_level_t                   xell.h:105-111
//  xell_app_log_callback_t                xell.h:118
//  xell_version_t                         xell.h:126-143
//  xell_frame_report_t                    xell.h:211-227
