// FidelityFX API: frame generation (FSR 3.1 / FSR 4 frame generation + frame-generation swapchain)
// Rust FFI transcription for DirectX 12 and Vulkan.
//
// Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK  tag v2.4.0 (released 2026-10-28)
// Headers transcribed (raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/...):
//   Kits/FidelityFX/api/include/ffx_api.h                                         ("api.h")
//   Kits/FidelityFX/api/include/ffx_api_types.h                                   ("types.h")
//   Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h                               ("api_dx12.h")
//   Kits/FidelityFX/api/include/vk/ffx_api_vk.h                                   ("api_vk.h")
//   Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h                 ("fg.h")
//   Kits/FidelityFX/framegeneration/include/ffx_framegeneration_api_types.h       ("fg_types.h")
//   Kits/FidelityFX/framegeneration/include/dx12/ffx_api_framegeneration_dx12.h   ("fg_dx12.h")
//   Kits/FidelityFX/framegeneration/include/vk/ffx_api_framegeneration_vk.h       ("fg_vk.h")
// "file:Lnn" below cites the line of the struct / define in that tag.
// Header versions in this tag: FFX_FRAMEGENERATION_VERSION 4.0.2,
// FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION 4.0.0, FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION 4.0.0.
// DLLs: amd_fidelityfx_loader_dx12.dll / amd_fidelityfx_loader_vk.dll (the loader pulls in
// amd_fidelityfx_framegeneration_dx12.dll / _vk.dll from next to the exe or next to the loader).
//
// Licence text as it appears at the top of every header above:
//
// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ---------------------------------------------------------------------------------------------
// Transcription notes
//  * C `bool` (<stdbool.h>, 1 byte) -> Rust `bool`. Padding after bool fields is produced by
//    #[repr(C)] exactly as the C compiler does; do NOT reorder fields.
//  * C `wchar_t` on Windows -> u16.
//  * All FFX entry points are `__declspec(dllexport)` with no explicit calling convention
//    -> extern "C". All FFX callback typedefs likewise have no convention -> extern "C".
//    Vulkan PFN_vk* types (VKAPI_PTR) -> extern "system" (ash's PFN types already are).
//    On x86_64 Windows "C" and "system" are the same ABI anyway.
//  * D3D12 / DXGI interface pointers -> *mut c_void (the header uses the real COM types, named
//    on each field). Vulkan types come from `ash::vk` (ash 0.38, already a dependency of
//    omsi-render): dispatchable handles (VkDevice, VkPhysicalDevice, VkQueue, VkCommandBuffer)
//    are pointer-sized, non-dispatchable (VkSwapchainKHR, VkImage, ...) are u64.
//  * The header struct and the types already used for upscaling are reused from
//    temporal/fidelityfx.rs: Header {kind: u64, next: *const Header} == ffxApiHeader,
//    Resource == FfxApiResource (48 bytes on 64-bit), Dimensions == FfxApiDimensions2D,
//    Coords == FfxApiFloatCoords2D. They are private there: either paste this file into that
//    module or make them pub(super). C declares `pNext` as non-const; *const is ABI-identical.
//  * The size asserts at the bottom were computed by hand for 64-bit targets from the C
//    layout rules; if one fires, a field is wrong.
// ---------------------------------------------------------------------------------------------

#![allow(non_camel_case_types, non_snake_case, dead_code)]
// (versions written as the headers write them: major, minor and patch shifted into place)
#![allow(clippy::identity_op)]

use core::ffi::c_void;

use ash::vk;

use crate::fidelityfx::{Coords, Dimensions, Header, Resource};

pub type ffxApiHeader = Header;
pub type ffxCreateContextDescHeader = Header;
pub type ffxConfigureDescHeader = Header;
pub type ffxQueryDescHeader = Header;
pub type ffxDispatchDescHeader = Header;
pub type FfxApiResource = Resource;
pub type FfxApiDimensions2D = Dimensions;
pub type FfxApiFloatCoords2D = Coords;

/// api.h:L45 `typedef void* ffxContext;`
pub type ffxContext = *mut c_void;
/// api.h:L46 `typedef uint32_t ffxReturnCode_t;`
pub type ffxReturnCode_t = u32;
/// api.h:L53
pub type ffxStructType_t = u64;

// =============================================================================================
// Entry points (api.h:L134-L158). Load from amd_fidelityfx_loader_{dx12,vk}.dll.
// =============================================================================================

/// api.h:L117 `typedef void* (*ffxAlloc)(void* pUserData, uint64_t size);`
pub type ffxAlloc = Option<unsafe extern "C" fn(pUserData: *mut c_void, size: u64) -> *mut c_void>;
/// api.h:L120 `typedef void (*ffxDealloc)(void* pUserData, void* pMem);`
pub type ffxDealloc = Option<unsafe extern "C" fn(pUserData: *mut c_void, pMem: *mut c_void)>;

/// api.h:L122
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ffxAllocationCallbacks {
    pub pUserData: *mut c_void,
    pub alloc: ffxAlloc,
    pub dealloc: ffxDealloc,
}

/// api.h:L134-135 `ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb);`
pub type PfnFfxCreateContext = unsafe extern "C" fn(context: *mut ffxContext, desc: *mut ffxCreateContextDescHeader, memCb: *const ffxAllocationCallbacks) -> ffxReturnCode_t;
/// api.h:L140-141 `ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb);`
pub type PfnFfxDestroyContext = unsafe extern "C" fn(context: *mut ffxContext, memCb: *const ffxAllocationCallbacks) -> ffxReturnCode_t;
/// api.h:L146-147 `ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc);`
/// "If context is null, configure operates on any global state."
pub type PfnFfxConfigure = unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxConfigureDescHeader) -> ffxReturnCode_t;
/// api.h:L152-153 `ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc);`
/// (desc is NON-const: queries write outputs.) "If context is null, query operates on any global state."
pub type PfnFfxQuery = unsafe extern "C" fn(context: *mut ffxContext, desc: *mut ffxQueryDescHeader) -> ffxReturnCode_t;
/// api.h:L157-158 `ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc);`
pub type PfnFfxDispatch = unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxDispatchDescHeader) -> ffxReturnCode_t;

// NOTE: every entry point takes `ffxContext*` (pointer TO the context handle), not the handle.

// =============================================================================================
// Return codes (api.h:L33 enum FfxApiReturnCodes)
// =============================================================================================
pub const FFX_API_RETURN_OK: ffxReturnCode_t = 0;
pub const FFX_API_RETURN_ERROR: ffxReturnCode_t = 1;
pub const FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: ffxReturnCode_t = 2;
pub const FFX_API_RETURN_ERROR_RUNTIME_ERROR: ffxReturnCode_t = 3;
pub const FFX_API_RETURN_NO_PROVIDER: ffxReturnCode_t = 4;
pub const FFX_API_RETURN_ERROR_MEMORY: ffxReturnCode_t = 5;
pub const FFX_API_RETURN_ERROR_PARAMETER: ffxReturnCode_t = 6;
pub const FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: ffxReturnCode_t = 7;

// =============================================================================================
// Id construction (api.h:L48-L180)
// =============================================================================================
pub const FFX_API_EFFECT_MASK: u32 = 0x00ff_0000;
pub const FFX_API_BACKEND_MASK: u32 = 0xff00_0000;
pub const FFX_API_EFFECT_ID_GENERAL: u32 = 0x0000_0000;
pub const FFX_API_EFFECT_ID_UPSCALE: u32 = 0x0001_0000;
pub const FFX_API_EFFECT_ID_FRAMEGENERATION: u32 = 0x0002_0000;
pub const FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN: u32 = 0x0003_0000;
/// api.h:L165 "Need to keep this ID around for the deprecated VK frame gen swapchain" -- but the
/// v2.4.0 VK swapchain descs (fg_vk.h) still use it; it is the live VK swapchain id.
pub const FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK: u32 = 0x0004_0000;
pub const FFX_API_BACKEND_ID_DX12: u32 = 0x0000_0000;
pub const FFX_API_BACKEND_ID_VK: u32 = 0x0100_0000;

/// api.h:L170
pub const fn ffx_make_effect_sub_id(effect_id: u32, sub: u32) -> u64 {
    ((effect_id & FFX_API_EFFECT_MASK) | (sub & !FFX_API_EFFECT_MASK)) as u64
}
/// api.h:L180
pub const fn ffx_make_backend_effect_sub_id(backend_id: u32, effect_id: u32, sub: u32) -> u64 {
    ((sub & !FFX_API_EFFECT_MASK) | (effect_id & FFX_API_EFFECT_MASK) | (backend_id & FFX_API_BACKEND_MASK) | (sub & !(FFX_API_BACKEND_MASK | FFX_API_EFFECT_MASK))) as u64
}

// Generic descs that are useful with frame generation (api.h)
pub const FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: u64 = 4; // api.h:L89
pub const FFX_API_DESC_TYPE_OVERRIDE_VERSION: u64 = 5; // api.h:L100
pub const FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: u64 = 6; // api.h:L107
pub const FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG: u64 = 7; // api.h:L80

// =============================================================================================
// Frame generation: versions, desc type ids (fg.h)
// =============================================================================================

/// fg.h:L28-33  FFX_FRAMEGENERATION_VERSION = MAKE_VERSION(4,0,2) = (4<<22)|(0<<12)|2
pub const FFX_FRAMEGENERATION_VERSION: u32 = (4 << 22) | (0 << 12) | 2; // 0x0100_0002

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION: u64 = 0x0002_0001; // fg.h:L68
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION: u64 = 0x0002_0002; // fg.h:L109
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: u64 = 0x0002_0003; // fg.h:L91
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE: u64 = 0x0002_0004; // fg.h:L130 (deprecated)
pub const FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT: u64 = 0x0002_0005; // fg.h:L78
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE: u64 = 0x0002_0006; // fg.h:L153
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE: u64 = 0x0002_0007; // fg.h:L168
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE: u64 = 0x0002_0008; // fg.h:L175
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS: u64 = 0x0002_0009; // fg.h:L183
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO: u64 = 0x0002_000a; // fg.h:L194 (deprecated)
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE_V2: u64 = 0x0002_000b; // fg.h:L208
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2: u64 = 0x0002_000c; // fg.h:L223
pub const FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT_PREMUL_ALPHA: u64 = 0x0002_000d; // fg.h:L251
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION: u64 = 0x0002_000e; // fg.h:L266
pub const FFX_API_FRAME_GENERATION_CONFIG: u64 = 0x0002_000f; // fg.h:L274 (internal swapchain config struct)

// enum FfxApiCreateContextFramegenerationFlags (fg.h:L40)
pub const FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT: u32 = 1 << 0;
pub const FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;
pub const FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION: u32 = 1 << 2;
pub const FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
pub const FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
pub const FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 5;
pub const FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING: u32 = 1 << 6;

// enum FfxApiDispatchFramegenerationFlags (fg.h:L51) -- used by configure.flags AND prepare.flags
pub const FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES: u32 = 1 << 0;
pub const FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS: u32 = 1 << 1;
pub const FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW: u32 = 1 << 2;
pub const FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY: u32 = 1 << 3;
pub const FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES: u32 = 1 << 4;
pub const FFX_FRAMEGENERATION_FLAG_RESERVED_1: u32 = 1 << 5;
pub const FFX_FRAMEGENERATION_FLAG_RESERVED_2: u32 = 1 << 6;

// enum FfxApiUiCompositionFlags (fg.h:L62) -- for the RegisterUiResource descs
pub const FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA: u32 = 1 << 0;
pub const FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING: u32 = 1 << 1;

// enum FfxApiConfigureFrameGenerationKey (fg.h:L162) -- keys for ffxConfigureDescFrameGenerationKeyValue
pub const FFX_API_CONFIGURE_FRAMEGENERATION_KEY_DEBUG_VIEW_MODE: u64 = 0;
pub const FFX_API_CONFIGURE_FRAMEGENERATION_KEY_DEBUG_VIEW_FLOW_SCALE: u64 = 1;

// enum FfxApiBackbufferTransferFunction (types.h:L132) -- ffxDispatchDescFrameGeneration.backbufferTransferFunction
pub const FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB: u32 = 0;
pub const FFX_API_BACKBUFFER_TRANSFER_FUNCTION_PQ: u32 = 1;
pub const FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SCRGB: u32 = 2;

// enum FfxApiResourceState (types.h:L91) -- the states the samples use for FG inputs
pub const FFX_API_RESOURCE_STATE_COMMON: u32 = 1 << 0;
pub const FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
pub const FFX_API_RESOURCE_STATE_COMPUTE_READ: u32 = 1 << 2;
pub const FFX_API_RESOURCE_STATE_PIXEL_READ: u32 = 1 << 3;
pub const FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ: u32 = (1 << 3) | (1 << 2);
pub const FFX_API_RESOURCE_STATE_COPY_SRC: u32 = 1 << 4;
pub const FFX_API_RESOURCE_STATE_COPY_DEST: u32 = 1 << 5;
pub const FFX_API_RESOURCE_STATE_GENERIC_READ: u32 = (1 << 4) | (1 << 2);
pub const FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT: u32 = 1 << 6;
pub const FFX_API_RESOURCE_STATE_PRESENT: u32 = 1 << 7;
pub const FFX_API_RESOURCE_STATE_RENDER_TARGET: u32 = 1 << 8;
pub const FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: u32 = 1 << 9;

// =============================================================================================
// Shared small types not used by upscaling (types.h)
// =============================================================================================

/// types.h:L162 struct FfxApiRect2D
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct FfxApiRect2D {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

/// types.h:L207 struct FfxApiEffectMemoryUsage
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct FfxApiEffectMemoryUsage {
    pub totalUsageInBytes: u64,
    pub aliasableUsageInBytes: u64,
}

// =============================================================================================
// Frame generation structs (fg.h)
// =============================================================================================

/// fg.h:L69 struct ffxCreateContextDescFrameGeneration
/// type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION. Chain a backend desc
/// (ffxCreateBackendDX12Desc / ffxCreateBackendVKDesc) and ffxCreateContextDescFrameGenerationVersion.
#[repr(C)]
pub struct ffxCreateContextDescFrameGeneration {
    pub header: ffxCreateContextDescHeader,
    pub flags: u32,                      // FfxApiCreateContextFramegenerationFlags
    pub displaySize: FfxApiDimensions2D, // resolution of rendered + generated frames
    pub maxRenderSize: FfxApiDimensions2D,
    pub backBufferFormat: u32, // FfxApiSurfaceFormat
}

/// fg.h:L79 struct ffxCallbackDescFrameGenerationPresent
/// Passed to the present (UI composition) callback; header.type = FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT.
#[repr(C)]
pub struct ffxCallbackDescFrameGenerationPresent {
    pub header: ffxDispatchDescHeader,
    pub device: *mut c_void,                   // ID3D12Device* / VkDevice
    pub commandList: *mut c_void,              // ID3D12GraphicsCommandList* / VkCommandBuffer, executed before present
    pub currentBackBuffer: FfxApiResource,     // rendered or generated image (SRV / SHADER_READ_ONLY_OPTIMAL)
    pub currentUI: FfxApiResource,             // registered UI image, else empty
    pub outputSwapChainBuffer: FfxApiResource, // image that will be presented -- write into this
    pub isGeneratedFrame: bool,
    pub frameID: u64,
}

/// fg.h:L92 struct ffxDispatchDescFrameGeneration
/// type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION. Normally built BY the swapchain and handed
/// to frameGenerationCallback; only filled by the app for manual dispatch.
#[repr(C)]
pub struct ffxDispatchDescFrameGeneration {
    pub header: ffxDispatchDescHeader,
    pub commandList: *mut c_void,     // ID3D12GraphicsCommandList* / VkCommandBuffer
    pub presentColor: FfxApiResource, // source color (current backbuffer)
    pub outputs: [FfxApiResource; 4], // one per generated frame
    pub numGeneratedFrames: u32,
    pub reset: bool,
    pub backbufferTransferFunction: u32, // FfxApiBackbufferTransferFunction
    pub minMaxLuminance: [f32; 2],       // nits
    pub generationRect: FfxApiRect2D,
    pub frameID: u64,
}

/// fg.h:L106 `typedef ffxReturnCode_t (*FfxApiPresentCallbackFunc)(ffxCallbackDescFrameGenerationPresent* params, void* pUserCtx);`
/// This is the "UI composition callback" (configure.presentCallback).
pub type FfxApiPresentCallbackFunc = Option<unsafe extern "C" fn(params: *mut ffxCallbackDescFrameGenerationPresent, pUserCtx: *mut c_void) -> ffxReturnCode_t>;
/// fg.h:L107 `typedef ffxReturnCode_t (*FfxApiFrameGenerationDispatchFunc)(ffxDispatchDescFrameGeneration* params, void* pUserCtx);`
pub type FfxApiFrameGenerationDispatchFunc = Option<unsafe extern "C" fn(params: *mut ffxDispatchDescFrameGeneration, pUserCtx: *mut c_void) -> ffxReturnCode_t>;

/// fg.h:L110 struct ffxConfigureDescFrameGeneration
/// type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION. Call ffxConfigure(&fgContext, ...) once per frame.
#[repr(C)]
pub struct ffxConfigureDescFrameGeneration {
    pub header: ffxConfigureDescHeader,
    pub swapChain: *mut c_void, // IDXGISwapChain* (the FFX proxy) / VkSwapchainKHR cast to void*
    pub presentCallback: FfxApiPresentCallbackFunc,
    pub presentCallbackUserContext: *mut c_void,
    pub frameGenerationCallback: FfxApiFrameGenerationDispatchFunc,
    pub frameGenerationCallbackUserContext: *mut c_void,
    pub frameGenerationEnabled: bool,
    pub allowAsyncWorkloads: bool,
    pub HUDLessColor: FfxApiResource, // may be empty
    pub flags: u32,                   // FfxApiDispatchFramegenerationFlags
    pub onlyPresentGenerated: bool,
    pub generationRect: FfxApiRect2D,
    pub frameID: u64,
}

/// fg.h:L131 struct ffxDispatchDescFrameGenerationPrepare  -- DEPRECATED since SDK 2.1, use PrepareV2.
/// type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE
#[repr(C)]
pub struct ffxDispatchDescFrameGenerationPrepare {
    pub header: ffxDispatchDescHeader,
    pub frameID: u64,
    pub flags: u32,
    pub commandList: *mut c_void,
    pub renderSize: FfxApiDimensions2D,
    pub jitterOffset: FfxApiFloatCoords2D,
    pub motionVectorScale: FfxApiFloatCoords2D,
    pub frameTimeDelta: f32,
    pub unused_reset: bool,
    pub cameraNear: f32,
    pub cameraFar: f32,
    pub cameraFovAngleVertical: f32,
    pub viewSpaceToMetersFactor: f32,
    pub depth: FfxApiResource,
    pub motionVectors: FfxApiResource,
}

/// fg.h:L154 struct ffxConfigureDescFrameGenerationKeyValue
/// type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE
#[repr(C)]
pub struct ffxConfigureDescFrameGenerationKeyValue {
    pub header: ffxConfigureDescHeader,
    pub key: u64,  // FfxApiConfigureFrameGenerationKey
    pub u64_: u64, // C name `u64`
    pub ptr: *mut c_void,
}

/// fg.h:L169 struct ffxQueryDescFrameGenerationGetGPUMemoryUsage
/// type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE (needs a live context)
#[repr(C)]
pub struct ffxQueryDescFrameGenerationGetGPUMemoryUsage {
    pub header: ffxQueryDescHeader,
    pub gpuMemoryUsageFrameGeneration: *mut FfxApiEffectMemoryUsage,
}

/// fg.h:L177 struct ffxConfigureDescFrameGenerationRegisterDistortionFieldResource
/// type = ..._REGISTERDISTORTIONRESOURCE. Optional; chain after ffxConfigureDescFrameGeneration.
#[repr(C)]
pub struct ffxConfigureDescFrameGenerationRegisterDistortionFieldResource {
    pub header: ffxConfigureDescHeader,
    pub distortionField: FfxApiResource, // RG: UV after distortion - UV before
}

/// fg.h:L185 struct ffxCreateContextDescFrameGenerationHudless
/// type = ..._HUDLESS. Optional; chain on create when HUDLessColor's format != backBufferFormat.
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationHudless {
    pub header: ffxCreateContextDescHeader,
    pub hudlessBackBufferFormat: u32, // FfxApiSurfaceFormat
}

/// fg.h:L197 struct ffxDispatchDescFrameGenerationPrepareCameraInfo -- DEPRECATED since SDK 2.1
/// (fields now live in PrepareV2). Note: the header field is declared ffxConfigureDescHeader in C
/// (same layout as every header). type = ..._PREPARE_CAMERAINFO; chained after the old Prepare.
#[repr(C)]
pub struct ffxDispatchDescFrameGenerationPrepareCameraInfo {
    pub header: ffxConfigureDescHeader,
    pub cameraPosition: [f32; 3],
    pub cameraUp: [f32; 3],
    pub cameraRight: [f32; 3],
    pub cameraForward: [f32; 3],
}

/// fg.h:L209 struct ffxQueryDescFrameGenerationGetGPUMemoryUsageV2
/// type = ..._GPU_MEMORY_USAGE_V2. Use with a NULL context, before creation.
#[repr(C)]
pub struct ffxQueryDescFrameGenerationGetGPUMemoryUsageV2 {
    pub header: ffxQueryDescHeader,
    pub device: *mut c_void, // DX12: ID3D12Device*; VK: pointer to VkDeviceContext
    pub maxRenderSize: FfxApiDimensions2D,
    pub displaySize: FfxApiDimensions2D,
    pub createFlags: u32,
    pub dispatchFlags: u32,
    pub backBufferFormat: u32,
    pub hudlessBackBufferFormat: u32,
    pub gpuMemoryUsageFrameGeneration: *mut FfxApiEffectMemoryUsage,
}

/// fg.h:L225 struct ffxDispatchDescFrameGenerationPrepareV2
/// type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2. ffxDispatch(&fgContext, ...) once per frame.
#[repr(C)]
pub struct ffxDispatchDescFrameGenerationPrepareV2 {
    pub header: ffxDispatchDescHeader,
    pub frameID: u64,             // same value as in this frame's configure
    pub flags: u32,               // FfxApiDispatchFramegenerationFlags
    pub commandList: *mut c_void, // ID3D12GraphicsCommandList* / VkCommandBuffer
    pub renderSize: FfxApiDimensions2D,
    pub jitterOffset: FfxApiFloatCoords2D,
    pub motionVectorScale: FfxApiFloatCoords2D, // renderSize for UV-space MVs, (1,1) for pixel MVs
    pub frameTimeDelta: f32,                    // milliseconds
    pub reset: bool,                            // honoured (unlike Prepare.unused_reset)
    pub cameraNear: f32,
    pub cameraFar: f32,
    pub cameraFovAngleVertical: f32, // radians
    pub viewSpaceToMetersFactor: f32,
    pub depth: FfxApiResource,
    pub motionVectors: FfxApiResource,
    pub cameraPosition: [f32; 3],
    pub cameraUp: [f32; 3],
    pub cameraRight: [f32; 3],
    pub cameraForward: [f32; 3],
}

/// fg.h:L254 struct ffxCallbackDescFrameGenerationPresentPremulAlpha
/// Attached by the SDK to ffxCallbackDescFrameGenerationPresent.header.pNext.
#[repr(C)]
pub struct ffxCallbackDescFrameGenerationPresentPremulAlpha {
    pub header: ffxApiHeader,
    pub usePremulAlpha: bool,
}

/// fg.h:L264 `typedef int32_t (*FfxWaitCallbackFunc)(wchar_t* fenceName, uint64_t fenceValueToWaitFor);`
/// Set via the swapchain key-value desc, key FFX_API_CONFIGURE_FG_SWAPCHAIN_KEY_WAITCALLBACK (in `ptr`).
pub type FfxWaitCallbackFunc = Option<unsafe extern "C" fn(fenceName: *mut u16, fenceValueToWaitFor: u64) -> i32>;

/// fg.h:L268 struct ffxCreateContextDescFrameGenerationVersion
/// type = ..._FRAMEGENERATION_VERSION; version = FFX_FRAMEGENERATION_VERSION. Required since SDK 2.1
/// ("omitting this will prevent new APIs from functioning").
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationVersion {
    pub header: ffxCreateContextDescHeader,
    pub version: u32,
}

/// fg.h:L278 struct FfxFrameGenerationConfig (type FFX_API_FRAME_GENERATION_CONFIG).
/// Internal config forwarded to the swapchain; applications use ffxConfigureDescFrameGeneration.
#[repr(C)]
pub struct FfxFrameGenerationConfig {
    pub header: ffxApiHeader,
    pub swapChain: *mut c_void,
    pub presentCallback: FfxApiPresentCallbackFunc,
    pub presentCallbackContext: *mut c_void,
    pub frameGenerationCallback: FfxApiFrameGenerationDispatchFunc,
    pub frameGenerationCallbackContext: *mut c_void,
    pub frameGenerationEnabled: bool,
    pub allowAsyncWorkloads: bool,
    pub allowAsyncPresent: bool,
    pub HUDLessColor: FfxApiResource,
    pub flags: u32,
    pub onlyPresentInterpolated: bool,
    pub interpolationRect: FfxApiRect2D,
    pub frameID: u64,
    pub drawDebugPacingLines: bool,
}

/// fg_types.h:L65 struct FfxApiSwapchainFramePacingTuning
/// Passed by pointer in the swapchain key-value desc, key FFX_API_CONFIGURE_FG_SWAPCHAIN_KEY_FRAMEPACINGTUNING.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfxApiSwapchainFramePacingTuning {
    pub safetyMarginInMs: f32,                 // default 0.1
    pub varianceFactor: f32,                   // [0,1], default 0.1
    pub allowHybridSpin: bool,                 // default false
    pub hybridSpinTime: u32,                   // default 2
    pub allowWaitForSingleObjectOnFence: bool, // default false
}

// =============================================================================================
// DX12 frame-generation swapchain (fg_dx12.h). All ids = MAKE_BACKEND_EFFECT_SUB_ID(DX12=0, 0x00030000, n)
// =============================================================================================

/// fg_dx12.h:L26-31 MAKE_VERSION(4,0,0)
pub const FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION: u32 = (4 << 22) | (0 << 12) | 0; // 0x0100_0000

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WRAP_DX12: u64 = 0x0003_0001; // L33
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_REGISTERUIRESOURCE_DX12: u64 = 0x0003_0002; // L63
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_INTERPOLATIONCOMMANDLIST_DX12: u64 = 0x0003_0003; // L71
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_INTERPOLATIONTEXTURE_DX12: u64 = 0x0003_0004; // L78
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_NEW_DX12: u64 = 0x0003_0005; // L41
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12: u64 = 0x0003_0006; // L51
pub const FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WAIT_FOR_PRESENTS_DX12: u64 = 0x0003_0007; // L85
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_KEYVALUE_DX12: u64 = 0x0003_0008; // L91
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_GPU_MEMORY_USAGE_DX12: u64 = 0x0003_0009; // L107
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_GPU_MEMORY_USAGE_DX12_V2: u64 = 0x0003_000a; // L114
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12: u64 = 0x0003_000b; // L128

// enum FfxApiConfigureFrameGenerationSwapChainKeyDX12 (fg_dx12.h:L101); same values in the VK enum
pub const FFX_API_CONFIGURE_FG_SWAPCHAIN_KEY_WAITCALLBACK: u64 = 0; // ptr = FfxWaitCallbackFunc
pub const FFX_API_CONFIGURE_FG_SWAPCHAIN_KEY_FRAMEPACINGTUNING: u64 = 2; // ptr = *FfxApiSwapchainFramePacingTuning

/// fg_dx12.h:L34 struct ffxCreateContextDescFrameGenerationSwapChainWrapDX12
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainWrapDX12 {
    pub header: ffxCreateContextDescHeader,
    pub swapchain: *mut *mut c_void, // IDXGISwapChain4** in: swapchain to wrap; out: FI proxy swapchain
    pub gameQueue: *mut c_void,      // ID3D12CommandQueue*
}

/// fg_dx12.h:L42 struct ffxCreateContextDescFrameGenerationSwapChainNewDX12
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainNewDX12 {
    pub header: ffxCreateContextDescHeader,
    pub swapchain: *mut *mut c_void, // IDXGISwapChain4** out
    pub desc: *mut c_void,           // DXGI_SWAP_CHAIN_DESC*
    pub dxgiFactory: *mut c_void,    // IDXGIFactory*
    pub gameQueue: *mut c_void,      // ID3D12CommandQueue*
}

/// fg_dx12.h:L52 struct ffxCreateContextDescFrameGenerationSwapChainForHwndDX12
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 {
    pub header: ffxCreateContextDescHeader,
    pub swapchain: *mut *mut c_void, // IDXGISwapChain4** out
    pub hwnd: *mut c_void,           // HWND
    pub desc: *mut c_void,           // DXGI_SWAP_CHAIN_DESC1*
    pub fullscreenDesc: *mut c_void, // DXGI_SWAP_CHAIN_FULLSCREEN_DESC* (NULL = windowed)
    pub dxgiFactory: *mut c_void,    // IDXGIFactory*
    pub gameQueue: *mut c_void,      // ID3D12CommandQueue*
}

/// fg_dx12.h:L64 struct ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12
/// ffxConfigure on the SWAPCHAIN context.
#[repr(C)]
pub struct ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12 {
    pub header: ffxConfigureDescHeader,
    pub uiResource: FfxApiResource, // may be empty
    pub flags: u32,                 // FfxApiUiCompositionFlags
}

/// fg_dx12.h:L72 struct ffxQueryDescFrameGenerationSwapChainInterpolationCommandListDX12
#[repr(C)]
pub struct ffxQueryDescFrameGenerationSwapChainInterpolationCommandListDX12 {
    pub header: ffxQueryDescHeader,
    pub pOutCommandList: *mut *mut c_void, // receives ID3D12GraphicsCommandList*
}

/// fg_dx12.h:L79 struct ffxQueryDescFrameGenerationSwapChainInterpolationTextureDX12
#[repr(C)]
pub struct ffxQueryDescFrameGenerationSwapChainInterpolationTextureDX12 {
    pub header: ffxQueryDescHeader,
    pub pOutTexture: *mut FfxApiResource,
}

/// fg_dx12.h:L86 struct ffxDispatchDescFrameGenerationSwapChainWaitForPresentsDX12 (header only)
#[repr(C)]
pub struct ffxDispatchDescFrameGenerationSwapChainWaitForPresentsDX12 {
    pub header: ffxDispatchDescHeader,
}

/// fg_dx12.h:L92 struct ffxConfigureDescFrameGenerationSwapChainKeyValueDX12
#[repr(C)]
pub struct ffxConfigureDescFrameGenerationSwapChainKeyValueDX12 {
    pub header: ffxConfigureDescHeader,
    pub key: u64,
    pub u64_: u64, // C name `u64`
    pub ptr: *mut c_void,
}

/// fg_dx12.h:L108 struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12
#[repr(C)]
pub struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12 {
    pub header: ffxQueryDescHeader,
    pub gpuMemoryUsageFrameGenerationSwapchain: *mut FfxApiEffectMemoryUsage,
}

/// fg_dx12.h:L115 struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12V2 (NULL context, pre-creation)
#[repr(C)]
pub struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12V2 {
    pub header: ffxQueryDescHeader,
    pub device: *mut c_void, // ID3D12Device*
    pub displaySize: FfxApiDimensions2D,
    pub backBufferFormat: u32,
    pub backBufferCount: u32,
    pub uiResourceSize: FfxApiDimensions2D,
    pub uiResourceFormat: u32,
    pub flags: u32,
    pub gpuMemoryUsageFrameGenerationSwapchain: *mut FfxApiEffectMemoryUsage,
}

/// fg_dx12.h:L129 struct ffxCreateContextDescFrameGenerationSwapChainVersionDX12
/// version = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION; chain on whichever DX12 swapchain create desc.
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainVersionDX12 {
    pub header: ffxCreateContextDescHeader,
    pub version: u32,
}

// =============================================================================================
// Vulkan frame-generation swapchain (api_vk.h + fg_vk.h).
// All ids = MAKE_EFFECT_SUB_ID(FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK = 0x00040000, n)
// (NO backend bits, unlike the VK backend desc 0x0100_0003).
// =============================================================================================

/// fg_vk.h:L26-31 MAKE_VERSION(4,0,0)
pub const FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION: u32 = (4 << 22) | (0 << 12) | 0; // 0x0100_0000

pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK: u64 = 0x0004_0001; // L33
pub const FFX_API_CONFIGURE_DESC_TYPE_FGSWAPCHAIN_REGISTERUIRESOURCE_VK: u64 = 0x0004_0002; // L48
pub const FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_INTERPOLATIONCOMMANDLIST_VK: u64 = 0x0004_0003; // L56
pub const FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_INTERPOLATIONTEXTURE_VK: u64 = 0x0004_0004; // L63
pub const FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK: u64 = 0x0004_0005; // L125
pub const FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK: u64 = 0x0004_0007; // L70
pub const FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_KEYVALUE_VK: u64 = 0x0004_0008; // L76
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_GPU_MEMORY_USAGE_VK: u64 = 0x0004_0009; // L92
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_VK: u64 = 0x0004_000a; // L145
pub const FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_GPU_MEMORY_USAGE_VK_V2: u64 = 0x0004_000b; // L101
pub const FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_MODE_VK: u64 = 0x0004_0010; // L138
// (FfxApiConfigureFrameGenerationSwapChainKeyVK, fg_vk.h:L86: WAITCALLBACK = 0, FRAMEPACINGTUNING = 2 -- same consts as DX12 above.)

/// api_vk.h:L29 `typedef VkResult(*PFN_vkQueueSubmitFFXAPI)(uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);`
/// (no VKAPI_PTR -> "C"). NB: no VkQueue parameter -- the callback owns/knows its queue.
pub type PFN_vkQueueSubmitFFXAPI = Option<unsafe extern "C" fn(submitCount: u32, pSubmits: *const vk::SubmitInfo<'static>, fence: vk::Fence) -> vk::Result>;

/// api_vk.h:L32 struct VkQueueInfoFFXAPI
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkQueueInfoFFXAPI {
    pub queue: vk::Queue,
    pub familyIndex: u32,                    // used for queue family ownership transfers
    pub submitFunc: PFN_vkQueueSubmitFFXAPI, // optional, for queues the app also submits to concurrently
}

/// api_vk.h:L51 VkDeviceContext (for reference: what ffxCreateBackendVKDesc wraps, and what the
/// GPU-memory V2 / GetVersions queries want a pointer to in their `device` field on Vulkan).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkDeviceContext {
    pub vkDevice: vk::Device,
    pub vkPhysicalDevice: vk::PhysicalDevice,
    pub vkDeviceProcAddr: vk::PFN_vkGetDeviceProcAddr,
}

/// fg_vk.h:L34 struct ffxCreateContextDescFrameGenerationSwapChainVK
/// VkSwapchainCreateInfoKHR is embedded BY VALUE (104 bytes on 64-bit).
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainVK {
    pub header: ffxCreateContextDescHeader,
    pub physicalDevice: vk::PhysicalDevice,
    pub device: vk::Device,
    pub swapchain: *mut vk::SwapchainKHR,                 // in: swapchain to replace (destroyed by create, may be NULL handle); out: FG swapchain
    pub allocator: *mut vk::AllocationCallbacks<'static>, // optional
    pub createInfo: vk::SwapchainCreateInfoKHR<'static>,  // if createInfo.oldSwapchain != NULL it must equal *swapchain
    pub gameQueue: VkQueueInfoFFXAPI,                     // graphics queue the app presents on
    pub asyncComputeQueue: VkQueueInfoFFXAPI,             // compute-capable
    pub presentQueue: VkQueueInfoFFXAPI,                  // transfer + present capable
    pub imageAcquireQueue: VkQueueInfoFFXAPI,             // any capability
}

/// fg_vk.h:L49 struct ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceVK
#[repr(C)]
pub struct ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceVK {
    pub header: ffxConfigureDescHeader,
    pub uiResource: FfxApiResource,
    pub flags: u32, // FfxApiUiCompositionFlags
}

/// fg_vk.h:L57 struct ffxQueryDescFrameGenerationSwapChainInterpolationCommandListVK
#[repr(C)]
pub struct ffxQueryDescFrameGenerationSwapChainInterpolationCommandListVK {
    pub header: ffxQueryDescHeader,
    pub pOutCommandList: *mut *mut c_void, // receives VkCommandBuffer
}

/// fg_vk.h:L64 struct ffxQueryDescFrameGenerationSwapChainInterpolationTextureVK
#[repr(C)]
pub struct ffxQueryDescFrameGenerationSwapChainInterpolationTextureVK {
    pub header: ffxQueryDescHeader,
    pub pOutTexture: *mut FfxApiResource,
}

/// fg_vk.h:L71 struct ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK (header only)
#[repr(C)]
pub struct ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK {
    pub header: ffxDispatchDescHeader,
}

/// fg_vk.h:L77 struct ffxConfigureDescFrameGenerationSwapChainKeyValueVK
#[repr(C)]
pub struct ffxConfigureDescFrameGenerationSwapChainKeyValueVK {
    pub header: ffxConfigureDescHeader,
    pub key: u64,
    pub u64_: u64, // C name `u64`
    pub ptr: *mut c_void,
}

/// fg_vk.h:L93 struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageVK
#[repr(C)]
pub struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageVK {
    pub header: ffxQueryDescHeader,
    pub gpuMemoryUsageFrameGenerationSwapchain: *mut FfxApiEffectMemoryUsage,
}

/// fg_vk.h:L102 struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageVKV2 (NULL context, pre-creation)
#[repr(C)]
pub struct ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageVKV2 {
    pub header: ffxQueryDescHeader,
    pub device: vk::Device,
    pub physicalDevice: vk::PhysicalDevice,
    pub displaySize: FfxApiDimensions2D,
    pub backBufferFormat: u32,
    pub backBufferCount: u32, // 1..=6
    pub uiResourceSize: FfxApiDimensions2D,
    pub uiResourceFormat: u32,
    pub flags: u32,
    pub composeOnGameQueue: u32,
    pub reserved: u32, // must be 0
    pub gpuMemoryUsageFrameGenerationSwapchain: *mut FfxApiEffectMemoryUsage,
}

/// fg_vk.h:L119 `typedef uint64_t(*PFN_getLastPresentCountFFXAPI)(VkSwapchainKHR);`
pub type PFN_getLastPresentCountFFXAPI = Option<unsafe extern "C" fn(swapchain: vk::SwapchainKHR) -> u64>;
/// fg_vk.h:L122 `typedef VkResult(*PFN_vkCreateSwapchainFFXAPI)(VkDevice, const VkSwapchainCreateInfoKHR*, const VkAllocationCallbacks*, VkSwapchainKHR*, void* pContext);`
pub type PFN_vkCreateSwapchainFFXAPI = Option<
    unsafe extern "C" fn(
        device: vk::Device,
        pCreateInfo: *const vk::SwapchainCreateInfoKHR<'static>,
        pAllocator: *const vk::AllocationCallbacks<'static>,
        pSwapchain: *mut vk::SwapchainKHR,
        pContext: *mut c_void,
    ) -> vk::Result,
>;
/// fg_vk.h:L123 `typedef void(*PFN_vkDestroySwapchainFFXAPI)(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*, void* pContext);`
pub type PFN_vkDestroySwapchainFFXAPI = Option<unsafe extern "C" fn(device: vk::Device, swapchain: vk::SwapchainKHR, pAllocator: *const vk::AllocationCallbacks<'static>, pContext: *mut c_void)>;

/// fg_vk.h:L126 struct ffxQueryDescSwapchainReplacementFunctionsVK
/// type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK; ffxQuery on the SWAPCHAIN context.
/// The four middle members are the standard Vulkan PFNs (VKAPI_PTR -> extern "system").
#[repr(C)]
pub struct ffxQueryDescSwapchainReplacementFunctionsVK {
    pub header: ffxQueryDescHeader,
    pub pOutCreateSwapchainFFXAPI: PFN_vkCreateSwapchainFFXAPI,   // replaces vkCreateSwapchainKHR (in-place recreate)
    pub pOutDestroySwapchainFFXAPI: PFN_vkDestroySwapchainFFXAPI, // replaces vkDestroySwapchainKHR (after ctx destroy)
    pub pOutGetSwapchainImagesKHR: Option<vk::PFN_vkGetSwapchainImagesKHR>,
    pub pOutAcquireNextImageKHR: Option<vk::PFN_vkAcquireNextImageKHR>,
    pub pOutQueuePresentKHR: Option<vk::PFN_vkQueuePresentKHR>,
    pub pOutSetHdrMetadataEXT: Option<vk::PFN_vkSetHdrMetadataEXT>,
    pub pOutGetLastPresentCountFFXAPI: PFN_getLastPresentCountFFXAPI,
}

/// fg_vk.h:L139 struct ffxCreateContextDescFrameGenerationSwapChainModeVK (optional, chain on create)
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainModeVK {
    pub header: ffxCreateContextDescHeader,
    pub composeOnPresentQueue: bool, // true: UI composition on present queue; false: on game queue
}

/// fg_vk.h:L146 struct ffxCreateContextDescFrameGenerationSwapChainVersionVK
/// version = FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION; chain on ffxCreateContextDescFrameGenerationSwapChainVK.
#[repr(C)]
pub struct ffxCreateContextDescFrameGenerationSwapChainVersionVK {
    pub header: ffxCreateContextDescHeader,
    pub version: u32,
}

// =============================================================================================
// Layout checks (64-bit, hand-computed from the C rules). FfxApiResource = 48, header = 16.
// =============================================================================================
#[cfg(target_pointer_width = "64")]
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<FfxApiResource>() == 48);
    assert!(size_of::<FfxApiRect2D>() == 16);
    assert!(size_of::<ffxCreateContextDescFrameGeneration>() == 40);
    assert!(size_of::<ffxCallbackDescFrameGenerationPresent>() == 192);
    assert!(offset_of!(ffxCallbackDescFrameGenerationPresent, isGeneratedFrame) == 176);
    assert!(size_of::<ffxDispatchDescFrameGeneration>() == 312);
    assert!(offset_of!(ffxDispatchDescFrameGeneration, numGeneratedFrames) == 264);
    assert!(offset_of!(ffxDispatchDescFrameGeneration, generationRect) == 284);
    assert!(offset_of!(ffxDispatchDescFrameGeneration, frameID) == 304);
    assert!(size_of::<ffxConfigureDescFrameGeneration>() == 144);
    assert!(offset_of!(ffxConfigureDescFrameGeneration, frameGenerationEnabled) == 56);
    assert!(offset_of!(ffxConfigureDescFrameGeneration, HUDLessColor) == 64);
    assert!(offset_of!(ffxConfigureDescFrameGeneration, onlyPresentGenerated) == 116);
    assert!(offset_of!(ffxConfigureDescFrameGeneration, generationRect) == 120);
    assert!(offset_of!(ffxConfigureDescFrameGeneration, frameID) == 136);
    assert!(size_of::<ffxDispatchDescFrameGenerationPrepare>() == 184);
    assert!(size_of::<ffxDispatchDescFrameGenerationPrepareV2>() == 232);
    assert!(offset_of!(ffxDispatchDescFrameGenerationPrepareV2, commandList) == 32);
    assert!(offset_of!(ffxDispatchDescFrameGenerationPrepareV2, reset) == 68);
    assert!(offset_of!(ffxDispatchDescFrameGenerationPrepareV2, depth) == 88);
    assert!(offset_of!(ffxDispatchDescFrameGenerationPrepareV2, cameraPosition) == 184);
    assert!(size_of::<ffxDispatchDescFrameGenerationPrepareCameraInfo>() == 64);
    assert!(size_of::<ffxConfigureDescFrameGenerationKeyValue>() == 40);
    assert!(size_of::<ffxQueryDescFrameGenerationGetGPUMemoryUsageV2>() == 64);
    assert!(size_of::<ffxCallbackDescFrameGenerationPresentPremulAlpha>() == 24);
    assert!(size_of::<FfxFrameGenerationConfig>() == 152);
    assert!(size_of::<FfxApiSwapchainFramePacingTuning>() == 20);
    assert!(size_of::<ffxCreateContextDescFrameGenerationSwapChainWrapDX12>() == 32);
    assert!(size_of::<ffxCreateContextDescFrameGenerationSwapChainNewDX12>() == 48);
    assert!(size_of::<ffxCreateContextDescFrameGenerationSwapChainForHwndDX12>() == 64);
    assert!(size_of::<ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12>() == 72);
    assert!(size_of::<ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12V2>() == 64);
    assert!(size_of::<VkQueueInfoFFXAPI>() == 24);
    assert!(size_of::<vk::SwapchainCreateInfoKHR<'static>>() == 104);
    assert!(size_of::<ffxCreateContextDescFrameGenerationSwapChainVK>() == 248);
    assert!(offset_of!(ffxCreateContextDescFrameGenerationSwapChainVK, createInfo) == 48);
    assert!(offset_of!(ffxCreateContextDescFrameGenerationSwapChainVK, gameQueue) == 152);
    assert!(size_of::<ffxQueryDescSwapchainReplacementFunctionsVK>() == 72);
    assert!(size_of::<ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageVKV2>() == 80);
    assert!(size_of::<ffxCreateContextDescFrameGenerationSwapChainModeVK>() == 24);
    assert!(ffx_make_effect_sub_id(FFX_API_EFFECT_ID_FRAMEGENERATION, 0x0c) == FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2);
    assert!(ffx_make_backend_effect_sub_id(FFX_API_BACKEND_ID_DX12, FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN, 0x06) == FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12);
    assert!(ffx_make_effect_sub_id(FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK, 0x10) == FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_MODE_VK);
};

// =============================================================================================
// Version differences vs. v1.1.4 (last 1.x tag, the original "FSR 3.1" ffx-api headers)
// =============================================================================================
// * Every struct above that exists in v1.1.4 has the identical field list and order, and every
//   numeric type id is identical (v1.1.4 wrote them as literals, e.g. 0x30006u, 0x40001u).
// * New since 1.1.4: PrepareV2 (0x2000c), GPU_MEMORY_USAGE_V2 (0x2000b), PRESENT_PREMUL_ALPHA
//   (0x2000d), FRAMEGENERATION_VERSION (0x2000e), FRAME_GENERATION_CONFIG (0x2000f), DX12
//   GPU_MEMORY_USAGE_DX12_V2 (0x3000a), SWAPCHAIN VERSION_DX12 (0x3000b), VK VERSION_VK
//   (0x4000a), VK GPU_MEMORY_USAGE_VK_V2 (0x4000b), return code 7.
// * In v1.1.4 the swapchain descs lived in ffx_api_dx12.h / ffx_api_vk.h; in 2.x they moved to
//   framegeneration/include/{dx12,vk}/ffx_api_framegeneration_*.h. The VK effect id macro was
//   renamed FFX_API_EFFECT_ID_FGSC_VK -> FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK (same 0x40000).
// * VK backend desc: 1.1.4 used type 0x3 with {vkDevice, vkPhysicalDevice, vkDeviceProcAddr} inline;
//   2.x uses 0x0100_0003 with a VkDeviceContext holding the same three fields in the same order
//   (0x3 survives as ffxCreateBackendVKDesc_DEPRECATED). temporal/fidelityfx.rs already uses 0x0100_0003.
// * Since SDK 2.1 the Prepare / PrepareCameraInfo pair is deprecated in favour of PrepareV2, and the
//   *Version create descs are required ("omitting this will prevent new APIs from functioning").
// * There is no separate "UI composition callback" typedef in any version: the UI composition
//   callback IS FfxApiPresentCallbackFunc (ffxConfigureDescFrameGeneration.presentCallback).

// =============================================================================================
// CALL SEQUENCE -- DirectX 12
// (from docs/techniques/frame-interpolation-api.md, frame-interpolation-swap-chain.md and
//  Samples/Upscalers/FidelityFX_FSR/internal/fsrapirendermodule.cpp in v2.4.0)
// =============================================================================================
//
// Contexts: two independent ffxContext handles -- the SWAPCHAIN context and the FRAMEGEN context.
// Both are created with ffxCreateContext from amd_fidelityfx_loader_dx12.dll. All descs are linked
// through header.next; order in the chain does not matter.
//
// 1. Create the proxy swapchain (swapchain context).
//    a. DX12 allows one swapchain per HWND. If a normal swapchain already exists, read its
//       HWND / DXGI_SWAP_CHAIN_DESC1 / DXGI_SWAP_CHAIN_FULLSCREEN_DESC / parent factory, then
//       Release() it completely BEFORE creating the proxy (the sample does exactly this).
//       (Alternative: ffxCreateContextDescFrameGenerationSwapChainWrapDX12 wraps an existing
//       IDXGISwapChain4 in place; ...NewDX12 creates from a DXGI_SWAP_CHAIN_DESC.)
//    b. let mut sc: *mut c_void = null_mut();  // will receive IDXGISwapChain4*
//       ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 {
//           header: { FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12, next: &ver },
//           swapchain: &mut sc, hwnd, desc: &mut desc1, fullscreenDesc: &mut fsDesc (or null),
//           dxgiFactory, gameQueue: the ID3D12CommandQueue you present from }
//       ffxCreateContextDescFrameGenerationSwapChainVersionDX12 ver {
//           FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12,
//           version: FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION }
//       ffxCreateContext(&mut swapchain_ctx, &mut forHwnd.header, null)
//    c. `sc` is now an IDXGISwapChain4* implemented by FFX ("implements the IDXGISwapChain4
//       interface, so once created it behaves just like a normal swapchain"). In windows-rs:
//       IDXGISwapChain4::from_raw(sc) takes ownership of that reference. Use it for
//       GetBuffer / GetCurrentBackBufferIndex / ResizeBuffers / Present exactly as before.
//       The sample also calls factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_WINDOW_CHANGES)
//       on the proxy's parent factory, and re-applies HDR metadata / colour space.
//    d. Optional: desc1.Flags |= DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT before create,
//       then GetFrameLatencyWaitableObject(); each call returns a NEW handle the app must
//       CloseHandle() after destroying the swapchain context. 2 backbuffers are sufficient.
//    e. Optional swapchain tuning: ffxConfigure(&swapchain_ctx, KeyValueDX12 { key: 2 /*FRAMEPACINGTUNING*/,
//       ptr: &FfxApiSwapchainFramePacingTuning }).
//
// 2. Create the framegen context.
//       ffxCreateContextDescFrameGeneration { header: {FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, next: &backend},
//           flags (e.g. DEPTH_INVERTED | ASYNC_WORKLOAD_SUPPORT | HIGH_DYNAMIC_RANGE),
//           displaySize, maxRenderSize, backBufferFormat (FfxApiSurfaceFormat of the swapchain) }
//       -> ffxCreateBackendDX12Desc { FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 (0x2), device }
//       -> ffxCreateContextDescFrameGenerationVersion { 0x2000e, FFX_FRAMEGENERATION_VERSION }
//       -> optional ffxCreateContextDescFrameGenerationHudless { 0x20009, hudless format }
//       ffxCreateContext(&mut fg_ctx, &mut createFg.header, null)
//    (Optional pre-check: ffxQuery(null, ffxQueryDescFrameGenerationGetGPUMemoryUsageV2) and
//     ffxQuery(null, ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12V2).)
//
// 3. Per frame, after rendering + upscaling, on the game thread, with frameID incremented by
//    exactly 1 per frame (any other step resets FG; the same frameID goes to configure and prepare):
//    a. ffxConfigure(&fg_ctx, ffxConfigureDescFrameGeneration {
//           swapChain: the proxy IDXGISwapChain4* (as void*),
//           frameGenerationEnabled: true/false,
//           frameGenerationCallback: Some(cb) where cb does `ffxDispatch(user_ctx as *mut ffxContext, &params.header)`,
//           frameGenerationCallbackUserContext: &mut fg_ctx as *mut _ as *mut c_void,  // pointer TO the handle
//           presentCallback: None (swapchain blits backbuffer + registered UI) or Some(ui_cb),
//           presentCallbackUserContext, allowAsyncWorkloads, HUDLessColor (or zeroed),
//           flags (debug flags; NO_SWAPCHAIN_CONTEXT_NOTIFY only when NOT using the proxy swapchain),
//           onlyPresentGenerated: false, generationRect: {0,0,displayW,displayH}, frameID })
//       Optionally chain ffxConfigureDescFrameGenerationRegisterDistortionFieldResource.
//    b. If FG enabled: ffxDispatch(&fg_ctx, ffxDispatchDescFrameGenerationPrepareV2 {
//           frameID, flags (same as configure), commandList: the frame's ID3D12GraphicsCommandList
//           (recorded, executed by you before Present), renderSize, jitterOffset (sample passes
//           -jitter, same convention as the upscaler dispatch), motionVectorScale (renderSize for
//           UV-space MVs), frameTimeDelta (ms), reset, cameraNear/Far, cameraFovAngleVertical (rad),
//           viewSpaceToMetersFactor, depth + motionVectors (render-res, state PIXEL_COMPUTE_READ /
//           COMPUTE_READ), cameraPosition/Up/Right/Forward (world space) })
//    c. If using a separate UI texture: ffxConfigure(&swapchain_ctx,
//           ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12 { uiResource, flags })
//       every frame (empty resource to unregister).
//    d. Execute your command lists, then call Present(sync, flags) on the proxy IDXGISwapChain4.
//
// 4. What happens in Present: the proxy takes the current backbuffer as presentColor, calls your
//    frameGenerationCallback on the game thread (it fills ffxDispatchDescFrameGeneration with its
//    own command list, presentColor and outputs[0] = its interpolation target), executes that on
//    the game queue or async compute, then its own present thread composes UI (registered UI
//    texture, or your presentCallback once per presented image, real and generated) into the real
//    swapchain image and paces two presents (generated, then real). Present also waits for the
//    previous frame's interpolation + UI composition, so the backbuffer is reusable on return.
//    If FG is disabled the proxy still copies + composes UI (one extra copy).
//    Manual alternative (discouraged): frameGenerationCallback = None, then each frame
//    ffxQuery(&swapchain_ctx, ...InterpolationCommandListDX12 { pOutCommandList: &mut dispatch.commandList })
//    and ffxQuery(&swapchain_ctx, ...InterpolationTextureDX12 { pOutTexture: &mut dispatch.outputs[0] }),
//    fill ffxDispatchDescFrameGeneration (presentColor = backbuffer in state PRESENT,
//    numGeneratedFrames 1, frameID, reset, backbufferTransferFunction, minMaxLuminance,
//    generationRect) and ffxDispatch(&fg_ctx, ...) before Present.
//
// 5. Threading: Present (when it runs the FG callback), PrepareV2 dispatch, manual FG dispatch,
//    create/destroy of the FG context and the proxy's destructor must be externally serialized
//    (one mutex).
//
// 6. Teardown (order matters):
//    a. ffxConfigure(&fg_ctx, ffxConfigureDescFrameGeneration { frameGenerationEnabled: false,
//       swapChain: proxy, presentCallback: None, HUDLessColor: zeroed, ... }) -- internally waits
//       for presents (flushes interpolation + UI composition GPU work). Otherwise D3D12 error #921
//       OBJECT_DELETED_WHILE_STILL_IN_USE. (ffxDispatch(&swapchain_ctx,
//       ffxDispatchDescFrameGenerationSwapChainWaitForPresentsDX12) is the explicit flush.)
//    b. ffxDestroyContext(&mut fg_ctx, null).
//    c. ffxDestroyContext(&mut swapchain_ctx, null) -- does NOT destroy the proxy swapchain object.
//    d. Release() the last reference to the proxy IDXGISwapChain4 (refcount must reach 0; this
//       runs the proxy destructor). Then CloseHandle any latency waitable handles.
//    e. Only now may a plain DXGI swapchain be created for that HWND again.
//
// =============================================================================================
// CALL SEQUENCE -- Vulkan
// (same sources + Cauldron2/framework/render/vk/device_vk.cpp)
// =============================================================================================
//
// 1. Device creation: request extra queues for FG -- a present queue (transfer+present), an
//    image-acquire queue (any), an async compute queue -- in addition to the graphics queue.
//    The sample gives them their own queue indices; if the app submits to any VkQueue it also
//    hands to FFX from another thread, provide VkQueueInfoFFXAPI.submitFunc so FFX submits
//    through the app's lock (PFN_vkQueueSubmitFFXAPI has no VkQueue param: one callback per queue).
//
// 2. Create a normal swapchain with the driver's vkCreateSwapchainKHR, then replace it:
//       let mut sc: vk::SwapchainKHR = existing;
//       ffxCreateContextDescFrameGenerationSwapChainVK {
//           header: { FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK (0x40001), next: &ver },
//           physicalDevice, device, swapchain: &mut sc, allocator: null,
//           createInfo: the same VkSwapchainCreateInfoKHR (by value),
//           gameQueue / asyncComputeQueue / presentQueue / imageAcquireQueue }
//       -> ffxCreateContextDescFrameGenerationSwapChainVersionVK { 0x4000a, FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION }
//       -> optional ffxCreateContextDescFrameGenerationSwapChainModeVK { 0x40010, composeOnPresentQueue }
//       ffxCreateContext(&mut swapchain_ctx, &mut create.header, null)
//    The old swapchain is DESTROYED by this call; `sc` now holds the FG swapchain handle.
//    Drop every other copy of the old handle first (the sample nulls its own before the call).
//
// 3. Get the replacement functions -- ffxQuery on the swapchain context:
//       ffxQueryDescSwapchainReplacementFunctionsVK { header: { FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK (0x40005) } }
//       ffxQuery(&mut swapchain_ctx, &mut q.header)
//    From now on, for the FG swapchain handle, call ONLY these instead of the driver's:
//       vkGetSwapchainImagesKHR -> pOutGetSwapchainImagesKHR(device, sc, &count, images)
//       vkAcquireNextImageKHR   -> pOutAcquireNextImageKHR(device, sc, timeout, sem, fence, &idx)
//       vkQueuePresentKHR       -> pOutQueuePresentKHR(gameQueue, &presentInfo)   (sample skips the call if swapchainCount == 0)
//       vkSetHdrMetadataEXT     -> pOutSetHdrMetadataEXT(device, 1, &sc, &meta)
//       vkCreateSwapchainKHR    -> pOutCreateSwapchainFFXAPI(device, &createInfo, null, &mut sc, sc as *mut c_void)
//                                  for in-place recreate (resize / vsync toggle); pContext and the
//                                  in/out handle are the existing FG swapchain
//       vkDestroySwapchainKHR   -> pOutDestroySwapchainFFXAPI(device, sc, null, null)  (only AFTER
//                                  destroying the swapchain context; pContext is unused/legacy)
//       extra: pOutGetLastPresentCountFFXAPI(sc) = presents since creation (use it for a CPU
//              frame limiter; there is no waitable object on Vulkan)
//    The swapchain images returned are FFX-owned app-facing backbuffers; render into them as usual.
//    These are plain function pointers returned by the query -- they are NOT exposed via
//    vkGetDeviceProcAddr and must not be mixed with the driver entry points for this handle.
//
// 4. Framegen context: identical to DX12 step 2, but the backend desc is
//       ffxCreateBackendVKDesc { header: {0x0100_0003}, deviceContext: VkDeviceContext { vkDevice,
//       vkPhysicalDevice, vkDeviceProcAddr: vkGetDeviceProcAddr } }.
//
// 5. Per frame: same as DX12 step 3, except configure.swapChain = sc.as_raw() as *mut c_void
//    (the FG VkSwapchainKHR), commandList fields are VkCommandBuffer, the UI desc is
//    ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceVK, and the frame ends with
//    pOutQueuePresentKHR(...) -- that call is where the frameGenerationCallback runs and
//    interpolation + paced presentation happen. The present callback receives the backbuffer in
//    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL. Manual-dispatch queries: ...InterpolationCommandListVK
//    (VkCommandBuffer) and ...InterpolationTextureVK.
//
// 6. Teardown:
//    a. ffxConfigure(&fg_ctx, configure { frameGenerationEnabled: false, swapChain: sc, presentCallback: None, HUDLessColor: zeroed })
//    b. ffxConfigure(&swapchain_ctx, RegisterUiResourceVK { uiResource: zeroed, flags: 0 })  (drop UI image)
//    c. ffxDestroyContext(&mut fg_ctx, null)
//    d. If presents may still be in flight: ffxDispatch(&swapchain_ctx,
//       ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK { 0x40007 })
//    e. ffxDestroyContext(&mut swapchain_ctx, null)
//    f. pOutDestroySwapchainFFXAPI(device, sc, null, null); forget the replacement table.
//    g. To continue without FG, create a fresh swapchain with the driver's vkCreateSwapchainKHR
//       from the saved VkSwapchainCreateInfoKHR (oldSwapchain = NULL).
// =============================================================================================
