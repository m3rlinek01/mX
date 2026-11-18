#![allow(dead_code)]
// The Streamline structures, constants and identifiers below are transcribed from NVIDIA's
// Streamline SDK headers (https://github.com/NVIDIA-RTX/Streamline), tag v2.14.16
// (commit 48bc490d53ca9b5a6cf421d9ce18e15e85fcd846, released 2026-10-16), which come under this
// licence:
//
// Copyright (c) 2022-2024 NVIDIA CORPORATION. All rights reserved
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Additions to `temporal/streamline.rs` for DLSS Ray Reconstruction (DLSS-RR / "DLSSD"),
// DLSS Frame Generation (DLSS-G), Reflex and PC Latency (PCL), and the swapchain interposition
// DLSS-G needs.
//
// This is a fragment meant to sit inside `streamline.rs`: it uses that file's `Base`,
// `StructType`, `ViewportHandle`, `Constants`, `Version`, `FeatureRequirements`, `ResourceTag`
// and its conventions (C++ `const T&` -> `*const T`, `T&` -> `*mut T`, `sl::Boolean` (a `char`)
// -> `u8`, an opaque `sl::FrameToken*` -> `*const c_void`, `sl::Result` -> `i32`).
//
// Citations are `header:line` in tag v2.14.16.

use crate::streamline::{Base, StructType, ViewportHandle};
use ash::vk;
use std::ffi::c_void;
use windows::core::{GUID, HRESULT};

// ---------------------------------------------------------------------------------------------
// Version (sl_version.h:24-34)
// ---------------------------------------------------------------------------------------------

// kSDKVersion = major << 48 | minor << 32 | patch << 16 | kSDKVersionMagic (0xfedc).
// streamline.rs has 2.14.1; v2.14.16 is the newest public tag.
pub(crate) const SDK_VERSION_2_14_16: u64 = (2 << 48) | (14 << 32) | (16 << 16) | 0xfedc;

// sl_struct.h:100-106 (struct versions; `Base::struct_version` is a size_t)
pub(crate) const STRUCT_VERSION_1: usize = 1;
pub(crate) const STRUCT_VERSION_2: usize = 2;
pub(crate) const STRUCT_VERSION_3: usize = 3;
pub(crate) const STRUCT_VERSION_4: usize = 4;
pub(crate) const STRUCT_VERSION_5: usize = 5;

// ---------------------------------------------------------------------------------------------
// Feature ids (sl_core_types.h:221-264, `using Feature = uint32_t`)
// ---------------------------------------------------------------------------------------------

pub(crate) const FEATURE_DLSS: u32 = 0; // :224
/// `kFeatureNRD_INVALID`: NRD was removed from Streamline; the id stays reserved. (:227)
pub(crate) const FEATURE_NRD_INVALID: u32 = 1;
pub(crate) const FEATURE_NIS: u32 = 2; // :230
pub(crate) const FEATURE_REFLEX: u32 = 3; // :233
pub(crate) const FEATURE_PCL: u32 = 4; // :236
pub(crate) const FEATURE_DEEP_DVC: u32 = 5; // :239
pub(crate) const FEATURE_LATEWARP: u32 = 6; // :241
pub(crate) const FEATURE_DLSS_G: u32 = 1000; // :244
pub(crate) const FEATURE_DLSS_RR: u32 = 1001; // :247
pub(crate) const FEATURE_NV_PERF: u32 = 1002; // :249
pub(crate) const FEATURE_DIRECT_SR: u32 = 1003; // :251
pub(crate) const FEATURE_DLSS_NR: u32 = 1004; // :253
pub(crate) const FEATURE_IMGUI: u32 = 9999; // :256
pub(crate) const FEATURE_COMMON: u32 = u32::MAX; // :264

// ---------------------------------------------------------------------------------------------
// Buffer types (sl_core_types.h:66-215, `using BufferType = uint32_t`)
// Already in streamline.rs: 0 depth, 1 mvec, 3 scaling input, 4 scaling output, 29 bias hint.
// ---------------------------------------------------------------------------------------------

pub(crate) const BUFFER_HUD_LESS_COLOR: u32 = 2; // :70  DLSS-G
pub(crate) const BUFFER_NORMALS: u32 = 5; // :76  RR
pub(crate) const BUFFER_ROUGHNESS: u32 = 6; // :78  RR
pub(crate) const BUFFER_ALBEDO: u32 = 7; // :80  RR diffuse albedo
pub(crate) const BUFFER_SPECULAR_ALBEDO: u32 = 8; // :82  RR
pub(crate) const BUFFER_INDIRECT_ALBEDO: u32 = 9; // :84
pub(crate) const BUFFER_SPECULAR_MOTION_VECTORS: u32 = 10; // :86  RR
pub(crate) const BUFFER_DISOCCLUSION_MASK: u32 = 11; // :88
pub(crate) const BUFFER_EMISSIVE: u32 = 12; // :90
pub(crate) const BUFFER_EXPOSURE: u32 = 13; // :92
pub(crate) const BUFFER_NORMAL_ROUGHNESS: u32 = 14; // :94  RR
pub(crate) const BUFFER_UI_COLOR_AND_ALPHA: u32 = 23; // :113 DLSS-G (premultiplied RGB + alpha)
pub(crate) const BUFFER_RAYTRACING_DISTANCE: u32 = 30; // :127
pub(crate) const BUFFER_REFLECTION_MOTION_VECTORS: u32 = 31; // :129
pub(crate) const BUFFER_POSITION: u32 = 32; // :131
pub(crate) const BUFFER_COLOR_BEFORE_PARTICLES: u32 = 39; // :145
pub(crate) const BUFFER_COLOR_BEFORE_TRANSPARENCY: u32 = 40; // :147 RR optional guide
pub(crate) const BUFFER_SPECULAR_HIT_DISTANCE: u32 = 42; // :151 RR (instead of specular mvec)
pub(crate) const BUFFER_SPECULAR_RAY_DIRECTION_HIT_DISTANCE: u32 = 43; // :153
pub(crate) const BUFFER_SPECULAR_RAY_DIRECTION: u32 = 44; // :155
pub(crate) const BUFFER_DIFFUSE_HIT_DISTANCE: u32 = 45; // :157
pub(crate) const BUFFER_DIFFUSE_RAY_DIRECTION_HIT_DISTANCE: u32 = 46; // :159
pub(crate) const BUFFER_DIFFUSE_RAY_DIRECTION: u32 = 47; // :161
pub(crate) const BUFFER_HI_RES_DEPTH: u32 = 48; // :163
pub(crate) const BUFFER_LINEAR_DEPTH: u32 = 49; // :165 RR (alternative to depth)
pub(crate) const BUFFER_BIDIRECTIONAL_DISTORTION_FIELD: u32 = 50; // :167 DLSS-G optional
pub(crate) const BUFFER_TRANSPARENCY_LAYER: u32 = 51; // :169 RR optional
pub(crate) const BUFFER_TRANSPARENCY_LAYER_OPACITY: u32 = 52; // :172 RR optional
pub(crate) const BUFFER_BACKBUFFER: u32 = 53; // :174 DLSS-G, only to pass a subrect (resource may be null)
pub(crate) const BUFFER_SSS_GUIDE: u32 = 58; // :184 kBufferTypeScreenSpaceSubsurfaceScatteringGuide, RR
pub(crate) const BUFFER_DEPTH_OF_FIELD_GUIDE: u32 = 64; // :196 RR optional
pub(crate) const BUFFER_SCALING_OUTPUT_ALPHA: u32 = 67; // :202
pub(crate) const BUFFER_UI_ALPHA: u32 = 69; // :208 DLSS-G, preferred over 23 when both are tagged

// sl::ResourceLifecycle (sl_core_types.h:397-405, plain C enum)
pub(crate) const ONLY_VALID_NOW: u32 = 0;
pub(crate) const VALID_UNTIL_PRESENT: u32 = 1;
pub(crate) const VALID_UNTIL_EVALUATE: u32 = 2;

// sl::Boolean (sl_consts.h:172-177, `enum Boolean : char`)
pub(crate) const BOOL_FALSE: u8 = 0;
pub(crate) const BOOL_TRUE: u8 = 1;
pub(crate) const BOOL_INVALID: u8 = 2;

pub(crate) const INVALID_UINT: u32 = 0xffff_ffff; // sl_consts.h:91

// ---------------------------------------------------------------------------------------------
// sl::PreferenceFlags (sl_core_types.h:520-553, uint64_t)
// ---------------------------------------------------------------------------------------------

pub(crate) const FLAG_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
pub(crate) const FLAG_DISABLE_DEBUG_TEXT: u64 = 1 << 1;
/// Manual hooking: SL does not hook anything globally; we hand it proxies via
/// `slUpgradeInterface` and native interfaces via `slGetNativeInterface`.
pub(crate) const FLAG_USE_MANUAL_HOOKING: u64 = 1 << 2;
pub(crate) const FLAG_ALLOW_OTA: u64 = 1 << 3;
pub(crate) const FLAG_BYPASS_OS_VERSION_CHECK: u64 = 1 << 4;
/// SL creates a DXGI factory proxy instead of patching the factory's v-table (helps with
/// injected overlays).
pub(crate) const FLAG_USE_DXGI_FACTORY_PROXY: u64 = 1 << 5;
pub(crate) const FLAG_LOAD_DOWNLOADED_PLUGINS: u64 = 1 << 6;
/// Required for `slSetTagForFrame` (and for evaluate using frame-based tags).
pub(crate) const FLAG_USE_FRAME_BASED_RESOURCE_TAGGING: u64 = 1 << 7;

// ---------------------------------------------------------------------------------------------
// sl::FeatureRequirementFlags (sl_core_types.h:625-638, uint32_t), for
// `slGetFeatureRequirements(kFeatureDLSS_G, ..)`: compare `os_version_detected` against
// `os_version_required` and `driver_version_detected` against `driver_version_required`.
// DLSS-G needs Win10 20H1 (build 19041+) and hardware-accelerated GPU scheduling
// (docs/ProgrammingGuideDLSS_G.md section 1.0).
// ---------------------------------------------------------------------------------------------

pub(crate) const REQ_D3D11_SUPPORTED: u32 = 1 << 0;
pub(crate) const REQ_D3D12_SUPPORTED: u32 = 1 << 1;
pub(crate) const REQ_VULKAN_SUPPORTED: u32 = 1 << 2;
pub(crate) const REQ_VSYNC_OFF_REQUIRED: u32 = 1 << 3;
pub(crate) const REQ_HARDWARE_SCHEDULING_REQUIRED: u32 = 1 << 4;

// FeatureRequirements is kStructVersion2 (sl_core_types.h:645); its layout is already in
// streamline.rs and is unchanged in v2.14.16.

// ---------------------------------------------------------------------------------------------
// GUIDs (StructType) of the new structures. Bytes exactly as in the headers.
// ---------------------------------------------------------------------------------------------

/// {0AD87504-774E-4BF3-9633-A44D1F7F9CB8}, kStructVersion3 (sl_dlss_d.h:62)
pub(crate) const DLSSD_OPTIONS: StructType = StructType {
    data1: 0x0ad87504,
    data2: 0x774e,
    data3: 0x4bf3,
    data4: [0x96, 0x33, 0xa4, 0x4d, 0x1f, 0x7f, 0x9c, 0xb8],
};
pub(crate) const DLSSD_OPTIONS_VERSION: usize = STRUCT_VERSION_3;

/// {FBD0C637-A28F-41F2-BC91-B421FAEE8E1E}, kStructVersion1 (sl_dlss_d.h:106)
pub(crate) const DLSSD_OPTIMAL_SETTINGS: StructType = StructType {
    data1: 0xfbd0c637,
    data2: 0xa28f,
    data3: 0x41f2,
    data4: [0xbc, 0x91, 0xb4, 0x21, 0xfa, 0xee, 0x8e, 0x1e],
};
pub(crate) const DLSSD_OPTIMAL_SETTINGS_VERSION: usize = STRUCT_VERSION_1;

/// {71873C14-F8CA-4767-9EAF-3B4393EA98FA}, kStructVersion1 (sl_dlss_d.h:128)
pub(crate) const DLSSD_STATE: StructType = StructType {
    data1: 0x71873c14,
    data2: 0xf8ca,
    data3: 0x4767,
    data4: [0x9e, 0xaf, 0x3b, 0x43, 0x93, 0xea, 0x98, 0xfa],
};
pub(crate) const DLSSD_STATE_VERSION: usize = STRUCT_VERSION_1;

/// {FAC5F1CB-2DFD-4F36-A1E6-3A9E865256C5}, kStructVersion5 (sl_dlss_g.h:72)
pub(crate) const DLSSG_OPTIONS: StructType = StructType {
    data1: 0xfac5f1cb,
    data2: 0x2dfd,
    data3: 0x4f36,
    data4: [0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5],
};
pub(crate) const DLSSG_OPTIONS_VERSION: usize = STRUCT_VERSION_5;

/// {CC8AC8E1-A179-44F5-97FA-E74112F9BC61}, kStructVersion4 (sl_dlss_g.h:149)
pub(crate) const DLSSG_STATE: StructType = StructType {
    data1: 0xcc8ac8e1,
    data2: 0xa179,
    data3: 0x44f5,
    data4: [0x97, 0xfa, 0xe7, 0x41, 0x12, 0xf9, 0xbc, 0x61],
};
pub(crate) const DLSSG_STATE_VERSION: usize = STRUCT_VERSION_4;

/// {F03AF81A-6D0B-4902-A651-C4965E215434}, kStructVersion1 (sl_reflex.h:42)
pub(crate) const REFLEX_OPTIONS: StructType = StructType {
    data1: 0xf03af81a,
    data2: 0x6d0b,
    data3: 0x4902,
    data4: [0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34],
};
pub(crate) const REFLEX_OPTIONS_VERSION: usize = STRUCT_VERSION_1;

/// {0D569B37-A1C8-4453-BE4D-40F4DE57952B}, kStructVersion1 (sl_reflex.h:65)
pub(crate) const REFLEX_REPORT: StructType = StructType {
    data1: 0x0d569b37,
    data2: 0xa1c8,
    data3: 0x4453,
    data4: [0xbe, 0x4d, 0x40, 0xf4, 0xde, 0x57, 0x95, 0x2b],
};

/// {68BB0632-5E1C-402B-899D-B49F633C56C2}, kStructVersion1 (sl_reflex.h:87)
pub(crate) const REFLEX_REPORT2: StructType = StructType {
    data1: 0x68bb0632,
    data2: 0x5e1c,
    data3: 0x402b,
    data4: [0x89, 0x9d, 0xb4, 0x9f, 0x63, 0x3c, 0x56, 0xc2],
};

/// {F0BB5985-DAF9-4728-B2FD-AE80A2BD7989}, kStructVersion2 (sl_reflex.h:97)
pub(crate) const REFLEX_STATE: StructType = StructType {
    data1: 0xf0bb5985,
    data2: 0xdaf9,
    data3: 0x4728,
    data4: [0xb2, 0xfd, 0xae, 0x80, 0xa2, 0xbd, 0x79, 0x89],
};
pub(crate) const REFLEX_STATE_VERSION: usize = STRUCT_VERSION_2;

/// {CFA32F9B-023C-420E-9056-6832B74F89B4}, kStructVersion1 (sl_pcl.h:42)
pub(crate) const PCL_OPTIONS: StructType = StructType {
    data1: 0xcfa32f9b,
    data2: 0x023c,
    data3: 0x420e,
    data4: [0x90, 0x56, 0x68, 0x32, 0xb7, 0x4f, 0x89, 0xb4],
};

/// {CFA32F9B-023C-420E-9056-6832B74F89B5}, kStructVersion1 (sl_pcl.h:53) (last byte b5)
pub(crate) const PCL_STATE: StructType = StructType {
    data1: 0xcfa32f9b,
    data2: 0x023c,
    data3: 0x420e,
    data4: [0x90, 0x56, 0x68, 0x32, 0xb7, 0x4f, 0x89, 0xb5],
};

/// {830A0F35-DB84-4171-A804-59B206499B18}, kStructVersion1 (sl_core_types.h:606). For
/// reference only: a FrameToken is created by `slGetNewFrameToken` and stays opaque (it is a
/// C++ class with a virtual `operator uint32_t`, so it has a v-table pointer; never build one).
pub(crate) const FRAME_TOKEN: StructType = StructType {
    data1: 0x830a0f35,
    data2: 0xdb84,
    data3: 0x4171,
    data4: [0xa8, 0x04, 0x59, 0xb2, 0x06, 0x49, 0x9b, 0x18],
};

/// {0EED6FD5-82CD-43A9-BDB5-47A5BA2F45D6}, kStructVersion3 (sl_helpers_vk.h:222). The struct
/// itself is already in streamline.rs; noted here because its version is 3.
pub(crate) const VULKAN_INFO_VERSION: usize = STRUCT_VERSION_3;

// ---------------------------------------------------------------------------------------------
// DLSS Ray Reconstruction (sl_dlss_d.h)
// ---------------------------------------------------------------------------------------------

// sl::DLSSMode (sl_dlss.h:34-44) is shared by DLSS-SR and DLSS-RR.
pub(crate) const DLSS_MODE_OFF: u32 = 0;
pub(crate) const DLSS_MODE_MAX_PERFORMANCE: u32 = 1;
pub(crate) const DLSS_MODE_BALANCED: u32 = 2;
pub(crate) const DLSS_MODE_MAX_QUALITY: u32 = 3;
pub(crate) const DLSS_MODE_ULTRA_PERFORMANCE: u32 = 4;
pub(crate) const DLSS_MODE_ULTRA_QUALITY: u32 = 5;
pub(crate) const DLSS_MODE_DLAA: u32 = 6;

// sl::DLSSDPreset (sl_dlss_d.h:30-51, uint32_t). A, B, C were removed; D and E are transformer
// models, F is "latest and default transformer model"; G to O revert to default.
pub(crate) const DLSSD_PRESET_DEFAULT: u32 = 0;
pub(crate) const DLSSD_PRESET_D: u32 = 4;
pub(crate) const DLSSD_PRESET_E: u32 = 5;
pub(crate) const DLSSD_PRESET_F: u32 = 6;
pub(crate) const DLSSD_PRESET_G: u32 = 7;
pub(crate) const DLSSD_PRESET_H: u32 = 8;
pub(crate) const DLSSD_PRESET_I: u32 = 9;
pub(crate) const DLSSD_PRESET_J: u32 = 10;
pub(crate) const DLSSD_PRESET_K: u32 = 11;
pub(crate) const DLSSD_PRESET_L: u32 = 12;
pub(crate) const DLSSD_PRESET_M: u32 = 13;
pub(crate) const DLSSD_PRESET_N: u32 = 14;
pub(crate) const DLSSD_PRESET_O: u32 = 15;

// sl::DLSSDNormalRoughnessMode (sl_dlss_d.h:53-59, uint32_t)
/// Normals (kBufferTypeNormals) and roughness (kBufferTypeRoughness) as separate resources.
pub(crate) const NORMAL_ROUGHNESS_UNPACKED: u32 = 0;
/// Roughness in the w channel of the normals (tag as kBufferTypeNormalRoughness).
pub(crate) const NORMAL_ROUGHNESS_PACKED: u32 = 1;

/// sl::float4x4: four row-major float4 rows (sl_consts.h:131-141), 64 bytes, align 4.
pub(crate) type Float4x4 = [[f32; 4]; 4];

/// sl::DLSSDOptions (sl_dlss_d.h:62-101), kStructVersion3.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DlssdOptions {
    pub(crate) base: Base,
    pub(crate) mode: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) sharpness: f32,
    pub(crate) pre_exposure: f32,
    pub(crate) exposure_scale: f32,
    pub(crate) color_buffers_hdr: u8,
    pub(crate) indicator_invert_axis_x: u8,
    pub(crate) indicator_invert_axis_y: u8,
    pub(crate) normal_roughness_mode: u32,
    /// Needed when kBufferTypeSpecularHitDistance is tagged instead of specular mvecs.
    pub(crate) world_to_camera_view: Float4x4,
    /// = world_to_camera_view.inverse()
    pub(crate) camera_view_to_world: Float4x4,
    pub(crate) alpha_upscaling_enabled: u8,
    pub(crate) dlaa_preset: u32,
    pub(crate) quality_preset: u32,
    pub(crate) balanced_preset: u32,
    pub(crate) performance_preset: u32,
    pub(crate) ultra_performance_preset: u32,
    pub(crate) ultra_quality_preset: u32,
}

/// sl::DLSSDOptimalSettings (sl_dlss_d.h:106-123), kStructVersion1. Unlike the DLSS-SR one in
/// streamline.rs (`DlssOptimalSettingsS`), this is flat: the base is the first member.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DlssdOptimalSettings {
    pub(crate) base: Base,
    pub(crate) optimal_render_width: u32,
    pub(crate) optimal_render_height: u32,
    pub(crate) optimal_sharpness: f32,
    pub(crate) render_width_min: u32,
    pub(crate) render_height_min: u32,
    pub(crate) render_width_max: u32,
    pub(crate) render_height_max: u32,
}

/// sl::DLSSDState (sl_dlss_d.h:128-133), kStructVersion1.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DlssdState {
    pub(crate) base: Base,
    pub(crate) estimated_vram_usage_in_bytes: u64,
}

// sl_dlss_d.h:146, 157, 168 (obtained through slGetFeatureFunction(kFeatureDLSS_RR, ..))
pub(crate) type FnDlssdGetOptimalSettings = unsafe extern "C" fn(*const DlssdOptions, *mut DlssdOptimalSettings) -> i32;
pub(crate) type FnDlssdGetState = unsafe extern "C" fn(*const ViewportHandle, *mut DlssdState) -> i32;
pub(crate) type FnDlssdSetOptions = unsafe extern "C" fn(*const ViewportHandle, *const DlssdOptions) -> i32;
pub(crate) const FN_DLSSD_GET_OPTIMAL_SETTINGS: &std::ffi::CStr = c"slDLSSDGetOptimalSettings";
pub(crate) const FN_DLSSD_GET_STATE: &std::ffi::CStr = c"slDLSSDGetState";
pub(crate) const FN_DLSSD_SET_OPTIONS: &std::ffi::CStr = c"slDLSSDSetOptions";

// ---------------------------------------------------------------------------------------------
// DLSS Frame Generation (sl_dlss_g.h)
// ---------------------------------------------------------------------------------------------

// sl::DLSSGMode (sl_dlss_g.h:34-41, uint32_t)
pub(crate) const DLSSG_MODE_OFF: u32 = 0;
pub(crate) const DLSSG_MODE_ON: u32 = 1;
/// Legacy: fixed multiplier, turns itself off when it would lower FPS.
pub(crate) const DLSSG_MODE_AUTO: u32 = 2;
/// Dynamic multi frame generation (D3D12 only, driver 595.41+; check
/// `DlssgState::b_is_dynamic_mfg_supported`). `num_frames_to_generate` is ignored.
pub(crate) const DLSSG_MODE_DYNAMIC: u32 = 3;

// sl::DLSSGFlags (sl_dlss_g.h:43-53, uint32_t)
pub(crate) const DLSSG_FLAG_SHOW_ONLY_INTERPOLATED_FRAME: u32 = 1 << 0;
pub(crate) const DLSSG_FLAG_DYNAMIC_RESOLUTION_ENABLED: u32 = 1 << 1;
/// Only with a non-null options pointer to slDLSSGGetState; expensive, never per frame.
pub(crate) const DLSSG_FLAG_REQUEST_VRAM_ESTIMATE: u32 = 1 << 2;
/// eOff suspends instead of freeing (no stutter on re-enable); free with slFreeResources.
pub(crate) const DLSSG_FLAG_RETAIN_RESOURCES_WHEN_OFF: u32 = 1 << 3;
/// Needs kBufferTypeUIColorAndAlpha.
pub(crate) const DLSSG_FLAG_ENABLE_FULLSCREEN_MENU_DETECTION: u32 = 1 << 4;

// sl::DLSSGQueueParallelismMode (sl_dlss_g.h:55-66, uint32_t)
pub(crate) const DLSSG_QUEUE_BLOCK_PRESENTING_CLIENT_QUEUE: u32 = 0;
/// Vulkan only (D3D falls back to the mode above); then the host must wait on
/// `inputs_processing_completion_fence` before touching last frame's tagged inputs.
pub(crate) const DLSSG_QUEUE_BLOCK_NO_CLIENT_QUEUES: u32 = 1;

// sl::DLSSGStatus (sl_dlss_g.h:125-143, uint32_t bit flags)
pub(crate) const DLSSG_STATUS_OK: u32 = 0;
pub(crate) const DLSSG_STATUS_FAIL_RESOLUTION_TOO_LOW: u32 = 1 << 0;
pub(crate) const DLSSG_STATUS_FAIL_REFLEX_NOT_DETECTED_AT_RUNTIME: u32 = 1 << 1;
pub(crate) const DLSSG_STATUS_FAIL_HDR_FORMAT_NOT_SUPPORTED: u32 = 1 << 2;
pub(crate) const DLSSG_STATUS_FAIL_COMMON_CONSTANTS_INVALID: u32 = 1 << 3;
/// D3D integrations must call GetCurrentBackBufferIndex on the (proxy) swapchain.
pub(crate) const DLSSG_STATUS_FAIL_GET_CURRENT_BACK_BUFFER_INDEX_NOT_CALLED: u32 = 1 << 4;
pub(crate) const DLSSG_STATUS_RESERVED5: u32 = 1 << 5;

/// `sl::PFunOnAPIErrorCallback = void(const APIError&)` (sl_core_types.h:508-517). `APIError`
/// is a union of `HRESULT hres` and `VkResult vkRes`: 4 bytes. Called on the present thread
/// only when the DXGI/Vulkan call returned non-zero; must return immediately.
pub(crate) type FnOnApiError = unsafe extern "C" fn(*const i32);

/// sl::DLSSGOptions (sl_dlss_g.h:72-122), kStructVersion5.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DlssgOptions {
    pub(crate) base: Base,
    pub(crate) mode: u32,
    /// Generated frames per real frame: 1 = 2x, 2 = 3x, 3 = 4x. At most
    /// `DlssgState::num_frames_to_generate_max`. Default 1.
    pub(crate) num_frames_to_generate: u32,
    pub(crate) flags: u32,
    pub(crate) dynamic_res_width: u32,
    pub(crate) dynamic_res_height: u32,
    pub(crate) num_back_buffers: u32,
    pub(crate) mvec_depth_width: u32,
    pub(crate) mvec_depth_height: u32,
    pub(crate) color_width: u32,
    pub(crate) color_height: u32,
    /// Native format (DXGI_FORMAT / VkFormat) of the swapchain back buffers.
    pub(crate) color_buffer_format: u32,
    pub(crate) mvec_buffer_format: u32,
    pub(crate) depth_buffer_format: u32,
    pub(crate) hud_less_buffer_format: u32,
    pub(crate) ui_buffer_format: u32,
    /// `PFunOnAPIErrorCallback*`
    pub(crate) on_error_callback: Option<FnOnApiError>,
    // kStructVersion2
    /// Header default `eInvalid` (2).
    pub(crate) b_reserved15: u8,
    // kStructVersion3
    pub(crate) queue_parallelism_mode: u32,
    // kStructVersion4
    /// Interpolate hudless and UI separately and recomposite (needs both tagged).
    pub(crate) enable_user_interface_recomposition: u8,
    // kStructVersion5
    /// Dynamic MFG target FPS; 0.0 = the display's refresh rate.
    pub(crate) dynamic_target_frame_rate: f32,
}

impl DlssgOptions {
    /// The header's default member initialisers.
    pub(crate) const fn new() -> DlssgOptions {
        DlssgOptions {
            base: Base::new(DLSSG_OPTIONS, DLSSG_OPTIONS_VERSION),
            mode: DLSSG_MODE_OFF,
            num_frames_to_generate: 1,
            flags: 0,
            dynamic_res_width: 0,
            dynamic_res_height: 0,
            num_back_buffers: 0,
            mvec_depth_width: 0,
            mvec_depth_height: 0,
            color_width: 0,
            color_height: 0,
            color_buffer_format: 0,
            mvec_buffer_format: 0,
            depth_buffer_format: 0,
            hud_less_buffer_format: 0,
            ui_buffer_format: 0,
            on_error_callback: None,
            b_reserved15: BOOL_INVALID,
            queue_parallelism_mode: DLSSG_QUEUE_BLOCK_PRESENTING_CLIENT_QUEUE,
            enable_user_interface_recomposition: BOOL_FALSE,
            dynamic_target_frame_rate: 0.0,
        }
    }
}

/// sl::DLSSGState (sl_dlss_g.h:149-183), kStructVersion4.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DlssgState {
    pub(crate) base: Base,
    pub(crate) estimated_vram_usage_in_bytes: u64,
    /// DLSSGStatus bits
    pub(crate) status: u32,
    pub(crate) min_width_or_height: u32,
    /// Frames presented since the last slDLSSGGetState call.
    pub(crate) num_frames_actually_presented: u32,
    // kStructVersion2
    pub(crate) num_frames_to_generate_max: u32,
    pub(crate) b_reserved4: u8,
    pub(crate) b_is_vsync_support_available: u8,
    /// ID3D12Fence* / VkSemaphore (timeline) owned by the plugin.
    pub(crate) inputs_processing_completion_fence: *mut c_void,
    pub(crate) last_present_inputs_processing_completion_fence_value: u64,
    // kStructVersion4 (no version-3 members in the header)
    pub(crate) b_is_dynamic_mfg_supported: u8,
}

// sl_dlss_g.h:197, 208 (obtained through slGetFeatureFunction(kFeatureDLSS_G, ..))
/// `options` may be null; pass it only with DLSSG_FLAG_REQUEST_VRAM_ESTIMATE.
pub(crate) type FnDlssgGetState = unsafe extern "C" fn(*const ViewportHandle, *mut DlssgState, *const DlssgOptions) -> i32;
pub(crate) type FnDlssgSetOptions = unsafe extern "C" fn(*const ViewportHandle, *const DlssgOptions) -> i32;
pub(crate) const FN_DLSSG_GET_STATE: &std::ffi::CStr = c"slDLSSGGetState";
pub(crate) const FN_DLSSG_SET_OPTIONS: &std::ffi::CStr = c"slDLSSGSetOptions";

// ---------------------------------------------------------------------------------------------
// Reflex (sl_reflex.h) - required whenever DLSS-G is on (status bit
// DLSSG_STATUS_FAIL_REFLEX_NOT_DETECTED_AT_RUNTIME otherwise).
// ---------------------------------------------------------------------------------------------

// sl::ReflexMode (sl_reflex.h:30-39): a plain C enum (int), not an enum class.
pub(crate) const REFLEX_MODE_OFF: u32 = 0;
pub(crate) const REFLEX_MODE_LOW_LATENCY: u32 = 1;
pub(crate) const REFLEX_MODE_LOW_LATENCY_WITH_BOOST: u32 = 2;

/// sl::ReflexOptions (sl_reflex.h:42-62), kStructVersion1.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ReflexOptions {
    pub(crate) base: Base,
    pub(crate) mode: u32,
    /// FPS cap in microseconds (0 = none); works with mode eOff too.
    pub(crate) frame_limit_us: u32,
    pub(crate) use_markers_to_optimize: bool,
    /// VK_F13/F14/F15 instead of the custom PCL ping message; 0 = message.
    pub(crate) virtual_key: u16,
    pub(crate) id_thread: u32,
}

/// sl::ReflexReport (sl_reflex.h:65-84), kStructVersion1. Arrayed inside ReflexState.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ReflexReport {
    pub(crate) base: Base,
    pub(crate) frame_id: u64,
    pub(crate) input_sample_time: u64,
    pub(crate) sim_start_time: u64,
    pub(crate) sim_end_time: u64,
    pub(crate) render_submit_start_time: u64,
    pub(crate) render_submit_end_time: u64,
    pub(crate) present_start_time: u64,
    pub(crate) present_end_time: u64,
    pub(crate) driver_start_time: u64,
    pub(crate) driver_end_time: u64,
    pub(crate) os_render_queue_start_time: u64,
    pub(crate) os_render_queue_end_time: u64,
    pub(crate) gpu_render_start_time: u64,
    pub(crate) gpu_render_end_time: u64,
    pub(crate) gpu_active_render_time_us: u32,
    pub(crate) gpu_frame_time_us: u32,
}

/// sl::ReflexReport2 (sl_reflex.h:87-92), kStructVersion1.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ReflexReport2 {
    pub(crate) base: Base,
    pub(crate) camera_constructed_time: u64,
    pub(crate) cross_adapter_copy_time_us: u32,
}

pub(crate) const REFLEX_FRAME_REPORT_COUNT: usize = 64; // sl_reflex.h:94

/// sl::ReflexState (sl_reflex.h:97-113), kStructVersion2. 12848 bytes: heap-allocate it.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ReflexState {
    pub(crate) base: Base,
    pub(crate) low_latency_available: bool,
    pub(crate) latency_report_available: bool,
    /// Windows message id of the PCL ping (when virtual_key is 0).
    pub(crate) stats_window_message: u32,
    pub(crate) frame_report: [ReflexReport; REFLEX_FRAME_REPORT_COUNT],
    pub(crate) flash_indicator_driver_controlled: bool,
    // kStructVersion2
    pub(crate) frame_report2: [ReflexReport2; REFLEX_FRAME_REPORT_COUNT],
}

// sl_reflex.h:156, 166, 176 (obtained through slGetFeatureFunction(kFeatureReflex, ..))
pub(crate) type FnReflexGetState = unsafe extern "C" fn(*mut ReflexState) -> i32;
/// `const sl::FrameToken&` -> the token pointer from slGetNewFrameToken.
pub(crate) type FnReflexSleep = unsafe extern "C" fn(*const c_void) -> i32;
pub(crate) type FnReflexSetOptions = unsafe extern "C" fn(*const ReflexOptions) -> i32;
pub(crate) const FN_REFLEX_GET_STATE: &std::ffi::CStr = c"slReflexGetState";
pub(crate) const FN_REFLEX_SLEEP: &std::ffi::CStr = c"slReflexSleep";
pub(crate) const FN_REFLEX_SET_OPTIONS: &std::ffi::CStr = c"slReflexSetOptions";
// (also slReflexSetCameraData / slReflexGetPredictedCameraData, sl_reflex.h:188, 200: Reflex
// Frame Warp only, not needed for DLSS-G)

// ---------------------------------------------------------------------------------------------
// PC Latency (sl_pcl.h)
// ---------------------------------------------------------------------------------------------

// sl::PCLMarker (sl_pcl.h:60-87, uint32_t)
pub(crate) const PCL_SIMULATION_START: u32 = 0;
pub(crate) const PCL_SIMULATION_END: u32 = 1;
pub(crate) const PCL_RENDER_SUBMIT_START: u32 = 2;
pub(crate) const PCL_RENDER_SUBMIT_END: u32 = 3;
pub(crate) const PCL_PRESENT_START: u32 = 4;
pub(crate) const PCL_PRESENT_END: u32 = 5;
// 6 was eInputSample, deprecated and removed
pub(crate) const PCL_TRIGGER_FLASH: u32 = 7;
pub(crate) const PCL_PC_LATENCY_PING: u32 = 8;
pub(crate) const PCL_OUT_OF_BAND_RENDER_SUBMIT_START: u32 = 9;
pub(crate) const PCL_OUT_OF_BAND_RENDER_SUBMIT_END: u32 = 10;
pub(crate) const PCL_OUT_OF_BAND_PRESENT_START: u32 = 11;
pub(crate) const PCL_OUT_OF_BAND_PRESENT_END: u32 = 12;
pub(crate) const PCL_CONTROLLER_INPUT_SAMPLE: u32 = 13;
pub(crate) const PCL_DELTA_T_CALCULATION: u32 = 14;
pub(crate) const PCL_LATE_WARP_PRESENT_START: u32 = 15;
pub(crate) const PCL_LATE_WARP_PRESENT_END: u32 = 16;
pub(crate) const PCL_CAMERA_CONSTRUCTED: u32 = 17;
pub(crate) const PCL_LATE_WARP_RENDER_SUBMIT_START: u32 = 18;
pub(crate) const PCL_LATE_WARP_RENDER_SUBMIT_END: u32 = 19;
pub(crate) const PCL_VENDOR_INTERNAL_ASYNC_PRESENT_START: u32 = 20;
pub(crate) const PCL_VENDOR_INTERNAL_ASYNC_PRESENT_END: u32 = 21;
pub(crate) const PCL_NUM_PRESENTS_IN_BATCH: u32 = 22;

// sl::PCLHotKey (sl_pcl.h:33-39, int16_t)
pub(crate) const PCL_HOTKEY_USE_PING_MESSAGE: i16 = 0;
pub(crate) const PCL_HOTKEY_VK_F13: i16 = 0x7c;
pub(crate) const PCL_HOTKEY_VK_F14: i16 = 0x7d;
pub(crate) const PCL_HOTKEY_VK_F15: i16 = 0x7e;

/// sl::PCLOptions (sl_pcl.h:42-50), kStructVersion1.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PclOptions {
    pub(crate) base: Base,
    pub(crate) virtual_key: i16,
    pub(crate) id_thread: u32,
}

/// sl::PCLState (sl_pcl.h:53-58), kStructVersion1.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PclState {
    pub(crate) base: Base,
    pub(crate) stats_window_message: u32,
}

// sl_pcl.h:122, 133, 143 (obtained through slGetFeatureFunction(kFeaturePCL, ..))
pub(crate) type FnPclGetState = unsafe extern "C" fn(*mut PclState) -> i32;
/// (marker, `const sl::FrameToken&`)
pub(crate) type FnPclSetMarker = unsafe extern "C" fn(u32, *const c_void) -> i32;
pub(crate) type FnPclSetOptions = unsafe extern "C" fn(*const PclOptions) -> i32;
pub(crate) const FN_PCL_GET_STATE: &std::ffi::CStr = c"slPCLGetState";
pub(crate) const FN_PCL_SET_MARKER: &std::ffi::CStr = c"slPCLSetMarker";
pub(crate) const FN_PCL_SET_OPTIONS: &std::ffi::CStr = c"slPCLSetOptions";

// ---------------------------------------------------------------------------------------------
// Core exports not yet in streamline.rs (sl_core_api.h:50-71; exports.def lists them all)
// ---------------------------------------------------------------------------------------------

/// slSetFeatureLoaded(Feature, bool) (sl_core_api.h:54). Loading/unloading sl.dlss_g decides
/// whether the next swapchain created through a proxy factory is a DLSS-G proxy.
pub(crate) type FnSetFeatureLoaded = unsafe extern "C" fn(u32, bool) -> i32;
/// slGetNativeInterface(void* proxy, void** base) (sl_core_api.h:68)
pub(crate) type FnGetNativeInterface = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
/// slAllocateResources(CommandBuffer*, Feature, const ViewportHandle&) (sl_core_api.h:56)
pub(crate) type FnAllocateResources = unsafe extern "C" fn(*mut c_void, u32, *const ViewportHandle) -> i32;
// Already in streamline.rs: slInit, slIsFeatureSupported, slIsFeatureLoaded, slSetD3DDevice
// (:71, takes the NATIVE ID3D12Device*), slSetVulkanInfo (sl_helpers_vk.h:245),
// slGetFeatureRequirements (:64), slGetFeatureFunction (:69), slGetNewFrameToken (:70),
// slSetTagForFrame (:63), slSetConstants (:67), slEvaluateFeature (:55), slFreeResources (:57),
// slUpgradeInterface (:66, `void**` replaced in place with the proxy).

// ---------------------------------------------------------------------------------------------
// Swapchain interposition
// ---------------------------------------------------------------------------------------------

// sl.interposer.dll also exports the D3D/DXGI entry points (source/core/sl.interposer/
// exports.def): D3D12CreateDevice, CreateDXGIFactory, CreateDXGIFactory1, CreateDXGIFactory2,
// DXGIGetDebugInterface1, ... Created through these, the factory/device are already SL proxies
// (no slUpgradeInterface needed). Signatures as in docs/ProgrammingGuide.md (WINAPI):
pub(crate) type FnCreateDxgiFactory2 = unsafe extern "system" fn(u32, *const GUID, *mut *mut c_void) -> HRESULT;
pub(crate) type FnCreateDxgiFactory1 = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;
pub(crate) const FN_CREATE_DXGI_FACTORY2: &std::ffi::CStr = c"CreateDXGIFactory2";

// Vulkan: load sl.interposer.dll's own `vkGetDeviceProcAddr` / `vkGetInstanceProcAddr`
// (exports). They return vulkan-1.dll functions for everything EXCEPT the hooked ones, for
// which they return SL proxies: vkCreateSwapchainKHR, vkDestroySwapchainKHR,
// vkGetSwapchainImagesKHR, vkAcquireNextImageKHR, vkQueuePresentKHR (mandatory), plus
// vkDeviceWaitIdle, vkCreateWin32SurfaceKHR, vkDestroySurfaceKHR, and optionally
// vkCreateInstance/vkCreateDevice (which add SL's extensions/queues; otherwise call
// slGetFeatureRequirements per feature yourself and then slSetVulkanInfo).
// (docs/ProgrammingGuideManualHooking.md 4.2)
pub(crate) type FnVkGetDeviceProcAddr = vk::PFN_vkGetDeviceProcAddr;
pub(crate) type FnVkGetInstanceProcAddr = vk::PFN_vkGetInstanceProcAddr;
pub(crate) const FN_VK_GET_DEVICE_PROC_ADDR: &std::ffi::CStr = c"vkGetDeviceProcAddr";
pub(crate) const FN_VK_GET_INSTANCE_PROC_ADDR: &std::ffi::CStr = c"vkGetInstanceProcAddr";

// sl::FunctionHookID (sl_hooks.h:46-79, uint32_t): the calls that must go through the proxy
// under manual hooking. No API takes these values; they name the hooked methods.
pub(crate) const HOOK_IDXGI_FACTORY_CREATE_SWAP_CHAIN: u32 = 0;
pub(crate) const HOOK_IDXGI_FACTORY_CREATE_SWAP_CHAIN_FOR_HWND: u32 = 1;
pub(crate) const HOOK_IDXGI_FACTORY_CREATE_SWAP_CHAIN_FOR_CORE_WINDOW: u32 = 2;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_PRESENT: u32 = 3;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_PRESENT1: u32 = 4;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_GET_BUFFER: u32 = 5;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_GET_DESC: u32 = 6;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_RESIZE_BUFFERS: u32 = 7;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_RESIZE_BUFFERS1: u32 = 8;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_GET_CURRENT_BACK_BUFFER_INDEX: u32 = 9;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_SET_FULLSCREEN_STATE: u32 = 10;
pub(crate) const HOOK_IDXGI_SWAP_CHAIN_DESTROYED: u32 = 11; // internal
pub(crate) const HOOK_ID3D12_DEVICE_CREATE_COMMAND_QUEUE: u32 = 12;
pub(crate) const HOOK_VULKAN_PRESENT: u32 = 13;
pub(crate) const HOOK_VULKAN_CREATE_SWAPCHAIN_KHR: u32 = 14;
pub(crate) const HOOK_VULKAN_DESTROY_SWAPCHAIN_KHR: u32 = 15;
pub(crate) const HOOK_VULKAN_GET_SWAPCHAIN_IMAGES_KHR: u32 = 16;
pub(crate) const HOOK_VULKAN_ACQUIRE_NEXT_IMAGE_KHR: u32 = 17;
pub(crate) const HOOK_VULKAN_DEVICE_WAIT_IDLE: u32 = 18;
pub(crate) const HOOK_VULKAN_CREATE_WIN32_SURFACE_KHR: u32 = 19;
pub(crate) const HOOK_VULKAN_DESTROY_SURFACE_KHR: u32 = 20;

// ---------------------------------------------------------------------------------------------
// Layout checks. These sizes/offsets are computed by hand from the MSVC x64 rules (no public
// NVIDIA AbiValidation numbers exist for them in the v2.14.16 tree), so they guard the Rust
// side against reordering, not against a header mismatch.
// ---------------------------------------------------------------------------------------------

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(offset_of!(DlssdOptions, normal_roughness_mode) == 60);
    assert!(offset_of!(DlssdOptions, world_to_camera_view) == 64);
    assert!(offset_of!(DlssdOptions, camera_view_to_world) == 128);
    assert!(offset_of!(DlssdOptions, alpha_upscaling_enabled) == 192);
    assert!(offset_of!(DlssdOptions, dlaa_preset) == 196);
    assert!(size_of::<DlssdOptions>() == 224);
    assert!(size_of::<DlssdOptimalSettings>() == 64);
    assert!(size_of::<DlssdState>() == 40);
    assert!(offset_of!(DlssgOptions, ui_buffer_format) == 88);
    assert!(offset_of!(DlssgOptions, on_error_callback) == 96);
    assert!(offset_of!(DlssgOptions, b_reserved15) == 104);
    assert!(offset_of!(DlssgOptions, queue_parallelism_mode) == 108);
    assert!(offset_of!(DlssgOptions, enable_user_interface_recomposition) == 112);
    assert!(offset_of!(DlssgOptions, dynamic_target_frame_rate) == 116);
    assert!(size_of::<DlssgOptions>() == 120);
    assert!(offset_of!(DlssgState, b_reserved4) == 56);
    assert!(offset_of!(DlssgState, inputs_processing_completion_fence) == 64);
    assert!(offset_of!(DlssgState, b_is_dynamic_mfg_supported) == 80);
    assert!(size_of::<DlssgState>() == 88);
    assert!(offset_of!(ReflexOptions, virtual_key) == 42);
    assert!(size_of::<ReflexOptions>() == 48);
    assert!(size_of::<ReflexReport>() == 152);
    assert!(size_of::<ReflexReport2>() == 48);
    assert!(offset_of!(ReflexState, frame_report) == 40);
    assert!(offset_of!(ReflexState, flash_indicator_driver_controlled) == 9768);
    assert!(offset_of!(ReflexState, frame_report2) == 9776);
    assert!(size_of::<ReflexState>() == 12848);
    assert!(size_of::<PclOptions>() == 40);
    assert!(size_of::<PclState>() == 40);
};

// =============================================================================================
// CALL SEQUENCES
// =============================================================================================
//
// ---- DLSS-G, D3D12, manual hooking (docs/ProgrammingGuideManualHooking.md, ..DLSS_G.md,
//      ..Reflex.md, ..PCL.md) ----
//
// Once:
//  1. LoadLibraryExW(sl.interposer.dll) after checking its signature; GetProcAddress the sl*
//     exports above.
//  2. slInit(Preferences { flags: FLAG_DISABLE_CL_STATE_TRACKING | FLAG_USE_MANUAL_HOOKING
//     | FLAG_USE_FRAME_BASED_RESOURCE_TAGGING (| FLAG_DISABLE_DEBUG_TEXT),
//     features_to_load: [FEATURE_DLSS, FEATURE_DLSS_G, FEATURE_REFLEX, FEATURE_PCL
//     (, FEATURE_DLSS_RR)], render_api: D3D12, .. }, SDK_VERSION_2_14_16)
//     BEFORE any swapchain is created. Under manual hooking the device may be created before
//     or after slInit (Vulkan: after).
//  3. Create the NATIVE ID3D12Device with d3d12.dll; slSetD3DDevice(native device).
//  4. slIsFeatureSupported(FEATURE_DLSS_G, AdapterInfo{LUID}); slGetFeatureRequirements(
//     FEATURE_DLSS_G, ..): check REQ_HARDWARE_SCHEDULING_REQUIRED / REQ_VSYNC_OFF_REQUIRED,
//     os_version_detected >= os_version_required, driver_version_detected >=
//     driver_version_required. Same for FEATURE_REFLEX and FEATURE_PCL.
//  5. slGetFeatureFunction for slDLSSGSetOptions / slDLSSGGetState (FEATURE_DLSS_G),
//     slReflexSetOptions / slReflexSleep / slReflexGetState (FEATURE_REFLEX), slPCLSetMarker
//     / slPCLGetState (FEATURE_PCL). Only valid after step 3.
//  6. Proxies: `proxy_device = native_device; slUpgradeInterface(&proxy_device)`, create the
//     present queue through proxy_device.CreateCommandQueue (hook 12). `proxy_factory =
//     CreateDXGIFactory2(..) from dxgi.dll; slUpgradeInterface(&proxy_factory)` (or call
//     sl.interposer's own CreateDXGIFactory2 export, which already returns a proxy; with
//     FLAG_USE_DXGI_FACTORY_PROXY SL wraps instead of v-table patching). With sl.dlss_g loaded
//     (slSetFeatureLoaded(FEATURE_DLSS_G, true)), proxy_factory.CreateSwapChainForHwnd(queue,
//     ..) returns a DLSS-G proxy swapchain; slUpgradeInterface on the swapchain right after
//     creation is what makes SL run its presentCommon() every frame. Call Present/Present1/
//     GetBuffer/GetDesc/ResizeBuffers(1)/GetCurrentBackBufferIndex/SetFullscreenState on the
//     proxy; everything else on slGetNativeInterface(proxy). Do NOT request
//     DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT and do not call
//     GetFrameLatencyWaitableObject/SetMaximumFrameLatency; pace with slReflexSleep.
//  7. slReflexSetOptions(ReflexOptions { mode: REFLEX_MODE_LOW_LATENCY, frame_limit_us: 0,
//     .. }) at least once (must be called even when Reflex is off).
//  8. slDLSSGGetState(viewport, &state, null): num_frames_to_generate_max,
//     b_is_vsync_support_available, b_is_dynamic_mfg_supported.
//
// Per frame (one FrameToken carries the frame through every call; viewport id the same in
// tags, constants, options):
//  1. slGetNewFrameToken(&token, &frame_index) at simulation start.
//  2. slReflexSleep(token)                         (always, even with Reflex off)
//  3. slPCLSetMarker(PCL_SIMULATION_START, token)
//     .. input + simulation ..
//     slPCLSetMarker(PCL_SIMULATION_END, token)
//  4. slPCLSetMarker(PCL_RENDER_SUBMIT_START, token)
//  5. slSetConstants(&constants, token, &viewport)  (as early as possible; no jitter in
//     matrices, row-major, mvec_scale normalises to [-1,1])
//  6. Scene: slSetTagForFrame(token, &viewport, [depth (BUFFER_DEPTH), mvec
//     (BUFFER_MOTION_VECTORS)], cmd_list) with VALID_UNTIL_PRESENT (fall back to
//     ONLY_VALID_NOW only if the content is wrong at present time). Manual hooking: each
//     Resource must carry the D3D12 state it will be in when SL uses it (at Present for
//     VALID_UNTIL_PRESENT; for ONLY_VALID_NOW, the state on this command list, SL copies it).
//     (DLSS-SR evaluate as today, if used.)
//  7. After post-processing, before UI: tag BUFFER_HUD_LESS_COLOR (VALID_UNTIL_PRESENT;
//     extent/size == back buffer).
//  8. UI: tag BUFFER_UI_ALPHA (69, preferred) or BUFFER_UI_COLOR_AND_ALPHA (23, RGB
//     premultiplied, 0 where there is no UI), same size as the back buffer. Optional:
//     BUFFER_BACKBUFFER with a null resource and the subrect as extent, only to restrict FG.
//  9. slDLSSGSetOptions(&viewport, &DlssgOptions { mode: DLSSG_MODE_ON,
//     num_frames_to_generate: 1..=max, flags: DLSSG_FLAG_RETAIN_RESOURCES_WHEN_OFF, .. }) on
//     the present thread; it takes effect at the next Present. Only needs calling on change.
// 10. ExecuteCommandLists; slPCLSetMarker(PCL_RENDER_SUBMIT_END, token)
// 11. slPCLSetMarker(PCL_PRESENT_START, token); proxy_swapchain.Present(..);
//     slPCLSetMarker(PCL_PRESENT_END, token). The token's frame index must equal the one given
//     to slSetConstants, or the log says "common constants cannot be found for frame N".
// 12. Next frame's render target index from proxy_swapchain.GetCurrentBackBufferIndex().
//     Optionally slDLSSGGetState(viewport, &state, null) after Present for
//     num_frames_actually_presented, status bits, and the inputs-processed fence (wait on it
//     before overwriting last frame's tagged inputs on a non-presenting queue).
// Menus/loading/pause/resize/fullscreen switch: mode DLSSG_MODE_OFF and null tags. Turning FG
// on/off for real: wait idle, release back buffers + swapchain, slSetFeatureLoaded(
// FEATURE_DLSS_G, on), recreate the swapchain through the proxy factory.
// PCL ping: when a window message equals PclState.stats_window_message, slPCLSetMarker(
// PCL_PC_LATENCY_PING, token).
// Vulkan: same calls, but swapchain/present/acquire go through the sl.interposer
// vkGetDeviceProcAddr proxies; present semaphore must be signalled by the last submit and the
// acquire semaphore waited on before the frame's first submit (SL waits/signals them).
//
// ---- DLSS-RR (docs/ProgrammingGuideDLSS_RR.md) ----
//
// Replaces DLSS-SR (evaluate one or the other, never both); uses the DLSS quality modes.
// Once / on settings change:
//  1. slDLSSDGetOptimalSettings(&DlssdOptions { mode, output_width, output_height, .. },
//     &mut DlssdOptimalSettings) -> render size. No dynamic resolution (changing the input
//     size re-initialises the denoiser).
//  2. slDLSSDSetOptions(&viewport, &DlssdOptions { mode, output_width, output_height,
//     color_buffers_hdr: BOOL_TRUE (required), normal_roughness_mode, world_to_camera_view,
//     camera_view_to_world (needed when using specular hit distance), presets (0 or D/E/F),
//     alpha_upscaling_enabled, .. }). sharpness and auto-exposure are ignored.
// Per frame:
//  3. slSetConstants(&constants, token, &viewport)  (same Constants as DLSS-SR)
//  4. slSetTagForFrame(token, &viewport, tags, cmd_list), all at render resolution except the
//     output:
//       BUFFER_SCALING_INPUT_COLOR (3)   noisy RT colour, HDR
//       BUFFER_SCALING_OUTPUT_COLOR (4)  output resolution
//       BUFFER_DEPTH (0) or BUFFER_LINEAR_DEPTH (49)
//       BUFFER_MOTION_VECTORS (1)
//       BUFFER_ALBEDO (7)                diffuse albedo, linear (no sRGB)
//       BUFFER_SPECULAR_ALBEDO (8)       linear; EnvBRDFApprox2 in the guide
//       BUFFER_NORMALS (5) + BUFFER_ROUGHNESS (6) with NORMAL_ROUGHNESS_UNPACKED, or
//       BUFFER_NORMAL_ROUGHNESS (14) with roughness in .w and NORMAL_ROUGHNESS_PACKED
//       BUFFER_SPECULAR_MOTION_VECTORS (10) or BUFFER_SPECULAR_HIT_DISTANCE (42) + the two
//       matrices in DlssdOptions
//       optional: BUFFER_TRANSPARENCY_LAYER (51) / _OPACITY (52),
//       BUFFER_COLOR_BEFORE_TRANSPARENCY (40), BUFFER_SSS_GUIDE (58),
//       BUFFER_DEPTH_OF_FIELD_GUIDE (64)
//  5. slEvaluateFeature(FEATURE_DLSS_RR, token, [&viewport as *const Base], 1, cmd_list), then
//     restore the command list state ourselves (manual hooking / CL state tracking off).
//  6. Optional: slDLSSDGetState(&viewport, &mut DlssdState) for VRAM; slFreeResources(
//     FEATURE_DLSS_RR, &viewport) when RR is switched off for good (mode eOff frees nothing).
