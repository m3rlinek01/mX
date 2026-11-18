// The Streamline structures, constants and identifiers below are transcribed from NVIDIA's
// Streamline SDK headers (https://github.com/NVIDIA-RTX/Streamline), which come under this
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

//! NVIDIA DLSS by way of NVIDIA Streamline: Super Resolution (and DLAA at full size), Ray
//! Reconstruction and Frame Generation; Windows, Direct3D 12 or Vulkan, a GeForce RTX card.
//!
//! One Streamline session a process (`Session`) serves them all, with manual hooking: DLSS
//! records into a command list of ours on mX's queue, after the work submitted so far (the
//! queue's order is the only synchronisation; textures go over in mX's resting states), and
//! DLSS-G makes the window's swapchain through Streamline's proxies (the surface's maker).

use crate::d3d12::{ListRing, wide};
use crate::frames::{FrameInputs, Show};
use crate::sl_ext::*;
use crate::vulkan::{VkGpu, VkRing};
use crate::{Constants as FrameConstants, Inputs, Method, Options, Quality as TemporalQuality, render_size as fallback_render_size};
use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use windows::Win32::Foundation::{FreeLibrary, HMODULE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    DXGI_FRAME_STATISTICS, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC, DXGI_SWAP_CHAIN_FLAG, IDXGIDeviceSubObject_Impl, IDXGIObject_Impl, IDXGIOutput, IDXGISwapChain, IDXGISwapChain_Impl, IDXGISwapChain1,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW};
use windows::core::{Interface, PCSTR, PCWSTR};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct StructType {
    pub(crate) data1: u32,
    pub(crate) data2: u16,
    pub(crate) data3: u16,
    pub(crate) data4: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Base {
    pub(crate) next: *mut c_void,
    pub(crate) struct_type: StructType,
    pub(crate) struct_version: usize,
}

impl Base {
    pub(crate) const fn new(struct_type: StructType, struct_version: usize) -> Base {
        Base {
            next: std::ptr::null_mut(),
            struct_type,
            struct_version,
        }
    }
}

const PREFERENCES: StructType = StructType {
    data1: 0x1ca10965,
    data2: 0xbf8e,
    data3: 0x432b,
    data4: [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14],
};
const RESOURCE: StructType = StructType {
    data1: 0x3a9d70cf,
    data2: 0x2418,
    data3: 0x4b72,
    data4: [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61],
};
const RESOURCE_TAG: StructType = StructType {
    data1: 0x4c6a5aad,
    data2: 0xb445,
    data3: 0x496c,
    data4: [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53],
};
const VIEWPORT_HANDLE: StructType = StructType {
    data1: 0x171b6435,
    data2: 0x9b3c,
    data3: 0x4fc8,
    data4: [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4],
};
const ADAPTER_INFO: StructType = StructType {
    data1: 0x0677315f,
    data2: 0xa746,
    data3: 0x4492,
    data4: [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4],
};
const CONSTANTS: StructType = StructType {
    data1: 0xdcd35ad7,
    data2: 0x4e4a,
    data3: 0x4bad,
    data4: [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe],
};
const DLSS_OPTIONS: StructType = StructType {
    data1: 0x6ac826e4,
    data2: 0x4c61,
    data3: 0x4101,
    data4: [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8],
};
const VULKAN_INFO: StructType = StructType {
    data1: 0x0eed6fd5,
    data2: 0x82cd,
    data3: 0x43a9,
    data4: [0xbd, 0xb5, 0x47, 0xa5, 0xba, 0x2f, 0x45, 0xd6],
};
const FEATURE_REQUIREMENTS: StructType = StructType {
    data1: 0x66714097,
    data2: 0xac6d,
    data3: 0x4bc6,
    data4: [0x89, 0x15, 0x1e, 0x0f, 0x55, 0xa6, 0xb6, 0x1f],
};
const DLSS_OPTIMAL_SETTINGS: StructType = StructType {
    data1: 0xef1d0957,
    data2: 0xfd58,
    data3: 0x4df7,
    data4: [0xb5, 0x04, 0x8b, 0x69, 0xd8, 0xaa, 0x6b, 0x76],
};

const SDK_VERSION: u64 = SDK_VERSION_2_14_16;
const FEATURE_DLSS: u32 = 0;
const BUFFER_DEPTH: u32 = 0;
const BUFFER_MOTION_VECTORS: u32 = 1;
// sl::DLSSPreset
const PRESET_K: u32 = 11;
const PRESET_L: u32 = 12;
const PRESET_M: u32 = 13;
// kBufferTypeBiasCurrentColorHint
const BUFFER_BIAS_CURRENT_COLOR: u32 = 29;
const BUFFER_SCALING_INPUT_COLOR: u32 = 3;
const BUFFER_SCALING_OUTPUT_COLOR: u32 = 4;
const VALID_UNTIL_EVALUATE: u32 = 2;
const FLAG_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
const FLAG_DISABLE_DEBUG_TEXT: u64 = 1 << 1;
const FLAG_USE_MANUAL_HOOKING: u64 = 1 << 2;
const FLAG_FRAME_BASED_TAGGING: u64 = 1 << 7;
const FALSE: u8 = 0;
const TRUE: u8 = 1;
/// The project id NGX is given: a custom engine has to name one (any GUID of its own).
const PROJECT_ID: &CStr = c"08d79c23-3148-4b8e-b9cd-1bbcaefa49a1";

#[repr(C)]
struct Preferences {
    base: Base,
    show_console: bool,
    log_level: u32,
    paths_to_plugins: *const *const u16,
    num_paths_to_plugins: u32,
    path_to_logs_and_data: *const u16,
    allocate_callback: *const c_void,
    release_callback: *const c_void,
    log_message_callback: Option<unsafe extern "C" fn(u32, *const c_char)>,
    flags: u64,
    features_to_load: *const u32,
    num_features_to_load: u32,
    application_id: u32,
    engine: u32,
    engine_version: *const c_char,
    project_id: *const c_char,
    render_api: u32,
}

#[repr(C)]
struct Resource {
    base: Base,
    kind: u8,
    native: *mut c_void,
    memory: *mut c_void,
    view: *mut c_void,
    state: u32,
    width: u32,
    height: u32,
    native_format: u32,
    mip_levels: u32,
    array_layers: u32,
    gpu_virtual_address: u64,
    flags: u32,
    usage: u32,
    internal_flags: u16,
    reserved: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Extent {
    top: u32,
    left: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
struct ResourceTag {
    base: Base,
    resource: *mut Resource,
    kind: u32,
    lifecycle: u32,
    extent: Extent,
}

#[repr(C)]
pub(crate) struct ViewportHandle {
    pub(crate) base: Base,
    pub(crate) value: u32,
}

#[repr(C)]
struct AdapterInfo {
    base: Base,
    device_luid: *mut u8,
    device_luid_size_in_bytes: u32,
    vk_physical_device: *mut c_void,
}

#[repr(C)]
struct VulkanInfo {
    base: Base,
    device: vk::Device,
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    compute_queue_index: u32,
    compute_queue_family: u32,
    graphics_queue_index: u32,
    graphics_queue_family: u32,
    optical_flow_queue_index: u32,
    optical_flow_queue_family: u32,
    use_native_optical_flow_mode: bool,
    compute_queue_create_flags: u32,
    graphics_queue_create_flags: u32,
    optical_flow_queue_create_flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Version {
    major: u32,
    minor: u32,
    build: u32,
}

#[repr(C)]
struct FeatureRequirements {
    base: Base,
    flags: u32,
    max_num_cpu_threads: u32,
    max_num_viewports: u32,
    num_required_tags: u32,
    required_tags: *const u32,
    os_version_detected: Version,
    os_version_required: Version,
    driver_version_detected: Version,
    driver_version_required: Version,
    vk_num_compute_queues_required: u32,
    vk_num_graphics_queues_required: u32,
    vk_num_device_extensions: u32,
    vk_device_extensions: *const *const c_char,
    vk_num_instance_extensions: u32,
    vk_instance_extensions: *const *const c_char,
    vk_num_features12: u32,
    vk_features12: *const *const c_char,
    vk_num_features13: u32,
    vk_features13: *const *const c_char,
    vk_num_optical_flow_queues_required: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float2 {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
struct Constants {
    base: Base,
    camera_view_to_clip: [[f32; 4]; 4],
    clip_to_camera_view: [[f32; 4]; 4],
    clip_to_lens_clip: [[f32; 4]; 4],
    clip_to_prev_clip: [[f32; 4]; 4],
    prev_clip_to_clip: [[f32; 4]; 4],
    jitter_offset: Float2,
    mvec_scale: Float2,
    camera_pinhole_offset: Float2,
    camera_pos: Float3,
    camera_up: Float3,
    camera_right: Float3,
    camera_fwd: Float3,
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    camera_aspect_ratio: f32,
    motion_vectors_invalid_value: f32,
    depth_inverted: u8,
    camera_motion_included: u8,
    motion_vectors_3d: u8,
    reset: u8,
    orthographic_projection: u8,
    motion_vectors_dilated: u8,
    motion_vectors_jittered: u8,
    min_relative_linear_depth_object_separation: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DlssOptions {
    base: Base,
    mode: u32,
    output_width: u32,
    output_height: u32,
    sharpness: f32,
    pre_exposure: f32,
    exposure_scale: f32,
    color_buffers_hdr: u8,
    indicator_invert_axis_x: u8,
    indicator_invert_axis_y: u8,
    dlaa_preset: u32,
    quality_preset: u32,
    balanced_preset: u32,
    performance_preset: u32,
    ultra_performance_preset: u32,
    ultra_quality_preset: u32,
    use_auto_exposure: u8,
    alpha_upscaling_enabled: u8,
}

#[repr(C)]
#[derive(Default)]
struct DlssOptimalSettings {
    optimal_render_width: u32,
    optimal_render_height: u32,
    optimal_sharpness: f32,
    render_width_min: u32,
    render_height_min: u32,
    render_width_max: u32,
    render_height_max: u32,
}

#[repr(C)]
struct DlssOptimalSettingsS {
    base: Base,
    s: DlssOptimalSettings,
}

// (the sizes Streamline's own `sl::test::AbiValidation` expects: a field off by a byte
// fails the build here, not in NVIDIA's DLL)
const _: () = {
    assert!(std::mem::size_of::<Base>() == 32);
    assert!(std::mem::size_of::<Preferences>() == 144);
    assert!(std::mem::size_of::<Resource>() == 112);
    assert!(std::mem::size_of::<ResourceTag>() == 64);
    assert!(std::mem::size_of::<ViewportHandle>() == 40);
    assert!(std::mem::size_of::<Constants>() == 456);
    assert!(std::mem::size_of::<DlssOptions>() == 88);
    assert!(std::mem::size_of::<VulkanInfo>() == 96);
    assert!(std::mem::offset_of!(FeatureRequirements, vk_num_compute_queues_required) == 104);
    assert!(std::mem::offset_of!(FeatureRequirements, vk_device_extensions) == 120);
    assert!(std::mem::size_of::<FeatureRequirements>() == 184);
};

type Fn0 = unsafe extern "C" fn() -> i32;
type FnInit = unsafe extern "C" fn(*const Preferences, u64) -> i32;
type FnIsFeatureSupported = unsafe extern "C" fn(u32, *const AdapterInfo) -> i32;
type FnIsFeatureLoaded = unsafe extern "C" fn(u32, *mut bool) -> i32;
type FnSetD3DDevice = unsafe extern "C" fn(*mut c_void) -> i32;
type FnSetVulkanInfo = unsafe extern "C" fn(*const VulkanInfo) -> i32;
type FnGetFeatureRequirements = unsafe extern "C" fn(u32, *mut FeatureRequirements) -> i32;
type FnGetFeatureFunction = unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> i32;
type FnGetNewFrameToken = unsafe extern "C" fn(*mut *const c_void, *const u32) -> i32;
type FnSetTagForFrame = unsafe extern "C" fn(*const c_void, *const ViewportHandle, *const ResourceTag, u32, *mut c_void) -> i32;
type FnSetConstants = unsafe extern "C" fn(*const Constants, *const c_void, *const ViewportHandle) -> i32;
type FnEvaluateFeature = unsafe extern "C" fn(u32, *const c_void, *const *const Base, u32, *mut c_void) -> i32;
type FnFreeResources = unsafe extern "C" fn(u32, *const ViewportHandle) -> i32;
type FnUpgradeInterface = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type FnDlssGetOptimalSettings = unsafe extern "C" fn(*const DlssOptions, *mut DlssOptimalSettingsS) -> i32;
type FnDlssSetOptions = unsafe extern "C" fn(*const ViewportHandle, *const DlssOptions) -> i32;

/// `sl::Result::eWarnOutOfVRAM`: a warning. The call did its work, the card is just over its
/// memory budget; treated as a failure it would switch DLSS off on a card that is merely full.
const WARN_OUT_OF_VRAM: i32 = 39;

fn result_name(r: i32) -> &'static str {
    const NAMES: [&str; 41] = [
        "eOk",
        "eErrorIO",
        "eErrorDriverOutOfDate",
        "eErrorOSOutOfDate",
        "eErrorOSDisabledHWS",
        "eErrorDeviceNotCreated",
        "eErrorNoSupportedAdapterFound",
        "eErrorAdapterNotSupported",
        "eErrorNoPlugins",
        "eErrorVulkanAPI",
        "eErrorDXGIAPI",
        "eErrorD3DAPI",
        "eErrorNRDAPI",
        "eErrorNVAPI",
        "eErrorReflexAPI",
        "eErrorNGXFailed",
        "eErrorJSONParsing",
        "eErrorMissingProxy",
        "eErrorMissingResourceState",
        "eErrorInvalidIntegration",
        "eErrorMissingInputParameter",
        "eErrorNotInitialized",
        "eErrorComputeFailed",
        "eErrorInitNotCalled",
        "eErrorExceptionHandler",
        "eErrorInvalidParameter",
        "eErrorMissingConstants",
        "eErrorDuplicatedConstants",
        "eErrorMissingOrInvalidAPI",
        "eErrorCommonConstantsMissing",
        "eErrorUnsupportedInterface",
        "eErrorFeatureMissing",
        "eErrorFeatureNotSupported",
        "eErrorFeatureMissingHooks",
        "eErrorFeatureFailedToLoad",
        "eErrorFeatureWrongPriority",
        "eErrorFeatureMissingDependency",
        "eErrorFeatureManagerInvalidState",
        "eErrorInvalidState",
        "eWarnOutOfVRAM",
        "?",
    ];
    NAMES.get(r as usize).copied().unwrap_or("?")
}

fn check(what: &str, r: i32) -> Result<()> {
    static VRAM_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    match r {
        0 => Ok(()),
        WARN_OUT_OF_VRAM => {
            if !VRAM_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!("{what}: Streamline says the graphics card is over its memory budget; expect stutter, or lower the video memory or the window size");
            }
            Ok(())
        }
        _ => Err(anyhow!("{what} failed in Streamline: {} ({r})", result_name(r))),
    }
}

unsafe extern "C" fn sl_log(kind: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let text = text.trim_end();
    match kind {
        2 => log::error!("Streamline: {text}"),
        1 => log::warn!("Streamline: {text}"),
        _ => log::debug!("Streamline: {text}"),
    }
}

/// The quality as `sl::DLSSMode` numbers it (0 off, 1 max performance, 2 balanced, 3 max
/// quality, 4 ultra performance, 5 ultra quality, 6 DLAA).
fn mode_code(quality: TemporalQuality) -> u32 {
    match quality {
        TemporalQuality::Performance => 1,
        TemporalQuality::Balanced => 2,
        TemporalQuality::Quality => 3,
        TemporalQuality::UltraPerformance => 4,
        TemporalQuality::UltraQuality => 5,
        TemporalQuality::Native => 6,
    }
}

fn dxgi_format(f: mx::TextureFormat) -> u32 {
    use mx::TextureFormat as Tf;
    (match f {
        Tf::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        Tf::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        Tf::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        Tf::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        Tf::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        Tf::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        Tf::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        Tf::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        Tf::R16Float => DXGI_FORMAT_R16_FLOAT,
        Tf::Depth32Float => DXGI_FORMAT_R32_FLOAT,
        _ => DXGI_FORMAT_UNKNOWN,
    })
    .0 as u32
}

/// A swap chain whose `Present` does nothing: Streamline's per-frame housekeeping runs on its
/// proxy's present, and the window's real swap chain is mX's (manual hooking).
#[windows::core::implement(IDXGISwapChain)]
struct NullSwapChain {
    device: ID3D12Device,
}

fn not_implemented<T>() -> windows::core::Result<T> {
    Err(windows::Win32::Foundation::E_NOTIMPL.into())
}

impl IDXGIObject_Impl for NullSwapChain_Impl {
    fn SetPrivateData(&self, _: *const windows::core::GUID, _: u32, _: *const c_void) -> windows::core::Result<()> {
        not_implemented()
    }
    fn SetPrivateDataInterface(&self, _: *const windows::core::GUID, _: windows::core::Ref<windows::core::IUnknown>) -> windows::core::Result<()> {
        not_implemented()
    }
    fn GetPrivateData(&self, _: *const windows::core::GUID, _: *mut u32, _: *mut c_void) -> windows::core::Result<()> {
        not_implemented()
    }
    fn GetParent(&self, _: *const windows::core::GUID, _: *mut *mut c_void) -> windows::core::Result<()> {
        not_implemented()
    }
}

impl IDXGIDeviceSubObject_Impl for NullSwapChain_Impl {
    fn GetDevice(&self, riid: *const windows::core::GUID, device: *mut *mut c_void) -> windows::core::Result<()> {
        unsafe { self.device.query(riid, device) }.ok()
    }
}

impl IDXGISwapChain_Impl for NullSwapChain_Impl {
    fn Present(&self, _: u32, _: DXGI_PRESENT) -> windows::core::HRESULT {
        windows::Win32::Foundation::S_OK
    }
    fn GetBuffer(&self, _: u32, _: *const windows::core::GUID, _: *mut *mut c_void) -> windows::core::Result<()> {
        not_implemented()
    }
    fn SetFullscreenState(&self, _: windows::core::BOOL, _: windows::core::Ref<IDXGIOutput>) -> windows::core::Result<()> {
        not_implemented()
    }
    fn GetFullscreenState(&self, _: *mut windows::core::BOOL, _: windows::core::OutRef<IDXGIOutput>) -> windows::core::Result<()> {
        not_implemented()
    }
    fn GetDesc(&self) -> windows::core::Result<DXGI_SWAP_CHAIN_DESC> {
        not_implemented()
    }
    fn ResizeBuffers(&self, _: u32, _: u32, _: u32, _: DXGI_FORMAT, _: &DXGI_SWAP_CHAIN_FLAG) -> windows::core::Result<()> {
        Ok(())
    }
    fn ResizeTarget(&self, _: *const DXGI_MODE_DESC) -> windows::core::Result<()> {
        not_implemented()
    }
    fn GetContainingOutput(&self) -> windows::core::Result<IDXGIOutput> {
        not_implemented()
    }
    fn GetFrameStatistics(&self, _: *mut DXGI_FRAME_STATISTICS) -> windows::core::Result<()> {
        not_implemented()
    }
    fn GetLastPresentCount(&self) -> windows::core::Result<u32> {
        not_implemented()
    }
}

fn load_fn<T: Copy>(module: HMODULE, name: &CStr) -> Result<T> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<usize>());
    let f = unsafe { GetProcAddress(module, PCSTR(name.as_ptr().cast())) }.ok_or_else(|| anyhow!("{} is not exported by sl.interposer.dll (an older Streamline?)", name.to_string_lossy()))?;
    Ok(unsafe { std::mem::transmute_copy::<_, T>(&f) })
}

fn init(module: HMODULE, dir: &std::path::Path, features: &'static [u32], render_api: u32) -> Result<(Vec<u16>, Vec<*const u16>)> {
    let init: FnInit = load_fn(module, c"slInit")?;
    let plugin_dir = wide(dir);
    let plugin_paths = vec![plugin_dir.as_ptr()];
    let engine_version = concat!("mX ", env!("CARGO_PKG_VERSION"), "\0");
    let prefs = Preferences {
        base: Base::new(PREFERENCES, 1),
        show_console: false,
        log_level: 1,
        paths_to_plugins: plugin_paths.as_ptr(),
        num_paths_to_plugins: 1,
        path_to_logs_and_data: std::ptr::null(),
        allocate_callback: std::ptr::null(),
        release_callback: std::ptr::null(),
        log_message_callback: Some(sl_log),
        flags: FLAG_DISABLE_CL_STATE_TRACKING | FLAG_DISABLE_DEBUG_TEXT | FLAG_USE_MANUAL_HOOKING | FLAG_FRAME_BASED_TAGGING,
        features_to_load: features.as_ptr(),
        num_features_to_load: features.len() as u32,
        application_id: 0,
        engine: 0,
        engine_version: engine_version.as_ptr().cast(),
        project_id: PROJECT_ID.as_ptr(),
        // (sl::RenderAPI: 1 Direct3D 12, 2 Vulkan)
        render_api,
    };
    check("slInit", unsafe { init(&prefs, SDK_VERSION) })?;
    Ok((plugin_dir, plugin_paths))
}

// ---------------------------------------------------------------------------------------------
// The session: one `slInit` a process, shared by DLSS (SR or RR) and DLSS-G

/// Everything Streamline is asked to load: features whose plugins are missing are simply not
/// loaded (`Session::loaded`).
static FEATURES: [u32; 5] = [FEATURE_DLSS, FEATURE_DLSS_RR, FEATURE_DLSS_G, FEATURE_REFLEX, FEATURE_PCL];

#[allow(clippy::large_enum_variant)] // (one per upscaler, made once)
enum Device {
    Dx12(ID3D12Device),
    Vulkan(VkGpu),
}

/// Streamline, initialised for one device.
pub(crate) struct Session {
    module: HMODULE,
    shutdown: Fn0,
    get_new_frame_token: FnGetNewFrameToken,
    set_tag_for_frame: FnSetTagForFrame,
    set_constants: FnSetConstants,
    evaluate_feature: FnEvaluateFeature,
    free_resources: FnFreeResources,
    get_feature_function: FnGetFeatureFunction,
    upgrade_interface: Option<FnUpgradeInterface>,
    /// (kept alive: `slInit` was given pointers into them)
    _paths: (Vec<u16>, Vec<*const u16>),
    device: Device,
    dir: std::path::PathBuf,
    /// The features loaded that this card runs.
    loaded: Vec<u32>,
    /// This frame's index, and who (1: the upscaler, 2: frame generation) asked for its token.
    frame: std::sync::Mutex<(u32, u8)>,
    /// A frame generator presents through Streamline's swapchain: the upscaler's stand-in
    /// present (`Runtime::pacer`) is not needed then.
    generating: std::sync::atomic::AtomicBool,
}

// (Streamline's functions and handles; its calls are made from one thread at a time)
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

static SESSION: std::sync::Mutex<std::sync::Weak<Session>> = std::sync::Mutex::new(std::sync::Weak::new());

impl Session {
    /// The process's session for `device`, started if there is none.
    fn get(device: &mx::Device, options: &Options) -> Result<Arc<Session>> {
        let mut slot = SESSION.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = slot.upgrade() {
            let same = match (&s.device, device.as_dx12(), VkGpu::of(device)) {
                (Device::Dx12(d), Some(m), _) => d.as_raw() == m.raw().as_raw(),
                (Device::Vulkan(g), _, Some(v)) => g.device.handle() == v.device.handle(),
                _ => false,
            };
            if same {
                return Ok(s);
            }
            bail!("Streamline is already running on another device (one a process)");
        }
        let s = Arc::new(Session::start(device, options)?);
        *slot = Arc::downgrade(&s);
        Ok(s)
    }

    fn start(device: &mx::Device, options: &Options) -> Result<Session> {
        let dir = options.library_dir.clone();
        if let Some(d) = device.as_dx12() {
            let raw_device = d.raw().clone();
            // (the program fell back from Vulkan: `slInit` must not run on a session still open)
            vulkan_abandon();
            let module = open_interposer(&dir)?;
            return (|| {
                let shutdown: Fn0 = load_fn(module, c"slShutdown")?;
                let paths = init(module, &dir, &FEATURES, 1)?;
                Self::begin(module, dir.clone(), paths, Device::Dx12(raw_device)).inspect_err(|_| unsafe {
                    shutdown();
                })
            })()
            .inspect_err(|_| unsafe {
                let _ = FreeLibrary(module);
            });
        }
        let gpu = VkGpu::of(device).ok_or_else(|| anyhow!("not running on Direct3D 12 or Vulkan"))?;
        let prepared =
            PREPARED.lock().unwrap_or_else(|e| e.into_inner()).take().ok_or_else(|| {
                anyhow!("on Vulkan, DLSS needs its device extensions when the device is made (`Upscaler::dlss_vulkan_extensions`, before the device); this device goes on without it")
            })?;
        let module = prepared.module;
        let shutdown: Fn0 = load_fn(module, c"slShutdown")?;
        Self::begin(module, dir, prepared.paths, Device::Vulkan(gpu)).inspect_err(|_| unsafe {
            shutdown();
            let _ = FreeLibrary(module);
        })
    }

    fn begin(module: HMODULE, dir: std::path::PathBuf, paths: (Vec<u16>, Vec<*const u16>), device: Device) -> Result<Session> {
        let shutdown: Fn0 = load_fn(module, c"slShutdown")?;
        let is_supported: FnIsFeatureSupported = load_fn(module, c"slIsFeatureSupported")?;
        let is_loaded: FnIsFeatureLoaded = load_fn(module, c"slIsFeatureLoaded")?;
        let mut luid = windows::Win32::Foundation::LUID::default();
        let adapter = match &device {
            Device::Dx12(d) => {
                let set_device: FnSetD3DDevice = load_fn(module, c"slSetD3DDevice")?;
                check("slSetD3DDevice", unsafe { set_device(d.as_raw()) })?;
                luid = unsafe { d.GetAdapterLuid() };
                AdapterInfo {
                    base: Base::new(ADAPTER_INFO, 1),
                    device_luid: (&mut luid as *mut windows::Win32::Foundation::LUID).cast(),
                    device_luid_size_in_bytes: std::mem::size_of::<windows::Win32::Foundation::LUID>() as u32,
                    vk_physical_device: std::ptr::null_mut(),
                }
            }
            Device::Vulkan(gpu) => {
                let set_vulkan_info: FnSetVulkanInfo = load_fn(module, c"slSetVulkanInfo")?;
                let info = VulkanInfo {
                    base: Base::new(VULKAN_INFO, 3),
                    device: gpu.device.handle(),
                    instance: gpu.instance,
                    physical_device: gpu.physical,
                    compute_queue_index: gpu.queue_index,
                    compute_queue_family: gpu.family,
                    graphics_queue_index: gpu.queue_index,
                    graphics_queue_family: gpu.family,
                    optical_flow_queue_index: 0,
                    optical_flow_queue_family: 0,
                    use_native_optical_flow_mode: false,
                    compute_queue_create_flags: 0,
                    graphics_queue_create_flags: 0,
                    optical_flow_queue_create_flags: 0,
                };
                check("slSetVulkanInfo", unsafe { set_vulkan_info(&info) })?;
                AdapterInfo {
                    base: Base::new(ADAPTER_INFO, 1),
                    device_luid: std::ptr::null_mut(),
                    device_luid_size_in_bytes: 0,
                    vk_physical_device: vk::Handle::as_raw(gpu.physical) as *mut c_void,
                }
            }
        };
        let loaded: Vec<u32> = FEATURES
            .iter()
            .copied()
            .filter(|&f| {
                let mut on = false;
                unsafe { is_supported(f, &adapter) == 0 && is_loaded(f, &mut on) == 0 && on }
            })
            .collect();
        let _ = luid;
        let api = if matches!(device, Device::Dx12(_)) { "Direct3D 12" } else { "Vulkan" };
        log::info!(
            "Streamline {}.{}.{} loaded from {} ({api}); this card runs features {loaded:?} (0 DLSS, 1001 Ray Reconstruction, 1000 Frame Generation, 3 Reflex, 4 PCL)",
            SDK_VERSION >> 48,
            (SDK_VERSION >> 32) & 0xffff,
            (SDK_VERSION >> 16) & 0xffff,
            dir.display()
        );
        Ok(Session {
            module,
            shutdown,
            get_new_frame_token: load_fn(module, c"slGetNewFrameToken")?,
            set_tag_for_frame: load_fn(module, c"slSetTagForFrame")?,
            set_constants: load_fn(module, c"slSetConstants")?,
            evaluate_feature: load_fn(module, c"slEvaluateFeature")?,
            free_resources: load_fn(module, c"slFreeResources")?,
            get_feature_function: load_fn(module, c"slGetFeatureFunction")?,
            upgrade_interface: load_fn(module, c"slUpgradeInterface").ok(),
            _paths: paths,
            device,
            dir,
            loaded,
            frame: std::sync::Mutex::new((0, 0)),
            generating: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn has(&self, feature: u32) -> bool {
        self.loaded.contains(&feature)
    }

    /// A function of `feature`'s plugin.
    fn function<T: Copy>(&self, feature: u32, name: &CStr) -> Result<T> {
        debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<usize>());
        let mut f: *mut c_void = std::ptr::null_mut();
        check(&format!("slGetFeatureFunction({})", name.to_string_lossy()), unsafe {
            (self.get_feature_function)(feature, name.as_ptr(), &mut f)
        })?;
        if f.is_null() {
            bail!("Streamline does not offer {}", name.to_string_lossy());
        }
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&f) })
    }

    /// This frame's token: `who` (1 the upscaler, 2 frame generation) asking for it a second
    /// time starts the next frame.
    fn token(&self, who: u8) -> Result<*const c_void> {
        let mut f = self.frame.lock().unwrap_or_else(|e| e.into_inner());
        if f.1 & who != 0 {
            f.0 = f.0.wrapping_add(1);
            f.1 = 0;
        }
        f.1 |= who;
        let index = f.0;
        let mut token: *const c_void = std::ptr::null();
        check("slGetNewFrameToken", unsafe { (self.get_new_frame_token)(&mut token, &index) })?;
        if token.is_null() {
            bail!("slGetNewFrameToken returned no token");
        }
        Ok(token)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            let r = (self.shutdown)();
            if r != 0 {
                log::warn!("Streamline did not shut down cleanly: {}", result_name(r));
            }
            let _ = FreeLibrary(self.module);
        }
    }
}

struct Prepared {
    module: HMODULE,
    paths: (Vec<u16>, Vec<*const u16>),
}

// (handles and strings handed over to the render thread once)
unsafe impl Send for Prepared {}

static PREPARED: std::sync::Mutex<Option<Prepared>> = std::sync::Mutex::new(None);

fn open_interposer(dir: &std::path::Path) -> Result<HMODULE> {
    let missing = Method::Dlss.missing(dir, mx::Backend::Dx12);
    if missing.len() == Method::Dlss.files(mx::Backend::Dx12).len() || missing.contains(&"sl.interposer.dll") {
        bail!("{} not found in {} (they come with NVIDIA's Streamline SDK, bin/x64)", missing.join(", "), dir.display());
    }
    let interposer = wide(&dir.join("sl.interposer.dll"));
    unsafe { LoadLibraryExW(PCWSTR(interposer.as_ptr()), None, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }.context("opening sl.interposer.dll")
}

fn static_names(count: u32, names: *const *const c_char) -> Vec<&'static CStr> {
    (0..count as usize)
        .filter_map(|i| {
            let p = unsafe { *names.add(i) };
            (!p.is_null()).then(|| &*Box::leak(unsafe { CStr::from_ptr(p) }.to_owned().into_boxed_c_str()))
        })
        .collect()
}

/// The Vulkan 1.2 features DLSS may ask for that mX's device is made with.
pub const VULKAN_FEATURES: [&str; 3] = ["timelineSemaphore", "descriptorIndexing", "bufferDeviceAddress"];

/// The device extensions Streamline's features want of a Vulkan device, Streamline started
/// for it (before the device is made). Features that want what mX's device cannot give (queues
/// of their own, other features) are left out; DLSS itself must fit.
pub fn vulkan_extensions(options: &Options) -> Result<Vec<&'static CStr>> {
    vulkan_abandon();
    let dir = options.library_dir.clone();
    let module = open_interposer(&dir)?;
    #[allow(clippy::type_complexity)]
    let started = (|| -> Result<(Vec<&'static CStr>, (Vec<u16>, Vec<*const u16>))> {
        let shutdown: Fn0 = load_fn(module, c"slShutdown")?;
        let requirements: FnGetFeatureRequirements = load_fn(module, c"slGetFeatureRequirements")?;
        let paths = init(module, &dir, &FEATURES, 2)?;
        (|| {
            let mut extensions: Vec<&'static CStr> = Vec::new();
            for feature in FEATURES {
                let mut r = FeatureRequirements {
                    base: Base::new(FEATURE_REQUIREMENTS, 2),
                    flags: 0,
                    max_num_cpu_threads: 0,
                    max_num_viewports: 0,
                    num_required_tags: 0,
                    required_tags: std::ptr::null(),
                    os_version_detected: Version::default(),
                    os_version_required: Version::default(),
                    driver_version_detected: Version::default(),
                    driver_version_required: Version::default(),
                    vk_num_compute_queues_required: 0,
                    vk_num_graphics_queues_required: 0,
                    vk_num_device_extensions: 0,
                    vk_device_extensions: std::ptr::null(),
                    vk_num_instance_extensions: 0,
                    vk_instance_extensions: std::ptr::null(),
                    vk_num_features12: 0,
                    vk_features12: std::ptr::null(),
                    vk_num_features13: 0,
                    vk_features13: std::ptr::null(),
                    vk_num_optical_flow_queues_required: 0,
                };
                let fits = (|| -> Result<()> {
                    check("slGetFeatureRequirements", unsafe { requirements(feature, &mut r) })?;
                    if r.flags & REQ_VULKAN_SUPPORTED == 0 {
                        bail!("its plugin does not run on Vulkan");
                    }
                    let wanted: Vec<String> = static_names(r.vk_num_features12, r.vk_features12)
                        .iter()
                        .chain(&static_names(r.vk_num_features13, r.vk_features13))
                        .map(|c| c.to_string_lossy().into_owned())
                        .collect();
                    let other: Vec<&String> = wanted.iter().filter(|f| !VULKAN_FEATURES.contains(&f.as_str())).collect();
                    if !other.is_empty() {
                        bail!(
                            "it asks for Vulkan features mX's device is not made with ({})",
                            other.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                        );
                    }
                    if r.vk_num_compute_queues_required + r.vk_num_graphics_queues_required + r.vk_num_optical_flow_queues_required > 0 {
                        bail!("it asks for Vulkan queues of its own, which mX's device does not make");
                    }
                    Ok(())
                })();
                match fits {
                    Ok(()) => {
                        for e in static_names(r.vk_num_device_extensions, r.vk_device_extensions) {
                            if !extensions.contains(&e) {
                                extensions.push(e);
                            }
                        }
                    }
                    Err(e) if feature == FEATURE_DLSS => return Err(e.context("DLSS on Vulkan")),
                    Err(e) => log::info!("Streamline feature {feature} is left out on Vulkan: {e:#}"),
                }
            }
            Ok((extensions, paths))
        })()
        .inspect_err(|_| unsafe {
            shutdown();
        })
    })();
    match started {
        Ok((extensions, paths)) => {
            *PREPARED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Prepared { module, paths });
            Ok(extensions)
        }
        Err(e) => {
            unsafe {
                let _ = FreeLibrary(module);
            }
            Err(e)
        }
    }
}

pub fn vulkan_abandon() {
    if let Some(p) = PREPARED.lock().unwrap_or_else(|e| e.into_inner()).take() {
        if let Ok(shutdown) = load_fn::<Fn0>(p.module, c"slShutdown") {
            unsafe { shutdown() };
        }
        unsafe {
            let _ = FreeLibrary(p.module);
        }
    }
}

/// A Streamline resource for `t`, in the state mX leaves it in (Direct3D 12: its home state;
/// Vulkan: `GENERAL`), `size` of it used, read as `format`.
fn sl_resource(device: &Device, t: &mx::Texture, view: Option<&mx::TextureView>, format: mx::TextureFormat, size: (u32, u32)) -> Result<Resource> {
    let mut r = Resource {
        base: Base::new(RESOURCE, 1),
        kind: 0,
        native: std::ptr::null_mut(),
        memory: std::ptr::null_mut(),
        view: std::ptr::null_mut(),
        state: 0,
        width: size.0,
        height: size.1,
        native_format: 0,
        mip_levels: 1,
        array_layers: 1,
        gpu_virtual_address: 0,
        flags: 0,
        usage: 0,
        internal_flags: 0,
        reserved: 0,
    };
    match device {
        Device::Dx12(_) => {
            let d = t.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 resource under a Streamline texture"))?;
            r.native = d.raw().as_raw();
            r.state = d.rest_state().0 as u32;
            r.native_format = dxgi_format(format);
        }
        Device::Vulkan(_) => {
            let (image, memory) = crate::vulkan::image_and_memory(t)?;
            let made;
            let view = match view {
                Some(v) => v,
                None => {
                    made = t.create_view(&mx::TextureViewDescriptor {
                        format: Some(format),
                        ..Default::default()
                    })?;
                    &made
                }
            };
            r.native = vk::Handle::as_raw(image) as *mut c_void;
            r.memory = vk::Handle::as_raw(memory) as *mut c_void;
            r.view = vk::Handle::as_raw(crate::vulkan::view(view)?) as *mut c_void;
            r.state = vk::ImageLayout::GENERAL.as_raw() as u32;
            r.native_format = crate::vulkan::format(format).as_raw() as u32;
            r.flags = if t.format() != format { vk::ImageCreateFlags::MUTABLE_FORMAT.as_raw() } else { 0 };
            r.usage = crate::vulkan::usage(t.usage(), t.format().has_depth()).as_raw();
        }
    }
    Ok(r)
}

/// The constants of a frame as Streamline takes them.
fn sl_constants(c: &FrameConstants, render: (u32, u32)) -> Constants {
    let f2 = |v: [f32; 2]| Float2 { x: v[0], y: v[1] };
    let f3 = |v: [f32; 3]| Float3 { x: v[0], y: v[1], z: v[2] };
    let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    Constants {
        base: Base::new(CONSTANTS, 2),
        camera_view_to_clip: c.view_to_clip,
        clip_to_camera_view: c.clip_to_view,
        clip_to_lens_clip: identity,
        clip_to_prev_clip: c.clip_to_prev_clip,
        prev_clip_to_clip: c.prev_clip_to_clip,
        jitter_offset: f2(c.jitter),
        mvec_scale: f2([1.0 / render.0.max(1) as f32, 1.0 / render.1.max(1) as f32]),
        camera_pinhole_offset: f2([0.0, 0.0]),
        camera_pos: f3(c.camera_pos),
        camera_up: f3(c.camera_up),
        camera_right: f3(c.camera_right),
        camera_fwd: f3(c.camera_fwd),
        camera_near: c.near,
        camera_far: c.far,
        camera_fov: c.fov,
        camera_aspect_ratio: c.aspect,
        motion_vectors_invalid_value: 0.0,
        depth_inverted: TRUE,
        camera_motion_included: TRUE,
        motion_vectors_3d: FALSE,
        reset: if c.reset { TRUE } else { FALSE },
        orthographic_projection: FALSE,
        motion_vectors_dilated: FALSE,
        motion_vectors_jittered: FALSE,
        min_relative_linear_depth_object_separation: 40.0,
    }
}

// ---------------------------------------------------------------------------------------------
// DLSS Super Resolution and Ray Reconstruction

#[allow(clippy::large_enum_variant)] // (one per upscaler, made once)
enum Backend {
    Dx12 { ring: ListRing, pacer: Option<IDXGISwapChain> },
    Vulkan { ring: VkRing },
}

impl Backend {
    fn wait(&mut self) {
        match self {
            Backend::Dx12 { ring, .. } => ring.wait(),
            Backend::Vulkan { ring, .. } => ring.wait(),
        }
    }
}

/// DLSS's options: Super Resolution's, or Ray Reconstruction's.
enum Feature {
    Sr {
        get_optimal_settings: FnDlssGetOptimalSettings,
        set_options: FnDlssSetOptions,
    },
    Rr {
        get_optimal_settings: FnDlssdGetOptimalSettings,
        set_options: FnDlssdSetOptions,
    },
}

pub struct Runtime {
    session: Arc<Session>,
    feature: Feature,
    preset: u32,
    viewport: ViewportHandle,
    options_set: Option<(TemporalQuality, (u32, u32), bool)>,
    #[allow(clippy::type_complexity)] // (quality and output size, to render size)
    optimal: Vec<((TemporalQuality, (u32, u32)), (u32, u32))>,
    backend: Backend,
    allocated: bool,
}

// (only one thread at a time calls into Streamline; what is held are raw handles)
unsafe impl Send for Runtime {}

impl Runtime {
    pub(crate) fn wait(&mut self) {
        self.backend.wait();
    }

    fn feature_id(&self) -> u32 {
        match self.feature {
            Feature::Sr { .. } => FEATURE_DLSS,
            Feature::Rr { .. } => FEATURE_DLSS_RR,
        }
    }

    pub fn load(device: &mx::Device, options: &Options, ray_reconstruction: bool) -> Result<Runtime> {
        let session = Session::get(device, options)?;
        let id = if ray_reconstruction { FEATURE_DLSS_RR } else { FEATURE_DLSS };
        if !session.has(id) {
            bail!(
                "{} does not run here (it takes a GeForce RTX, an up-to-date driver and {} in {})",
                if ray_reconstruction { "DLSS Ray Reconstruction" } else { "DLSS" },
                if ray_reconstruction {
                    "sl.dlss_d.dll and nvngx_dlssd.dll"
                } else {
                    "sl.dlss.dll and nvngx_dlss.dll"
                },
                session.dir.display()
            );
        }
        let feature = if ray_reconstruction {
            Feature::Rr {
                get_optimal_settings: session.function(FEATURE_DLSS_RR, FN_DLSSD_GET_OPTIMAL_SETTINGS)?,
                set_options: session.function(FEATURE_DLSS_RR, FN_DLSSD_SET_OPTIONS)?,
            }
        } else {
            Feature::Sr {
                get_optimal_settings: session.function(FEATURE_DLSS, c"slDLSSGetOptimalSettings")?,
                set_options: session.function(FEATURE_DLSS, c"slDLSSSetOptions")?,
            }
        };
        let backend = match &session.device {
            Device::Dx12(device) => {
                let ring = ListRing::new(device).context("DLSS's command lists")?;
                // (Streamline's per-frame housekeeping runs on a swapchain proxy's present: a stand-in
                // one when no frame generator presents through Streamline's own)
                let pacer: IDXGISwapChain = NullSwapChain { device: device.clone() }.into();
                let mut raw = pacer.as_raw();
                let pacer = match session.upgrade_interface {
                    Some(upgrade) if unsafe { upgrade(&mut raw) } == 0 && !raw.is_null() && raw != pacer.as_raw() => Some(unsafe { IDXGISwapChain::from_raw(raw) }),
                    _ => {
                        log::warn!("DLSS: no swapchain proxy from Streamline, so its per-frame housekeeping is skipped");
                        None
                    }
                };
                Backend::Dx12 { ring, pacer }
            }
            Device::Vulkan(gpu) => Backend::Vulkan {
                ring: VkRing::new(gpu).context("DLSS's command buffers")?,
            },
        };
        Ok(Runtime {
            session,
            feature,
            preset: options.dlss_preset,
            viewport: ViewportHandle {
                base: Base::new(VIEWPORT_HANDLE, 1),
                value: 0,
            },
            options_set: None,
            optimal: Vec::new(),
            backend,
            allocated: false,
        })
    }

    /// `sl::DLSSOptions`, with the newer SDK's default presets set by hand: an older nvngx_dlss.dll
    /// would pick the CNN model.
    fn options(&self, quality: TemporalQuality, (w, h): (u32, u32), hdr: bool) -> DlssOptions {
        let p = |default| if self.preset != 0 { self.preset } else { default };
        DlssOptions {
            base: Base::new(DLSS_OPTIONS, 3),
            mode: mode_code(quality),
            output_width: w,
            output_height: h,
            sharpness: 0.0,
            pre_exposure: 1.0,
            exposure_scale: 1.0,
            color_buffers_hdr: if hdr { TRUE } else { FALSE },
            indicator_invert_axis_x: FALSE,
            indicator_invert_axis_y: FALSE,
            dlaa_preset: p(PRESET_K),
            quality_preset: p(PRESET_K),
            balanced_preset: p(PRESET_K),
            performance_preset: p(PRESET_M),
            ultra_performance_preset: p(PRESET_L),
            ultra_quality_preset: p(PRESET_K),
            use_auto_exposure: if hdr { FALSE } else { TRUE },
            alpha_upscaling_enabled: FALSE,
        }
    }

    /// `sl::DLSSDOptions` (Ray Reconstruction: HDR input always; normals and roughness packed).
    fn rr_options(&self, quality: TemporalQuality, (w, h): (u32, u32), c: &FrameConstants) -> DlssdOptions {
        DlssdOptions {
            base: Base::new(DLSSD_OPTIONS, DLSSD_OPTIONS_VERSION),
            mode: mode_code(quality),
            output_width: w,
            output_height: h,
            sharpness: 0.0,
            pre_exposure: c.pre_exposure,
            exposure_scale: 1.0,
            color_buffers_hdr: TRUE,
            indicator_invert_axis_x: FALSE,
            indicator_invert_axis_y: FALSE,
            normal_roughness_mode: NORMAL_ROUGHNESS_PACKED,
            world_to_camera_view: c.world_to_view,
            camera_view_to_world: c.view_to_world,
            alpha_upscaling_enabled: FALSE,
            dlaa_preset: self.preset,
            quality_preset: self.preset,
            balanced_preset: self.preset,
            performance_preset: self.preset,
            ultra_performance_preset: self.preset,
            ultra_quality_preset: self.preset,
        }
    }

    pub fn render_size(&mut self, quality: TemporalQuality, out: (u32, u32)) -> (u32, u32) {
        if quality == TemporalQuality::Native {
            return out;
        }
        if let Some((_, s)) = self.optimal.iter().find(|(k, _)| *k == (quality, out)) {
            return *s;
        }
        let (r, w, h) = match &self.feature {
            Feature::Sr { get_optimal_settings, .. } => {
                let options = self.options(quality, out, false);
                let mut settings = DlssOptimalSettingsS {
                    base: Base::new(DLSS_OPTIMAL_SETTINGS, 1),
                    s: DlssOptimalSettings::default(),
                };
                let r = unsafe { get_optimal_settings(&options, &mut settings) };
                (r, settings.s.optimal_render_width, settings.s.optimal_render_height)
            }
            Feature::Rr { get_optimal_settings, .. } => {
                let options = self.rr_options(quality, out, &FrameConstants::default());
                let mut settings = DlssdOptimalSettings {
                    base: Base::new(DLSSD_OPTIMAL_SETTINGS, DLSSD_OPTIMAL_SETTINGS_VERSION),
                    optimal_render_width: 0,
                    optimal_render_height: 0,
                    optimal_sharpness: 0.0,
                    render_width_min: 0,
                    render_height_min: 0,
                    render_width_max: 0,
                    render_height_max: 0,
                };
                let r = unsafe { get_optimal_settings(&options, &mut settings) };
                (r, settings.optimal_render_width, settings.optimal_render_height)
            }
        };
        let size = if r == 0 && w > 0 && h > 0 {
            (w.min(out.0), h.min(out.1))
        } else {
            log::warn!(
                "DLSS: Streamline gave no picture size for {quality:?} in a {}x{} window ({}); taking NVIDIA's ratio",
                out.0,
                out.1,
                result_name(r)
            );
            fallback_render_size(quality, out)
        };
        // (only this window's sizes are kept and told: a live resize asks for one at every step)
        if self.optimal.is_empty() {
            log::info!("DLSS {quality:?}: the {}x{} window is drawn at {}x{}", out.0, out.1, size.0, size.1);
        }
        self.optimal.retain(|((_, window), _)| *window == out);
        self.optimal.push(((quality, out), size));
        size
    }

    pub fn evaluate(&mut self, queue: &mx::Queue, quality: TemporalQuality, inputs: &Inputs, c: &FrameConstants) -> Result<()> {
        let feature_id = self.feature_id();
        if self.options_set != Some((quality, inputs.output_size, c.hdr)) || matches!(self.feature, Feature::Rr { .. }) {
            let changed = self.options_set.is_some_and(|(q, s, h)| (q, s, h) != (quality, inputs.output_size, c.hdr));
            // (a new quality or window size: resizing the proxy makes Streamline drop what it
            // made for the old one)
            if let (Backend::Dx12 { pacer: Some(p), .. }, true) = (&self.backend, changed) {
                let _ = unsafe { p.ResizeBuffers(0, inputs.output_size.0, inputs.output_size.1, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG(0)) };
            }
            // (on Vulkan no present of ours ever empties Streamline's queue of old contexts: without
            // this every window size kept its own, up to a full card)
            if matches!(self.backend, Backend::Vulkan { .. }) && self.allocated && changed {
                self.backend.wait();
                check("slFreeResources(DLSS)", unsafe { (self.session.free_resources)(feature_id, &self.viewport) })?;
                self.allocated = false;
            }
            match &self.feature {
                Feature::Sr { set_options, .. } => {
                    let options = self.options(quality, inputs.output_size, c.hdr);
                    check("slDLSSSetOptions", unsafe { set_options(&self.viewport, &options) })?;
                }
                Feature::Rr { set_options, .. } => {
                    // (every frame: the camera's matrices are in the options)
                    let options = self.rr_options(quality, inputs.output_size, c);
                    check("slDLSSDSetOptions", unsafe { set_options(&self.viewport, &options) })?;
                }
            }
            self.options_set = Some((quality, inputs.output_size, c.hdr));
        }
        let render = inputs.render_size;
        let output = inputs.output_size;
        let need = |t: Option<&mx::Texture>, what: &str| -> Result<mx::Texture> { t.cloned().ok_or_else(|| anyhow!("DLSS needs the {what}")) };
        let depth = need(inputs.depth, "depth")?;
        let motion = need(inputs.motion, "motion vectors")?;
        // (the picture's resource is plain UNORM and drawn through an sRGB view: DLSS takes its
        // sRGB-encoded numbers as they are, as FSR 3 does: read through an sRGB view it made
        // them linear and gave them back so, too dark)
        let mut list: Vec<(mx::Texture, u32, (u32, u32))> = vec![
            (inputs.color.clone(), BUFFER_SCALING_INPUT_COLOR, render),
            (depth, BUFFER_DEPTH, render),
            (motion, BUFFER_MOTION_VECTORS, render),
            (inputs.output.clone(), BUFFER_SCALING_OUTPUT_COLOR, output),
        ];
        if let Some(mask) = inputs.mask
            && matches!(self.feature, Feature::Sr { .. })
        {
            list.push((mask.clone(), BUFFER_BIAS_CURRENT_COLOR, render));
        }
        if let Some(g) = &inputs.guides {
            list.push((g.diffuse_albedo.clone(), BUFFER_ALBEDO, render));
            list.push((g.specular_albedo.clone(), BUFFER_SPECULAR_ALBEDO, render));
            list.push((g.normal_roughness.clone(), BUFFER_NORMAL_ROUGHNESS, render));
            if let Some(d) = g.specular_hit_distance {
                list.push((d.clone(), BUFFER_SPECULAR_HIT_DISTANCE, render));
            }
        }
        let mut resources: Vec<Resource> = Vec::with_capacity(list.len());
        for (t, _, size) in &list {
            resources.push(sl_resource(&self.session.device, t, None, t.format(), *size)?);
        }
        let tags: Vec<ResourceTag> = resources
            .iter_mut()
            .zip(&list)
            .map(|(r, (_, kind, (w, h)))| ResourceTag {
                base: Base::new(RESOURCE_TAG, 1),
                resource: r,
                kind: *kind,
                lifecycle: VALID_UNTIL_EVALUATE,
                extent: Extent {
                    top: 0,
                    left: 0,
                    width: *w,
                    height: *h,
                },
            })
            .collect();
        let token = self.session.token(1)?;
        let constants = sl_constants(c, render);
        let (list_ptr, dx12_list) = match &mut self.backend {
            Backend::Dx12 { ring, .. } => {
                let list = ring.begin()?;
                (list.as_raw(), Some(list))
            }
            Backend::Vulkan { ring, .. } => (vk::Handle::as_raw(ring.begin()?) as *mut c_void, None),
        };
        let session = &self.session;
        let viewport = &self.viewport;
        let recorded = (|| -> Result<()> {
            check("slSetConstants", unsafe { (session.set_constants)(&constants, token, viewport) })?;
            check("slSetTagForFrame", unsafe { (session.set_tag_for_frame)(token, viewport, tags.as_ptr(), tags.len() as u32, list_ptr) })?;
            let inputs: [*const Base; 1] = [&viewport.base];
            check("slEvaluateFeature(DLSS)", unsafe { (session.evaluate_feature)(feature_id, token, inputs.as_ptr(), 1, list_ptr) })?;
            Ok(())
        })();
        // (closed and submitted even when a call above failed: an empty list does no harm, and
        // the fence then still says when this slot is free)
        drop(dx12_list);
        match &mut self.backend {
            Backend::Dx12 { ring, .. } => {
                let raw_queue = queue.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 queue under mX's"))?.raw().clone();
                ring.submit(&raw_queue)?;
            }
            Backend::Vulkan { ring, .. } => crate::vulkan::with_queue(queue, |q| ring.submit(q))??,
        }
        recorded?;
        self.allocated = true;
        if let Backend::Dx12 { pacer: Some(p), .. } = &self.backend
            && !self.session.generating.load(std::sync::atomic::Ordering::Acquire)
        {
            let _ = unsafe { p.Present(0, DXGI_PRESENT(0)) };
        }
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.backend.wait();
        unsafe {
            if self.allocated {
                let _ = (self.session.free_resources)(self.feature_id(), &self.viewport);
            }
        }
        if let Backend::Dx12 { pacer, .. } = &mut self.backend {
            *pacer = None;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// DLSS Frame Generation

struct GenState {
    session: Arc<Session>,
    set_options: FnDlssgSetOptions,
    #[allow(dead_code)]
    get_state: FnDlssgGetState,
    reflex_sleep: Option<FnReflexSleep>,
    pcl_marker: Option<FnPclSetMarker>,
    viewport: ViewportHandle,
    generated: u32,
    /// What was last given to `slDLSSGSetOptions`.
    options: Option<(u32, (u32, u32))>,
    ring: Option<ListRing>,
    vk_ring: Option<VkRing>,
    /// Streamline made the window's swapchain.
    swapchain: bool,
    token: *const c_void,
    stopped: bool,
    presented: u64,
    show: bool,
}

// (Streamline's handles, used under the lock)
unsafe impl Send for GenState {}

/// DLSS-G on one window.
pub struct Generator {
    state: Arc<std::sync::Mutex<GenState>>,
}

impl Generator {
    pub fn new(device: &mx::Device, _queue: &mx::Queue, surface: &mx::Surface<'_>, generated: u32, options: &Options) -> Result<Generator> {
        let session = Session::get(device, options)?;
        if !session.has(FEATURE_DLSS_G) {
            bail!(
                "DLSS Frame Generation does not run here (it takes a GeForce RTX 40 or newer, hardware-accelerated GPU scheduling, an up-to-date driver and sl.dlss_g.dll with nvngx_dlssg.dll in {})",
                session.dir.display()
            );
        }
        if !session.has(FEATURE_REFLEX) {
            bail!("DLSS Frame Generation needs Reflex (sl.reflex.dll in {})", session.dir.display());
        }
        let set_options: FnDlssgSetOptions = session.function(FEATURE_DLSS_G, FN_DLSSG_SET_OPTIONS)?;
        let get_state: FnDlssgGetState = session.function(FEATURE_DLSS_G, FN_DLSSG_GET_STATE)?;
        let reflex_set: FnReflexSetOptions = session.function(FEATURE_REFLEX, FN_REFLEX_SET_OPTIONS)?;
        let reflex_sleep: Option<FnReflexSleep> = session.function(FEATURE_REFLEX, FN_REFLEX_SLEEP).ok();
        let pcl_marker: Option<FnPclSetMarker> = if session.has(FEATURE_PCL) { session.function(FEATURE_PCL, FN_PCL_SET_MARKER).ok() } else { None };
        let reflex = ReflexOptions {
            base: Base::new(REFLEX_OPTIONS, REFLEX_OPTIONS_VERSION),
            mode: REFLEX_MODE_LOW_LATENCY,
            frame_limit_us: 0,
            use_markers_to_optimize: false,
            virtual_key: 0,
            id_thread: 0,
        };
        check("slReflexSetOptions", unsafe { reflex_set(&reflex) })?;
        let viewport = ViewportHandle {
            base: Base::new(VIEWPORT_HANDLE, 1),
            value: 0,
        };
        let mut st = DlssgState {
            base: Base::new(DLSSG_STATE, DLSSG_STATE_VERSION),
            estimated_vram_usage_in_bytes: 0,
            status: 0,
            min_width_or_height: 0,
            num_frames_actually_presented: 0,
            num_frames_to_generate_max: 1,
            b_reserved4: 0,
            b_is_vsync_support_available: 0,
            inputs_processing_completion_fence: std::ptr::null_mut(),
            last_present_inputs_processing_completion_fence_value: 0,
            b_is_dynamic_mfg_supported: 0,
        };
        let _ = unsafe { get_state(&viewport, &mut st, std::ptr::null()) };
        let max = st.num_frames_to_generate_max.max(1);
        let generated = generated.clamp(1, max);
        let (ring, vk_ring) = match &session.device {
            Device::Dx12(d) => (Some(ListRing::new(d).context("DLSS-G's command lists")?), None),
            Device::Vulkan(g) => (None, Some(VkRing::new(g).context("DLSS-G's command buffers")?)),
        };
        let state = Arc::new(std::sync::Mutex::new(GenState {
            session: session.clone(),
            set_options,
            get_state,
            reflex_sleep,
            pcl_marker,
            viewport,
            generated,
            options: None,
            ring,
            vk_ring,
            swapchain: false,
            token: std::ptr::null(),
            stopped: false,
            presented: 0,
            show: options.show != Show::Both,
        }));
        let installed = match &session.device {
            Device::Dx12(_) => {
                let st = state.clone();
                surface.with_dx12(move |s| s.set_swapchain_maker(Some(Arc::new(move |req: &mx::mx_dx12::SwapchainRequest<'_>| make_dx12(&st, req)))))
            }
            Device::Vulkan(_) => {
                let interposer = session.module;
                let get: vk::PFN_vkGetDeviceProcAddr = load_fn(interposer, c"vkGetDeviceProcAddr")?;
                let st = state.clone();
                surface.with_vulkan(move |s| s.set_swapchain_maker(Some(Arc::new(move |dev, info: &vk::SwapchainCreateInfoKHR<'_>| make_vulkan(&st, get, dev, info)))))
            }
        };
        if installed.is_none() {
            bail!("DLSS Frame Generation needs the window on the device's graphics API (configure the surface first)");
        }
        session.generating.store(true, std::sync::atomic::Ordering::Release);
        log::info!("DLSS Frame Generation: {}x (this card makes up to {max} pictures between two)", generated + 1);
        Ok(Generator { state })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GenState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn prepare(&self, queue: &mx::Queue, inputs: &FrameInputs) -> Result<()> {
        let mut st = self.lock();
        if st.stopped || !st.swapchain {
            return Ok(());
        }
        let result = (|| -> Result<()> {
            let token = st.session.token(2)?;
            st.token = token;
            if let Some(sleep) = st.reflex_sleep {
                let _ = unsafe { sleep(token) };
            }
            let device = &st.session.device;
            let c = &inputs.constants;
            let render = inputs.render_size;
            let output = inputs.output_size;
            let mut list: Vec<(mx::Texture, u32, (u32, u32))> = vec![
                (inputs.depth.clone(), BUFFER_DEPTH, render),
                (inputs.motion.clone(), BUFFER_MOTION_VECTORS, render),
                (inputs.hudless.clone(), BUFFER_HUD_LESS_COLOR, output),
            ];
            if let Some(ui) = &inputs.ui {
                list.push((ui.clone(), BUFFER_UI_COLOR_AND_ALPHA, output));
            }
            let mut resources: Vec<Resource> = Vec::with_capacity(list.len());
            for (t, _, size) in &list {
                resources.push(sl_resource(device, t, None, t.format(), *size)?);
            }
            let tags: Vec<ResourceTag> = resources
                .iter_mut()
                .zip(&list)
                .map(|(r, (_, kind, (w, h)))| ResourceTag {
                    base: Base::new(RESOURCE_TAG, 1),
                    resource: r,
                    kind: *kind,
                    // (copied by Streamline on this command list: the textures are mX's, and may be
                    // drawn into again before the present)
                    lifecycle: ONLY_VALID_NOW,
                    extent: Extent {
                        top: 0,
                        left: 0,
                        width: *w,
                        height: *h,
                    },
                })
                .collect();
            if st.options != Some((st.generated, output)) {
                let mut o = DlssgOptions::new();
                o.mode = DLSSG_MODE_ON;
                o.num_frames_to_generate = st.generated;
                o.flags = DLSSG_FLAG_RETAIN_RESOURCES_WHEN_OFF | if st.show { DLSSG_FLAG_SHOW_ONLY_INTERPOLATED_FRAME } else { 0 };
                o.mvec_depth_width = render.0;
                o.mvec_depth_height = render.1;
                o.color_width = output.0;
                o.color_height = output.1;
                check("slDLSSGSetOptions", unsafe { (st.set_options)(&st.viewport, &o) })?;
                st.options = Some((st.generated, output));
            }
            let constants = sl_constants(c, render);
            let session = st.session.clone();
            let viewport = ViewportHandle {
                base: Base::new(VIEWPORT_HANDLE, 1),
                value: st.viewport.value,
            };
            let marker = st.pcl_marker;
            if let Some(m) = marker {
                let _ = unsafe { m(PCL_SIMULATION_START, token) };
                let _ = unsafe { m(PCL_SIMULATION_END, token) };
                let _ = unsafe { m(PCL_RENDER_SUBMIT_START, token) };
            }
            check("slSetConstants", unsafe { (session.set_constants)(&constants, token, &viewport) })?;
            let record = |list_ptr: *mut c_void| -> Result<()> { check("slSetTagForFrame", unsafe { (session.set_tag_for_frame)(token, &viewport, tags.as_ptr(), tags.len() as u32, list_ptr) }) };
            if let Some(ring) = st.ring.as_mut() {
                let raw_queue = queue.as_dx12().ok_or_else(|| anyhow!("no Direct3D 12 queue under mX's"))?.raw().clone();
                let list = ring.begin()?;
                let recorded = record(list.as_raw());
                drop(list);
                ring.submit(&raw_queue)?;
                recorded?;
            } else if let Some(ring) = st.vk_ring.as_mut() {
                let buffer = ring.begin()?;
                let recorded = record(vk::Handle::as_raw(buffer) as *mut c_void);
                crate::vulkan::with_queue(queue, |q| ring.submit(q))??;
                recorded?;
            }
            if let Some(m) = marker {
                let _ = unsafe { m(PCL_RENDER_SUBMIT_END, token) };
            }
            Ok(())
        })();
        if let Err(e) = &result {
            log::error!("DLSS Frame Generation stopped working ({e:#}); frames are shown without it from now on");
            st.stopped = true;
            let mut o = DlssgOptions::new();
            o.mode = DLSSG_MODE_OFF;
            let _ = unsafe { (st.set_options)(&st.viewport, &o) };
            st.options = None;
        }
        result
    }

    /// Streamline's markers around the window's present.
    pub fn presenting(&self, start: bool) {
        let mut st = self.lock();
        if let (Some(m), false) = (st.pcl_marker, st.token.is_null()) {
            let _ = unsafe { m(if start { PCL_PRESENT_START } else { PCL_PRESENT_END }, st.token) };
        }
        if !start {
            st.presented += 1;
        }
    }

    pub fn skip(&self) {
        let mut st = self.lock();
        if st.options.is_some() {
            let mut o = DlssgOptions::new();
            o.mode = DLSSG_MODE_OFF;
            o.flags = DLSSG_FLAG_RETAIN_RESOURCES_WHEN_OFF;
            let _ = unsafe { (st.set_options)(&st.viewport, &o) };
            st.options = None;
        }
    }

    pub fn working(&self) -> bool {
        !self.lock().stopped
    }

    pub fn generated(&self) -> u64 {
        let st = self.lock();
        st.presented * st.generated as u64
    }

    pub fn wait(&self) {
        let mut st = self.lock();
        if let Some(r) = st.ring.as_mut() {
            r.wait();
        }
        if let Some(r) = st.vk_ring.as_mut() {
            r.wait();
        }
    }

    pub fn stop(self, surface: &mx::Surface<'_>) {
        self.wait();
        {
            let mut st = self.lock();
            let mut o = DlssgOptions::new();
            o.mode = DLSSG_MODE_OFF;
            let _ = unsafe { (st.set_options)(&st.viewport, &o) };
            let _ = unsafe { (st.session.free_resources)(FEATURE_DLSS_G, &st.viewport) };
            st.session.generating.store(false, std::sync::atomic::Ordering::Release);
            st.swapchain = false;
            st.stopped = true;
        }
        // (the window's own swapchain again, made the next time the surface is configured)
        let _ = surface.with_dx12(|s| s.set_swapchain_maker(None));
        let _ = surface.with_vulkan(|s| s.set_swapchain_maker(None));
    }
}

/// The surface's swapchain, made through Streamline's factory proxy: a DLSS-G swapchain that
/// makes the pictures in its present (Direct3D 12).
fn make_dx12(state: &Arc<std::sync::Mutex<GenState>>, req: &mx::mx_dx12::SwapchainRequest<'_>) -> windows::core::Result<IDXGISwapChain1> {
    let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
    let own = || unsafe { req.factory.CreateSwapChainForHwnd(req.queue, req.hwnd, req.desc, None, None) };
    let Some(upgrade) = st.session.upgrade_interface else {
        st.stopped = true;
        return own();
    };
    // (Streamline paces the presents itself: no frame latency object)
    let mut desc = *req.desc;
    desc.Flags &= !(windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0 as u32);
    let mut factory: *mut c_void = req.factory.as_raw();
    // (an AddRef'd copy, upgraded in place to Streamline's proxy)
    let _ = std::mem::ManuallyDrop::new(req.factory.clone());
    let r = unsafe { upgrade(&mut factory) };
    if r != 0 || factory.is_null() {
        log::error!("DLSS Frame Generation: no factory proxy from Streamline ({}); the window's own swapchain is used", result_name(r));
        st.stopped = true;
        return own();
    }
    // SAFETY: the proxy's reference (ours, from the clone above).
    let proxy = unsafe { windows::Win32::Graphics::Dxgi::IDXGIFactory2::from_raw(factory) };
    let swap = unsafe { proxy.CreateSwapChainForHwnd(req.queue, req.hwnd, &desc, None, None) }?;
    // (upgraded right after it is made, so that Streamline's present runs every frame)
    let mut raw = swap.as_raw();
    let _ = std::mem::ManuallyDrop::new(swap.clone());
    let r = unsafe { upgrade(&mut raw) };
    let swap = if r == 0 && !raw.is_null() {
        // SAFETY: the proxy's reference (the clone above).
        unsafe { IDXGISwapChain1::from_raw(raw) }
    } else {
        swap
    };
    st.swapchain = true;
    st.stopped = false;
    st.options = None;
    Ok(swap)
}

/// The surface's swapchain, made with Streamline's own Vulkan functions (sl.interposer's
/// `vkGetDeviceProcAddr`).
fn make_vulkan(
    state: &Arc<std::sync::Mutex<GenState>>,
    get: vk::PFN_vkGetDeviceProcAddr,
    device: vk::Device,
    info: &vk::SwapchainCreateInfoKHR<'_>,
) -> std::result::Result<(vk::SwapchainKHR, mx::mx_vulkan::SwapchainFns), vk::Result> {
    let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: Streamline's functions by their Vulkan names.
    let f = |name: &CStr| unsafe { get(device, name.as_ptr()).ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT) };
    let create: vk::PFN_vkCreateSwapchainKHR = unsafe { std::mem::transmute(f(c"vkCreateSwapchainKHR")?) };
    // (each table field is the type the transmute makes)
    #[allow(clippy::missing_transmute_annotations)]
    let fns = mx::mx_vulkan::SwapchainFns {
        destroy: unsafe { std::mem::transmute(f(c"vkDestroySwapchainKHR")?) },
        get_images: unsafe { std::mem::transmute(f(c"vkGetSwapchainImagesKHR")?) },
        acquire: unsafe { std::mem::transmute(f(c"vkAcquireNextImageKHR")?) },
        present: unsafe { std::mem::transmute(f(c"vkQueuePresentKHR")?) },
    };
    let mut sc = vk::SwapchainKHR::null();
    // SAFETY: the surface's own description.
    unsafe { create(device, info, std::ptr::null(), &mut sc) }.result()?;
    st.swapchain = true;
    st.stopped = false;
    st.options = None;
    Ok((sc, fns))
}

impl Drop for Generator {
    fn drop(&mut self) {
        let st = self.lock();
        st.session.generating.store(false, std::sync::atomic::Ordering::Release);
    }
}
