# Status

What is implemented, and how far each part has been tested. Tested here means on an Apple M4
unless the row says otherwise.

## Backends

| Area | Metal | Vulkan | Direct3D 12 |
|------|-------|--------|-------------|
| Buffers, textures, views, samplers, the heap | tested | tested on MoltenVK | compiled |
| Render and compute pipelines, root bind groups, immediates, copies | tested | tested on MoltenVK | compiled |
| MSAA resolve, render bundles, indirect and multi-draw indirect | tested | tested on MoltenVK | compiled |
| Timestamp and occlusion queries | tested | tested on MoltenVK | compiled |
| Surfaces and swapchains | tested | tested on MoltenVK (Win32, Xlib, XCB, Wayland and Android surfaces compiled) | compiled (DXGI flip model) |
| Zero initialisation, `clear_texture` | tested | tested on MoltenVK | compiled (`clear_texture`; committed resources start zeroed) |
| Workgroup memory | tested | tested on MoltenVK | compiled |
| Staging ring for queue writes | tested | tested on MoltenVK | compiled |
| Ray queries, refits | tested | compiled (MoltenVK has no ray queries) | compiled (DXR 1.1) |
| Native handles, `texture_from_raw` | tested (MetalFX in openOMSI) | compiled | compiled |
| Residency set (macOS 15+) | tested | | |

"Compiled" means built and linted in CI but never run on a GPU.

## Test suites

| Suite | Result |
|-------|--------|
| Conformance tests, 22 (`crates/mx/tests/conformance.rs`) | pass on Metal, with and without the residency set, under Metal's API and shader validation; pass on MoltenVK |
| Pixel parity with wgpu, 7 scenes and 21 of openOMSI's full-screen passes | identical bytes on Metal and on MoltenVK |
| openOMSI's 27 WGSL modules | compile for every backend; the HLSL has not been through DXC yet |
| `wgpu-mx` through wgpu's API (`crates/wgpu-mx/tests`) | pass on Metal and MoltenVK |
| Khronos validation layer with synchronization validation (MoltenVK) | no synchronization findings; two known reports, below |

## openOMSI on mX

openOMSI builds on `wgpu-mx` with two changes: its `wgpu` dependency points at `wgpu-mx`, and
MetalFX takes mX's native handles instead of wgpu-hal's. Grundorf runs clean under Metal's
validation layer with Enhanced graphics and with ray tracing. With Enhanced graphics it runs
at 104 fps against 82 fps for wgpu ([README](../README.md#performance) has the full numbers).

## Known reports from the Vulkan validation layer

- `VUID-vkCreateDescriptorSetLayout-support-09582` on MoltenVK only. MoltenVK rejects the heap
  layout because of its `maxPerSetDescriptors` of 1212, yet handles it. mX keeps the requested
  heap size on MoltenVK and shrinks it on drivers that actually reject the layout.
- `VUID-StandaloneSpirv-None-10684`. naga's SPIR-V output puts `ArrayStride` on arrays in
  function memory, for wgpu as well as for mX. Drivers accept it.

## Not done yet

- Running Vulkan on Linux and Android drivers.
- Running Direct3D 12 on Windows hardware, and compiling the generated HLSL with DXC.
- Comparing whole openOMSI frames (scene, sky, weather, ray tracing) with a wgpu build of the
  same source. The full-screen passes are compared; the 3D passes need real scenes.
