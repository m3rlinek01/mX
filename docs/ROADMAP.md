# Roadmap

Where mX goes from here, decided on 2026-11-17. The current state is in [STATUS.md](STATUS.md).

## What is wrong today

- **openOMSI does not use mX's own API.** It runs through `wgpu-mx`, which implements wgpu 29's
  API on mX. The game's code still has 26 places that create bind group layouts, 64 that create
  bind groups and 72 that set them, and mX turns all of them into "root groups". The heap is there, but the game
  does not use it, so most of the cost and most of wgpu's quirks stay.
- **Not fully bindless.** Buffers never went through the heap: uniforms and storage buffers
  were root groups 0 to 6, and on Direct3D 12 their descriptors are copied into a ring on every
  draw. (Phase 1, step 2 adds the buffer heap.)
- **Little FPS gain.** On the M4 the main pass takes 5.1 ms with mX and with wgpu, because the
  game draws exactly the same way. The gain (82 → 104 fps) comes from CPU recording alone.
- **Synchronisation is coarse.** Vulkan puts one full memory barrier before every pass and copy.
  Metal declares textures to its hazard tracker instead of using fences.
- **Effects differ from wgpu.** Rain and other weather effects look wrong in game. Only
  full-screen passes are compared with wgpu; the 3D and weather passes are not.
- **Upscalers.** Only MetalFX upscaling works under mX. DLSS (Streamline), FSR 3 with frame
  generation and MetalFX frame interpolation reach wgpu-hal through `as_hal` and are switched
  off on mX. XeSS is missing in both.
- **No legacy APIs.** openOMSI on wgpu also runs on OpenGL (and ANGLE over Direct3D 11 on
  Windows); mX has nothing for Metal 2, Direct3D 11 or OpenGL.
- **Untested backends.** Direct3D 12 has never run on a GPU, Vulkan only on MoltenVK.

## Phase 1: native and fast

Written on 2026-11-17; built and linted for every backend, unit tests run, the GPU tests not
run yet (they are the testing phase's first job, see [STATUS.md](STATUS.md)).

1. **Baseline.** `cargo run --release -p mx --example bench` draws the same 20000 materials
   three ways (a bind group per draw, heap indices as immediates, one GPU-driven batch) and
   prints CPU record time and GPU pass time per way, with a JSON line per way for comparing
   builds and backends. The whole-game baseline (Grundorf with weather, wgpu build against mX
   build) belongs to phase 3, where the frames are compared anyway.
2. **Buffers in the heap.** `Buffer::create_view` gives a buffer range a heap index; WGSL
   reads it through `mx_buffers[i]`, `mx_rw_buffers[i]` or its own typed views (bindings 16 to
   63 of group 7). Metal: an argument buffer of GPU addresses, slots assigned per entry point.
   Vulkan: one descriptor-indexed array of storage buffers that every view aliases. Direct3D 12:
   raw SRVs and UAVs in the CBV/SRV/UAV heap, with naga's HLSL writer extended for binding arrays
   of buffers (`vendor/naga`). Root groups stay only for `wgpu-mx`; native code needs none, and
   Direct3D 12 copies no descriptors per draw for it.
3. **Precise synchronisation.**
   - Vulkan: the full barrier before every command is gone. Each command declares its stages
     and accesses, and the barrier before it waits only for what ran and makes visible only what
     was written and is not visible yet (see ARCHITECTURE, Synchronisation).
   - Direct3D 12: enhanced barriers where the driver has them, legacy transitions otherwise.
   - Metal: kept on Metal's hazard tracking, deliberately. Untracked resources and `MTLFence`s
     were planned, but the residency set already removed the cost they were meant to remove
     (declaring the heap per encoder), Metal tracks per resource and so only orders encoders
     that really conflict (one fence chain would order all of them), and MetalFX encodes into
     mX's command buffers without waiting on mX's fences, so untracked textures would race
     with it.
4. **GPU-driven drawing.** `IndirectDraws`: batches whose arguments and count the GPU writes,
   executed natively (Metal indirect command buffers encoded by a kernel, Vulkan
   `vkCmdDraw[Indexed]IndirectCount`, Direct3D 12 `ExecuteIndirect` with a count buffer). The
   culling dispatches that fill them are the renderer's, in phase 2.
5. **Direct3D 12 shader model 6.6: not needed.** mX's heap on Direct3D 12 is already bindless
   the shader model 5.1 way, one descriptor table over the whole heap, which every resource
   binding tier 2 device runs; `ResourceDescriptorHeap` would remove only the table's root
   parameter, for a large change to naga's HLSL writer.

Moved to phase 2: an async compute queue. Which passes run on it, and where they wait for the
graphics queue, are the renderer's decisions; mX gets the second queue when the ported renderer
has passes to put there.

## Order

On 2026-11-17 phase 5 moved before phase 2: openOMSI on wgpu runs on OpenGL (ANGLE on Windows,
GL on Linux), and porting the renderer to mX's API before mX has GL would leave those players
without the game. [BOUND_TIER.md](BOUND_TIER.md) has its design and status.

## Phase 2: openOMSI's renderer on mX

Port `omsi-render` (68 files using `wgpu::`) and the other crates that draw (`omsi-app`,
`omsi-navigator`, `omsi-scene`, `omsi-ui`, `omsi-plugin-ui`, `omsi-xr`) from wgpu's API to mX's,
one pass at a time. Materials become heap indices; bind group layouts go away. Every ported
pass gets a pixel test against the wgpu build before the next one starts. `wgpu-mx` stays as a
fallback until the port is complete.

## Phase 3: every effect identical

Whole-frame comparisons with the wgpu build in fixed scenes: rain, snow, puddles and their
reflections, fog and lamps, night, ray tracing, cockpit and exterior. Each difference gets a
cause and a regression test. Rain is first, because it is visibly broken now.

## Phase 4: upscalers and frame generation in mX

A `mx-upscale` crate that works on mX's own objects, without `as_hal`:

| Technology | Backends |
|------------|----------|
| MetalFX spatial and temporal upscaling, frame interpolation | Metal |
| DLSS Super Resolution, Ray Reconstruction, Frame Generation (Streamline) | Direct3D 12, Vulkan |
| FSR 3.1 upscaling and frame generation (FidelityFX SDK) | Direct3D 12, Vulkan |
| XeSS upscaling (and frame generation on Direct3D 12) | Direct3D 12, Vulkan |

FSR 1, FXAA and the other shader-only filters stay in the renderer; they need no backend.

## Phase 5: the bound tier (started, see BOUND_TIER.md)

Backends for hardware without working bindless: Metal 2, Direct3D 11, and OpenGL 3.3, 4.3 and
ES 3.0 (with ANGLE on Windows). The shaders stay the same WGSL. Where a shader indexes the heap
with a per-draw immediate, the backend reads the immediates at draw time, binds those views to
fixed slots and rewrites the access to that slot. Indexing the heap with values read from
buffers needs real bindless, so passes that do it get a simpler fallback on this tier.

## What can be tested where

This Mac runs Metal and Vulkan through MoltenVK. Direct3D 12, Direct3D 11, OpenGL on Windows and
Linux drivers, DLSS, FSR 3 and XeSS need Windows or Linux hardware; until that exists they are
compiled in CI and tested by whoever has the hardware.
