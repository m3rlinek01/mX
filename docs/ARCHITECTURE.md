# Architecture

mX exists so that openOMSI owns the whole path from its renderer to the driver. It replaced
wgpu under openOMSI's renderer without changing the picture, and it changes how resources
reach shaders: through one bindless heap instead of bind groups.

## Goals

1. The same picture as wgpu. Tests render the same scenes with both and compare bytes.
2. Bindless. Shaders index one global heap of textures and samplers; a draw is a pipeline,
   some buffers and a few bytes of immediates.
3. Thin. One layer between the API and Metal, Vulkan or Direct3D 12, and the native handles
   are public for interop with MetalFX, DLSS, FSR 3 and OpenXR.
4. Familiar. Names and descriptor shapes follow wgpu where the concept is the same.

WGSL stays the shader language and naga compiles it. naga is a standalone compiler from the
wgpu project; mX uses it and nothing else from wgpu at run time. A second WGSL compiler would
be years of work and a second source of shading differences.

## Crates and backend selection

`mx` re-exports one backend, chosen at compile time:

| Target | Backend |
|--------|---------|
| Apple platforms | `mx-metal` (Metal 3: macOS 13+, iOS 16+) |
| Windows | `mx-dx12` (feature level 11.0, resource binding tier 2) |
| Linux, Android and others | `mx-vulkan` (Vulkan 1.3, or 1.2 with dynamic rendering and synchronization2) |
| Any target with the `vulkan` feature | `mx-vulkan` (on macOS through MoltenVK) |

There is no dispatch layer: `mx::Device` is the backend's `Device`. Every backend implements
the same items with the same signatures ([BACKEND_API.md](BACKEND_API.md)), and the conformance
tests hold them to the same behaviour.

`mx-types` holds the plain data the backends share. `mx-shader` turns WGSL into each backend's
language with naga and reflects what an entry point uses. `wgpu-mx` implements wgpu 29's API on
top of `mx`.

## Binding model

There are no bind group layouts or pipeline layouts. A backend reflects what an entry point
uses from the naga module and places each WGSL `(group, binding)` where its API wants it.

- Group 7 is the heap, declared by the prelude that mX appends to every module
  ([GUIDE.md](GUIDE.md) lists the arrays).
- Groups 0 to 6 are root groups: uniform and storage buffers, storage textures, textures,
  samplers and acceleration structures, set with layout-free bind groups.
- `var<immediate>` holds per-draw bytes.

The heap is group 7 rather than 0 so that openOMSI's shaders, which use groups 0 to 3, compile
unchanged.

A texture view gets one index and that index is valid in every typed array the view can be
read through. Freed indices are reused only after every submission that could read them has
completed.

### Metal

Two `MTLBuffer`s of `MTLResourceID`s (8 bytes per slot), one for textures and one for
samplers. naga writes unbounded binding arrays as argument buffer pointers, so the texture
buffer is bound at all six texture slots.

| Buffer slot | Contents |
|-------------|----------|
| 0 to 5 | texture heap (the same buffer at each typed array's slot) |
| 6 and 7 | sampler heap |
| 8 | immediates (`setVertexBytes`, `setFragmentBytes`) |
| 9 | sizes of runtime-sized storage buffers, then of vertex buffers |
| 10 to 22 | root buffers in `(group, binding)` order |
| 23 to 30 | vertex buffers, slot `30 - index` |

Root textures and samplers take texture and sampler slots from 0 in `(group, binding)` order.
Each entry point gets its own assignment and counts only the root bindings it uses, so a vertex
shader and a fragment shader from different modules work together.

Vertex buffers are read by the vertex shader itself (naga's vertex pulling, as in wgpu-hal),
because Metal's fixed-function vertex fetch converts normalised formats one ulp differently.

### Vulkan

Descriptor set 0 is the heap: one `UPDATE_AFTER_BIND | PARTIALLY_BOUND` array per prelude
binding. A view is written at its index into every binding its dimension and sample type
allow, so a 2D depth view goes into `mx_tex2d` and `mx_texdepth`. Root group `g` is descriptor
set `g + 1`, with layouts built from what the pipeline's entry points use, sets allocated per
encoder and cached per bind group. Immediates are push constants.

Above `maxPerSetDescriptors` a layout is valid only where `vkGetDescriptorSetLayoutSupport`
accepts it, so mX halves the heap until the driver does. MoltenVK answers by its
`maxPerSetDescriptors` of 1212 alone although its argument buffers handle much larger arrays,
so mX keeps the requested size there and says so in the log.

### Direct3D 12

One shader-visible CBV/SRV/UAV heap whose first `T` descriptors are the texture slots, and a
shader-visible sampler heap of 2048. Every root signature has a table over the texture slots
with one SRV range per typed array, all starting at descriptor 0 so the indices are shared.
The ranges are `T` long because naga's HLSL writer needs a size for a binding array. Samplers
go through naga's sampler heap and per-group sampler index buffers, the way wgpu-hal binds
them. Root group `g` is a descriptor table whose descriptors are copied per draw into a ring
in the same heap; immediates are root constants. `mx_shader::hlsl` has the register layout.

## Shader compilation

Shaders are compiled the way wgpu-hal compiles them for the same API, so the same WGSL runs
the same code:

- naga's bounds-check policies as wgpu-hal sets them (`Restrict` on Metal),
- zeroed workgroup memory and bounded loops,
- on Metal, the newest MSL version the OS has (3.0 on macOS 13, 3.1 on 14, 3.2 from 15),
  `preserveInvariance`, and Metal's default fast math,
- on Direct3D 12, DXC with wgpu-hal's arguments, and FXC (shader model 5.1) when
  `dxcompiler.dll` is missing.

The parity tests found three differences from wgpu early on (bounds checks, the MSL version
and vertex fetch); all three are fixed and the tests guard against new ones.

## Synchronisation

mX keeps wgpu's guarantees: queue writes land before the next submission's commands, a pass
sees what earlier passes wrote, and no barriers are written by hand.

On Metal, hazards are tracked by Metal for the resources an encoder uses or declares. Textures
read through the heap are declared with `useResources`; see Residency below for which ones.

On Vulkan every image stays in `GENERAL` layout (swapchain images move to `PRESENT_SRC` to be
shown), so the heap's descriptors stay valid whatever a pass does. One full memory barrier
orders each pass or copy after the work before it, and the dispatches of a compute pass after
each other. Narrower barriers would be faster and are left for later.

On Direct3D 12 every resource has a home state: `PIXEL_SHADER_RESOURCE |
NON_PIXEL_SHADER_RESOURCE` for textures with `TEXTURE_BINDING`, so the heap can sample them in
any pass, and `COMMON` otherwise. A per-subresource tracker moves what a pass, copy, draw or
dispatch uses into the state it needs and returns textures home afterwards, so command lists
can be submitted in any order. A UAV barrier orders storage writes before the next draw or
dispatch.

### Zero initialisation

Textures are zeroed before the first submission after their creation, except for the mips
and layers that a queue write fills completely (`mx_types::Unwritten` tracks them). Render
targets and depth are cleared by a render pass, other textures by a copy from a buffer of
zeros. Vulkan fills new buffers with zeros. Metal's new buffers and Direct3D 12's committed
resources are zeroed by the system.

## Residency and staging

On macOS 15 and later the heap's views live in an `MTLResidencySet` that the queue keeps
resident. Encoders then declare with `useResources` only the textures that Metal's hazard
tracking must know about: render targets, storage textures and textures made outside mX at all
times; a texture written by a copy, a clear, a mipmap generation or zero initialisation until
the submission that wrote it has finished; and the textures the current command buffer copied
into. Recording 40 passes with 20000 textures in the heap takes about 0.5 ms this way and
112 ms when every encoder declares the whole heap, which is what happens on macOS 13 and 14
or with `MX_NO_RESIDENCY_SET` set.

Queue writes go through a staging ring on every backend: 4 MiB blocks of host-visible memory,
filled one after another and reused once the submission that read them is done. Writes larger
than 1 MiB get a buffer of their own.

## Known platform behaviour

When several Metal devices in one process sample GPU counters at the same time, the M4
sometimes resolves both timestamps of a pass as 0. wgpu-hal samples the same way and has the
same problem. A game opens one device; the conformance tests take the GPU one at a time.

OpenGL is not a backend, because the hardware that still needs it has no bindless that works.
