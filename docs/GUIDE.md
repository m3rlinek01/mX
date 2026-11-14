# Using mX

This guide covers the parts of mX that differ from wgpu. Names and descriptor shapes follow
wgpu wherever the concept is the same, so `TextureDescriptor`, `RenderPassDescriptor`,
`queue.write_buffer` and the rest work as you would expect from wgpu.

## Device

```rust
let instance = mx::Instance::new();
let adapter = instance.default_adapter()?;            // a discrete GPU first, then an integrated one
let (device, queue) = adapter.request_device(&mx::DeviceDescriptor {
    label: Some("game"),
    bindless_textures: 1 << 16,                       // the defaults
    bindless_samplers: 2048,
})?;
```

`bindless_textures` and `bindless_samplers` size the heap. mX lowers them to what the device
supports; `device.limits()` reports the sizes it got.

`Instance::enumerate_adapters` lists only adapters that can run mX. A Metal device needs
Metal 3 and Tier 2 argument buffers; a Vulkan device needs descriptor indexing, dynamic
rendering and synchronization2.

## The heap

Every texture view created from a texture with `TextureUsages::TEXTURE_BINDING` receives an
index, and so does every sampler:

```rust
let view = texture.create_view(&Default::default())?;
let sampler = device.create_sampler(&mx::SamplerDescriptor::default())?;
let (t, s) = (view.index(), sampler.index());
```

Shaders read the heap through arrays that mX declares for them, in group 7:

| Array | Type |
|-------|------|
| `mx_tex2d` | `binding_array<texture_2d<f32>>` |
| `mx_tex2d_array` | `binding_array<texture_2d_array<f32>>` |
| `mx_texcube` | `binding_array<texture_cube<f32>>` |
| `mx_tex3d` | `binding_array<texture_3d<f32>>` |
| `mx_texdepth` | `binding_array<texture_depth_2d>` |
| `mx_tex2d_u32` | `binding_array<texture_2d<u32>>` |
| `mx_samplers` | `binding_array<sampler>` |
| `mx_samplers_cmp` | `binding_array<sampler_comparison>` |

A texture index is valid in the array that matches the view's dimension and sample type: a 2D
view is `mx_tex2d[i]`, a cube view `mx_texcube[i]`, a 2D depth view both `mx_tex2d[i]` and
`mx_texdepth[i]`. Samplers have their own index space, and comparison samplers are read
through `mx_samplers_cmp`.

When a view or sampler is dropped, its index returns to the heap only after every submission
that could still read it has finished on the GPU. Holding on to an index after dropping the
view is a bug in the caller; mX does not detect it.

## Immediates

Small per-draw data, typically heap indices and a few constants, goes in immediates
(push constants on Vulkan, root constants on Direct3D 12):

```wgsl
struct Draw { albedo: u32, normal: u32, smp: u32, tint: f32 }
var<immediate> draw: Draw;
```

```rust
pass.set_immediates(0, bytemuck::bytes_of(&draw));
```

`device.limits().max_immediate_size` gives the size available: 4096 bytes on Metal, the
device's push constant limit on Vulkan (at least 128 bytes).

## Bind groups

Groups 0 to 6 work like wgpu's bind groups, without layouts. A pipeline reflects what its
shaders use, so a bind group is just its entries:

```rust
let group = device.create_bind_group(&mx::BindGroupDescriptor {
    label: Some("frame"),
    entries: &[mx::BindGroupEntry {
        binding: 0,
        resource: mx::BindingResource::Buffer(mx::BufferBinding { buffer: &uniforms, offset: 0, size: None }),
    }],
});
pass.set_bind_group(0, &group);
```

Use them for uniform and storage buffers, storage textures and acceleration structures, and
for shaders that have not moved to the heap yet. Textures and samplers can be bound here too,
which is how wgpu code runs unchanged on `wgpu-mx`.

## Pipelines and shaders

```rust
let module = device.create_shader_module(&mx::ShaderModuleDescriptor { label: Some("mesh"), source: WGSL })?;
let pipeline = device.create_render_pipeline(&mx::RenderPipelineDescriptor {
    label: Some("mesh"),
    vertex: mx::VertexState { module: &module, entry_point: None, constants: &[], buffers: &[layout] },
    primitive: Default::default(),
    depth_stencil: None,
    multisample: Default::default(),
    fragment: Some(mx::FragmentState { module: &module, entry_point: None, constants: &[], targets: &[Some(format.into())] }),
})?;
```

There is no `layout` field. `constants` sets WGSL `override` values. `device.precompile(&module)`
compiles every entry point at load time, so shader errors and the driver's compile time show
up there instead of at the first draw.

mX appends its prelude after your source, so line numbers in error messages are your own.
Shader errors come back as `Error::Shader` with naga's message.

## Synchronisation

You do not write barriers. mX keeps wgpu's rules:

- `queue.write_buffer` and `queue.write_texture` take effect before the commands of the next
  `submit`.
- A pass sees everything that earlier passes and copies wrote, including writes to textures
  it reads through the heap.
- A new texture or buffer reads as zeros until something writes it.

`device.poll()` reclaims finished work without blocking; `device.wait(index)` blocks until a
submission is done. Mapping is synchronous: `buffer.map_read` and `buffer.map_write` wait for
the GPU to be idle. `wgpu-mx` implements wgpu's asynchronous `map_async` on top of
`Buffer::mapped_ptr` and `Device::is_done`.

## Ray queries

When `adapter.supports_ray_query()` is true, WGSL ray queries work in any shader:

```rust
let blas = device.create_blas(&Default::default(), &[size])?;
let mut tlas = device.create_tlas(&mx::CreateTlasDescriptor { label: None, max_instances: 1024, flags, update_mode })?;
tlas[0] = Some(mx::TlasInstance::new(blas.clone(), transform, custom_data, 0xff));
encoder.build_acceleration_structures([&entry], [&tlas]);
```

A structure created with `AccelerationStructureFlags::ALLOW_UPDATE` and
`AccelerationStructureUpdateMode::PreferUpdate` is refitted instead of rebuilt after its
first build. For a top-level structure that holds when it has the same number of instances of
the same bottom-level structures as in its previous build.

## Native handles

`Device::raw()`, `Queue::raw()`, `Buffer::raw()`, `Texture::raw()` and `CommandEncoder::raw()`
return the backend's own objects (`objc2-metal` protocol objects, `ash::vk` handles or
`windows` Direct3D 12 interfaces). `Device::texture_from_raw` wraps a texture made elsewhere,
such as an OpenXR swapchain image or an upscaler's output. openOMSI runs MetalFX this way.

## Moving from wgpu

The quickest route is `wgpu-mx`: replace the `wgpu` dependency with it and keep the code. From
there, a renderer can move pass by pass to the heap. Bind a pass's textures through the heap
instead of its bind groups, check that the frame is unchanged, and continue with the next
pass.
