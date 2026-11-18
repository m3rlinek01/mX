# mX's changes to naga 30.0.2

This is naga 30.0.2 from crates.io with one change, used through `[patch.crates-io]` in the
workspace's `Cargo.toml`. Shaders that do not use the feature below compile exactly as with
the published crate.

## Binding arrays of storage buffers in the HLSL writer

naga validates `binding_array<T>` of storage buffers and writes it for SPIR-V and MSL, but its
HLSL writer hit `unreachable!()` on any access through one. mX's buffer heap needs it on
Direct3D 12.

- `back/hlsl/storage.rs`: `SubAccess::BindingArrayIndex` / `BindingArrayConstant` record which
  element of the array an access chain selects. `fill_access_chain` pushes one when it steps
  from a binding array to its element; `write_storage_address` skips them (they are not part of
  the byte address); the new `write_storage_var` writes the variable as `name[i]`, with
  `NonUniformResourceIndex` when the index is not uniform.
- Every `Load`, `Store` and `Interlocked*` on a storage buffer (`storage.rs`, `writer.rs`) writes
  its variable through `write_storage_var`.

`arrayLength` of an element is not supported; mX refuses it before naga sees it.

To be offered upstream (gfx-rs/wgpu).

## Samplers in their own registers below shader model 5.1

`back/hlsl/writer.rs`, `write_global_sampler`: naga writes every sampler as an element of a
2048-sampler heap indexed through a buffer, which shader model 5.0 (Direct3D 11) cannot
declare. Below 5.1 a sampler is written as `SamplerState name : register(sN);` at its binding's
register, as naga did before its sampler heaps. mX's Direct3D 11 backend (the bound tier)
needs it.

## Constant buffers below shader model 5.1

`back/hlsl/writer.rs`: the special constants (first vertex and instance) and the immediates
were written as `ConstantBuffer<T>`, which came with shader model 5.1. Below it they are written
as `cbuffer NameBuffer : register(bN) { T Name; }`, which reads the same.
