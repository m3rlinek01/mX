<!-- logo goes here -->

<h1 align="center">mX</h1>

<p align="center">
  A bindless graphics library in Rust for Metal 3, Vulkan and Direct3D 12,<br>
  with wgpu's API on top so that existing wgpu code runs on it unchanged.
</p>

<p align="center">
  <a href="https://github.com/openOMSI-org/mX/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/openOMSI-org/mX/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: AGPL-3.0-only" src="https://img.shields.io/badge/license-AGPL--3.0--only-f47f30"></a>
  <img alt="Rust 1.88+" src="https://img.shields.io/badge/rust-1.88%2B-2d3138?logo=rust&logoColor=white">
</p>

mX is the renderer backend of [openOMSI](https://github.com/openOMSI-org/openOMSI) and replaces
wgpu there. Every texture view and sampler gets an index in a heap, and shaders pick them by
that index. A draw needs a pipeline, a few buffers and some bytes of immediates. There are no
bind group layouts and no bind groups per material.

Shaders stay in WGSL. mX compiles them with [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
using the options wgpu uses, and its tests compare the result with wgpu pixel by pixel, so a
renderer that moves to mX draws the same picture.

> [!NOTE]
> mX is developed together with openOMSI, its first user. The API may still change.

## Example

```rust
let adapter = mx::Instance::new().default_adapter()?;
let (device, queue) = adapter.request_device(&Default::default())?;

let view = texture.create_view(&Default::default())?; // the view gets a heap index
pass.set_pipeline(&pipeline);
pass.set_immediates(0, bytemuck::cast_slice(&[view.index(), sampler.index()]));
pass.draw(0..3, 0..1);
```

```wgsl
struct Draw { tex: u32, smp: u32 }
var<immediate> draw: Draw;

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(mx_tex2d[draw.tex], mx_samplers[draw.smp], uv);
}
```

[`crates/mx/examples/window.rs`](crates/mx/examples/window.rs) is a complete program that opens a
window and draws into it: `cargo run -p mx --example window`.

## Features

- A bindless heap that any WGSL shader reads through `mx_tex2d[i]`, `mx_samplers[j]` and the
  other prelude arrays. Bind groups 0 to 6 remain for uniforms, storage buffers and code that
  is being ported.
- Native Metal 3, Vulkan and Direct3D 12 backends, chosen at compile time with no dispatch
  layer between the API and the driver.
- `wgpu-mx`, an implementation of wgpu 29's API on mX. Point a `wgpu` dependency at it and a
  wgpu renderer builds and runs as it is; openOMSI's does.
- Render and compute pipelines, MSAA, render bundles, indirect and multi-draw indirect draws,
  timestamp and occlusion queries, ray queries with acceleration structure refits, surfaces,
  and the native handles that MetalFX, DLSS, FSR 3 and OpenXR need.
- wgpu's guarantees: a new texture or buffer reads as zeros, queue writes land before the next
  submission, and no barriers are written by hand.
- Low CPU cost per pass: a residency set on macOS 15 and later, and staging rings for queue
  writes on every backend.

## Platforms

| Platform | Backend | State |
|----------|---------|-------|
| macOS 13+, iOS 16+ | Metal 3 | Working. openOMSI runs on it, frames match wgpu in every scene and pass compared, and Metal's API and shader validation report nothing. |
| Linux, Android | Vulkan 1.3, or 1.2 with dynamic rendering and synchronization2 | Passes every test on MoltenVK and matches wgpu's Vulkan backend there; clean under the Khronos validation layer with synchronization validation. Not yet run on Linux or Android drivers. |
| Windows | Direct3D 12, or Vulkan with the `vulkan` feature | Complete and compiled for `x86_64-pc-windows-msvc` in CI; not yet run on a GPU. |

The `vulkan` feature selects the Vulkan backend on any platform, including macOS through
MoltenVK. OpenGL is not supported, because the hardware that still needs it has no working
bindless. [docs/STATUS.md](docs/STATUS.md) lists what is tested and what is not.

## Performance

openOMSI on the Grundorf map with 30 AI vehicles, Enhanced graphics, MetalFX and 2x MSAA on an
Apple M4. Each build ran twice for 40 seconds, alternating with the other.

| | mX | wgpu |
|---|---|---|
| Frames per second | 104 | 82 |
| GPU time per frame, all passes | 6.8 ms | 8.0 ms |
| CPU time to record a frame | 1.35 ms | 1.9 ms |
| Frame time, p50 / p95 / p99 | 8.6 / 13.5 / 16.6 ms | 12.1 / 17.5 / 21.7 ms |

The wgpu numbers come from the openOMSI release built two days earlier than the mX build. The
main pass, which takes most of the GPU time, runs in 5.1 ms on both.

## Crates

| Crate | Contents |
|-------|----------|
| [`mx`](crates/mx) | The public API. It re-exports the platform's backend. |
| [`mx-types`](crates/mx-types) | Data shared by the backends: formats, usages, states and descriptors. |
| [`mx-shader`](crates/mx-shader) | WGSL through naga to MSL, SPIR-V and HLSL, with the heap prelude. |
| [`mx-metal`](crates/mx-metal) | Metal 3 backend. |
| [`mx-vulkan`](crates/mx-vulkan) | Vulkan backend. |
| [`mx-dx12`](crates/mx-dx12) | Direct3D 12 backend. |
| [`wgpu-mx`](crates/wgpu-mx) | wgpu 29's API on mX. |
| [`mx-parity`](crates/mx-parity) | Tests that draw the same scenes with wgpu and with mX and compare the pixels. |

## Using mX

```toml
[dependencies]
mx = { git = "https://github.com/openOMSI-org/mX" }
```

A wgpu renderer can switch by replacing the crate behind the `wgpu` name:

```toml
[workspace.dependencies]
wgpu = { git = "https://github.com/openOMSI-org/mX", package = "wgpu-mx" }
```

`wgpu-mx` covers what openOMSI uses from wgpu, including ray queries. It does not offer binding
arrays; with mX, shaders reach textures through the heap instead.

The [guide](docs/GUIDE.md) explains the heap, bind groups, immediates and the shader prelude.

## Building and testing

You need Rust 1.88 or newer. On macOS, the Vulkan backend also needs MoltenVK and the Vulkan
loader (`brew install molten-vk vulkan-loader`).

```sh
cargo test --workspace
cargo test -p mx --features vulkan
cargo check -p mx-dx12 --target x86_64-pc-windows-msvc
```

The tests in [`crates/mx/tests`](crates/mx/tests) run on the GPU and check the pixels they read
back. [docs/TESTING.md](docs/TESTING.md) covers validation layers, the parity tests against
wgpu and the CI setup.

## Documentation

- [Guide](docs/GUIDE.md): how to use mX
- [Architecture](docs/ARCHITECTURE.md): design, binding model and synchronisation
- [Backend contract](docs/BACKEND_API.md): what every backend implements
- [Testing](docs/TESTING.md): the test suites and how to run them
- [Status](docs/STATUS.md): what is done and tested on each platform
- [Contributing](CONTRIBUTING.md)

## License

mX is free software under the [GNU Affero General Public License, version 3](LICENSE)
(`SPDX-License-Identifier: AGPL-3.0-only`), the same license as openOMSI.

mX is part of [openOMSI.org](https://github.com/openOMSI-org). The openOMSI name and logos fall
under openOMSI's [trademark policy](https://github.com/openOMSI-org/openOMSI/blob/main/TRADEMARKS.md),
and the license does not grant permission to use them.
