# Contributing to mX

mX is developed together with [openOMSI](https://github.com/openOMSI-org/openOMSI), so the first
question for any change is what it does to openOMSI's picture and frame time.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing a backend: the binding model
and the synchronisation rules explain most of the code. For anything larger than a fix, open
an issue first and describe the approach.

## What a change needs

- The picture stays the same. Shaders are compiled with the options wgpu uses. A change to
  what a shader computes needs a reason and a parity test.
- A test on the GPU. A bug fix comes with a conformance test in
  `crates/mx/tests/conformance.rs` that fails without the fix. Prefer tests that check pixels
  over tests that check calls.
- All three backends. A new API goes into Metal, Vulkan and Direct3D 12 together, with the same
  behaviour. If you cannot run one of them, it must still compile; CI checks Direct3D 12 on
  Windows.
- Clean validation on the backends you touched: `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`
  for Metal, the Khronos layer for Vulkan ([docs/TESTING.md](docs/TESTING.md)).
- `cargo fmt --all` and `cargo clippy --workspace` without new warnings.

## Pull requests

Keep one topic per pull request. The description says what changed and why, and what you
could not test, such as a platform, a driver or a GPU. Performance work needs numbers from
before and after, taken on the same machine with the same scene and settings.

## License

mX is licensed under the [GNU AGPL, version 3 only](LICENSE). By contributing you agree that
your contribution is licensed under the same terms.
