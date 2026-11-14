## What changes, and why

## How it was tested
<!-- Which backends and GPUs; conformance or parity tests added; validation layers run. -->

## What could break
<!-- What you could not test (a platform, a driver), and anything reviewers should look at first. -->

## Checklist
- [ ] Tests pass on the backends I could run (`cargo test -p mx`, with `--features vulkan` where it applies)
- [ ] Clean under Metal's validation / the Khronos layer for the backends touched
- [ ] Direct3D 12 still compiles (`cargo check -p mx-dx12 --target x86_64-pc-windows-msvc`)
- [ ] `cargo fmt --all` and `cargo clippy --workspace` without new warnings
