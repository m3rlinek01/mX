//! openOMSI's own WGSL modules, as its renderer assembles them, compile through mX unchanged.
//!
//! Needs the modules on disk: `python3 tools/openomsi_shaders.py <openOMSI checkout> <dir>`,
//! then `MX_OPENOMSI_SHADERS=<dir> cargo test -p mx --test openomsi_shaders`. Without the
//! variable the test says so and passes.

use mx::*;

#[test]
fn every_openomsi_module_compiles_for_the_gpu() {
    let Ok(dir) = std::env::var("MX_OPENOMSI_SHADERS") else {
        eprintln!("skipped: set MX_OPENOMSI_SHADERS (see tools/openomsi_shaders.py)");
        return;
    };
    let adapter = Instance::new().default_adapter().expect("a GPU mX can use");
    let (device, _queue) = adapter.request_device(&Default::default()).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "wgsl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .wgsl files in {dir}");
    let mut failed = Vec::new();
    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        if name.starts_with("rt") && !adapter.supports_ray_query() {
            eprintln!("{name}: skipped, no ray queries");
            continue;
        }
        let source = std::fs::read_to_string(path).unwrap();
        let t = std::time::Instant::now();
        let r = device
            .create_shader_module(&ShaderModuleDescriptor { label: Some(&name), source: &source })
            .and_then(|m| device.precompile(&m));
        match r {
            Ok(()) => eprintln!("{name}: ok in {:.2} s", t.elapsed().as_secs_f32()),
            Err(e) => {
                eprintln!("{name}: FAILED\n{e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{} of {} modules failed: {failed:?}", failed.len(), files.len());
}
