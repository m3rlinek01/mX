//! openOMSI's own full-screen passes (post-processing, FSR 1, upscaling, mips) drawn by wgpu
//! and by mX from the same inputs, compared byte for byte.
//!
//! Needs openOMSI's modules on disk (see `tools/openomsi_shaders.py`):
//! `MX_OPENOMSI_SHADERS=<dir> cargo test -p mx-parity --test openomsi_passes`. Each fragment
//! entry point is drawn over a full-screen triangle into 128×128 `Rgba16Float` targets; its
//! uniforms hold a fixed pattern of floats, its textures a fixed 64×64 high-range picture,
//! its samplers filter linearly. Entry points that read anything else are listed as skipped.

use naga::{AddressSpace, ImageClass, ImageDimension, ScalarKind, ShaderStage, TypeInner};

const SIZE: u32 = 128;
const TEX: u32 = 64;
const ROW: u32 = SIZE * 8; // Rgba16Float, already a multiple of 256

enum Res {
    Uniform(u32, Vec<u8>),
    Texture(u32),
    Sampler(u32),
}

struct Pass<'a> {
    wgsl: &'a str,
    vs: String,
    fs: String,
    resources: Vec<Res>,
    targets: u32,
}

fn f16(x: f32) -> u16 {
    // (the pattern's values are normal halves: no rounding mode or subnormal question)
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let man = ((b >> 13) & 0x3FF) as u16;
    if x == 0.0 { sign } else { sign | ((exp as u16) << 10) | man }
}

/// A 64×64 high-range picture: smooth gradients, a few lamps far above white, fine detail.
fn picture() -> Vec<u8> {
    let mut out = Vec::with_capacity((TEX * TEX * 8) as usize);
    for y in 0..TEX {
        for x in 0..TEX {
            let (u, v) = (x as f32 / TEX as f32, y as f32 / TEX as f32);
            let lamp = if (x % 21 == 7) && (y % 17 == 5) { 40.0 } else { 0.0 };
            let detail = if (x + y) % 2 == 0 { 0.08 } else { 0.0 };
            let px = [u * 1.5 + lamp + detail, v * 0.9 + lamp * 0.8, (1.0 - u) * v * 2.0 + lamp * 0.5, 1.0];
            for c in px {
                out.extend_from_slice(&f16(c).to_le_bytes());
            }
        }
    }
    out
}

/// Plausible parameter values: between 0.05 and 1.5, varied.
fn uniform(size: u32) -> Vec<u8> {
    (0..size / 4).flat_map(|i| (0.05 + 0.725 * (1.0 + (i as f32 * 1.37).sin())).to_le_bytes()).collect()
}

/// The full-screen passes of a module: one per fragment entry point it can feed.
fn passes(wgsl: &str) -> (Vec<Pass<'_>>, Vec<String>) {
    let module = naga::front::wgsl::parse_str(wgsl).expect("parses");
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .expect("validates");
    let index = |stage, name: &str| module.entry_points.iter().position(|e| e.stage == stage && e.name == name).unwrap();
    let vertex: Vec<&str> = module.entry_points.iter().filter(|e| e.stage == ShaderStage::Vertex).map(|e| e.name.as_str()).collect();
    let (mut out, mut skipped) = (Vec::new(), Vec::new());
    let [vs] = vertex[..] else { return (out, vec!["(not one vertex entry point)".into()]) };
    for ep in module.entry_points.iter().filter(|e| e.stage == ShaderStage::Fragment) {
        let used =
            |h: naga::Handle<naga::GlobalVariable>| !info.get_entry_point(index(ShaderStage::Vertex, vs))[h].is_empty() || !info.get_entry_point(index(ShaderStage::Fragment, &ep.name))[h].is_empty();
        let mut resources = Vec::new();
        let mut why = None;
        for (h, var) in module.global_variables.iter() {
            let Some(rb) = var.binding else { continue };
            if !used(h) {
                continue;
            }
            let inner = &module.types[var.ty].inner;
            match (var.space, inner) {
                _ if rb.group != 0 => why = Some(format!("group {}", rb.group)),
                (AddressSpace::Uniform, t) => resources.push(Res::Uniform(rb.binding, uniform(t.size(module.to_ctx())))),
                (
                    _,
                    TypeInner::Image {
                        dim: ImageDimension::D2,
                        arrayed: false,
                        class: ImageClass::Sampled {
                            kind: ScalarKind::Float,
                            multi: false,
                        },
                    },
                ) => resources.push(Res::Texture(rb.binding)),
                (_, TypeInner::Sampler { comparison: false }) => resources.push(Res::Sampler(rb.binding)),
                (_, other) => why = Some(format!("{} is {other:?}", var.name.as_deref().unwrap_or("?"))),
            }
        }
        let targets = match &ep.function.result {
            Some(r) => match (&r.binding, &module.types[r.ty].inner) {
                (Some(_), _) => 1,
                (None, TypeInner::Struct { members, .. }) => members.iter().filter(|m| matches!(m.binding, Some(naga::Binding::Location { .. }))).count() as u32,
                _ => 0,
            },
            None => 0,
        };
        if targets == 0 {
            why = Some("no colour output".into());
        }
        match why {
            Some(w) => skipped.push(format!("{}: {w}", ep.name)),
            None => out.push(Pass {
                wgsl,
                vs: vs.into(),
                fs: ep.name.clone(),
                resources,
                targets,
            }),
        }
    }
    (out, skipped)
}

// ---------------------------------------------------------------------------------------------

fn wgpu_draw(device: &wgpu::Device, queue: &wgpu::Queue, p: &Pass<'_>) -> Vec<Vec<u8>> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(p.wgsl.into()),
    });
    let fmt = wgpu::TextureFormat::Rgba16Float;
    let targets: Vec<Option<wgpu::ColorTargetState>> = (0..p.targets).map(|_| Some(fmt.into())).collect();
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(&p.vs),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(&p.fs),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    });
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: TEX,
            height: TEX,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        tex.as_image_copy(),
        &picture(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TEX * 8),
            rows_per_image: None,
        },
        tex.size(),
    );
    let view = tex.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let buffers: Vec<(u32, wgpu::Buffer)> = p
        .resources
        .iter()
        .filter_map(|r| match r {
            Res::Uniform(b, data) => {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: data.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, data);
                Some((*b, buf))
            }
            _ => None,
        })
        .collect();
    let entries: Vec<wgpu::BindGroupEntry<'_>> = p
        .resources
        .iter()
        .map(|r| match r {
            Res::Uniform(b, _) => wgpu::BindGroupEntry {
                binding: *b,
                resource: buffers.iter().find(|(x, _)| x == b).unwrap().1.as_entire_binding(),
            },
            Res::Texture(b) => wgpu::BindGroupEntry {
                binding: *b,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            Res::Sampler(b) => wgpu::BindGroupEntry {
                binding: *b,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        })
        .collect();
    let bind_group = (!entries.is_empty()).then(|| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    });
    let outs: Vec<wgpu::Texture> = (0..p.targets)
        .map(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
        .collect();
    let views: Vec<wgpu::TextureView> = outs.iter().map(|t| t.create_view(&Default::default())).collect();
    let rbs: Vec<wgpu::Buffer> = outs
        .iter()
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (ROW * SIZE) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let atts: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = views
            .iter()
            .map(|v| {
                Some(wgpu::RenderPassColorAttachment {
                    view: v,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &atts,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&pipeline);
        if let Some(bg) = &bind_group {
            rp.set_bind_group(0, bg, &[]);
        }
        rp.draw(0..3, 0..1);
    }
    for (t, rb) in outs.iter().zip(&rbs) {
        enc.copy_texture_to_buffer(
            t.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: None,
                },
            },
            t.size(),
        );
    }
    queue.submit([enc.finish()]);
    rbs.iter()
        .map(|rb| {
            rb.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let v = rb.slice(..).get_mapped_range().to_vec();
            rb.unmap();
            v
        })
        .collect()
}

fn mx_draw(device: &mx::Device, queue: &mx::Queue, p: &Pass<'_>) -> Vec<Vec<u8>> {
    use mx::*;
    let module = device
        .create_shader_module(&ShaderModuleDescriptor { label: Some(&p.fs), source: p.wgsl })
        .unwrap_or_else(|e| panic!("{e}"));
    let fmt = TextureFormat::Rgba16Float;
    let targets: Vec<Option<ColorTargetState>> = (0..p.targets).map(|_| Some(fmt.into())).collect();
    let pipeline = device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(&p.fs),
            vertex: VertexState {
                module: &module,
                entry_point: Some(&p.vs),
                constants: &[],
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some(&p.fs),
                constants: &[],
                targets: &targets,
            }),
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let tex = device
        .create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: TEX,
                height: TEX,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .unwrap();
    queue
        .write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &picture(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX * 8),
                rows_per_image: None,
            },
            tex.size(),
        )
        .unwrap();
    let view = tex.create_view(&Default::default()).unwrap();
    let sampler = device
        .create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        })
        .unwrap();
    let buffers: Vec<(u32, Buffer)> = p
        .resources
        .iter()
        .filter_map(|r| match r {
            Res::Uniform(b, data) => Some((
                *b,
                device
                    .create_buffer_init(&BufferInitDescriptor {
                        label: None,
                        contents: data,
                        usage: BufferUsages::UNIFORM,
                    })
                    .unwrap(),
            )),
            _ => None,
        })
        .collect();
    let entries: Vec<BindGroupEntry<'_>> = p
        .resources
        .iter()
        .map(|r| match r {
            Res::Uniform(b, _) => BindGroupEntry {
                binding: *b,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &buffers.iter().find(|(x, _)| x == b).unwrap().1,
                    offset: 0,
                    size: None,
                }),
            },
            Res::Texture(b) => BindGroupEntry {
                binding: *b,
                resource: BindingResource::TextureView(&view),
            },
            Res::Sampler(b) => BindGroupEntry {
                binding: *b,
                resource: BindingResource::Sampler(&sampler),
            },
        })
        .collect();
    let bind_group = (!entries.is_empty()).then(|| device.create_bind_group(&BindGroupDescriptor { label: None, entries: &entries }));
    let outs: Vec<Texture> = (0..p.targets)
        .map(|_| {
            device
                .create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: fmt,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .unwrap()
        })
        .collect();
    let views: Vec<TextureView> = outs.iter().map(|t| t.create_view(&Default::default()).unwrap()).collect();
    let rbs: Vec<Buffer> = outs
        .iter()
        .map(|_| {
            device
                .create_buffer(&BufferDescriptor {
                    label: None,
                    size: (ROW * SIZE) as u64,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                })
                .unwrap()
        })
        .collect();
    let mut enc = device.create_command_encoder(None);
    {
        let atts: Vec<Option<RenderPassColorAttachment<'_>>> = views
            .iter()
            .map(|v| {
                Some(RenderPassColorAttachment {
                    view: v,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })
            })
            .collect();
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &atts,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&pipeline);
        if let Some(bg) = &bind_group {
            rp.set_bind_group(0, bg);
        }
        rp.draw(0..3, 0..1);
    }
    for (t, rb) in outs.iter().zip(&rbs) {
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: t,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: rb,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: None,
                },
            },
            t.size(),
        );
    }
    let i = queue.submit([enc.finish()]);
    device.wait(i).unwrap();
    rbs.iter().map(|rb| rb.map_read(0..rb.size(), |b| b.to_vec()).unwrap()).collect()
}

#[test]
fn openomsi_fullscreen_passes_match_wgpu() {
    let Ok(dir) = std::env::var("MX_OPENOMSI_SHADERS") else {
        eprintln!("skipped: set MX_OPENOMSI_SHADERS (see tools/openomsi_shaders.py)");
        return;
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if cfg!(feature = "vulkan") {
            wgpu::Backends::VULKAN
        } else if cfg!(feature = "dx12") {
            wgpu::Backends::DX12
        } else {
            wgpu::Backends::METAL
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (wd, wq) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let (md, mq) = mx::Instance::new().default_adapter().unwrap().request_device(&Default::default()).unwrap();
    let (mut compared, mut failed) = (0, Vec::new());
    for name in ["post", "fsr1_srgb", "fsr1_linear", "upscale", "mip", "overlay", "lamp_sight"] {
        let path = std::path::Path::new(&dir).join(format!("{name}.wgsl"));
        let Ok(src) = std::fs::read_to_string(&path) else {
            eprintln!("{name}: not in {dir}");
            continue;
        };
        let (list, skipped) = passes(&src);
        for s in skipped {
            eprintln!("{name}: skipped {s}");
        }
        for p in &list {
            let (a, b) = (wgpu_draw(&wd, &wq, p), mx_draw(&md, &mq, p));
            for (t, (a, b)) in a.iter().zip(&b).enumerate() {
                let distinct: std::collections::HashSet<&[u8]> = a.chunks(8).collect();
                let differ = a.chunks(8).zip(b.chunks(8)).filter(|(x, y)| x != y).count();
                eprintln!("{name}::{} target {t}: {} distinct texels, {differ} differ", p.fs, distinct.len());
                if differ > 0 {
                    failed.push(format!("{name}::{} target {t} ({differ} texels)", p.fs));
                }
                compared += 1;
            }
        }
    }
    assert!(compared > 0, "nothing compared");
    assert!(failed.is_empty(), "{failed:?}");
}
