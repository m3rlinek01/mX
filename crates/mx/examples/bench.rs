//! The draw-cost benchmark: the same scene of `MX_DRAWS` small quads, each with its own
//! material (a colour and a texture), drawn three ways, off-screen:
//!
//! - `bind-groups`: a bind group per material, set before each draw (how code ported from
//!   wgpu draws),
//! - `heap`: the material's buffer view and texture indices as immediates (mX's bindless way),
//! - `gpu-driven`: one GPU-driven batch; each instance finds its material in a heap buffer.
//!
//! For each it prints the CPU time to record a frame and the GPU time of the pass (timestamp
//! queries, where the device has them), averaged over `MX_FRAMES` frames after 10 warm-up
//! frames, and a JSON line per mode for comparing builds and backends.
//!
//! `cargo run --release -p mx --example bench` (add `--features vulkan` for Vulkan).

use std::time::Instant;

use mx::*;

const W: u32 = 1920;
const H: u32 = 1080;
const TEXTURES: u32 = 64;

const SHADER: &str = "
struct Material { colour: vec4<f32>, texture: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
struct Materials { m: array<Material> }
@group(7) @binding(16) var<storage, read> materials: binding_array<Materials>;

// bind-groups: the material as a uniform and a root texture
struct Uniform { colour: vec4<f32> }
@group(0) @binding(0) var<uniform> material: Uniform;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

// heap and gpu-driven: indices
struct Draw { materials: u32, material: u32, sampler: u32, per_instance: u32 }
var<immediate> draw: Draw;

struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) @interpolate(flat) id: u32 }

@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) inst: u32) -> Out {
    let id = select(draw.material, inst, draw.per_instance != 0u);
    let corner = vec2<f32>(f32((v + 1u) % 6u / 3u), f32(v % 2u));
    let cols = 200u;
    let cell = vec2<f32>(f32(id % cols), f32(id / cols % 110u));
    let p = (cell + corner * 0.8) / vec2<f32>(f32(cols), 110.0);
    return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), corner, id);
}

@fragment fn fs_groups(i: Out) -> @location(0) vec4<f32> {
    return material.colour * textureSample(tex, smp, i.uv);
}

@fragment fn fs_heap(i: Out) -> @location(0) vec4<f32> {
    let m = materials[draw.materials].m[i.id];
    return m.colour * textureSample(mx_tex2d[m.texture], mx_samplers[draw.sampler], i.uv);
}
";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Material {
    colour: [f32; 4],
    texture: u32,
    _pad: [u32; 3],
}

fn env(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Scene {
    device: Device,
    queue: Queue,
    target: TextureView,
    groups: RenderPipeline,
    heap: RenderPipeline,
    bind_groups: Vec<BindGroup>,
    materials: BufferView,
    textures: Vec<TextureView>,
    sampler: Sampler,
    batch: Option<(IndirectDraws, Buffer, Buffer)>,
    timestamps: Option<(QuerySet, Buffer, Buffer)>,
    draws: u32,
}

fn main() {
    let adapter = Instance::new().default_adapter().expect("a GPU mX can use");
    let info = adapter.info();
    let (device, queue) = adapter.request_device(&DeviceDescriptor::default()).expect("a device");
    let draws = env("MX_DRAWS", 20_000);
    let frames = env("MX_FRAMES", 100);
    println!("mX bench on {} ({:?}): {draws} draws, {frames} frames", info.name, info.backend);

    let tex_desc = |label| TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let textures: Vec<TextureView> = (0..TEXTURES)
        .map(|i| {
            let t = device.create_texture(&tex_desc("texture")).unwrap();
            let texels: Vec<u8> = (0..16 * 16).flat_map(|p| [(p * 7 + i) as u8, (i * 4) as u8, 200, 255]).collect();
            queue
                .write_texture(
                    TexelCopyTextureInfo {
                        texture: &t,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &texels,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(64),
                        rows_per_image: None,
                    },
                    Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 1,
                    },
                )
                .unwrap();
            t.create_view(&Default::default()).unwrap()
        })
        .collect();
    let sampler = device.create_sampler(&SamplerDescriptor::default()).unwrap();
    let mats: Vec<Material> = (0..draws)
        .map(|i| Material {
            colour: [(i % 7) as f32 / 7.0, (i % 11) as f32 / 11.0, 1.0, 1.0],
            texture: textures[(i % TEXTURES) as usize].index(),
            _pad: [0; 3],
        })
        .collect();
    let mat_buf = device
        .create_buffer_init(&BufferInitDescriptor {
            label: Some("materials"),
            contents: bytemuck::cast_slice(&mats),
            usage: BufferUsages::STORAGE,
        })
        .unwrap();
    let materials = mat_buf.create_view(&Default::default()).unwrap();
    let bind_groups: Vec<BindGroup> = (0..draws)
        .map(|i| {
            let u = device
                .create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&mats[i as usize].colour),
                    usage: BufferUsages::UNIFORM,
                })
                .unwrap();
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding { buffer: &u, offset: 0, size: None }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&textures[(i % TEXTURES) as usize]),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ],
            })
        })
        .collect();

    let module = device.create_shader_module(&ShaderModuleDescriptor { label: Some("bench"), source: SHADER }).unwrap();
    let pipeline = |fs: &str| {
        device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(fs),
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    constants: &[],
                    buffers: &[],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    constants: &[],
                    targets: &[Some(TextureFormat::Rgba8Unorm.into())],
                }),
            })
            .unwrap()
    };
    let (groups, heap) = (pipeline("fs_groups"), pipeline("fs_heap"));
    let target = device
        .create_texture(&TextureDescriptor {
            label: Some("target"),
            size: Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap()
        .create_view(&Default::default())
        .unwrap();

    let batch = adapter.supports_indirect_count().then(|| {
        let args = DrawIndirectArgs {
            vertex_count: 6,
            instance_count: draws,
            first_vertex: 0,
            first_instance: 0,
        };
        let a = device
            .create_buffer_init(&BufferInitDescriptor {
                label: Some("args"),
                contents: bytemuck::bytes_of(&args),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            })
            .unwrap();
        let c = device
            .create_buffer_init(&BufferInitDescriptor {
                label: Some("count"),
                contents: bytemuck::bytes_of(&1u32),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            })
            .unwrap();
        (
            device
                .create_indirect_draws(&IndirectDrawsDescriptor {
                    label: Some("batch"),
                    max_draws: 1,
                    indexed: false,
                })
                .unwrap(),
            a,
            c,
        )
    });
    let timestamps = adapter.supports_timestamps().then(|| {
        let set = device
            .create_query_set(&QuerySetDescriptor {
                label: None,
                ty: QueryType::Timestamp,
                count: 2,
            })
            .unwrap();
        let resolve = device
            .create_buffer(&BufferDescriptor {
                label: None,
                size: 16,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            })
            .unwrap();
        let read = device
            .create_buffer(&BufferDescriptor {
                label: None,
                size: 16,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            })
            .unwrap();
        (set, resolve, read)
    });
    let scene = Scene {
        device,
        queue,
        target,
        groups,
        heap,
        bind_groups,
        materials,
        textures,
        sampler,
        batch,
        timestamps,
        draws,
    };

    for mode in ["bind-groups", "heap", "gpu-driven"] {
        if mode == "gpu-driven" && scene.batch.is_none() {
            println!("{mode:>12}: no GPU-driven draw counts on this device");
            continue;
        }
        let (mut cpu, mut gpu, mut gpu_n) = (0.0f64, 0.0f64, 0u32);
        for f in 0..frames + 10 {
            let (record, pass) = scene.frame(mode);
            if f >= 10 {
                cpu += record;
                if let Some(p) = pass {
                    gpu += p;
                    gpu_n += 1;
                }
            }
        }
        let cpu_ms = cpu / frames as f64 * 1e3;
        let gpu_ms = (gpu_n > 0).then(|| gpu / gpu_n as f64 * 1e3);
        match gpu_ms {
            Some(g) => println!("{mode:>12}: CPU {cpu_ms:7.3} ms to record, GPU {g:7.3} ms per pass"),
            None => println!("{mode:>12}: CPU {cpu_ms:7.3} ms to record (no GPU timestamps)"),
        }
        println!(
            "{{\"mode\":\"{mode}\",\"backend\":\"{:?}\",\"draws\":{},\"cpu_ms\":{cpu_ms:.4},\"gpu_ms\":{}}}",
            info.backend,
            scene.draws,
            gpu_ms.map_or("null".into(), |g| format!("{g:.4}"))
        );
    }
    drop(scene.textures);
}

impl Scene {
    /// Record and run one frame: the CPU seconds spent recording, and the pass's GPU seconds.
    fn frame(&self, mode: &str) -> (f64, Option<f64>) {
        let start = Instant::now();
        let mut enc = self.device.create_command_encoder(Some(mode));
        if let (Some((draws, args, count)), "gpu-driven") = (&self.batch, mode) {
            enc.encode_indirect_draws(draws, &self.heap, args, 0, count, 0, None);
        }
        {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some(mode),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.target,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self.timestamps.as_ref().map(|(set, _, _)| PassTimestampWrites {
                    query_set: set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                occlusion_query_set: None,
            });
            match mode {
                "bind-groups" => {
                    rp.set_pipeline(&self.groups);
                    rp.set_immediates(0, bytemuck::cast_slice(&[0u32, 0, 0, 0]));
                    for (i, bg) in self.bind_groups.iter().enumerate() {
                        rp.set_bind_group(0, bg);
                        rp.set_immediates(4, bytemuck::bytes_of(&(i as u32)));
                        rp.draw(0..6, 0..1);
                    }
                }
                "heap" => {
                    rp.set_pipeline(&self.heap);
                    rp.set_immediates(0, bytemuck::cast_slice(&[self.materials.index(), 0, self.sampler.index(), 0]));
                    for i in 0..self.draws {
                        rp.set_immediates(4, bytemuck::bytes_of(&i));
                        rp.draw(0..6, 0..1);
                    }
                }
                _ => {
                    let (draws, _, _) = self.batch.as_ref().unwrap();
                    rp.set_pipeline(&self.heap);
                    rp.set_immediates(0, bytemuck::cast_slice(&[self.materials.index(), 0, self.sampler.index(), 1]));
                    rp.execute_indirect_draws(draws);
                }
            }
        }
        if let Some((set, resolve, read)) = &self.timestamps {
            enc.resolve_query_set(set, 0..2, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, read, 0, 16);
        }
        let cb = enc.finish();
        let record = start.elapsed().as_secs_f64();
        let i = self.queue.submit([cb]);
        self.device.wait(i).unwrap();
        let pass = self.timestamps.as_ref().and_then(|(_, _, read)| {
            let q: Vec<u64> = read.map_read(0..16, |b| bytemuck::cast_slice(b).to_vec()).ok()?;
            (q[0] > 0 && q[1] > q[0]).then(|| (q[1] - q[0]) as f64 * self.queue.get_timestamp_period() as f64 * 1e-9)
        });
        (record, pass)
    }
}
