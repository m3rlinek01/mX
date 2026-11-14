//! CPU cost of making the heap's textures resident in each render pass (`useResources` on
//! Metal). Ignored by default: `cargo test -p mx --release --test residency_cost -- --ignored --nocapture`.

use mx::*;

#[test]
#[ignore]
fn heap_residency_cost_per_pass() {
    let adapter = Instance::new().default_adapter().unwrap();
    let (device, queue) = adapter.request_device(&DeviceDescriptor::default()).unwrap();
    let texture = |usage| {
        device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage,
                view_formats: &[],
            })
            .unwrap()
    };
    let target = texture(TextureUsages::RENDER_ATTACHMENT);
    let tv = target.create_view(&Default::default()).unwrap();
    let m = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: "struct D { t: u32 }\nvar<immediate> d: D;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(f32(i & 1u), f32(i >> 1u), 0.0, 1.0); }
@fragment fn fs() -> @location(0) vec4<f32> { return textureLoad(mx_tex2d[d.t], vec2<i32>(0, 0), 0); }",
        })
        .unwrap();
    let p = device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            vertex: VertexState {
                module: &m,
                entry_point: None,
                constants: &[],
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &m,
                entry_point: None,
                constants: &[],
                targets: &[Some(TextureFormat::Rgba8Unorm.into())],
            }),
        })
        .unwrap();
    for n in [100usize, 1000, 5000, 20000] {
        let textures: Vec<(Texture, TextureView)> = (0..n)
            .map(|_| {
                let t = texture(TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST);
                let v = t.create_view(&Default::default()).unwrap();
                (t, v)
            })
            .collect();
        let passes = 40;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let mut enc = device.create_command_encoder(None);
            for _ in 0..passes {
                let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &tv,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rp.set_pipeline(&p);
                rp.set_immediates(0, &textures[0].1.index().to_le_bytes());
                rp.draw(0..3, 0..1);
            }
            let cb = enc.finish();
            best = best.min(t0.elapsed().as_secs_f64());
            let i = queue.submit([cb]);
            device.wait(i).unwrap();
        }
        println!("{n:6} heap textures, {passes} passes: {:.3} ms to record", best * 1000.0);
    }
}
