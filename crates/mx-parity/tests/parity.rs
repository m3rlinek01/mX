//! The same scenes drawn by wgpu (the revision openOMSI ships) and by mX, compared byte for
//! byte. mX's promise is that switching changes how resources reach the GPU, not a single
//! pixel of what the shaders compute; these tests hold it to that on this machine's GPU.

const CLEAR: [f64; 4] = [0.1, 0.2, 0.3, 1.0];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Rgba8,
    Rgba8Srgb,
    Rgba16F,
    Rgba32F,
}

impl Fmt {
    fn bytes(self) -> u32 {
        match self {
            Fmt::Rgba8 | Fmt::Rgba8Srgb => 4,
            Fmt::Rgba16F => 8,
            Fmt::Rgba32F => 16,
        }
    }
    fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            Fmt::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            Fmt::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Fmt::Rgba16F => wgpu::TextureFormat::Rgba16Float,
            Fmt::Rgba32F => wgpu::TextureFormat::Rgba32Float,
        }
    }
    fn mx(self) -> mx::TextureFormat {
        match self {
            Fmt::Rgba8 => mx::TextureFormat::Rgba8Unorm,
            Fmt::Rgba8Srgb => mx::TextureFormat::Rgba8UnormSrgb,
            Fmt::Rgba16F => mx::TextureFormat::Rgba16Float,
            Fmt::Rgba32F => mx::TextureFormat::Rgba32Float,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Blend {
    Replace,
    Alpha,
    Add,
}

/// An sRGB texture with its whole mip chain, sampled through a repeating trilinear
/// sampler with 16× anisotropy.
struct Tex {
    size: u32,
    mips: Vec<Vec<u8>>,
}

struct Scene {
    wgsl: String,
    fmt: Fmt,
    blend: Blend,
    samples: u32,
    depth: bool,
    vertices: u32,
    instances: u32,
    texture: Option<Tex>,
    vertex: Option<Verts>,
}

/// One vertex buffer (slot 0) of interleaved attributes.
struct Verts {
    data: Vec<u8>,
    stride: u64,
    attrs: Vec<(Vf, u64, u32)>,
}

#[derive(Clone, Copy, Debug)]
enum Vf {
    Float32x2,
    Float32x3,
    Float16x2,
    Unorm8x4,
    Snorm8x4,
    Unorm16x2,
    Snorm16x2,
    Uint8x4,
    Sint16x2,
}

impl Vf {
    fn wgpu(self) -> wgpu::VertexFormat {
        match self {
            Vf::Float32x2 => wgpu::VertexFormat::Float32x2,
            Vf::Float32x3 => wgpu::VertexFormat::Float32x3,
            Vf::Float16x2 => wgpu::VertexFormat::Float16x2,
            Vf::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
            Vf::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
            Vf::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
            Vf::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
            Vf::Uint8x4 => wgpu::VertexFormat::Uint8x4,
            Vf::Sint16x2 => wgpu::VertexFormat::Sint16x2,
        }
    }
    fn mx(self) -> mx::VertexFormat {
        match self {
            Vf::Float32x2 => mx::VertexFormat::Float32x2,
            Vf::Float32x3 => mx::VertexFormat::Float32x3,
            Vf::Float16x2 => mx::VertexFormat::Float16x2,
            Vf::Unorm8x4 => mx::VertexFormat::Unorm8x4,
            Vf::Snorm8x4 => mx::VertexFormat::Snorm8x4,
            Vf::Unorm16x2 => mx::VertexFormat::Unorm16x2,
            Vf::Snorm16x2 => mx::VertexFormat::Snorm16x2,
            Vf::Uint8x4 => mx::VertexFormat::Uint8x4,
            Vf::Sint16x2 => mx::VertexFormat::Sint16x2,
        }
    }
}

const SIZE: u32 = 128;

fn padded_row(fmt: Fmt) -> u32 {
    (SIZE * fmt.bytes()).next_multiple_of(256)
}

/// The rows of a read-back picture without their padding.
fn unpad(fmt: Fmt, data: &[u8]) -> Vec<u8> {
    let (row, padded) = ((SIZE * fmt.bytes()) as usize, padded_row(fmt) as usize);
    data.chunks(padded).flat_map(|r| &r[..row]).copied().collect()
}

// ---------------------------------------------------------------------------------------------
// wgpu

struct Wgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn wgpu_gpu() -> Wgpu {
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
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a wgpu adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        ..Default::default()
    }))
    .expect("a wgpu device");
    Wgpu { device, queue }
}

impl Wgpu {
    fn read_buffer(&self, b: &wgpu::Buffer) -> Vec<u8> {
        b.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let v = b.slice(..).get_mapped_range().to_vec();
        b.unmap();
        v
    }

    fn draw(&self, s: &Scene) -> Vec<u8> {
        let d = &self.device;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(s.wgsl.as_str().into()),
        });
        let blend = match s.blend {
            Blend::Replace => None,
            Blend::Alpha => Some(wgpu::BlendState::ALPHA_BLENDING),
            Blend::Add => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
        };
        let attrs: Vec<wgpu::VertexAttribute> = s
            .vertex
            .iter()
            .flat_map(|v| {
                v.attrs.iter().map(|&(f, offset, shader_location)| wgpu::VertexAttribute {
                    format: f.wgpu(),
                    offset,
                    shader_location,
                })
            })
            .collect();
        let layouts: Vec<wgpu::VertexBufferLayout<'_>> = s
            .vertex
            .iter()
            .map(|v| wgpu::VertexBufferLayout {
                array_stride: v.stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attrs,
            })
            .collect();
        let vbuf = s.vertex.as_ref().map(|v| {
            let b = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: v.data.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&b, 0, &v.data);
            b
        });
        let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &layouts,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: s.depth.then(|| wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: s.samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: s.fmt.wgpu(),
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let target = |format, samples, usage| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = target(s.fmt.wgpu(), 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let color_view = color.create_view(&Default::default());
        let ms = (s.samples > 1).then(|| target(s.fmt.wgpu(), s.samples, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default()));
        let depth = s
            .depth
            .then(|| target(wgpu::TextureFormat::Depth32Float, s.samples, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default()));
        let bind_group = s.texture.as_ref().map(|t| {
            let tex = d.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: t.size,
                    height: t.size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: t.mips.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (level, data) in t.mips.iter().enumerate() {
                let w = (t.size >> level).max(1);
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(w * 4),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: w,
                        height: w,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            });
            let view = tex.create_view(&Default::default());
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        });
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded_row(s.fmt) * SIZE) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let [r, g, b, a] = CLEAR;
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ms.as_ref().unwrap_or(&color_view),
                    depth_slice: None,
                    resolve_target: ms.as_ref().map(|_| &color_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth.as_ref().map(|v| wgpu::RenderPassDepthStencilAttachment {
                    view: v,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&pipeline);
            if let Some(b) = &vbuf {
                rp.set_vertex_buffer(0, b.slice(..));
            }
            if let Some(bg) = &bind_group {
                rp.set_bind_group(0, bg, &[]);
            }
            rp.draw(0..s.vertices, 0..s.instances);
        }
        enc.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row(s.fmt)),
                    rows_per_image: None,
                },
            },
            color.size(),
        );
        self.queue.submit([enc.finish()]);
        unpad(s.fmt, &self.read_buffer(&readback))
    }

    fn compute(&self, wgsl: &str, n: u32) -> Vec<u8> {
        let d = &self.device;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let size = n as u64 * 16;
        let out = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out.as_entire_binding(),
            }],
        });
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(n / 64, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rb, 0, size);
        self.queue.submit([enc.finish()]);
        self.read_buffer(&rb)
    }
}

// ---------------------------------------------------------------------------------------------
// mX

struct Mx {
    device: mx::Device,
    queue: mx::Queue,
}

fn mx_gpu() -> Mx {
    let adapter = mx::Instance::with_backends(if cfg!(feature = "vulkan") {
        mx::Backends::VULKAN
    } else {
        mx::Backends::PRIMARY - mx::Backends::VULKAN
    })
    .default_adapter()
    .expect("an mX adapter");
    let (device, queue) = adapter.request_device(&Default::default()).unwrap();
    Mx { device, queue }
}

impl Mx {
    fn draw(&self, s: &Scene) -> Vec<u8> {
        use mx::*;
        let d = &self.device;
        let module = d
            .create_shader_module(&ShaderModuleDescriptor {
                label: Some("parity"),
                source: &s.wgsl,
            })
            .unwrap_or_else(|e| panic!("{e}"));
        let blend = match s.blend {
            Blend::Replace => None,
            Blend::Alpha => Some(BlendState::ALPHA_BLENDING),
            Blend::Add => Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent::REPLACE,
            }),
        };
        let attrs: Vec<VertexAttribute> = s
            .vertex
            .iter()
            .flat_map(|v| {
                v.attrs.iter().map(|&(f, offset, shader_location)| VertexAttribute {
                    format: f.mx(),
                    offset,
                    shader_location,
                })
            })
            .collect();
        let layouts: Vec<VertexBufferLayout<'_>> = s
            .vertex
            .iter()
            .map(|v| VertexBufferLayout {
                array_stride: v.stride,
                step_mode: VertexStepMode::Vertex,
                attributes: &attrs,
            })
            .collect();
        let vbuf = s.vertex.as_ref().map(|v| {
            d.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: &v.data,
                usage: BufferUsages::VERTEX,
            })
            .unwrap()
        });
        let pipeline = d
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("parity"),
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    constants: &[],
                    buffers: &layouts,
                },
                primitive: PrimitiveState::default(),
                depth_stencil: s.depth.then(|| DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: s.samples,
                    ..Default::default()
                },
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    constants: &[],
                    targets: &[Some(ColorTargetState {
                        format: s.fmt.mx(),
                        blend,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            })
            .unwrap_or_else(|e| panic!("{e}"));
        let target = |format, samples, usage| {
            d.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .unwrap()
        };
        let color = target(s.fmt.mx(), 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
        let color_view = color.create_view(&Default::default()).unwrap();
        let ms = (s.samples > 1).then(|| target(s.fmt.mx(), s.samples, TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default()).unwrap());
        let depth = s.depth.then(|| {
            target(TextureFormat::Depth32Float, s.samples, TextureUsages::RENDER_ATTACHMENT)
                .create_view(&Default::default())
                .unwrap()
        });
        let bind_group = s.texture.as_ref().map(|t| {
            let tex = d
                .create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: t.size,
                        height: t.size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: t.mips.len() as u32,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                })
                .unwrap();
            for (level, data) in t.mips.iter().enumerate() {
                let w = (t.size >> level).max(1);
                self.queue
                    .write_texture(
                        TexelCopyTextureInfo {
                            texture: &tex,
                            mip_level: level as u32,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        data,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * 4),
                            rows_per_image: None,
                        },
                        Extent3d {
                            width: w,
                            height: w,
                            depth_or_array_layers: 1,
                        },
                    )
                    .unwrap();
            }
            let sampler = d
                .create_sampler(&SamplerDescriptor {
                    address_mode_u: AddressMode::Repeat,
                    address_mode_v: AddressMode::Repeat,
                    address_mode_w: AddressMode::Repeat,
                    mag_filter: FilterMode::Linear,
                    min_filter: FilterMode::Linear,
                    mipmap_filter: MipmapFilterMode::Linear,
                    anisotropy_clamp: 16,
                    ..Default::default()
                })
                .unwrap();
            let view = tex.create_view(&Default::default()).unwrap();
            let bg = d.create_bind_group(&BindGroupDescriptor {
                label: None,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ],
            });
            (bg, tex, view, sampler)
        });
        let readback = d
            .create_buffer(&BufferDescriptor {
                label: None,
                size: (padded_row(s.fmt) * SIZE) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            })
            .unwrap();
        let mut enc = d.create_command_encoder(None);
        {
            let [r, g, b, a] = CLEAR;
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: ms.as_ref().unwrap_or(&color_view),
                    resolve_target: ms.as_ref().map(|_| &color_view),
                    ops: Operations {
                        load: LoadOp::Clear(Color { r, g, b, a }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth.as_ref().map(|v| RenderPassDepthStencilAttachment {
                    view: v,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&pipeline);
            if let Some(b) = &vbuf {
                rp.set_vertex_buffer(0, b, 0);
            }
            if let Some((bg, ..)) = &bind_group {
                rp.set_bind_group(0, bg);
            }
            rp.draw(0..s.vertices, 0..s.instances);
        }
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row(s.fmt)),
                    rows_per_image: None,
                },
            },
            color.size(),
        );
        let i = self.queue.submit([enc.finish()]);
        d.wait(i).unwrap();
        unpad(s.fmt, &readback.map_read(0..readback.size(), |b| b.to_vec()).unwrap())
    }

    fn compute(&self, wgsl: &str, n: u32) -> Vec<u8> {
        use mx::*;
        let d = &self.device;
        let module = d
            .create_shader_module(&ShaderModuleDescriptor { label: Some("parity"), source: wgsl })
            .unwrap_or_else(|e| panic!("{e}"));
        let pipeline = d
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                module: &module,
                entry_point: Some("main"),
                constants: &[],
            })
            .unwrap_or_else(|e| panic!("{e}"));
        let size = n as u64 * 16;
        let out = d
            .create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
            .unwrap();
        let rb = d
            .create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            })
            .unwrap();
        let bg = d.create_bind_group(&BindGroupDescriptor {
            label: None,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding { buffer: &out, offset: 0, size: None }),
            }],
        });
        let mut enc = d.create_command_encoder(None);
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg);
            cp.dispatch_workgroups(n / 64, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rb, 0, size);
        let i = self.queue.submit([enc.finish()]);
        d.wait(i).unwrap();
        rb.map_read(0..size, |b| b.to_vec()).unwrap()
    }
}

// ---------------------------------------------------------------------------------------------
// Comparison

/// Panics with where and by how much the pictures differ, if they do at all.
fn same(name: &str, texel: usize, wgpu: &[u8], mx: &[u8]) {
    assert_eq!(wgpu.len(), mx.len(), "{name}: sizes differ");
    if wgpu == mx {
        return;
    }
    let differ: Vec<usize> = (0..wgpu.len() / texel).filter(|i| wgpu[i * texel..(i + 1) * texel] != mx[i * texel..(i + 1) * texel]).collect();
    let first = differ[0];
    panic!(
        "{name}: {} of {} texels differ; first at texel {first} (x {}, y {}): wgpu {:?}, mX {:?}",
        differ.len(),
        wgpu.len() / texel,
        first % SIZE as usize,
        first / SIZE as usize,
        &wgpu[first * texel..(first + 1) * texel],
        &mx[first * texel..(first + 1) * texel]
    );
}

fn check(name: &str, s: Scene) {
    let (w, m) = (wgpu_gpu(), mx_gpu());
    let (a, b) = (w.draw(&s), m.draw(&s));
    // (a picture of one colour would prove nothing)
    let texel = s.fmt.bytes() as usize;
    let distinct: std::collections::HashSet<&[u8]> = a.chunks(texel).collect();
    assert!(distinct.len() > 16, "{name}: the scene drew only {} distinct texels", distinct.len());
    same(name, texel, &a, &b);
}

const FULLSCREEN_VS: &str = "
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y));
}
";

/// Overlapping triangles, one per instance, at depths and with colours (alpha 0.3..0.8)
/// that differ by instance and vertex.
const TRIANGLES_VS: &str = "
struct Out { @builtin(position) pos: vec4<f32>, @location(0) c: vec4<f32> }
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) k: u32) -> Out {
    let fk = f32(k);
    let a = fk * 0.523 + f32(v) * 2.0944;
    let center = vec2<f32>(sin(fk * 1.7), cos(fk * 2.3)) * 0.3;
    let p = center + vec2<f32>(cos(a), sin(a)) * (0.9 - fk * 0.05);
    let c = vec4<f32>(fract(fk * 0.37 + f32(v) * 0.11), fract(fk * 0.61 + 0.2), fract(fk * 0.13 + 0.5 + f32(v) * 0.07), 0.3 + fract(fk * 0.29) * 0.5);
    return Out(vec4<f32>(p, fract(fk * 0.43) * 0.9, 1.0), c);
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return i.c; }
";

#[test]
fn shading_math_is_bit_identical() {
    check(
        "math",
        Scene {
            wgsl: format!(
                "{FULLSCREEN_VS}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {{
    let p = i.uv * 6.2831853 - 3.14159265;
    let a = sin(p.x * 3.7) * cos(p.y * 2.3) + tan(p.x * 0.31);
    let b = exp(-dot(p, p) * 0.25) + pow(abs(p.x) + 0.01, 2.2) + log2(abs(p.y) + 1.5);
    let n = normalize(vec3<f32>(p, 1.0 + a));
    let c = atan2(p.y, p.x) + sqrt(abs(a * b)) + inverseSqrt(b + 1.0) + fract(a * 13.7) + smoothstep(-1.0, 1.0, a);
    return vec4<f32>(a, b, c, n.x * n.z + length(p) / (1.0 + distance(p, vec2<f32>(1.0, 0.5))));
}}"
            ),
            fmt: Fmt::Rgba32F,
            blend: Blend::Replace,
            samples: 1,
            depth: false,
            vertices: 3,
            instances: 1,
            texture: None,
            vertex: None,
        },
    );
}

/// A procedural sRGB texture (checks and noise) with a box-filtered mip chain.
fn pattern(size: u32) -> Tex {
    let mut hash = 0x9E37_79B9u32;
    let mut base = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            hash ^= hash << 13;
            hash ^= hash >> 17;
            hash ^= hash << 5;
            let check = ((x / 16 + y / 16) % 2) as u8 * 150;
            base.extend_from_slice(&[check.wrapping_add((hash & 0x3F) as u8), (x * 255 / size) as u8, ((hash >> 8) & 0xFF) as u8, 255]);
        }
    }
    let mut mips = vec![base];
    let mut s = size;
    while s > 1 {
        let prev = mips.last().unwrap();
        let n = s / 2;
        let mut next = Vec::with_capacity((n * n * 4) as usize);
        for y in 0..n {
            for x in 0..n {
                for c in 0..4 {
                    let at = |dx: u32, dy: u32| prev[(((2 * y + dy) * s + 2 * x + dx) * 4 + c) as usize] as u32;
                    next.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
                }
            }
        }
        mips.push(next);
        s = n;
    }
    Tex { size, mips }
}

#[test]
fn trilinear_anisotropic_srgb_sampling_is_identical() {
    check(
        "ground plane",
        Scene {
            wgsl: "
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let c = array<vec2<f32>, 6>(vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(-1.0, 1.0), vec2(-1.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));
    let q = c[i];
    let world = vec3<f32>(q.x * 6.0, -1.0, mix(1.0, 40.0, q.y));
    let clip = vec4<f32>(world.x * 1.2, world.y * 1.6 + world.z * 0.02, world.z * 0.98 - 0.5, world.z);
    return Out(clip, vec2<f32>(world.x + 0.3 * world.z, world.z) * 0.75);
}
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return textureSample(t, s, i.uv); }"
                .into(),
            fmt: Fmt::Rgba8Srgb,
            blend: Blend::Replace,
            samples: 1,
            depth: false,
            vertices: 6,
            instances: 1,
            texture: Some(pattern(256)),
            vertex: None,
        },
    );
}

#[test]
fn alpha_blending_into_srgb_is_identical() {
    check(
        "alpha blending",
        Scene {
            wgsl: TRIANGLES_VS.into(),
            fmt: Fmt::Rgba8Srgb,
            blend: Blend::Alpha,
            samples: 1,
            depth: false,
            vertices: 3,
            instances: 12,
            texture: None,
            vertex: None,
        },
    );
}

#[test]
fn additive_blending_into_half_floats_is_identical() {
    check(
        "additive blending",
        Scene {
            wgsl: TRIANGLES_VS.into(),
            fmt: Fmt::Rgba16F,
            blend: Blend::Add,
            samples: 1,
            depth: false,
            vertices: 3,
            instances: 12,
            texture: None,
            vertex: None,
        },
    );
}

#[test]
fn depth_tested_msaa_resolve_is_identical() {
    check(
        "depth + MSAA",
        Scene {
            wgsl: TRIANGLES_VS.into(),
            fmt: Fmt::Rgba8,
            blend: Blend::Replace,
            samples: 4,
            depth: true,
            vertices: 3,
            instances: 12,
            texture: None,
            vertex: None,
        },
    );
}

#[test]
fn compute_math_is_bit_identical() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = f32(id.x) * 0.0137 - 3.0;
    out[id.x] = vec4<f32>(sin(x) * exp(x * 0.1), pow(abs(x), 0.4545), atan2(x, 1.3) + sqrt(abs(x)), fma(x, x, -2.0) / (x * x + 1.0));
}";
    let n = 4096;
    let (a, b) = (wgpu_gpu().compute(src, n), mx_gpu().compute(src, n));
    same("compute", 16, &a, &b);
}

/// Every vertex format openOMSI uses (Float32x2/x3, Float16x2, Unorm8x4) and the other
/// normalised and integer ones: wgpu-hal fetches vertices in the shader (naga's vertex
/// pulling), mX through Metal's vertex descriptors; the converted values must not differ.
#[test]
fn vertex_formats_convert_identically() {
    let (tris, stride) = (16u32, 48u64);
    let mut hash = 0x2545_F491u32;
    let mut next = || {
        hash ^= hash << 13;
        hash ^= hash >> 17;
        hash ^= hash << 5;
        hash
    };
    let mut data = Vec::new();
    for _ in 0..tris * 3 {
        let mut f = |scale: f32| (next() as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale;
        let pos = [f(1.0), f(1.0)];
        let f3 = [f(3.0), f(100.0), f(0.001)];
        let h2 = [half(f(4.0)), half(f(0.01))];
        let mut v = Vec::new();
        v.extend(bytemuck::bytes_of(&pos));
        for _ in 0..4 {
            v.extend(next().to_le_bytes()); // unorm8x4, snorm8x4, unorm16x2, snorm16x2
        }
        v.extend(bytemuck::bytes_of(&h2));
        v.extend(bytemuck::bytes_of(&f3));
        v.extend(next().to_le_bytes()); // uint8x4
        v.extend(next().to_le_bytes()); // sint16x2
        assert_eq!(v.len() as u64, stride);
        data.extend(v);
    }
    // (snorm's -128 and -32768 come up and must clamp to -1 on both)
    let attrs = vec![
        (Vf::Float32x2, 0, 0),
        (Vf::Unorm8x4, 8, 1),
        (Vf::Snorm8x4, 12, 2),
        (Vf::Unorm16x2, 16, 3),
        (Vf::Snorm16x2, 20, 4),
        (Vf::Float16x2, 24, 5),
        (Vf::Float32x3, 28, 6),
        (Vf::Uint8x4, 40, 7),
        (Vf::Sint16x2, 44, 8),
    ];
    check(
        "vertex formats",
        Scene {
            wgsl: "
struct In {
    @location(0) pos: vec2<f32>, @location(1) a: vec4<f32>, @location(2) b: vec4<f32>, @location(3) c: vec2<f32>,
    @location(4) d: vec2<f32>, @location(5) e: vec2<f32>, @location(6) f: vec3<f32>, @location(7) g: vec4<u32>, @location(8) h: vec2<i32>,
}
struct Out {
    @builtin(position) p: vec4<f32>,
    @location(0) @interpolate(flat) a: vec4<f32>, @location(1) @interpolate(flat) b: vec4<f32>, @location(2) @interpolate(flat) cd: vec4<f32>,
    @location(3) @interpolate(flat) ef: vec4<f32>, @location(4) @interpolate(flat) fgh: vec4<f32>,
}
@vertex fn vs(i: In) -> Out {
    return Out(vec4<f32>(i.pos, 0.5, 1.0), i.a, i.b, vec4<f32>(i.c, i.d), vec4<f32>(i.e, i.f.xy), vec4<f32>(i.f.z, f32(i.g.x + i.g.y * 256u), f32(i.g.z + i.g.w * 256u), f32(i.h.x * 65536 + i.h.y)));
}
// (each column of pixels shows one group of attributes, so no value hides in a sum)
@fragment fn fs(o: Out) -> @location(0) vec4<f32> {
    switch u32(o.p.x) % 5u {
        case 0u: { return o.a; }
        case 1u: { return o.b; }
        case 2u: { return o.cd; }
        case 3u: { return o.ef; }
        default: { return o.fgh; }
    }
}"
            .into(),
            fmt: Fmt::Rgba32F,
            blend: Blend::Replace,
            samples: 1,
            depth: false,
            vertices: tris * 3,
            instances: 1,
            texture: None,
            vertex: Some(Verts { data, stride, attrs }),
        },
    );
}

fn half(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}
