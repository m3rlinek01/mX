//! A window drawn by mX: a mipmapped checkerboard reached through the bindless heap,
//! scrolling, with the texture and sampler indices passed as immediates.
//!
//! `MX_FRAMES=n` quits after n frames; `MX_SCREENSHOT=out.png` saves the last frame.

use std::sync::Arc;

use mx::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const SHADER: &str = "
struct Draw { tex: u32, smp: u32, time: f32, _pad: u32 }
var<immediate> draw: Draw;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return Out(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(p.x, 1.0 - p.y));
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let uv = i.uv * vec2<f32>(4.0, 3.0) + vec2<f32>(draw.time * 0.1, 0.0);
    return textureSample(mx_tex2d[draw.tex], mx_samplers[draw.smp], uv);
}
";

struct Gfx {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    _checker: Texture,
    view: TextureView,
    sampler: Sampler,
    frame: u32,
}

#[derive(Default)]
struct App {
    gfx: Option<Gfx>,
    max_frames: Option<u32>,
    screenshot: Option<String>,
}

fn checkerboard(device: &Device, queue: &Queue) -> Texture {
    let n = 256u32;
    let mips = 9;
    let size = Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    let tex = device
        .create_texture(&TextureDescriptor {
            label: Some("checker"),
            size,
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .unwrap();
    let mut data = Vec::with_capacity((n * n * 4) as usize);
    for y in 0..n {
        for x in 0..n {
            let on = ((x / 32) + (y / 32)) % 2 == 0;
            data.extend_from_slice(if on { &[230, 190, 40, 255] } else { &[20, 60, 110, 255] });
        }
    }
    queue
        .write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * 4),
                rows_per_image: None,
            },
            size,
        )
        .unwrap();
    let mut enc = device.create_command_encoder(Some("mips"));
    enc.generate_mipmaps(&tex);
    queue.submit([enc.finish()]);
    tex
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) {
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("mX").with_inner_size(winit::dpi::LogicalSize::new(800, 600)))
                .unwrap(),
        );
        let instance = Instance::new();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.default_adapter().unwrap();
        println!("adapter: {:?}", adapter.info());
        let (device, queue) = adapter.request_device(&Default::default()).unwrap();
        let caps = surface.get_capabilities(&adapter);
        let size = window.inner_size();
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            format: caps.formats[0],
            width: size.width,
            height: size.height,
            present_mode: PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let module = device
            .create_shader_module(&ShaderModuleDescriptor {
                label: Some("window"),
                source: SHADER,
            })
            .unwrap();
        let pipeline = device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("window"),
                vertex: VertexState {
                    module: &module,
                    entry_point: None,
                    constants: &[],
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: None,
                    constants: &[],
                    targets: &[Some(config.format.into())],
                }),
            })
            .unwrap();
        let checker = checkerboard(&device, &queue);
        let view = checker.create_view(&Default::default()).unwrap();
        let sampler = device
            .create_sampler(&SamplerDescriptor {
                address_mode_u: AddressMode::Repeat,
                address_mode_v: AddressMode::Repeat,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: MipmapFilterMode::Linear,
                anisotropy_clamp: 8,
                ..Default::default()
            })
            .unwrap();
        window.request_redraw();
        self.gfx = Some(Gfx {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            _checker: checker,
            view,
            sampler,
            frame: 0,
        });
    }

    fn redraw(&mut self, el: &ActiveEventLoop) {
        let g = self.gfx.as_mut().unwrap();
        let frame = match g.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(f) => f,
            CurrentSurfaceTexture::Suboptimal(f) => {
                g.surface.configure(&g.device, &g.config);
                f
            }
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                g.surface.configure(&g.device, &g.config);
                g.window.request_redraw();
                return;
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                g.window.request_redraw();
                return;
            }
        };
        let target = frame.texture.create_view(&Default::default()).unwrap();
        let mut enc = g.device.create_command_encoder(Some("frame"));
        {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&g.pipeline);
            let time = g.frame as f32 / 60.0;
            pass.set_immediates(0, bytemuck::cast_slice(&[g.view.index(), g.sampler.index(), time.to_bits(), 0]));
            pass.draw(0..3, 0..1);
        }
        g.frame += 1;
        let last = self.max_frames.is_some_and(|m| g.frame >= m);
        let readback = (last && self.screenshot.is_some()).then(|| {
            let (w, h) = (frame.texture.width(), frame.texture.height());
            let bpr = (w * 4).next_multiple_of(256);
            let buf = g
                .device
                .create_buffer(&BufferDescriptor {
                    label: Some("screenshot"),
                    size: (bpr * h) as u64,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                })
                .unwrap();
            enc.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bpr),
                        rows_per_image: Some(h),
                    },
                },
                frame.texture.size(),
            );
            (buf, w, h, bpr, frame.texture.format())
        });
        g.queue.submit([enc.finish()]);
        frame.present();
        if let (Some((buf, w, h, bpr, format)), Some(path)) = (readback, &self.screenshot) {
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            buf.map_read(0..buf.size(), |b| {
                for row in b.chunks(bpr as usize).take(h as usize) {
                    for px in row[..(w * 4) as usize].chunks(4) {
                        // (BGRA surfaces)
                        let bgra = matches!(format, TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb);
                        rgba.extend_from_slice(&if bgra { [px[2], px[1], px[0], 255] } else { [px[0], px[1], px[2], 255] });
                    }
                }
            })
            .unwrap();
            let file = std::fs::File::create(path).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&rgba).unwrap();
            println!("saved {path} ({w}x{h}, {format:?})");
        }
        if last {
            g.device.wait_idle().unwrap();
            println!("presented {} frames", g.frame);
            el.exit();
        } else {
            g.window.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
            self.init(el);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                if let Some(g) = &mut self.gfx {
                    g.config.width = s.width.max(1);
                    g.config.height = s.height.max(1);
                    g.surface.configure(&g.device, &g.config);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(el),
            _ => {}
        }
    }
}

fn main() {
    let mut app = App {
        max_frames: std::env::var("MX_FRAMES").ok().and_then(|v| v.parse().ok()),
        screenshot: std::env::var("MX_SCREENSHOT").ok(),
        ..Default::default()
    };
    EventLoop::new().unwrap().run_app(&mut app).unwrap();
}
