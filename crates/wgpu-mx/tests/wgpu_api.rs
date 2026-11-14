//! wgpu-mx through wgpu's own API, as openOMSI calls it: ray queries and texture clears.

use wgpu::util::DeviceExt;
use wgpu_mx as wgpu;

/// wgpu-mx's futures are ready at once.
fn now<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
        std::task::Poll::Ready(v) => v,
        std::task::Poll::Pending => panic!("wgpu-mx future not ready"),
    }
}

/// The GPU one test at a time (as mX's conformance tests).
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn open(features: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = now(instance.request_adapter(&Default::default())).ok()?;
    if !adapter.features().contains(features) {
        return None;
    }
    now(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .ok()
}

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let v = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    v
}

#[test]
fn ray_queries_through_wgpus_api() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((device, queue)) = open(wgpu::Features::EXPERIMENTAL_RAY_QUERY) else {
        eprintln!("skipped: no ray queries on this GPU");
        return;
    };
    let verts: [f32; 9] = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    let indices: [u32; 3] = [0, 1, 2];
    let bytes = |f: &[f32]| f.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &bytes(&verts),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &indices.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(3),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        max_instances: 4,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let at = |x: f32| [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, at(0.0), 7, 0xFF));
    tlas[2] = Some(wgpu::TlasInstance::new(&blas, at(10.0), 0x123456, 0xFF));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "enable wgpu_ray_query;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var acc: acceleration_structure;
@compute @workgroup_size(3) fn main(@builtin(local_invocation_index) i: u32) {
    let x = array<f32, 3>(0.0, 10.0, 5.0)[i];
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 100.0, vec3<f32>(x, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE) { out[i] = hit.instance_custom_data; } else { out[i] = 0xFFFFFFFFu; }
}"
            .into(),
        ),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
        ],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 12,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let run = |tlas: &wgpu::Tlas, build: bool| {
        let mut enc = device.create_command_encoder(&Default::default());
        let entry = wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vb,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&ib),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        };
        let entries = if build { vec![entry] } else { vec![] };
        enc.build_acceleration_structures(entries.iter(), std::iter::once(tlas));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tlas.as_binding(),
                },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rb, 0, 12);
        queue.submit([enc.finish()]);
        let b = read(&device, &rb);
        b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>()
    };
    assert_eq!(run(&tlas, true), [7, 0x123456, u32::MAX]);
    // instances changed through wgpu's API are what the next build uses
    tlas[0] = None;
    tlas[3] = Some(wgpu::TlasInstance::new(&blas, at(5.0), 42, 0xFF));
    assert_eq!(run(&tlas, false), [u32::MAX, 0x123456, 42]);
}

#[test]
fn clear_texture_through_wgpus_api() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((device, queue)) = open(wgpu::Features::CLEAR_TEXTURE) else {
        panic!("wgpu-mx offers CLEAR_TEXTURE everywhere");
    };
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    queue.write_texture(
        t.as_image_copy(),
        &[200u8; 64 * 64 * 4],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: None,
        },
        t.size(),
    );
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64 * 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.clear_texture(&t, &wgpu::ImageSubresourceRange::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        t.size(),
    );
    queue.submit([enc.finish()]);
    assert!(read(&device, &rb).iter().all(|b| *b == 0), "clear_texture left texels");
}
