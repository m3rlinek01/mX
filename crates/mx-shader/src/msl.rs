//! Metal Shading Language output and mX's Metal slot layout.
//!
//! | buffer slot | use |
//! |-------------|-----|
//! | 0..=7       | heap bindings (`heap::*`), each an argument buffer of `MTLResourceID`s |
//! | 8           | immediates |
//! | 9           | sizes of runtime-sized storage buffers, then of the vertex buffers |
//! | 10..=22     | root buffers, in (group, binding) order |
//! | 23..=30     | vertex buffers, `30 - index` |
//!
//! Root textures and samplers take texture / sampler slots from 0 in (group, binding) order.

use std::collections::BTreeMap;

use crate::{HEAP_GROUP, Module, RootKind, ShaderError, heap};
use naga::back::msl;

pub use msl::{AttributeMapping, VertexBufferMapping, VertexBufferStepMode};
pub use naga_types::VertexFormat;

pub const IMMEDIATES_SLOT: u32 = 8;
pub const SIZES_SLOT: u32 = 9;
pub const FIRST_ROOT_BUFFER_SLOT: u32 = 10;
pub const MAX_ROOT_BUFFERS: u32 = 13;
pub const MAX_VERTEX_BUFFERS: u32 = 8;
pub const MAX_ROOT_TEXTURES: u32 = 96;
pub const MAX_ROOT_SAMPLERS: u32 = 16;

pub const fn vertex_buffer_slot(index: u32) -> u32 {
    30 - index
}

/// Where a root binding lives in Metal's argument tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootSlot {
    Buffer {
        slot: u32,
        kind: RootKind,
    },
    Texture(u32),
    Sampler(u32),
    /// A top-level acceleration structure (ray queries), in a buffer slot.
    AccelerationStructure(u32),
}

/// The Metal slots of every root binding of a module.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub roots: BTreeMap<(u32, u32), RootSlot>,
}

impl Layout {
    /// Slots for every root binding of the module.
    pub fn for_module(m: &Module) -> Result<Layout, ShaderError> {
        Self::assign(m, m.roots.iter())
    }

    /// Slots for the root bindings one entry point uses. As in wgpu, Metal's limits count
    /// what the entry point reads, not everything its module declares (openOMSI's motion
    /// vectors share the scene module, which declares more buffers than one stage takes).
    pub fn for_entry_point(m: &Module, stage: naga::ShaderStage, name: &str) -> Result<Layout, ShaderError> {
        Self::assign(m, m.entry_point_roots(stage, name)?.iter())
    }

    fn assign<'a>(m: &Module, list: impl Iterator<Item = &'a crate::RootBinding>) -> Result<Layout, ShaderError> {
        let (mut b, mut t, mut s) = (0, 0, 0);
        let mut roots = BTreeMap::new();
        for r in list {
            let slot = match r.kind {
                RootKind::UniformBuffer | RootKind::StorageBuffer { .. } => {
                    b += 1;
                    RootSlot::Buffer {
                        slot: FIRST_ROOT_BUFFER_SLOT + b - 1,
                        kind: r.kind,
                    }
                }
                RootKind::Texture | RootKind::StorageTexture => {
                    t += 1;
                    RootSlot::Texture(t - 1)
                }
                RootKind::Sampler => {
                    s += 1;
                    RootSlot::Sampler(s - 1)
                }
                RootKind::AccelerationStructure => {
                    b += 1;
                    RootSlot::AccelerationStructure(FIRST_ROOT_BUFFER_SLOT + b - 1)
                }
            };
            roots.insert((r.group, r.binding), slot);
        }
        let over = |n: u32, max: u32, what: &str| {
            (n > max).then(|| ShaderError::Layout {
                label: m.label.clone(),
                message: format!("{n} root {what}; Metal takes {max}"),
            })
        };
        if let Some(e) = over(b, MAX_ROOT_BUFFERS, "buffers")
            .or(over(t, MAX_ROOT_TEXTURES, "textures"))
            .or(over(s, MAX_ROOT_SAMPLERS, "samplers"))
        {
            return Err(e);
        }
        Ok(Layout { roots })
    }
}

/// One entry point in MSL.
#[derive(Clone, Debug)]
pub struct Output {
    pub source: String,
    /// The function's name in `source` (naga renames names MSL reserves).
    pub entry_point: String,
    /// Which heap bindings the entry point reads (bit per `heap::*`).
    pub heap_bindings_used: u32,
    pub uses_immediates: bool,
    /// The root buffers whose byte sizes go to `SIZES_SLOT`, as consecutive `u32`s in this
    /// order (naga's: the order of declaration among the globals this entry point keeps).
    pub sized_buffers: Vec<(u32, u32)>,
    pub workgroup_size: [u32; 3],
    /// Bytes of each `var<workgroup>` the entry point uses, in threadgroup-memory index
    /// order: Metal needs each length set before a dispatch (as wgpu-hal sets it).
    pub workgroup_memory: Vec<u32>,
}

/// Write one entry point of `m` as MSL `version` (3.0 at least; wgpu-hal picks it by OS
/// version, see `mx_metal`), with `constants` applied to its `override`s. A vertex stage
/// fetches its attributes from `vertex_buffers` itself (naga's vertex pulling, as wgpu-hal
/// compiles every vertex shader): the conversions of normalised formats are then the
/// shader's arithmetic, not Metal's vertex fetch, which differs from it by an ulp.
pub fn write(
    m: &Module,
    layout: &Layout,
    stage: naga::ShaderStage,
    entry: Option<&str>,
    constants: &[(&str, f64)],
    version: (u8, u8),
    vertex_buffers: &[VertexBufferMapping],
) -> Result<Output, ShaderError> {
    let name = m.entry_point(stage, entry)?.name.clone();
    let (module, info) = m.specialize(stage, &name, constants)?;
    let backend_err = |message: String| ShaderError::Backend {
        label: m.label.clone(),
        target: "MSL",
        message,
    };

    let mut resources = BTreeMap::new();
    for b in 0..heap::COUNT {
        let target = msl::BindTarget {
            buffer: Some(b as u8),
            ..Default::default()
        };
        resources.insert(naga::ResourceBinding { group: HEAP_GROUP, binding: b }, target);
    }
    for (&(group, binding), slot) in &layout.roots {
        let target = match *slot {
            RootSlot::Buffer { slot, .. } | RootSlot::AccelerationStructure(slot) => msl::BindTarget {
                buffer: Some(slot as u8),
                ..Default::default()
            },
            RootSlot::Texture(t) => msl::BindTarget {
                texture: Some(t as u8),
                mutable: true,
                ..Default::default()
            },
            RootSlot::Sampler(s) => msl::BindTarget {
                sampler: Some(msl::BindSamplerTarget::Resource(s as u8)),
                ..Default::default()
            },
        };
        resources.insert(naga::ResourceBinding { group, binding }, target);
    }
    let ep_resources = msl::EntryPointResources {
        resources,
        immediates_buffer: Some(IMMEDIATES_SLOT as u8),
        sizes_buffer: Some(SIZES_SLOT as u8),
    };

    let mut options = msl::Options {
        lang_version: version,
        bounds_check_policies: crate::bounds_check_policies(),
        zero_initialize_workgroup_memory: true,
        force_loop_bounding: true,
        fake_missing_bindings: false,
        ..Default::default()
    };
    options.per_entry_point_map.insert(name.clone(), ep_resources);

    let pipeline = msl::PipelineOptions {
        entry_point: Some((stage, name.clone())),
        vertex_pulling_transform: stage == naga::ShaderStage::Vertex,
        vertex_buffer_mappings: vertex_buffers.to_vec(),
        ..Default::default()
    };
    let (source, tinfo) = msl::write_string(&module, &info, &options, &pipeline).map_err(|e| backend_err(e.to_string()))?;

    let (ep_index, ep) = module
        .entry_points
        .iter()
        .enumerate()
        .find(|(_, e)| e.stage == stage && e.name == name)
        .ok_or_else(|| backend_err("entry point lost".into()))?;
    let entry_point = match tinfo.entry_point_names.get(ep_index) {
        Some(Ok(n)) => n.clone(),
        Some(Err(e)) => return Err(backend_err(e.to_string())),
        // (only the asked entry point is written, so it is the only name)
        None => tinfo.entry_point_names.iter().flatten().next().cloned().ok_or_else(|| backend_err("no entry point written".into()))?,
    };

    let ep_info = info.get_entry_point(ep_index);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).map_err(|e| backend_err(e.to_string()))?;
    let workgroup_memory = module
        .global_variables
        .iter()
        .filter(|(h, v)| v.space == naga::AddressSpace::WorkGroup && !ep_info[*h].is_empty())
        .map(|(_, v)| layouter[v.ty].size)
        .collect();
    let mut heap_bindings_used = 0;
    let mut uses_immediates = false;
    let sized_buffers = module
        .global_variables
        .iter()
        .filter(|(_, v)| module.types[v.ty].inner.needs_host_buffer_byte_size(&module.types))
        .filter_map(|(_, v)| v.binding.map(|b| (b.group, b.binding)))
        .collect();
    for (h, var) in module.global_variables.iter() {
        if ep_info[h].is_empty() {
            continue;
        }
        if var.space == naga::AddressSpace::Immediate {
            uses_immediates = true;
        }
        if let Some(rb) = var.binding
            && rb.group == HEAP_GROUP
        {
            heap_bindings_used |= 1 << rb.binding;
        }
    }
    Ok(Output {
        source,
        entry_point,
        heap_bindings_used,
        uses_immediates,
        sized_buffers,
        workgroup_size: ep.workgroup_size,
        workgroup_memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_and_roots_land_in_their_slots() {
        let m = Module::from_wgsl(
            "t",
            "struct C { k: vec4<f32> }\n@group(0) @binding(0) var<uniform> c: C;\n@group(2) @binding(1) var<storage, read> xs: array<f32>;\nstruct D { t: u32 }\nvar<immediate> d: D;\n@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return c.k * xs[0] * textureSample(mx_tex2d[d.t], mx_samplers[0], uv); }",
        )
        .unwrap();
        let layout = Layout::for_module(&m).unwrap();
        let out = write(&m, &layout, naga::ShaderStage::Fragment, None, &[], (3, 0), &[]).unwrap();
        assert!(out.source.contains("[[buffer(0)]]"), "{}", out.source);
        assert!(out.source.contains("[[buffer(6)]]"));
        assert!(out.source.contains("[[buffer(8)]]"));
        assert!(out.source.contains("[[buffer(10)]]"));
        assert!(out.source.contains("[[buffer(11)]]"));
        assert_eq!(out.heap_bindings_used, 1 << heap::TEX_2D | 1 << heap::SAMPLERS);
        assert!(out.uses_immediates);
    }

    #[test]
    fn overrides_are_applied() {
        let m = Module::from_wgsl("t", "override K: f32 = 1.0;\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(K); }").unwrap();
        let layout = Layout::for_module(&m).unwrap();
        let out = write(&m, &layout, naga::ShaderStage::Fragment, Some("fs"), &[("K", 0.25)], (3, 0), &[]).unwrap();
        assert!(out.source.contains("0.25"), "{}", out.source);
    }
}
#[cfg(test)]
#[test]
fn sizes_follow_declaration_order() {
    let m = Module::from_wgsl(
        "t",
        "@group(0) @binding(3) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<u32>;\n@group(0) @binding(0) var<storage, read> unused: array<u32>;\n@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(a[arrayLength(&a)-1u] + f32(b[2])); }\n@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(f32(unused[0])); }",
    )
    .unwrap();
    let l = Layout::for_module(&m).unwrap();
    let out = write(&m, &l, naga::ShaderStage::Fragment, None, &[], (3, 0), &[]).unwrap();
    assert_eq!(out.sized_buffers, vec![(0, 3), (0, 1)], "{}", out.source);
}

#[cfg(test)]
#[test]
fn ray_queries_get_a_buffer_slot() {
    let m = Module::from_wgsl(
        "t",
        "enable wgpu_ray_query;\n@group(0) @binding(1) var acc: acceleration_structure;\n@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(1) fn main() { var rq: ray_query; rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 10.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0))); rayQueryProceed(&rq); out[0] = rayQueryGetCommittedIntersection(&rq).instance_custom_data; }",
    )
    .unwrap();
    let l = Layout::for_module(&m).unwrap();
    assert_eq!(l.roots.get(&(0, 1)), Some(&RootSlot::AccelerationStructure(FIRST_ROOT_BUFFER_SLOT + 1)));
    let out = write(&m, &l, naga::ShaderStage::Compute, None, &[], (3, 0), &[]).unwrap();
    assert!(out.source.contains("instance_acceleration_structure"), "{}", out.source);
}

#[test]
fn each_entry_point_counts_only_the_roots_it_uses() {
    // 14 buffers in the module, 13 at most per entry point: as in wgpu, this compiles
    let decls: String = (0..14).map(|i| format!("@group(0) @binding({i}) var<uniform> u{i}: vec4<f32>;\n")).collect();
    let src = format!("{decls}@vertex fn vs() -> @builtin(position) vec4<f32> {{ return u0 + u13; }}\n@fragment fn fs() -> @location(0) vec4<f32> {{ return u1; }}");
    let m = Module::from_wgsl("t", &src).unwrap();
    assert!(Layout::for_module(&m).is_err());
    let vs = Layout::for_entry_point(&m, naga::ShaderStage::Vertex, "vs").unwrap();
    assert_eq!(vs.roots.keys().copied().collect::<Vec<_>>(), [(0, 0), (0, 13)]);
    assert_eq!(
        vs.roots[&(0, 13)],
        RootSlot::Buffer {
            slot: FIRST_ROOT_BUFFER_SLOT + 1,
            kind: RootKind::UniformBuffer
        }
    );
    let fs = Layout::for_entry_point(&m, naga::ShaderStage::Fragment, "fs").unwrap();
    assert_eq!(fs.roots.keys().copied().collect::<Vec<_>>(), [(0, 1)]);
}
