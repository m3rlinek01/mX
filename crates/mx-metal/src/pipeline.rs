//! Shader modules and pipelines.

use std::sync::Arc;

use mx_shader::msl;
use mx_shader::naga::ShaderStage;
use mx_types as t;
use objc2_metal::*;

use crate::{Device, Error, Mt, Po, Result, conv};

pub struct ShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    /// WGSL. The heap prelude (`mx_tex2d`, `mx_samplers`, ...) is added by mX.
    pub source: &'a str,
}

pub(crate) struct ModuleInner {
    pub module: mx_shader::Module,
}

#[derive(Clone)]
pub struct ShaderModule(pub(crate) Arc<ModuleInner>);

/// What a pipeline stage reads, for binding at draw time.
pub(crate) struct StageInfo {
    pub layout: msl::Layout,
    pub heap_used: u32,
    pub uses_immediates: bool,
    pub sized_buffers: Vec<(u32, u32)>,
    /// The vertex buffers (by `set_vertex_buffer` slot) whose sizes follow `sized_buffers`'
    /// at `SIZES_SLOT`, in this order: the vertex stage fetches its attributes itself.
    pub vertex_buffers: Vec<u32>,
    /// Bytes of each threadgroup-memory argument (`var<workgroup>`), by index.
    pub workgroup_memory: Vec<u32>,
}

pub(crate) struct RenderPipelineInner {
    pub state: Mt<Po<dyn MTLRenderPipelineState>>,
    pub depth_stencil: Option<Mt<Po<dyn MTLDepthStencilState>>>,
    pub primitive: MTLPrimitiveType,
    pub cull: MTLCullMode,
    pub winding: MTLWinding,
    pub clamp_depth: bool,
    pub bias: t::DepthBiasState,
    pub vertex: StageInfo,
    pub fragment: Option<StageInfo>,
    pub immediate_size: u32,
}

#[derive(Clone)]
pub struct RenderPipeline(pub(crate) Arc<RenderPipelineInner>);

pub(crate) struct ComputePipelineInner {
    pub state: Mt<Po<dyn MTLComputePipelineState>>,
    pub stage: StageInfo,
    pub workgroup_size: MTLSize,
    pub immediate_size: u32,
}

#[derive(Clone)]
pub struct ComputePipeline(pub(crate) Arc<ComputePipelineInner>);

/// The MSL version wgpu-hal compiles for on this OS (`private_caps.msl_version`): the newest
/// the OS has, from 3.0 (macOS 13, mX's minimum) to 3.2.
fn msl_version() -> ((u8, u8), MTLLanguageVersion) {
    if objc2::available!(macos = 15.0, ios = 18.0, tvos = 18.0, visionos = 2.0) {
        ((3, 2), MTLLanguageVersion::Version3_2)
    } else if objc2::available!(macos = 14.0, ios = 17.0, tvos = 17.0, visionos = 1.0) {
        ((3, 1), MTLLanguageVersion::Version3_1)
    } else {
        ((3, 0), MTLLanguageVersion::Version3_0)
    }
}

fn immediate_size(m: &mx_shader::Module) -> u32 {
    let module = &m.module;
    module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == mx_shader::naga::AddressSpace::Immediate)
        .map(|(_, v)| module.types[v.ty].inner.size(module.to_ctx()))
        .max()
        .unwrap_or(0)
}

impl Device {
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor<'_>) -> Result<ShaderModule> {
        let module = mx_shader::Module::from_wgsl(desc.label.unwrap_or("shader"), desc.source)?;
        Ok(ShaderModule(Arc::new(ModuleInner { module })))
    }

    fn function(
        &self,
        m: &ShaderModule,
        stage: ShaderStage,
        entry: Option<&str>,
        constants: &[(&str, f64)],
        vertex_buffers: &[msl::VertexBufferMapping],
    ) -> Result<(Po<dyn MTLFunction>, StageInfo, [u32; 3])> {
        let name = m.0.module.entry_point(stage, entry)?.name.clone();
        let layout = msl::Layout::for_entry_point(&m.0.module, stage, &name)?;
        let (version, lang) = msl_version();
        let out = msl::write(&m.0.module, &layout, stage, Some(&name), constants, version, vertex_buffers)?;
        log::trace!("{} ({stage:?}) MSL:\n{}", m.0.module.label, out.source);
        let options = MTLCompileOptions::new();
        // As wgpu: the OS's MSL version, invariance kept (a depth prepass and the main pass
        // must compute bit-identical positions), and Metal's default fast math left on, so the
        // shading matches what openOMSI drew on wgpu.
        options.setLanguageVersion(lang);
        options.setPreserveInvariance(true);
        let lib = self
            .shared
            .device
            .newLibraryWithSource_options_error(&crate::ns(&out.source), Some(&options))
            .map_err(|e| Error::Compile {
                label: m.0.module.label.clone(),
                message: e.localizedDescription().to_string(),
            })?;
        let f = lib.newFunctionWithName(&crate::ns(&out.entry_point)).ok_or_else(|| Error::Compile {
            label: m.0.module.label.clone(),
            message: format!("`{}` missing from the compiled library", out.entry_point),
        })?;
        let info = StageInfo {
            layout,
            heap_used: out.heap_bindings_used,
            uses_immediates: out.uses_immediates,
            sized_buffers: out.sized_buffers,
            vertex_buffers: Vec::new(),
            workgroup_memory: out.workgroup_memory,
        };
        Ok((f, info, out.workgroup_size))
    }

    /// Compile every entry point of `module` now (with its overrides' defaults), so shader
    /// errors and the driver's compile time show up at load instead of at the first draw.
    pub fn precompile(&self, module: &ShaderModule) -> Result<()> {
        for ep in &module.0.module.module.entry_points {
            if matches!(ep.stage, ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute) {
                self.function(module, ep.stage, Some(&ep.name), &[], &[])?;
            }
        }
        Ok(())
    }

    pub fn create_render_pipeline(&self, desc: &t::RenderPipelineDescriptor<'_, ShaderModule>) -> Result<RenderPipeline> {
        let label = desc.label.unwrap_or("render pipeline");
        let pipe_err = |message: String| Error::Pipeline { label: label.into(), message };
        let d = MTLRenderPipelineDescriptor::new();
        if let Some(l) = desc.label {
            d.setLabel(Some(&crate::ns(l)));
        }

        // vertex buffers: fetched by the vertex shader itself, as wgpu-hal does (see `msl::write`)
        if desc.vertex.buffers.len() as u32 > msl::MAX_VERTEX_BUFFERS {
            return Err(pipe_err(format!("{} vertex buffers; mX takes {}", desc.vertex.buffers.len(), msl::MAX_VERTEX_BUFFERS)));
        }
        let mappings: Vec<msl::VertexBufferMapping> = desc
            .vertex
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| msl::VertexBufferMapping {
                id: msl::vertex_buffer_slot(i as u32),
                // (a stride of 0: every vertex reads the same element, as in wgpu)
                stride: if b.array_stride > 0 {
                    b.array_stride as u32
                } else {
                    b.attributes.iter().map(|a| a.offset + a.format.size()).max().unwrap_or(0) as u32
                },
                step_mode: match (b.array_stride == 0, b.step_mode) {
                    (true, _) => msl::VertexBufferStepMode::Constant,
                    (false, t::VertexStepMode::Vertex) => msl::VertexBufferStepMode::ByVertex,
                    (false, t::VertexStepMode::Instance) => msl::VertexBufferStepMode::ByInstance,
                },
                attributes: b
                    .attributes
                    .iter()
                    .map(|a| msl::AttributeMapping {
                        shader_location: a.shader_location,
                        offset: a.offset as u32,
                        format: conv::naga_vertex_format(a.format),
                    })
                    .collect(),
            })
            .collect();
        let (vf, mut vinfo, _) = self.function(desc.vertex.module, ShaderStage::Vertex, desc.vertex.entry_point, desc.vertex.constants, &mappings)?;
        vinfo.vertex_buffers = (0..mappings.len() as u32).collect();
        d.setVertexFunction(Some(&vf));
        let fragment = match &desc.fragment {
            Some(fs) => {
                let (ff, finfo, _) = self.function(fs.module, ShaderStage::Fragment, fs.entry_point, fs.constants, &[])?;
                d.setFragmentFunction(Some(&ff));
                Some(finfo)
            }
            None => None,
        };

        // targets
        if let Some(fs) = &desc.fragment {
            for (i, target) in fs.targets.iter().enumerate() {
                let Some(target) = target else { continue };
                // SAFETY: `i` is below the 8 colour attachments (checked by Metal).
                let ca = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
                ca.setPixelFormat(conv::pixel_format(target.format));
                ca.setWriteMask(conv::write_mask(target.write_mask));
                if let Some(b) = target.blend {
                    ca.setBlendingEnabled(true);
                    ca.setSourceRGBBlendFactor(conv::blend_factor(b.color.src_factor));
                    ca.setDestinationRGBBlendFactor(conv::blend_factor(b.color.dst_factor));
                    ca.setRgbBlendOperation(conv::blend_op(b.color.operation));
                    ca.setSourceAlphaBlendFactor(conv::blend_factor(b.alpha.src_factor));
                    ca.setDestinationAlphaBlendFactor(conv::blend_factor(b.alpha.dst_factor));
                    ca.setAlphaBlendOperation(conv::blend_op(b.alpha.operation));
                }
            }
        }
        let depth_stencil = match &desc.depth_stencil {
            Some(ds) => {
                let pf = conv::pixel_format(ds.format);
                if ds.format.has_depth() {
                    d.setDepthAttachmentPixelFormat(pf);
                }
                if ds.format.has_stencil() {
                    d.setStencilAttachmentPixelFormat(pf);
                }
                let dsd = MTLDepthStencilDescriptor::new();
                dsd.setDepthCompareFunction(conv::compare(ds.depth_compare));
                dsd.setDepthWriteEnabled(ds.depth_write_enabled);
                if ds.format.has_stencil() {
                    let face = |f: &t::StencilFaceState| {
                        let s = MTLStencilDescriptor::new();
                        s.setStencilCompareFunction(conv::compare(f.compare));
                        s.setStencilFailureOperation(conv::stencil_op(f.fail_op));
                        s.setDepthFailureOperation(conv::stencil_op(f.depth_fail_op));
                        s.setDepthStencilPassOperation(conv::stencil_op(f.pass_op));
                        s.setReadMask(ds.stencil.read_mask);
                        s.setWriteMask(ds.stencil.write_mask);
                        s
                    };
                    dsd.setFrontFaceStencil(Some(&face(&ds.stencil.front)));
                    dsd.setBackFaceStencil(Some(&face(&ds.stencil.back)));
                }
                let s = self.shared.device.newDepthStencilStateWithDescriptor(&dsd).ok_or_else(|| pipe_err("depth-stencil state".into()))?;
                Some(Mt(s))
            }
            None => None,
        };
        d.setRasterSampleCount(desc.multisample.count as usize);
        d.setAlphaToCoverageEnabled(desc.multisample.alpha_to_coverage_enabled);
        // SAFETY: a valid topology class for the pipeline's primitives.
        unsafe { d.setInputPrimitiveTopology(conv::topology_class(desc.primitive.topology)) };

        let state = self
            .shared
            .device
            .newRenderPipelineStateWithDescriptor_error(&d)
            .map_err(|e| pipe_err(e.localizedDescription().to_string()))?;
        let immediate_size = immediate_size(&desc.vertex.module.0.module).max(desc.fragment.as_ref().map_or(0, |f| immediate_size(&f.module.0.module)));
        Ok(RenderPipeline(Arc::new(RenderPipelineInner {
            state: Mt(state),
            depth_stencil,
            primitive: conv::primitive_type(desc.primitive.topology),
            cull: conv::cull(desc.primitive.cull_mode),
            winding: conv::winding(desc.primitive.front_face),
            clamp_depth: desc.primitive.unclipped_depth,
            bias: desc.depth_stencil.map(|d| d.bias).unwrap_or_default(),
            vertex: vinfo,
            fragment,
            immediate_size,
        })))
    }

    pub fn create_compute_pipeline(&self, desc: &t::ComputePipelineDescriptor<'_, ShaderModule>) -> Result<ComputePipeline> {
        let label = desc.label.unwrap_or("compute pipeline");
        let (f, stage, wg) = self.function(desc.module, ShaderStage::Compute, desc.entry_point, desc.constants, &[])?;
        let state = self.shared.device.newComputePipelineStateWithFunction_error(&f).map_err(|e| Error::Pipeline {
            label: label.into(),
            message: e.localizedDescription().to_string(),
        })?;
        Ok(ComputePipeline(Arc::new(ComputePipelineInner {
            state: Mt(state),
            stage,
            workgroup_size: MTLSize {
                width: wg[0] as usize,
                height: wg[1] as usize,
                depth: wg[2] as usize,
            },
            immediate_size: immediate_size(&desc.module.0.module),
        })))
    }
}
