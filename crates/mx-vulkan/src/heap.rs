//! The bindless heap: descriptor set 0, one `UPDATE_AFTER_BIND | PARTIALLY_BOUND` array per
//! texture and sampler binding of the heap and one array of storage buffers that every buffer
//! view of a shader aliases, slot allocation with deferred reuse.

use ash::vk;
use mx_shader::heap;

use crate::{Error, Result, vk_err};

/// Indices of one kind and their deferred release.
struct Slots {
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    /// Released slots and the submission after whose completion they may be reused.
    deferred: Vec<(u64, u32)>,
}

impl Slots {
    fn new(capacity: u32) -> Slots {
        Slots {
            capacity,
            next: 0,
            free: Vec::new(),
            deferred: Vec::new(),
        }
    }

    fn alloc(&mut self, what: &'static str) -> Result<u32> {
        if let Some(s) = self.free.pop() {
            return Ok(s);
        }
        if self.next == self.capacity {
            return Err(Error::HeapFull(what));
        }
        self.next += 1;
        Ok(self.next - 1)
    }

    fn reclaim(&mut self, completed: u64) {
        let free = &mut self.free;
        self.deferred.retain(|&(after, slot)| {
            let done = after <= completed;
            if done {
                free.push(slot);
            }
            !done
        });
    }
}

pub(crate) struct Heap {
    pool: vk::DescriptorPool,
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    textures: Slots,
    samplers: Slots,
    buffers: Slots,
    in_use: usize,
    buffers_in_use: usize,
}

/// The descriptor bindings of the heap's set: textures, samplers, then the buffers.
const SET_BINDINGS: u32 = heap::BUFFERS + 1;

impl Heap {
    /// The heap with up to `textures` / `samplers` / `buffers` slots: fewer where the driver says a
    /// layout that large is not supported (above `maxPerSetDescriptors` a layout is valid only
    /// where `vkGetDescriptorSetLayoutSupport` says so). `portability`: MoltenVK, whose answer
    /// only compares with its `maxPerSetDescriptors` (1212) although its argument buffers
    /// take far larger arrays: the size asked for is kept there.
    pub fn new(device: &ash::Device, textures: u32, samplers: u32, buffers: u32, update_unused_while_pending: bool, portability: bool) -> Result<Heap> {
        let mut flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        if update_unused_while_pending {
            flags |= vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
        }
        let bindings_for = |textures: u32, samplers: u32, buffers: u32| -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
            (0..SET_BINDINGS)
                .map(|b| {
                    let (ty, count) = if heap::is_texture(b) {
                        (vk::DescriptorType::SAMPLED_IMAGE, textures)
                    } else if heap::is_sampler(b) {
                        (vk::DescriptorType::SAMPLER, samplers)
                    } else {
                        (vk::DescriptorType::STORAGE_BUFFER, buffers)
                    };
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(ty)
                        .descriptor_count(count)
                        .stage_flags(vk::ShaderStageFlags::ALL)
                })
                .collect()
        };
        let binding_flags = vec![flags; SET_BINDINGS as usize];
        let supported = |textures: u32, samplers: u32, buffers: u32| -> bool {
            let bindings = bindings_for(textures, samplers, buffers);
            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
            let info = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .bindings(&bindings)
                .push_next(&mut flags_info);
            let mut support = vk::DescriptorSetLayoutSupport::default();
            // SAFETY: a plain query (Vulkan 1.1).
            unsafe { device.get_descriptor_set_layout_support(&info, &mut support) };
            support.supported != 0
        };
        let (mut textures, mut samplers, mut buffers) = (textures, samplers, buffers);
        if !supported(textures, samplers, buffers) {
            if portability {
                log::info!("mX: MoltenVK reports the heap layout ({textures} textures, {buffers} buffers) as unsupported by its maxPerSetDescriptors; kept, as its argument buffers take it");
            } else {
                while !supported(textures, samplers, buffers) {
                    if textures <= 256 {
                        return Err(crate::Error::Unsupported(format!("a bindless heap of {textures} textures")));
                    }
                    textures /= 2;
                    buffers = buffers.min(textures).max(64);
                    samplers = samplers.min(textures).max(16);
                }
                log::info!("mX: the driver takes a heap of {textures} textures, {samplers} samplers and {buffers} buffers");
            }
        }
        let bindings = bindings_for(textures, samplers, buffers);
        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut flags_info);
        let tex_bindings = heap::SAMPLERS;
        debug_assert_eq!((heap::SAMPLERS_CMP, heap::BUFFERS), (heap::SAMPLERS + 1, heap::SAMPLERS + 2));
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: textures * tex_bindings,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: samplers * 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: buffers,
            },
        ];
        // SAFETY: plain creation of the heap's layout, pool and set.
        unsafe {
            let layout = device.create_descriptor_set_layout(&info, None).map_err(|e| vk_err("heap layout", e))?;
            let pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| vk_err("heap pool", e))?;
            let layouts = [layout];
            let set = device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))
                .map_err(|e| vk_err("heap descriptor set", e))?[0];
            Ok(Heap {
                pool,
                layout,
                set,
                textures: Slots::new(textures),
                samplers: Slots::new(samplers),
                buffers: Slots::new(buffers),
                in_use: 0,
                buffers_in_use: 0,
            })
        }
    }

    /// The slots the heap has: (textures, samplers, buffers).
    pub fn capacity(&self) -> (u32, u32, u32) {
        (self.textures.capacity, self.samplers.capacity, self.buffers.capacity)
    }

    /// Put a view in the heap: written into every heap binding in `bindings` (bit per
    /// `heap::*`) at its slot.
    pub fn add_texture(&mut self, device: &ash::Device, view: vk::ImageView, bindings: u32) -> Result<u32> {
        let slot = self.textures.alloc("textures")?;
        let info = [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL)];
        let writes: Vec<vk::WriteDescriptorSet> = (0..heap::SAMPLERS)
            .filter(|b| bindings & (1 << b) != 0)
            .map(|b| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(b)
                    .dst_array_element(slot)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&info)
            })
            .collect();
        // SAFETY: the slot is not read by any pending work (fresh, or past its deferred
        // release); the heap lock serialises updates of the set.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        self.in_use += 1;
        Ok(slot)
    }

    /// Take a view out of the heap. Its slot is reused once submission `after` is done.
    pub fn remove_texture(&mut self, slot: u32, after: u64) {
        self.textures.deferred.push((after, slot));
        self.in_use -= 1;
    }

    pub fn add_sampler(&mut self, device: &ash::Device, sampler: vk::Sampler, compare: bool) -> Result<u32> {
        let slot = self.samplers.alloc("samplers")?;
        let info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(if compare { heap::SAMPLERS_CMP } else { heap::SAMPLERS })
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&info)];
        // SAFETY: as in `add_texture`.
        unsafe { device.update_descriptor_sets(&write, &[]) };
        Ok(slot)
    }

    pub fn remove_sampler(&mut self, slot: u32, after: u64) {
        self.samplers.deferred.push((after, slot));
    }

    /// Put `range` bytes of `buffer` from `offset` in the heap, as a storage buffer.
    pub fn add_buffer(&mut self, device: &ash::Device, buffer: vk::Buffer, offset: u64, range: u64) -> Result<u32> {
        let slot = self.buffers.alloc("buffers")?;
        let info = [vk::DescriptorBufferInfo::default().buffer(buffer).offset(offset).range(range)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(heap::BUFFERS)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info)];
        // SAFETY: as in `add_texture`.
        unsafe { device.update_descriptor_sets(&write, &[]) };
        self.buffers_in_use += 1;
        Ok(slot)
    }

    /// Take a buffer view out of the heap. Its slot is reused once submission `after` is done.
    pub fn remove_buffer(&mut self, slot: u32, after: u64) {
        self.buffers.deferred.push((after, slot));
        self.buffers_in_use -= 1;
    }

    pub fn buffers_in_use(&self) -> usize {
        self.buffers_in_use
    }

    /// Everything released before submission `completed` finished is free again.
    pub fn reclaim(&mut self, completed: u64) {
        self.textures.reclaim(completed);
        self.samplers.reclaim(completed);
        self.buffers.reclaim(completed);
    }

    pub fn textures_in_use(&self) -> usize {
        self.in_use
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: called once, when the device is idle and going away.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

/// The heap bindings a view of `dimension` and `format` is reachable through.
pub(crate) fn bindings_for(dimension: mx_types::TextureViewDimension, format: mx_types::TextureFormat, aspect: mx_types::TextureAspect, samples: u32) -> u32 {
    use mx_types::{TextureSampleType as S, TextureViewDimension as D};
    if samples > 1 {
        return 0;
    }
    let sample = if format.has_depth() && aspect == mx_types::TextureAspect::StencilOnly {
        S::Uint
    } else {
        format.sample_type()
    };
    match (dimension, sample) {
        (D::D2, S::Float { .. }) => 1 << heap::TEX_2D,
        (D::D2, S::Depth) => 1 << heap::TEX_2D | 1 << heap::TEX_DEPTH,
        (D::D2, S::Uint) => 1 << heap::TEX_2D_U32,
        (D::D2Array, S::Float { .. } | S::Depth) => 1 << heap::TEX_2D_ARRAY,
        (D::Cube, S::Float { .. } | S::Depth) => 1 << heap::TEX_CUBE,
        (D::D3, S::Float { .. }) => 1 << heap::TEX_3D,
        _ => 0,
    }
}
