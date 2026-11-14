//! Descriptor heaps: the shader-visible heaps (bindless slots, the sampler index buffer of the
//! heap group, the ring root tables are copied into) and the CPU-only pools views and bind
//! groups keep their descriptors in.
//!
//! Shader-visible CBV/SRV/UAV heap:
//!
//! | descriptors | use |
//! |-------------|-----|
//! | `0 .. T` | bindless texture slots (`TextureView::index`) |
//! | `T` | the heap group's sampler index buffer (the identity: entry `i` holds `i`) |
//! | `T + 1 ..` | blocks of [`BLOCK`] descriptors for root tables, one encoder at a time |

use windows::Win32::Graphics::Direct3D12::*;

use crate::{Error, Result, Texture, hr_err};

/// Descriptors per root-table block.
pub const BLOCK: u32 = 1024;
/// Root-table blocks in the shader-visible heap.
pub const BLOCKS: u32 = 128;
/// D3D12's largest shader-visible CBV/SRV/UAV heap on resource binding tiers 1 and 2.
pub const MAX_VIEW_HEAP: u32 = 1_000_000;

/// One allocator of slots with deferred reuse.
struct Slots {
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    /// Released slots and the submission after whose completion they may be reused.
    deferred: Vec<(u64, u32)>,
    in_use: usize,
}

impl Slots {
    fn new(capacity: u32) -> Slots {
        Slots {
            capacity,
            next: 0,
            free: Vec::new(),
            deferred: Vec::new(),
            in_use: 0,
        }
    }

    fn alloc(&mut self, what: &'static str) -> Result<u32> {
        let s = match self.free.pop() {
            Some(s) => s,
            None if self.next == self.capacity => return Err(Error::HeapFull(what)),
            None => {
                self.next += 1;
                self.next - 1
            }
        };
        self.in_use += 1;
        Ok(s)
    }

    fn release(&mut self, slot: u32, after: u64) {
        self.in_use -= 1;
        self.deferred.push((after, slot));
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

/// The two shader-visible heaps. Immutable after creation; descriptors are written into them
/// through their CPU handles (slots no submission can be reading).
pub(crate) struct Views {
    pub view_heap: ID3D12DescriptorHeap,
    pub view_cpu: usize,
    pub view_gpu: u64,
    pub view_inc: u32,
    pub sampler_heap: ID3D12DescriptorHeap,
    pub sampler_cpu: usize,
    pub sampler_gpu: u64,
    pub sampler_inc: u32,
    /// Bindless texture slots (`T`).
    pub textures: u32,
}

impl Views {
    pub fn new(device: &ID3D12Device, textures: u32) -> Result<Views> {
        let heap = |ty, n: u32, what: &str| -> Result<ID3D12DescriptorHeap> {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: ty,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: a plain creation from a filled description.
            let h: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| hr_err(what, e))?;
            crate::set_name(&h, what);
            Ok(h)
        };
        let view_heap = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, textures + 1 + BLOCK * BLOCKS, "mX view heap")?;
        let sampler_heap = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, mx_shader::hlsl::SAMPLER_HEAP_SIZE, "mX sampler heap")?;
        // SAFETY: plain getters on live objects.
        unsafe {
            Ok(Views {
                view_cpu: view_heap.GetCPUDescriptorHandleForHeapStart().ptr,
                view_gpu: view_heap.GetGPUDescriptorHandleForHeapStart().ptr,
                view_inc: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                sampler_cpu: sampler_heap.GetCPUDescriptorHandleForHeapStart().ptr,
                sampler_gpu: sampler_heap.GetGPUDescriptorHandleForHeapStart().ptr,
                sampler_inc: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER),
                view_heap,
                sampler_heap,
                textures,
            })
        }
    }

    pub fn view_cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.view_cpu + index as usize * self.view_inc as usize,
        }
    }

    pub fn view_gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.view_gpu + index as u64 * self.view_inc as u64,
        }
    }

    pub fn sampler_cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.sampler_cpu + index as usize * self.sampler_inc as usize,
        }
    }

    /// The heap group's sampler index buffer descriptor.
    pub fn sampler_index_slot(&self) -> u32 {
        self.textures
    }

    /// The first descriptor of root-table block `b`.
    pub fn block_start(&self, b: u32) -> u32 {
        self.textures + 1 + b * BLOCK
    }
}

/// Slot bookkeeping of the bindless heap and the root-table blocks.
pub(crate) struct Heap {
    textures: Slots,
    samplers: Slots,
    /// Textures whose heap view was released but which a submission may still read.
    retired: Vec<(u64, Texture)>,
    free_blocks: Vec<u32>,
}

impl Heap {
    pub fn new(textures: u32, samplers: u32) -> Heap {
        Heap {
            textures: Slots::new(textures),
            samplers: Slots::new(samplers),
            retired: Vec::new(),
            free_blocks: (0..BLOCKS).rev().collect(),
        }
    }

    pub fn alloc_texture(&mut self) -> Result<u32> {
        self.textures.alloc("textures")
    }

    /// Release a texture slot (reused after submission `after` is done) and keep the texture
    /// alive until then.
    pub fn release_texture(&mut self, slot: u32, after: u64, texture: Texture) {
        self.textures.release(slot, after);
        self.retired.push((after, texture));
    }

    pub fn alloc_sampler(&mut self) -> Result<u32> {
        self.samplers.alloc("samplers")
    }

    pub fn release_sampler(&mut self, slot: u32, after: u64) {
        self.samplers.release(slot, after);
    }

    pub fn alloc_block(&mut self) -> Option<u32> {
        self.free_blocks.pop()
    }

    /// Blocks whose command lists finished (or were never submitted).
    pub fn free_blocks(&mut self, blocks: impl IntoIterator<Item = u32>) {
        self.free_blocks.extend(blocks);
    }

    /// Everything released before submission `completed` finished is free again. Returns the
    /// textures to drop outside the lock.
    pub fn reclaim(&mut self, completed: u64) -> Vec<Texture> {
        self.textures.reclaim(completed);
        self.samplers.reclaim(completed);
        let mut done = Vec::new();
        self.retired.retain_mut(|(after, t)| {
            if *after <= completed {
                done.push(t.clone());
                false
            } else {
                true
            }
        });
        done
    }

    pub fn textures_in_use(&self) -> usize {
        self.textures.in_use
    }
}

/// A CPU-only descriptor (in a [`CpuPool`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cpu(pub usize);

impl Cpu {
    pub fn handle(self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.0 }
    }
}

/// Non-shader-visible descriptors of one type, in chunks that are never freed.
pub(crate) struct CpuPool {
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
    inc: usize,
    chunks: Vec<ID3D12DescriptorHeap>,
    free: Vec<usize>,
}

const CHUNK: u32 = 256;

impl CpuPool {
    pub fn new(device: &ID3D12Device, ty: D3D12_DESCRIPTOR_HEAP_TYPE) -> CpuPool {
        CpuPool {
            ty,
            // SAFETY: a plain getter.
            inc: unsafe { device.GetDescriptorHandleIncrementSize(ty) } as usize,
            chunks: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn alloc(&mut self, device: &ID3D12Device) -> Result<Cpu> {
        if let Some(p) = self.free.pop() {
            return Ok(Cpu(p));
        }
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: self.ty,
            NumDescriptors: CHUNK,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: a plain creation from a filled description.
        let h: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&desc) }.map_err(|e| hr_err("CPU descriptor heap", e))?;
        // SAFETY: a plain getter.
        let start = unsafe { h.GetCPUDescriptorHandleForHeapStart() }.ptr;
        self.chunks.push(h);
        self.free.extend((1..CHUNK as usize).rev().map(|i| start + i * self.inc));
        Ok(Cpu(start))
    }

    /// Return a descriptor. CPU descriptors are copied (into tables) or read (render targets)
    /// when a command is recorded, so they may be reused at once.
    pub fn free(&mut self, d: Cpu) {
        self.free.push(d.0);
    }
}

/// The CPU-only pools of a device.
pub(crate) struct CpuPools {
    pub rtv: CpuPool,
    pub dsv: CpuPool,
    pub view: CpuPool,
}
