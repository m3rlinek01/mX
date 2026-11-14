//! The bindless heap: argument buffers of `MTLResourceID`s, slot allocation with deferred
//! reuse, and the residency list the encoders hand to `useResources`.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_metal::*;

use crate::{Error, Mt, Po, Result};

/// One argument buffer and its slots.
struct Slots {
    buffer: Mt<Po<dyn MTLBuffer>>,
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    /// Released slots and the submission after whose completion they may be reused.
    deferred: Vec<(u64, u32)>,
}

impl Slots {
    fn new(device: &ProtocolObject<dyn MTLDevice>, capacity: u32, label: &str) -> Result<Slots> {
        let bytes = capacity as usize * 8;
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::OutOfMemory(label.into()))?;
        buffer.setLabel(Some(&crate::ns(label)));
        // (empty slots read as a null resource: undefined to sample, but never a stale one)
        // SAFETY: the buffer is `bytes` long and nothing uses it yet.
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, bytes) };
        Ok(Slots {
            buffer: Mt(buffer),
            capacity,
            next: 0,
            free: Vec::new(),
            deferred: Vec::new(),
        })
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

    fn write(&self, slot: u32, id: MTLResourceID) {
        debug_assert!(slot < self.capacity);
        // SAFETY: `MTLResourceID` is `#[repr(C)]` around one `u64`, the layout an argument
        // buffer of `texture2d`/`sampler` entries has on Tier 2 (Metal 3) devices.
        let raw: u64 = unsafe { std::mem::transmute(id) };
        // SAFETY: the slot is inside the buffer; the GPU reads only slots in use, and this one
        // is not (fresh or past its deferred release).
        unsafe { self.buffer.contents().as_ptr().cast::<u64>().add(slot as usize).write(raw) };
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
    textures: Slots,
    samplers: Slots,
    /// The textures in the heap, dense, for `useResources`.
    resident: Vec<Mt<Po<dyn MTLTexture>>>,
    /// Per texture slot: its place in `resident` (`u32::MAX`: none).
    dense_of: Vec<u32>,
    /// Per place in `resident`: its slot.
    slot_of: Vec<u32>,
    /// Textures released but maybe still read by the GPU, kept alive until then.
    retired: Vec<(u64, Mt<Po<dyn MTLTexture>>)>,
    /// Bumped whenever what encoders must declare changes, so they call `useResources` again.
    pub generation: u64,
    /// macOS 15+: every heap texture is in this residency set, which the queue keeps
    /// resident; encoders then declare (`useResources`) only the textures the GPU may be
    /// writing, which Metal's hazard tracking needs to know about. Without it (macOS 13 and
    /// 14) every encoder declares the whole heap, which costs more with every texture.
    set: Option<Mt<Po<dyn MTLResidencySet>>>,
    set_dirty: bool,
    /// Per slot: the texture the view belongs to (its `Arc` address) and whether passes may
    /// write it (render targets, storage textures, textures made outside mX).
    owner: HashMap<u32, (usize, bool)>,
    /// The heap's views of each texture.
    views_of: HashMap<usize, Vec<u32>>,
    /// Views passes may write: declared by every encoder.
    hot: HashMap<u32, Mt<Po<dyn MTLTexture>>>,
    /// Views of textures copied into, declared until the submission that wrote them is done.
    recent: Vec<(u64, Mt<Po<dyn MTLTexture>>)>,
}

impl Heap {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, textures: u32, samplers: u32) -> Result<Heap> {
        Ok(Heap {
            textures: Slots::new(device, textures, "mX texture heap")?,
            samplers: Slots::new(device, samplers, "mX sampler heap")?,
            resident: Vec::new(),
            dense_of: vec![u32::MAX; textures as usize],
            slot_of: Vec::new(),
            retired: Vec::new(),
            generation: 1,
            set: residency_set(device),
            set_dirty: false,
            owner: HashMap::new(),
            views_of: HashMap::new(),
            hot: HashMap::new(),
            recent: Vec::new(),
        })
    }

    /// Whether the heap is kept resident by a residency set (macOS 15+).
    pub fn residency_set(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.set.as_deref().map(|s| &**s)
    }

    /// Put view `tex` of texture `owner` (its `Arc` address) in the heap. `hot`: passes
    /// may write the texture.
    pub fn add_texture(&mut self, tex: &Po<dyn MTLTexture>, owner: usize, hot: bool) -> Result<u32> {
        let slot = self.textures.alloc("textures")?;
        self.textures.write(slot, tex.gpuResourceID());
        self.dense_of[slot as usize] = self.resident.len() as u32;
        self.slot_of.push(slot);
        self.resident.push(Mt(tex.clone()));
        self.owner.insert(slot, (owner, hot));
        self.views_of.entry(owner).or_default().push(slot);
        if hot {
            self.hot.insert(slot, Mt(tex.clone()));
        }
        if let Some(set) = &self.set {
            set.addAllocation(ProtocolObject::from_ref(&**tex));
            self.set_dirty = true;
        }
        self.generation += 1;
        Ok(slot)
    }

    /// Texture `owner` was written by a copy (or zeroed) in submission `until`: encoders
    /// declare its views until that submission is done.
    pub fn written(&mut self, owner: usize, until: u64) {
        if self.set.is_none() {
            return;
        }
        let Some(slots) = self.views_of.get(&owner) else { return };
        for &slot in slots {
            if self.owner.get(&slot).is_some_and(|(_, hot)| !hot) {
                let dense = self.dense_of[slot as usize] as usize;
                self.recent.push((until, self.resident[dense].clone()));
            }
        }
        self.generation += 1;
    }

    /// Changes to the residency set take effect for command buffers committed after this.
    pub fn commit_residency(&mut self) {
        if let (Some(set), true) = (&self.set, self.set_dirty) {
            set.commit();
            self.set_dirty = false;
        }
    }

    /// Take a texture out of the heap. Its slot is reused once submission `after` is done.
    pub fn remove_texture(&mut self, slot: u32, after: u64) {
        let dense = std::mem::replace(&mut self.dense_of[slot as usize], u32::MAX) as usize;
        let tex = self.resident.swap_remove(dense);
        self.slot_of.swap_remove(dense);
        if let Some(&moved) = self.slot_of.get(dense) {
            self.dense_of[moved as usize] = dense as u32;
        }
        if let Some((owner, _)) = self.owner.remove(&slot)
            && let Some(v) = self.views_of.get_mut(&owner)
        {
            v.retain(|s| *s != slot);
            if v.is_empty() {
                self.views_of.remove(&owner);
            }
        }
        self.hot.remove(&slot);
        self.retired.push((after, tex));
        self.textures.deferred.push((after, slot));
        self.generation += 1;
    }

    pub fn add_sampler(&mut self, s: &Po<dyn MTLSamplerState>) -> Result<u32> {
        let slot = self.samplers.alloc("samplers")?;
        self.samplers.write(slot, s.gpuResourceID());
        Ok(slot)
    }

    pub fn remove_sampler(&mut self, slot: u32, after: u64) {
        self.samplers.deferred.push((after, slot));
    }

    /// Everything released before submission `completed` finished is free again.
    pub fn reclaim(&mut self, completed: u64) {
        self.textures.reclaim(completed);
        self.samplers.reclaim(completed);
        let set = &self.set;
        let mut removed = false;
        self.retired.retain(|(after, tex)| {
            let keep = *after > completed;
            if !keep && let Some(set) = set {
                set.removeAllocation(ProtocolObject::from_ref(&***tex));
                removed = true;
            }
            keep
        });
        self.set_dirty |= removed;
        let before = self.recent.len();
        self.recent.retain(|(until, _)| *until > completed);
        if self.recent.len() != before {
            self.generation += 1;
        }
    }

    pub fn texture_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.textures.buffer
    }

    pub fn sampler_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.samplers.buffer
    }

    /// Declare the heap's textures to a render encoder: with a residency set, those passes
    /// may write (and the views of `written`, textures this command buffer copied into);
    /// without, all of them.
    pub fn use_on_render(&self, enc: &ProtocolObject<dyn MTLRenderCommandEncoder>, written: &[usize]) {
        let list = self.resource_list(written);
        if let Some(ptr) = NonNull::new(list.as_ptr().cast_mut()) {
            // SAFETY: `list` holds valid resources kept alive by `self.resident` (the caller
            // holds the heap lock).
            unsafe { enc.useResources_count_usage_stages(ptr, list.len(), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment) };
        }
    }

    pub fn use_on_compute(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, written: &[usize]) {
        let list = self.resource_list(written);
        if let Some(ptr) = NonNull::new(list.as_ptr().cast_mut()) {
            // SAFETY: as in `use_on_render`.
            unsafe { enc.useResources_count_usage(ptr, list.len(), MTLResourceUsage::Read) };
        }
    }

    fn resource_list(&self, written: &[usize]) -> Vec<NonNull<ProtocolObject<dyn MTLResource>>> {
        let ptr = |t: &Mt<Po<dyn MTLTexture>>| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&*t.0));
        if self.set.is_none() {
            return self.resident.iter().map(ptr).collect();
        }
        let mut list: Vec<_> = self.hot.values().map(ptr).chain(self.recent.iter().map(|(_, t)| ptr(t))).collect();
        for owner in written {
            for &slot in self.views_of.get(owner).into_iter().flatten() {
                list.push(ptr(&self.resident[self.dense_of[slot as usize] as usize]));
            }
        }
        list
    }

    pub fn textures_in_use(&self) -> usize {
        self.resident.len()
    }
}

/// A residency set for the heap where Metal has them (macOS 15+), unless
/// `MX_NO_RESIDENCY_SET` is set.
fn residency_set(device: &ProtocolObject<dyn MTLDevice>) -> Option<Mt<Po<dyn MTLResidencySet>>> {
    if std::env::var_os("MX_NO_RESIDENCY_SET").is_some() || !device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
        return None;
    }
    let d = MTLResidencySetDescriptor::new();
    d.setLabel(Some(&crate::ns("mX heap")));
    match device.newResidencySetWithDescriptor_error(&d) {
        Ok(set) => Some(Mt(set)),
        Err(e) => {
            log::warn!("mX: no residency set ({}); the heap is declared in every pass", e.localizedDescription());
            None
        }
    }
}
