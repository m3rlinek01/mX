//! Resource states within one command list.
//!
//! Every resource has a home state it is in whenever no command list is running (and, for
//! textures, between operations): `ALL_SHADER_RESOURCE` for textures the heap may read,
//! `COMMON` otherwise, or a fixed state for upload / readback buffers. A command list starts
//! with everything at home, moves subresources to the states its commands need (merging read
//! states), and puts textures back after each pass, copy or dispatch, buffers at the end.
//!
//! Where the driver has enhanced barriers, each state change is recorded as one: a buffer
//! barrier (synchronisation and access only: buffers have no layout) or a texture barrier with
//! the layouts the states stand for, and the UAV barrier as a global barrier of the shading
//! stages. Otherwise as legacy `ResourceBarrier` transitions.

use std::collections::HashMap;
use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// What the tracker needs to know about a resource.
#[derive(Clone)]
pub(crate) struct Track {
    pub raw: ID3D12Resource,
    pub home: D3D12_RESOURCE_STATES,
    /// Upload and readback buffers never change state.
    pub fixed: bool,
    pub mips: u32,
    pub layers: u32,
    pub planes: u32,
    pub is_texture: bool,
}

impl Track {
    pub fn subresources(&self) -> u32 {
        self.mips * self.layers * self.planes
    }

    pub fn subresource(&self, mip: u32, layer: u32, plane: u32) -> u32 {
        mip + layer * self.mips + plane * self.mips * self.layers
    }
}

/// A set of subresources: mips × layers × planes.
#[derive(Clone, Debug)]
pub(crate) struct Range {
    pub mips: std::ops::Range<u32>,
    pub layers: std::ops::Range<u32>,
    pub planes: std::ops::Range<u32>,
}

struct Entry {
    track: Track,
    states: Vec<D3D12_RESOURCE_STATES>,
}

const READ_ONLY: i32 = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0
    | D3D12_RESOURCE_STATE_INDEX_BUFFER.0
    | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
    | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0
    | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0
    | D3D12_RESOURCE_STATE_COPY_SOURCE.0
    | D3D12_RESOURCE_STATE_DEPTH_READ.0
    | D3D12_RESOURCE_STATE_RESOLVE_SOURCE.0;

fn read_only(s: D3D12_RESOURCE_STATES) -> bool {
    s.0 != 0 && s.0 & !READ_ONLY == 0
}

/// Whether the resource reaches `state` without a barrier: a buffer in `COMMON` is promoted to
/// a read state on first use (and decays back when the list is done); with enhanced barriers a
/// buffer has no layout, and its first access in a list waits for nothing. Every command list
/// starts with each buffer at home (`COMMON`), so a barrier here came between two draws for
/// each vertex, index and uniform buffer first drawn from in the list: about 500 a frame in
/// openOMSI, each one holding the GPU up. Changes from the promoted state (another read state,
/// a write) are barriers, as before.
fn implicit(is_texture: bool, cur: D3D12_RESOURCE_STATES, state: D3D12_RESOURCE_STATES) -> bool {
    !is_texture && cur == D3D12_RESOURCE_STATE_COMMON && read_only(state)
}

#[derive(Default)]
pub(crate) struct Tracker {
    entries: HashMap<usize, Entry>,
    barriers: Vec<D3D12_RESOURCE_BARRIER>,
    /// Record enhanced barriers instead (see the module's documentation).
    enhanced: bool,
    buffer_barriers: Vec<D3D12_BUFFER_BARRIER>,
    texture_barriers: Vec<D3D12_TEXTURE_BARRIER>,
    global_barriers: Vec<D3D12_GLOBAL_BARRIER>,
    /// Textures the current operation uses (see [`Tracker::end_scope`]).
    scope: Vec<usize>,
    /// A buffer changed state since [`Tracker::buffers_settled`].
    buffers_moved: bool,
}

fn key(r: &ID3D12Resource) -> usize {
    r.as_raw() as usize
}

impl Tracker {
    pub fn new(enhanced: bool) -> Tracker {
        Tracker { enhanced, ..Default::default() }
    }

    /// Need `state` on `range` of the resource (the whole of it with `None`).
    pub fn need(&mut self, t: &Track, range: Option<Range>, state: D3D12_RESOURCE_STATES) {
        if t.fixed {
            return;
        }
        let k = key(&t.raw);
        let e = self.entries.entry(k).or_insert_with(|| Entry {
            states: vec![t.home; t.subresources() as usize],
            track: t.clone(),
        });
        if t.is_texture && !self.scope.contains(&k) {
            self.scope.push(k);
        }
        let subs: Vec<u32> = match range {
            None => (0..t.subresources()).collect(),
            Some(r) => {
                let mut v = Vec::new();
                for p in r.planes.clone() {
                    for l in r.layers.clone() {
                        for m in r.mips.clone() {
                            v.push(t.subresource(m, l, p));
                        }
                    }
                }
                v
            }
        };
        let mut changes: Vec<(u32, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> = Vec::new();
        for s in subs {
            let cur = e.states[s as usize];
            // (not a move `buffers_moved` waits for: a buffer only reaches a read state here)
            if implicit(t.is_texture, cur, state) {
                e.states[s as usize] = state;
                continue;
            }
            let next = if cur == state || (read_only(cur) && read_only(state) && cur.0 & state.0 == state.0) {
                continue;
            } else if read_only(cur) && read_only(state) {
                D3D12_RESOURCE_STATES(cur.0 | state.0)
            } else {
                state
            };
            changes.push((s, cur, next));
            e.states[s as usize] = next;
        }
        if !t.is_texture && !changes.is_empty() {
            self.buffers_moved = true;
        }
        if changes.is_empty() {
            return;
        }
        let (raw, is_texture, total) = (e.track.raw.clone(), e.track.is_texture, e.states.len() as u32);
        self.push(&raw, is_texture, total, &changes);
    }

    fn push(&mut self, raw: &ID3D12Resource, is_texture: bool, total: u32, changes: &[(u32, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)]) {
        if !self.enhanced {
            push_changes(&mut self.barriers, raw, total, changes);
            return;
        }
        if changes.is_empty() {
            return;
        }
        let whole = changes.len() as u32 == total && changes.iter().all(|&(_, b, a)| (b, a) == (changes[0].1, changes[0].2));
        if !is_texture {
            let (_, b, a) = changes[0];
            let (before, after) = (enhanced(b), enhanced(a));
            self.buffer_barriers.push(D3D12_BUFFER_BARRIER {
                SyncBefore: before.0,
                SyncAfter: after.0,
                AccessBefore: before.1,
                AccessAfter: after.1,
                // SAFETY: the tracker's entry owns `raw` until the list is retired.
                pResource: unsafe { crate::borrow(raw) },
                Offset: 0,
                Size: u64::MAX,
            });
            return;
        }
        let texture = |sub: u32, b, a| {
            let (before, after) = (enhanced(b), enhanced(a));
            D3D12_TEXTURE_BARRIER {
                SyncBefore: before.0,
                SyncAfter: after.0,
                AccessBefore: before.1,
                AccessAfter: after.1,
                LayoutBefore: before.2,
                LayoutAfter: after.2,
                // SAFETY: as above.
                pResource: unsafe { crate::borrow(raw) },
                // (`NumMipLevels` 0: `IndexOrFirstMipLevel` is a subresource index, as in
                // legacy barriers)
                Subresources: D3D12_BARRIER_SUBRESOURCE_RANGE {
                    IndexOrFirstMipLevel: sub,
                    NumMipLevels: 0,
                    FirstArraySlice: 0,
                    NumArraySlices: 0,
                    FirstPlane: 0,
                    NumPlanes: 0,
                },
                Flags: D3D12_TEXTURE_BARRIER_FLAG_NONE,
            }
        };
        if whole {
            let (_, b, a) = changes[0];
            self.texture_barriers.push(texture(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, b, a));
        } else {
            self.texture_barriers.extend(changes.iter().map(|&(s, b, a)| texture(s, b, a)));
        }
    }

    /// Whether a buffer changed state since the last call (see `need_heap_buffers`).
    pub fn buffers_settled(&mut self) -> bool {
        !std::mem::take(&mut self.buffers_moved)
    }

    /// Put every texture back in its home state (the end of a pass or copy).
    pub fn restore_textures(&mut self) {
        self.scope.clear();
        self.restore(|e| e.track.is_texture);
    }

    /// Start of an operation (a draw's or dispatch's bindings): textures it does not ask for
    /// with [`Tracker::need`] before [`Tracker::end_scope`] go back home then.
    pub fn begin_scope(&mut self) {
        self.scope.clear();
    }

    /// Return textures the operation did not use to their home state.
    pub fn end_scope(&mut self) {
        let scope = std::mem::take(&mut self.scope);
        self.restore(|e| e.track.is_texture && !scope.contains(&key(&e.track.raw)));
        self.scope = scope;
    }

    /// Everything back home (the end of a command list).
    pub fn restore_all(&mut self) {
        self.scope.clear();
        self.restore(|_| true);
    }

    fn restore(&mut self, which: impl Fn(&Entry) -> bool) {
        let mut changes_out = Vec::new();
        for e in self.entries.values_mut() {
            if !which(e) {
                continue;
            }
            let home = e.track.home;
            let changes: Vec<_> = e.states.iter().enumerate().filter(|(_, s)| **s != home).map(|(i, s)| (i as u32, *s, home)).collect();
            e.states.fill(home);
            let (raw, is_texture, total) = (e.track.raw.clone(), e.track.is_texture, e.states.len() as u32);
            changes_out.push((raw, is_texture, total, changes));
        }
        for (raw, is_texture, total, changes) in changes_out {
            self.push(&raw, is_texture, total, &changes);
        }
    }

    /// A UAV barrier on everything (orders storage writes before the next reads and writes).
    pub fn uav_barrier(&mut self) {
        if self.enhanced {
            self.global_barriers.push(D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL_SHADING,
                SyncAfter: D3D12_BARRIER_SYNC_ALL_SHADING,
                AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            });
            return;
        }
        self.barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }),
            },
        });
    }

    /// Record the barriers asked for so far.
    pub fn flush(&mut self, list: &ID3D12GraphicsCommandList) {
        if self.enhanced && !(self.global_barriers.is_empty() && self.buffer_barriers.is_empty() && self.texture_barriers.is_empty()) {
            let list7: ID3D12GraphicsCommandList7 = list.cast().expect("enhanced barriers need ID3D12GraphicsCommandList7");
            let mut groups = Vec::new();
            if !self.global_barriers.is_empty() {
                groups.push(D3D12_BARRIER_GROUP {
                    Type: D3D12_BARRIER_TYPE_GLOBAL,
                    NumBarriers: self.global_barriers.len() as u32,
                    Anonymous: D3D12_BARRIER_GROUP_0 {
                        pGlobalBarriers: self.global_barriers.as_ptr(),
                    },
                });
            }
            if !self.buffer_barriers.is_empty() {
                groups.push(D3D12_BARRIER_GROUP {
                    Type: D3D12_BARRIER_TYPE_BUFFER,
                    NumBarriers: self.buffer_barriers.len() as u32,
                    Anonymous: D3D12_BARRIER_GROUP_0 {
                        pBufferBarriers: self.buffer_barriers.as_ptr(),
                    },
                });
            }
            if !self.texture_barriers.is_empty() {
                groups.push(D3D12_BARRIER_GROUP {
                    Type: D3D12_BARRIER_TYPE_TEXTURE,
                    NumBarriers: self.texture_barriers.len() as u32,
                    Anonymous: D3D12_BARRIER_GROUP_0 {
                        pTextureBarriers: self.texture_barriers.as_ptr(),
                    },
                });
            }
            // SAFETY: the barriers point at resources the tracker keeps alive.
            unsafe { list7.Barrier(&groups) };
            self.global_barriers.clear();
            self.buffer_barriers.clear();
            self.texture_barriers.clear();
        }
        if !self.barriers.is_empty() {
            // SAFETY: the barriers point at resources the tracker keeps alive.
            unsafe { list.ResourceBarrier(&self.barriers) };
            self.barriers.clear();
        }
    }
}

fn push_changes(out: &mut Vec<D3D12_RESOURCE_BARRIER>, raw: &ID3D12Resource, total: u32, changes: &[(u32, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)]) {
    if changes.is_empty() {
        return;
    }
    let transition = |sub: u32, before, after| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the tracker's entry owns `raw` until the list is retired.
                pResource: unsafe { crate::borrow(raw) },
                Subresource: sub,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    let (_, b0, a0) = changes[0];
    if changes.len() as u32 == total && changes.iter().all(|&(_, b, a)| b == b0 && a == a0) {
        out.push(transition(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, b0, a0));
    } else {
        out.extend(changes.iter().map(|&(s, b, a)| transition(s, b, a)));
    }
}

/// The synchronisation, access and (for textures) layout a legacy state stands for, in
/// enhanced barriers. Read states merged by the tracker merge here too; a texture in several
/// read states is in `GENERIC_READ` (or `DEPTH_STENCIL_READ` when one of them is the depth
/// read). Home (`COMMON`) waits for and is reached by everything, as a legacy transition from
/// or to it does.
fn enhanced(state: D3D12_RESOURCE_STATES) -> (D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT) {
    use D3D12_RESOURCE_STATES as St;
    if state == D3D12_RESOURCE_STATE_COMMON {
        return (D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_LAYOUT_COMMON);
    }
    let table: [(St, D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS, D3D12_BARRIER_LAYOUT); 14] = [
        (
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_BARRIER_SYNC_ALL_SHADING,
            D3D12_BARRIER_ACCESS(D3D12_BARRIER_ACCESS_VERTEX_BUFFER.0 | D3D12_BARRIER_ACCESS_CONSTANT_BUFFER.0),
            D3D12_BARRIER_LAYOUT_GENERIC_READ,
        ),
        (
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            D3D12_BARRIER_SYNC_INDEX_INPUT,
            D3D12_BARRIER_ACCESS_INDEX_BUFFER,
            D3D12_BARRIER_LAYOUT_GENERIC_READ,
        ),
        (
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_BARRIER_SYNC_RENDER_TARGET,
            D3D12_BARRIER_ACCESS_RENDER_TARGET,
            D3D12_BARRIER_LAYOUT_RENDER_TARGET,
        ),
        (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_BARRIER_SYNC_ALL_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
        ),
        (
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
            D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
        ),
        (
            D3D12_RESOURCE_STATE_DEPTH_READ,
            D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ,
            D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ,
        ),
        (
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_BARRIER_SYNC_NON_PIXEL_SHADING,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
        ),
        (
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_BARRIER_SYNC_PIXEL_SHADING,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
        ),
        (
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
            D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            D3D12_BARRIER_LAYOUT_GENERIC_READ,
        ),
        (D3D12_RESOURCE_STATE_COPY_DEST, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST),
        (
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_COPY_SOURCE,
            D3D12_BARRIER_LAYOUT_COPY_SOURCE,
        ),
        (
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
            D3D12_BARRIER_SYNC_RESOLVE,
            D3D12_BARRIER_ACCESS_RESOLVE_DEST,
            D3D12_BARRIER_LAYOUT_RESOLVE_DEST,
        ),
        (
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            D3D12_BARRIER_SYNC_RESOLVE,
            D3D12_BARRIER_ACCESS_RESOLVE_SOURCE,
            D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE,
        ),
        (
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_BARRIER_SYNC_ALL_SHADING,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
            D3D12_BARRIER_LAYOUT_COMMON,
        ),
    ];
    let (mut sync, mut access, mut layouts) = (0, 0, Vec::new());
    for (st, sy, ac, la) in table {
        if state.0 & st.0 == st.0 {
            sync |= sy.0;
            access |= ac.0;
            layouts.push(la);
        }
    }
    let layout = match layouts.as_slice() {
        [one] => *one,
        many if many.contains(&D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ) => D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ,
        _ if layouts.iter().all(|l| *l == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE) => D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
        _ => D3D12_BARRIER_LAYOUT_GENERIC_READ,
    };
    (D3D12_BARRIER_SYNC(sync), D3D12_BARRIER_ACCESS(access), layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_in_common_is_promoted_to_read_states_without_a_barrier() {
        let common = D3D12_RESOURCE_STATE_COMMON;
        for read in [
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            crate::conv::SHADER_READ,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        ] {
            assert!(implicit(false, common, read), "{read:?}");
            // (textures are not promoted: they rest in their own home states)
            assert!(!implicit(true, common, read), "{read:?}");
        }
        // writes, and anything from a state the buffer was already moved to, take a barrier
        assert!(!implicit(false, common, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        assert!(!implicit(false, common, D3D12_RESOURCE_STATE_COPY_DEST));
        assert!(!implicit(false, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, crate::conv::SHADER_READ));
    }
}
