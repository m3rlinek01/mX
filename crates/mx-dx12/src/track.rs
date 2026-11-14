//! Resource states within one command list.
//!
//! Every resource has a home state it is in whenever no command list is running (and, for
//! textures, between operations): `ALL_SHADER_RESOURCE` for textures the heap may read,
//! `COMMON` otherwise, or a fixed state for upload / readback buffers. A command list starts
//! with everything at home, moves subresources to the states its commands need (merging read
//! states), and puts textures back after each pass, copy or dispatch, buffers at the end.

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

#[derive(Default)]
pub(crate) struct Tracker {
    entries: HashMap<usize, Entry>,
    barriers: Vec<D3D12_RESOURCE_BARRIER>,
    /// Textures the current operation uses (see [`Tracker::end_scope`]).
    scope: Vec<usize>,
}

fn key(r: &ID3D12Resource) -> usize {
    r.as_raw() as usize
}

impl Tracker {
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
        push_changes(&mut self.barriers, &e.track.raw, e.states.len() as u32, &changes);
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
        for e in self.entries.values_mut() {
            if !which(e) {
                continue;
            }
            let home = e.track.home;
            let changes: Vec<_> = e.states.iter().enumerate().filter(|(_, s)| **s != home).map(|(i, s)| (i as u32, *s, home)).collect();
            e.states.fill(home);
            push_changes(&mut self.barriers, &e.track.raw, e.states.len() as u32, &changes);
        }
    }

    /// A UAV barrier on everything (orders storage writes before the next reads and writes).
    pub fn uav_barrier(&mut self) {
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
