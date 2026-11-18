//! Query sets: occlusion queries and timestamps.
//!
//! Timestamps are measured under a disjoint query begun at the submission's first timestamp
//! and ended with the submission (or at a resolve, which needs its frequency): resolving waits
//! for the GPU (the bound tier's one way to read a query) and writes nanoseconds.

use std::sync::Arc;

use mx_types as t;
use windows::Win32::Graphics::Direct3D11::*;

use crate::device::{Ctx, Shared};
use windows::core::{HRESULT, Interface};

/// `GetData`, with its result as it is: `S_FALSE` (not ready) is a success code, which the
/// `windows` crate's wrapper turns into `Ok`.
///
/// # Safety
/// `data` holds `size` bytes, the size of the query's result.
pub(crate) unsafe fn get_data(c: &ID3D11DeviceContext1, q: &ID3D11Query, data: *mut std::ffi::c_void, size: u32) -> HRESULT {
    // SAFETY: the context's own vtable, called with its own pointer.
    unsafe { (Interface::vtable(c).base__.GetData)(Interface::as_raw(c), Interface::as_raw(q), data, size, 0) }
}
use crate::{Device, Error, Result, hr};

pub(crate) struct QuerySetInner {
    pub ty: t::QueryType,
    pub queries: Vec<ID3D11Query>,
}

// SAFETY: queries are used only under the context's lock.
unsafe impl Send for QuerySetInner {}
unsafe impl Sync for QuerySetInner {}

#[derive(Clone)]
pub struct QuerySet(pub(crate) Arc<QuerySetInner>);

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.0.queries.len() as u32
    }
    pub fn ty(&self) -> t::QueryType {
        self.0.ty
    }
}

impl Device {
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        let kind = match desc.ty {
            t::QueryType::Occlusion => D3D11_QUERY_OCCLUSION,
            t::QueryType::Timestamp => D3D11_QUERY_TIMESTAMP,
        };
        let mut queries = Vec::with_capacity(desc.count as usize);
        for _ in 0..desc.count {
            let d = D3D11_QUERY_DESC { Query: kind, MiscFlags: 0 };
            let mut q = None;
            // SAFETY: plain creation.
            unsafe { self.shared.device.CreateQuery(&d, Some(&mut q)) }.map_err(|e| hr("query", e))?;
            queries.push(q.ok_or_else(|| Error::OutOfMemory("query".into()))?);
        }
        Ok(QuerySet(Arc::new(QuerySetInner { ty: desc.ty, queries })))
    }
}

/// Begin the disjoint query timestamps need, if none is running.
pub(crate) fn begin_disjoint(ctx: &mut Ctx, shared: &Shared) {
    if ctx.disjoint.is_some() {
        return;
    }
    let d = D3D11_QUERY_DESC {
        Query: D3D11_QUERY_TIMESTAMP_DISJOINT,
        MiscFlags: 0,
    };
    let mut q = None;
    // SAFETY: plain creation; begun on the immediate context.
    if unsafe { shared.device.CreateQuery(&d, Some(&mut q)) }.is_ok()
        && let Some(q) = q
    {
        unsafe { ctx.context.Begin(&q) };
        ctx.disjoint = Some(q);
    }
}

/// End the disjoint query and read its frequency (waiting for the GPU).
pub(crate) fn end_disjoint(ctx: &mut Ctx) {
    let Some(q) = ctx.disjoint.take() else { return };
    let mut data = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
    // SAFETY: the query was begun on this context; its result is the struct read into.
    unsafe {
        ctx.context.End(&q);
        while get_data(
            &ctx.context,
            &q,
            (&mut data as *mut D3D11_QUERY_DATA_TIMESTAMP_DISJOINT).cast(),
            std::mem::size_of::<D3D11_QUERY_DATA_TIMESTAMP_DISJOINT>() as u32,
        ) == windows::Win32::Foundation::S_FALSE
        {
            std::thread::yield_now();
        }
    }
    if data.Frequency > 0 {
        ctx.timestamp_frequency = data.Frequency;
    }
    if data.Disjoint.as_bool() {
        log::debug!("mX: the GPU's clock changed while timestamps were taken; they may be off");
    }
}

/// The results of `range` as `u64`s: sample counts, or timestamps in nanoseconds. Waits for
/// the GPU.
pub(crate) fn results(ctx: &mut Ctx, set: &QuerySet, range: std::ops::Range<u32>) -> Vec<u64> {
    if set.ty() == t::QueryType::Timestamp {
        end_disjoint(ctx);
    }
    let freq = ctx.timestamp_frequency.max(1);
    let mut out = Vec::with_capacity(range.len());
    for i in range {
        let q = &set.0.queries[i as usize];
        let mut v = 0u64;
        // SAFETY: a query of this device; its result is a u64 for both kinds.
        unsafe {
            loop {
                let r = get_data(&ctx.context, q, (&mut v as *mut u64).cast(), 8);
                // (a query never issued reports nothing: zero, as WebGPU leaves it)
                if r != windows::Win32::Foundation::S_FALSE {
                    break;
                }
                std::thread::yield_now();
            }
        }
        out.push(if set.ty() == t::QueryType::Timestamp {
            (v as u128 * 1_000_000_000 / freq as u128) as u64
        } else {
            v
        });
    }
    out
}
