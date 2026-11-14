//! Query sets: timestamp and occlusion query heaps.

use std::ops::Range;
use std::sync::Arc;

use mx_types as t;
use windows::Win32::Graphics::Direct3D12::*;

use crate::{Buffer, CommandEncoder, Device, Error, Result, hr_err};

pub(crate) struct QuerySetInner {
    pub raw: ID3D12QueryHeap,
    pub count: u32,
    pub ty: t::QueryType,
}

#[derive(Clone)]
pub struct QuerySet(pub(crate) Arc<QuerySetInner>);

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.0.count
    }

    pub fn ty(&self) -> t::QueryType {
        self.0.ty
    }

    /// The `ID3D12QueryHeap`, for interop.
    pub fn raw(&self) -> &ID3D12QueryHeap {
        &self.0.raw
    }

    pub(crate) fn write(&self, list: &ID3D12GraphicsCommandList, index: u32) {
        assert_eq!(self.0.ty, t::QueryType::Timestamp, "timestamps written to an occlusion query set");
        assert!(index < self.0.count, "timestamp index {index} past the set");
        // SAFETY: an index inside the heap.
        unsafe { list.EndQuery(&self.0.raw, D3D12_QUERY_TYPE_TIMESTAMP, index) };
    }

    pub(crate) fn begin_occlusion(&self, list: &ID3D12GraphicsCommandList, index: u32) {
        assert_eq!(self.0.ty, t::QueryType::Occlusion, "occlusion queries in a timestamp query set");
        assert!(index < self.0.count, "occlusion query index {index} past the set");
        // SAFETY: an index inside the heap.
        unsafe { list.BeginQuery(&self.0.raw, D3D12_QUERY_TYPE_OCCLUSION, index) };
    }

    pub(crate) fn end_occlusion(&self, list: &ID3D12GraphicsCommandList, index: u32) {
        // SAFETY: the query begun before.
        unsafe { list.EndQuery(&self.0.raw, D3D12_QUERY_TYPE_OCCLUSION, index) };
    }
}

impl Device {
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        let d = D3D12_QUERY_HEAP_DESC {
            Type: match desc.ty {
                t::QueryType::Timestamp => D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                t::QueryType::Occlusion => D3D12_QUERY_HEAP_TYPE_OCCLUSION,
            },
            Count: desc.count.max(1),
            NodeMask: 0,
        };
        let mut raw: Option<ID3D12QueryHeap> = None;
        // SAFETY: a plain creation.
        unsafe { self.shared.device.CreateQueryHeap(&d, &mut raw) }.map_err(|e| hr_err("query set", e))?;
        let raw = raw.ok_or_else(|| Error::OutOfMemory("query set".into()))?;
        if let Some(l) = desc.label {
            crate::set_name(&raw, l);
        }
        Ok(QuerySet(Arc::new(QuerySetInner { raw, count: desc.count, ty: desc.ty })))
    }
}

impl CommandEncoder {
    /// Write `set`'s results `range` into `dst` at `dst_offset` (`u64` each: ticks for
    /// timestamps, passing samples for occlusion queries).
    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64) {
        assert!(range.end <= set.count(), "resolve_query_set range past the set");
        assert!(dst_offset.is_multiple_of(t::QUERY_RESOLVE_BUFFER_ALIGNMENT), "resolve_query_set: dst_offset must be a multiple of 256");
        assert!(dst_offset + range.len() as u64 * 8 <= dst.size(), "resolve_query_set past the end of the buffer");
        if range.is_empty() {
            return;
        }
        let ty = match set.ty() {
            t::QueryType::Timestamp => D3D12_QUERY_TYPE_TIMESTAMP,
            t::QueryType::Occlusion => D3D12_QUERY_TYPE_OCCLUSION,
        };
        let rec = &mut self.rec;
        rec.track.need(&dst.0.track, None, D3D12_RESOURCE_STATE_COPY_DEST);
        rec.barriers();
        // SAFETY: ranges checked above.
        unsafe { rec.list().ResolveQueryData(&set.0.raw, ty, range.start, range.len() as u32, &dst.0.raw, dst_offset) };
        rec.keep(set.clone());
    }

    /// A timestamp between the commands before and after.
    pub fn write_timestamp(&mut self, set: &QuerySet, index: u32) {
        set.write(self.rec.list(), index);
        self.rec.keep(set.clone());
    }
}
