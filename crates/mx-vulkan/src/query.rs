//! Query sets: timestamp and occlusion query pools.

use std::ops::Range;
use std::sync::Arc;

use ash::vk;
use mx_types as t;

use crate::device::{Garbage, Shared};
use crate::{Buffer, CommandEncoder, Device, Error, Queue, Result, vk_err};

pub(crate) struct QuerySetInner {
    pub raw: vk::QueryPool,
    pub count: u32,
    pub ty: t::QueryType,
    shared: Arc<Shared>,
}

impl Drop for QuerySetInner {
    fn drop(&mut self) {
        self.shared.retire(Garbage::QueryPool(self.raw));
    }
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

    /// The `VkQueryPool`, for interop.
    pub fn raw(&self) -> vk::QueryPool {
        self.0.raw
    }
}

impl Device {
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        let shared = &self.shared;
        let ty = match desc.ty {
            t::QueryType::Timestamp => {
                if !shared.caps.timestamps {
                    return Err(Error::Unsupported("timestamp queries".into()));
                }
                vk::QueryType::TIMESTAMP
            }
            t::QueryType::Occlusion => vk::QueryType::OCCLUSION,
        };
        let info = vk::QueryPoolCreateInfo::default().query_type(ty).query_count(desc.count.max(1));
        // SAFETY: plain creation.
        let raw = unsafe { shared.raw.create_query_pool(&info, None) }.map_err(|e| vk_err("query set", e))?;
        if let Some(l) = desc.label {
            shared.name(raw, l);
        }
        Ok(QuerySet(Arc::new(QuerySetInner {
            raw,
            count: desc.count,
            ty: desc.ty,
            shared: shared.clone(),
        })))
    }
}

impl Queue {
    /// Nanoseconds per timestamp tick (the device's `timestampPeriod`).
    pub fn get_timestamp_period(&self) -> f32 {
        self.shared.caps.timestamp_period
    }
}

impl CommandEncoder {
    /// Write `set`'s results `range` into `dst` at `dst_offset` (`u64` each).
    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64) {
        assert!(range.end <= set.count(), "resolve_query_set range past the set");
        assert!(dst_offset.is_multiple_of(t::QUERY_RESOLVE_BUFFER_ALIGNMENT), "resolve_query_set: dst_offset must be a multiple of 256");
        assert!(dst_offset + range.len() as u64 * 8 <= dst.size(), "resolve_query_set past the end of the buffer");
        if range.is_empty() {
            return;
        }
        if set.ty() == t::QueryType::Timestamp && self.shared.caps.portability {
            self.split();
        }
        self.before(crate::command::Op::ANY);
        // SAFETY: ranges checked above; the queries were written before (WAIT, as wgpu-hal).
        unsafe {
            self.shared.raw.cmd_copy_query_pool_results(
                self.raw,
                set.0.raw,
                range.start,
                range.len() as u32,
                dst.0.raw,
                dst_offset,
                t::QUERY_SIZE as u64,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        self.keep(set);
        self.keep(dst);
        self.after(crate::command::Op::ANY);
    }

    /// A timestamp between the commands before and after.
    pub fn write_timestamp(&mut self, set: &QuerySet, index: u32) {
        self.before(crate::command::Op::ANY);
        self.reset_timestamp(set, index);
        self.timestamp(set, index);
        self.after(crate::command::Op::ANY);
    }
}
