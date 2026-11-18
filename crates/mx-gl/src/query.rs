//! Occlusion and timestamp queries, read back on the CPU when resolved.

use std::ops::Range;
use std::sync::Arc;

use glow::HasContext;
use mx_types as t;

use crate::device::{Garbage, Shared};
use crate::{Device, Result};

pub(crate) struct QuerySetInner {
    queries: Vec<glow::Query>,
    ty: t::QueryType,
    shared: Arc<Shared>,
}

impl Drop for QuerySetInner {
    fn drop(&mut self) {
        for q in &self.queries {
            self.shared.retire(Garbage::Query(*q));
        }
    }
}

#[derive(Clone)]
pub struct QuerySet(pub(crate) Arc<QuerySetInner>);

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.0.queries.len() as u32
    }

    pub fn ty(&self) -> t::QueryType {
        self.0.ty
    }

    /// GL ES counts only whether any sample passed; desktop GL counts them.
    fn occlusion_target(&self) -> u32 {
        if self.0.shared.ctx.api == crate::context::Api::Es {
            glow::ANY_SAMPLES_PASSED_CONSERVATIVE
        } else {
            glow::SAMPLES_PASSED
        }
    }

    pub(crate) fn begin_occlusion(&self, gl: &glow::Context, index: u32) {
        // SAFETY: a query of this device, on the current context.
        unsafe { gl.begin_query(self.occlusion_target(), self.0.queries[index as usize]) };
    }

    pub(crate) fn end_occlusion(&self, gl: &glow::Context) {
        // SAFETY: closes the query begun before.
        unsafe { gl.end_query(self.occlusion_target()) };
    }

    pub(crate) fn timestamp(&self, gl: &glow::Context, index: u32) {
        if self.0.shared.caps.timestamps {
            // SAFETY: a query of this device, on the current context.
            unsafe { gl.query_counter(self.0.queries[index as usize], glow::TIMESTAMP) };
        }
    }

    /// The results of `range`, waiting for them (GL keeps them on the GPU until asked).
    pub(crate) fn results(&self, gl: &glow::Context, range: Range<u32>) -> Vec<u64> {
        range
            .map(|i| {
                let mut v = [0u8; 8];
                // SAFETY: a query of this device whose commands were issued before.
                unsafe { gl.get_query_parameter_u64_with_offset(self.0.queries[i as usize], glow::QUERY_RESULT, v.as_mut_ptr() as usize) };
                u64::from_le_bytes(v)
            })
            .collect()
    }
}

impl Device {
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        let gl = self.shared.ctx.lock();
        let queries = (0..desc.count)
            // SAFETY: plain creation.
            .map(|_| unsafe { gl.create_query() }.map_err(|e| crate::gl_err("query", e)))
            .collect::<Result<Vec<_>>>()?;
        Ok(QuerySet(Arc::new(QuerySetInner {
            queries,
            ty: desc.ty,
            shared: self.shared.clone(),
        })))
    }
}
