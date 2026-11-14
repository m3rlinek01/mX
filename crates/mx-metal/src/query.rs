//! Query sets: timestamps (counter sample buffers) and occlusion (visibility result buffers).

use std::ops::Range;
use std::sync::Arc;

use mx_types as t;
use objc2_foundation::NSRange;
use objc2_metal::*;

use crate::{Buffer, CommandEncoder, Device, Error, Mt, Po, Queue, Result};

pub(crate) enum QueryRaw {
    Timestamp(Mt<Po<dyn MTLCounterSampleBuffer>>),
    Occlusion(Mt<Po<dyn MTLBuffer>>),
}

pub(crate) struct QuerySetInner {
    pub raw: QueryRaw,
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

    pub(crate) fn sample_buffer(&self) -> &ProtocolObjectSampleBuffer {
        match &self.0.raw {
            QueryRaw::Timestamp(b) => b,
            QueryRaw::Occlusion(_) => panic!("timestamps written to an occlusion query set"),
        }
    }

    pub(crate) fn visibility_buffer(&self) -> &objc2::runtime::ProtocolObject<dyn MTLBuffer> {
        match &self.0.raw {
            QueryRaw::Occlusion(b) => b,
            QueryRaw::Timestamp(_) => panic!("occlusion queries in a timestamp query set"),
        }
    }
}

pub(crate) type ProtocolObjectSampleBuffer = objc2::runtime::ProtocolObject<dyn MTLCounterSampleBuffer>;

/// The timestamp counter set, where the device can sample at pass boundaries.
pub(crate) fn timestamp_counter_set(device: &objc2::runtime::ProtocolObject<dyn MTLDevice>) -> Option<Po<dyn MTLCounterSet>> {
    if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    // SAFETY: reading an extern static string provided by Metal.
    let name = unsafe { MTLCommonCounterSetTimestamp };
    device.counterSets()?.iter().find(|s| s.name().isEqualToString(name))
}

impl Device {
    pub fn create_query_set(&self, desc: &t::QuerySetDescriptor<'_>) -> Result<QuerySet> {
        let device = &self.shared.device;
        let count = desc.count.max(1);
        let raw = match desc.ty {
            t::QueryType::Timestamp => {
                let set = timestamp_counter_set(device).ok_or_else(|| Error::Unsupported("timestamp queries".into()))?;
                let d = MTLCounterSampleBufferDescriptor::new();
                d.setCounterSet(Some(&set));
                d.setStorageMode(MTLStorageMode::Shared);
                // SAFETY: a positive count.
                unsafe { d.setSampleCount(count as usize) };
                if let Some(l) = desc.label {
                    d.setLabel(&crate::ns(l));
                }
                let b = device
                    .newCounterSampleBufferWithDescriptor_error(&d)
                    .map_err(|e| Error::Unsupported(format!("timestamp queries: {}", e.localizedDescription())))?;
                QueryRaw::Timestamp(Mt(b))
            }
            t::QueryType::Occlusion => {
                let b = device
                    .newBufferWithLength_options(count as usize * 8, MTLResourceOptions::StorageModePrivate)
                    .ok_or_else(|| Error::OutOfMemory("occlusion query set".into()))?;
                QueryRaw::Occlusion(Mt(b))
            }
        };
        Ok(QuerySet(Arc::new(QuerySetInner { raw, count: desc.count, ty: desc.ty })))
    }
}

impl Queue {
    /// Nanoseconds per timestamp tick. Apple GPUs count nanoseconds; other Metal 3 GPUs are
    /// measured against the CPU clock when the device opens.
    pub fn get_timestamp_period(&self) -> f32 {
        self.shared.timestamp_period
    }
}

/// Nanoseconds per GPU timestamp tick on `device`.
pub(crate) fn measure_timestamp_period(device: &objc2::runtime::ProtocolObject<dyn MTLDevice>) -> f32 {
    if device.supportsFamily(MTLGPUFamily::Apple7) {
        return 1.0;
    }
    let sample = || {
        let (mut cpu, mut gpu) = (0u64, 0u64);
        // SAFETY: two valid out-pointers.
        unsafe { device.sampleTimestamps_gpuTimestamp(std::ptr::NonNull::from(&mut cpu), std::ptr::NonNull::from(&mut gpu)) };
        (cpu, gpu)
    };
    let (c0, g0) = sample();
    std::thread::sleep(std::time::Duration::from_millis(3));
    let (c1, g1) = sample();
    // (the CPU timestamp is mach_absolute_time; on Intel Macs its unit is a nanosecond)
    if g1 > g0 && c1 > c0 { (c1 - c0) as f32 / (g1 - g0) as f32 } else { 1.0 }
}

impl CommandEncoder {
    /// Write `set`'s results `range` into `dst` at `dst_offset` (`u64` each).
    pub fn resolve_query_set(&mut self, set: &QuerySet, range: Range<u32>, dst: &Buffer, dst_offset: u64) {
        assert!(range.end <= set.count(), "resolve_query_set range past the set");
        assert!(dst_offset.is_multiple_of(t::QUERY_RESOLVE_BUFFER_ALIGNMENT), "resolve_query_set: dst_offset must be a multiple of 256");
        assert!(dst_offset + range.len() as u64 * 8 <= dst.size(), "resolve_query_set past the end of the buffer");
        let n = range.len();
        if n == 0 {
            return;
        }
        let b = self.raw.blitCommandEncoder().expect("Metal returned no blit encoder");
        // SAFETY: ranges checked above.
        unsafe {
            match &set.0.raw {
                QueryRaw::Timestamp(sb) => {
                    b.resolveCounters_inRange_destinationBuffer_destinationOffset(sb, NSRange::new(range.start as usize, n), &dst.0.raw, dst_offset as usize);
                }
                QueryRaw::Occlusion(vb) => {
                    b.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(vb, range.start as usize * 8, &dst.0.raw, dst_offset as usize, n * 8);
                }
            }
        }
        b.endEncoding();
    }

    /// A timestamp between the commands before and after.
    pub fn write_timestamp(&mut self, set: &QuerySet, index: u32) {
        assert!(index < set.count(), "write_timestamp index past the set");
        let d = MTLBlitPassDescriptor::new();
        // SAFETY: index 0 of the attachment array; the sample index is inside the set.
        unsafe {
            let a = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
            a.setSampleBuffer(Some(set.sample_buffer()));
            a.setStartOfEncoderSampleIndex(index as usize);
            a.setEndOfEncoderSampleIndex(MTLCounterDontSample);
        }
        let b = self.raw.blitCommandEncoderWithDescriptor(&d).expect("Metal returned no blit encoder");
        b.endEncoding();
    }
}
