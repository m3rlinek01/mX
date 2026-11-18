//! What the vendors need of Direct3D 12 beside mX: command lists of our own on mX's queue.

use anyhow::{Context, Result};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

pub(crate) fn wide(s: &std::path::Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}

struct Slot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    done_at: u64,
}

pub(crate) struct ListRing {
    slots: Vec<Slot>,
    next: usize,
    fence: ID3D12Fence,
    value: u64,
}

impl ListRing {
    pub(crate) fn new(device: &ID3D12Device) -> Result<ListRing> {
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.context("creating a fence")?;
        let slots = (0..3)
            .map(|_| -> Result<Slot> {
                let allocator: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
                let list: ID3D12GraphicsCommandList = unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None::<&ID3D12PipelineState>) }?;
                unsafe { list.Close() }?;
                Ok(Slot { allocator, list, done_at: 0 })
            })
            .collect::<Result<Vec<_>>>()
            .context("creating command lists")?;
        Ok(ListRing { slots, next: 0, fence, value: 0 })
    }

    pub(crate) fn begin(&mut self) -> Result<ID3D12GraphicsCommandList> {
        let slot = &mut self.slots[self.next];
        if unsafe { self.fence.GetCompletedValue() } < slot.done_at {
            // (a null event handle makes this call block until the fence is reached)
            unsafe { self.fence.SetEventOnCompletion(slot.done_at, HANDLE::default()) }.context("waiting for a command list")?;
        }
        unsafe {
            slot.allocator.Reset()?;
            slot.list.Reset(&slot.allocator, None::<&ID3D12PipelineState>)?;
        }
        Ok(slot.list.clone())
    }

    pub(crate) fn submit(&mut self, queue: &ID3D12CommandQueue) -> Result<()> {
        let count = self.slots.len();
        let slot = &mut self.slots[self.next];
        self.next = (self.next + 1) % count;
        unsafe { slot.list.Close() }?;
        let list: ID3D12CommandList = slot.list.cast()?;
        unsafe { queue.ExecuteCommandLists(&[Some(list)]) };
        self.value += 1;
        unsafe { queue.Signal(&self.fence, self.value) }?;
        slot.done_at = self.value;
        Ok(())
    }

    pub(crate) fn wait(&self) {
        if unsafe { self.fence.GetCompletedValue() } < self.value {
            let _ = unsafe { self.fence.SetEventOnCompletion(self.value, HANDLE::default()) };
        }
    }
}
