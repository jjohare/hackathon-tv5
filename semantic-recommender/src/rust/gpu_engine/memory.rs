/// GPU Memory Management
///
/// Provides efficient memory allocation, pooling, and transfer
/// operations with pinned memory for optimal performance.

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr};
use std::collections::HashMap;
use std::sync::Arc;

use super::*;

/// GPU memory pool for efficient allocation
pub struct MemoryPool {
    device: Arc<CudaDevice>,
    total_size: usize,
    allocated: usize,
    peak_allocated: usize,
    allocations: HashMap<usize, usize>, // ptr -> size
    free_blocks: Vec<(usize, usize)>,   // (size, ptr)
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(device: Arc<CudaDevice>, total_size: usize) -> GpuResult<Self> {
        Ok(Self {
            device,
            total_size,
            allocated: 0,
            peak_allocated: 0,
            allocations: HashMap::new(),
            free_blocks: Vec::new(),
        })
    }

    /// Allocate device memory from pool
    pub fn alloc<T: DeviceRepr>(&mut self, num_elements: usize) -> GpuResult<CudaSlice<T>> {
        let size_bytes = num_elements * std::mem::size_of::<T>();

        // Try to find a free block
        if let Some(idx) = self.free_blocks.iter().position(|(size, _)| *size >= size_bytes) {
            let (block_size, ptr) = self.free_blocks.remove(idx);
            self.allocated += size_bytes;
            self.peak_allocated = self.peak_allocated.max(self.allocated);
            self.allocations.insert(ptr, size_bytes);

            // Create slice from existing allocation
            unsafe {
                let device_ptr = DevicePtr::from_raw(ptr as *mut T)?;
                return Ok(CudaSlice::from_raw(device_ptr, num_elements));
            }
        }

        // Allocate new memory
        if self.allocated + size_bytes > self.total_size {
            return Err(GpuError::Memory(format!(
                "Pool exhausted: requested {}, available {}",
                size_bytes,
                self.total_size - self.allocated
            )));
        }

        let slice = self.device.alloc_zeros::<T>(num_elements)?;
        let ptr = slice.device_ptr().as_raw() as usize;

        self.allocated += size_bytes;
        self.peak_allocated = self.peak_allocated.max(self.allocated);
        self.allocations.insert(ptr, size_bytes);

        Ok(slice)
    }

    /// Free device memory back to pool
    pub fn free<T: DeviceRepr>(&mut self, slice: CudaSlice<T>) {
        let ptr = slice.device_ptr().as_raw() as usize;

        if let Some(size) = self.allocations.remove(&ptr) {
            self.allocated -= size;
            self.free_blocks.push((size, ptr));

            // Coalesce adjacent free blocks
            self.coalesce_free_blocks();
        }
    }

    /// Coalesce adjacent free memory blocks
    fn coalesce_free_blocks(&mut self) {
        if self.free_blocks.len() < 2 {
            return;
        }

        self.free_blocks.sort_by_key(|(_, ptr)| *ptr);

        let mut i = 0;
        while i < self.free_blocks.len() - 1 {
            let (size1, ptr1) = self.free_blocks[i];
            let (size2, ptr2) = self.free_blocks[i + 1];

            if ptr1 + size1 == ptr2 {
                self.free_blocks[i] = (size1 + size2, ptr1);
                self.free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Get memory pool statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.allocated, self.peak_allocated, self.total_size)
    }

    /// Clear all allocations (unsafe - assumes no outstanding references)
    pub unsafe fn clear(&mut self) {
        self.allocations.clear();
        self.free_blocks.clear();
        self.allocated = 0;
    }
}

/// Device buffer with automatic cleanup
pub struct DeviceBuffer<T: DeviceRepr> {
    slice: Option<CudaSlice<T>>,
    pool: Arc<tokio::sync::RwLock<MemoryPool>>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Create a new device buffer
    pub async fn new(
        pool: Arc<tokio::sync::RwLock<MemoryPool>>,
        num_elements: usize,
    ) -> GpuResult<Self> {
        let slice = {
            let mut pool = pool.write().await;
            pool.alloc::<T>(num_elements)?
        };

        Ok(Self {
            slice: Some(slice),
            pool,
        })
    }

    /// Get reference to underlying slice
    pub fn slice(&self) -> &CudaSlice<T> {
        self.slice.as_ref().unwrap()
    }

    /// Get mutable reference to underlying slice
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        self.slice.as_mut().unwrap()
    }

    /// Take ownership of slice (prevents auto-cleanup)
    pub fn take(mut self) -> CudaSlice<T> {
        self.slice.take().unwrap()
    }
}

impl<T: DeviceRepr> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.slice.take() {
            let pool = self.pool.clone();
            tokio::spawn(async move {
                let mut pool = pool.write().await;
                pool.free(slice);
            });
        }
    }
}

/// Pinned host memory for efficient transfers
pub struct PinnedBuffer<T> {
    data: Vec<T>,
    capacity: usize,
}

impl<T: Clone + Default> PinnedBuffer<T> {
    /// Create a new pinned buffer
    pub fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, T::default());

        Self { data, capacity }
    }

    /// Get slice of data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice of data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Resize buffer
    pub fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, T::default());
        self.capacity = new_size;
    }

    /// Copy from host to device
    pub async fn to_device<D: DeviceRepr>(
        &self,
        device: &CudaDevice,
    ) -> GpuResult<CudaSlice<D>>
    where
        T: bytemuck::Pod,
        D: bytemuck::Pod,
    {
        // Safety: Pod types have no invalid bit patterns
        let bytes = bytemuck::cast_slice(&self.data);
        let slice = device.htod_copy(bytes.to_vec())?;
        Ok(slice)
    }

    /// Copy from device to host
    pub async fn from_device<D: DeviceRepr>(
        &mut self,
        slice: &CudaSlice<D>,
    ) -> GpuResult<()>
    where
        T: bytemuck::Pod,
        D: bytemuck::Pod,
    {
        let bytes = slice.dtoh()?;
        let data: &[T] = bytemuck::cast_slice(&bytes);
        self.data.clear();
        self.data.extend_from_slice(data);
        Ok(())
    }
}

/// Transfer data between host and device asynchronously
pub struct AsyncTransfer {
    device: Arc<CudaDevice>,
}

impl AsyncTransfer {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Transfer host data to device asynchronously
    pub async fn htod<T: DeviceRepr + bytemuck::Pod>(
        &self,
        data: &[T],
    ) -> GpuResult<CudaSlice<T>> {
        let slice = self.device.htod_copy(data.to_vec())?;
        Ok(slice)
    }

    /// Transfer device data to host asynchronously
    pub async fn dtoh<T: DeviceRepr + bytemuck::Pod>(
        &self,
        slice: &CudaSlice<T>,
    ) -> GpuResult<Vec<T>> {
        let data = slice.dtoh()?;
        Ok(data)
    }

    /// Transfer device to device (copy)
    pub async fn dtod<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut CudaSlice<T>,
    ) -> GpuResult<()> {
        if src.len() != dst.len() {
            return Err(GpuError::Memory(
                "Source and destination sizes must match".to_string()
            ));
        }

        self.device.dtod_copy(src, dst)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let mut pool = MemoryPool::new(device, 1024 * 1024).unwrap();

        let slice1 = pool.alloc::<f32>(256).unwrap();
        assert_eq!(pool.allocated, 256 * 4);

        pool.free(slice1);
        assert_eq!(pool.allocated, 0);

        let (allocated, peak, total) = pool.stats();
        assert_eq!(allocated, 0);
        assert_eq!(peak, 1024);
        assert_eq!(total, 1024 * 1024);
    }

    #[test]
    fn test_pinned_buffer() {
        let mut buffer = PinnedBuffer::<f32>::new(100);
        assert_eq!(buffer.as_slice().len(), 100);

        buffer.resize(200);
        assert_eq!(buffer.as_slice().len(), 200);
    }
}
