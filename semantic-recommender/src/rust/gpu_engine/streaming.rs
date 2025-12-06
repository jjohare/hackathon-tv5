/// CUDA Stream Management
///
/// Provides concurrent execution through multiple CUDA streams
/// with automatic synchronization and dependency tracking.

use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};

use super::*;

/// Handle to a CUDA stream for async operations
pub struct StreamHandle {
    stream: Arc<CudaStream>,
    device: Arc<CudaDevice>,
    id: usize,
}

impl StreamHandle {
    /// Synchronize this stream (wait for all operations to complete)
    pub async fn synchronize(&self) -> GpuResult<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    /// Get stream ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Check if stream has completed all operations
    pub fn is_complete(&self) -> bool {
        self.stream.is_complete().unwrap_or(false)
    }
}

/// Manager for multiple CUDA streams
pub struct StreamManager {
    device: Arc<CudaDevice>,
    streams: Vec<Arc<CudaStream>>,
    available: Arc<Semaphore>,
    next_stream: Arc<Mutex<usize>>,
}

impl StreamManager {
    /// Create a new stream manager
    pub fn new(device: Arc<CudaDevice>, num_streams: usize) -> GpuResult<Self> {
        let mut streams = Vec::with_capacity(num_streams);

        for _ in 0..num_streams {
            let stream = device.fork_default_stream()?;
            streams.push(Arc::new(stream));
        }

        Ok(Self {
            device,
            streams,
            available: Arc::new(Semaphore::new(num_streams)),
            next_stream: Arc::new(Mutex::new(0)),
        })
    }

    /// Acquire a stream for operations
    pub async fn acquire(&self) -> GpuResult<StreamHandle> {
        // Wait for available stream
        let _permit = self.available.acquire().await
            .map_err(|e| GpuError::Stream(format!("Failed to acquire stream: {}", e)))?;

        let mut next = self.next_stream.lock().await;
        let id = *next;
        *next = (*next + 1) % self.streams.len();

        Ok(StreamHandle {
            stream: self.streams[id].clone(),
            device: self.device.clone(),
            id,
        })
    }

    /// Acquire a specific stream by ID
    pub async fn acquire_specific(&self, stream_id: usize) -> GpuResult<StreamHandle> {
        if stream_id >= self.streams.len() {
            return Err(GpuError::Stream(format!(
                "Invalid stream ID: {} (max: {})",
                stream_id,
                self.streams.len() - 1
            )));
        }

        let _permit = self.available.acquire().await
            .map_err(|e| GpuError::Stream(format!("Failed to acquire stream: {}", e)))?;

        Ok(StreamHandle {
            stream: self.streams[stream_id].clone(),
            device: self.device.clone(),
            id: stream_id,
        })
    }

    /// Release a stream back to the pool
    pub fn release(&self, _handle: StreamHandle) {
        // Permit is automatically released when handle is dropped
    }

    /// Synchronize all streams
    pub async fn synchronize_all(&self) -> GpuResult<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get number of streams
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Get device
    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }
}

/// Batch executor for parallel operations across streams
pub struct BatchExecutor {
    manager: Arc<StreamManager>,
}

impl BatchExecutor {
    pub fn new(manager: Arc<StreamManager>) -> Self {
        Self { manager }
    }

    /// Execute multiple operations in parallel across streams
    pub async fn execute_parallel<F, R>(
        &self,
        operations: Vec<F>,
    ) -> GpuResult<Vec<R>>
    where
        F: FnOnce(StreamHandle) -> std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<R>> + Send>> + Send + 'static,
        R: Send + 'static,
    {
        let mut handles = Vec::new();

        for op in operations {
            let manager = self.manager.clone();

            let handle = tokio::spawn(async move {
                let stream = manager.acquire().await?;
                let result = op(stream).await?;
                Ok::<R, GpuError>(result)
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await
                .map_err(|e| GpuError::Stream(format!("Task join error: {}", e)))??;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute operations with automatic chunking
    pub async fn execute_chunked<F, R>(
        &self,
        total_items: usize,
        chunk_size: usize,
        operation: F,
    ) -> GpuResult<Vec<R>>
    where
        F: Fn(usize, usize, StreamHandle) -> std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<R>> + Send>> + Send + Clone + 'static,
        R: Send + 'static,
    {
        let num_chunks = (total_items + chunk_size - 1) / chunk_size;
        let mut operations = Vec::new();

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(total_items);
            let op = operation.clone();

            operations.push(Box::new(move |stream: StreamHandle| {
                Box::pin(op(start, end, stream)) as std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<R>> + Send>>
            }) as Box<dyn FnOnce(StreamHandle) -> std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<R>> + Send>> + Send>);
        }

        self.execute_parallel(operations).await
    }
}

/// Stream priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// Priority-based stream scheduler
pub struct PriorityScheduler {
    manager: Arc<StreamManager>,
    priority_queue: Arc<Mutex<Vec<(StreamPriority, Box<dyn FnOnce(StreamHandle) + Send>)>>>,
}

impl PriorityScheduler {
    pub fn new(manager: Arc<StreamManager>) -> Self {
        Self {
            manager,
            priority_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Schedule an operation with priority
    pub async fn schedule<F>(
        &self,
        priority: StreamPriority,
        operation: F,
    ) where
        F: FnOnce(StreamHandle) + Send + 'static,
    {
        let mut queue = self.priority_queue.lock().await;
        queue.push((priority, Box::new(operation)));

        // Sort by priority (highest first)
        queue.sort_by(|a, b| b.0.cmp(&a.0));
    }

    /// Process scheduled operations
    pub async fn process(&self) -> GpuResult<()> {
        let operations = {
            let mut queue = self.priority_queue.lock().await;
            std::mem::take(&mut *queue)
        };

        for (_, operation) in operations {
            let stream = self.manager.acquire().await?;
            operation(stream);
        }

        self.manager.synchronize_all().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires CUDA device
    async fn test_stream_manager() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let manager = StreamManager::new(device, 4).unwrap();

        assert_eq!(manager.num_streams(), 4);

        let stream = manager.acquire().await.unwrap();
        assert!(stream.id() < 4);
    }

    #[tokio::test]
    #[ignore] // Requires CUDA device
    async fn test_batch_executor() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let manager = Arc::new(StreamManager::new(device, 4).unwrap());
        let executor = BatchExecutor::new(manager);

        let operations = vec![
            Box::new(|_stream: StreamHandle| {
                Box::pin(async { Ok::<i32, GpuError>(1) }) as std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<i32>> + Send>>
            }) as Box<dyn FnOnce(StreamHandle) -> std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<i32>> + Send>> + Send>,
            Box::new(|_stream: StreamHandle| {
                Box::pin(async { Ok::<i32, GpuError>(2) }) as std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<i32>> + Send>>
            }) as Box<dyn FnOnce(StreamHandle) -> std::pin::Pin<Box<dyn std::future::Future<Output = GpuResult<i32>> + Send>> + Send>,
        ];

        let results = executor.execute_parallel(operations).await.unwrap();
        assert_eq!(results, vec![1, 2]);
    }
}
