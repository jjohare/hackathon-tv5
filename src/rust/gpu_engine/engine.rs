/// Main GPU Semantic Engine
///
/// Orchestrates all GPU operations with optimized memory management,
/// stream coordination, and performance monitoring.

use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::*;

/// Configuration for GPU engine initialization
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// CUDA device ID to use (None = auto-select)
    pub device_id: Option<usize>,

    /// Path to compiled PTX kernels directory
    pub ptx_path: String,

    /// Memory pool size in bytes (None = auto-size)
    pub memory_pool_size: Option<usize>,

    /// Number of concurrent CUDA streams
    pub num_streams: usize,

    /// Enable performance metrics collection
    pub enable_metrics: bool,

    /// Enable kernel timing
    pub enable_timing: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            ptx_path: "./cuda_kernels/build".to_string(),
            memory_pool_size: None,
            num_streams: 4,
            enable_metrics: true,
            enable_timing: true,
        }
    }
}

/// Main GPU semantic engine for accelerated operations
pub struct GpuSemanticEngine {
    /// CUDA device handle
    device: Arc<CudaDevice>,

    /// Loaded kernel modules
    modules: Arc<KernelModules>,

    /// GPU memory pool for efficient allocation
    memory_pool: Arc<RwLock<MemoryPool>>,

    /// CUDA stream manager for concurrent operations
    streams: Arc<StreamManager>,

    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,

    /// Configuration
    config: GpuConfig,
}

impl GpuSemanticEngine {
    /// Create a new GPU semantic engine
    ///
    /// # Arguments
    ///
    /// * `config` - Engine configuration
    ///
    /// # Returns
    ///
    /// Initialized engine instance or error
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = GpuConfig::default();
    /// let engine = GpuSemanticEngine::new(config).await?;
    /// ```
    pub async fn new(config: GpuConfig) -> GpuResult<Self> {
        // Initialize CUDA device
        let device = Self::init_device(config.device_id)?;
        let device = Arc::new(device);

        // Load kernel modules
        let modules = KernelModules::load(device.clone(), &config.ptx_path)?;
        let modules = Arc::new(modules);

        // Initialize memory pool
        let pool_size = config.memory_pool_size.unwrap_or_else(|| {
            Self::calculate_pool_size(&device)
        });
        let memory_pool = MemoryPool::new(device.clone(), pool_size)?;
        let memory_pool = Arc::new(RwLock::new(memory_pool));

        // Initialize stream manager
        let streams = StreamManager::new(device.clone(), config.num_streams)?;
        let streams = Arc::new(streams);

        // Initialize metrics
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::default()));

        Ok(Self {
            device,
            modules,
            memory_pool,
            streams,
            metrics,
            config,
        })
    }

    /// Initialize CUDA device
    fn init_device(device_id: Option<usize>) -> GpuResult<CudaDevice> {
        let device = if let Some(id) = device_id {
            CudaDevice::new(id)?
        } else {
            // Auto-select device with most memory
            let device_count = CudaDevice::count()?;
            if device_count == 0 {
                return Err(GpuError::Config("No CUDA devices found".to_string()));
            }

            let mut best_device = 0;
            let mut best_memory = 0;

            for i in 0..device_count {
                let device = CudaDevice::new(i)?;
                let total_mem = device.total_memory()?;
                if total_mem > best_memory {
                    best_memory = total_mem;
                    best_device = i;
                }
            }

            CudaDevice::new(best_device)?
        };

        Ok(device)
    }

    /// Calculate optimal memory pool size based on device memory
    fn calculate_pool_size(device: &CudaDevice) -> usize {
        // Use 80% of available device memory for pool
        let total_memory = device.total_memory().unwrap_or(0);
        (total_memory as f64 * 0.8) as usize
    }

    /// Get device information
    pub fn device_info(&self) -> GpuResult<DeviceInfo> {
        Ok(DeviceInfo {
            device_id: self.device.ordinal(),
            name: self.device.name()?,
            compute_capability: self.device.compute_capability()?,
            total_memory: self.device.total_memory()?,
            multiprocessor_count: self.device.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            )?,
            max_threads_per_block: self.device.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            )?,
            max_shared_memory: self.device.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
            )?,
        })
    }

    /// Get current performance metrics
    pub async fn metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset performance metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = PerformanceMetrics::default();
    }

    /// Compute semantic similarity for a batch of embeddings
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Input embeddings matrix [batch_size, embedding_dim]
    /// * `config` - Similarity computation configuration
    ///
    /// # Returns
    ///
    /// Similarity matrix [batch_size, batch_size]
    pub async fn compute_similarity_batch(
        &self,
        embeddings: &[f32],
        config: &SimilarityConfig,
    ) -> GpuResult<SimilarityMatrix> {
        let start = std::time::Instant::now();

        let result = similarity::compute_similarity_batch(
            &self.device,
            &self.modules,
            &self.memory_pool,
            &self.streams,
            embeddings,
            config,
        ).await?;

        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.total_operations += 1;
            metrics.total_compute_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        }

        Ok(result)
    }

    /// Enforce ontology constraints on entity relationships
    ///
    /// # Arguments
    ///
    /// * `constraints` - Ontology constraint definitions
    /// * `entities` - Entity relationship graph
    ///
    /// # Returns
    ///
    /// Reasoning result with validated relationships
    pub async fn enforce_ontology_constraints(
        &self,
        constraints: &OntologyConstraints,
        entities: &[u32],
    ) -> GpuResult<ReasoningResult> {
        let start = std::time::Instant::now();

        let result = reasoning::enforce_ontology_constraints(
            &self.device,
            &self.modules,
            &self.memory_pool,
            &self.streams,
            constraints,
            entities,
        ).await?;

        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.total_operations += 1;
            metrics.total_compute_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        }

        Ok(result)
    }

    /// Find shortest paths in knowledge graph
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph adjacency data
    /// * `sources` - Source node IDs
    /// * `targets` - Target node IDs
    /// * `config` - Pathfinding configuration
    ///
    /// # Returns
    ///
    /// Vector of shortest paths
    pub async fn find_shortest_paths(
        &self,
        graph: &[u32],
        sources: &[u32],
        targets: &[u32],
        config: &PathfindingConfig,
    ) -> GpuResult<Vec<Path>> {
        let start = std::time::Instant::now();

        let result = pathfinding::find_shortest_paths(
            &self.device,
            &self.modules,
            &self.memory_pool,
            &self.streams,
            graph,
            sources,
            targets,
            config,
        ).await?;

        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.total_operations += 1;
            metrics.total_compute_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        }

        Ok(result)
    }

    /// Synchronize all pending GPU operations
    pub async fn synchronize(&self) -> GpuResult<()> {
        self.device.synchronize()?;
        Ok(())
    }

    /// Get memory pool statistics
    pub async fn memory_stats(&self) -> (usize, usize, usize) {
        let pool = self.memory_pool.read().await;
        pool.stats()
    }
}

/// RAII guard for GPU metrics
pub struct GpuMetrics {
    engine: Arc<GpuSemanticEngine>,
    start: std::time::Instant,
    operation_name: String,
}

impl GpuMetrics {
    pub fn new(engine: Arc<GpuSemanticEngine>, operation_name: String) -> Self {
        Self {
            engine,
            start: std::time::Instant::now(),
            operation_name,
        }
    }
}

impl Drop for GpuMetrics {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64() * 1000.0;

        tokio::spawn({
            let metrics = self.engine.metrics.clone();
            let operation_name = self.operation_name.clone();
            async move {
                let mut m = metrics.write().await;
                m.total_operations += 1;
                m.total_compute_time_ms += duration;

                tracing::debug!(
                    operation = %operation_name,
                    duration_ms = duration,
                    "GPU operation completed"
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires CUDA device
    async fn test_engine_initialization() {
        let config = GpuConfig::default();
        let engine = GpuSemanticEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires CUDA device
    async fn test_device_info() {
        let config = GpuConfig::default();
        let engine = GpuSemanticEngine::new(config).await.unwrap();
        let info = engine.device_info().unwrap();

        assert!(!info.name.is_empty());
        assert!(info.total_memory > 0);
        assert!(info.multiprocessor_count > 0);
    }
}
