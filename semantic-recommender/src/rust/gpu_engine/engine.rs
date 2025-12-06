/// Main GPU Semantic Engine
///
/// Orchestrates all GPU operations with optimized memory management,
/// stream coordination, and performance monitoring.

use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::*;
use crate::adaptive_sssp::{
    AdaptiveSsspEngine, AdaptiveSsspConfig, AlgorithmMode, SsspMetrics,
};

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

    /// Adaptive SSSP engine for intelligent algorithm selection
    adaptive_sssp: Arc<RwLock<AdaptiveSsspEngine>>,
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

        // Initialize adaptive SSSP with default configuration
        let sssp_config = AdaptiveSsspConfig::default();
        let adaptive_sssp = Arc::new(RwLock::new(
            AdaptiveSsspEngine::new(sssp_config)
        ));

        Ok(Self {
            device,
            modules,
            memory_pool,
            streams,
            metrics,
            config,
            adaptive_sssp,
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

    /// Find shortest paths in knowledge graph with adaptive algorithm selection
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph adjacency data
    /// * `sources` - Source node IDs
    /// * `targets` - Target node IDs
    /// * `config` - Pathfinding configuration
    /// * `algorithm` - Optional algorithm override (None = auto-select)
    ///
    /// # Returns
    ///
    /// Vector of shortest paths with execution metrics
    pub async fn find_shortest_paths(
        &self,
        graph: &[u32],
        sources: &[u32],
        targets: &[u32],
        config: &PathfindingConfig,
    ) -> GpuResult<Vec<Path>> {
        self.find_shortest_paths_with_algorithm(graph, sources, targets, config, None).await
    }

    /// Find shortest paths with explicit algorithm selection
    pub async fn find_shortest_paths_with_algorithm(
        &self,
        graph: &[u32],
        sources: &[u32],
        targets: &[u32],
        config: &PathfindingConfig,
        algorithm: Option<AlgorithmMode>,
    ) -> GpuResult<Vec<Path>> {
        let start = std::time::Instant::now();

        // Determine which algorithm to use
        let selected_algorithm = if let Some(mode) = algorithm {
            mode
        } else {
            // Auto-select based on graph characteristics
            let mut sssp = self.adaptive_sssp.write().await;
            let num_nodes = graph.len() / 2; // Approximate
            let num_edges = graph.len();
            sssp.update_graph_stats(num_nodes, num_edges);
            sssp.select_algorithm()
        };

        // Execute with selected algorithm
        let result = match selected_algorithm {
            AlgorithmMode::Auto | AlgorithmMode::GpuDijkstra => {
                // Use existing GPU pathfinding
                pathfinding::find_shortest_paths(
                    &self.device,
                    &self.modules,
                    &self.memory_pool,
                    &self.streams,
                    graph,
                    sources,
                    targets,
                    config,
                ).await?
            }
            AlgorithmMode::LandmarkApsp => {
                // Use landmark-based APSP
                self.find_paths_landmark(graph, sources, targets, config).await?
            }
            AlgorithmMode::Duan => {
                // Future: Duan et al. implementation
                // For now, fall back to GPU Dijkstra
                pathfinding::find_shortest_paths(
                    &self.device,
                    &self.modules,
                    &self.memory_pool,
                    &self.streams,
                    graph,
                    sources,
                    targets,
                    config,
                ).await?
            }
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        // Record metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.total_operations += 1;
            metrics.total_compute_time_ms += elapsed;

            // Record SSSP-specific metrics
            let sssp_metrics = SsspMetrics {
                algorithm_used: format!("{:?}", selected_algorithm),
                total_time_ms: elapsed as f32,
                gpu_time_ms: Some(elapsed as f32 * 0.9),
                nodes_processed: graph.len() / 2,
                edges_relaxed: graph.len(),
                landmarks_used: None,
                complexity_factor: None,
            };

            let sssp = self.adaptive_sssp.read().await;
            sssp.record_metrics(sssp_metrics);
        }

        Ok(result)
    }

    /// Landmark-based pathfinding implementation
    async fn find_paths_landmark(
        &self,
        graph: &[u32],
        sources: &[u32],
        targets: &[u32],
        config: &PathfindingConfig,
    ) -> GpuResult<Vec<Path>> {
        // Simplified landmark implementation
        // In production: use full landmark APSP from adaptive_sssp module

        // For now, delegate to standard pathfinding
        pathfinding::find_shortest_paths(
            &self.device,
            &self.modules,
            &self.memory_pool,
            &self.streams,
            graph,
            sources,
            targets,
            config,
        ).await
    }

    /// Get adaptive SSSP metrics
    pub async fn get_sssp_metrics(&self) -> Option<SsspMetrics> {
        let sssp = self.adaptive_sssp.read().await;
        sssp.last_metrics()
    }

    /// Get current algorithm selection
    pub async fn get_selected_algorithm(&self) -> AlgorithmMode {
        let sssp = self.adaptive_sssp.read().await;
        sssp.select_algorithm()
    }

    /// Set algorithm mode (override auto-selection)
    pub async fn set_algorithm_mode(&self, mode: AlgorithmMode) -> GpuResult<()> {
        let mut sssp = self.adaptive_sssp.write().await;
        let mut config = AdaptiveSsspConfig::default();
        config.mode = mode;
        *sssp = AdaptiveSsspEngine::new(config);
        Ok(())
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
