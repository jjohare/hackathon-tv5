/// GPU Engine Orchestration Layer
///
/// This module provides a high-level Rust API for GPU-accelerated semantic operations
/// using CUDA kernels. It handles device management, memory optimization, and stream
/// coordination for maximum throughput.
///
/// # Architecture
///
/// - `engine`: Main orchestration and API surface
/// - `kernels`: CUDA kernel FFI bindings and launch wrappers
/// - `memory`: GPU memory pool and transfer optimization
/// - `streaming`: CUDA stream management for concurrent operations
/// - `similarity`: Semantic similarity computation operations
/// - `reasoning`: Ontology constraint enforcement operations
/// - `pathfinding`: Graph search and pathfinding operations
///
/// # Example
///
/// ```rust
/// use gpu_engine::GpuSemanticEngine;
///
/// let engine = GpuSemanticEngine::new().await?;
/// let similarities = engine.compute_similarity_batch(&embeddings).await?;
/// ```

pub mod engine;
pub mod kernels;
pub mod memory;
pub mod streaming;
pub mod similarity;
pub mod reasoning;
pub mod pathfinding;
pub mod gpu_bridge;

// Unified pipeline module (all 3 optimization phases)
pub mod unified_gpu;

pub use engine::{GpuSemanticEngine, GpuConfig, GpuMetrics};
pub use kernels::{KernelModules, KernelError};
pub use memory::{MemoryPool, DeviceBuffer, PinnedBuffer};
pub use streaming::{StreamManager, StreamHandle};
pub use similarity::{SimilarityMatrix, SimilarityConfig};
pub use reasoning::{OntologyConstraints, ReasoningResult};
pub use pathfinding::{Path, PathfindingConfig};
pub use gpu_bridge::{GpuBridge, GpuTransferManager, GpuBridgeError, ParsedAxiom, ViolationReport, SerializationStats, GpuHandle};

// Unified pipeline exports
pub use unified_gpu::{GPUPipeline, GPUPipelineBuilder, PipelineStats};

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

/// Unified error type for GPU operations
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("Kernel error: {0}")]
    Kernel(#[from] KernelError),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory: i32,
}

/// Performance metrics for GPU operations
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    pub total_operations: u64,
    pub total_compute_time_ms: f64,
    pub total_transfer_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_allocated_bytes: usize,
    pub memory_peak_bytes: usize,
}

impl PerformanceMetrics {
    pub fn average_compute_time_ms(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.total_compute_time_ms / self.total_operations as f64
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();
        metrics.total_operations = 100;
        metrics.total_compute_time_ms = 1000.0;
        metrics.cache_hits = 80;
        metrics.cache_misses = 20;

        assert_eq!(metrics.average_compute_time_ms(), 10.0);
        assert_eq!(metrics.cache_hit_rate(), 0.8);
    }
}
