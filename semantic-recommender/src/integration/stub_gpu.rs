/// GPU stub types for when GPU engine is not available
///
/// These provide graceful fallback when CUDA is not present

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub device_id: Option<usize>,
    pub ptx_path: String,
    pub memory_pool_size: Option<usize>,
    pub num_streams: usize,
    pub enable_metrics: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory: i32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_operations: u64,
    pub total_compute_time_ms: f64,
    pub total_transfer_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_allocated_bytes: usize,
    pub memory_peak_bytes: usize,
}

#[derive(Debug)]
pub struct GpuSemanticEngine {
    _config: GpuConfig,
}

impl GpuSemanticEngine {
    pub async fn new(_config: GpuConfig) -> Result<Self, String> {
        // Stub implementation - would fail gracefully if GPU not available
        Err("GPU engine not available - using CPU fallback".to_string())
    }

    pub fn device_info(&self) -> Result<DeviceInfo, String> {
        Err("No GPU device available".to_string())
    }

    pub async fn metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics::default()
    }
}
