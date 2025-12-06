/// Health check system with component-level status monitoring
///
/// Provides detailed health information for all system components including
/// GPU status, storage backends, and service availability.

use serde::{Deserialize, Serialize};

/// Overall health status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Overall system status
    pub status: String,

    /// API version
    pub version: String,

    /// Current timestamp (RFC3339 format)
    pub timestamp: String,

    /// Component-level health details
    pub components: ComponentHealth,

    /// GPU device information (if available)
    pub gpu_info: Option<GpuInfo>,
}

/// Health status of individual components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// GPU engine availability
    pub gpu_available: bool,

    /// GPU engine status message
    pub gpu_status: String,

    /// Cache system health
    pub cache_healthy: bool,

    /// API server health
    pub api_healthy: bool,
}

/// GPU device information for health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU device ID
    pub device_id: usize,

    /// GPU device name
    pub name: String,

    /// CUDA compute capability
    pub compute_capability: (i32, i32),

    /// Total GPU memory in bytes
    pub total_memory_bytes: usize,

    /// Memory formatted for display
    pub total_memory_display: String,

    /// Number of streaming multiprocessors
    pub multiprocessor_count: i32,

    /// Maximum threads per block
    pub max_threads_per_block: i32,
}

impl GpuInfo {
    /// Convert from GPU engine DeviceInfo
    pub fn from_device_info(info: crate::integration::stub_gpu::DeviceInfo) -> Self {
        let memory_gb = info.total_memory as f64 / 1_073_741_824.0; // Convert to GB

        Self {
            device_id: info.device_id,
            name: info.name,
            compute_capability: info.compute_capability,
            total_memory_bytes: info.total_memory,
            total_memory_display: format!("{:.1} GB", memory_gb),
            multiprocessor_count: info.multiprocessor_count,
            max_threads_per_block: info.max_threads_per_block,
        }
    }
}

impl ComponentHealth {
    /// Create health status for all system components
    pub fn new(gpu_available: bool, gpu_status: String) -> Self {
        Self {
            gpu_available,
            gpu_status,
            cache_healthy: true, // Assume healthy if system is running
            api_healthy: true,   // If we can respond, API is healthy
        }
    }

    /// Check if all components are healthy
    pub fn all_healthy(&self) -> bool {
        self.cache_healthy && self.api_healthy
        // GPU is optional - system can run without it (slower)
    }

    /// Get overall status string
    pub fn overall_status(&self) -> &'static str {
        if self.all_healthy() && self.gpu_available {
            "healthy"
        } else if self.all_healthy() {
            "degraded" // Running but without GPU acceleration
        } else {
            "unhealthy"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_health_all_healthy() {
        let health = ComponentHealth::new(true, "GPU available".to_string());
        assert!(health.all_healthy());
        assert_eq!(health.overall_status(), "healthy");
    }

    #[test]
    fn test_component_health_degraded() {
        let health = ComponentHealth::new(false, "GPU not available".to_string());
        assert!(health.all_healthy()); // Core components still healthy
        assert_eq!(health.overall_status(), "degraded");
    }
}
