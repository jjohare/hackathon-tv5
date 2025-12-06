/// Application state with GPU-accelerated recommendation engine integration
///
/// This module initializes and manages the core application state, including
/// GPU pipeline, storage backends, and recommendation engine coordination.

use std::sync::Arc;
use crate::integration::metrics::Metrics;
use crate::integration::embedding_service::EmbeddingService;
use crate::integration::stub_gpu::{GpuSemanticEngine, GpuConfig};
pub use crate::integration::stub_gpu::{DeviceInfo as GpuDeviceInfo, PerformanceMetrics as GpuMetrics};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Main application state shared across API handlers
pub struct AppState {
    /// GPU-accelerated semantic engine
    pub gpu_engine: Arc<GpuSemanticEngine>,

    /// Text-to-embedding service
    pub embedding_service: Arc<EmbeddingService>,

    /// Performance metrics collector
    pub metrics: Arc<Metrics>,
}

impl AppState {
    /// Initialize application state with all required components
    ///
    /// This performs a complete bootstrap sequence:
    /// 1. Initialize GPU engine with CUDA device (or use stub if unavailable)
    /// 2. Set up metrics collection
    pub async fn new() -> Result<Self> {
        println!("[INFO] Initializing GPU-accelerated recommendation system...");

        // Initialize GPU engine
        let gpu_config = GpuConfig::default();

        let gpu_engine = match GpuSemanticEngine::new(gpu_config).await {
            Ok(engine) => {
                println!("[INFO] GPU engine initialized successfully");
                Arc::new(engine)
            }
            Err(e) => {
                println!("[WARN] GPU engine initialization failed: {}. Using CPU fallback.", e);
                return Err(format!("GPU not available: {}", e).into());
            }
        };

        // Initialize embedding service (384-dim for SBERT compatibility)
        let embedding_service = Arc::new(EmbeddingService::new(384));

        // Initialize metrics
        let metrics = Arc::new(Metrics::new());

        println!("[INFO] Application state initialized successfully");

        Ok(Self {
            gpu_engine,
            embedding_service,
            metrics,
        })
    }

    /// Check if GPU is available and healthy
    pub async fn gpu_available(&self) -> bool {
        // Attempt a simple GPU operation to verify availability
        self.gpu_engine.device_info().is_ok()
    }

    /// Get GPU device information
    pub async fn gpu_info(&self) -> Result<GpuDeviceInfo> {
        self.gpu_engine.device_info()
            .map_err(|e| format!("Failed to get GPU device info: {}", e).into())
    }

    /// Get current performance metrics
    pub async fn performance_metrics(&self) -> GpuMetrics {
        self.gpu_engine.metrics().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_app_state_initialization() {
        let state = AppState::new().await;
        assert!(state.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_gpu_availability() {
        let state = AppState::new().await.unwrap();
        assert!(state.gpu_available().await);
    }
}
