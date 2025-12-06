/// Integration module tests
///
/// Tests for AppState, Health, and Metrics functionality

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_recording() {
        let metrics = Metrics::new();

        // Record some operations
        metrics.record_success(Duration::from_millis(100));
        metrics.record_success(Duration::from_millis(200));
        metrics.record_failure();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.successful_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert!((snapshot.success_rate - 66.66).abs() < 0.1);
        assert!(snapshot.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = Metrics::new();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.cache_hits, 3);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.cache_hit_rate, 75.0);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new();

        metrics.record_success(Duration::from_millis(50));
        metrics.record_cache_hit();

        metrics.reset();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.cache_hits, 0);
    }

    #[test]
    fn test_component_health() {
        let health = health::ComponentHealth::new(
            true,
            "GPU available".to_string(),
        );

        assert!(health.all_healthy());
        assert_eq!(health.overall_status(), "healthy");
    }

    #[test]
    fn test_component_health_degraded() {
        let health = health::ComponentHealth::new(
            false,
            "GPU not available".to_string(),
        );

        assert!(health.all_healthy()); // Core components still healthy
        assert_eq!(health.overall_status(), "degraded"); // But degraded without GPU
    }

    #[test]
    fn test_gpu_info_conversion() {
        let device_info = stub_gpu::DeviceInfo {
            device_id: 0,
            name: "Test GPU".to_string(),
            compute_capability: (8, 6),
            total_memory: 25_769_803_776, // 24 GB
            multiprocessor_count: 82,
            max_threads_per_block: 1024,
            max_shared_memory: 49152,
        };

        let gpu_info = health::GpuInfo::from_device_info(device_info);

        assert_eq!(gpu_info.device_id, 0);
        assert_eq!(gpu_info.name, "Test GPU");
        assert_eq!(gpu_info.compute_capability, (8, 6));
        assert!(gpu_info.total_memory_display.contains("24"));
        assert!(gpu_info.total_memory_display.contains("GB"));
    }

    #[test]
    fn test_gpu_config_default() {
        let config = stub_gpu::GpuConfig::default();

        assert_eq!(config.device_id, None);
        assert_eq!(config.num_streams, 4);
        assert!(config.enable_metrics);
        assert!(config.enable_timing);
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = stub_gpu::PerformanceMetrics::default();

        assert_eq!(metrics.total_operations, 0);
        assert_eq!(metrics.total_compute_time_ms, 0.0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
    }

    #[tokio::test]
    async fn test_gpu_stub_initialization() {
        let config = stub_gpu::GpuConfig::default();
        let result = stub_gpu::GpuSemanticEngine::new(config).await;

        // Stub should fail gracefully
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not available"));
    }

    #[test]
    fn test_metrics_snapshot_calculations() {
        let metrics = Metrics::new();

        // Record 10 successful requests with varying latency
        for i in 1..=10 {
            metrics.record_success(Duration::from_millis(i * 10));
        }

        // Record 5 cache hits and 5 misses
        for _ in 0..5 {
            metrics.record_cache_hit();
            metrics.record_cache_miss();
        }

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_requests, 10);
        assert_eq!(snapshot.successful_requests, 10);
        assert_eq!(snapshot.success_rate, 100.0);
        assert_eq!(snapshot.cache_hits, 5);
        assert_eq!(snapshot.cache_misses, 5);
        assert_eq!(snapshot.cache_hit_rate, 50.0);
        assert!(snapshot.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_health_response_serialization() {
        use serde_json;

        let health = health::HealthResponse {
            status: "healthy".to_string(),
            version: "1.0.0".to_string(),
            timestamp: "2025-12-04T14:48:00Z".to_string(),
            components: health::ComponentHealth::new(
                true,
                "GPU available".to_string(),
            ),
            gpu_info: None,
        };

        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_metrics_snapshot_serialization() {
        use serde_json;

        let snapshot = metrics::MetricsSnapshot {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            avg_latency_ms: 45.3,
            success_rate: 95.0,
            cache_hits: 70,
            cache_misses: 30,
            cache_hit_rate: 70.0,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("100"));
        assert!(json.contains("95"));
        assert!(json.contains("45.3"));
    }
}
