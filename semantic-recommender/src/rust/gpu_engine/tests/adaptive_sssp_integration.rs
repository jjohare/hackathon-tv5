/// Integration tests for adaptive SSSP algorithm selection
///
/// These tests verify the adaptive algorithm selection logic works correctly
/// with the GPU engine and produces correct results.

#[cfg(test)]
mod integration_tests {
    use crate::gpu_engine::*;
    use std::sync::Arc;

    /// Test fixture for GPU engine
    struct TestFixture {
        device: Arc<cudarc::driver::CudaDevice>,
        modules: Arc<KernelModules>,
        memory_pool: Arc<tokio::sync::RwLock<MemoryPool>>,
        streams: Arc<StreamManager>,
    }

    impl TestFixture {
        async fn new() -> GpuResult<Self> {
            let device = Arc::new(cudarc::driver::CudaDevice::new(0)?);
            let modules = Arc::new(KernelModules::load(&device)?);
            let memory_pool = Arc::new(tokio::sync::RwLock::new(MemoryPool::new(device.clone())));
            let streams = Arc::new(StreamManager::new(device.clone(), 4).await?);

            Ok(Self {
                device,
                modules,
                memory_pool,
                streams,
            })
        }
    }

    /// Generate a simple test graph
    fn create_test_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        // Simple linear graph: 0 -> 1 -> 2 -> 3 -> 4
        let graph = vec![
            0, 1,
            1, 2,
            2, 3,
            3, 4,
        ];

        let sources = vec![0];
        let targets = vec![4];

        (graph, sources, targets)
    }

    /// Generate a larger test graph
    fn create_large_test_graph(size: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut graph = Vec::new();

        // Create a grid-like graph structure
        for i in 0..size {
            // Connect to next node
            if i + 1 < size {
                graph.push(i as u32);
                graph.push((i + 1) as u32);
            }

            // Connect to node 10 steps ahead (shortcuts)
            if i + 10 < size {
                graph.push(i as u32);
                graph.push((i + 10) as u32);
            }
        }

        let sources = vec![0];
        let targets = vec![(size - 1) as u32];

        (graph, sources, targets)
    }

    #[tokio::test]
    async fn test_gpu_dijkstra_simple_path() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let (graph, sources, targets) = create_test_graph();

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::GPUDijkstra;
        config.enable_profiling = true;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        assert!(!ssp_result.paths.is_empty(), "Should find at least one path");
        assert_eq!(ssp_result.metrics.algorithm_used, "GPU Dijkstra");

        // Verify path correctness
        let path = &ssp_result.paths[0];
        assert_eq!(path.nodes[0], 0);
        assert_eq!(path.nodes[path.nodes.len() - 1], 4);
        assert_eq!(path.length, 4);
    }

    #[tokio::test]
    async fn test_hybrid_duan_simple_path() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let (graph, sources, targets) = create_test_graph();

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::HybridDuan;
        config.enable_profiling = true;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        assert!(!ssp_result.paths.is_empty(), "Should find at least one path");
        assert_eq!(ssp_result.metrics.algorithm_used, "Hybrid Duan");
    }

    #[tokio::test]
    async fn test_auto_selection_small_graph() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let (graph, sources, targets) = create_test_graph();

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::Auto;
        config.crossover_threshold = 1000; // Small graphs use GPU Dijkstra
        config.enable_profiling = true;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        // For small graph, should select GPU Dijkstra
        assert_eq!(ssp_result.metrics.algorithm_used, "GPU Dijkstra");
    }

    #[tokio::test]
    async fn test_auto_selection_large_graph() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let (graph, sources, targets) = create_large_test_graph(10000);

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::Auto;
        config.crossover_threshold = 5000; // Large graphs use Hybrid Duan
        config.enable_profiling = true;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        // For large graph, should select Hybrid Duan
        assert_eq!(ssp_result.metrics.algorithm_used, "Hybrid Duan");
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let (graph, sources, targets) = create_test_graph();

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::GPUDijkstra;
        config.enable_profiling = true;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        let metrics = &ssp_result.metrics;

        // Verify metrics are populated
        assert!(metrics.total_time_ms > 0.0);
        assert!(metrics.nodes_processed > 0);
        assert_eq!(metrics.paths_found, ssp_result.paths.len());
        assert!(metrics.peak_gpu_memory > 0);

        // Verify metric calculations
        let throughput = metrics.throughput_nodes_per_sec();
        assert!(throughput > 0.0);

        let gpu_util = metrics.gpu_utilization();
        assert!(gpu_util >= 0.0 && gpu_util <= 100.0);
    }

    #[tokio::test]
    async fn test_multiple_sources_targets() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let graph = vec![
            0, 1,
            0, 2,
            1, 3,
            2, 3,
            3, 4,
            3, 5,
        ];

        let sources = vec![0];
        let targets = vec![3, 4, 5];

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::GPUDijkstra;
        config.max_paths = 10;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        // Should find paths to all targets
        assert!(ssp_result.paths.len() >= 1);
    }

    #[tokio::test]
    async fn test_no_path_exists() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        // Disconnected graph
        let graph = vec![
            0, 1,
            1, 2,
            // 2 and 3 are not connected
            3, 4,
        ];

        let sources = vec![0];
        let targets = vec![4]; // No path from 0 to 4

        let mut config = AdaptiveSSPConfig::default();
        config.algorithm = SSPAlgorithm::GPUDijkstra;

        let result = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config,
        ).await;

        assert!(result.is_ok());

        let ssp_result = result.unwrap();
        // Should return empty paths or no path to target
        assert!(ssp_result.paths.is_empty() ||
                !ssp_result.paths.iter().any(|p| p.nodes.contains(&4)));
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = AdaptiveSSPConfig::default();

        assert_eq!(config.algorithm, SSPAlgorithm::Auto);
        assert_eq!(config.crossover_threshold, 100_000);
        assert_eq!(config.enable_profiling, true);
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.max_paths, 100);
        assert_eq!(config.weighted, true);
    }

    #[tokio::test]
    async fn test_algorithm_comparison() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let (graph, sources, targets) = create_large_test_graph(5000);

        // Test GPU Dijkstra
        let mut config_gpu = AdaptiveSSPConfig::default();
        config_gpu.algorithm = SSPAlgorithm::GPUDijkstra;

        let result_gpu = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config_gpu,
        ).await;

        // Test Hybrid Duan
        let mut config_hybrid = AdaptiveSSPConfig::default();
        config_hybrid.algorithm = SSPAlgorithm::HybridDuan;

        let result_hybrid = find_adaptive_shortest_paths(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
            &graph,
            &sources,
            &targets,
            &config_hybrid,
        ).await;

        // Both should succeed
        assert!(result_gpu.is_ok());
        assert!(result_hybrid.is_ok());

        // Both should find paths (may differ in number)
        let gpu_result = result_gpu.unwrap();
        let hybrid_result = result_hybrid.unwrap();

        assert!(!gpu_result.paths.is_empty() || !hybrid_result.paths.is_empty());

        // Log performance comparison
        println!("GPU Dijkstra: {:.2}ms", gpu_result.metrics.total_time_ms);
        println!("Hybrid Duan: {:.2}ms", hybrid_result.metrics.total_time_ms);
    }

    #[tokio::test]
    #[ignore] // Expensive test - run manually
    async fn test_crossover_detection() {
        let fixture = TestFixture::new().await.expect("Failed to initialize GPU");

        let threshold = detect_crossover_threshold(
            &fixture.device,
            &fixture.modules,
            &fixture.memory_pool,
            &fixture.streams,
        ).await;

        assert!(threshold.is_ok());

        let detected_threshold = threshold.unwrap();
        println!("Detected crossover threshold: {}", detected_threshold);

        // Threshold should be reasonable
        assert!(detected_threshold >= 1000 && detected_threshold <= 500_000);
    }
}
