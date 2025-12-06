/// Example: Adaptive SSSP Algorithm Selection
///
/// Demonstrates how to use the adaptive shortest path algorithm selection
/// to automatically choose between GPU Dijkstra and Hybrid Duan based on
/// graph characteristics.

use std::sync::Arc;
use tokio::sync::RwLock;

// Note: This assumes the module structure matches your project
// Adjust imports as needed
use crate::gpu_engine::{
    GpuResult, SSPAlgorithm, AdaptiveSSPConfig, SSPMetrics,
    find_adaptive_shortest_paths, detect_crossover_threshold,
};
use cudarc::driver::CudaDevice;

/// Example 1: Automatic algorithm selection
async fn example_auto_selection() -> GpuResult<()> {
    println!("Example 1: Automatic Algorithm Selection");
    println!("=========================================\n");

    // Initialize GPU engine
    let device = Arc::new(CudaDevice::new(0)?);
    let modules = Arc::new(crate::gpu_engine::KernelModules::load(&device)?);
    let memory_pool = Arc::new(RwLock::new(crate::gpu_engine::MemoryPool::new(device.clone())));
    let streams = Arc::new(crate::gpu_engine::StreamManager::new(device.clone(), 4).await?);

    // Create a sample graph (10,000 nodes)
    let (graph, sources, targets) = create_sample_graph(10_000);

    // Configure for automatic selection
    let config = AdaptiveSSPConfig {
        algorithm: SSPAlgorithm::Auto,
        crossover_threshold: 50_000, // Switch to Hybrid at 50K nodes
        enable_profiling: true,
        max_depth: 20,
        max_paths: 100,
        weighted: true,
        ..Default::default()
    };

    println!("Graph: {} nodes, {} edges", 10_000, graph.len() / 2);
    println!("Config: Auto selection with threshold={}", config.crossover_threshold);
    println!();

    // Execute SSSP
    let result = find_adaptive_shortest_paths(
        &device,
        &modules,
        &memory_pool,
        &streams,
        &graph,
        &sources,
        &targets,
        &config,
    ).await?;

    // Display results
    print_results(&result.paths, &result.metrics);

    Ok(())
}

/// Example 2: Force GPU Dijkstra
async fn example_gpu_dijkstra() -> GpuResult<()> {
    println!("\nExample 2: Force GPU Dijkstra");
    println!("=============================\n");

    // Initialize GPU engine (same as above)
    let device = Arc::new(CudaDevice::new(0)?);
    let modules = Arc::new(crate::gpu_engine::KernelModules::load(&device)?);
    let memory_pool = Arc::new(RwLock::new(crate::gpu_engine::MemoryPool::new(device.clone())));
    let streams = Arc::new(crate::gpu_engine::StreamManager::new(device.clone(), 4).await?);

    // Create a small graph
    let (graph, sources, targets) = create_sample_graph(5_000);

    // Force GPU Dijkstra
    let config = AdaptiveSSPConfig {
        algorithm: SSPAlgorithm::GPUDijkstra,
        enable_profiling: true,
        ..Default::default()
    };

    println!("Graph: {} nodes, {} edges", 5_000, graph.len() / 2);
    println!("Algorithm: GPU Dijkstra (forced)");
    println!();

    let result = find_adaptive_shortest_paths(
        &device,
        &modules,
        &memory_pool,
        &streams,
        &graph,
        &sources,
        &targets,
        &config,
    ).await?;

    print_results(&result.paths, &result.metrics);

    Ok(())
}

/// Example 3: Force Hybrid Duan
async fn example_hybrid_duan() -> GpuResult<()> {
    println!("\nExample 3: Force Hybrid Duan");
    println!("============================\n");

    // Initialize GPU engine
    let device = Arc::new(CudaDevice::new(0)?);
    let modules = Arc::new(crate::gpu_engine::KernelModules::load(&device)?);
    let memory_pool = Arc::new(RwLock::new(crate::gpu_engine::MemoryPool::new(device.clone())));
    let streams = Arc::new(crate::gpu_engine::StreamManager::new(device.clone(), 4).await?);

    // Create a large graph
    let (graph, sources, targets) = create_sample_graph(200_000);

    // Force Hybrid Duan
    let config = AdaptiveSSPConfig {
        algorithm: SSPAlgorithm::HybridDuan,
        enable_profiling: true,
        hybrid_cpu_threads: 8,
        hybrid_batch_size: 20_000,
        ..Default::default()
    };

    println!("Graph: {} nodes, {} edges", 200_000, graph.len() / 2);
    println!("Algorithm: Hybrid Duan (forced)");
    println!("CPU threads: {}, Batch size: {}", config.hybrid_cpu_threads, config.hybrid_batch_size);
    println!();

    let result = find_adaptive_shortest_paths(
        &device,
        &modules,
        &memory_pool,
        &streams,
        &graph,
        &sources,
        &targets,
        &config,
    ).await?;

    print_results(&result.paths, &result.metrics);

    Ok(())
}

/// Example 4: Detect optimal crossover threshold
async fn example_crossover_detection() -> GpuResult<()> {
    println!("\nExample 4: Crossover Threshold Detection");
    println!("========================================\n");

    // Initialize GPU engine
    let device = Arc::new(CudaDevice::new(0)?);
    let modules = Arc::new(crate::gpu_engine::KernelModules::load(&device)?);
    let memory_pool = Arc::new(RwLock::new(crate::gpu_engine::MemoryPool::new(device.clone())));
    let streams = Arc::new(crate::gpu_engine::StreamManager::new(device.clone(), 4).await?);

    println!("Running benchmarks to detect optimal crossover threshold...");
    println!("This may take several minutes...\n");

    let threshold = detect_crossover_threshold(
        &device,
        &modules,
        &memory_pool,
        &streams,
    ).await?;

    println!("Detected optimal threshold: {} nodes", threshold);
    println!();
    println!("Recommendation:");
    println!("  - Graphs with < {} nodes: Use GPU Dijkstra", threshold);
    println!("  - Graphs with >= {} nodes: Use Hybrid Duan", threshold);

    Ok(())
}

/// Example 5: Compare algorithms
async fn example_algorithm_comparison() -> GpuResult<()> {
    println!("\nExample 5: Algorithm Comparison");
    println!("================================\n");

    // Initialize GPU engine
    let device = Arc::new(CudaDevice::new(0)?);
    let modules = Arc::new(crate::gpu_engine::KernelModules::load(&device)?);
    let memory_pool = Arc::new(RwLock::new(crate::gpu_engine::MemoryPool::new(device.clone())));
    let streams = Arc::new(crate::gpu_engine::StreamManager::new(device.clone(), 4).await?);

    // Test different graph sizes
    let sizes = vec![1_000, 10_000, 50_000, 100_000];

    println!("{:<12} {:<15} {:<15} {:<12}", "Graph Size", "GPU Dijkstra", "Hybrid Duan", "Winner");
    println!("{:-<60}", "");

    for size in sizes {
        let (graph, sources, targets) = create_sample_graph(size);

        // Test GPU Dijkstra
        let mut config_gpu = AdaptiveSSPConfig::default();
        config_gpu.algorithm = SSPAlgorithm::GPUDijkstra;
        config_gpu.enable_profiling = false;

        let result_gpu = find_adaptive_shortest_paths(
            &device,
            &modules,
            &memory_pool,
            &streams,
            &graph,
            &sources,
            &targets,
            &config_gpu,
        ).await?;

        // Test Hybrid Duan
        let mut config_hybrid = AdaptiveSSPConfig::default();
        config_hybrid.algorithm = SSPAlgorithm::HybridDuan;
        config_hybrid.enable_profiling = false;

        let result_hybrid = find_adaptive_shortest_paths(
            &device,
            &modules,
            &memory_pool,
            &streams,
            &graph,
            &sources,
            &targets,
            &config_hybrid,
        ).await?;

        let gpu_time = result_gpu.metrics.total_time_ms;
        let hybrid_time = result_hybrid.metrics.total_time_ms;
        let winner = if gpu_time < hybrid_time { "GPU" } else { "Hybrid" };

        println!(
            "{:<12} {:<15.2} {:<15.2} {:<12}",
            format!("{} nodes", size),
            format!("{:.2}ms", gpu_time),
            format!("{:.2}ms", hybrid_time),
            winner
        );
    }

    Ok(())
}

/// Helper: Create a sample graph
fn create_sample_graph(num_nodes: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let edges_per_node = 8;
    let num_edges = num_nodes * edges_per_node;

    let mut graph = Vec::with_capacity(num_edges * 2);

    for node in 0..num_nodes {
        for _ in 0..edges_per_node {
            let target = rng.gen_range(0..num_nodes) as u32;
            graph.push(node as u32);
            graph.push(target);
        }
    }

    let sources = vec![0];
    let targets = vec![(num_nodes - 1) as u32];

    (graph, sources, targets)
}

/// Helper: Print results
fn print_results(paths: &[crate::gpu_engine::Path], metrics: &SSPMetrics) {
    println!("Results:");
    println!("--------");
    println!("Algorithm used: {}", metrics.algorithm_used);
    println!("Paths found: {}", paths.len());
    println!("Execution time: {:.2}ms", metrics.total_time_ms);
    println!("  - GPU compute: {:.2}ms ({:.1}%)",
             metrics.gpu_compute_ms, metrics.gpu_utilization());
    println!("  - CPU compute: {:.2}ms", metrics.cpu_compute_ms);
    println!("  - Transfer: {:.2}ms", metrics.transfer_time_ms);
    println!("Throughput: {:.0} nodes/sec", metrics.throughput_nodes_per_sec());
    println!("Memory: {:.2}MB", metrics.peak_gpu_memory as f64 / 1_048_576.0);

    if !paths.is_empty() {
        println!("Average path length: {:.1}", metrics.avg_path_length);

        println!("\nSample paths:");
        for (i, path) in paths.iter().take(3).enumerate() {
            println!("  Path {}: length={}, cost={:.2}",
                     i + 1, path.length, path.cost);
            let nodes_preview = if path.nodes.len() > 10 {
                format!("{:?}...", &path.nodes[..10])
            } else {
                format!("{:?}", path.nodes)
            };
            println!("    Nodes: {}", nodes_preview);
        }
    }
}

/// Main entry point
#[tokio::main]
async fn main() -> GpuResult<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("Adaptive SSSP Algorithm Selection Examples");
    println!("==========================================\n");

    // Run examples
    example_auto_selection().await?;
    example_gpu_dijkstra().await?;
    example_hybrid_duan().await?;

    // Uncomment to run expensive examples:
    // example_crossover_detection().await?;
    // example_algorithm_comparison().await?;

    println!("\nAll examples completed successfully!");

    Ok(())
}
