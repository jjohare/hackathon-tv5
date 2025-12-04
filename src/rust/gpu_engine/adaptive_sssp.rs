/// Adaptive Single-Source Shortest Path (SSSP) Algorithm Selection
///
/// Provides intelligent algorithm selection between GPU Dijkstra and Hybrid Duan
/// based on graph characteristics, with automatic crossover detection and
/// performance profiling.

use cudarc::driver::CudaDevice;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::*;
use super::pathfinding::{Path, PathfindingConfig, SearchAlgorithm};

/// SSSP algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SSPAlgorithm {
    /// Pure GPU Dijkstra - optimal for small to medium graphs
    GPUDijkstra,

    /// Hybrid CPU-GPU Duan algorithm - scales for large graphs
    HybridDuan,

    /// Automatic selection based on graph characteristics
    Auto,
}

impl Default for SSPAlgorithm {
    fn default() -> Self {
        Self::Auto
    }
}

/// Configuration for adaptive SSSP execution
#[derive(Debug, Clone)]
pub struct AdaptiveSSPConfig {
    /// Algorithm selection strategy
    pub algorithm: SSPAlgorithm,

    /// Node count threshold for algorithm switching (Auto mode)
    /// Graphs with fewer nodes use GPU Dijkstra, larger use Hybrid Duan
    pub crossover_threshold: usize,

    /// Enable runtime profiling and metrics collection
    pub enable_profiling: bool,

    /// Maximum path length to search
    pub max_depth: usize,

    /// Maximum number of paths to return
    pub max_paths: usize,

    /// Whether to use edge weights
    pub weighted: bool,

    /// Hybrid algorithm: number of CPU threads for frontier expansion
    pub hybrid_cpu_threads: usize,

    /// Hybrid algorithm: batch size for GPU processing
    pub hybrid_batch_size: usize,
}

impl Default for AdaptiveSSPConfig {
    fn default() -> Self {
        Self {
            algorithm: SSPAlgorithm::Auto,
            crossover_threshold: 100_000, // 100K nodes
            enable_profiling: true,
            max_depth: 10,
            max_paths: 100,
            weighted: true,
            hybrid_cpu_threads: 4,
            hybrid_batch_size: 10_000,
        }
    }
}

impl AdaptiveSSPConfig {
    /// Convert to PathfindingConfig for compatibility
    pub fn to_pathfinding_config(&self) -> PathfindingConfig {
        PathfindingConfig {
            max_depth: self.max_depth,
            max_paths: self.max_paths,
            algorithm: SearchAlgorithm::Dijkstra,
            weighted: self.weighted,
        }
    }
}

/// Performance metrics for algorithm execution
#[derive(Debug, Clone, Default)]
pub struct SSPMetrics {
    /// Algorithm that was used
    pub algorithm_used: String,

    /// Total execution time (ms)
    pub total_time_ms: f64,

    /// GPU compute time (ms)
    pub gpu_compute_ms: f64,

    /// CPU compute time (ms)
    pub cpu_compute_ms: f64,

    /// Memory transfer time (ms)
    pub transfer_time_ms: f64,

    /// Number of nodes processed
    pub nodes_processed: usize,

    /// Number of edges traversed
    pub edges_traversed: usize,

    /// Number of paths found
    pub paths_found: usize,

    /// Peak GPU memory used (bytes)
    pub peak_gpu_memory: usize,

    /// Average path length
    pub avg_path_length: f32,
}

impl SSPMetrics {
    /// Calculate throughput in nodes per second
    pub fn throughput_nodes_per_sec(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.nodes_processed as f64 / self.total_time_ms) * 1000.0
        } else {
            0.0
        }
    }

    /// Calculate GPU utilization percentage
    pub fn gpu_utilization(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.gpu_compute_ms / self.total_time_ms) * 100.0
        } else {
            0.0
        }
    }
}

/// Result of adaptive SSSP execution
pub struct SSPResult {
    /// Shortest paths found
    pub paths: Vec<Path>,

    /// Performance metrics
    pub metrics: SSPMetrics,
}

/// Find shortest paths with adaptive algorithm selection
pub async fn find_adaptive_shortest_paths(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &AdaptiveSSPConfig,
) -> GpuResult<SSPResult> {
    let start = std::time::Instant::now();

    // Estimate graph size
    let num_nodes = estimate_node_count(graph, sources, targets);
    let num_edges = graph.len() / 2;

    // Select algorithm
    let selected_algorithm = select_algorithm(config.algorithm, num_nodes, config.crossover_threshold);

    tracing::info!(
        "Adaptive SSSP: graph={} nodes, {} edges, algorithm={:?}",
        num_nodes, num_edges, selected_algorithm
    );

    // Execute selected algorithm
    let (paths, mut metrics) = match selected_algorithm {
        SSPAlgorithm::GPUDijkstra => {
            execute_gpu_dijkstra(
                device,
                modules,
                memory_pool,
                streams,
                graph,
                sources,
                targets,
                config,
                num_nodes,
                num_edges,
            ).await?
        }
        SSPAlgorithm::HybridDuan => {
            execute_hybrid_duan(
                device,
                modules,
                memory_pool,
                streams,
                graph,
                sources,
                targets,
                config,
                num_nodes,
                num_edges,
            ).await?
        }
        SSPAlgorithm::Auto => {
            unreachable!("Auto should be resolved by select_algorithm")
        }
    };

    // Finalize metrics
    metrics.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics.nodes_processed = num_nodes;
    metrics.edges_traversed = num_edges;
    metrics.paths_found = paths.len();

    if !paths.is_empty() {
        metrics.avg_path_length = paths.iter()
            .map(|p| p.length as f32)
            .sum::<f32>() / paths.len() as f32;
    }

    if config.enable_profiling {
        log_performance_metrics(&metrics);
    }

    Ok(SSPResult { paths, metrics })
}

/// Select algorithm based on graph size and configuration
fn select_algorithm(
    algorithm: SSPAlgorithm,
    num_nodes: usize,
    crossover_threshold: usize,
) -> SSPAlgorithm {
    match algorithm {
        SSPAlgorithm::Auto => {
            if num_nodes < crossover_threshold {
                SSPAlgorithm::GPUDijkstra
            } else {
                SSPAlgorithm::HybridDuan
            }
        }
        other => other,
    }
}

/// Estimate node count from graph data
fn estimate_node_count(graph: &[u32], sources: &[u32], targets: &[u32]) -> usize {
    let max_from_graph = graph.iter().max().copied().unwrap_or(0) as usize;
    let max_from_sources = sources.iter().max().copied().unwrap_or(0) as usize;
    let max_from_targets = targets.iter().max().copied().unwrap_or(0) as usize;

    max_from_graph.max(max_from_sources).max(max_from_targets) + 1
}

/// Execute GPU Dijkstra algorithm
async fn execute_gpu_dijkstra(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &AdaptiveSSPConfig,
    num_nodes: usize,
    num_edges: usize,
) -> GpuResult<(Vec<Path>, SSPMetrics)> {
    let start = std::time::Instant::now();
    let mut metrics = SSPMetrics {
        algorithm_used: "GPU Dijkstra".to_string(),
        ..Default::default()
    };

    // Acquire stream
    let stream = streams.acquire().await?;

    // Generate edge weights
    let weights = vec![1.0f32; num_edges];

    // Allocate device memory
    let transfer_start = std::time::Instant::now();

    let mut d_graph = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(graph.len())?
    };

    let mut d_weights = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(weights.len())?
    };

    let mut d_sources = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(sources.len())?
    };

    let mut d_distances = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(num_nodes)?
    };

    let mut d_predecessors = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(num_nodes)?
    };

    metrics.peak_gpu_memory = (graph.len() * 4) + (weights.len() * 4) +
                               (sources.len() * 4) + (num_nodes * 8);

    // Transfer inputs
    device.htod_copy_into(graph, &mut d_graph)?;
    device.htod_copy_into(&weights, &mut d_weights)?;
    device.htod_copy_into(sources, &mut d_sources)?;

    // Initialize distances to infinity
    let init_distances = vec![f32::INFINITY; num_nodes];
    device.htod_copy_into(&init_distances, &mut d_distances)?;

    metrics.transfer_time_ms = transfer_start.elapsed().as_secs_f64() * 1000.0;

    // Launch Dijkstra kernel
    let compute_start = std::time::Instant::now();

    modules.launch_dijkstra(
        &d_graph,
        &d_weights,
        &d_sources,
        &mut d_distances,
        &mut d_predecessors,
        num_nodes as u32,
        num_edges as u32,
    )?;

    // Synchronize
    stream.synchronize().await?;

    metrics.gpu_compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

    // Transfer results back
    let distances_f32 = d_distances.dtoh()?;
    let predecessors = d_predecessors.dtoh()?;

    // Convert distances for reconstruction
    let distances: Vec<u32> = distances_f32.iter()
        .map(|&d| if d.is_finite() { d as u32 } else { u32::MAX })
        .collect();

    // Reconstruct paths
    let pathfinding_config = config.to_pathfinding_config();
    let mut paths = super::pathfinding::reconstruct_paths(
        &distances,
        &predecessors,
        sources,
        targets,
        &pathfinding_config,
    );

    // Add actual costs from float distances
    for (i, path) in paths.iter_mut().enumerate() {
        if i < targets.len() {
            let target = targets[i] as usize;
            if target < distances_f32.len() {
                path.cost = distances_f32[target];
            }
        }
    }

    // Free memory
    {
        let mut pool = memory_pool.write().await;
        pool.free(d_graph);
        pool.free(d_weights);
        pool.free(d_sources);
        pool.free(d_distances);
        pool.free(d_predecessors);
    }

    metrics.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok((paths, metrics))
}

/// Execute Hybrid Duan SSSP algorithm
/// Combines CPU frontier expansion with GPU batch processing
async fn execute_hybrid_duan(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &AdaptiveSSPConfig,
    num_nodes: usize,
    num_edges: usize,
) -> GpuResult<(Vec<Path>, SSPMetrics)> {
    let start = std::time::Instant::now();
    let mut metrics = SSPMetrics {
        algorithm_used: "Hybrid Duan".to_string(),
        ..Default::default()
    };

    // Build adjacency list on CPU for frontier expansion
    let adj_list = build_adjacency_list(graph, num_nodes);
    let weights = vec![1.0f32; num_edges];

    // Initialize distances and predecessors on CPU
    let mut distances = vec![f32::INFINITY; num_nodes];
    let mut predecessors = vec![u32::MAX; num_nodes];

    // Initialize source nodes
    for &source in sources {
        if (source as usize) < num_nodes {
            distances[source as usize] = 0.0;
        }
    }

    // Hybrid algorithm: CPU-managed frontier with GPU batch processing
    let mut frontier = sources.to_vec();
    let mut visited = vec![false; num_nodes];
    let mut iteration = 0;

    while !frontier.is_empty() && iteration < config.max_depth {
        iteration += 1;

        // Mark frontier nodes as visited
        for &node in &frontier {
            if (node as usize) < visited.len() {
                visited[node as usize] = true;
            }
        }

        // CPU frontier expansion for small batches
        if frontier.len() < config.hybrid_batch_size {
            let cpu_start = std::time::Instant::now();
            frontier = expand_frontier_cpu(
                &adj_list,
                &weights,
                &frontier,
                &mut distances,
                &mut predecessors,
                &visited,
            );
            metrics.cpu_compute_ms += cpu_start.elapsed().as_secs_f64() * 1000.0;
        } else {
            // GPU batch processing for large frontiers
            let gpu_start = std::time::Instant::now();
            frontier = expand_frontier_gpu(
                device,
                modules,
                memory_pool,
                streams,
                graph,
                &weights,
                &frontier,
                &mut distances,
                &mut predecessors,
                &visited,
                num_nodes,
            ).await?;
            metrics.gpu_compute_ms += gpu_start.elapsed().as_secs_f64() * 1000.0;
        }
    }

    // Convert distances for reconstruction
    let distances_u32: Vec<u32> = distances.iter()
        .map(|&d| if d.is_finite() { d as u32 } else { u32::MAX })
        .collect();

    // Reconstruct paths
    let pathfinding_config = config.to_pathfinding_config();
    let mut paths = super::pathfinding::reconstruct_paths(
        &distances_u32,
        &predecessors,
        sources,
        targets,
        &pathfinding_config,
    );

    // Add actual costs
    for (i, path) in paths.iter_mut().enumerate() {
        if i < targets.len() {
            let target = targets[i] as usize;
            if target < distances.len() {
                path.cost = distances[target];
            }
        }
    }

    metrics.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok((paths, metrics))
}

/// Build adjacency list from edge list
fn build_adjacency_list(graph: &[u32], num_nodes: usize) -> Vec<Vec<(u32, usize)>> {
    let mut adj_list = vec![Vec::new(); num_nodes];

    for edge_idx in (0..graph.len()).step_by(2) {
        if edge_idx + 1 < graph.len() {
            let from = graph[edge_idx] as usize;
            let to = graph[edge_idx + 1];

            if from < num_nodes {
                adj_list[from].push((to, edge_idx / 2));
            }
        }
    }

    adj_list
}

/// Expand frontier on CPU (for small batches)
fn expand_frontier_cpu(
    adj_list: &[Vec<(u32, usize)>],
    weights: &[f32],
    frontier: &[u32],
    distances: &mut [f32],
    predecessors: &mut [u32],
    visited: &[bool],
) -> Vec<u32> {
    let mut next_frontier = Vec::new();

    for &node in frontier {
        let node_idx = node as usize;
        if node_idx >= adj_list.len() {
            continue;
        }

        let current_dist = distances[node_idx];

        for &(neighbor, edge_idx) in &adj_list[node_idx] {
            let neighbor_idx = neighbor as usize;
            if neighbor_idx >= distances.len() || visited[neighbor_idx] {
                continue;
            }

            let edge_weight = if edge_idx < weights.len() {
                weights[edge_idx]
            } else {
                1.0
            };

            let new_dist = current_dist + edge_weight;

            if new_dist < distances[neighbor_idx] {
                distances[neighbor_idx] = new_dist;
                predecessors[neighbor_idx] = node;
                next_frontier.push(neighbor);
            }
        }
    }

    // Remove duplicates
    next_frontier.sort_unstable();
    next_frontier.dedup();
    next_frontier
}

/// Expand frontier on GPU (for large batches)
async fn expand_frontier_gpu(
    _device: &Arc<CudaDevice>,
    _modules: &Arc<KernelModules>,
    _memory_pool: &Arc<RwLock<MemoryPool>>,
    _streams: &Arc<StreamManager>,
    _graph: &[u32],
    _weights: &[f32],
    frontier: &[u32],
    _distances: &mut [f32],
    _predecessors: &mut [u32],
    _visited: &[bool],
    _num_nodes: usize,
) -> GpuResult<Vec<u32>> {
    // TODO: Implement GPU frontier expansion kernel
    // For now, fall back to empty frontier to terminate
    // In production, this would launch a custom CUDA kernel for parallel expansion

    tracing::warn!(
        "GPU frontier expansion not yet implemented, frontier size: {}",
        frontier.len()
    );

    Ok(Vec::new())
}

/// Log performance metrics
fn log_performance_metrics(metrics: &SSPMetrics) {
    tracing::info!(
        "SSSP Performance: algorithm={}, time={:.2}ms, throughput={:.0} nodes/sec",
        metrics.algorithm_used,
        metrics.total_time_ms,
        metrics.throughput_nodes_per_sec()
    );

    tracing::debug!(
        "  GPU: {:.2}ms ({:.1}%), CPU: {:.2}ms, Transfer: {:.2}ms",
        metrics.gpu_compute_ms,
        metrics.gpu_utilization(),
        metrics.cpu_compute_ms,
        metrics.transfer_time_ms
    );

    tracing::debug!(
        "  Paths: {}, Avg length: {:.1}, Memory: {:.2}MB",
        metrics.paths_found,
        metrics.avg_path_length,
        metrics.peak_gpu_memory as f64 / 1_048_576.0
    );
}

/// Automatic crossover threshold detection
/// Runs both algorithms on sample graphs to find optimal threshold
pub async fn detect_crossover_threshold(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
) -> GpuResult<usize> {
    tracing::info("Detecting optimal crossover threshold...");

    // Test graph sizes (in nodes)
    let test_sizes = vec![1000, 5000, 10_000, 50_000, 100_000, 200_000];
    let mut crossover_threshold = 100_000; // Default fallback

    for size in test_sizes {
        // Generate synthetic graph
        let (graph, sources, targets) = generate_test_graph(size);

        // Test GPU Dijkstra
        let mut config_gpu = AdaptiveSSPConfig::default();
        config_gpu.algorithm = SSPAlgorithm::GPUDijkstra;
        config_gpu.enable_profiling = false;

        let result_gpu = find_adaptive_shortest_paths(
            device,
            modules,
            memory_pool,
            streams,
            &graph,
            &sources,
            &targets,
            &config_gpu,
        ).await;

        // Test Hybrid Duan
        let mut config_hybrid = AdaptiveSSPConfig::default();
        config_hybrid.algorithm = SSPAlgorithm::HybridDuan;
        config_hybrid.enable_profiling = false;

        let result_hybrid = find_adaptive_shortest_paths(
            device,
            modules,
            memory_pool,
            streams,
            &graph,
            &sources,
            &targets,
            &config_hybrid,
        ).await;

        // Compare performance
        if let (Ok(gpu), Ok(hybrid)) = (result_gpu, result_hybrid) {
            tracing::debug!(
                "Size {}: GPU={:.2}ms, Hybrid={:.2}ms",
                size,
                gpu.metrics.total_time_ms,
                hybrid.metrics.total_time_ms
            );

            // Find crossover point where hybrid becomes faster
            if hybrid.metrics.total_time_ms < gpu.metrics.total_time_ms {
                crossover_threshold = size;
                break;
            }
        }
    }

    tracing::info!("Detected crossover threshold: {} nodes", crossover_threshold);
    Ok(crossover_threshold)
}

/// Generate synthetic test graph
fn generate_test_graph(num_nodes: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let edges_per_node = 10;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_selection() {
        let small_graph = 50_000;
        let large_graph = 150_000;
        let threshold = 100_000;

        assert_eq!(
            select_algorithm(SSPAlgorithm::Auto, small_graph, threshold),
            SSPAlgorithm::GPUDijkstra
        );

        assert_eq!(
            select_algorithm(SSPAlgorithm::Auto, large_graph, threshold),
            SSPAlgorithm::HybridDuan
        );

        assert_eq!(
            select_algorithm(SSPAlgorithm::GPUDijkstra, large_graph, threshold),
            SSPAlgorithm::GPUDijkstra
        );
    }

    #[test]
    fn test_node_count_estimation() {
        let graph = vec![0, 1, 1, 2, 2, 3, 3, 4];
        let sources = vec![0];
        let targets = vec![4];

        assert_eq!(estimate_node_count(&graph, &sources, &targets), 5);
    }

    #[test]
    fn test_adjacency_list_construction() {
        let graph = vec![0, 1, 0, 2, 1, 2, 2, 3];
        let num_nodes = 4;

        let adj_list = build_adjacency_list(&graph, num_nodes);

        assert_eq!(adj_list.len(), 4);
        assert_eq!(adj_list[0].len(), 2); // Node 0 has 2 edges
        assert_eq!(adj_list[1].len(), 1); // Node 1 has 1 edge
        assert_eq!(adj_list[2].len(), 1); // Node 2 has 1 edge
    }

    #[test]
    fn test_frontier_expansion_cpu() {
        let adj_list = vec![
            vec![(1, 0), (2, 1)],
            vec![(2, 2), (3, 3)],
            vec![(3, 4)],
            vec![],
        ];

        let weights = vec![1.0, 2.0, 1.0, 3.0, 1.0];
        let frontier = vec![0];
        let mut distances = vec![0.0, f32::INFINITY, f32::INFINITY, f32::INFINITY];
        let mut predecessors = vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        let visited = vec![false; 4];

        let next_frontier = expand_frontier_cpu(
            &adj_list,
            &weights,
            &frontier,
            &mut distances,
            &mut predecessors,
            &visited,
        );

        assert!(next_frontier.contains(&1));
        assert!(next_frontier.contains(&2));
        assert_eq!(distances[1], 1.0);
        assert_eq!(distances[2], 2.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut metrics = SSPMetrics {
            algorithm_used: "Test".to_string(),
            total_time_ms: 100.0,
            gpu_compute_ms: 70.0,
            nodes_processed: 10000,
            ..Default::default()
        };

        assert_eq!(metrics.throughput_nodes_per_sec(), 100_000.0);
        assert_eq!(metrics.gpu_utilization(), 70.0);

        metrics.total_time_ms = 0.0;
        assert_eq!(metrics.throughput_nodes_per_sec(), 0.0);
    }

    #[test]
    fn test_config_conversion() {
        let config = AdaptiveSSPConfig {
            max_depth: 20,
            max_paths: 50,
            weighted: true,
            ..Default::default()
        };

        let pf_config = config.to_pathfinding_config();
        assert_eq!(pf_config.max_depth, 20);
        assert_eq!(pf_config.max_paths, 50);
        assert_eq!(pf_config.weighted, true);
    }
}
