/// Graph Pathfinding Operations
///
/// GPU-accelerated graph search algorithms for knowledge graph traversal.

use cudarc::driver::CudaDevice;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::*;

/// Configuration for pathfinding operations
#[derive(Debug, Clone)]
pub struct PathfindingConfig {
    /// Maximum path length to search
    pub max_depth: usize,

    /// Maximum number of paths to return per query
    pub max_paths: usize,

    /// Search algorithm
    pub algorithm: SearchAlgorithm,

    /// Whether to consider edge weights
    pub weighted: bool,
}

impl Default for PathfindingConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_paths: 100,
            algorithm: SearchAlgorithm::BFS,
            weighted: false,
        }
    }
}

/// Graph search algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchAlgorithm {
    BFS,
    Dijkstra,
    AStar,
}

/// A path through the graph
#[derive(Debug, Clone)]
pub struct Path {
    /// Node IDs in path order
    pub nodes: Vec<u32>,

    /// Total path cost (if weighted)
    pub cost: f32,

    /// Path length (number of edges)
    pub length: usize,
}

impl Path {
    /// Check if path contains a node
    pub fn contains(&self, node: u32) -> bool {
        self.nodes.contains(&node)
    }

    /// Get path as edge list
    pub fn edges(&self) -> Vec<(u32, u32)> {
        self.nodes.windows(2).map(|w| (w[0], w[1])).collect()
    }
}

/// Find shortest paths in knowledge graph
pub async fn find_shortest_paths(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
) -> GpuResult<Vec<Path>> {
    match config.algorithm {
        SearchAlgorithm::BFS => {
            find_paths_bfs(
                device,
                modules,
                memory_pool,
                streams,
                graph,
                sources,
                targets,
                config,
            ).await
        }
        SearchAlgorithm::Dijkstra => {
            find_paths_dijkstra(
                device,
                modules,
                memory_pool,
                streams,
                graph,
                sources,
                targets,
                config,
            ).await
        }
        SearchAlgorithm::AStar => {
            // A* would require heuristic function
            // Fall back to Dijkstra for now
            find_paths_dijkstra(
                device,
                modules,
                memory_pool,
                streams,
                graph,
                sources,
                targets,
                config,
            ).await
        }
    }
}

/// BFS-based pathfinding
async fn find_paths_bfs(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
) -> GpuResult<Vec<Path>> {
    let start = std::time::Instant::now();

    // Acquire stream
    let stream = streams.acquire().await?;

    // Estimate number of nodes from graph size
    let num_nodes = (graph.len() / 2).max(sources.iter().max().unwrap_or(&0) + 1) as usize;
    let num_edges = graph.len() / 2;

    // Allocate device memory
    let mut d_graph = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(graph.len())?
    };

    let mut d_sources = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(sources.len())?
    };

    let mut d_distances = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(num_nodes)?
    };

    let mut d_predecessors = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<u32>(num_nodes)?
    };

    // Transfer inputs
    device.htod_copy_into(graph, &mut d_graph)?;
    device.htod_copy_into(sources, &mut d_sources)?;

    // Initialize distances to max
    let init_distances = vec![u32::MAX; num_nodes];
    device.htod_copy_into(&init_distances, &mut d_distances)?;

    // Launch BFS kernel
    modules.launch_bfs(
        &d_graph,
        &d_sources,
        &mut d_distances,
        &mut d_predecessors,
        num_nodes as u32,
        num_edges as u32,
    )?;

    // Synchronize
    stream.synchronize().await?;

    // Transfer results
    let distances = d_distances.dtoh()?;
    let predecessors = d_predecessors.dtoh()?;

    // Reconstruct paths
    let paths = reconstruct_paths(&distances, &predecessors, sources, targets, config);

    // Free memory
    {
        let mut pool = memory_pool.write().await;
        pool.free(d_graph);
        pool.free(d_sources);
        pool.free(d_distances);
        pool.free(d_predecessors);
    }

    tracing::debug!(
        "BFS pathfinding completed in {:.2}ms, found {} paths",
        start.elapsed().as_secs_f64() * 1000.0,
        paths.len()
    );

    Ok(paths)
}

/// Dijkstra's algorithm pathfinding
async fn find_paths_dijkstra(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
) -> GpuResult<Vec<Path>> {
    let start = std::time::Instant::now();

    // Acquire stream
    let stream = streams.acquire().await?;

    let num_nodes = (graph.len() / 2).max(sources.iter().max().unwrap_or(&0) + 1) as usize;
    let num_edges = graph.len() / 2;

    // Generate dummy weights (would come from graph in real implementation)
    let weights = vec![1.0f32; num_edges];

    // Allocate device memory
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

    // Transfer inputs
    device.htod_copy_into(graph, &mut d_graph)?;
    device.htod_copy_into(&weights, &mut d_weights)?;
    device.htod_copy_into(sources, &mut d_sources)?;

    // Initialize distances to infinity
    let init_distances = vec![f32::INFINITY; num_nodes];
    device.htod_copy_into(&init_distances, &mut d_distances)?;

    // Launch Dijkstra kernel
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

    // Transfer results
    let distances_f32 = d_distances.dtoh()?;
    let predecessors = d_predecessors.dtoh()?;

    // Convert float distances to u32 for reconstruction
    let distances: Vec<u32> = distances_f32.iter()
        .map(|&d| if d.is_finite() { d as u32 } else { u32::MAX })
        .collect();

    // Reconstruct paths
    let mut paths = reconstruct_paths(&distances, &predecessors, sources, targets, config);

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

    tracing::debug!(
        "Dijkstra pathfinding completed in {:.2}ms, found {} paths",
        start.elapsed().as_secs_f64() * 1000.0,
        paths.len()
    );

    Ok(paths)
}

/// Reconstruct paths from BFS/Dijkstra results
fn reconstruct_paths(
    distances: &[u32],
    predecessors: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
) -> Vec<Path> {
    let mut paths = Vec::new();

    for &target in targets {
        let target_idx = target as usize;

        if target_idx >= distances.len() || distances[target_idx] == u32::MAX {
            continue; // No path found
        }

        // Backtrack from target to source
        let mut nodes = vec![target];
        let mut current = target_idx;

        while current < predecessors.len() && predecessors[current] != u32::MAX {
            let pred = predecessors[current];
            nodes.push(pred);
            current = pred as usize;

            if nodes.len() > config.max_depth {
                break; // Path too long
            }
        }

        // Check if we reached a source
        if sources.contains(&nodes[nodes.len() - 1]) {
            nodes.reverse();

            paths.push(Path {
                length: nodes.len() - 1,
                cost: distances[target_idx] as f32,
                nodes,
            });

            if paths.len() >= config.max_paths {
                break;
            }
        }
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_operations() {
        let path = Path {
            nodes: vec![1, 2, 3, 4],
            cost: 3.0,
            length: 3,
        };

        assert!(path.contains(2));
        assert!(!path.contains(5));

        let edges = path.edges();
        assert_eq!(edges, vec![(1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_reconstruct_paths() {
        let distances = vec![0, 1, 2, u32::MAX];
        let predecessors = vec![u32::MAX, 0, 1, u32::MAX];
        let sources = vec![0];
        let targets = vec![2];
        let config = PathfindingConfig::default();

        let paths = reconstruct_paths(&distances, &predecessors, &sources, &targets, &config);

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].nodes, vec![0, 1, 2]);
        assert_eq!(paths[0].length, 2);
    }
}
