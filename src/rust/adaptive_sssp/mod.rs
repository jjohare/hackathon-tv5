/// Adaptive SSSP Algorithm Selector
///
/// Intelligently selects the optimal shortest path algorithm based on graph
/// characteristics, providing seamless integration with recommendation engines.
///
/// Algorithms supported:
/// - GPU Dijkstra: Fast for small-medium graphs (<10M nodes)
/// - Duan et al.: Optimal for large graphs (>10M nodes), O(m log^(2/3) n)
/// - Landmark APSP: Precomputed distances for frequent queries
/// - Auto: Intelligent selection based on runtime analysis

use std::sync::Arc;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

pub mod gpu_dijkstra;
pub mod landmark_apsp;
pub mod metrics;

/// Algorithm selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgorithmMode {
    /// Automatically select best algorithm based on graph characteristics
    Auto,
    /// Force GPU-parallel Dijkstra (best for <10M nodes)
    GpuDijkstra,
    /// Force Landmark APSP with k-pivot approximation
    LandmarkApsp,
    /// Force Duan et al. O(m log^(2/3) n) algorithm (future: large graphs)
    #[allow(dead_code)]
    Duan,
}

impl Default for AlgorithmMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Configuration for adaptive SSSP
#[derive(Debug, Clone)]
pub struct AdaptiveSsspConfig {
    /// Algorithm selection mode
    pub mode: AlgorithmMode,

    /// Number of landmarks for APSP (default: 32)
    pub landmark_count: usize,

    /// Graph size threshold for algorithm switching (default: 10M nodes)
    pub large_graph_threshold: usize,

    /// Enable performance metrics collection
    pub collect_metrics: bool,
}

impl Default for AdaptiveSsspConfig {
    fn default() -> Self {
        Self {
            mode: AlgorithmMode::Auto,
            landmark_count: 32,
            large_graph_threshold: 10_000_000,
            collect_metrics: true,
        }
    }
}

/// Graph statistics for algorithm selection
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub is_sparse: bool,
}

impl GraphStats {
    pub fn new(num_nodes: usize, num_edges: usize) -> Self {
        let avg_degree = if num_nodes > 0 {
            num_edges as f32 / num_nodes as f32
        } else {
            0.0
        };

        // Graph is sparse if avg degree < log(n)
        let is_sparse = avg_degree < (num_nodes as f32).log2();

        Self {
            num_nodes,
            num_edges,
            avg_degree,
            is_sparse,
        }
    }
}

/// Algorithm execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsspMetrics {
    /// Which algorithm was actually used
    pub algorithm_used: String,

    /// Total execution time in milliseconds
    pub total_time_ms: f32,

    /// GPU kernel execution time (if applicable)
    pub gpu_time_ms: Option<f32>,

    /// Number of nodes processed
    pub nodes_processed: usize,

    /// Number of edges relaxed
    pub edges_relaxed: usize,

    /// Number of landmarks used (for APSP)
    pub landmarks_used: Option<usize>,

    /// Complexity factor (actual ops / theoretical ops)
    pub complexity_factor: Option<f32>,
}

impl Default for SsspMetrics {
    fn default() -> Self {
        Self {
            algorithm_used: "unknown".to_string(),
            total_time_ms: 0.0,
            gpu_time_ms: None,
            nodes_processed: 0,
            edges_relaxed: 0,
            landmarks_used: None,
            complexity_factor: None,
        }
    }
}

/// Result of shortest path computation
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Distances from source to all nodes
    pub distances: Vec<f32>,

    /// Execution metrics
    pub metrics: SsspMetrics,
}

/// Adaptive SSSP engine with intelligent algorithm selection
pub struct AdaptiveSsspEngine {
    config: AdaptiveSsspConfig,
    graph_stats: Option<GraphStats>,
    last_metrics: Arc<std::sync::RwLock<Option<SsspMetrics>>>,
}

impl AdaptiveSsspEngine {
    /// Create new adaptive SSSP engine
    pub fn new(config: AdaptiveSsspConfig) -> Self {
        Self {
            config,
            graph_stats: None,
            last_metrics: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    /// Update graph statistics for algorithm selection
    pub fn update_graph_stats(&mut self, num_nodes: usize, num_edges: usize) {
        self.graph_stats = Some(GraphStats::new(num_nodes, num_edges));
    }

    /// Select optimal algorithm based on graph characteristics
    pub fn select_algorithm(&self) -> AlgorithmMode {
        match self.config.mode {
            AlgorithmMode::Auto => {
                if let Some(stats) = &self.graph_stats {
                    self.auto_select(stats)
                } else {
                    // Default to GPU Dijkstra if no stats available
                    AlgorithmMode::GpuDijkstra
                }
            }
            mode => mode,
        }
    }

    /// Automatic algorithm selection logic
    fn auto_select(&self, stats: &GraphStats) -> AlgorithmMode {
        // Decision tree based on graph characteristics

        // For very large graphs (>10M nodes), use Landmark APSP
        if stats.num_nodes > self.config.large_graph_threshold {
            return AlgorithmMode::LandmarkApsp;
        }

        // For medium sparse graphs with high query frequency, use Landmark APSP
        if stats.is_sparse && stats.num_nodes > 100_000 {
            return AlgorithmMode::LandmarkApsp;
        }

        // For small-medium graphs, GPU Dijkstra is fastest
        AlgorithmMode::GpuDijkstra
    }

    /// Get the last execution metrics
    pub fn last_metrics(&self) -> Option<SsspMetrics> {
        self.last_metrics.read().unwrap().clone()
    }

    /// Record metrics from algorithm execution
    pub fn record_metrics(&self, metrics: SsspMetrics) {
        if self.config.collect_metrics {
            *self.last_metrics.write().unwrap() = Some(metrics);
        }
    }

    /// Get algorithm description for the selected mode
    pub fn algorithm_description(&self, mode: AlgorithmMode) -> &'static str {
        match mode {
            AlgorithmMode::Auto => "Auto-select based on graph characteristics",
            AlgorithmMode::GpuDijkstra => "GPU-parallel Dijkstra (O(m + n log n))",
            AlgorithmMode::LandmarkApsp => "Landmark APSP with k-pivot approximation",
            AlgorithmMode::Duan => "Duan et al. O(m log^(2/3) n) algorithm",
        }
    }

    /// Calculate theoretical complexity for comparison
    pub fn theoretical_complexity(&self, mode: AlgorithmMode) -> Option<f32> {
        if let Some(stats) = &self.graph_stats {
            let n = stats.num_nodes as f32;
            let m = stats.num_edges as f32;

            match mode {
                AlgorithmMode::GpuDijkstra => {
                    // O(m + n log n)
                    Some(m + n * n.log2())
                }
                AlgorithmMode::LandmarkApsp => {
                    // O(k * (m + n log n)) for k landmarks
                    let k = self.config.landmark_count as f32;
                    Some(k * (m + n * n.log2()))
                }
                AlgorithmMode::Duan => {
                    // O(m * log^(2/3) n)
                    Some(m * n.log2().powf(2.0 / 3.0))
                }
                AlgorithmMode::Auto => None,
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_selection() {
        let mut engine = AdaptiveSsspEngine::new(AdaptiveSsspConfig::default());

        // Small graph: should select GPU Dijkstra
        engine.update_graph_stats(10_000, 50_000);
        assert_eq!(engine.select_algorithm(), AlgorithmMode::GpuDijkstra);

        // Large graph: should select Landmark APSP
        engine.update_graph_stats(20_000_000, 100_000_000);
        assert_eq!(engine.select_algorithm(), AlgorithmMode::LandmarkApsp);
    }

    #[test]
    fn test_graph_stats() {
        let stats = GraphStats::new(10_000, 30_000);
        assert_eq!(stats.num_nodes, 10_000);
        assert_eq!(stats.num_edges, 30_000);
        assert_eq!(stats.avg_degree, 3.0);
    }

    #[test]
    fn test_theoretical_complexity() {
        let mut engine = AdaptiveSsspEngine::new(AdaptiveSsspConfig::default());
        engine.update_graph_stats(10_000, 50_000);

        let dijkstra_complexity = engine.theoretical_complexity(AlgorithmMode::GpuDijkstra);
        assert!(dijkstra_complexity.is_some());
        assert!(dijkstra_complexity.unwrap() > 0.0);
    }
}
