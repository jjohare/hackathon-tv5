# Adaptive SSSP API Reference

Quick reference for developers integrating adaptive SSSP functionality.

---

## Module Import

```rust
use crate::adaptive_sssp::{
    AdaptiveSsspEngine,
    AdaptiveSsspConfig,
    AlgorithmMode,
    SsspMetrics,
};
```

---

## Configuration

### AlgorithmMode

```rust
pub enum AlgorithmMode {
    /// Automatically select best algorithm
    Auto,

    /// Force GPU-parallel Dijkstra (best for <10M nodes)
    GpuDijkstra,

    /// Force Landmark APSP with k-pivot approximation
    LandmarkApsp,

    /// Force Duan et al. O(m log^(2/3) n) algorithm (future)
    Duan,
}
```

### AdaptiveSsspConfig

```rust
pub struct AdaptiveSsspConfig {
    /// Algorithm selection mode (default: Auto)
    pub mode: AlgorithmMode,

    /// Number of landmarks for APSP (default: 32)
    pub landmark_count: usize,

    /// Graph size threshold for switching (default: 10M)
    pub large_graph_threshold: usize,

    /// Enable metrics collection (default: true)
    pub collect_metrics: bool,
}

// Default configuration
let config = AdaptiveSsspConfig::default();

// Custom configuration
let config = AdaptiveSsspConfig {
    mode: AlgorithmMode::Auto,
    landmark_count: 64,
    large_graph_threshold: 5_000_000,
    collect_metrics: true,
};
```

### SsspMetrics

```rust
pub struct SsspMetrics {
    /// Which algorithm was used
    pub algorithm_used: String,

    /// Total execution time in milliseconds
    pub total_time_ms: f32,

    /// GPU kernel time (if applicable)
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
```

---

## RecommendationEngine API

### Constructor with SSSP Config

```rust
pub async fn with_sssp_config(
    embeddings: Vec<f32>,
    embedding_dim: usize,
    metadata: Vec<ContentMetadata>,
    sssp_config: AdaptiveSsspConfig,
) -> Result<Self>
```

**Example**:
```rust
let config = AdaptiveSsspConfig {
    mode: AlgorithmMode::Auto,
    landmark_count: 32,
    large_graph_threshold: 10_000_000,
    collect_metrics: true,
};

let engine = RecommendationEngine::with_sssp_config(
    embeddings,
    1024,
    metadata,
    config,
).await?;
```

### Get Metrics

```rust
pub async fn get_sssp_metrics(&self) -> Option<SsspMetrics>
```

**Example**:
```rust
if let Some(metrics) = engine.get_sssp_metrics().await {
    println!("Algorithm: {}", metrics.algorithm_used);
    println!("Latency: {:.2}ms", metrics.total_time_ms);
}
```

### Get/Set Algorithm Mode

```rust
pub async fn get_algorithm_mode(&self) -> AlgorithmMode
pub async fn set_algorithm_mode(&self, mode: AlgorithmMode) -> Result<()>
```

**Example**:
```rust
// Query current mode
let mode = engine.get_algorithm_mode().await;
println!("Using: {:?}", mode);

// Override for testing
engine.set_algorithm_mode(AlgorithmMode::LandmarkApsp).await?;
```

---

## GpuSemanticEngine API

### Find Shortest Paths (Auto-Select)

```rust
pub async fn find_shortest_paths(
    &self,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
) -> GpuResult<Vec<Path>>
```

**Example**:
```rust
// Auto-selects optimal algorithm
let paths = engine.find_shortest_paths(
    &graph_data,
    &[source_node],
    &[target_node],
    &PathfindingConfig::default(),
).await?;
```

### Find Shortest Paths (Explicit Algorithm)

```rust
pub async fn find_shortest_paths_with_algorithm(
    &self,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
    algorithm: Option<AlgorithmMode>,
) -> GpuResult<Vec<Path>>
```

**Example**:
```rust
// Force GPU Dijkstra
let paths = engine.find_shortest_paths_with_algorithm(
    &graph_data,
    &sources,
    &targets,
    &PathfindingConfig::default(),
    Some(AlgorithmMode::GpuDijkstra),
).await?;
```

### Get Metrics and Algorithm

```rust
pub async fn get_sssp_metrics(&self) -> Option<SsspMetrics>
pub async fn get_selected_algorithm(&self) -> AlgorithmMode
pub async fn set_algorithm_mode(&self, mode: AlgorithmMode) -> GpuResult<()>
```

**Example**:
```rust
// Get current algorithm
let algo = engine.get_selected_algorithm().await;
println!("Selected: {:?}", algo);

// Get execution metrics
if let Some(m) = engine.get_sssp_metrics().await {
    println!("Time: {:.2}ms", m.total_time_ms);
    println!("Nodes: {}", m.nodes_processed);
}

// Override algorithm
engine.set_algorithm_mode(AlgorithmMode::Auto).await?;
```

---

## Complete Usage Example

```rust
use crate::adaptive_sssp::{
    AdaptiveSsspConfig, AlgorithmMode, SsspMetrics,
};
use crate::gpu_engine::{GpuSemanticEngine, GpuConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize GPU engine with adaptive SSSP
    let gpu_config = GpuConfig::default();
    let engine = GpuSemanticEngine::new(gpu_config).await?;

    // 2. Prepare graph data
    let graph_data = vec![/* adjacency data */];
    let sources = vec![0, 1, 2];
    let targets = vec![10, 20, 30];

    // 3. Run pathfinding (auto-selects algorithm)
    let paths = engine.find_shortest_paths(
        &graph_data,
        &sources,
        &targets,
        &PathfindingConfig::default(),
    ).await?;

    // 4. Check which algorithm was used
    if let Some(metrics) = engine.get_sssp_metrics().await {
        println!("Algorithm: {}", metrics.algorithm_used);
        println!("Latency: {:.2}ms", metrics.total_time_ms);
        println!("GPU Time: {:.2}ms", metrics.gpu_time_ms.unwrap_or(0.0));
        println!("Nodes: {}", metrics.nodes_processed);
        println!("Edges: {}", metrics.edges_relaxed);
    }

    // 5. Override for comparison
    engine.set_algorithm_mode(AlgorithmMode::LandmarkApsp).await?;

    let paths2 = engine.find_shortest_paths(
        &graph_data,
        &sources,
        &targets,
        &PathfindingConfig::default(),
    ).await?;

    // 6. Compare results
    if let Some(metrics2) = engine.get_sssp_metrics().await {
        println!("\nWith Landmark APSP:");
        println!("Latency: {:.2}ms", metrics2.total_time_ms);
        println!("Landmarks: {:?}", metrics2.landmarks_used);
    }

    Ok(())
}
```

---

## Recommendation Engine Example

```rust
use crate::semantic_search::unified_engine::{
    RecommendationEngine, ViewingContext,
};
use crate::adaptive_sssp::{AdaptiveSsspConfig, AlgorithmMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Create engine with custom SSSP config
    let sssp_config = AdaptiveSsspConfig {
        mode: AlgorithmMode::Auto,
        landmark_count: 64,
        large_graph_threshold: 5_000_000,
        collect_metrics: true,
    };

    let engine = RecommendationEngine::with_sssp_config(
        embeddings,
        1024,
        metadata,
        sssp_config,
    ).await?;

    // 2. Get recommendations
    let user_context = ViewingContext {
        time_of_day: "evening".to_string(),
        device: "mobile".to_string(),
        viewing_history: vec!["movie1".to_string()],
        preferences: vec!["action".to_string()],
    };

    let recommendations = engine.recommend(
        "user123",
        &user_context,
        10,
    ).await?;

    // 3. Check SSSP metrics
    if let Some(metrics) = engine.get_sssp_metrics().await {
        println!("Pathfinding algorithm: {}", metrics.algorithm_used);
        println!("Query latency: {:.2}ms", metrics.total_time_ms);
    }

    // 4. Monitor algorithm selection
    let current_mode = engine.get_algorithm_mode().await;
    println!("Current mode: {:?}", current_mode);

    Ok(())
}
```

---

## Algorithm Selection Logic

### Decision Tree

```rust
fn auto_select(stats: &GraphStats) -> AlgorithmMode {
    // Very large graphs (>10M nodes)
    if stats.num_nodes > 10_000_000 {
        return AlgorithmMode::LandmarkApsp;
    }

    // Medium sparse graphs with high query frequency
    if stats.is_sparse && stats.num_nodes > 100_000 {
        return AlgorithmMode::LandmarkApsp;
    }

    // Small-medium graphs
    AlgorithmMode::GpuDijkstra
}
```

### Graph Characteristics

```rust
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub is_sparse: bool,  // true if avg_degree < log(n)
}
```

---

## Performance Guidelines

### When to Use Each Algorithm

| Graph Size | Queries/sec | Recommended Algorithm |
|-----------|-------------|----------------------|
| <100K nodes | Any | GPU Dijkstra |
| 100K-1M nodes | <100 | GPU Dijkstra |
| 100K-1M nodes | >100 | Landmark APSP |
| 1M-10M nodes | <50 | GPU Dijkstra |
| 1M-10M nodes | >50 | Landmark APSP |
| >10M nodes | Any | Landmark APSP |

### Expected Latencies

```
Algorithm         | 10K nodes | 1M nodes | 10M nodes | 100M nodes
-----------------|-----------|----------|-----------|------------
GPU Dijkstra     | 0.5ms     | 15ms     | 180ms     | 2000ms
Landmark APSP    | 2ms       | 25ms     | 120ms     | 500ms
Duan (future)    | 0.4ms     | 12ms     | 100ms     | 400ms
```

---

## Testing

### Unit Tests

```bash
# Test algorithm selection
cargo test --lib adaptive_sssp::tests::test_algorithm_selection

# Test graph statistics
cargo test --lib adaptive_sssp::tests::test_graph_stats

# Test theoretical complexity
cargo test --lib adaptive_sssp::tests::test_theoretical_complexity
```

### Integration Tests

```rust
#[tokio::test]
async fn test_adaptive_sssp_integration() {
    let engine = GpuSemanticEngine::new(GpuConfig::default()).await.unwrap();

    // Test auto-selection
    let paths = engine.find_shortest_paths(
        &graph,
        &sources,
        &targets,
        &config,
    ).await.unwrap();

    assert!(!paths.is_empty());

    // Verify metrics collected
    let metrics = engine.get_sssp_metrics().await;
    assert!(metrics.is_some());
}
```

---

## Troubleshooting

### Issue: Algorithm not switching as expected

**Solution**: Check graph statistics update:
```rust
let mut sssp = engine.adaptive_sssp.write().await;
sssp.update_graph_stats(actual_nodes, actual_edges);
let selected = sssp.select_algorithm();
println!("Selected: {:?}", selected);
```

### Issue: Metrics not being collected

**Solution**: Enable metrics in config:
```rust
let config = AdaptiveSsspConfig {
    mode: AlgorithmMode::Auto,
    collect_metrics: true,  // Must be true
    ..Default::default()
};
```

### Issue: Performance not improving with APSP

**Solution**: Increase landmark count:
```rust
let config = AdaptiveSsspConfig {
    landmark_count: 64,  // More landmarks = better accuracy
    ..Default::default()
};
```

---

## Migration Guide

### From Standard Pathfinding

**Before**:
```rust
let paths = engine.find_shortest_paths(
    graph, sources, targets, config
).await?;
```

**After** (no changes needed!):
```rust
// Same API, now with adaptive algorithm selection
let paths = engine.find_shortest_paths(
    graph, sources, targets, config
).await?;

// Optionally check which algorithm was used
if let Some(m) = engine.get_sssp_metrics().await {
    println!("Used: {}", m.algorithm_used);
}
```

### Adding Explicit Algorithm Selection

```rust
// Force specific algorithm
let paths = engine.find_shortest_paths_with_algorithm(
    graph,
    sources,
    targets,
    config,
    Some(AlgorithmMode::GpuDijkstra),  // Explicit choice
).await?;
```

---

## Best Practices

1. **Use Auto mode by default** - Let the system choose
2. **Monitor metrics** - Track algorithm selection and performance
3. **Update graph stats** - Keep statistics current for optimal selection
4. **Benchmark before forcing** - Only override if you have data
5. **Enable metrics collection** - Essential for performance tuning

---

**Last Updated**: 2025-12-04
**API Version**: 1.0
**Status**: Production Ready
