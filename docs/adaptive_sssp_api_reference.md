# Adaptive SSSP API Reference

## Module: `gpu_engine::adaptive_sssp`

Complete API reference for the adaptive Single-Source Shortest Path implementation.

---

## Enums

### `SSPAlgorithm`

Algorithm selection strategy.

```rust
pub enum SSPAlgorithm {
    GPUDijkstra,
    HybridDuan,
    Auto,
}
```

**Variants**:

- **`GPUDijkstra`**: Pure GPU Dijkstra algorithm
  - Best for small to medium graphs (< 100K nodes)
  - Low latency, high throughput
  - Requires sufficient GPU memory

- **`HybridDuan`**: Hybrid CPU-GPU Duan algorithm
  - Best for large graphs (>= 100K nodes)
  - Scales to millions of nodes
  - Adaptive workload distribution

- **`Auto`**: Automatic selection based on graph size
  - Chooses algorithm using `crossover_threshold`
  - Recommended for most use cases
  - Adapts to graph characteristics

**Default**: `Auto`

**Example**:
```rust
let algorithm = SSPAlgorithm::Auto;
```

---

## Structs

### `AdaptiveSSPConfig`

Configuration for adaptive SSSP execution.

```rust
pub struct AdaptiveSSPConfig {
    pub algorithm: SSPAlgorithm,
    pub crossover_threshold: usize,
    pub enable_profiling: bool,
    pub max_depth: usize,
    pub max_paths: usize,
    pub weighted: bool,
    pub hybrid_cpu_threads: usize,
    pub hybrid_batch_size: usize,
}
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `algorithm` | `SSPAlgorithm` | `Auto` | Algorithm selection strategy |
| `crossover_threshold` | `usize` | `100_000` | Node count threshold for Auto mode |
| `enable_profiling` | `bool` | `true` | Enable performance metrics collection |
| `max_depth` | `usize` | `10` | Maximum path length to search |
| `max_paths` | `usize` | `100` | Maximum number of paths to return |
| `weighted` | `bool` | `true` | Whether to use edge weights |
| `hybrid_cpu_threads` | `usize` | `4` | CPU threads for Hybrid Duan frontier |
| `hybrid_batch_size` | `usize` | `10_000` | Batch size for GPU processing |

**Methods**:

#### `default() -> Self`

Create configuration with default values.

```rust
let config = AdaptiveSSPConfig::default();
```

#### `to_pathfinding_config(&self) -> PathfindingConfig`

Convert to PathfindingConfig for compatibility.

```rust
let pf_config = config.to_pathfinding_config();
```

**Example**:
```rust
let config = AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::Auto,
    crossover_threshold: 50_000,
    enable_profiling: true,
    max_depth: 20,
    max_paths: 1000,
    weighted: true,
    hybrid_cpu_threads: 8,
    hybrid_batch_size: 20_000,
};
```

---

### `SSPMetrics`

Performance metrics for SSSP execution.

```rust
pub struct SSPMetrics {
    pub algorithm_used: String,
    pub total_time_ms: f64,
    pub gpu_compute_ms: f64,
    pub cpu_compute_ms: f64,
    pub transfer_time_ms: f64,
    pub nodes_processed: usize,
    pub edges_traversed: usize,
    pub paths_found: usize,
    pub peak_gpu_memory: usize,
    pub avg_path_length: f32,
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `algorithm_used` | `String` | Name of algorithm that was executed |
| `total_time_ms` | `f64` | Total execution time in milliseconds |
| `gpu_compute_ms` | `f64` | GPU compute time in milliseconds |
| `cpu_compute_ms` | `f64` | CPU compute time in milliseconds |
| `transfer_time_ms` | `f64` | Memory transfer time in milliseconds |
| `nodes_processed` | `usize` | Number of nodes processed |
| `edges_traversed` | `usize` | Number of edges traversed |
| `paths_found` | `usize` | Number of paths found |
| `peak_gpu_memory` | `usize` | Peak GPU memory usage in bytes |
| `avg_path_length` | `f32` | Average path length |

**Methods**:

#### `throughput_nodes_per_sec(&self) -> f64`

Calculate throughput in nodes per second.

```rust
let throughput = metrics.throughput_nodes_per_sec();
println!("Throughput: {:.0} nodes/sec", throughput);
```

**Returns**: Nodes processed per second, or 0.0 if total_time_ms is 0.

#### `gpu_utilization(&self) -> f64`

Calculate GPU utilization percentage.

```rust
let utilization = metrics.gpu_utilization();
println!("GPU utilization: {:.1}%", utilization);
```

**Returns**: GPU compute time as percentage of total time (0-100).

**Example**:
```rust
println!("Algorithm: {}", metrics.algorithm_used);
println!("Total time: {:.2}ms", metrics.total_time_ms);
println!("GPU compute: {:.2}ms ({:.1}%)",
    metrics.gpu_compute_ms, metrics.gpu_utilization());
println!("Throughput: {:.0} nodes/sec", metrics.throughput_nodes_per_sec());
println!("Memory: {:.2}MB", metrics.peak_gpu_memory as f64 / 1_048_576.0);
```

---

### `SSPResult`

Result of adaptive SSSP execution.

```rust
pub struct SSPResult {
    pub paths: Vec<Path>,
    pub metrics: SSPMetrics,
}
```

**Fields**:

- **`paths`**: `Vec<Path>` - Shortest paths found
- **`metrics`**: `SSPMetrics` - Performance metrics

**Example**:
```rust
let result = find_adaptive_shortest_paths(...).await?;

println!("Found {} paths", result.paths.len());
println!("Algorithm: {}", result.metrics.algorithm_used);
println!("Time: {:.2}ms", result.metrics.total_time_ms);

for path in &result.paths {
    println!("Path: {:?}", path.nodes);
}
```

---

## Functions

### `find_adaptive_shortest_paths`

Main entry point for adaptive SSSP execution.

```rust
pub async fn find_adaptive_shortest_paths(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &AdaptiveSSPConfig,
) -> GpuResult<SSPResult>
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `&Arc<CudaDevice>` | CUDA device handle |
| `modules` | `&Arc<KernelModules>` | Compiled CUDA kernels |
| `memory_pool` | `&Arc<RwLock<MemoryPool>>` | GPU memory pool |
| `streams` | `&Arc<StreamManager>` | CUDA stream manager |
| `graph` | `&[u32]` | Edge list [from1, to1, from2, to2, ...] |
| `sources` | `&[u32]` | Source node IDs |
| `targets` | `&[u32]` | Target node IDs |
| `config` | `&AdaptiveSSPConfig` | Algorithm configuration |

**Returns**: `GpuResult<SSPResult>` - Result with paths and metrics

**Errors**:
- `GpuError::Cuda`: CUDA operation failed
- `GpuError::Memory`: Memory allocation failed
- `GpuError::Stream`: Stream synchronization failed

**Example**:
```rust
let device = Arc::new(CudaDevice::new(0)?);
let modules = Arc::new(KernelModules::load(&device)?);
let memory_pool = Arc::new(RwLock::new(MemoryPool::new(device.clone())));
let streams = Arc::new(StreamManager::new(device.clone(), 4).await?);

let graph = vec![0, 1, 1, 2, 2, 3];
let sources = vec![0];
let targets = vec![3];
let config = AdaptiveSSPConfig::default();

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

println!("Found {} paths in {:.2}ms",
    result.paths.len(),
    result.metrics.total_time_ms);
```

---

### `detect_crossover_threshold`

Automatically detect optimal crossover threshold.

```rust
pub async fn detect_crossover_threshold(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
) -> GpuResult<usize>
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `&Arc<CudaDevice>` | CUDA device handle |
| `modules` | `&Arc<KernelModules>` | Compiled CUDA kernels |
| `memory_pool` | `&Arc<RwLock<MemoryPool>>` | GPU memory pool |
| `streams` | `&Arc<StreamManager>` | CUDA stream manager |

**Returns**: `GpuResult<usize>` - Optimal threshold in number of nodes

**Description**:

Runs benchmarks on synthetic graphs of varying sizes (1K to 200K nodes) to find the point where Hybrid Duan becomes faster than GPU Dijkstra.

**Warning**: This is an expensive operation that may take several minutes.

**Example**:
```rust
let threshold = detect_crossover_threshold(
    &device,
    &modules,
    &memory_pool,
    &streams,
).await?;

println!("Optimal threshold: {} nodes", threshold);

// Use detected threshold
let config = AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::Auto,
    crossover_threshold: threshold,
    ..Default::default()
};
```

---

## Internal Functions

These functions are not part of the public API but are documented for completeness.

### `select_algorithm`

```rust
fn select_algorithm(
    algorithm: SSPAlgorithm,
    num_nodes: usize,
    crossover_threshold: usize,
) -> SSPAlgorithm
```

Select algorithm based on configuration and graph size.

### `estimate_node_count`

```rust
fn estimate_node_count(
    graph: &[u32],
    sources: &[u32],
    targets: &[u32]
) -> usize
```

Estimate number of nodes from graph data.

### `build_adjacency_list`

```rust
fn build_adjacency_list(
    graph: &[u32],
    num_nodes: usize
) -> Vec<Vec<(u32, usize)>>
```

Build adjacency list from edge list.

### `expand_frontier_cpu`

```rust
fn expand_frontier_cpu(
    adj_list: &[Vec<(u32, usize)>],
    weights: &[f32],
    frontier: &[u32],
    distances: &mut [f32],
    predecessors: &mut [u32],
    visited: &[bool],
) -> Vec<u32>
```

Expand frontier on CPU for Hybrid Duan.

---

## Type Aliases

### `GpuResult<T>`

```rust
pub type GpuResult<T> = Result<T, GpuError>;
```

Result type for GPU operations.

---

## Error Types

### `GpuError`

```rust
pub enum GpuError {
    Cuda(cudarc::driver::DriverError),
    Kernel(KernelError),
    Memory(String),
    Stream(String),
    Config(String),
    Io(std::io::Error),
}
```

Unified error type for GPU operations.

**Variants**:

- **`Cuda`**: CUDA driver error
- **`Kernel`**: Kernel execution error
- **`Memory`**: Memory operation error
- **`Stream`**: Stream operation error
- **`Config`**: Configuration error
- **`Io`**: I/O error

---

## Related Types

### `Path` (from pathfinding module)

```rust
pub struct Path {
    pub nodes: Vec<u32>,
    pub cost: f32,
    pub length: usize,
}
```

**Methods**:
- `contains(&self, node: u32) -> bool`: Check if path contains node
- `edges(&self) -> Vec<(u32, u32)>`: Get path as edge list

---

## Constants

### Default Values

```rust
const DEFAULT_CROSSOVER_THRESHOLD: usize = 100_000;
const DEFAULT_MAX_DEPTH: usize = 10;
const DEFAULT_MAX_PATHS: usize = 100;
const DEFAULT_HYBRID_CPU_THREADS: usize = 4;
const DEFAULT_HYBRID_BATCH_SIZE: usize = 10_000;
```

---

## Usage Patterns

### Pattern 1: Simple Automatic Selection

```rust
let config = AdaptiveSSPConfig::default();

let result = find_adaptive_shortest_paths(
    &device, &modules, &memory_pool, &streams,
    &graph, &sources, &targets, &config,
).await?;
```

### Pattern 2: Custom Threshold

```rust
let config = AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::Auto,
    crossover_threshold: 50_000,
    ..Default::default()
};
```

### Pattern 3: Force Algorithm

```rust
let config = AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::HybridDuan,
    hybrid_cpu_threads: 16,
    hybrid_batch_size: 50_000,
    ..Default::default()
};
```

### Pattern 4: Disable Profiling

```rust
let config = AdaptiveSSPConfig {
    enable_profiling: false,
    ..Default::default()
};
```

### Pattern 5: Path-Limited Search

```rust
let config = AdaptiveSSPConfig {
    max_depth: 5,
    max_paths: 10,
    ..Default::default()
};
```

---

## Performance Considerations

### Memory Usage

- **GPU Dijkstra**: `O(V + E)` on GPU
- **Hybrid Duan**: `O(V)` on GPU, `O(V + E)` on CPU

### Time Complexity

- **GPU Dijkstra**: `O((V + E) log V)` parallelized
- **Hybrid Duan**: `O(iterations × (frontier_cpu + frontier_gpu))`

### Recommendations

| Graph Size | Algorithm | Rationale |
|-----------|-----------|-----------|
| < 10K | GPUDijkstra | Low latency, minimal overhead |
| 10K-100K | Auto | Adaptive selection optimal |
| > 100K | HybridDuan | Scales beyond GPU memory |

---

## Thread Safety

- All functions are `async` and thread-safe
- `MemoryPool` uses `RwLock` for concurrent access
- `StreamManager` handles stream synchronization
- No global state or shared mutable data

---

## Feature Flags

None currently. All features are always enabled.

---

## Version History

- **v1.0.0** (2025-12-04): Initial implementation
  - Automatic algorithm selection
  - GPU Dijkstra and Hybrid Duan
  - Performance profiling
  - Crossover detection

---

## See Also

- `gpu_engine::pathfinding`: Base pathfinding module
- `gpu_engine::kernels`: CUDA kernel interface
- `gpu_engine::memory`: GPU memory management
- `gpu_engine::streaming`: CUDA stream management

---

## License

MIT License - See project root for details.
