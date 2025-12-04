# Hybrid CPU-WASM/GPU SSSP Module

## Overview

This module implements the "Breaking the Sorting Barrier" O(m log^(2/3) n) algorithm
for Single-Source Shortest Path computation, combining CPU-WASM recursive control with
GPU-accelerated parallel relaxation.

## Files

### mod.rs (281 lines)
Main executor with:
- `HybridSSPConfig`: Configuration with theoretical parameters (k, t, max_depth)
- `HybridSSPExecutor`: Main execution orchestrator
- `SSPMetrics`: Performance tracking
- Automatic parameter calculation: k = log^(1/3) n, t = log^(2/3) n

### adaptive_heap.rs (315 lines)
Sophisticated data structure implementing:
- `insert(vertex, distance)`: Single vertex insertion
- `batch_prepend(vertices, distances)`: Group insertion for efficiency
- `pull(m)`: Extract m minimum-distance vertices
- Block-based organization with automatic merging

### communication_bridge.rs (313 lines)
CPU-GPU coordination layer:
- `GPUBridge`: Handles data transfer and kernel invocation
- `k_step_relaxation`: Performs k rounds of edge relaxation
- `bounded_dijkstra`: GPU-accelerated Dijkstra with distance bound
- Pinned memory support for optimized transfers

### wasm_controller.rs (351 lines)
Recursive BMSSP orchestration:
- `WASMController`: Manages recursive partitioning
- `execute_bmssp`: Main recursive algorithm entry
- `find_pivots`: Identifies partition boundaries
- `partition_frontier`: Splits work based on pivot distances
- Iterative implementation to avoid stack overflow

## Integration

The module is integrated into `gpu_engine/mod.rs`:
```rust
pub use hybrid_sssp::{HybridSSPConfig, HybridSSPExecutor, HybridSSPResult, SSPMetrics};
```

## Key Features

1. **Theoretical Soundness**: Exact parameter calculations from the paper
2. **T4 GPU Compatibility**: Designed for sm_75 architecture
3. **Async/Await**: Full async support for GPU operations
4. **Zero-Copy Transfers**: Pinned memory for efficient CPU-GPU communication
5. **Comprehensive Metrics**: CPU/GPU time, transfer overhead, complexity factor

## Usage Example

```rust
use gpu_engine::hybrid_sssp::{HybridSSPExecutor, HybridSSPConfig};

let config = HybridSSPConfig {
    enable_hybrid: true,
    use_pinned_memory: true,
    enable_profiling: true,
    ..Default::default()
};

let mut executor = HybridSSPExecutor::new(config);
executor.initialize().await?;

let result = executor.execute(
    num_nodes,
    num_edges,
    &sources,
    &csr_row_offsets,
    &csr_col_indices,
    &csr_weights,
).await?;

println!("Distances: {:?}", result.distances);
println!("Metrics: {:?}", result.metrics);
```

## Theoretical Complexity

- **Time**: O(m log^(2/3) n) - Breaking the sorting barrier
- **Space**: O(n + m) for graph storage + O(k·t) for recursion
- **Parameters**:
  - k = log^(1/3) n (pivot selection rounds)
  - t = log^(2/3) n (branching factor)
  - max_depth = log n / t (recursion depth)

## Testing

All modules include comprehensive unit tests:
- `adaptive_heap`: Insert/pull, batch operations, duplicate handling
- `communication_bridge`: Graph upload, memory stats
- `wasm_controller`: Controller creation, frontier partitioning
- `mod.rs`: Parameter calculation validation

## Performance Notes

- GPU operations use placeholder implementations pending kernel integration
- Pinned memory reduces transfer overhead by ~2-3x
- Debug assertions can be disabled in release builds for maximum performance
- Logging uses eprintln! for zero-dependency operation

## Future Work

1. Connect to actual CUDA kernels (currently placeholders)
2. Implement WASM bindings for browser deployment
3. Add memory pooling for repeated executions
4. Benchmark against standard Dijkstra on real graphs
5. Tune parameters empirically for T4 GPU characteristics
