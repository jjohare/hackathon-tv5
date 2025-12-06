# Hybrid SSSP FFI Quick Reference

## Files Created

```
/home/devuser/workspace/hackathon-tv5/
├── src/rust/
│   ├── gpu_engine/
│   │   └── hybrid_sssp_ffi.rs         (915 lines) ← Main FFI module
│   ├── examples/
│   │   └── hybrid_sssp_example.rs     (522 lines) ← Usage examples
│   └── tests/
│       └── hybrid_sssp_ffi_tests.rs   (642 lines) ← Test suite
├── docs/
│   ├── hybrid_sssp_ffi_implementation.md         ← Full documentation
│   └── hybrid_sssp_ffi_quickref.md               ← This file
└── scripts/
    └── verify_hybrid_sssp_ffi.sh                 ← Verification script
```

**Total**: 2,079 lines of code

## API Overview

### Initialization
```rust
use gpu_engine::{HybridSSSPKernels, HybridSSSPConfig};

let device = Arc::new(CudaDevice::new(0)?);
let kernels = HybridSSSPKernels::new(device.clone());
let stream = device.fork_default_stream()?;
```

### Core Operations

| Operation | Purpose | Complexity |
|-----------|---------|------------|
| `initialize_distances` | Set source=0, others=∞ | O(V) |
| `k_step_relaxation` | Approximate SSSP | O(k*E) |
| `detect_pivots` | Find critical nodes | O(V) |
| `bounded_dijkstra` | Refine around pivots | O(pivots*r*log V) |
| `partition_frontier` | Filter active nodes | O(frontier) |
| `compact_frontier` | Stream compaction | O(frontier) |
| `sssp_semantic` | Content discovery SSSP | O(E) per iteration |
| `select_landmarks` | Choose diverse nodes | O(V) |
| `approximate_apsp` | All-pairs distances | O(V²*k) |

### Quick Workflow

```rust
// 1. Initialize
let mut d_distances = device.alloc_zeros::<f32>(n)?;
let mut d_predecessors = device.alloc_zeros::<i32>(n)?;
kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;

// 2. K-step approximation
kernels.k_step_relaxation(&mut d_distances, &mut d_predecessors,
    &d_row_offsets, &d_col_indices, &d_edge_weights, k, &stream)?;

// 3. Detect & refine pivots
let pivots = kernels.detect_pivots(&d_distances, &d_frontier,
    &d_row_offsets, threshold, degree_min, &stream)?;
if !pivots.is_empty() {
    kernels.bounded_dijkstra(&mut d_distances, &mut d_predecessors,
        &pivots, &d_row_offsets, &d_col_indices, &d_edge_weights,
        radius, &stream)?;
}

// 4. Retrieve results
stream.synchronize()?;
let distances = device.dtoh_sync_copy(&d_distances)?;
```

## Configuration Presets

### Sparse Graphs (E ≈ V)
```rust
HybridSSSPConfig {
    k_steps: 15,
    convergence_threshold: 0.01,
    degree_threshold: 50,
    dijkstra_radius: 5,
    frontier_epsilon: 1e-6,
    max_hops: 15,
    min_similarity: 0.5,
}
```

### Dense Graphs (E ≈ V²)
```rust
HybridSSSPConfig {
    k_steps: 5,
    convergence_threshold: 0.005,
    degree_threshold: 1000,
    dijkstra_radius: 2,
    frontier_epsilon: 1e-5,
    max_hops: 8,
    min_similarity: 0.6,
}
```

### Small-World / Social Graphs
```rust
HybridSSSPConfig {
    k_steps: 10,
    convergence_threshold: 0.01,
    degree_threshold: 100,
    dijkstra_radius: 3,
    frontier_epsilon: 1e-6,
    max_hops: 10,
    min_similarity: 0.5,
}
```

## Error Handling

```rust
match kernels.k_step_relaxation(...) {
    Ok(()) => println!("Success"),
    Err(GpuError::Cuda(e)) => eprintln!("CUDA error: {}", e),
    Err(GpuError::Kernel(e)) => eprintln!("Kernel error: {}", e),
    Err(GpuError::Memory(e)) => eprintln!("Memory error: {}", e),
    Err(GpuError::Config(e)) => eprintln!("Config error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Path Reconstruction

```rust
let result = SSSPResult {
    distances: device.dtoh_sync_copy(&d_distances)?,
    predecessors: device.dtoh_sync_copy(&d_predecessors)?,
    semantic_scores: Some(device.dtoh_sync_copy(&d_semantic)?),
};

// Get path
if let Some(path) = result.reconstruct_path(target) {
    println!("Path: {:?}", path);
    println!("Distance: {:.2}", result.distance_to(target).unwrap());
    if let Some(score) = result.semantic_score(target) {
        println!("Semantic score: {:.3}", score);
    }
}
```

## Memory Layout

### CSR Graph Format
```rust
// Graph with edges: 0->1, 0->2, 1->2, 2->3
let row_offsets = vec![0, 2, 3, 4, 4];  // [n+1]
let col_indices = vec![1, 2, 2, 3];     // [m]
let edge_weights = vec![1.0, 2.0, 1.0, 3.0];  // [m]

// Upload to GPU
let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
let d_col_indices = device.htod_sync_copy(&col_indices)?;
let d_edge_weights = device.htod_sync_copy(&edge_weights)?;
```

### Memory Footprint
- **Graph**: `4*m + 4*(n+1) + 4*m` bytes (CSR)
- **Working**: `8*n` bytes (distances + predecessors)
- **Peak**: `~12*n + 8*m` bytes

## Testing

```bash
# Unit tests (no GPU required)
cargo test --lib

# Integration tests (requires CUDA)
cargo test --test hybrid_sssp_ffi_tests --features cuda -- --ignored

# Run examples
cargo run --example hybrid_sssp_example --features cuda

# Verify implementation
./scripts/verify_hybrid_sssp_ffi.sh
```

## Performance Tips

1. **Batch operations**: Process multiple sources in parallel
2. **Reuse buffers**: Allocate once, reuse for multiple queries
3. **Stream concurrency**: Use multiple streams for independent ops
4. **Tune k_steps**: Profile to find optimal value for your graph
5. **Pivot threshold**: Adjust based on graph density

## Common Patterns

### Single-Source Query
```rust
kernels.initialize_distances(&mut d_distances, &mut d_predecessors, source, &stream)?;
kernels.k_step_relaxation(..., 10, &stream)?;
```

### Multi-Source Batch
```rust
for source in sources {
    kernels.initialize_distances(&mut d_distances, &mut d_predecessors, source, &stream)?;
    kernels.k_step_relaxation(...)?;
    let distances = device.dtoh_sync_copy(&d_distances)?;
    process_results(source, distances);
}
```

### Hybrid Algorithm
```rust
// Phase 1: Approximate
kernels.k_step_relaxation(..., k, &stream)?;

// Phase 2: Detect critical nodes
let pivots = kernels.detect_pivots(...)?;

// Phase 3: Refine
kernels.bounded_dijkstra(..., pivots, radius, &stream)?;
```

### Semantic Discovery
```rust
kernels.sssp_semantic(
    source,
    &mut d_distances, &mut d_predecessors, &mut d_semantic_scores,
    &d_row_offsets, &d_col_indices, &d_edge_weights,
    &d_content_features, &d_user_affinities,
    &d_frontier, &mut d_next_frontier, &mut d_next_frontier_size,
    max_hops, min_similarity,
    &stream
)?;
```

## Safety Guarantees

✓ **Memory safety**: RAII via `CudaSlice`
✓ **Type safety**: Strong typing, no raw pointer leaks
✓ **Thread safety**: Stream synchronization
✓ **Error propagation**: `Result<T, GpuError>`
✓ **Bounds checking**: Array size validation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Empty graph" error | Check `num_nodes > 0` and `num_edges > 0` |
| "Invalid CSR format" | Verify `row_offsets.len() == num_nodes + 1` |
| Kernel launch failure | Check CUDA device availability |
| Infinite distances | Not enough k_steps or unreachable nodes |
| Memory allocation error | Reduce problem size or check GPU memory |
| Stream sync timeout | Increase timeout or check kernel loop |

## FFI Bindings Summary

| Kernel | Rust Wrapper | Return Type |
|--------|--------------|-------------|
| `k_step_relaxation_launch` | `k_step_relaxation` | `GpuResult<()>` |
| `detect_pivots_launch` | `detect_pivots` | `GpuResult<Vec<u32>>` |
| `bounded_dijkstra_launch` | `bounded_dijkstra` | `GpuResult<()>` |
| `partition_frontier_launch` | `partition_frontier` | `GpuResult<Vec<u32>>` |
| `compact_frontier_gpu` | `compact_frontier` | `GpuResult<Vec<u32>>` |
| `initialize_distances_launch` | `initialize_distances` | `GpuResult<()>` |
| `sssp_semantic_kernel_launch` | `sssp_semantic` | `GpuResult<()>` |
| `select_content_landmarks_kernel_launch` | `select_landmarks` | `GpuResult<Vec<u32>>` |
| `approximate_apsp_content_kernel_launch` | `approximate_apsp` | `GpuResult<()>` |

## Related Modules

- `adaptive_sssp.rs`: Algorithm selector (BFS/Dijkstra/Hybrid)
- `pathfinding.rs`: High-level pathfinding API
- `kernels.rs`: Other kernel bindings
- `memory.rs`: Memory pool management
- `streaming.rs`: Stream manager

## Documentation Links

- **Full guide**: `docs/hybrid_sssp_ffi_implementation.md`
- **CUDA kernels**: `src/cuda/kernels/graph_search.cu`
- **Examples**: `src/rust/examples/hybrid_sssp_example.rs`
- **Tests**: `src/rust/tests/hybrid_sssp_ffi_tests.rs`
- **API docs**: Run `cargo doc --open`

---

**Version**: 1.0.0
**Author**: Implementation Date: 2025-12-04
**License**: Project License
