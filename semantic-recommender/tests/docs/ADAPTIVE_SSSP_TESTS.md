# Adaptive SSSP Integration Tests

## Overview

Comprehensive test suite for Adaptive Single-Source Shortest Path (SSSP) algorithm selection system. Tests algorithm selection, correctness, performance, and graceful degradation.

## Test Structure

### File: `tests/adaptive_sssp_tests.rs`

Total: **13 test cases** covering:
- Algorithm selection logic
- Correctness validation
- Performance benchmarks
- Edge case handling
- Fallback mechanisms

## Test Categories

### 1. Algorithm Selection Tests

#### `test_small_graph_selects_gpu_dijkstra`
- **Graph**: 1K nodes, sparse (Erdős-Rényi, p=0.01)
- **Expected**: GPU Dijkstra
- **Validates**: Small graph detection

#### `test_medium_graph_selects_gpu_dijkstra`
- **Graph**: 100K nodes, scale-free (avg degree 5)
- **Expected**: GPU Dijkstra
- **Validates**: Medium graph detection

#### `test_large_graph_selects_hybrid_duan`
- **Graph**: 10M nodes, scale-free (avg degree 15)
- **Expected**: Hybrid Duan
- **Validates**: Large high-degree graph detection

### 2. Correctness Tests

#### `test_correctness_gpu_dijkstra`
- **Graph**: 10K node grid (100x100)
- **Validation**: Compare GPU Dijkstra vs CPU ground truth
- **Tolerance**: < 1e-5 error
- **Checks**: All distances match exactly

#### `test_correctness_hybrid_duan`
- **Graph**: 5K node scale-free (avg degree 8)
- **Validation**: Compare Hybrid Duan vs CPU ground truth
- **Tolerance**: < 1e-5 error
- **Checks**: Hub-based optimization correctness

#### `test_correctness_both_algorithms_match`
- **Graph**: 2K node Erdős-Rényi
- **Validation**: GPU Dijkstra results == Hybrid Duan results
- **Checks**: Algorithm equivalence

### 3. Performance Tests

#### `test_performance_comparison`
- **Graphs**: 1K, 5K, 10K nodes (scale-free)
- **Metrics**:
  - GPU Dijkstra time
  - Hybrid Duan time
  - Speedup factor
- **Runs**: 5 iterations per size (averaged)
- **Output**: Performance table with speedups

#### `test_crossover_point_detection`
- **Graphs**: 50K, 100K, 200K, 300K nodes
- **Goal**: Identify performance crossover point
- **Validates**: Threshold accuracy (1M nodes with high degree)

#### `test_memory_efficiency`
- **Graph**: 10K nodes, 8 avg degree
- **Checks**: Memory increase < 10MB
- **Validates**: Efficient memory usage

### 4. Adaptive Behavior Tests

#### `test_adaptive_switching`
- **Test Cases**:
  1. Small sparse (1K nodes, degree 5) → GPU Dijkstra
  2. Medium (50K nodes, degree 8) → GPU Dijkstra
  3. Large dense (1.5M nodes, degree 15) → Hybrid Duan
  4. Large sparse (2M nodes, degree 3) → GPU Dijkstra
- **Validates**: Correct algorithm selection for diverse inputs

#### `test_graceful_fallback`
- **Scenario**: Force Hybrid Duan selection but simulate failure
- **Expected**: Automatic fallback to GPU Dijkstra
- **Validates**: Fault tolerance

### 5. Edge Case Tests

#### `test_edge_cases`
- **Cases**:
  1. Single node graph
  2. Disconnected components
  3. Self-loops
  4. Zero-weight edges
  5. Unreachable nodes
- **Validates**: Robust handling of corner cases

### 6. Benchmark Suite

#### `benchmark_adaptive_sssp`
- **Configurations**:
  - Small Random: 1K nodes, Erdős-Rényi
  - Medium Grid: 10K nodes, 100x100 grid
  - Large Scale-Free: 50K nodes, preferential attachment
- **Metrics**:
  - Total execution time
  - Selected algorithm
  - Reachable node count
- **Output**: Full benchmark report

## Graph Generators

### `generate_erdos_renyi_graph(n, p, seed)`
Random graph with edge probability `p`.
- **Use**: General sparse graphs
- **Properties**: Uniform degree distribution

### `generate_scale_free_graph(n, avg_degree, seed)`
Preferential attachment (Barabási-Albert) model.
- **Use**: Real-world network simulation (social, web)
- **Properties**: Power-law degree distribution, hub nodes

### `generate_grid_graph(width, height)`
2D lattice graph.
- **Use**: Spatial networks, path planning
- **Properties**: Regular structure, bounded degree

## Implementation Details

### Algorithm Selection Logic

```rust
fn select_algorithm(graph: &Graph) -> SSSPAlgorithm {
    let avg_degree = edges.len() / nodes;

    if nodes >= 1_000_000 && avg_degree > 10.0 {
        SSSPAlgorithm::HybridDuan  // Large dense
    } else {
        SSSPAlgorithm::GPUDijkstra  // Small/medium or sparse
    }
}
```

**Thresholds**:
- Small: < 100K nodes
- Large: > 1M nodes
- High degree: > 10 avg degree

### GPU Dijkstra (Mock)

Current implementation uses CPU Dijkstra with timing simulation:
- Base computation: CPU priority queue
- GPU overhead: 50μs simulated transfer time
- **Production**: Replace with CUDA kernel FFI

### Hybrid Duan (Mock)

Three-phase approach:
1. **Hub Detection**: Identify nodes with degree > 2×avg
2. **GPU Phase**: Process hubs in parallel
3. **CPU Phase**: Sequential processing of remaining nodes
- Overhead: 100μs + O(n/1000)
- **Production**: Implement actual hybrid GPU/CPU execution

## Running Tests

### Run All Tests
```bash
cd /home/devuser/workspace/hackathon-tv5
cargo test --test adaptive_sssp_tests
```

### Run Specific Test
```bash
cargo test --test adaptive_sssp_tests test_small_graph_selects_gpu_dijkstra
```

### Run with Output
```bash
cargo test --test adaptive_sssp_tests -- --nocapture
```

### Run Benchmarks Only
```bash
cargo test --test adaptive_sssp_tests benchmark_adaptive_sssp -- --nocapture
```

## Expected Output

### Performance Comparison Example
```
Performance Comparison:
Graph Size   GPU Dijkstra    Hybrid Duan     Speedup
-------------------------------------------------------
1K           152μs           251μs           0.61x
5K           412μs           523μs           0.79x
10K          789μs           856μs           0.92x
```

### Adaptive Switching Example
```
Adaptive Algorithm Selection:
Graph Type      Nodes        Avg Deg    Selected Algorithm
-------------------------------------------------------------
Small sparse    1000         5          GPUDijkstra
Medium          50000        8          GPUDijkstra
Large dense     1500000      15         HybridDuan
Large sparse    2000000      3          GPUDijkstra
```

### Benchmark Suite Example
```
======================================================================
ADAPTIVE SSSP BENCHMARK SUITE
======================================================================

Test: Small Random
  Nodes: 1000, Edges: 9980
  Selected: GPUDijkstra
  Time: 152.3μs
  Reachable: 998/1000 nodes

Test: Medium Grid
  Nodes: 10000, Edges: 19800
  Selected: GPUDijkstra
  Time: 1.24ms
  Reachable: 10000/10000 nodes

Test: Large Scale-Free
  Nodes: 50000, Edges: 399984
  Selected: GPUDijkstra
  Time: 8.76ms
  Reachable: 49998/50000 nodes

======================================================================
✓ Benchmark suite completed
```

## Test Data Characteristics

### Small Graph (1K nodes)
- Edges: ~10K (p=0.01 Erdős-Rényi)
- Avg degree: ~10
- Diameter: ~3-5 hops
- Components: Typically 1 (connected)

### Medium Graph (100K nodes)
- Edges: ~500K (scale-free, k=5)
- Avg degree: ~5
- Diameter: ~log(n) ≈ 17
- Hub nodes: ~√n ≈ 316

### Large Graph (10M nodes, simulated)
- Edges: ~150M (scale-free, k=15)
- Avg degree: ~15
- Diameter: ~log(n) ≈ 23
- Hub nodes: ~√n ≈ 3162

## Validation Criteria

### ✅ Pass Conditions

1. **Algorithm Selection**: 100% accuracy on test cases
2. **Correctness**: Max error < 1e-5 vs ground truth
3. **Performance**: Speedup measured and logged
4. **Fallback**: Graceful degradation verified
5. **Edge Cases**: All corner cases handled

### ❌ Fail Conditions

1. Wrong algorithm selected for graph characteristics
2. Distance errors > 1e-5
3. Crashes or panics on edge cases
4. Memory usage > 10MB for 10K node graph
5. Fallback fails to recover from errors

## Performance Targets

### GPU Dijkstra
- **Small (1K)**: < 500μs
- **Medium (10K)**: < 5ms
- **Large (100K)**: < 50ms

### Hybrid Duan
- **Medium (10K)**: < 10ms
- **Large (100K)**: < 100ms
- **Very Large (1M)**: < 1s

**Note**: Current mock implementations have artificial delays. Real CUDA kernels will be significantly faster.

## Future Enhancements

### Phase 1: CUDA Integration
- [ ] Replace mock GPU Dijkstra with actual CUDA kernels
- [ ] Implement real Hybrid Duan GPU/CPU split
- [ ] Add CUDA error handling and fallback

### Phase 2: Advanced Testing
- [ ] Add concurrent query tests (multi-source)
- [ ] Test bidirectional Dijkstra
- [ ] Add A* with heuristics
- [ ] Test graph streaming/dynamic updates

### Phase 3: Real-World Graphs
- [ ] Test on actual social network data
- [ ] Validate on road networks
- [ ] Test on knowledge graph datasets

### Phase 4: Optimization Tests
- [ ] Test HNSW indexing integration
- [ ] Validate landmark selection
- [ ] Test approximate APSP

## Dependencies

```toml
[dependencies]
anyhow = "1.0"
rand = "0.8"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

## Integration with CUDA Kernels

The test suite is designed to integrate with the existing CUDA implementation:

### GPU Pathfinding Module
**File**: `src/rust/gpu_engine/pathfinding.rs`
- `find_shortest_paths()` - Main entry point
- `find_paths_dijkstra()` - GPU Dijkstra implementation
- `PathfindingConfig` - Configuration struct

### CUDA Headers
**File**: `src/cuda/kernels/graph_search.cuh`
- `launch_sssp_semantic()` - SSSP with semantic scoring
- `launch_bounded_dijkstra()` - Localized search
- `launch_k_shortest_paths()` - Alternative paths

### Integration Points

1. **Replace Mock Functions**:
   - `gpu_dijkstra()` → Call `find_paths_dijkstra()` from pathfinding module
   - Add device/memory pool initialization

2. **Add Real Hybrid Implementation**:
   - Use `launch_bounded_dijkstra()` for local neighborhoods
   - Implement hub detection and CPU coordination

3. **Performance Validation**:
   - Re-run benchmarks with real CUDA
   - Validate speedup claims
   - Tune crossover thresholds

## Troubleshooting

### "Unable to find library -lunified_gpu"
The main project requires CUDA libraries. Run tests standalone:
```bash
rustc --test tests/adaptive_sssp_tests.rs --crate-type lib
./adaptive_sssp_tests
```

### Test Timeouts
Reduce graph sizes for faster testing:
```rust
let graph = generate_scale_free_graph(100, 5, 42); // Smaller
```

### Memory Issues
Large graphs may require streaming or batching:
```rust
// Process in chunks
for chunk in sources.chunks(1000) {
    adaptive.compute(&graph, chunk)?;
}
```

## Contact

For issues or enhancements to the test suite:
- File: `/home/devuser/workspace/hackathon-tv5/tests/adaptive_sssp_tests.rs`
- Docs: `/home/devuser/workspace/hackathon-tv5/tests/docs/ADAPTIVE_SSSP_TESTS.md`

---

**Last Updated**: 2025-12-04
**Test Suite Version**: 1.0.0
**Status**: ✅ Comprehensive test coverage implemented
