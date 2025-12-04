# Adaptive SSSP Quick Reference

## One-Liner

```rust
use gpu_engine::{AdaptiveSSPConfig, SSPAlgorithm, find_adaptive_shortest_paths};
```

## Basic Usage (Auto Mode)

```rust
let config = AdaptiveSSPConfig::default();
let result = find_adaptive_shortest_paths(
    &device, &modules, &memory_pool, &streams,
    &graph, &sources, &targets, &config
).await?;
println!("Found {} paths in {:.2}ms", result.paths.len(), result.metrics.total_time_ms);
```

## Algorithms

| Algorithm | Best For | Node Count |
|-----------|----------|------------|
| `GPUDijkstra` | Small-Medium | < 100K |
| `HybridDuan` | Large | >= 100K |
| `Auto` | All (Recommended) | Any |

## Quick Config Templates

### Production (Balanced)
```rust
AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::Auto,
    enable_profiling: false,
    ..Default::default()
}
```

### Small Graphs (Low Latency)
```rust
AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::GPUDijkstra,
    enable_profiling: true,
    ..Default::default()
}
```

### Large Graphs (Scalability)
```rust
AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::HybridDuan,
    hybrid_cpu_threads: 16,
    hybrid_batch_size: 50_000,
    ..Default::default()
}
```

### Path-Limited Search
```rust
AdaptiveSSPConfig {
    max_depth: 5,
    max_paths: 10,
    ..Default::default()
}
```

### Custom Threshold
```rust
AdaptiveSSPConfig {
    algorithm: SSPAlgorithm::Auto,
    crossover_threshold: 50_000,
    ..Default::default()
}
```

## Graph Format

```rust
// Edge list: [from, to, from, to, ...]
let graph = vec![0, 1, 1, 2, 2, 3];
let sources = vec![0];
let targets = vec![3];
```

## Accessing Results

```rust
let result = find_adaptive_shortest_paths(...).await?;

// Paths
for path in &result.paths {
    println!("{:?}", path.nodes);  // [0, 1, 2, 3]
    println!("{}", path.cost);     // 3.0
    println!("{}", path.length);   // 3
}

// Metrics
let m = &result.metrics;
println!("Algorithm: {}", m.algorithm_used);
println!("Time: {:.2}ms", m.total_time_ms);
println!("Throughput: {:.0} nodes/sec", m.throughput_nodes_per_sec());
println!("GPU: {:.1}%", m.gpu_utilization());
```

## Detect Optimal Threshold

```rust
let threshold = detect_crossover_threshold(
    &device, &modules, &memory_pool, &streams
).await?;
println!("Optimal: {} nodes", threshold);
```

## Common Patterns

### Error Handling
```rust
match find_adaptive_shortest_paths(...).await {
    Ok(result) => println!("Success: {} paths", result.paths.len()),
    Err(e) => eprintln!("Error: {}", e),
}
```

### Algorithm Comparison
```rust
let gpu_config = AdaptiveSSPConfig { algorithm: SSPAlgorithm::GPUDijkstra, ..Default::default() };
let hybrid_config = AdaptiveSSPConfig { algorithm: SSPAlgorithm::HybridDuan, ..Default::default() };

let gpu_result = find_adaptive_shortest_paths(..., &gpu_config).await?;
let hybrid_result = find_adaptive_shortest_paths(..., &hybrid_config).await?;

println!("GPU: {:.2}ms", gpu_result.metrics.total_time_ms);
println!("Hybrid: {:.2}ms", hybrid_result.metrics.total_time_ms);
```

### Batch Processing
```rust
for (graph, sources, targets) in graphs {
    let result = find_adaptive_shortest_paths(
        &device, &modules, &memory_pool, &streams,
        &graph, &sources, &targets, &config
    ).await?;
    process_paths(result.paths);
}
```

## Performance Hints

- **< 10K nodes**: Use `GPUDijkstra` for low latency
- **10K-100K nodes**: Use `Auto` for optimal selection
- **> 100K nodes**: Use `HybridDuan` for scalability
- **Dense graphs**: Prefer `GPUDijkstra`
- **Sparse graphs**: Prefer `HybridDuan`
- **Production**: Disable profiling (`enable_profiling: false`)
- **Development**: Enable profiling for insights

## Tuning Parameters

| Parameter | Small Graphs | Large Graphs |
|-----------|--------------|--------------|
| `crossover_threshold` | 50,000 | 150,000 |
| `hybrid_cpu_threads` | 4 | 16 |
| `hybrid_batch_size` | 10,000 | 50,000 |
| `max_depth` | 10 | 20 |
| `max_paths` | 100 | 1,000 |

## Metrics Cheat Sheet

```rust
metrics.total_time_ms           // Total execution time
metrics.gpu_compute_ms          // GPU kernel time
metrics.cpu_compute_ms          // CPU processing time
metrics.transfer_time_ms        // Memory transfer time
metrics.nodes_processed         // Nodes examined
metrics.edges_traversed         // Edges examined
metrics.paths_found             // Paths returned
metrics.peak_gpu_memory         // Peak memory (bytes)
metrics.avg_path_length         // Average path length
metrics.throughput_nodes_per_sec() // Nodes/sec
metrics.gpu_utilization()       // GPU usage %
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `GpuError::Cuda` | CUDA failure | Check GPU availability |
| `GpuError::Memory` | OOM | Reduce graph size or use Hybrid |
| `GpuError::Stream` | Sync failed | Check stream initialization |

## Testing

```bash
# Unit tests
cargo test adaptive_sssp

# Integration tests
cargo test --test adaptive_sssp_integration

# Run example
cargo run --example adaptive_sssp_example --release
```

## File Locations

- **Core**: `/src/rust/gpu_engine/adaptive_sssp.rs`
- **Tests**: `/src/rust/gpu_engine/tests/adaptive_sssp_integration.rs`
- **Example**: `/src/rust/examples/adaptive_sssp_example.rs`
- **Docs**: `/docs/adaptive_sssp_readme.md`
- **API**: `/docs/adaptive_sssp_api_reference.md`

## Need More?

- Full docs: `docs/adaptive_sssp_readme.md`
- API reference: `docs/adaptive_sssp_api_reference.md`
- Implementation: `docs/adaptive_sssp_implementation_summary.md`
- Example code: `src/rust/examples/adaptive_sssp_example.rs`
