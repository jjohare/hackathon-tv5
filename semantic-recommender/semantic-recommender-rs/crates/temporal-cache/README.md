# temporal-cache

GPU-accelerated temporal similarity cache with exponential decay for real-time recommendation systems.

## Features

- **GPU Acceleration**: Batch matrix multiplication using CUDA and cuBLAS
- **Temporal Decay**: Exponential decay weights: `w(t) = exp(-λ * t)`
- **Sub-millisecond Latency**: <0.16ms target for cache hits
- **Atomic Tracking**: Thread-safe hit/miss counters
- **Memory Efficient**: 2.48 GB GPU memory for 10K×62K similarity matrix

## Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Cache Hit | <0.16ms | >6,000 QPS |
| Cache Miss | <2ms | >500 QPS |
| Rebuild (10K items) | <100ms | N/A |

## Usage

### Basic Setup

```rust
use temporal_cache::TemporalGPUCache;

// Load item embeddings (num_items * embed_dim)
let num_items = 62_000;
let embed_dim = 256;
let embeddings: Vec<f32> = load_embeddings();

// Create cache with top 10K popular items
let cache = TemporalGPUCache::new(
    &embeddings,
    num_items,
    embed_dim,
    Some(10_000),  // num_popular
    Some(0.1),     // decay_rate (λ)
)?;
```

### Query Similar Items

```rust
// Query cached item (fast path)
let result = cache.get_similar_items(item_id)?;

if result.from_cache {
    println!("Cache HIT: {:.3}ms", result.latency_ms);
} else {
    println!("Cache MISS: {:.3}ms", result.latency_ms);
}

// Get top-K similar items
let mut scored: Vec<_> = result.similarities
    .iter()
    .enumerate()
    .map(|(idx, &score)| (idx, score))
    .collect();

scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
let top_k = &scored[0..10];
```

### Custom Query Embeddings

```rust
// Query with user embedding
let user_embedding: Vec<f32> = get_user_embedding(user_id);

let result = cache.get_similarities(
    &user_embedding,
    std::time::Instant::now()
)?;

// Result contains similarities for all items
```

### Temporal Decay

```rust
// Update temporal weights based on cache age
cache.update_temporal_weights()?;

// Rebuild cache with new popular items
let new_popular_indices: Vec<usize> = get_trending_items();
cache.update_popular_indices(new_popular_indices)?;
```

### Cache Statistics

```rust
let stats = cache.cache_stats();

println!("Hit Rate: {:.2}%", stats.hit_rate * 100.0);
println!("Total Hits: {}", stats.total_hits);
println!("Total Misses: {}", stats.total_misses);
println!("Avg Hit Latency: {:.3}ms", stats.avg_hit_latency_ms);
println!("Cache Age: {:.1}s", stats.cache_age_secs);
```

## Architecture

### Memory Layout

```
GPU Memory:
┌─────────────────────────────────┐
│ item_embeddings                 │  62K × 256 × 4B = 63.5 MB
│ (num_items, embed_dim)          │
├─────────────────────────────────┤
│ popular_similarities            │  10K × 62K × 4B = 2.48 GB
│ (num_popular, num_items)        │
├─────────────────────────────────┤
│ temporal_weights                │  10K × 4B = 40 KB
│ (num_popular,)                  │
└─────────────────────────────────┘
Total: ~2.54 GB
```

### Cache Rebuild

```rust
// Batch matrix multiplication on GPU
let popular_emb = extract_popular_embeddings();  // (10K, 256)
let all_emb = item_embeddings;                   // (62K, 256)

// GEMM: similarities = popular_emb @ all_emb^T
// Result: (10K, 62K) similarity matrix
similarities = cuBLAS::gemm(popular_emb, all_emb.T);

// Apply temporal decay
temporal_weights = exp(-λ * cache_age);
```

### Query Flow

```
Query(item_id)
     │
     ├─ In Cache?
     │      │
     │     YES ──> GPU Copy (offset, length) ──> <0.16ms
     │      │
     │      NO ──> Extract Embedding ──> GEMM ──> <2ms
     │
     └─> Update Stats (atomic)
```

## Configuration

### Decay Rate (λ)

Controls how quickly cached similarities become stale:

- `λ = 0.1`: Half-life ~7 seconds (fast decay)
- `λ = 0.01`: Half-life ~70 seconds (moderate)
- `λ = 0.001`: Half-life ~700 seconds (slow)

Formula: `half_life = ln(2) / λ`

### Number of Popular Items

Trade-off between memory and hit rate:

| num_popular | GPU Memory | Expected Hit Rate |
|-------------|------------|-------------------|
| 1,000 | 250 MB | 60-70% |
| 5,000 | 1.2 GB | 75-85% |
| 10,000 | 2.5 GB | 85-95% |

## Benchmarks

Run benchmarks with:

```bash
cargo bench --features cuda
```

### Results (NVIDIA A100)

```
cache_hit/hit          time: [156.2 μs, 161.4 μs, 167.3 μs]
cache_miss/miss        time: [1.842 ms, 1.897 ms, 1.956 ms]
cache_rebuild/10000    time: [87.34 ms, 89.12 ms, 91.05 ms]
mixed_workload/80_20   time: [21.45 ms, 22.18 ms, 22.94 ms]
```

## Testing

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run with logging
RUST_LOG=debug cargo test
```

## Error Handling

```rust
use temporal_cache::{CacheError, Result};

match cache.get_similar_items(item_id) {
    Ok(result) => { /* process result */ },
    Err(CacheError::ItemNotFound(id)) => {
        eprintln!("Item {} not found", id);
    },
    Err(CacheError::GpuOperation(msg)) => {
        eprintln!("GPU error: {}", msg);
    },
    Err(e) => {
        eprintln!("Cache error: {}", e);
    }
}
```

## Requirements

- CUDA 11.7+ (or CPU fallback)
- GPU with ≥3GB VRAM for production scale
- Rust 1.70+

## License

MIT OR Apache-2.0
