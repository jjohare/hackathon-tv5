# Temporal GPU Cache

Rust implementation of the TemporalGPUCache from `scripts/gpu_hyper_personalization.py`.

## Features

- **Pre-computed Similarity Matrix**: 10K × 62K GPU-accelerated cache
- **Sub-millisecond Lookups**: <0.16ms cache hits vs 0.5ms cold computation
- **Temporal Decay**: Exponential time-based weighting for recency bias
- **Cache Rebuild**: Hourly refresh mechanism for dynamic catalogs
- **80-90% Hit Rate**: Optimized for Zipf-distributed access patterns

## Architecture

```
TemporalGPUCache
├── item_embeddings: Tensor (62K × 384)      # Full catalog on GPU
├── popular_indices: Tensor (10K)            # Top items by frequency
├── popular_similarities: Tensor (10K × 62K) # Pre-computed matrix (2.48 GB)
└── temporal_weights: Tensor (62K)           # Exponential decay weights
```

## Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Cache Hit | <0.16ms | 2.48 GB GPU |
| Cache Miss | ~0.5ms | - |
| Rebuild | ~2-5s | - |

## Python Original

Ported from lines 160-259 in `scripts/gpu_hyper_personalization.py`:
- `__init__`: Initialize with embeddings and popular item count
- `rebuild_cache()`: Batch matrix multiplication (10K × 384) @ (384 × 62K)
- `get_similar_items()`: Fast lookup with temporal weighting
- `_compute_temporal_weights()`: Exponential decay w_i = exp(-λ * age_i)

## Usage

```rust
use temporal_cache::TemporalGPUCache;
use tch::{Device, Kind, Tensor};

// Load item embeddings (62K × 384)
let embeddings = Tensor::randn(&[62_000, 384], (Kind::Float, Device::Cuda(0)));

// Initialize cache with top 10K popular items
let cache = TemporalGPUCache::new(embeddings, 10_000)?;

// Fast lookup with temporal decay
let result = cache.get_similar_items(
    item_id,
    top_k: 10,
    apply_temporal: true,
)?;

println!("Cache hit: {}, Time: {:.3}ms",
    result.cache_hit,
    result.lookup_time_ms
);

// Periodic rebuild (e.g., hourly cron)
cache.rebuild_cache()?;
```

## Build Requirements

**PyTorch Installation Required**:
```bash
export LIBTORCH_USE_PYTORCH=1
pip install torch  # Must match tch-rs version (0.16.x -> PyTorch 2.0+)
cargo build --package temporal-cache
```

Or download libtorch manually and set `LIBTORCH` environment variable.

## Testing

```bash
cargo test --package temporal-cache
```

## Benchmarks

```bash
cargo bench --package temporal-cache
```

Expected results:
- `cache_hit_lookup`: <0.16ms
- `cache_miss_lookup`: ~0.5ms
- `cache_rebuild`: ~2-5s (62K items)

## Dependencies

- `tch`: PyTorch Rust bindings for GPU tensors
- `tokio`: Async runtime for background rebuild
- `tracing`: Instrumentation and logging

## License

MIT
