# Hyper-Personalization System

GPU-accelerated hyper-personalization system for semantic search that integrates all components into a unified end-to-end pipeline.

## Architecture

```
┌─────────┐
│  Query  │
└────┬────┘
     │
     ▼
┌──────────────────┐
│ Semantic Model   │  ← ONNX inference
│ (Query Encoding) │
└────┬─────────────┘
     │ embedding (384D)
     ▼
┌──────────────────┐
│  User Fusion     │  ← GPU user embeddings
│  (0.7Q + 0.3U)   │
└────┬─────────────┘
     │ fused embedding
     ▼
┌──────────────────┐
│ Temporal Cache   │  ← Hot query cache
│ (Check/Miss)     │
└────┬─────────────┘
     │ similarities
     ▼
┌──────────────────┐
│ GPU Similarity   │  ← CUDA GEMM
│ (Batch MatMul)   │
└────┬─────────────┘
     │ top-2K candidates
     ▼
┌──────────────────┐
│ Attention        │  ← PyTorch reranking
│ (Context-Aware)  │
└────┬─────────────┘
     │ top-K results
     ▼
┌──────────────────┐
│ Search Result    │
└──────────────────┘
```

## Features

- **End-to-End Pipeline**: Integrates all components (semantic model, user embeddings, cache, attention)
- **GPU Acceleration**: Full CUDA acceleration for embeddings and similarity computation
- **Temporal Caching**: Sub-millisecond cache hits for popular queries
- **Attention Reranking**: Context-aware result reranking with multi-head attention
- **Real-time Updates**: User preference updates with immediate effect
- **Performance Tracking**: Detailed timing breakdown and metrics

## Performance Targets

- **End-to-end latency**: <10ms (p95)
- **Cache hit latency**: <0.2ms
- **GPU similarity**: <2ms for 100K items
- **Attention reranking**: <0.3ms for top-20 candidates
- **Throughput**: >1000 QPS

## Usage

```rust
use hyper_personalization::{HyperPersonalizationSystem, SystemConfig};

// Initialize system
let config = SystemConfig {
    embedding_dim: 384,
    max_query_length: 512,
    cache_size: 10_000,
    cache_ttl_secs: 3600,
    use_gpu: true,
};

let mut system = HyperPersonalizationSystem::new(
    "models/semantic_model.onnx",
    "models/tokenizer.json",
    "embeddings/items.bin",
    "embeddings/users.bin",
    config,
)?;

// Perform personalized search
let result = system.personalized_search(
    "user123",           // User ID
    "action movie",      // Query
    10,                  // Top-K
    None,                // Context (optional)
)?;

// Print results
println!("Found {} items in {:.2}ms",
    result.items.len(),
    result.timing.total_ms
);

for (idx, score) in result.items {
    println!("  Item {}: {:.4}", idx, score);
}

// Update user preferences
system.update_user_preferences("user123", 42, 0.9)?;
```

## Components

### 1. Semantic Model (`semantic-model`)

- **Backend**: ONNX Runtime
- **Model**: Sentence transformer (e.g., MiniLM)
- **Features**:
  - Fast query encoding
  - Mean/max/CLS pooling
  - GPU acceleration via CUDA

### 2. User Embeddings (`gpu-embeddings`)

- **Backend**: CUDA
- **Features**:
  - Real-time updates
  - Embedding fusion (query + user)
  - Efficient GPU storage

### 3. Temporal Cache (`temporal-cache`)

- **Backend**: CUDA + cuBLAS
- **Features**:
  - Precomputed similarities for hot items
  - Exponential temporal decay
  - Sub-ms cache hits

### 4. Attention Reranker (`attention`)

- **Backend**: PyTorch (libtorch)
- **Features**:
  - Multi-head attention
  - Context injection (time, genre, social)
  - Fast GPU inference

## Timing Breakdown

```
Total: 8.2ms
├─ Query Encoding:    2.1ms  (ONNX)
├─ User Fusion:       0.3ms  (GPU)
├─ Similarity:        4.5ms  (CUDA GEMM)
├─ Attention Rerank:  1.1ms  (PyTorch)
└─ Top-K Selection:   0.2ms  (CPU)
```

## Integration Tests

Run integration tests (requires test data):

```bash
# Generate test data
cargo run --bin generate-test-data

# Run tests
cargo test --package hyper-personalization --test integration_test -- --ignored
```

## Performance Benchmarks

```bash
# Run benchmarks
cargo bench --package hyper-personalization

# Compare with Python baseline
python scripts/benchmark_comparison.py
```

## Memory Usage

For 100K items (384D embeddings):

- **Item embeddings**: ~153 MB (100K × 384 × 4 bytes)
- **User embeddings**: ~15 MB (10K users)
- **Temporal cache**: ~2.4 GB (10K popular × 100K items × 4 bytes)
- **Attention weights**: ~10 MB
- **Total GPU**: ~2.6 GB

## Configuration

```rust
pub struct SystemConfig {
    pub embedding_dim: usize,        // 384 for MiniLM
    pub max_query_length: usize,     // 512 tokens
    pub cache_size: usize,           // 10K popular items
    pub cache_ttl_secs: u64,         // 3600 (1 hour)
    pub use_gpu: bool,               // true for production
}
```

## Error Handling

```rust
use hyper_personalization::HyperPersonalizationError;

match system.personalized_search(...) {
    Ok(result) => { /* handle success */ },
    Err(HyperPersonalizationError::Model(e)) => { /* model error */ },
    Err(HyperPersonalizationError::Gpu(e)) => { /* GPU error */ },
    Err(HyperPersonalizationError::UserNotFound(id)) => { /* user not found */ },
    Err(e) => { /* other error */ },
}
```

## Monitoring

```rust
// Get cache statistics
let cache_stats = system.cache_stats();
println!("Cache hit rate: {:.2}%", cache_stats.hit_rate * 100.0);

// Get performance metrics
let metrics = system.metrics();
println!("P95 latency: {:.2}ms", metrics.p95_latency_ms);

// Reset metrics
system.reset_metrics();
```

## Dependencies

- **cudarc**: CUDA operations
- **ort**: ONNX Runtime
- **tch**: PyTorch bindings
- **tokenizers**: HuggingFace tokenizers
- **serde**: Serialization
- **tracing**: Logging

## License

MIT
