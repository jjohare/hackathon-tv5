# Attention Reranker - Rust Implementation

## Overview

Rust implementation of the MultiHeadAttentionReranker from Python (`scripts/gpu_hyper_personalization.py` lines 177-275), providing context-aware reranking for hyper-personalized recommendations.

**Location:** `/home/devuser/workspace/hackathon-tv5/semantic-recommender/src/rust/attention_reranker.rs`

## Architecture

```
Query Embedding + Context
      ↓
Q/K/V Projections (384-dim)
      ↓
Scaled Dot-Product Attention
      ↓
Attention Weights + Candidates
      ↓
Reranked Scores
```

## Key Components

### 1. ContextFeatures

Encodes personalization context into a fixed-size representation:

```rust
pub struct ContextFeatures {
    pub time_of_day: [f32; 3],    // [morning, afternoon, evening]
    pub genre_prefs: [f32; 3],    // [action, drama, comedy]
    pub social_signal: [f32; 2],  // [solo, group]
}
```

**Factory Methods:**
- `ContextFeatures::default()` - Neutral preferences
- `ContextFeatures::morning()` - Morning viewing context
- `ContextFeatures::evening()` - Evening viewing context
- `ContextFeatures::with_genres(action, drama, comedy)` - Custom genre preferences

**Encoding:**
- `encode()` → 8-dimensional feature vector
- Used for context injection into query embeddings

### 2. AttentionReranker

Main reranking engine with simplified single-head attention:

```rust
pub struct AttentionReranker {
    embed_dim: usize,
    query_proj: LinearProjection,   // Q projection
    key_proj: LinearProjection,     // K projection
    value_proj: LinearProjection,   // V projection
    out_proj: LinearProjection,     // Output projection
    context_proj: LinearProjection, // Context encoder
    last_inference_time: Cell<f64>, // Performance tracking
}
```

**Key Methods:**

#### `new(embed_dim: usize) -> Self`
Initialize reranker with Xavier-initialized weights.

#### `encode_context(&self, context: &ContextFeatures) -> Array1<f32>`
Project 8-dim context vector to embed_dim space.

#### `forward(&self, query_emb, candidate_embs, context) -> Array1<f32>`
Compute attention-weighted scores:
1. Add context to query (if provided): `query' = query + 0.3 * context_vec`
2. Project to Q/K/V: `Q = W_q * query'`, `K = W_k * candidates`, `V = W_v * candidates`
3. Scaled dot-product: `scores = (Q · K^T) / sqrt(d)`
4. Softmax: `attention_weights = softmax(scores)`
5. Apply attention: `attended = sum(attention_weights * V)`
6. Output projection: `output = W_o * attended`
7. Final scores: `output · candidates^T`

#### `rerank(&self, candidates, candidate_embs, query_emb, base_scores, context) -> Vec<(usize, f32)>`
Full reranking pipeline:
- Combines attention scores with base similarity scores
- Weighted average: `final = 0.7 * attention + 0.3 * base`
- Returns sorted (candidate_id, score) pairs

## Implementation Details

### Simplified Architecture

Unlike the Python multi-head version, this implements **single-head attention** for performance:

```python
# Python (multi-head, not used)
self.num_heads = 8
self.head_dim = embed_dim // num_heads

# Rust (single-head, actually used in Python forward())
# Just uses full embed_dim with single attention head
```

This matches the actual Python implementation in `forward()` which uses simplified single-head attention.

### Linear Projections

```rust
struct LinearProjection {
    weight: Array2<f32>,  // (out_dim, in_dim)
    bias: Array1<f32>,    // (out_dim,)
}
```

**Initialization:** Xavier uniform: `scale = sqrt(6 / (in_dim + out_dim))`

**Forward:**
- Single: `y = W * x + b`
- Batch: `Y = X * W^T + b`

### Scaled Dot-Product Attention

```rust
// Scale factor
let scale = (embed_dim as f32).sqrt();

// Attention scores
let scores = k.dot(&q) / scale;  // (N,)

// Softmax normalization
let attention_weights = softmax(scores);

// Weighted sum
let attended = sum(attention_weights * V)
```

## Memory Usage

| Component | Size | Details |
|-----------|------|---------|
| Query projection | ~150 KB | 384 × 384 weights + 384 bias |
| Key projection | ~150 KB | 384 × 384 weights + 384 bias |
| Value projection | ~150 KB | 384 × 384 weights + 384 bias |
| Output projection | ~150 KB | 384 × 384 weights + 384 bias |
| Context projection | ~12 KB | 8 × 384 weights + 384 bias |
| **Total** | **~612 KB** | All parameters in f32 |

Additional runtime memory:
- Attention weights: `4 * N bytes` (N = num candidates)
- Intermediate activations: `~15 KB` (Q/K/V/attended vectors)

## Performance

### Target

**<0.2ms** reranking overhead for 100 candidates

### Current Implementation (CPU)

Based on the architecture (pure Rust ndarray):

| Candidates | Expected Time | Note |
|------------|---------------|------|
| 10 | <0.1ms | Likely meets target |
| 100 | 0.2-1.0ms | May exceed target on CPU |
| 1000 | 2-10ms | GPU acceleration needed |

### Performance Tracking

The reranker tracks inference time internally:

```rust
let time_ms = reranker.last_inference_time_ms();
println!("Last inference: {:.4}ms", time_ms);
```

### GPU Migration Path

Structure is ready for GPU acceleration:

**Current:** ndarray (CPU)
```rust
use ndarray::{Array1, Array2};
let scores = candidates.dot(&query);
```

**Future:** tch-rs or cudarc (GPU)
```rust
use tch::{Tensor, Device};
let device = Device::Cuda(0);
let scores = candidates.matmul(&query).to_device(device);
```

## Testing

### Unit Tests (in module)

```bash
# All tests in attention_reranker.rs
cargo test --lib attention_reranker
```

**Test Coverage:**
- `test_context_features_default` - Default context creation
- `test_context_encoding` - 8-dim encoding
- `test_linear_projection` - Projection layers
- `test_softmax` - Softmax normalization
- `test_attention_reranker_creation` - Initialization
- `test_attention_forward` - Forward pass without context
- `test_attention_forward_with_context` - Forward pass with context
- `test_rerank` - Full reranking pipeline
- `test_performance_target` - Performance benchmark (100 candidates)
- `test_context_variants` - Different context effects

### Demo (standalone)

```bash
cargo run --example attention_demo
```

**Note:** Requires `ndarray` in workspace dependencies:
```toml
[workspace.dependencies]
ndarray = "0.15"
```

## Usage Example

```rust
use recommendation_engine::attention_reranker::{AttentionReranker, ContextFeatures};
use ndarray::{Array1, Array2};

// Initialize reranker
let reranker = AttentionReranker::new(384);

// Create context
let context = ContextFeatures::evening()
    .with_genres(0.7, 0.2, 0.1);  // Action fan

// Prepare data
let query = Array1::from_elem(384, 0.1);
let candidate_ids = vec![0, 1, 2, 3, 4];
let candidate_embs = Array2::from_elem((5, 384), 0.05);
let base_scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];

// Rerank
let reranked = reranker.rerank(
    &candidate_ids,
    &candidate_embs.view(),
    &query.view(),
    &base_scores,
    Some(&context),
);

// Process results
for (id, score) in reranked.iter().take(3) {
    println!("Candidate {}: {:.4}", id, score);
}
```

## Comparison with Python

| Feature | Python | Rust |
|---------|--------|------|
| Architecture | Multi-head (declared) | Single-head |
| Actual implementation | Single-head | Single-head |
| Embed dim | 384 | 384 |
| Context features | 8 (time, genre, social) | 8 (same) |
| Projections | Q/K/V/Out | Q/K/V/Out + Context |
| Framework | PyTorch | ndarray (pure Rust) |
| Device | CUDA | CPU (GPU-ready) |
| Memory | <1 MB | ~612 KB |
| Performance target | +0.1ms | <0.2ms |

## Python Source Reference

**File:** `scripts/gpu_hyper_personalization.py`
**Lines:** 261-360

**Key sections ported:**
1. **Lines 270-280:** Projection layers → `LinearProjection` struct
2. **Lines 284-314:** Context encoding → `encode_context()` method
3. **Lines 316-360:** Forward pass → `forward()` method

**Simplification note:** Python forward() uses simplified single-head attention (line 345), not true multi-head. Rust implementation matches this.

## Dependencies

Required in `src/rust/Cargo.toml`:

```toml
[dependencies]
ndarray = "0.15"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
```

## Future Enhancements

### GPU Acceleration
1. Migrate to `tch-rs` for PyTorch Rust bindings
2. Or use `cudarc` for direct CUDA integration
3. Add GPU memory management
4. Implement batched operations

### Model Loading
```rust
impl AttentionReranker {
    pub fn load_from_pytorch(path: &str) -> Result<Self> {
        // Load pre-trained weights from Python model
    }

    pub fn save(&self, path: &str) -> Result<()> {
        // Save weights for persistence
    }
}
```

### Multi-Head Support (if needed)
```rust
struct MultiHeadAttentionReranker {
    num_heads: usize,
    heads: Vec<AttentionReranker>,
}
```

### Quantization
```rust
pub struct QuantizedAttentionReranker {
    // Int8 weights for 4x memory reduction
    // Runtime dequantization
}
```

## Integration Points

### Recommendation Pipeline

```rust
// 1. Initial retrieval (similarity search)
let candidates = gpu_engine.search(&query, 100)?;

// 2. Context-aware reranking
let context = ContextFeatures::from_user_session(&session);
let reranked = reranker.rerank(
    &candidate_ids,
    &candidate_embs,
    &query_emb,
    &base_scores,
    Some(&context),
)?;

// 3. Return top-k
reranked.into_iter().take(10).collect()
```

### AgentDB Memory Integration

Store reranking decisions for learning:

```rust
// After reranking
let decision = json!({
    "query_id": query_id,
    "context": context,
    "reranked_ids": reranked.iter().map(|(id, _)| id).collect::<Vec<_>>(),
    "timestamp": Utc::now(),
});

agentdb.store("reranking_decisions", &decision).await?;
```

## Benchmarking

### Micro-benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_forward(c: &mut Criterion) {
    let reranker = AttentionReranker::new(384);
    let query = Array1::from_elem(384, 0.1);
    let candidates = Array2::from_elem((100, 384), 0.05);

    c.bench_function("attention_forward_100", |b| {
        b.iter(|| {
            reranker.forward(
                black_box(&query.view()),
                black_box(&candidates.view()),
                None
            )
        })
    });
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
```

### Run benchmarks

```bash
cargo bench --bench attention_benchmarks
```

## Troubleshooting

### Compilation Errors

If `ndarray` is not found:
```bash
# Add to workspace dependencies in root Cargo.toml
[workspace.dependencies]
ndarray = "0.15"

# Then in src/rust/Cargo.toml
[dependencies]
ndarray.workspace = true
```

### Performance Issues

If performance target not met:
1. Profile with `cargo flamegraph`
2. Check for unnecessary allocations
3. Consider GPU acceleration
4. Enable BLAS backend for ndarray:
   ```toml
   ndarray = { version = "0.15", features = ["blas"] }
   blas-src = { version = "0.9", default-features = false, features = ["openblas"] }
   ```

### Memory Issues

Monitor allocations:
```rust
let start_mem = get_memory_usage();
let _result = reranker.forward(&query, &candidates, None);
let end_mem = get_memory_usage();
println!("Memory delta: {} KB", (end_mem - start_mem) / 1024);
```

## References

- **Python Source:** `scripts/gpu_hyper_personalization.py` (lines 261-360)
- **ndarray Documentation:** https://docs.rs/ndarray/
- **Attention Mechanism:** "Attention Is All You Need" (Vaswani et al., 2017)
- **Xavier Initialization:** "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)

## Author

Ported from Python to Rust
Date: 2025-12-07
Version: 1.0.0
