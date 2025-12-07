# Attention Reranker Crate

GPU-accelerated attention-based reranking with context awareness using PyTorch (tch-rs).

## Features

- **Multi-head Attention**: Scaled dot-product attention for semantic reranking
- **Context Awareness**: Inject temporal, genre, and social signals
- **GPU Acceleration**: Full GPU execution with tch-rs (PyTorch bindings)
- **High Performance**: Target <0.2ms latency for 100 candidates on GPU
- **Xavier Initialization**: Proper weight initialization for stable training

## Architecture

```rust
pub struct AttentionReranker {
    query_proj: nn::Linear,    // Query projection
    key_proj: nn::Linear,      // Key projection
    value_proj: nn::Linear,    // Value projection
    out_proj: nn::Linear,      // Output projection
    context_proj: nn::Linear,  // Context encoding (8 -> embed_dim)
}
```

### Context Features

```rust
pub struct ContextFeatures {
    time_of_day: [f32; 3],    // morning, afternoon, evening
    genre_prefs: [f32; 3],    // action, drama, comedy
    social_signal: [f32; 2],  // solo, group
}
```

## Usage

```rust
use attention::{AttentionReranker, ContextFeatures};
use tch::{nn, Device, Tensor, Kind};

// Initialize on GPU
let device = Device::Cuda(0);
let vs = nn::VarStore::new(device);
let reranker = AttentionReranker::new(&vs.root(), 384);

// Create query and candidate embeddings
let query = Tensor::randn(&[384], (Kind::Float, device));
let candidates = Tensor::randn(&[100, 384], (Kind::Float, device));

// Optional: Add context
let context = ContextFeatures::new(
    [1.0, 0.0, 0.0],  // Morning
    [0.0, 1.0, 0.0],  // Drama
    [1.0, 0.0],       // Solo
);

// Forward pass
let scores = reranker.forward(&query, &candidates, Some(&context));

// Top-k selection
let (_, top_k_indices) = scores.topk(10, -1, true, true);
```

## Implementation Details

### Attention Mechanism

1. **Project inputs**:
   ```
   Q = query @ W_q
   K = candidates @ W_k
   V = candidates @ W_v
   ```

2. **Compute attention scores**:
   ```
   scores = (Q @ K^T) / sqrt(d_k)
   weights = softmax(scores)
   ```

3. **Weighted aggregation**:
   ```
   context = weights @ V
   output = context @ W_out
   ```

4. **Context injection**:
   ```
   query' = query + 0.3 * encode_context(context)
   ```

### Scoring Formula

Final scores combine attention and base similarity:

```
final_score = 0.7 * attention_score + 0.3 * base_score
```

## Requirements

### System Dependencies

**PyTorch/LibTorch**: Required for tch-rs

```bash
# Option 1: Use Python PyTorch
export LIBTORCH_USE_PYTORCH=1

# Option 2: Install LibTorch manually
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-*.zip
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Option 3: System-wide install
sudo ln -s /path/to/libtorch/lib/libtorch.so /usr/lib/
```

**CUDA** (for GPU acceleration):
```bash
export CUDA_PATH=/usr/local/cuda
export CUDA_LIBRARY_PATH=/usr/local/cuda/lib64
```

### Cargo Features

```toml
[features]
default = ["cuda"]
cuda = []              # Enable CUDA support (requires GPU)
cpu-fallback = []      # CPU-only execution
```

## Performance Targets

| Metric | Target | Hardware |
|--------|--------|----------|
| Latency (100 candidates) | <0.2ms | GPU (A100) |
| Latency (100 candidates) | <2ms | CPU (16 cores) |
| Throughput (batch=16) | >80K QPS | GPU (A100) |
| Memory (384-dim) | <50MB | GPU |

## Benchmarks

```bash
cargo bench -p attention
```

Available benchmarks:
- `context_encoding` - Context feature encoding
- `attention_forward` - Forward pass (10-500 candidates)
- `attention_batched` - Batched queries (1-16 batch size)
- `rerank_pipeline` - End-to-end reranking
- `gpu_transfer` - CPU↔GPU transfer overhead
- `end_to_end_latency` - Full pipeline latency

## Testing

```bash
# Run tests (requires LibTorch)
cargo test -p attention

# Run specific test
cargo test -p attention test_forward_with_context
```

### Test Coverage

- ✅ Context encoding
- ✅ Single query forward pass
- ✅ Batched query forward pass
- ✅ Context injection
- ✅ Reranking pipeline
- ✅ Model save/load
- ✅ Edge cases (empty candidates)

## File Structure

```
attention/
├── src/
│   ├── lib.rs          # Main reranker implementation
│   ├── error.rs        # Error types
│   └── utils.rs        # Utility functions
├── benches/
│   └── attention_bench.rs  # Performance benchmarks
├── Cargo.toml
└── README.md
```

## Integration Example

```rust
// In hyper-personalization system
use attention::{AttentionReranker, ContextFeatures};

impl RecommendationEngine {
    fn rerank_with_context(
        &self,
        query_emb: &Tensor,
        candidates: Vec<Content>,
        context: UserContext,
    ) -> Vec<Content> {
        // Encode candidates
        let cand_embs = self.encode_batch(&candidates);

        // Create context features
        let ctx = ContextFeatures::new(
            context.time_preferences,
            context.genre_preferences,
            context.social_signal,
        );

        // Rerank with attention
        let scores = self.reranker.forward(
            query_emb,
            &cand_embs,
            Some(&ctx),
        );

        // Return sorted candidates
        self.sort_by_scores(candidates, scores)
    }
}
```

## GPU Optimization Tips

1. **Batch Processing**: Process multiple queries simultaneously
2. **Persistent GPU Memory**: Keep embeddings on GPU
3. **Mixed Precision**: Use FP16 for faster computation
4. **Kernel Fusion**: Minimize GPU kernel launches
5. **Asynchronous Execution**: Overlap CPU and GPU work

## Troubleshooting

### LibTorch Not Found

```
Error: Cannot find a libtorch install
```

**Solution**: Set `LIBTORCH_USE_PYTORCH=1` or install LibTorch manually.

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or embedding dimension.

### Slow Performance

**Check**:
1. GPU utilization: `nvidia-smi`
2. Data transfer: Minimize CPU↔GPU copies
3. Batch size: Increase for better GPU utilization

## References

- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

MIT
