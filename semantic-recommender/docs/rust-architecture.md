# Rust Crate Architecture for Semantic Recommender System

## Architecture Decision Record (ADR-001)

**Date**: 2025-12-07
**Status**: Proposed
**Context**: Converting GPU hyper-personalization system from Python (PyTorch) to Rust

---

## System Overview

Converting 1,010 lines of Python (gpu_hyper_personalization.py + benchmark_hyper_personalization.py) into a high-performance, memory-safe Rust implementation with GPU acceleration.

### Python Implementation Analysis

**Core Components**:
1. **GPUUserEmbeddings** (610 lines)
   - 10M users × 384 dims = 15.36 GB GPU memory
   - Adaptive learning rate (α = 0.15)
   - Sparse → Dense storage transition
   - Real-time embedding updates (<0.1ms)

2. **TemporalGPUCache** (400 lines)
   - 10K popular × 62K items = 2.48 GB GPU memory
   - Precomputed similarity matrices
   - Temporal decay (exponential)
   - Cache hit rate: 80-90%

3. **MultiHeadAttentionReranker**
   - 8 heads × 48 dims per head
   - Context-aware scoring
   - <1 MB GPU memory
   - +20-40% quality improvement

4. **GPUHyperPersonalization** (Integration)
   - Total GPU memory: ~18 GB / 42 GB (43%)
   - Expected latency: <0.5ms
   - Throughput: 500K+ QPS

---

## Proposed Rust Crate Structure

```
semantic-recommender-rs/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── gpu-embeddings/           # User embedding management
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── user_embeddings.rs
│   │   │   ├── hybrid_query.rs
│   │   │   └── memory.rs
│   │   └── benches/
│   │       └── embedding_bench.rs
│   │
│   ├── temporal-cache/           # GPU similarity caching
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── cache.rs
│   │   │   ├── temporal_decay.rs
│   │   │   └── popularity.rs
│   │   └── benches/
│   │       └── cache_bench.rs
│   │
│   ├── attention/                # Multi-head attention
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── reranker.rs
│   │   │   ├── context.rs
│   │   │   └── projection.rs
│   │   └── benches/
│   │       └── attention_bench.rs
│   │
│   ├── semantic-model/           # Sentence transformer wrapper
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── encoder.rs
│   │   │   └── onnx_runtime.rs
│   │   └── models/
│   │       └── paraphrase-multilingual-MiniLM-L12-v2.onnx
│   │
│   ├── hyper-personalization/    # Integration layer
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── system.rs
│   │   │   ├── search.rs
│   │   │   └── preferences.rs
│   │   └── examples/
│   │       └── demo.rs
│   │
│   ├── benchmarks/               # Criterion benchmarks
│   │   ├── Cargo.toml
│   │   ├── benches/
│   │   │   ├── latency.rs
│   │   │   ├── throughput.rs
│   │   │   ├── memory.rs
│   │   │   └── quality.rs
│   │   └── results/
│   │
│   └── cli/                      # Command-line interface
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
```

---

## Dependency Selection Matrix

### GPU Acceleration Framework

| Framework | Pros | Cons | Decision |
|-----------|------|------|----------|
| **tch-rs** | PyTorch bindings, mature | Requires libtorch (heavy), C++ deps | ❌ Too heavy |
| **burn.rs** | Pure Rust, modern | Immature, limited CUDA support | ❌ Not production ready |
| **candle** | HuggingFace, lightweight | Limited ops, early stage | ⚠️ Consider for inference |
| **cudarc** | Direct CUDA control, minimal overhead | Low-level, manual memory | ✅ **SELECTED** |

**Decision**: **cudarc** for maximum performance and memory control
- Direct CuBLAS/CuDNN bindings
- Minimal overhead (no Python/C++ layer)
- Full control over GPU memory lifecycle
- Compatible with CUDA 11.7+ (A100 support)

### Async Runtime

| Runtime | Pros | Cons | Decision |
|---------|------|------|----------|
| **tokio** | Industry standard, full-featured | Heavier than alternatives | ✅ **SELECTED** |
| **async-std** | Simpler API | Less ecosystem support | ❌ |
| **smol** | Lightweight | Limited GPU integration | ❌ |

**Decision**: **tokio** for async GPU operations and concurrent request handling

### Serialization

| Crate | Pros | Cons | Decision |
|-------|------|------|----------|
| **serde** | Universal standard | N/A | ✅ **SELECTED** |
| **bincode** | Fast binary | Limited cross-lang | ✅ For internal cache |
| **rmp-serde** | MessagePack | Slower than bincode | ⚠️ Optional |

### ONNX Runtime (Semantic Model)

| Crate | Pros | Cons | Decision |
|-------|------|------|----------|
| **ort** | Official bindings, GPU support | Large runtime | ✅ **SELECTED** |
| **tract** | Pure Rust | Limited ops | ❌ Missing transformer ops |

---

## Memory Safety Strategy

### GPU Tensor Lifecycle Management

```rust
// Problem: CUDA memory is not managed by Rust's ownership system
// Solution: RAII wrapper with Drop implementation

pub struct GpuTensor<T> {
    device_ptr: CudaSlice<T>,
    shape: Vec<usize>,
    device: CudaDevice,
}

impl<T> Drop for GpuTensor<T> {
    fn drop(&mut self) {
        // Automatic GPU memory deallocation
        // cudarc handles synchronization
    }
}
```

### Send/Sync for CUDA Contexts

```rust
// Problem: CudaDevice is not Send by default
// Solution: Arc + Mutex with thread-local streams

pub struct GpuContext {
    device: Arc<CudaDevice>,
    streams: Arc<Mutex<Vec<CudaStream>>>,
}

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

// Thread-local stream pool
thread_local! {
    static STREAM: RefCell<Option<CudaStream>> = RefCell::new(None);
}
```

### Shared State Patterns

```rust
// User embeddings: Arc for shared immutable refs
pub struct UserEmbeddings {
    dense_embeddings: Arc<RwLock<GpuTensor<f32>>>,
    user_id_map: Arc<DashMap<String, usize>>,
    interaction_counts: Arc<DashMap<String, u32>>,
}

// Temporal cache: Arc for read-heavy workload
pub struct TemporalCache {
    similarities: Arc<GpuTensor<f32>>,  // Immutable after rebuild
    popular_indices: Arc<GpuTensor<i32>>,
    temporal_weights: Arc<GpuTensor<f32>>,
}
```

### Zero-Copy Patterns

```rust
// Avoid CPU↔GPU copies with pinned memory
use cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP;

pub struct PinnedBuffer<T> {
    host_ptr: *mut T,
    device_ptr: CudaDevicePtr,
    len: usize,
}

// Zero-copy transfer
impl<T> PinnedBuffer<T> {
    pub fn new(len: usize, device: &CudaDevice) -> Result<Self> {
        let host_ptr = unsafe {
            cuMemHostAlloc(len * size_of::<T>(), CU_MEMHOSTALLOC_DEVICEMAP)
        }?;
        let device_ptr = cuMemHostGetDevicePointer(host_ptr)?;
        Ok(Self { host_ptr, device_ptr, len })
    }
}
```

---

## Performance Optimizations

### Profile Configuration

```toml
[profile.release]
opt-level = 3              # Maximum optimizations
lto = "thin"               # Thin LTO (faster builds than "fat")
codegen-units = 1          # Single codegen unit for max inlining
panic = "abort"            # Smaller binary, faster panics
strip = true               # Strip debug symbols

[profile.bench]
inherits = "release"
debug = true               # Keep debug info for profiling

[profile.release-with-debug]
inherits = "release"
debug = true
```

### CUDA Kernel Optimizations

```rust
// Use cuBLAS for matrix operations (optimized for A100)
use cudarc::cublas::{CudaBlas, Gemm};

pub struct OptimizedMatmul {
    cublas: CudaBlas,
}

impl OptimizedMatmul {
    pub fn matmul_f32(
        &self,
        a: &GpuTensor<f32>,  // (M, K)
        b: &GpuTensor<f32>,  // (K, N)
    ) -> Result<GpuTensor<f32>> {
        // cuBLAS uses column-major, need transpose
        // C = A @ B → cublas_gemm(B^T, A^T)^T
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let c = GpuTensor::zeros(&[m, n], &self.device)?;

        unsafe {
            self.cublas.gemm(
                1.0,  // alpha
                b,    // B^T
                a,    // A^T
                0.0,  // beta
                &c,   // C
            )?;
        }

        Ok(c)
    }
}
```

---

## Feature Flags

```toml
[features]
default = ["cuda", "onnx"]
cuda = ["cudarc/cuda-11070"]
cpu-only = []
onnx = ["ort"]
benchmarks = ["criterion"]
distributed = ["tonic", "prost"]
metrics = ["prometheus"]
```

---

## API Design

### High-Level Interface

```rust
use hyper_personalization::{HyperPersonalization, SearchRequest, Context};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize system
    let system = HyperPersonalization::builder()
        .device(0)  // GPU 0
        .num_users(10_000_000)
        .num_popular_items(10_000)
        .load_embeddings("data/embeddings/media")?
        .build()
        .await?;

    // Personalized search
    let request = SearchRequest::new("sci-fi movies with time travel")
        .user_id("user_demo_001")
        .top_k(10)
        .context(Context {
            time_of_day: [0.2, 0.1, 0.7],
            genre_prefs: [0.7, 0.2, 0.1],
            social_signal: [1.0, 0.0],
        });

    let results = system.search(request).await?;

    println!("Latency: {:.2}ms", results.timing.total_ms);
    for (i, item) in results.items.iter().enumerate() {
        println!("{}: {} (score: {:.3})", i+1, item.title, item.score);
    }

    Ok(())
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_user_embedding_update() {
        let embeddings = UserEmbeddings::new(100, 384, 0)?;
        let item_emb = GpuTensor::randn(&[384], &embeddings.device)?;

        embeddings.update_from_interaction(
            "user_1",
            &item_emb,
            0.8,
        ).await?;

        let user_emb = embeddings.get_user_embedding("user_1").await?;
        assert!(user_emb.norm() > 0.0);
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_end_to_end_search() {
    let system = HyperPersonalization::builder()
        .device(0)
        .num_users(1000)
        .load_embeddings("tests/data/embeddings")?
        .build()
        .await?;

    let results = system.search(
        SearchRequest::new("action movies")
            .user_id("test_user")
            .top_k(5)
    ).await?;

    assert_eq!(results.items.len(), 5);
    assert!(results.timing.total_ms < 1.0);
}
```

### Benchmark Suite

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let system = rt.block_on(async {
        HyperPersonalization::builder()
            .device(0)
            .build()
            .await
            .unwrap()
    });

    c.bench_function("search_latency", |b| {
        b.to_async(&rt).iter(|| async {
            system.search(
                SearchRequest::new("test query")
                    .user_id("bench_user")
                    .top_k(10)
            ).await.unwrap()
        })
    });
}

criterion_group!(benches, bench_latency);
criterion_main!(benches);
```

---

## Migration Strategy

### Phase 1: Core Infrastructure
1. Create workspace structure
2. Implement GPU tensor abstractions
3. Port user embeddings logic
4. Unit tests for embeddings

### Phase 2: Caching Layer
1. Implement temporal cache
2. Port similarity computation
3. Benchmark cache performance

### Phase 3: Attention Mechanism
1. Port multi-head attention
2. Context encoding
3. Integration tests

### Phase 4: Semantic Model
1. Export PyTorch model to ONNX
2. ONNX runtime integration
3. Inference benchmarks

### Phase 5: Integration & Benchmarks
1. Wire all components
2. End-to-end tests
3. Performance validation vs Python

---

## Quality Attributes

### Performance
- **Target Latency**: <0.5ms (P95)
- **Target Throughput**: 500K+ QPS
- **GPU Memory**: <20 GB (A100 48GB)
- **CPU Memory**: <2 GB

### Reliability
- **Error Handling**: Result<T> everywhere, no panics in library code
- **GPU Recovery**: Automatic context reset on CUDA errors
- **Graceful Degradation**: Fallback to CPU if GPU unavailable

### Maintainability
- **Code Coverage**: >80% for core logic
- **Documentation**: All public APIs documented
- **Examples**: Comprehensive usage examples
- **Benchmarks**: Track performance regressions

### Security
- **Input Validation**: Sanitize all user inputs
- **Memory Safety**: Zero unsafe outside cudarc bindings
- **Dependency Audit**: cargo-audit in CI

---

## Technology Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| cudarc API instability | High | Medium | Pin to specific version, vendor if needed |
| ONNX runtime compatibility | Medium | Low | Thorough testing, fallback to PyTorch export |
| Memory fragmentation | Medium | Medium | Custom allocator, periodic defragmentation |
| CUDA context conflicts | High | Low | Thread-local contexts, careful synchronization |

---

## Success Metrics

### Functional
- [ ] All Python functionality ported
- [ ] API parity with Python version
- [ ] Integration tests passing

### Performance
- [ ] Latency <0.5ms (P95)
- [ ] Throughput >500K QPS
- [ ] Memory <20 GB GPU

### Quality
- [ ] Code coverage >80%
- [ ] Zero unsafe in business logic
- [ ] Documentation complete

---

## Next Steps

1. Create workspace Cargo.toml
2. Implement gpu-embeddings crate
3. Port user embedding logic
4. Write unit tests
5. Benchmark against Python baseline
