# Rust Crate Architecture Diagram

## System Layer Diagram (C4 Model - Level 2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Client Applications                             │
│                    (HTTP API, CLI, Embedded Systems)                     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    hyper-personalization (Integration)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   System     │  │   Search     │  │ Preferences  │                  │
│  │   Manager    │  │   Engine     │  │   Update     │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                  │                  │                          │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Core Components                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │gpu-embeddings│  │temporal-cache│  │  attention   │                  │
│  │              │  │              │  │              │                  │
│  │ • User Emb   │  │ • Similarity │  │ • Multi-Head │                  │
│  │ • Hybrid Q   │  │ • Temporal   │  │ • Context    │                  │
│  │ • Updates    │  │ • Popularity │  │ • Reranking  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                  │                  │                          │
│         │                  │                  │                          │
│  ┌──────┴──────────────────┴──────────────────┴───────┐                 │
│  │           semantic-model (ONNX Runtime)             │                 │
│  │  • Tokenization  • Inference  • GPU Acceleration    │                 │
│  └─────────────────────────────────────────────────────┘                 │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      GPU Acceleration Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   cudarc     │  │   cuBLAS     │  │   cuDNN      │                  │
│  │ CUDA bindings│  │ Matrix ops   │  │ Neural ops   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      NVIDIA A100 GPU (42 GB)                             │
│  • 15.36 GB: User Embeddings                                             │
│  •  2.48 GB: Temporal Cache                                              │
│  •  0.29 GB: Item Embeddings                                             │
│  •  0.50 GB: Model Parameters                                            │
│  • ~18.63 GB Total (44% utilization)                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
cli
 └─▶ hyper-personalization
      ├─▶ gpu-embeddings
      │    └─▶ cudarc (GPU ops)
      ├─▶ temporal-cache
      │    ├─▶ cudarc (GPU ops)
      │    └─▶ gpu-embeddings
      ├─▶ attention
      │    ├─▶ cudarc (GPU ops)
      │    └─▶ gpu-embeddings
      └─▶ semantic-model
           ├─▶ ort (ONNX Runtime)
           └─▶ gpu-embeddings

benchmarks
 └─▶ hyper-personalization
      └─▶ (all transitive deps)
```

## Data Flow Diagram

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Semantic Model (ONNX)            │
│    • Tokenize query                 │
│    • Generate embedding (384-dim)   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    GPU User Embeddings              │
│    • Fetch user profile             │
│    • Hybrid query = 0.7*Q + 0.3*U   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Temporal Cache (GPU)             │
│    • Check cache (10K popular)      │
│    • Compute similarities           │
│    • Top-100 candidates             │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Multi-Head Attention             │
│    • Encode context                 │
│    • Rerank with attention          │
│    • Top-K final results            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Results   │
│   (JSON)    │
└─────────────┘
```

## GPU Memory Layout

```
A100 GPU (42 GB Total)
┌────────────────────────────────────────────────────┐
│ User Embeddings                        15.36 GB    │ ████████████████
│ (100K active × 384 dims × 4 bytes)                 │
├────────────────────────────────────────────────────┤
│ Temporal Cache                          2.48 GB    │ ███
│ (10K × 62K × 4 bytes)                               │
├────────────────────────────────────────────────────┤
│ Item Embeddings                         0.29 GB    │ ▌
│ (62K × 384 × 4 bytes)                               │
├────────────────────────────────────────────────────┤
│ Model Parameters                        0.50 GB    │ ▌
│ (ONNX Runtime + Attention)                          │
├────────────────────────────────────────────────────┤
│ Workspace Buffers                       1.00 GB    │ █
│ (Temporary computations)                            │
├────────────────────────────────────────────────────┤
│ Free Memory                            22.37 GB    │ ████████████████████
└────────────────────────────────────────────────────┘

Total Used: 19.63 GB (47%)
Total Free: 22.37 GB (53%)
```

## Thread Architecture

```
Main Thread
 │
 ├─▶ Tokio Runtime (async)
 │    ├─▶ Request Handler Tasks
 │    ├─▶ GPU Operation Tasks
 │    └─▶ Background Workers
 │
 ├─▶ CUDA Streams (per-thread)
 │    ├─▶ Stream 0: Embedding updates
 │    ├─▶ Stream 1: Similarity computation
 │    └─▶ Stream 2: Attention reranking
 │
 └─▶ Rayon Thread Pool (CPU)
      ├─▶ Preprocessing
      ├─▶ Postprocessing
      └─▶ Metrics aggregation
```

## Crate API Surface

### gpu-embeddings
```rust
pub struct UserEmbeddings { /* ... */ }

impl UserEmbeddings {
    pub fn new(num_users: usize, dim: usize, device: usize) -> Result<Self>;
    pub async fn get_user_embedding(&self, user_id: &str) -> Result<GpuTensor<f32>>;
    pub async fn update_from_interaction(&self, user_id: &str, item_emb: &GpuTensor<f32>, rating: f32) -> Result<()>;
    pub async fn hybrid_query_embedding(&self, query: &GpuTensor<f32>, user_id: &str, weight: f32) -> Result<GpuTensor<f32>>;
}
```

### temporal-cache
```rust
pub struct TemporalCache { /* ... */ }

impl TemporalCache {
    pub fn new(item_embeddings: GpuTensor<f32>, num_popular: usize) -> Result<Self>;
    pub async fn rebuild_cache(&self) -> Result<()>;
    pub async fn get_similar_items(&self, item_id: usize, top_k: usize, temporal: bool) -> Result<(Vec<usize>, Vec<f32>)>;
}
```

### attention
```rust
pub struct MultiHeadAttention { /* ... */ }

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self>;
    pub async fn forward(&self, query: &GpuTensor<f32>, candidates: &GpuTensor<f32>, context: Option<&Context>) -> Result<Vec<f32>>;
}
```

### semantic-model
```rust
pub struct SemanticEncoder { /* ... */ }

impl SemanticEncoder {
    pub fn new(model_path: &Path) -> Result<Self>;
    pub async fn encode(&self, text: &str) -> Result<GpuTensor<f32>>;
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<GpuTensor<f32>>>;
}
```

### hyper-personalization
```rust
pub struct HyperPersonalization { /* ... */ }

impl HyperPersonalization {
    pub fn builder() -> HyperPersonalizationBuilder;
    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse>;
    pub async fn update_preferences(&self, user_id: &str, item_id: &str, rating: f32) -> Result<()>;
}
```

## Build Profiles Performance Impact

```
Profile              | Binary Size | Link Time | Runtime Perf
---------------------|-------------|-----------|---------------
dev                  | 450 MB      | 12s       | Baseline (1x)
release              | 28 MB       | 45s       | 15x faster
release-lto          | 22 MB       | 180s      | 18x faster
release-with-debug   | 35 MB       | 50s       | 15x faster
```

## Feature Flag Matrix

```
Feature        | Default | GPU Memory | Latency Impact
---------------|---------|------------|---------------
cuda           | ✓       | +18 GB     | -95% (0.5ms)
cpu-only       |         | 0 GB       | Baseline (10ms)
onnx           | ✓       | +0.5 GB    | -20%
distributed    |         | +2 GB/node | -50% (parallel)
metrics        |         | +10 MB     | +0.01ms
simd           |         | 0 GB       | -10% (CPU)
```

## Testing Strategy Matrix

```
Crate               | Unit Tests | Integration | Benchmarks | Coverage
--------------------|------------|-------------|------------|----------
gpu-embeddings      | ✓✓✓        | ✓           | ✓✓         | >85%
temporal-cache      | ✓✓✓        | ✓           | ✓✓         | >80%
attention           | ✓✓✓        | ✓           | ✓✓         | >85%
semantic-model      | ✓✓         | ✓✓          | ✓          | >75%
hyper-personalization| ✓✓        | ✓✓✓         | ✓✓✓        | >80%
benchmarks          | ✓          | ✓✓✓         | ✓✓✓        | N/A
cli                 | ✓          | ✓✓          | ✓          | >70%
```

## Performance Bottleneck Analysis

```
Operation                    | Time (μs) | % Total | Optimization
-----------------------------|-----------|---------|-------------------
Query encoding (ONNX)        | 150       | 30%     | Batch inference
User fusion                  | 10        | 2%      | ✓ Optimized
GPU similarity (62K items)   | 200       | 40%     | cuBLAS GEMM
Attention reranking (100→10) | 100       | 20%     | Fused kernels
Result formatting            | 40        | 8%      | Zero-copy serde
-----------------------------|-----------|---------|-------------------
Total                        | 500       | 100%    | Target: <500μs
```

## Error Handling Strategy

```rust
// Custom error types per crate
#[derive(thiserror::Error, Debug)]
pub enum GpuEmbeddingError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("User not found: {0}")]
    UserNotFound(String),

    #[error("Out of memory: needed {needed} GB, available {available} GB")]
    OutOfMemory { needed: f32, available: f32 },
}

// Top-level error aggregation
#[derive(thiserror::Error, Debug)]
pub enum HyperPersonalizationError {
    #[error("GPU embeddings: {0}")]
    Embeddings(#[from] gpu_embeddings::GpuEmbeddingError),

    #[error("Temporal cache: {0}")]
    Cache(#[from] temporal_cache::CacheError),

    #[error("Attention: {0}")]
    Attention(#[from] attention::AttentionError),

    #[error("Semantic model: {0}")]
    SemanticModel(#[from] semantic_model::EncoderError),
}
```
