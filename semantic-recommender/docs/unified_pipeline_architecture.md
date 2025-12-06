# Unified GPU Pipeline Architecture

## Overview

The unified GPU pipeline integrates all three CUDA optimization phases into a single high-performance system for semantic similarity search.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified GPU Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Query Embeddings [batch_size, embedding_dim]        │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Phase 3: Candidate Generation (10-100x reduction)   │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ LSH Hash Computation                                 │   │
│  │  • 8 hash tables × 10 bits = 1024 buckets          │   │
│  │  • Parallel hash computation via GPU                │   │
│  │  • Candidate retrieval: O(batch_size × bucket_size)│   │
│  │                                                      │   │
│  │ HNSW Index (Optional)                              │   │
│  │  • Hierarchical graph search                        │   │
│  │  • Warp-level parallelism for beam search          │   │
│  │  • Shared memory priority queues                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│         Candidates: ~1000 per query                         │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Phase 2: Memory Optimization (4-5x speedup)        │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Batch Sorting                                        │   │
│  │  • Sort pairs by candidate_idx                      │   │
│  │  • Enables coalesced memory access                  │   │
│  │  • Memory bandwidth: 60 GB/s → 280+ GB/s           │   │
│  │                                                      │   │
│  │ Prefetching & Caching                               │   │
│  │  • Precomputed L2 norms                            │   │
│  │  • Shared memory embedding cache                    │   │
│  │  • Double buffering for overlap                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│         Sorted pairs ready for similarity computation        │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Phase 1: Tensor Core Acceleration (8-10x speedup)  │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ FP16 Matrix Multiply                                │   │
│  │  • WMMA (Warp Matrix Multiply-Accumulate)          │   │
│  │  • 16×16×16 tiles processed per warp               │   │
│  │  • Peak throughput: 65 TFLOPS on T4 GPU            │   │
│  │                                                      │   │
│  │ Cosine Similarity                                   │   │
│  │  • Dot product via tensor cores                     │   │
│  │  • Normalization with cached norms                  │   │
│  │  • Vectorized half2 operations                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│         Similarity scores for all candidates                 │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Top-K Selection                                      │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ Warp-Level Reduction                                │   │
│  │  • Shared memory priority queues                    │   │
│  │  • Per-query top-k extraction                       │   │
│  │  • Parallel sorting for final ranking               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  Output: Top-K results [batch_size, k]                     │
│         + Similarity scores                                  │
└─────────────────────────────────────────────────────────────┘
```

## Performance Targets

### Individual Phase Speedups
- **Phase 1 (Tensor Cores)**: 8-10x vs FP32 scalar operations
- **Phase 2 (Memory Opt)**: 4-5x vs random access patterns
- **Phase 3 (Indexing)**: 10-100x candidate reduction

### Combined Performance
- **Target**: 300-500x speedup vs naive baseline
- **Latency**: <5ms for 1M vectors @ 1024-dim, k=10
- **Throughput**: 200+ QPS on single T4 GPU

## Memory Layout

### Embeddings Storage
```
FP16 format: [num_embeddings, embedding_dim]
Memory: num_embeddings × embedding_dim × 2 bytes
Example: 1M × 1024 × 2 = 2 GB

Sorted for coalesced access (Phase 2)
```

### LSH Hash Tables
```
Structure: [num_tables, num_buckets, max_bucket_size]
Example: 8 × 1024 × 256 = 2M entries × 4 bytes = 8 MB
```

### Precomputed Norms
```
Format: FP32
Memory: num_embeddings × 4 bytes
Example: 1M × 4 = 4 MB
```

## Key Optimizations

### 1. Multi-Stream Execution
- 4 CUDA streams for overlapping operations
- Pipeline: Hash → Retrieve → Sort → Compute
- Hides memory latency with computation

### 2. Shared Memory Usage
- 48 KB per SM on T4
- Embedding caching: 16 KB
- Priority queues: 8 KB
- Scratch space: 24 KB

### 3. Coalesced Memory Access
- Consecutive threads access consecutive memory
- Achieved via sorting pairs by candidate_idx
- 4-5x bandwidth improvement

### 4. Tensor Core Utilization
- Requires 16-byte aligned data
- Processes 16×16 tiles per warp
- ~95% theoretical peak on tensor workloads

## Integration with Rust

### FFI Boundary
```rust
extern "C" {
    fn unified_pipeline_create(...) -> *mut Pipeline;
    fn unified_pipeline_search_knn(...);
    fn unified_pipeline_destroy(...);
}
```

### Safety Guarantees
- RAII wrapper for automatic cleanup
- Send + Sync for thread safety
- Error propagation via Result<T>

### Zero-Copy Data Transfer
- Direct GPU memory access
- Pinned host memory for faster transfers
- Memory pooling for reusable buffers

## Benchmarking

### Test Configurations
1. **Small**: 10K vectors, 1024-dim
2. **Medium**: 100K vectors, 1024-dim
3. **Large**: 1M vectors, 1024-dim
4. **Extra Large**: 10M vectors, 1024-dim

### Metrics
- Latency (p50, p95, p99)
- Throughput (QPS)
- Memory usage
- GPU utilization

### Comparison Baselines
- CPU SIMD (AVX2)
- GPU naive implementation
- FAISS library
- Qdrant vector database

## Future Enhancements

### Phase 4: Dynamic Pruning
- Adaptive candidate generation
- Early termination based on confidence
- Expected: 2-3x additional speedup

### Phase 5: Multi-GPU Scaling
- Sharded embeddings across GPUs
- Parallel search with aggregation
- Linear scaling to 8 GPUs

### Phase 6: Compression
- 8-bit quantization (2x memory reduction)
- Product quantization integration
- Minimal accuracy loss (<1%)

## References

- [NVIDIA Tensor Core Programming Guide](https://docs.nvidia.com/cuda/tensor-core-programming/)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [LSH Survey](https://arxiv.org/abs/1411.3787)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
