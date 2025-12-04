# CUDA Optimization Guide

## Table of Contents
- [Overview](#overview)
- [Phase 1: Tensor Core Optimization](#phase-1-tensor-core-optimization)
- [Phase 2: Memory Optimization](#phase-2-memory-optimization)
- [Phase 3: Algorithm Complexity Analysis](#phase-3-algorithm-complexity-analysis)
- [Benchmarking Guide](#benchmarking-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers three progressive optimization phases for GPU-accelerated vector similarity search:

| Phase | Focus | Expected Speedup | Target Hardware |
|-------|-------|------------------|-----------------|
| 1 | Tensor Cores | 5-10x | V100, A100, H100 |
| 2 | Memory Management | 2-4x additional | All CUDA GPUs |
| 3 | Algorithm Optimization | 10-100x total | Production systems |

### System Requirements
- CUDA 11.0+ (12.0+ recommended)
- cuBLAS, cuDNN libraries
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- 16GB+ GPU memory (32GB+ recommended)

---

## Phase 1: Tensor Core Optimization

### What Are Tensor Cores?

Tensor Cores are specialized hardware units on NVIDIA GPUs (Volta, Turing, Ampere, Hopper architectures) designed for mixed-precision matrix operations.

**Key Capabilities:**
- **FP16/BF16 Matrix Multiplication**: 8x-16x faster than FP32
- **Mixed Precision**: FP16 computation with FP32 accumulation
- **Automatic Casting**: PyTorch AMP handles precision management

### Implementation Strategy

#### 1. Baseline Implementation (FP32)
```python
# Standard similarity search - NO tensor cores
def similarity_search_fp32(query_vectors, database_vectors):
    """
    Args:
        query_vectors: [batch_size, dim] float32
        database_vectors: [num_vectors, dim] float32
    Returns:
        similarities: [batch_size, num_vectors] float32
    """
    return torch.mm(query_vectors, database_vectors.T)
```

**Performance**: ~500 GFLOPS on A100

#### 2. Tensor Core Optimized (FP16)
```python
import torch
from torch.cuda.amp import autocast

@autocast(dtype=torch.float16)
def similarity_search_fp16(query_vectors, database_vectors):
    """
    Args:
        query_vectors: [batch_size, dim] float32 (auto-cast to fp16)
        database_vectors: [num_vectors, dim] float32 (auto-cast to fp16)
    Returns:
        similarities: [batch_size, num_vectors] float32
    """
    # Computation happens in FP16 using Tensor Cores
    # Result auto-cast back to FP32
    return torch.mm(query_vectors, database_vectors.T)
```

**Performance**: ~4000 GFLOPS on A100 (8x speedup)

#### 3. Optimized Data Layout
```python
def prepare_tensors_for_tensor_cores(vectors):
    """
    Tensor Cores perform best with specific alignment:
    - Dimensions divisible by 8 (FP16) or 16 (INT8)
    - Contiguous memory layout
    - Proper batch sizes (multiples of 8)
    """
    # Ensure contiguous memory
    vectors = vectors.contiguous()

    # Pad dimensions to multiples of 8
    dim = vectors.shape[-1]
    if dim % 8 != 0:
        pad_size = 8 - (dim % 8)
        vectors = torch.nn.functional.pad(vectors, (0, pad_size))

    return vectors
```

### Tensor Core Activation Checklist

✅ **Use PyTorch AMP**: Automatic mixed precision
✅ **Matrix Dimensions**: Divisible by 8 (FP16) or 16 (INT8)
✅ **Batch Sizes**: Multiples of 8 for optimal occupancy
✅ **Contiguous Memory**: Call `.contiguous()` before operations
✅ **Proper Data Types**: torch.float16 or torch.bfloat16

### Precision Considerations

| Precision | Range | Tensor Cores | Use Case |
|-----------|-------|--------------|----------|
| FP32 | ±3.4e38 | ❌ | Baseline, debugging |
| FP16 | ±65,504 | ✅ | Most vector search |
| BF16 | ±3.4e38 | ✅ | Training, large models |
| INT8 | ±127 | ✅ | Extreme optimization |

**Recommendation**: Start with FP16, use BF16 if encountering overflow.

### Code Example: Full Pipeline

```python
class TensorCoreVectorSearch:
    def __init__(self, database_vectors, use_amp=True):
        """
        Args:
            database_vectors: [N, D] numpy array or torch tensor
            use_amp: Enable Automatic Mixed Precision
        """
        self.device = torch.device('cuda')

        # Prepare database vectors
        db = torch.tensor(database_vectors, dtype=torch.float32)
        db = prepare_tensors_for_tensor_cores(db)
        self.db_vectors = db.to(self.device)

        self.use_amp = use_amp

        # Pre-normalize for cosine similarity
        self.db_vectors = torch.nn.functional.normalize(
            self.db_vectors, p=2, dim=1
        )

    def search(self, query_vectors, top_k=10):
        """
        Args:
            query_vectors: [B, D] batch of query vectors
            top_k: Return top-k most similar
        Returns:
            scores: [B, top_k] similarity scores
            indices: [B, top_k] database indices
        """
        queries = torch.tensor(query_vectors, dtype=torch.float32)
        queries = prepare_tensors_for_tensor_cores(queries)
        queries = queries.to(self.device)

        # Normalize queries
        queries = torch.nn.functional.normalize(queries, p=2, dim=1)

        # Compute similarities with Tensor Cores
        with autocast(enabled=self.use_amp, dtype=torch.float16):
            similarities = torch.mm(queries, self.db_vectors.T)

        # Top-k selection (happens in FP32)
        scores, indices = torch.topk(similarities, k=top_k, dim=1)

        return scores.cpu().numpy(), indices.cpu().numpy()
```

### Verification

```python
# Check if Tensor Cores are being used
def verify_tensor_core_usage():
    import torch.cuda.profiler as profiler

    # Profile a sample operation
    query = torch.randn(128, 768, device='cuda', dtype=torch.float16)
    db = torch.randn(10000, 768, device='cuda', dtype=torch.float16)

    profiler.start()
    result = torch.mm(query, db.T)
    profiler.stop()

    # Check CUDA kernel names in nvidia-smi or nsys
    # Look for: "volta_fp16_*" or "ampere_fp16_*" kernels
```

**Expected Kernel Names:**
- Volta: `volta_fp16_s884gemm_fp16_*`
- Ampere: `ampere_fp16_s16816gemm_*`
- Hopper: `hopper_fp16_tf32_*`

---

## Phase 2: Memory Optimization

### Memory Hierarchy on GPU

```
L1 Cache (128KB per SM)     [Fastest, smallest]
    ↓ ~28 cycles
Shared Memory (64-100KB)    [Programmable, per-block]
    ↓ ~200 cycles
L2 Cache (40-80MB)          [Shared across SMs]
    ↓ ~200-500 cycles
HBM (16-80GB)               [Main GPU memory]
    ↓ ~500+ cycles
Host Memory (RAM)           [Slowest, largest]
```

### Optimization Techniques

#### 1. Shared Memory for Tiling

```python
def matmul_with_shared_memory(A, B):
    """
    Manual tiling implementation using shared memory.
    PyTorch handles this automatically, but understanding helps.
    """
    # Tile sizes optimized for shared memory
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32

    # PyTorch cuBLAS automatically does this
    # This is conceptual - actual CUDA kernel would be in C++
    result = torch.mm(A, B)  # PyTorch handles tiling

    return result
```

**Key Insight**: PyTorch's cuBLAS backend automatically tiles operations. Focus on:
- Keeping working sets in L2 cache
- Minimizing memory transfers
- Using contiguous tensors

#### 2. Memory Access Patterns

```python
# ❌ BAD: Non-contiguous access (cache misses)
def bad_memory_pattern(vectors):
    result = []
    for i in range(len(vectors)):
        result.append(vectors[i, ::2])  # Strided access
    return torch.stack(result)

# ✅ GOOD: Contiguous access (cache hits)
def good_memory_pattern(vectors):
    # Reshape and slice contiguously
    return vectors[:, ::2].contiguous()
```

#### 3. Batch Processing for Cache Reuse

```python
class BatchedVectorSearch:
    def __init__(self, database_vectors, batch_size=1024):
        self.db_vectors = database_vectors
        self.batch_size = batch_size

    def search(self, query_vectors, top_k=10):
        """
        Process queries in batches to fit in L2 cache
        """
        all_scores = []
        all_indices = []

        # Split database into chunks that fit in L2 cache
        db_chunk_size = 100000  # ~300MB for 768-dim FP16

        for db_start in range(0, len(self.db_vectors), db_chunk_size):
            db_chunk = self.db_vectors[db_start:db_start + db_chunk_size]

            # Process queries in batches
            for q_start in range(0, len(query_vectors), self.batch_size):
                q_batch = query_vectors[q_start:q_start + self.batch_size]

                with autocast(dtype=torch.float16):
                    scores = torch.mm(q_batch, db_chunk.T)

                batch_scores, batch_indices = torch.topk(scores, k=top_k, dim=1)
                batch_indices += db_start  # Offset for global indices

                all_scores.append(batch_scores)
                all_indices.append(batch_indices)

        # Merge results across chunks
        return self.merge_topk_results(all_scores, all_indices, top_k)
```

#### 4. Pinned Memory for Fast Transfers

```python
# Standard host-to-device transfer (~10 GB/s)
def slow_transfer(cpu_data):
    gpu_data = torch.tensor(cpu_data).cuda()
    return gpu_data

# Pinned memory transfer (~25 GB/s on PCIe 4.0)
def fast_transfer(cpu_data):
    # Allocate pinned (page-locked) memory
    pinned = torch.tensor(cpu_data).pin_memory()
    gpu_data = pinned.cuda(non_blocking=True)
    return gpu_data
```

#### 5. Memory Pool Management

```python
# Avoid frequent allocations
class MemoryEfficientSearch:
    def __init__(self, max_batch_size, max_db_size, embedding_dim):
        # Pre-allocate buffers
        self.similarity_buffer = torch.empty(
            (max_batch_size, max_db_size),
            dtype=torch.float16,
            device='cuda'
        )
        self.topk_scores = torch.empty(
            (max_batch_size, 100),
            dtype=torch.float32,
            device='cuda'
        )
        self.topk_indices = torch.empty(
            (max_batch_size, 100),
            dtype=torch.int64,
            device='cuda'
        )

    def search(self, queries, database, top_k):
        # Reuse pre-allocated buffers - no allocation overhead
        batch_size = queries.shape[0]
        db_size = database.shape[0]

        sim_view = self.similarity_buffer[:batch_size, :db_size]
        torch.mm(queries, database.T, out=sim_view)

        torch.topk(sim_view, k=top_k, out=(self.topk_scores, self.topk_indices))

        return self.topk_scores[:batch_size, :top_k], \
               self.topk_indices[:batch_size, :top_k]
```

### Memory Optimization Checklist

✅ **Use `.contiguous()`**: Ensure contiguous memory layout
✅ **Batch Operations**: Process multiple items together
✅ **Pre-allocate Buffers**: Reuse memory across calls
✅ **Pinned Memory**: For host↔device transfers
✅ **Avoid Copies**: Use in-place operations when possible
✅ **Profile Memory**: Use `torch.cuda.max_memory_allocated()`

### Profiling Memory Usage

```python
import torch

def profile_memory_usage(func, *args):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    result = func(*args)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

    print(f"Peak GPU memory: {peak_memory:.2f} GB")
    return result
```

---

## Phase 3: Algorithm Complexity Analysis

### Brute Force vs. Approximate Search

| Method | Time Complexity | Space Complexity | Recall |
|--------|----------------|------------------|--------|
| Brute Force | O(N·D) | O(N·D) | 100% |
| HNSW | O(log N·D) | O(N·D·M) | 95-99% |
| LSH | O(L·K·D) | O(N·D) | 80-95% |
| PQ | O(N·D/8) | O(N·D/8) | 85-95% |

Where:
- N = number of vectors
- D = embedding dimension
- M = HNSW connectivity (typically 16-48)
- L = number of hash tables (LSH)
- K = hash length (LSH)

### When to Use Each Algorithm

#### Brute Force (GPU-Accelerated)
**Use when:**
- Database size < 1M vectors
- Recall must be 100%
- GPU memory available
- Real-time updates needed

**Implementation:**
```python
# Optimized brute force with Tensor Cores
def brute_force_search(queries, database, top_k=10):
    with autocast(dtype=torch.float16):
        similarities = torch.mm(queries, database.T)
    scores, indices = torch.topk(similarities, k=top_k, dim=1)
    return scores, indices

# Scalability: ~1M vectors in 20-50ms (A100)
```

#### HNSW (Hierarchical Navigable Small World)
**Use when:**
- Database size: 1M - 100M vectors
- Need 95%+ recall
- Can tolerate 2-5x memory overhead
- Infrequent updates

**Performance Characteristics:**
```python
# HNSW complexity analysis
def hnsw_complexity(N, D, M=16, ef_search=100):
    """
    N: number of vectors
    D: dimension
    M: max connections per layer
    ef_search: search quality parameter
    """
    # Construction time
    construction = N * math.log(N) * D * M  # O(N log N D M)

    # Search time per query
    search = math.log(N) * D * ef_search  # O(log N D ef)

    # Memory overhead
    memory = N * D * 4 + N * M * 4 * math.log2(N)  # ~2-3x base size

    return {
        'construction_time': construction,
        'search_time_per_query': search,
        'memory_bytes': memory
    }
```

**Tuning Parameters:**
- `M` (connectivity): Higher = better recall, more memory (16-48)
- `ef_construction`: Build quality (100-200)
- `ef_search`: Search quality (50-500)

#### LSH (Locality Sensitive Hashing)
**Use when:**
- Database size: 10M - 1B vectors
- Can tolerate 80-90% recall
- Need fast updates
- Memory constrained

**Implementation Strategy:**
```python
class LSHIndex:
    def __init__(self, num_tables=8, hash_length=12):
        """
        num_tables (L): More tables = higher recall, slower
        hash_length (K): Longer = fewer collisions, more memory
        """
        self.L = num_tables
        self.K = hash_length

        # Random projection matrices
        self.projections = [
            torch.randn(embedding_dim, hash_length)
            for _ in range(num_tables)
        ]

    def hash_vector(self, vector):
        """
        Time complexity: O(L·K·D)
        Space per hash: K bits = K/8 bytes
        """
        hashes = []
        for projection in self.projections:
            # Project to K dimensions
            proj = torch.mm(vector, projection)
            # Binary hash (sign of projection)
            hash_code = (proj > 0).int()
            hashes.append(hash_code)
        return hashes

    def search(self, query, top_k=10):
        """
        1. Hash query: O(L·K·D)
        2. Lookup candidates: O(L·C) where C = candidates per table
        3. Rank candidates: O(C·D)

        Total: O(L·K·D + L·C + C·D)
        """
        # Get candidate buckets
        query_hashes = self.hash_vector(query)
        candidates = self.get_candidates(query_hashes)

        # Rank candidates (brute force on subset)
        return self.rank_candidates(query, candidates, top_k)
```

**Complexity vs. Accuracy Trade-off:**
```python
# More tables = higher recall but slower
L=4:  80% recall, 10ms search
L=8:  90% recall, 20ms search
L=16: 95% recall, 40ms search

# Longer hashes = fewer false positives
K=8:  Many collisions, 85% precision
K=12: Moderate collisions, 92% precision
K=16: Few collisions, 97% precision
```

#### Product Quantization (PQ)
**Use when:**
- Database size: 100M - 10B vectors
- Memory extremely constrained (8x compression)
- Can tolerate 85-95% recall
- Read-heavy workload

**Compression Mechanism:**
```python
class ProductQuantizer:
    def __init__(self, dim, num_subspaces=8, num_centroids=256):
        """
        Compress D-dimensional vector to M bytes

        dim: embedding dimension (e.g., 768)
        num_subspaces (M): split into M subspaces (e.g., 8)
        num_centroids: codebook size per subspace (typically 256 = 1 byte)

        Compression ratio: (D×4 bytes) / M bytes = 4D/M
        Example: 768×4 / 8 = 384x compression!
        """
        self.M = num_subspaces
        self.K = num_centroids
        self.subspace_dim = dim // num_subspaces

        # Learned codebooks (centroids) per subspace
        self.codebooks = [
            torch.randn(num_centroids, self.subspace_dim)
            for _ in range(num_subspaces)
        ]

    def encode(self, vector):
        """
        Time: O(M·K·D/M) = O(K·D)
        Output: M bytes (one byte per subspace)
        """
        codes = []
        for i, codebook in enumerate(self.codebooks):
            # Extract subspace
            start = i * self.subspace_dim
            end = start + self.subspace_dim
            subvector = vector[start:end]

            # Find nearest centroid
            distances = torch.cdist(subvector.unsqueeze(0), codebook)
            code = distances.argmin(dim=1).item()  # 0-255
            codes.append(code)

        return codes  # M bytes total

    def search(self, query, compressed_db, top_k=10):
        """
        Asymmetric Distance Computation (ADC)
        Time: O(N·M·K) = O(N·2048) for M=8, K=256
        Compare to brute force: O(N·D) = O(N·768)

        3x faster due to lookup table optimization!
        """
        # Pre-compute distance table: query subspace to all centroids
        distance_table = []  # [M, K] table
        for i, codebook in enumerate(self.codebooks):
            start = i * self.subspace_dim
            end = start + self.subspace_dim
            query_sub = query[start:end]

            # Distances from query subspace to all 256 centroids
            dists = torch.cdist(query_sub.unsqueeze(0), codebook)  # [1, 256]
            distance_table.append(dists)

        # Compute approximate distances to all N vectors
        N = len(compressed_db)
        approx_distances = torch.zeros(N)

        for n in range(N):
            codes = compressed_db[n]  # M bytes
            # Sum distances using lookup table
            total_dist = sum(
                distance_table[m][0, codes[m]]
                for m in range(self.M)
            )
            approx_distances[n] = total_dist

        # Top-k selection
        scores, indices = torch.topk(approx_distances, k=top_k, largest=False)
        return scores, indices
```

**PQ Complexity Analysis:**
```
Encoding:     O(K·D) per vector
Search:       O(N·M·K) ≈ O(N·2048) vs O(N·D) ≈ O(N·768)
              BUT: Lookup table makes this 3-5x faster in practice
Memory:       N·M bytes vs N·D·4 bytes
              Example: 100M vectors × 8 bytes = 800 MB
                       vs 100M × 768 × 4 = 307 GB
              Compression ratio: ~384x
```

### Hybrid Strategy

Combine algorithms for optimal performance:

```python
class HybridIndex:
    """
    Multi-stage search pipeline:
    1. LSH for coarse filtering (reduce N to N/100)
    2. PQ for rapid ranking (reduce N/100 to N/1000)
    3. Brute force on GPU for exact top-k (final N/1000)
    """
    def __init__(self, vectors, embedding_dim):
        # Stage 1: LSH for 10x reduction
        self.lsh = LSHIndex(num_tables=8, hash_length=12)

        # Stage 2: PQ for 10x reduction
        self.pq = ProductQuantizer(dim=embedding_dim, num_subspaces=8)

        # Stage 3: Full precision vectors for final ranking
        self.exact_vectors = vectors.to('cuda')

    def search(self, query, top_k=10):
        # Stage 1: LSH → ~10K candidates from 100M
        lsh_candidates = self.lsh.search(query, top_k=10000)  # ~1ms

        # Stage 2: PQ on candidates → ~1K candidates
        pq_candidates = self.pq.search(
            query,
            self.pq_compressed[lsh_candidates],
            top_k=1000
        )  # ~5ms

        # Stage 3: GPU brute force on 1K vectors
        candidate_vectors = self.exact_vectors[pq_candidates]
        with autocast(dtype=torch.float16):
            scores = torch.mm(query, candidate_vectors.T)
        final_scores, final_indices = torch.topk(scores, k=top_k)  # ~1ms

        # Map back to original indices
        return final_scores, pq_candidates[final_indices]

# Total time: ~7ms for 100M vectors vs 2000ms brute force
# Recall: ~95% (vs 100% brute force)
```

### Algorithm Selection Decision Tree

```
Database Size?
├─ < 1M vectors
│  └─ Use: GPU Brute Force (100% recall, 20-50ms)
│
├─ 1M - 10M vectors
│  ├─ Need 100% recall?
│  │  └─ Use: GPU Brute Force (50-500ms)
│  └─ 95%+ recall OK?
│     └─ Use: HNSW (5-20ms, 2x memory)
│
├─ 10M - 100M vectors
│  ├─ Memory available (3x)?
│  │  └─ Use: HNSW (10-50ms, 95-99% recall)
│  ├─ Memory constrained?
│  │  └─ Use: LSH → PQ → GPU (10-30ms, 90-95% recall)
│  └─ Updates frequent?
│     └─ Use: LSH (20-40ms, 85-90% recall)
│
└─ > 100M vectors
   ├─ Ultra low latency (<10ms)?
   │  └─ Use: PQ + GPU reranking (8x compression, 85-90% recall)
   ├─ High recall (95%+)?
   │  └─ Use: HNSW + PQ hybrid (50-100ms)
   └─ Extreme scale (1B+)?
      └─ Use: Sharded LSH → PQ → GPU (distributed)
```

---

## Benchmarking Guide

### GPU Performance Metrics

```python
import torch
import time

class GPUBenchmark:
    def __init__(self, device='cuda'):
        self.device = device

    def benchmark_operation(self, func, *args, num_runs=100, warmup=10):
        """
        Accurate GPU benchmarking with warmup and synchronization
        """
        # Warmup runs
        for _ in range(warmup):
            func(*args)
        torch.cuda.synchronize()

        # Timed runs
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(num_runs):
            start_event.record()
            result = func(*args)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        return {
            'mean_ms': sum(times) / len(times),
            'median_ms': sorted(times)[len(times) // 2],
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }

    def benchmark_tensor_cores(self, batch_size=128, dim=768, db_size=10000):
        """
        Compare FP32 vs FP16 (Tensor Cores)
        """
        queries = torch.randn(batch_size, dim, device=self.device)
        database = torch.randn(db_size, dim, device=self.device)

        # FP32 baseline
        def fp32_search():
            return torch.mm(queries, database.T)

        # FP16 with Tensor Cores
        def fp16_search():
            with autocast(dtype=torch.float16):
                return torch.mm(queries, database.T)

        fp32_time = self.benchmark_operation(fp32_search)
        fp16_time = self.benchmark_operation(fp16_search)

        speedup = fp32_time['mean_ms'] / fp16_time['mean_ms']

        print(f"FP32: {fp32_time['mean_ms']:.2f}ms")
        print(f"FP16: {fp16_time['mean_ms']:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")

        return {'fp32': fp32_time, 'fp16': fp16_time, 'speedup': speedup}

    def benchmark_memory_bandwidth(self):
        """
        Measure effective memory bandwidth
        """
        size_gb = 2.0
        num_elements = int(size_gb * 1024**3 / 4)  # FP32 elements

        src = torch.randn(num_elements, device=self.device)
        dst = torch.empty_like(src)

        def copy_operation():
            dst.copy_(src)

        result = self.benchmark_operation(copy_operation, num_runs=20)

        bandwidth_gb_s = (size_gb / result['mean_ms']) * 1000

        print(f"Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")
        print(f"Theoretical max (A100): 2039 GB/s")
        print(f"Efficiency: {bandwidth_gb_s / 2039 * 100:.1f}%")

        return bandwidth_gb_s
```

### End-to-End Search Benchmark

```python
def benchmark_search_pipeline(index, queries, top_k=10):
    """
    Benchmark complete search pipeline with metrics
    """
    metrics = {
        'latency_ms': [],
        'throughput_qps': 0,
        'recall': [],
        'memory_gb': 0
    }

    # Memory usage
    torch.cuda.reset_peak_memory_stats()

    # Latency measurement
    start = time.time()
    for i, query in enumerate(queries):
        query_start = time.time()
        results = index.search(query, top_k=top_k)
        query_time = (time.time() - query_start) * 1000
        metrics['latency_ms'].append(query_time)

    total_time = time.time() - start
    metrics['throughput_qps'] = len(queries) / total_time
    metrics['memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3

    # Latency percentiles
    latencies = sorted(metrics['latency_ms'])
    print(f"Latency p50: {latencies[len(latencies)//2]:.2f}ms")
    print(f"Latency p95: {latencies[int(len(latencies)*0.95)]:.2f}ms")
    print(f"Latency p99: {latencies[int(len(latencies)*0.99)]:.2f}ms")
    print(f"Throughput: {metrics['throughput_qps']:.2f} QPS")
    print(f"Peak memory: {metrics['memory_gb']:.2f} GB")

    return metrics
```

### Recall Measurement

```python
def measure_recall(approximate_results, exact_results, k=10):
    """
    Recall@k: percentage of true top-k found by approximation
    """
    recalls = []
    for approx, exact in zip(approximate_results, exact_results):
        approx_set = set(approx[:k])
        exact_set = set(exact[:k])
        recall = len(approx_set & exact_set) / k
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls)
    print(f"Recall@{k}: {avg_recall*100:.2f}%")
    return avg_recall
```

---

## Troubleshooting

### Common Issues

#### 1. Tensor Cores Not Activating

**Symptoms:**
- FP16 performance same as FP32
- No speedup from AMP

**Diagnosis:**
```python
# Check GPU capability
print(torch.cuda.get_device_capability())  # Should be >= (7, 0) for Tensor Cores

# Profile CUDA kernels
torch.cuda.profiler.start()
result = torch.mm(a, b)
torch.cuda.profiler.stop()
# Check for "s884gemm" or "s16816gemm" kernels
```

**Solutions:**
- Ensure dimensions divisible by 8
- Use `torch.float16` or `torch.bfloat16`
- Call `.contiguous()` on tensors
- Update PyTorch to 2.0+

#### 2. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
batch_size = 64  # Instead of 256

# Process database in chunks
for chunk in database.split(chunk_size):
    results = search(query, chunk)

# Use gradient checkpointing (training)
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

#### 3. Slow Host-to-Device Transfer

**Symptoms:**
- High latency before GPU computation
- CPU bottleneck

**Solutions:**
```python
# Use pinned memory
tensor = tensor.pin_memory().cuda(non_blocking=True)

# Pre-load on GPU
database = database.cuda()  # Keep on GPU between queries

# Use asynchronous transfers
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    tensor.cuda(non_blocking=True)
```

#### 4. Poor Recall in Approximate Search

**Symptoms:**
- Results don't match expectations
- Low recall scores

**Solutions:**
```python
# HNSW: Increase search quality
ef_search = 200  # Default 50, higher = better recall

# LSH: Increase number of tables
num_tables = 16  # Default 8

# PQ: Increase subspace centroids
num_centroids = 512  # Default 256

# Hybrid: Add reranking stage
top_k_candidates = 10 * final_top_k  # Get more candidates, rerank
```

#### 5. Slow Index Construction

**Symptoms:**
- Hours to build HNSW index
- LSH initialization slow

**Solutions:**
```python
# HNSW: Reduce construction quality
ef_construction = 100  # Default 200

# Parallelize construction
index.add_items(vectors, num_threads=16)

# LSH: Use fewer hash functions
hash_length = 8  # Default 12

# PQ: Train on sample
sample_size = 100000  # Train codebooks on subset
pq.train(vectors[:sample_size])
```

### Performance Debugging Checklist

✅ **GPU Utilization**: `nvidia-smi dmon` should show >80%
✅ **Memory Usage**: Not at capacity (leave 10% headroom)
✅ **Tensor Cores**: Use CUDA profiler to verify
✅ **Batch Size**: Test different sizes for throughput
✅ **Data Layout**: All tensors `.contiguous()`
✅ **Precision**: FP16 for computation, FP32 for accuracy-critical
✅ **Synchronization**: Minimize `torch.cuda.synchronize()` calls
✅ **Streams**: Use multiple streams for overlap

### Profiling Tools

```bash
# NVIDIA Nsight Systems (timeline profiling)
nsys profile -o profile.qdrep python script.py
nsys-ui profile.qdrep

# NVIDIA Nsight Compute (kernel profiling)
ncu -o profile python script.py
ncu-ui profile.ncu-rep

# PyTorch profiler
python -m torch.utils.bottleneck script.py
```

---

## Optimization Summary

| Phase | Key Technique | Expected Gain | Complexity |
|-------|---------------|---------------|------------|
| 1 | Tensor Cores (FP16) | 5-10x | Low |
| 2 | Memory Management | 2-4x | Medium |
| 3 | Algorithm Choice | 10-100x | High |

**Recommended Path:**
1. Start with Phase 1 (Tensor Cores) - easy wins
2. Profile and optimize memory (Phase 2) if needed
3. Switch to approximate algorithms (Phase 3) for scale

**Final Performance Target:**
- **1M vectors**: <50ms, 100% recall (GPU brute force)
- **10M vectors**: <20ms, 95%+ recall (HNSW)
- **100M vectors**: <10ms, 90%+ recall (Hybrid LSH+PQ+GPU)
