# Phase 3: Algorithmic Optimizations - Breaking O(N²) Complexity

## Overview

Phase 3 implements advanced algorithmic techniques to break the O(N²) complexity of all-pairs similarity comparison, achieving O(log N) search complexity through hierarchical navigation and locality-sensitive hashing.

## Complexity Analysis

### Exact k-NN Search
- **Complexity**: O(N·D) per query
  - N: number of items in database
  - D: embedding dimensionality
  - For 10K items, 1024 dims: ~10M operations per query
  - For 100K items: ~100M operations per query

### HNSW (Hierarchical Navigable Small World)
- **Build complexity**: O(N·log N·D)
- **Search complexity**: O(log N·D) per query
- **Expected speedup**: 100-1000x for large datasets
- **Recall**: 90-99% with proper tuning

### LSH (Locality Sensitive Hashing)
- **Build complexity**: O(N·L·K·D)
  - L: number of hash tables
  - K: number of hash functions per table
- **Search complexity**: O(L·B + C·D)
  - B: average bucket size
  - C: number of candidates
- **Expected speedup**: 10-100x
- **Recall**: 70-95% (tunable)

### Product Quantization
- **Memory reduction**: 8-64x (1 byte per 8-64 dims)
- **Search complexity**: O(N·M·K)
  - M: number of subspaces
  - K: number of centroids (typically 256)
- **Distance computation speedup**: 8-16x

## Implementation Components

### 1. HNSW GPU (`hnsw_gpu.cuh`)

Hierarchical graph structure for approximate nearest neighbor search:

**Key Features:**
- Multi-layer graph with exponentially decreasing connectivity
- Warp-level parallelism for neighbor exploration
- Shared memory for visited bitmap tracking
- Tensor core acceleration for distance computation

**Data Structure:**
```cuda
struct HNSW_GPU {
    HNSWLayer* layers;       // [num_layers]
    int entry_point;         // Top-level entry
    int M;                   // Base connectivity
};

struct HNSWLayer {
    int* neighbors;          // [num_nodes * M]
    float* distances;        // Precomputed
    int num_nodes;
};
```

**Search Algorithm:**
1. Start from top layer (coarse approximation)
2. Greedy search to find local minimum
3. Descend to next layer
4. Repeat until bottom layer
5. Collect k nearest neighbors

**GPU Optimizations:**
- Each block processes one query
- Threads explore neighbors in parallel
- Atomic operations for visited tracking
- Shared memory for candidate management

### 2. LSH GPU (`lsh_gpu.cu`)

Locality-sensitive hashing for fast candidate generation:

**Hash Function:**
- SimHash: random projection + sign bit extraction
- Maps similar items to same bucket with high probability

**Multi-table Strategy:**
- Use L independent hash tables
- Union candidates from all tables
- Trade-off: more tables = higher recall, more computation

**GPU Implementation:**
```cuda
__device__ uint32_t compute_hash(embedding, table_id) {
    hash = 0;
    for each projection:
        if dot(embedding, random_vector) > 0:
            set bit in hash
    return hash % num_buckets;
}
```

**Advanced Features:**
- Multi-probe LSH: query neighboring buckets
- Adaptive LSH: adjust tables based on query difficulty
- Deduplication using shared memory bitmaps

### 3. Product Quantization (`product_quantization.cu`)

Compress embeddings for memory-efficient storage:

**Approach:**
1. Split embedding into M subspaces
2. Quantize each subspace independently using k-means
3. Store as M bytes (one per subspace)
4. Distance computation via lookup tables

**Asymmetric Distance Computation (ADT):**
- Precompute query distances to all centroids
- Distance to database item = sum of table lookups
- 8-16x faster than full distance computation

**Memory Usage:**
- Original: N·D·2 bytes (fp16)
- PQ: N·M bytes (M=64 typical)
- Reduction: 32x for 1024-dim embeddings

### 4. Hybrid Index (`hybrid_index.cu`)

Combines multiple techniques for optimal performance:

**Search Modes:**

1. **EXACT**: Pure HNSW
   - Use for: small k (<10), high recall (>95%)

2. **LSH_HNSW**: LSH candidates + HNSW refinement
   - Use for: medium k (10-100), medium-high recall (85-95%)
   - Pipeline: LSH → 1000 candidates → HNSW refine → top k

3. **LSH_PQ**: LSH candidates + PQ distances
   - Use for: large k (>100), medium recall (70-85%)
   - Most memory efficient

4. **ADAPTIVE**: Per-query mode selection
   - Analyzes query characteristics
   - Selects optimal strategy dynamically

**Implementation:**
```cuda
__global__ void hybrid_search_lsh_hnsw(...) {
    // Stage 1: LSH candidate generation (fast)
    candidates = lsh_query(query);  // ~1000 items

    // Stage 2: HNSW refinement (accurate)
    for each candidate:
        distance = compute_exact_distance(query, candidate);

    // Stage 3: Top-k selection
    sort and select k nearest;
}
```

## Performance Characteristics

### Expected Speedup vs Exact Search

| Dataset Size | HNSW | LSH+HNSW | LSH+PQ |
|--------------|------|----------|--------|
| 10K items    | 10x  | 8x       | 15x    |
| 100K items   | 100x | 80x      | 120x   |
| 1M items     | 1000x| 800x     | 1200x  |

### Recall vs Speed Trade-off

| Method | Recall | Relative Speed |
|--------|--------|----------------|
| Exact  | 100%   | 1x             |
| HNSW (ef=100) | 95% | 100x      |
| HNSW (ef=50)  | 90% | 200x      |
| LSH+HNSW | 92% | 150x           |
| LSH+PQ   | 85% | 250x           |

### Memory Usage

| Method | Memory per Item | Total for 100K |
|--------|-----------------|----------------|
| Exact (fp16) | 2KB | 200MB      |
| HNSW         | 2.5KB | 250MB    |
| LSH          | 2.1KB | 210MB    |
| PQ           | 128B  | 12.8MB   |
| Hybrid       | 2.2KB | 220MB    |

## Configuration Guidelines

### HNSW Parameters

```
M = 16-32           // Higher M = better recall, more memory
ef_construction = 200  // Higher = better graph quality
ef_search = 50-200     // Higher = better recall, slower search
```

### LSH Parameters

```
num_tables = 4-16      // More tables = higher recall
num_projections = 12-24 // More bits = finer granularity
num_buckets = sqrt(N)   // Balance collision rate
```

### PQ Parameters

```
num_subspaces = 32-128  // Higher = more precision
                        // Must divide embedding_dim
subspace_dim = 8-32     // Higher = larger codebook
```

## Benchmarking

Use `benchmark_algorithms.cu` to validate performance:

```bash
nvcc -o benchmark benchmark_algorithms.cu -arch=sm_80
./benchmark
```

**Output:**
- Build time for each method
- Search time and throughput (QPS)
- Recall vs ground truth
- Memory usage
- Speedup vs exact search

## Integration with Phase 2

Combines with Phase 2 tensor core optimizations:
- HNSW uses tensor cores for distance computation
- LSH hash computation vectorized
- PQ lookup tables cached in shared memory
- All methods use fp16 for memory bandwidth

## Scaling to Large Datasets

**10M items:**
- Exact search: infeasible (hours per query)
- HNSW: <1ms per query with ef=100
- Hybrid: <0.5ms per query with 90% recall

**100M items:**
- Shard across multiple GPUs
- Use LSH for coarse filtering
- HNSW within shards
- Achieve <5ms per query

## Future Optimizations

1. **GPU-aware HNSW construction**: Build index on GPU
2. **Dynamic index updates**: Insert/delete without rebuild
3. **Learned indices**: Use neural networks for routing
4. **Distributed search**: Multi-GPU coordination
5. **Quantized HNSW**: Combine with PQ for memory efficiency

## References

- HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- LSH: Indyk & Motwani, "Approximate nearest neighbors: towards removing the curse of dimensionality"
- Product Quantization: Jégou et al., "Product quantization for nearest neighbor search"
