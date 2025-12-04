# CUDA Optimization Roadmap: T4 GPU Performance Enhancement

## Overview

Systematic plan to optimize 4 CUDA kernels from 15-25% to 80-90% of T4 GPU theoretical performance.

**Target Improvement**: 50-100x overall speedup
**Timeline**: 6-8 weeks for complete implementation
**Risk Level**: Medium (incremental improvements with validation)

---

## Phase 1: Quick Wins (Week 1-2)
**Expected Speedup**: 8-12x
**Risk**: Low
**Effort**: Low-Medium

### 1.1: Fix Memory Coalescing (Priority: CRITICAL)

**Target**: semantic_similarity_fp16.cu, semantic_similarity.cu
**Current**: 60-80 GB/s memory bandwidth
**Goal**: 240-280 GB/s (75-85% of peak)

#### Implementation

```cuda
// BEFORE: Scattered access via random pairs
for (int idx = tid; idx < num_pairs; idx += stride) {
    int src = item_pairs_src[idx];  // Non-sequential
    int tgt = item_pairs_tgt[idx];
    float sim = compute_similarity(&embeddings[src * dim], &embeddings[tgt * dim], dim);
}

// AFTER: Tile-based processing with coalesced access
__global__ void tiled_similarity_kernel(
    const __half* __restrict__ embeddings,  // [num_items][embedding_dim]
    float* __restrict__ similarity_matrix,  // [num_items][num_items]
    const int num_items,
    const int embedding_dim
) {
    // Each block computes a 16x16 tile of similarity matrix
    const int TILE_SIZE = 16;
    int tile_row = blockIdx.y * TILE_SIZE;
    int tile_col = blockIdx.x * TILE_SIZE;
    int thread_row = tile_row + threadIdx.y;
    int thread_col = tile_col + threadIdx.x;

    // Shared memory for embeddings (coalesced loads)
    __shared__ __half tile_a[TILE_SIZE][768];  // Max embedding dim
    __shared__ __half tile_b[TILE_SIZE][768];

    // Coalesced load into shared memory
    for (int d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
        if (thread_row < num_items) {
            tile_a[threadIdx.y][d] = embeddings[thread_row * embedding_dim + d];
        }
        if (thread_col < num_items) {
            tile_b[threadIdx.x][d] = embeddings[thread_col * embedding_dim + d];
        }
    }
    __syncthreads();

    // Compute similarity from shared memory
    if (thread_row < num_items && thread_col < num_items) {
        float sim = compute_similarity_shared(tile_a[threadIdx.y], tile_b[threadIdx.x], embedding_dim);
        similarity_matrix[thread_row * num_items + thread_col] = sim;
    }
}

// Launch with 2D grid
dim3 block(16, 16);
dim3 grid((num_items + 15) / 16, (num_items + 15) / 16);
tiled_similarity_kernel<<<grid, block>>>(embeddings, similarity_matrix, num_items, dim);
```

**Expected Impact**: 3-4x speedup
**Validation**: Measure global memory transactions per warp (should drop from 32 to 1-2)

---

### 1.2: Enable Tensor Core Batch Processing (Priority: CRITICAL)

**Target**: semantic_similarity_fp16.cu
**Current**: 10-15 TFLOPS (scalar operations)
**Goal**: 50-60 TFLOPS (tensor core acceleration)

#### Implementation

```cuda
// Compute dot products for similarity matrix using tensor cores
__global__ void tensor_core_similarity_kernel(
    const __half* __restrict__ embeddings,  // [num_items][embedding_dim] row-major
    float* __restrict__ dot_products,       // [num_items][num_items] output
    const int num_items,
    const int embedding_dim
) {
    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // Each block computes a 16x16 output tile
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);

    // Iterate over K dimension in 16-element chunks
    for (int k = 0; k < embedding_dim; k += 16) {
        // Load 16x16 tiles from A and B
        int a_idx = warp_row * embedding_dim + k;
        int b_idx = warp_col * embedding_dim + k;

        wmma::load_matrix_sync(a_frag, embeddings + a_idx, embedding_dim);
        wmma::load_matrix_sync(b_frag, embeddings + b_idx, embedding_dim);

        // Perform tensor core multiply-accumulate
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store 16x16 result tile
    int out_idx = warp_row * num_items + warp_col;
    wmma::store_matrix_sync(dot_products + out_idx, acc_frag, num_items, wmma::mem_row_major);
}

// Separate kernel to normalize dot products into cosine similarities
__global__ void normalize_to_cosine_kernel(
    const float* __restrict__ dot_products,
    const __half* __restrict__ embeddings,
    float* __restrict__ cosine_similarities,
    const int num_items,
    const int embedding_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= num_items || col >= num_items) return;

    // Compute norms (could be precomputed and cached)
    float norm_a = 0.0f, norm_b = 0.0f;
    for (int d = 0; d < embedding_dim; d++) {
        float val_a = __half2float(embeddings[row * embedding_dim + d]);
        float val_b = __half2float(embeddings[col * embedding_dim + d]);
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    float dot = dot_products[row * num_items + col];
    cosine_similarities[row * num_items + col] = dot / (norm_a * norm_b + 1e-6f);
}

// Launch configuration
dim3 tc_block(128, 2);  // 256 threads = 8 warps
dim3 tc_grid((num_items + 15) / 16, (num_items + 15) / 16);
tensor_core_similarity_kernel<<<tc_grid, tc_block>>>(embeddings, dots, num_items, dim);

dim3 norm_block(16, 16);
dim3 norm_grid((num_items + 15) / 16, (num_items + 15) / 16);
normalize_to_cosine_kernel<<<norm_grid, norm_block>>>(dots, embeddings, similarities, num_items, dim);
```

**Expected Impact**: 5-8x speedup for similarity computation
**Validation**: Check TFLOPS via nsight compute (should see 50-60 TFLOPS)

---

### 1.3: Add Shared Memory Caching (Priority: HIGH)

**Target**: All kernels
**Current**: 4KB shared memory usage (8% of available)
**Goal**: 32-48KB usage (70-100% of available)

#### Implementation Template

```cuda
__global__ void cached_computation_kernel(...) {
    const int CACHE_SIZE = 32;  // Number of embeddings to cache
    const int EMB_DIM = 768;

    // Allocate shared memory for embedding cache
    __shared__ __half emb_cache[CACHE_SIZE][EMB_DIM];
    __shared__ int cache_ids[CACHE_SIZE];

    // Cooperative loading: all threads in block contribute
    int cache_start = blockIdx.x * CACHE_SIZE;
    for (int c = 0; c < CACHE_SIZE; c++) {
        int item_id = cache_start + c;
        if (item_id < num_items) {
            cache_ids[c] = item_id;
            // Coalesced load: each thread loads one dimension
            for (int d = threadIdx.x; d < EMB_DIM; d += blockDim.x) {
                emb_cache[c][d] = embeddings[item_id * EMB_DIM + d];
            }
        }
    }
    __syncthreads();

    // Now use cached embeddings for computation
    // 10-100x faster than repeated global memory access
}
```

**Expected Impact**: 2-3x speedup
**Validation**: Measure global memory load transactions (should drop 70-80%)

---

### 1.4: Optimize Warp Reductions (Priority: MEDIUM)

**Target**: semantic_similarity_fp16.cu (lines 94-98)
**Current**: 3 separate shuffle operations per reduction
**Goal**: Single shuffle with packed data

#### Implementation

```cuda
// BEFORE: Three separate reductions
for (int offset = 16; offset > 0; offset /= 2) {
    dot += __shfl_down_sync(0xffffffff, dot, offset);
    norm_a += __shfl_down_sync(0xffffffff, norm_a, offset);
    norm_b += __shfl_down_sync(0xffffffff, norm_b, offset);
}

// AFTER: Single packed reduction
struct DotNorms {
    float dot;
    float norm_a;
    float norm_b;
};

__device__ __forceinline__ DotNorms warp_reduce_packed(DotNorms val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val.dot += __shfl_down_sync(0xffffffff, val.dot, offset);
        val.norm_a += __shfl_down_sync(0xffffffff, val.norm_a, offset);
        val.norm_b += __shfl_down_sync(0xffffffff, val.norm_b, offset);
    }
    return val;
}

// Or using float4 for even better packing
__device__ __forceinline__ float4 warp_reduce_float4(float4 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
        val.z += __shfl_down_sync(0xffffffff, val.z, offset);
        val.w += __shfl_down_sync(0xffffffff, val.w, offset);
    }
    return val;
}
```

**Expected Impact**: 1.3-1.5x speedup
**Validation**: Instruction count per warp (should drop 30%)

---

## Phase 2: Algorithmic Improvements (Week 3-4)
**Expected Speedup**: 5-10x (additional)
**Risk**: Medium
**Effort**: Medium-High

### 2.1: Replace Linear Search with Hash Tables (Priority: CRITICAL)

**Target**: ontology_reasoning.cu (all constraint kernels)
**Current**: O(N) linear search, 95% of kernel time
**Goal**: O(1) hash lookup, <5% overhead

#### Implementation

```cuda
// Preprocessing: Build hash table on CPU or GPU
struct NodeIndex {
    uint32_t node_id;
    uint32_t graph_id;
    uint32_t array_index;
};

// Use thrust for GPU-side sorting and searching
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

// Build sorted index
void build_node_index(
    MediaOntologyNode* nodes,
    int num_nodes,
    NodeIndex** d_index
) {
    thrust::device_vector<NodeIndex> index(num_nodes);

    // Populate index
    thrust::counting_iterator<int> iter(0);
    thrust::transform(iter, iter + num_nodes,
        index.begin(),
        [nodes] __device__ (int i) {
            NodeIndex idx;
            idx.node_id = nodes[i].node_id;
            idx.graph_id = nodes[i].graph_id;
            idx.array_index = i;
            return idx;
        });

    // Sort by (graph_id, node_id)
    thrust::sort(index.begin(), index.end(),
        [] __device__ (const NodeIndex& a, const NodeIndex& b) {
            if (a.graph_id != b.graph_id) return a.graph_id < b.graph_id;
            return a.node_id < b.node_id;
        });

    *d_index = thrust::raw_pointer_cast(index.data());
}

// Lookup function (O(log N) binary search)
__device__ int lookup_node_index(
    const NodeIndex* index,
    int num_nodes,
    uint32_t node_id,
    uint32_t graph_id
) {
    NodeIndex key{node_id, graph_id, 0};

    // Binary search
    int left = 0, right = num_nodes - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        const NodeIndex& midval = index[mid];

        if (midval.graph_id < key.graph_id ||
            (midval.graph_id == key.graph_id && midval.node_id < key.node_id)) {
            left = mid + 1;
        } else if (midval.graph_id > key.graph_id ||
                   (midval.graph_id == key.graph_id && midval.node_id > key.node_id)) {
            right = mid - 1;
        } else {
            return midval.array_index;  // Found
        }
    }
    return -1;  // Not found
}

// Updated kernel using index
__global__ void apply_disjoint_genres_kernel_indexed(
    MediaOntologyNode* nodes,
    int num_nodes,
    const NodeIndex* node_index,  // Sorted index
    MediaOntologyConstraint* constraints,
    int num_constraints,
    // ...
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_constraints) return;

    MediaOntologyConstraint constraint = constraints[idx];

    // FAST: O(log N) lookup instead of O(N) scan
    int source_idx = lookup_node_index(node_index, num_nodes,
                                       constraint.source_id, constraint.graph_id);
    int target_idx = lookup_node_index(node_index, num_nodes,
                                       constraint.target_id, constraint.graph_id);

    if (source_idx < 0 || target_idx < 0) return;

    // Rest of computation unchanged...
}
```

**Expected Impact**: 10-15x speedup for ontology kernels
**Validation**: Kernel runtime should drop from 15ms to <2ms per iteration

---

### 2.2: Reduce Warp Divergence with Constraint Sorting (Priority: HIGH)

**Target**: ontology_reasoning.cu
**Current**: 30-40% warp efficiency
**Goal**: 70-90% warp efficiency

#### Implementation

```cuda
// Sort constraints by type before kernel launch
void sort_constraints_by_type(
    MediaOntologyConstraint* constraints,
    int num_constraints
) {
    thrust::device_ptr<MediaOntologyConstraint> ptr(constraints);
    thrust::sort(ptr, ptr + num_constraints,
        [] __device__ (const MediaOntologyConstraint& a,
                       const MediaOntologyConstraint& b) {
            return a.type < b.type;
        });
}

// Launch separate kernels per constraint type
void process_constraints_batched(
    MediaOntologyNode* nodes,
    int num_nodes,
    MediaOntologyConstraint* constraints,
    int num_constraints
) {
    // Find boundaries for each constraint type
    int type_starts[11] = {0};  // 10 constraint types
    int type_counts[11] = {0};

    // Count constraints per type (parallel reduction)
    thrust::device_vector<int> types(num_constraints);
    thrust::transform(constraints, constraints + num_constraints,
                     types.begin(),
                     [] __device__ (const MediaOntologyConstraint& c) {
                         return c.type;
                     });
    // ... compute starts/counts ...

    // Launch type-specific kernels (no divergence!)
    if (type_counts[CONSTRAINT_DISJOINT_GENRES] > 0) {
        apply_disjoint_genres_kernel<<<grid, block>>>(
            nodes, num_nodes,
            constraints + type_starts[CONSTRAINT_DISJOINT_GENRES],
            type_counts[CONSTRAINT_DISJOINT_GENRES],
            // ...
        );
    }
    // Repeat for each type...
}
```

**Expected Impact**: 2-3x speedup
**Validation**: Warp efficiency metric should improve to 70-90%

---

### 2.3: Atomic Operation Reduction (Priority: HIGH)

**Target**: ontology_reasoning.cu, graph_search.cu
**Current**: Direct atomics causing serialization
**Goal**: Warp-level aggregation before atomics

#### Implementation

```cuda
// BEFORE: Every thread does atomic
atomic_add_float3(&nodes[idx].velocity, force * delta_time);

// AFTER: Warp-level aggregation
__device__ __forceinline__ void warp_aggregate_force(
    MediaOntologyNode* nodes,
    int node_idx,
    float3 force,
    float delta_time
) {
    // Step 1: All threads in warp vote on same target node
    unsigned int active_mask = __activemask();
    unsigned int target_mask = __match_any_sync(active_mask, node_idx);

    // Step 2: Leader thread accumulates forces from matching threads
    int leader = __ffs(target_mask) - 1;  // First set bit

    // Step 3: Warp-level reduction for matching threads
    float3 aggregate = force;
    for (int i = 0; i < 32; i++) {
        if ((target_mask & (1 << i)) && i != threadIdx.x) {
            // Shuffle force components from other threads
            aggregate.x += __shfl_sync(target_mask, force.x, i);
            aggregate.y += __shfl_sync(target_mask, force.y, i);
            aggregate.z += __shfl_sync(target_mask, force.z, i);
        }
    }

    // Step 4: Only leader does atomic
    if (threadIdx.x % 32 == leader) {
        atomic_add_float3(&nodes[node_idx].velocity, aggregate * delta_time);
    }
}
```

**Expected Impact**: 3-5x speedup on hub nodes
**Validation**: Atomic operation count should drop 70-90%

---

### 2.4: Precompute Similarity Matrices (Priority: MEDIUM)

**Target**: semantic_similarity.cu
**Current**: Recompute similarities in each kernel
**Goal**: Compute once, reuse across kernels

#### Implementation

```cuda
// Phase 1: Compute all similarities once
__global__ void batch_compute_all_similarities(
    const float* embeddings,
    float* similarity_matrix,  // [num_items][num_items]
    int num_items,
    int embedding_dim
) {
    // Use tiled approach with tensor cores (from 1.2)
    // Store results in global similarity matrix
}

// Phase 2: Kernels read from precomputed matrix
__global__ void apply_genre_cluster_force_cached(
    const float* precomputed_similarities,  // NEW: read-only
    const int* item_genres,
    float3* positions,
    float3* forces,
    int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    for (int i = 0; i < num_items; i++) {
        if (i == idx) continue;

        // Read precomputed similarity (fast!)
        float similarity = precomputed_similarities[idx * num_items + i];

        if (similarity < threshold) continue;

        // Use similarity to compute force (no recomputation)
        // ...
    }
}
```

**Trade-off Analysis**:
- **Memory**: 10K items = 400MB (fits easily)
- **Compute**: One-time cost << multiple recomputations
- **Update**: Recompute when embeddings change (daily for media system)

**Expected Impact**: 3-5x speedup for multi-kernel workflows
**Validation**: Kernel compute time should drop 70-80%

---

## Phase 3: Advanced Optimizations (Week 5-6)
**Expected Speedup**: 2-5x (additional)
**Risk**: High
**Effort**: High

### 3.1: Hierarchical Spatial Partitioning (Priority: HIGH)

**Target**: semantic_similarity.cu (break O(N²) complexity)
**Current**: 10K items = 100M comparisons
**Goal**: ~10-20M comparisons (10x reduction)

#### Implementation Approach

```cuda
// Build spatial grid on GPU
struct SpatialGrid {
    int3 grid_dims;  // e.g., 32x32x32
    float3 cell_size;
    int* cell_starts;  // [grid_dims.x * grid_dims.y * grid_dims.z]
    int* cell_counts;  // [grid_dims.x * grid_dims.y * grid_dims.z]
    int* sorted_items; // [num_items]
};

// Build grid (one-time or when positions update)
void build_spatial_grid(
    const float3* positions,
    int num_items,
    SpatialGrid* grid
) {
    // 1. Compute grid cell for each item
    thrust::device_vector<int> cell_ids(num_items);
    thrust::transform(positions, positions + num_items,
                     cell_ids.begin(),
                     [grid] __device__ (float3 pos) {
                         int3 cell = compute_cell(pos, grid);
                         return cell.x + cell.y * grid->grid_dims.x +
                                cell.z * grid->grid_dims.x * grid->grid_dims.y;
                     });

    // 2. Sort items by cell ID
    thrust::sort_by_key(cell_ids.begin(), cell_ids.end(),
                        thrust::make_counting_iterator(0));

    // 3. Find cell boundaries
    // ... compute cell_starts and cell_counts ...
}

// Query only neighboring cells
__global__ void apply_force_spatial(
    const SpatialGrid* grid,
    const float3* positions,
    float3* forces,
    int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    float3 pos = positions[idx];
    int3 cell = compute_cell(pos, grid);

    // Check 27 neighboring cells (3x3x3)
    for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        int3 neighbor_cell = cell + make_int3(dx, dy, dz);
        if (!is_valid_cell(neighbor_cell, grid)) continue;

        int cell_id = flatten_cell_id(neighbor_cell, grid);
        int start = grid->cell_starts[cell_id];
        int count = grid->cell_counts[cell_id];

        // Only check items in neighboring cell
        for (int i = start; i < start + count; i++) {
            int other_idx = grid->sorted_items[i];
            if (other_idx == idx) continue;

            // Compute force (only if within distance threshold)
            float3 delta = positions[other_idx] - pos;
            if (length(delta) < max_interaction_distance) {
                // Compute and accumulate force
            }
        }
    }}}
}
```

**Expected Impact**: 5-10x speedup for large-scale problems
**Validation**: Comparison count should drop 80-90%

---

### 3.2: Kernel Fusion (Priority: MEDIUM)

**Target**: Multiple small kernels
**Current**: 8 separate kernel launches, each with global memory round-trip
**Goal**: 2-3 fused kernels

#### Implementation

```cuda
// Fuse multiple force computations into single kernel
__global__ void compute_all_forces_fused(
    // All data structures
    const int* item_genres,
    const float* mood_vectors,
    const float* precomputed_similarities,
    float3* positions,
    float3* forces,  // Single output accumulator
    int num_items,
    MediaRecommendationConfig config  // All parameters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

    // Compute all enabled forces in one pass
    if (config.genre_cluster.enabled) {
        total_force = total_force + compute_genre_force(idx, item_genres, positions, config);
    }

    if (config.mood_cluster.enabled) {
        total_force = total_force + compute_mood_force(idx, mood_vectors, positions, config);
    }

    if (config.content_similarity.enabled) {
        total_force = total_force + compute_similarity_force(idx, precomputed_similarities,
                                                              positions, config);
    }

    // Write once to global memory
    forces[idx] = total_force;
}
```

**Expected Impact**: 1.5-2x speedup
**Validation**: Total kernel launch overhead should drop 80%

---

### 3.3: Persistent Threads for Graph Search (Priority: MEDIUM)

**Target**: graph_search.cu
**Current**: Kernel relaunch overhead per BFS level (5-10μs each)
**Goal**: Single kernel launch for entire SSSP

#### Implementation

```cuda
__global__ void persistent_sssp_kernel(
    int source,
    float* distances,
    int* predecessors,
    // ... graph data ...
    int* frontier_a,
    int* frontier_b,
    int* frontier_sizes,
    int num_nodes,
    int max_iterations
) {
    __shared__ int* current_frontier;
    __shared__ int* next_frontier;
    __shared__ int current_size;

    // Initialize
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        frontier_a[0] = source;
        frontier_sizes[0] = 1;
        distances[source] = 0.0f;
    }
    __syncthreads();
    __threadfence();

    // Persistent loop (avoid relaunch)
    for (int iter = 0; iter < max_iterations; iter++) {
        // Determine active frontier
        if (threadIdx.x == 0) {
            current_frontier = (iter % 2 == 0) ? frontier_a : frontier_b;
            next_frontier = (iter % 2 == 0) ? frontier_b : frontier_a;
            current_size = frontier_sizes[iter % 2];
        }
        __syncthreads();

        if (current_size == 0) break;  // Converged

        // Process current frontier (standard logic)
        // ...

        __syncthreads();
        __threadfence();  // Ensure writes visible to all blocks
    }
}
```

**Expected Impact**: 1.5-2x speedup for multi-iteration algorithms
**Validation**: Total runtime including overheads

---

### 3.4: CUDA Graphs for Repeated Workflows (Priority: LOW)

**Target**: Entire pipeline
**Current**: Kernel launch overhead adds up
**Goal**: Capture graph once, replay efficiently

#### Implementation

```cuda
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

// Capture sequence of kernel launches
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Launch all kernels in workflow
compute_similarities<<<grid, block, 0, stream>>>(args);
apply_genre_forces<<<grid, block, 0, stream>>>(args);
apply_mood_forces<<<grid, block, 0, stream>>>(args);
// ... more kernels ...

cudaStreamEndCapture(stream, &graph);

// Instantiate executable graph
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

// Replay graph (very low overhead)
for (int i = 0; i < num_iterations; i++) {
    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);
}
```

**Expected Impact**: 1.2-1.5x speedup for repeated executions
**Validation**: Launch overhead measurement

---

## Phase 4: Launch Configuration Tuning (Week 7)
**Expected Speedup**: 1.5-2x
**Risk**: Low
**Effort**: Low

### 4.1: Occupancy Optimization

#### Tools and Process

```bash
# Use CUDA Occupancy Calculator API
cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                   kernel_function, 0, 0);

# Or manual calculation based on profiling
ncu --metrics "sm__warps_active.avg.pct_of_peak,\
              sm__maximum_warps_per_active_cycle_pct" \
    ./application

# Target: 75-100% occupancy for memory-bound kernels
#         50-75% for compute-bound (registers limiting factor)
```

#### Per-Kernel Tuning

| Kernel | Optimal Block Size | Grid Size | Expected Occupancy |
|--------|-------------------|-----------|-------------------|
| tensor_core_similarity | 128-256 | (N+15)/16 × 2 | 75-100% |
| apply_genre_forces | 256-512 | N/block_size | 60-80% |
| sssp_semantic | 256 | variable | 70-90% |
| multimodal_similarity | 256 | (N+255)/256 | 80-100% |

---

### 4.2: Launch Configuration Function

```cuda
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
};

template<typename KernelFunc>
LaunchConfig compute_launch_config(
    KernelFunc kernel,
    int problem_size,
    int registers_per_thread,
    size_t shared_mem_per_block
) {
    LaunchConfig config;

    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Calculate occupancy for different block sizes
    int max_occupancy = 0;
    int best_block_size = 128;

    for (int bs = 128; bs <= 1024; bs += 32) {
        int occupancy;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, kernel, bs, shared_mem_per_block);

        int total_occupancy = occupancy * prop.multiProcessorCount;
        if (total_occupancy > max_occupancy) {
            max_occupancy = total_occupancy;
            best_block_size = bs;
        }
    }

    config.block = dim3(best_block_size);
    config.grid = dim3((problem_size + best_block_size - 1) / best_block_size);
    config.shared_mem = shared_mem_per_block;

    // Ensure grid utilizes all SMs
    int min_grid = prop.multiProcessorCount * 2;  // 2 blocks per SM minimum
    if (config.grid.x < min_grid) {
        config.grid.x = min(min_grid, (problem_size + best_block_size/2 - 1) / (best_block_size/2));
        config.block.x = best_block_size / 2;
    }

    return config;
}
```

---

## Phase 5: Memory and Streaming (Week 8)
**Expected Speedup**: 1.3-1.8x
**Risk**: Medium
**Effort**: Medium

### 5.1: Streaming for Large Datasets

```cuda
void process_large_dataset_streaming(
    const __half* embeddings,
    int num_items,
    int embedding_dim,
    float* results
) {
    const int NUM_STREAMS = 4;
    const int BATCH_SIZE = num_items / NUM_STREAMS;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Overlapping data transfer and compute
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * BATCH_SIZE;
        int size = (i == NUM_STREAMS - 1) ? (num_items - offset) : BATCH_SIZE;

        // Async copy H2D
        cudaMemcpyAsync(d_batch_embeddings[i], &embeddings[offset * embedding_dim],
                       size * embedding_dim * sizeof(__half),
                       cudaMemcpyHostToDevice, streams[i]);

        // Compute on GPU
        process_batch_kernel<<<grid, block, 0, streams[i]>>>(
            d_batch_embeddings[i], d_batch_results[i], size, embedding_dim);

        // Async copy D2H
        cudaMemcpyAsync(&results[offset], d_batch_results[i],
                       size * sizeof(float),
                       cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}
```

**Expected Impact**: 1.3-1.5x for large datasets (>100K items)

---

### 5.2: Pinned Memory and Zero-Copy

```cuda
// Allocate pinned (page-locked) memory for faster H2D/D2H transfers
__half* h_embeddings_pinned;
cudaMallocHost(&h_embeddings_pinned, num_items * embedding_dim * sizeof(__half));

// Copy data to pinned memory (once)
memcpy(h_embeddings_pinned, embeddings, num_items * embedding_dim * sizeof(__half));

// Async transfers are now 2-3x faster
cudaMemcpyAsync(d_embeddings, h_embeddings_pinned, size, cudaMemcpyHostToDevice, stream);
```

---

## Validation and Measurement Plan

### Key Metrics to Track

1. **Memory Bandwidth**
   ```bash
   ncu --metrics dram__throughput.avg.pct_of_peak \
       --csv ./application > bandwidth.csv
   ```
   - **Target**: 75-85% of peak (240-270 GB/s on T4)

2. **Compute Throughput**
   ```bash
   ncu --metrics sm__sass_inst_executed_op_fp16_pred_on.sum.per_second \
       --csv ./application > compute.csv
   ```
   - **Target**: 50-60 TFLOPS (FP16 with tensor cores)

3. **Warp Efficiency**
   ```bash
   ncu --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct \
       --csv ./application > warp_eff.csv
   ```
   - **Target**: 70-90% active warps

4. **Occupancy**
   ```bash
   ncu --metrics sm__warps_active.avg.pct_of_peak \
       --csv ./application > occupancy.csv
   ```
   - **Target**: 60-100% depending on kernel type

5. **End-to-End Performance**
   - Items/second throughput
   - Latency per query
   - Multi-query batching efficiency

---

## Risk Mitigation

### Testing Strategy
1. **Unit Tests**: Each optimization validated independently
2. **Correctness Tests**: Compare output vs baseline (< 0.1% error)
3. **Performance Tests**: Measure speedup at each phase
4. **Regression Tests**: Ensure no performance degradation on edge cases

### Rollback Plan
- Each phase in separate git branch
- Preserve baseline implementation
- A/B testing infrastructure for production deployment

### Alternative Approaches
- If tensor cores don't deliver: focus on memory optimizations
- If spatial partitioning too complex: use approximate nearest neighbors library (FAISS)
- If persistent threads unstable: optimize launch overhead with CUDA graphs

---

## Expected Final Results

| Metric | Baseline | After Phase 1-2 | After Phase 3-5 | Target |
|--------|----------|-----------------|-----------------|--------|
| Memory BW | 60-80 GB/s | 200-240 GB/s | 240-280 GB/s | 280 GB/s |
| Compute | 10-15 TFLOPS | 40-50 TFLOPS | 50-60 TFLOPS | 60 TFLOPS |
| Warp Eff | 30-50% | 60-75% | 70-90% | 90% |
| Occupancy | 40-75% | 65-85% | 70-95% | 90% |
| **Overall Speedup** | **1x** | **20-30x** | **50-100x** | **100x** |

---

## Success Criteria

✅ **Phase 1 Complete**: 8-12x speedup, <10ms per similarity computation
✅ **Phase 2 Complete**: 40-50x speedup, <3ms per constraint evaluation
✅ **Phase 3 Complete**: 80-100x speedup, <1ms per graph operation
✅ **Phase 4 Complete**: Consistent 75%+ occupancy across all kernels
✅ **Phase 5 Complete**: Handle 100K+ items within memory budget

**Final Goal**: Real-time media recommendation (<10ms end-to-end) for 10K-100K content items.
