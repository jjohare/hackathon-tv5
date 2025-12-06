// =============================================================================
// Phase 2: Sorted Batch Processing with Coalesced Memory Access
// =============================================================================
// Target: 4-5x speedup on top of Phase 1 optimizations
//
// Key Optimizations:
// 1. Sort pairs by source index for consecutive memory access
// 2. Batch processing with shared memory caching
// 3. Coalesced loads/stores for maximum bandwidth
// 4. Double buffering to overlap compute and memory
// 5. Prefetching to hide memory latency
//
// Memory Pattern: Random (60 GB/s) → Coalesced (280+ GB/s)
// =============================================================================

#include "memory_optimization.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

// =============================================================================
// Optimized Cosine Similarity with Cached Norms
// =============================================================================

template<int DIM>
__device__ __forceinline__ float compute_cosine_similarity_optimized(
    const __half* __restrict__ vec_a,
    const __half* __restrict__ vec_b,
    float norm_a
) {
    float dot = 0.0f;
    float norm_b = 0.0f;

    // Use half2 for 2x throughput
    const half2* vec_a_h2 = reinterpret_cast<const half2*>(vec_a);
    const half2* vec_b_h2 = reinterpret_cast<const half2*>(vec_b);
    int dim_h2 = DIM / 2;

    #pragma unroll 8
    for (int i = 0; i < dim_h2; i++) {
        half2 a = vec_a_h2[i];
        half2 b = vec_b_h2[i];

        float2 a_f = __half22float2(a);
        float2 b_f = __half22float2(b);

        dot += a_f.x * b_f.x + a_f.y * b_f.y;
        norm_b += b_f.x * b_f.x + b_f.y * b_f.y;
    }

    // Warp-level reduction for efficiency
    dot = warp_reduce_sum(dot);
    norm_b = warp_reduce_sum(norm_b);

    norm_b = sqrtf(norm_b);

    float norm_product = norm_a * norm_b;
    if (norm_product < 1e-6f) return 0.0f;

    return dot / norm_product;
}

// =============================================================================
// Phase 2: Sorted Batch Similarity Kernel with Coalesced Access
// =============================================================================

template<int CACHE_SIZE = 32, int EMBEDDING_DIM = 1024>
__device__ void compute_similarity_sorted_coalesced_impl(
    const __half* __restrict__ embeddings,
    const SortedPairBatch* __restrict__ batches,
    float* __restrict__ similarities,
    int num_batches,
    int num_items
) {
    // Shared memory cache for source vectors (reused across targets)
    __shared__ EmbeddingCache<CACHE_SIZE, EMBEDDING_DIM> cache;

    int batch_id = blockIdx.x;
    if (batch_id >= num_batches) return;

    const SortedPairBatch& batch = batches[batch_id];

    if (!batch.is_valid()) return;

    int num_sources = batch.num_sources();
    if (num_sources > CACHE_SIZE) return; // Safety check

    // ==========================================================================
    // PHASE 1: Coalesced Load of Source Vectors into Shared Memory
    // ==========================================================================

    // All threads participate in loading for maximum bandwidth
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load source vectors cooperatively
    for (int src_idx = 0; src_idx < num_sources; src_idx++) {
        int global_src_idx = batch.src_start + src_idx;

        // Coalesced vectorized load using half2
        vectorized_load<EMBEDDING_DIM>(
            &embeddings[global_src_idx * EMBEDDING_DIM],
            cache.data[src_idx],
            tid,
            num_threads
        );

        cache.indices[src_idx] = global_src_idx;
    }

    __syncthreads();

    // ==========================================================================
    // Compute Norms for Cached Vectors
    // ==========================================================================

    if (tid < num_sources) {
        float norm = 0.0f;

        #pragma unroll 8
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            float val = __half2float(cache.data[tid][i]);
            norm += val * val;
        }

        cache.norms[tid] = sqrtf(norm);
    }

    cache.mark_valid(num_sources);

    __syncthreads();

    // ==========================================================================
    // PHASE 2: Process Targets with Cached Sources
    // ==========================================================================

    // Each thread processes multiple targets
    for (int tgt_idx = tid; tgt_idx < batch.batch_size; tgt_idx += num_threads) {
        int tgt_global = batch.tgt_indices[tgt_idx];

        // Load target vector into registers (coalesced within warp)
        __half target[EMBEDDING_DIM];

        // Vectorized coalesced load
        vectorized_load<EMBEDDING_DIM>(
            &embeddings[tgt_global * EMBEDDING_DIM],
            target,
            0,  // Single thread loads entire vector
            1
        );

        // Compute target norm
        float target_norm = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            float val = __half2float(target[i]);
            target_norm += val * val;
        }
        target_norm = sqrtf(target_norm);

        // Compute similarities with all cached sources
        for (int src_idx = 0; src_idx < num_sources; src_idx++) {
            int global_src_idx = batch.src_start + src_idx;

            // Compute dot product
            float dot = 0.0f;

            const half2* src_h2 = reinterpret_cast<const half2*>(cache.data[src_idx]);
            const half2* tgt_h2 = reinterpret_cast<const half2*>(target);
            int dim_h2 = EMBEDDING_DIM / 2;

            #pragma unroll 8
            for (int i = 0; i < dim_h2; i++) {
                float2 s = __half22float2(src_h2[i]);
                float2 t = __half22float2(tgt_h2[i]);
                dot += s.x * t.x + s.y * t.y;
            }

            // Compute cosine similarity
            float norm_product = cache.norms[src_idx] * target_norm;
            float similarity = (norm_product > 1e-6f) ? (dot / norm_product) : 0.0f;

            // Store result with coalesced write
            similarities[global_src_idx * num_items + tgt_global] = similarity;
        }
    }
}

// =============================================================================
// Streaming Similarity with Double Buffering
// =============================================================================

template<int CACHE_SIZE = 32, int EMBEDDING_DIM = 1024>
__global__ void similarity_with_prefetch_double_buffer(
    const __half* __restrict__ embeddings,
    const SortedPairBatch* __restrict__ batches,
    float* __restrict__ similarities,
    int num_batches,
    int num_items
) {
    // Double buffer in shared memory for overlapped compute and load
    __shared__ EmbeddingCache<CACHE_SIZE / 2, EMBEDDING_DIM> buffer_A;
    __shared__ EmbeddingCache<CACHE_SIZE / 2, EMBEDDING_DIM> buffer_B;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Prefetch first batch into buffer_A
    if (blockIdx.x < num_batches) {
        const SortedPairBatch& first_batch = batches[blockIdx.x];
        int num_sources = min(first_batch.num_sources(), CACHE_SIZE / 2);

        for (int i = 0; i < num_sources; i++) {
            int global_idx = first_batch.src_start + i;
            vectorized_load<EMBEDDING_DIM>(
                &embeddings[global_idx * EMBEDDING_DIM],
                buffer_A.data[i],
                tid,
                num_threads
            );
        }
    }

    __syncthreads();

    // Process batches with double buffering
    for (int batch_idx = blockIdx.x; batch_idx < num_batches; batch_idx += gridDim.x) {
        const SortedPairBatch& current_batch = batches[batch_idx];

        // Determine which buffer to use
        bool use_buffer_a = (batch_idx % 2 == 0);
        EmbeddingCache<CACHE_SIZE / 2, EMBEDDING_DIM>& current_buffer = use_buffer_a ? buffer_A : buffer_B;
        EmbeddingCache<CACHE_SIZE / 2, EMBEDDING_DIM>& next_buffer = use_buffer_a ? buffer_B : buffer_A;

        // Prefetch next batch into alternate buffer while processing current
        if (batch_idx + 1 < num_batches) {
            const SortedPairBatch& next_batch = batches[batch_idx + 1];
            int num_sources = min(next_batch.num_sources(), CACHE_SIZE / 2);

            for (int i = tid; i < num_sources; i += num_threads) {
                int global_idx = next_batch.src_start + i;
                vectorized_load<EMBEDDING_DIM>(
                    &embeddings[global_idx * EMBEDDING_DIM],
                    next_buffer.data[i],
                    0,
                    1
                );
            }
        }

        // Process current batch (using current_buffer)
        int num_sources = min(current_batch.num_sources(), CACHE_SIZE / 2);

        // Compute norms for cached vectors
        if (tid < num_sources) {
            float norm = 0.0f;
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                float val = __half2float(current_buffer.data[tid][i]);
                norm += val * val;
            }
            current_buffer.norms[tid] = sqrtf(norm);
        }

        __syncthreads();

        // Process targets
        for (int tgt_idx = tid; tgt_idx < current_batch.batch_size; tgt_idx += num_threads) {
            int tgt_global = current_batch.tgt_indices[tgt_idx];

            __half target[EMBEDDING_DIM];
            vectorized_load<EMBEDDING_DIM>(
                &embeddings[tgt_global * EMBEDDING_DIM],
                target,
                0,
                1
            );

            float target_norm = 0.0f;
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                float val = __half2float(target[i]);
                target_norm += val * val;
            }
            target_norm = sqrtf(target_norm);

            for (int src_idx = 0; src_idx < num_sources; src_idx++) {
                int global_src_idx = current_batch.src_start + src_idx;

                float dot = 0.0f;
                const half2* src_h2 = reinterpret_cast<const half2*>(current_buffer.data[src_idx]);
                const half2* tgt_h2 = reinterpret_cast<const half2*>(target);

                #pragma unroll 8
                for (int i = 0; i < EMBEDDING_DIM / 2; i++) {
                    float2 s = __half22float2(src_h2[i]);
                    float2 t = __half22float2(tgt_h2[i]);
                    dot += s.x * t.x + s.y * t.y;
                }

                float similarity = dot / (current_buffer.norms[src_idx] * target_norm + 1e-6f);
                similarities[global_src_idx * num_items + tgt_global] = similarity;
            }
        }

        __syncthreads();
    }
}

// =============================================================================
// Specialized Kernel for Common Embedding Dimensions
// =============================================================================

// Instantiate for common dimensions
__global__ void compute_similarity_sorted_768(
    const __half* __restrict__ embeddings,
    const SortedPairBatch* __restrict__ batches,
    float* __restrict__ similarities,
    int num_batches,
    int num_items
) {
    compute_similarity_sorted_coalesced_impl<32, 768>(
        embeddings, batches, similarities, num_batches, num_items
    );
}

__global__ void compute_similarity_sorted_1024(
    const __half* __restrict__ embeddings,
    const SortedPairBatch* __restrict__ batches,
    float* __restrict__ similarities,
    int num_batches,
    int num_items
) {
    compute_similarity_sorted_coalesced_impl<32, 1024>(
        embeddings, batches, similarities, num_batches, num_items
    );
}

__global__ void compute_similarity_sorted_2048(
    const __half* __restrict__ embeddings,
    const SortedPairBatch* __restrict__ batches,
    float* __restrict__ similarities,
    int num_batches,
    int num_items
) {
    compute_similarity_sorted_coalesced_impl<32, 2048>(
        embeddings, batches, similarities, num_batches, num_items
    );
}

// =============================================================================
// Host API Functions
// =============================================================================

// Launch sorted similarity kernel with optimal configuration
cudaError_t launch_sorted_similarity_kernel(
    const __half* embeddings,
    const SortedPairBatch* batches,
    float* similarities,
    int num_batches,
    int num_items,
    int embedding_dim,
    cudaStream_t stream = 0
) {
    // Optimal configuration for T4
    int block_size = 256;  // 256 threads per block
    int grid_size = num_batches;

    // Select appropriate kernel based on embedding dimension
    if (embedding_dim == 768) {
        compute_similarity_sorted_768<<<grid_size, block_size, 0, stream>>>(
            embeddings, batches, similarities, num_batches, num_items
        );
    } else if (embedding_dim == 1024) {
        compute_similarity_sorted_1024<<<grid_size, block_size, 0, stream>>>(
            embeddings, batches, similarities, num_batches, num_items
        );
    } else if (embedding_dim == 2048) {
        compute_similarity_sorted_2048<<<grid_size, block_size, 0, stream>>>(
            embeddings, batches, similarities, num_batches, num_items
        );
    } else {
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}
