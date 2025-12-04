// =============================================================================
// Memory Layout Optimization - Transpose and Reordering
// =============================================================================
// Optimizes memory layout for coalesced access patterns
//
// Key Operations:
// 1. Transpose embeddings: Row-major → Column-major for better access
// 2. Sort pairs by source index for consecutive memory access
// 3. Batch generation with optimal cache utilization
// 4. Memory alignment and padding for coalescing
//
// Expected Impact: 3-4x bandwidth improvement
// =============================================================================

#include "memory_optimization.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

extern "C" {

// =============================================================================
// Transpose Embeddings for Coalesced Access
// =============================================================================

template<int TILE_SIZE = 32>
__global__ void transpose_embeddings_for_coalescing(
    const __half* __restrict__ row_major,    // [N, D] row-major layout
    __half* __restrict__ col_major,          // [D, N] column-major layout
    int num_embeddings,
    int embedding_dim
) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ __half tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts

    // Block and thread indices
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global input coordinates
    int x_in = bx + tx;
    int y_in = by + ty;

    // Load from row-major with coalescing (consecutive threads load consecutive elements)
    if (x_in < embedding_dim && y_in < num_embeddings) {
        tile[ty][tx] = row_major[y_in * embedding_dim + x_in];
    }

    __syncthreads();

    // Global output coordinates (transposed)
    int x_out = by + tx;
    int y_out = bx + ty;

    // Store to column-major with coalescing
    if (x_out < num_embeddings && y_out < embedding_dim) {
        col_major[y_out * num_embeddings + x_out] = tile[tx][ty];
    }
}

// =============================================================================
// Sort Pairs by Source Index
// =============================================================================

struct IndexPair {
    int src;
    int tgt;

    __host__ __device__ IndexPair() : src(-1), tgt(-1) {}
    __host__ __device__ IndexPair(int s, int t) : src(s), tgt(t) {}

    // Comparison operator for sorting by source
    __host__ __device__ bool operator<(const IndexPair& other) const {
        if (src != other.src) return src < other.src;
        return tgt < other.tgt;
    }
};

// Sort pairs by source index using thrust
void sort_pairs_by_source(
    const int* src_indices,
    const int* tgt_indices,
    int* sorted_src,
    int* sorted_tgt,
    int num_pairs
) {
    // Create device vectors
    thrust::device_vector<IndexPair> pairs(num_pairs);

    // Copy data to device
    thrust::device_ptr<const int> d_src(src_indices);
    thrust::device_ptr<const int> d_tgt(tgt_indices);

    // Create pairs on device
    auto create_pairs = [] __device__ (int src, int tgt) {
        return IndexPair(src, tgt);
    };

    thrust::transform(
        d_src, d_src + num_pairs,
        d_tgt,
        pairs.begin(),
        create_pairs
    );

    // Sort pairs by source index
    thrust::sort(pairs.begin(), pairs.end());

    // Extract sorted indices
    thrust::device_ptr<int> d_sorted_src(sorted_src);
    thrust::device_ptr<int> d_sorted_tgt(sorted_tgt);

    auto extract_src = [] __device__ (const IndexPair& p) { return p.src; };
    auto extract_tgt = [] __device__ (const IndexPair& p) { return p.tgt; };

    thrust::transform(pairs.begin(), pairs.end(), d_sorted_src, extract_src);
    thrust::transform(pairs.begin(), pairs.end(), d_sorted_tgt, extract_tgt);
}

// =============================================================================
// Generate Sorted Batches for Optimal Caching
// =============================================================================

__global__ void generate_sorted_batches(
    const int* __restrict__ sorted_src_indices,
    const int* __restrict__ sorted_tgt_indices,
    SortedPairBatch* __restrict__ batches,
    int* __restrict__ batch_count,
    int num_pairs,
    int max_batch_size = 256
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int current_batch = 0;
    int current_src_start = sorted_src_indices[0];
    int pair_idx = 0;

    while (pair_idx < num_pairs) {
        int batch_start = pair_idx;
        int src_start = sorted_src_indices[pair_idx];
        int src_end = src_start;

        // Collect consecutive source indices
        while (pair_idx < num_pairs &&
               sorted_src_indices[pair_idx] == src_end &&
               (pair_idx - batch_start) < max_batch_size) {
            src_end = sorted_src_indices[pair_idx] + 1;
            pair_idx++;
        }

        // Create batch
        SortedPairBatch batch;
        batch.src_start = src_start;
        batch.src_end = src_end;
        batch.tgt_indices = const_cast<int*>(&sorted_tgt_indices[batch_start]);
        batch.batch_size = pair_idx - batch_start;
        batch.batch_id = current_batch;

        batches[current_batch] = batch;
        current_batch++;
    }

    *batch_count = current_batch;
}

// =============================================================================
// Reorder Embeddings for Better Locality
// =============================================================================

__global__ void reorder_embeddings_by_frequency(
    const __half* __restrict__ original_embeddings,
    __half* __restrict__ reordered_embeddings,
    const int* __restrict__ reorder_map,    // Maps original index → new index
    int num_embeddings,
    int embedding_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_embeddings; idx += stride) {
        int new_idx = reorder_map[idx];

        // Copy entire embedding vector
        for (int d = 0; d < embedding_dim; d++) {
            reordered_embeddings[new_idx * embedding_dim + d] =
                original_embeddings[idx * embedding_dim + d];
        }
    }
}

// =============================================================================
// Pad Embeddings for Alignment
// =============================================================================

__global__ void pad_embeddings_for_alignment(
    const __half* __restrict__ unpadded,
    __half* __restrict__ padded,
    int num_embeddings,
    int original_dim,
    int padded_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_embeddings) return;

    // Copy original data
    for (int d = 0; d < original_dim; d++) {
        padded[idx * padded_dim + d] = unpadded[idx * original_dim + d];
    }

    // Zero-pad remaining dimensions
    for (int d = original_dim; d < padded_dim; d++) {
        padded[idx * padded_dim + d] = __float2half(0.0f);
    }
}

// =============================================================================
// Streaming Data Transfer with Pinned Memory
// =============================================================================

class StreamingDataTransfer {
private:
    static constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    __half* device_buffers[NUM_STREAMS];
    __half* pinned_host_buffer;
    size_t buffer_size_bytes;

public:
    StreamingDataTransfer(int chunk_size, int embedding_dim) {
        buffer_size_bytes = chunk_size * embedding_dim * sizeof(__half);

        // Create streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            cudaMalloc(&device_buffers[i], buffer_size_bytes);
        }

        // Allocate pinned host memory for faster transfers
        cudaMallocHost(&pinned_host_buffer, buffer_size_bytes * NUM_STREAMS);
    }

    ~StreamingDataTransfer() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
            cudaFree(device_buffers[i]);
        }
        cudaFreeHost(pinned_host_buffer);
    }

    // Transfer large dataset in streaming chunks
    void transfer_large_dataset(
        const __half* host_embeddings,
        __half* device_embeddings,
        int num_embeddings,
        int embedding_dim
    ) {
        int chunk_size = buffer_size_bytes / (embedding_dim * sizeof(__half));
        int num_chunks = (num_embeddings + chunk_size - 1) / chunk_size;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int stream_id = chunk % NUM_STREAMS;
            int chunk_start = chunk * chunk_size;
            int chunk_elements = min(chunk_size, num_embeddings - chunk_start);
            size_t chunk_bytes = chunk_elements * embedding_dim * sizeof(__half);

            // Async copy to device
            cudaMemcpyAsync(
                device_buffers[stream_id],
                host_embeddings + chunk_start * embedding_dim,
                chunk_bytes,
                cudaMemcpyHostToDevice,
                streams[stream_id]
            );

            // Copy from staging buffer to final location
            cudaMemcpyAsync(
                device_embeddings + chunk_start * embedding_dim,
                device_buffers[stream_id],
                chunk_bytes,
                cudaMemcpyDeviceToDevice,
                streams[stream_id]
            );
        }

        // Wait for all streams to complete
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }
};

// =============================================================================
// L2 Cache-Aware Batching
// =============================================================================

// T4 has 4MB L2 cache - optimize batch size to fit in L2
constexpr int T4_L2_CACHE_SIZE = 4 * 1024 * 1024;  // 4MB

// Calculate optimal batch size for L2 cache residency
__host__ int calculate_optimal_batch_size(int embedding_dim, float cache_utilization = 0.8f) {
    size_t available_cache = static_cast<size_t>(T4_L2_CACHE_SIZE * cache_utilization);
    size_t embedding_size = embedding_dim * sizeof(__half);

    // Number of embeddings that fit in L2 cache
    int max_embeddings_in_cache = available_cache / embedding_size;

    // Round down to nearest power of 2 for alignment
    int batch_size = 1;
    while (batch_size * 2 <= max_embeddings_in_cache) {
        batch_size *= 2;
    }

    return batch_size;
}

// =============================================================================
// Memory Access Pattern Analyzer
// =============================================================================

struct MemoryAccessPattern {
    int num_sequential_accesses;
    int num_random_accesses;
    float sequential_ratio;

    __host__ void analyze_pairs(const int* src_indices, int num_pairs) {
        num_sequential_accesses = 0;
        num_random_accesses = 0;

        for (int i = 1; i < num_pairs; i++) {
            if (src_indices[i] == src_indices[i-1] + 1) {
                num_sequential_accesses++;
            } else {
                num_random_accesses++;
            }
        }

        sequential_ratio = static_cast<float>(num_sequential_accesses) / num_pairs;
    }

    __host__ void print_analysis() const {
        printf("Memory Access Pattern Analysis:\n");
        printf("  Sequential accesses: %d\n", num_sequential_accesses);
        printf("  Random accesses: %d\n", num_random_accesses);
        printf("  Sequential ratio: %.2f%%\n", sequential_ratio * 100.0f);
        printf("  Expected bandwidth improvement: %.2fx\n", 1.0f + sequential_ratio * 3.0f);
    }
};

// =============================================================================
// Host API Functions
// =============================================================================

// Transpose embeddings on host
cudaError_t transpose_embeddings(
    const __half* row_major,
    __half* col_major,
    int num_embeddings,
    int embedding_dim
) {
    constexpr int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (embedding_dim + TILE_SIZE - 1) / TILE_SIZE,
        (num_embeddings + TILE_SIZE - 1) / TILE_SIZE
    );

    transpose_embeddings_for_coalescing<TILE_SIZE><<<grid_size, block_size>>>(
        row_major, col_major, num_embeddings, embedding_dim
    );

    return cudaGetLastError();
}

} // extern "C"
