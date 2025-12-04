// =============================================================================
// Phase 2 Memory Optimization - Data Structures and Utilities
// =============================================================================
// Target: 4-5x speedup through coalesced memory access and shared memory caching
//
// Key Optimizations:
// 1. Sorted pair batching for consecutive memory access
// 2. Shared memory caching with proper alignment
// 3. Double buffering for overlapped compute and load
// 4. Bank conflict avoidance in shared memory
// 5. Prefetching for latency hiding
//
// Expected Bandwidth: 60 GB/s → 280+ GB/s (4.6x improvement)
// =============================================================================

#ifndef MEMORY_OPTIMIZATION_CUH
#define MEMORY_OPTIMIZATION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Memory Access Patterns
// =============================================================================

// Alignment requirements for coalesced access
#define COALESCE_ALIGNMENT 128  // 128-byte alignment for optimal coalescing
#define CACHE_LINE_SIZE 32      // 32 elements per cache line
#define WARP_SIZE 32

// =============================================================================
// Sorted Batch Structure for Coalesced Access
// =============================================================================

struct SortedPairBatch {
    int src_start;           // Starting index of consecutive source vectors
    int src_end;             // Ending index (exclusive)
    int* tgt_indices;        // Sorted target indices for this batch
    int batch_size;          // Number of targets in batch
    int batch_id;            // Unique batch identifier

    __host__ __device__ int num_sources() const {
        return src_end - src_start;
    }

    __host__ __device__ bool is_valid() const {
        return src_start >= 0 && src_end > src_start && tgt_indices != nullptr && batch_size > 0;
    }
};

// =============================================================================
// Embedding Cache with Alignment for Coalesced Access
// =============================================================================

template<int CACHE_SIZE = 32, int EMBEDDING_DIM = 1024>
struct EmbeddingCache {
    // Aligned data for coalesced access (32 vectors × 1024 dims)
    // Using __align__ directive for 128-byte alignment
    __align__(COALESCE_ALIGNMENT) __half data[CACHE_SIZE][EMBEDDING_DIM];

    // Pre-computed norms for fast similarity computation
    float norms[CACHE_SIZE];

    // Indices of cached vectors (for validation)
    int indices[CACHE_SIZE];

    // Cache metadata
    int num_cached;
    bool is_valid;

    __device__ void invalidate() {
        is_valid = false;
        num_cached = 0;
    }

    __device__ void mark_valid(int count) {
        is_valid = true;
        num_cached = count;
    }

    __device__ __forceinline__ const __half* get_vector(int idx) const {
        return data[idx];
    }

    __device__ __forceinline__ float get_norm(int idx) const {
        return norms[idx];
    }
};

// Specialized cache for smaller embeddings (768 dims - BERT/transformers)
template<>
struct EmbeddingCache<32, 768> {
    __align__(COALESCE_ALIGNMENT) __half data[32][768];
    float norms[32];
    int indices[32];
    int num_cached;
    bool is_valid;

    __device__ void invalidate() { is_valid = false; num_cached = 0; }
    __device__ void mark_valid(int count) { is_valid = true; num_cached = count; }
    __device__ __forceinline__ const __half* get_vector(int idx) const { return data[idx]; }
    __device__ __forceinline__ float get_norm(int idx) const { return norms[idx]; }
};

// =============================================================================
// Tiled Loading for Shared Memory
// =============================================================================

// Tile descriptor for efficient data movement
struct TileDescriptor {
    int tile_row;            // Row index of tile
    int tile_col;            // Column index of tile
    int global_row_start;    // Global row start
    int global_col_start;    // Global column start
    int tile_height;         // Height of tile
    int tile_width;          // Width of tile

    __device__ bool contains(int row, int col) const {
        return row >= global_row_start && row < global_row_start + tile_height &&
               col >= global_col_start && col < global_col_start + tile_width;
    }
};

// =============================================================================
// Coalesced Memory Access Helpers
// =============================================================================

// Load vector with coalesced access pattern
// All threads in warp cooperatively load one vector
template<int DIM>
__device__ __forceinline__ void coalesced_load_vector(
    const __half* __restrict__ global_vec,
    __half* __restrict__ shared_vec,
    int thread_id,
    int num_threads
) {
    // Each thread loads (DIM / num_threads) elements
    // Consecutive threads load consecutive elements → coalesced
    #pragma unroll 4
    for (int i = thread_id; i < DIM; i += num_threads) {
        shared_vec[i] = global_vec[i];
    }
}

// Store vector with coalesced access pattern
template<int DIM>
__device__ __forceinline__ void coalesced_store_vector(
    __half* __restrict__ global_vec,
    const __half* __restrict__ shared_vec,
    int thread_id,
    int num_threads
) {
    #pragma unroll 4
    for (int i = thread_id; i < DIM; i += num_threads) {
        global_vec[i] = shared_vec[i];
    }
}

// Vectorized load using half2 for 2x throughput
template<int DIM>
__device__ __forceinline__ void vectorized_load(
    const __half* __restrict__ global_vec,
    __half* __restrict__ shared_vec,
    int thread_id,
    int num_threads
) {
    static_assert(DIM % 2 == 0, "DIM must be even for half2 operations");

    const half2* global_h2 = reinterpret_cast<const half2*>(global_vec);
    half2* shared_h2 = reinterpret_cast<half2*>(shared_vec);
    int dim_h2 = DIM / 2;

    #pragma unroll 4
    for (int i = thread_id; i < dim_h2; i += num_threads) {
        shared_h2[i] = global_h2[i];
    }
}

// =============================================================================
// Bank Conflict Avoidance
// =============================================================================

// Compute padding to avoid bank conflicts in shared memory
// T4 has 32 banks, each 4 bytes wide
constexpr int compute_bank_conflict_padding(int num_elements, int element_size_bytes) {
    // Add 1 element padding if size aligns with bank stride
    int num_banks = 32;
    int bank_width = 4;

    int elements_per_bank_stride = (num_banks * bank_width) / element_size_bytes;

    if (num_elements % elements_per_bank_stride == 0) {
        return 1; // Add padding
    }
    return 0;
}

// Padded shared memory array to avoid bank conflicts
template<typename T, int ROWS, int COLS>
struct BankConflictFreeArray {
    static constexpr int PADDING = compute_bank_conflict_padding(COLS, sizeof(T));
    static constexpr int PADDED_COLS = COLS + PADDING;

    T data[ROWS][PADDED_COLS];

    __device__ __forceinline__ T& operator()(int row, int col) {
        return data[row][col];
    }

    __device__ __forceinline__ const T& operator()(int row, int col) const {
        return data[row][col];
    }
};

// =============================================================================
// Prefetch Hints
// =============================================================================

// Prefetch data into L2 cache
template<typename T>
__device__ __forceinline__ void prefetch_l2(const T* addr) {
    #if __CUDA_ARCH__ >= 750  // Turing and later
    __builtin_nontemporal_load(addr);
    #endif
}

// Prefetch for read-only access
template<typename T>
__device__ __forceinline__ void prefetch_readonly(const T* addr) {
    #if __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L2 [%0];" :: "l"(addr));
    #endif
}

// =============================================================================
// Warp-Level Primitives for Fast Reductions
// =============================================================================

// Warp-level reduction for dot product accumulation
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max finding
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction with index tracking (for argmax)
__device__ __forceinline__ void warp_reduce_max_with_index(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);

        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// =============================================================================
// Memory Bandwidth Monitoring
// =============================================================================

struct MemoryBandwidthStats {
    unsigned long long bytes_read;
    unsigned long long bytes_written;
    float kernel_time_ms;

    __host__ float compute_bandwidth_gbps() const {
        if (kernel_time_ms <= 0.0f) return 0.0f;

        unsigned long long total_bytes = bytes_read + bytes_written;
        float time_seconds = kernel_time_ms / 1000.0f;
        float bandwidth_bps = static_cast<float>(total_bytes) / time_seconds;
        return bandwidth_bps / (1024.0f * 1024.0f * 1024.0f); // Convert to GB/s
    }

    __host__ float compute_efficiency_percent(float peak_bandwidth_gbps = 320.0f) const {
        float actual_bw = compute_bandwidth_gbps();
        return (actual_bw / peak_bandwidth_gbps) * 100.0f;
    }

    __host__ void print_stats() const {
        printf("Memory Bandwidth Statistics:\n");
        printf("  Bytes Read:    %llu (%.2f MB)\n", bytes_read, bytes_read / (1024.0 * 1024.0));
        printf("  Bytes Written: %llu (%.2f MB)\n", bytes_written, bytes_written / (1024.0 * 1024.0));
        printf("  Kernel Time:   %.2f ms\n", kernel_time_ms);
        printf("  Bandwidth:     %.2f GB/s\n", compute_bandwidth_gbps());
        printf("  Efficiency:    %.1f%% of peak\n", compute_efficiency_percent());
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

// Round up to nearest multiple (for alignment)
__host__ __device__ constexpr int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// Check if pointer is aligned
template<typename T>
__host__ __device__ bool is_aligned(const T* ptr, int alignment = COALESCE_ALIGNMENT) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// Get optimal grid size for GPU occupancy
__host__ dim3 get_optimal_grid_size(int total_elements, int block_size, int sm_count = 40) {
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Target 2-4 blocks per SM for optimal occupancy
    int max_blocks = sm_count * 4;
    num_blocks = min(num_blocks, max_blocks);

    return dim3(num_blocks, 1, 1);
}

#endif // MEMORY_OPTIMIZATION_CUH
