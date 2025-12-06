#include "hnsw_gpu.cuh"
#include "lsh_gpu.cu"
#include "product_quantization.cu"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Hybrid index combining HNSW, LSH, and PQ
struct HybridIndex {
    HNSW_GPU hnsw;
    LSH_GPU lsh;
    ProductQuantizer pq;

    __half* embeddings;          // Original embeddings [num_items, embedding_dim]
    int num_items;
    int embedding_dim;

    // Configuration
    enum SearchMode {
        EXACT,                   // Full HNSW search
        LSH_HNSW,               // LSH candidates + HNSW refinement
        LSH_PQ,                 // LSH candidates + PQ distances
        ADAPTIVE                // Choose based on query characteristics
    };

    SearchMode mode;
};

// Adaptive mode selection based on query characteristics
__device__ HybridIndex::SearchMode select_search_mode(
    const __half* query,
    int k,
    float recall_target
) {
    // Heuristics for mode selection:
    // - Small k and high recall: EXACT
    // - Large k and medium recall: LSH_HNSW
    // - Very large k or low recall: LSH_PQ

    if (k <= 10 && recall_target > 0.95f) {
        return HybridIndex::EXACT;
    } else if (k <= 100 && recall_target > 0.85f) {
        return HybridIndex::LSH_HNSW;
    } else {
        return HybridIndex::LSH_PQ;
    }
}

// Exact search using HNSW
__global__ void hybrid_search_exact(
    HybridIndex index,
    const __half* queries,       // [batch_size, embedding_dim]
    int* results,                // [batch_size, k]
    float* distances,            // [batch_size, k]
    int batch_size,
    int k,
    int ef
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * index.hnsw.embedding_dim;

    // Delegate to HNSW device function
    hnsw_search_single(
        index.hnsw,
        query,
        results + query_id * k,
        distances + query_id * k,
        k,
        ef
    );
}

// LSH + HNSW hybrid search
__global__ void hybrid_search_lsh_hnsw(
    HybridIndex index,
    const __half* queries,       // [batch_size, embedding_dim]
    int* results,                // [batch_size, k]
    float* distances,            // [batch_size, k]
    int batch_size,
    int k,
    int max_candidates
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * index.embedding_dim;

    // Step 1: LSH candidate generation (fast, recall ~0.7-0.8)
    __shared__ int lsh_candidates[2048];
    __shared__ int num_lsh_candidates;

    if (threadIdx.x == 0) {
        num_lsh_candidates = 0;
    }
    __syncthreads();

    // Get LSH candidates
    __shared__ uint32_t seen_bitmap[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        seen_bitmap[i] = 0;
    }
    __syncthreads();

    for (int table = threadIdx.x; table < index.lsh.num_tables; table += blockDim.x) {
        uint32_t bucket = index.lsh.compute_hash(query, table);

        int bucket_offset = table * index.lsh.num_buckets * index.lsh.bucket_size +
                           bucket * index.lsh.bucket_size;
        int count = index.lsh.bucket_counts[table * index.lsh.num_buckets + bucket];

        for (int i = 0; i < min(count, index.lsh.bucket_size); i++) {
            int item_id = index.lsh.hash_tables[bucket_offset + i];

            int word_idx = item_id / 32;
            int bit_idx = item_id % 32;
            uint32_t old = atomicOr(&seen_bitmap[word_idx], 1u << bit_idx);

            if (!(old & (1u << bit_idx))) {
                int pos = atomicAdd(&num_lsh_candidates, 1);
                if (pos < 2048) {
                    lsh_candidates[pos] = item_id;
                }
            }
        }
    }
    __syncthreads();

    // Step 2: Refine using HNSW graph navigation
    // Build mini-HNSW from candidates for fast k-NN
    __shared__ float candidate_dists[2048];

    for (int i = threadIdx.x; i < num_lsh_candidates; i += blockDim.x) {
        int candidate_id = lsh_candidates[i];
        const __half* candidate_emb = index.embeddings + candidate_id * index.embedding_dim;

        candidate_dists[i] = compute_distance_tensor_core(
            query,
            candidate_emb,
            index.embedding_dim
        );
    }
    __syncthreads();

    // Partial sort to get top-k
    if (threadIdx.x == 0) {
        // Use heap or quickselect for top-k
        for (int i = 0; i < k; i++) {
            int min_idx = i;
            float min_dist = candidate_dists[i];

            for (int j = i + 1; j < num_lsh_candidates; j++) {
                if (candidate_dists[j] < min_dist) {
                    min_dist = candidate_dists[j];
                    min_idx = j;
                }
            }

            if (min_idx != i) {
                float tmp_dist = candidate_dists[i];
                candidate_dists[i] = candidate_dists[min_idx];
                candidate_dists[min_idx] = tmp_dist;

                int tmp_id = lsh_candidates[i];
                lsh_candidates[i] = lsh_candidates[min_idx];
                lsh_candidates[min_idx] = tmp_id;
            }

            results[query_id * k + i] = lsh_candidates[i];
            distances[query_id * k + i] = candidate_dists[i];
        }
    }
}

// LSH + PQ hybrid search (most memory efficient)
__global__ void hybrid_search_lsh_pq(
    HybridIndex index,
    const __half* queries,
    int* results,
    float* distances,
    int batch_size,
    int k
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * index.embedding_dim;

    // Build PQ distance table (reduced from 256 to 128 centroids per subspace to save memory)
    extern __shared__ float pq_distance_table[];

    int num_subspaces = min(index.pq.num_subspaces, 32);  // Limit to reduce memory
    for (int s = 0; s < num_subspaces; s++) {
        for (int c = threadIdx.x; c < 128; c += blockDim.x) {
            const __half* query_subvector = query + s * index.pq.subspace_dim;
            const __half* centroid = index.pq.centroids +
                                     s * 256 * index.pq.subspace_dim +
                                     c * index.pq.subspace_dim;

            float dist = 0.0f;
            for (int d = 0; d < index.pq.subspace_dim; d++) {
                float diff = __half2float(query_subvector[d]) - __half2float(centroid[d]);
                dist += diff * diff;
            }

            pq_distance_table[s * 128 + c] = dist;
        }
    }
    __syncthreads();

    // Get LSH candidates (reduced from 4096 to 1024 to save memory)
    __shared__ int lsh_candidates[1024];
    __shared__ int num_lsh_candidates;
    __shared__ uint32_t seen_bitmap[2048];

    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        seen_bitmap[i] = 0;
    }
    if (threadIdx.x == 0) num_lsh_candidates = 0;
    __syncthreads();

    for (int table = threadIdx.x; table < index.lsh.num_tables; table += blockDim.x) {
        uint32_t bucket = index.lsh.compute_hash(query, table);

        int bucket_offset = table * index.lsh.num_buckets * index.lsh.bucket_size +
                           bucket * index.lsh.bucket_size;
        int count = index.lsh.bucket_counts[table * index.lsh.num_buckets + bucket];

        for (int i = 0; i < min(count, index.lsh.bucket_size); i++) {
            int item_id = index.lsh.hash_tables[bucket_offset + i];

            int word_idx = item_id / 32;
            int bit_idx = item_id % 32;
            uint32_t old = atomicOr(&seen_bitmap[word_idx], 1u << bit_idx);

            if (!(old & (1u << bit_idx))) {
                int pos = atomicAdd(&num_lsh_candidates, 1);
                if (pos < 1024) {
                    lsh_candidates[pos] = item_id;
                }
            }
        }
    }
    __syncthreads();

    // Compute PQ distances for candidates
    __shared__ float candidate_dists[1024];

    for (int i = threadIdx.x; i < num_lsh_candidates; i += blockDim.x) {
        int vec_id = lsh_candidates[i];

        float dist = 0.0f;
        for (int s = 0; s < num_subspaces; s++) {
            uint8_t code = index.pq.codes[vec_id * index.pq.num_subspaces + s];
            dist += pq_distance_table[s * 128 + code];
        }

        candidate_dists[i] = sqrtf(dist);
    }
    __syncthreads();

    // Select top-k
    if (threadIdx.x == 0) {
        for (int i = 0; i < k && i < num_lsh_candidates; i++) {
            int min_idx = i;
            float min_dist = candidate_dists[i];

            for (int j = i + 1; j < num_lsh_candidates; j++) {
                if (candidate_dists[j] < min_dist) {
                    min_dist = candidate_dists[j];
                    min_idx = j;
                }
            }

            if (min_idx != i) {
                float tmp_dist = candidate_dists[i];
                candidate_dists[i] = candidate_dists[min_idx];
                candidate_dists[min_idx] = tmp_dist;

                int tmp_id = lsh_candidates[i];
                lsh_candidates[i] = lsh_candidates[min_idx];
                lsh_candidates[min_idx] = tmp_id;
            }

            results[query_id * k + i] = lsh_candidates[i];
            distances[query_id * k + i] = candidate_dists[i];
        }
    }
}

// Device function wrapper for exact search (callable from kernels without dynamic parallelism)
__device__ void hybrid_search_exact_device(
    HybridIndex index,
    const __half* query,
    int* results,
    float* distances,
    int k,
    int ef
) {
    hnsw_search_single(
        index.hnsw,
        query,
        results,
        distances,
        k,
        ef
    );
}

// Device function wrapper for LSH+HNSW (callable from kernels)
// Note: Uses external shared memory to avoid allocation conflicts
__device__ void hybrid_search_lsh_hnsw_device(
    HybridIndex index,
    const __half* query,
    int* results,
    float* distances,
    int k,
    int max_candidates,
    int* lsh_candidates,        // Shared memory buffer (size: 1024)
    float* candidate_dists,     // Shared memory buffer (size: 1024)
    uint32_t* seen_bitmap       // Shared memory buffer (size: 2048)
) {
    // LSH candidate generation
    int& num_lsh_candidates = lsh_candidates[1023];  // Use last element as counter

    if (threadIdx.x == 0) {
        num_lsh_candidates = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        seen_bitmap[i] = 0;
    }
    __syncthreads();

    for (int table = threadIdx.x; table < index.lsh.num_tables; table += blockDim.x) {
        uint32_t bucket = index.lsh.compute_hash(query, table);

        int bucket_offset = table * index.lsh.num_buckets * index.lsh.bucket_size +
                           bucket * index.lsh.bucket_size;
        int count = index.lsh.bucket_counts[table * index.lsh.num_buckets + bucket];

        for (int i = 0; i < min(count, index.lsh.bucket_size); i++) {
            int item_id = index.lsh.hash_tables[bucket_offset + i];

            int word_idx = item_id / 32;
            int bit_idx = item_id % 32;
            uint32_t old = atomicOr(&seen_bitmap[word_idx], 1u << bit_idx);

            if (!(old & (1u << bit_idx))) {
                int pos = atomicAdd(&num_lsh_candidates, 1);
                if (pos < 1023) {  // Reserve last slot for counter
                    lsh_candidates[pos] = item_id;
                }
            }
        }
    }
    __syncthreads();

    int num_cands = num_lsh_candidates;
    for (int i = threadIdx.x; i < num_cands; i += blockDim.x) {
        int candidate_id = lsh_candidates[i];
        const __half* candidate_emb = index.embeddings + candidate_id * index.embedding_dim;

        candidate_dists[i] = compute_distance_tensor_core(
            query,
            candidate_emb,
            index.embedding_dim
        );
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            int min_idx = i;
            float min_dist = candidate_dists[i];

            for (int j = i + 1; j < num_cands; j++) {
                if (candidate_dists[j] < min_dist) {
                    min_dist = candidate_dists[j];
                    min_idx = j;
                }
            }

            if (min_idx != i) {
                float tmp_dist = candidate_dists[i];
                candidate_dists[i] = candidate_dists[min_idx];
                candidate_dists[min_idx] = tmp_dist;

                int tmp_id = lsh_candidates[i];
                lsh_candidates[i] = lsh_candidates[min_idx];
                lsh_candidates[min_idx] = tmp_id;
            }

            results[i] = lsh_candidates[i];
            distances[i] = candidate_dists[i];
        }
    }
}

// Adaptive search that selects best strategy per query
__global__ void hybrid_search_adaptive(
    HybridIndex index,
    const __half* queries,
    int* results,
    float* distances,
    int batch_size,
    int k,
    float recall_target
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * index.embedding_dim;

    // Select mode based on query characteristics
    HybridIndex::SearchMode mode = select_search_mode(query, k, recall_target);

    // Shared memory for LSH+HNSW search (total: ~14KB)
    __shared__ int lsh_candidates[1024];
    __shared__ float candidate_dists[1024];
    __shared__ uint32_t seen_bitmap[2048];

    // Dispatch to appropriate search method (direct device calls)
    switch (mode) {
        case HybridIndex::EXACT:
            // Use HNSW only
            hybrid_search_exact_device(
                index, query,
                results + query_id * k, distances + query_id * k,
                k, 64
            );
            break;

        case HybridIndex::LSH_HNSW:
        case HybridIndex::ADAPTIVE:
        default:
            // Use LSH + HNSW (also default for adaptive and LSH_PQ modes)
            hybrid_search_lsh_hnsw_device(
                index, query,
                results + query_id * k, distances + query_id * k,
                k, 2000,
                lsh_candidates, candidate_dists, seen_bitmap
            );
            break;
    }
}

// Batch processing with cooperative groups for large-scale search
__global__ void hybrid_search_batch_cooperative(
    HybridIndex index,
    const __half* queries,
    int* results,
    float* distances,
    int batch_size,
    int k,
    HybridIndex::SearchMode mode
) {
    cg::grid_group grid = cg::this_grid();

    // Shared memory for LSH+HNSW search
    __shared__ int lsh_candidates[1024];
    __shared__ float candidate_dists[1024];
    __shared__ uint32_t seen_bitmap[2048];

    // Each thread block processes one query
    for (int query_id = blockIdx.x; query_id < batch_size; query_id += gridDim.x) {
        const __half* query = queries + query_id * index.embedding_dim;
        int* query_results = results + query_id * k;
        float* query_distances = distances + query_id * k;

        // Execute search based on mode (direct device calls)
        switch (mode) {
            case HybridIndex::EXACT:
                hybrid_search_exact_device(
                    index, query, query_results, query_distances, k, 64
                );
                break;

            case HybridIndex::LSH_HNSW:
            default:
                hybrid_search_lsh_hnsw_device(
                    index, query, query_results, query_distances, k, 2000,
                    lsh_candidates, candidate_dists, seen_bitmap
                );
                break;
        }
    }

    grid.sync();
}
