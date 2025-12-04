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
    // Delegate to HNSW
    hnsw_search_batch(
        index.hnsw,
        queries,
        results,
        distances,
        batch_size,
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

    // Build PQ distance table
    extern __shared__ float pq_distance_table[];

    for (int s = 0; s < index.pq.num_subspaces; s++) {
        for (int c = threadIdx.x; c < 256; c += blockDim.x) {
            const __half* query_subvector = query + s * index.pq.subspace_dim;
            const __half* centroid = index.pq.centroids +
                                     s * 256 * index.pq.subspace_dim +
                                     c * index.pq.subspace_dim;

            float dist = 0.0f;
            for (int d = 0; d < index.pq.subspace_dim; d++) {
                float diff = __half2float(query_subvector[d]) - __half2float(centroid[d]);
                dist += diff * diff;
            }

            pq_distance_table[s * 256 + c] = dist;
        }
    }
    __syncthreads();

    // Get LSH candidates
    __shared__ int lsh_candidates[4096];
    __shared__ int num_lsh_candidates;
    __shared__ uint32_t seen_bitmap[4096];

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
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
                if (pos < 4096) {
                    lsh_candidates[pos] = item_id;
                }
            }
        }
    }
    __syncthreads();

    // Compute PQ distances for candidates
    __shared__ float candidate_dists[4096];

    for (int i = threadIdx.x; i < num_lsh_candidates; i += blockDim.x) {
        int vec_id = lsh_candidates[i];

        float dist = 0.0f;
        for (int s = 0; s < index.pq.num_subspaces; s++) {
            uint8_t code = index.pq.codes[vec_id * index.pq.num_subspaces + s];
            dist += pq_distance_table[s * 256 + code];
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

    // Dispatch to appropriate search method
    switch (mode) {
        case HybridIndex::EXACT:
            // Use HNSW only
            if (query_id == blockIdx.x) {
                hybrid_search_exact<<<1, 256>>>(
                    index, queries + query_id * index.embedding_dim,
                    results + query_id * k, distances + query_id * k,
                    1, k, 64
                );
            }
            break;

        case HybridIndex::LSH_HNSW:
            // Use LSH + HNSW
            hybrid_search_lsh_hnsw<<<1, 256>>>(
                index, queries + query_id * index.embedding_dim,
                results + query_id * k, distances + query_id * k,
                1, k, 2000
            );
            break;

        case HybridIndex::LSH_PQ:
            // Use LSH + PQ
            hybrid_search_lsh_pq<<<1, 256, index.pq.num_subspaces * 256 * sizeof(float)>>>(
                index, queries + query_id * index.embedding_dim,
                results + query_id * k, distances + query_id * k,
                1, k
            );
            break;

        case HybridIndex::ADAPTIVE:
        default:
            // Default to LSH_HNSW
            hybrid_search_lsh_hnsw<<<1, 256>>>(
                index, queries + query_id * index.embedding_dim,
                results + query_id * k, distances + query_id * k,
                1, k, 2000
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

    // Each thread block processes one query
    for (int query_id = blockIdx.x; query_id < batch_size; query_id += gridDim.x) {
        const __half* query = queries + query_id * index.embedding_dim;
        int* query_results = results + query_id * k;
        float* query_distances = distances + query_id * k;

        // Execute search based on mode
        switch (mode) {
            case HybridIndex::LSH_PQ:
                hybrid_search_lsh_pq<<<1, blockDim.x,
                    index.pq.num_subspaces * 256 * sizeof(float)>>>(
                    index, query, query_results, query_distances, 1, k
                );
                break;

            default:
                hybrid_search_lsh_hnsw<<<1, blockDim.x>>>(
                    index, query, query_results, query_distances, 1, k, 2000
                );
                break;
        }
    }

    grid.sync();
}
