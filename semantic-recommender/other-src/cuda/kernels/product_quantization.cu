#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Product Quantization structure for memory-efficient storage
struct ProductQuantizer {
    __half* centroids;           // [num_subspaces, 256, subspace_dim]
    uint8_t* codes;              // [num_vectors, num_subspaces]
    float* distance_tables;      // Precomputed distance tables
    int num_subspaces;
    int subspace_dim;
    int num_vectors;
    int embedding_dim;

    __device__ float compute_distance_pq(
        const __half* query,
        int vector_id
    ) const {
        float dist = 0.0f;

        // Sum distances across all subspaces
        for (int s = 0; s < num_subspaces; s++) {
            uint8_t code = codes[vector_id * num_subspaces + s];

            // Compute distance to quantized subvector
            int centroid_offset = s * 256 * subspace_dim + code * subspace_dim;

            for (int d = 0; d < subspace_dim; d++) {
                float diff = __half2float(query[s * subspace_dim + d]) -
                            __half2float(centroids[centroid_offset + d]);
                dist += diff * diff;
            }
        }

        return sqrtf(dist);
    }

    __device__ float compute_distance_adt(
        int query_id,
        int vector_id
    ) const {
        // Asymmetric Distance Computation using precomputed table
        float dist = 0.0f;

        const float* table = distance_tables + query_id * num_subspaces * 256;

        for (int s = 0; s < num_subspaces; s++) {
            uint8_t code = codes[vector_id * num_subspaces + s];
            dist += table[s * 256 + code];
        }

        return dist;
    }
};

// Build distance table for asymmetric distance computation
__global__ void pq_build_distance_table(
    ProductQuantizer pq,
    const __half* queries,       // [batch_size, embedding_dim]
    float* distance_tables,      // [batch_size, num_subspaces, 256]
    int batch_size
) {
    int query_id = blockIdx.x;
    int subspace = blockIdx.y;
    int centroid_id = threadIdx.x; // 0-255

    if (query_id >= batch_size || subspace >= pq.num_subspaces || centroid_id >= 256)
        return;

    const __half* query_subvector = queries +
                                     query_id * pq.embedding_dim +
                                     subspace * pq.subspace_dim;

    const __half* centroid = pq.centroids +
                             subspace * 256 * pq.subspace_dim +
                             centroid_id * pq.subspace_dim;

    // Compute distance between query subvector and centroid
    float dist = 0.0f;
    for (int d = 0; d < pq.subspace_dim; d++) {
        float diff = __half2float(query_subvector[d]) - __half2float(centroid[d]);
        dist += diff * diff;
    }

    int table_offset = query_id * pq.num_subspaces * 256 + subspace * 256 + centroid_id;
    distance_tables[table_offset] = dist;
}

// Encode embeddings using product quantization
__global__ void pq_encode_batch(
    ProductQuantizer pq,
    const __half* embeddings,    // [num_items, embedding_dim]
    uint8_t* codes,              // [num_items, num_subspaces]
    int num_items
) {
    int item_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (item_id >= num_items) return;

    const __half* embedding = embeddings + item_id * pq.embedding_dim;

    // Encode each subspace
    for (int s = 0; s < pq.num_subspaces; s++) {
        const __half* subvector = embedding + s * pq.subspace_dim;

        // Find nearest centroid in this subspace
        float min_dist = INFINITY;
        uint8_t best_code = 0;

        for (int c = 0; c < 256; c++) {
            const __half* centroid = pq.centroids +
                                     s * 256 * pq.subspace_dim +
                                     c * pq.subspace_dim;

            float dist = 0.0f;
            for (int d = 0; d < pq.subspace_dim; d++) {
                float diff = __half2float(subvector[d]) - __half2float(centroid[d]);
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_code = c;
            }
        }

        codes[item_id * pq.num_subspaces + s] = best_code;
    }
}

// Search using product quantization with ADT
__global__ void pq_search_batch(
    ProductQuantizer pq,
    const __half* queries,       // [batch_size, embedding_dim]
    int* results,                // [batch_size, k]
    float* distances,            // [batch_size, k]
    int batch_size,
    int k
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    // Build distance table for this query (in shared memory if possible)
    extern __shared__ float shared_distance_table[];

    // Each thread computes distances for subset of centroids
    for (int s = 0; s < pq.num_subspaces; s++) {
        for (int c = threadIdx.x; c < 256; c += blockDim.x) {
            const __half* query_subvector = queries +
                                             query_id * pq.embedding_dim +
                                             s * pq.subspace_dim;

            const __half* centroid = pq.centroids +
                                     s * 256 * pq.subspace_dim +
                                     c * pq.subspace_dim;

            float dist = 0.0f;
            for (int d = 0; d < pq.subspace_dim; d++) {
                float diff = __half2float(query_subvector[d]) - __half2float(centroid[d]);
                dist += diff * diff;
            }

            shared_distance_table[s * 256 + c] = dist;
        }
    }
    __syncthreads();

    // Find k nearest using heap
    __shared__ int heap_ids[256];
    __shared__ float heap_dists[256];
    __shared__ int heap_size;

    if (threadIdx.x == 0) {
        heap_size = 0;
    }
    __syncthreads();

    // Scan all vectors (parallelized)
    for (int vec_id = threadIdx.x; vec_id < pq.num_vectors; vec_id += blockDim.x) {
        // Compute distance using ADT
        float dist = 0.0f;
        for (int s = 0; s < pq.num_subspaces; s++) {
            uint8_t code = pq.codes[vec_id * pq.num_subspaces + s];
            dist += shared_distance_table[s * 256 + code];
        }

        // Insert into heap (requires atomic operations)
        // Simplified: use atomic max to maintain top-k
        if (heap_size < k) {
            int pos = atomicAdd(&heap_size, 1);
            if (pos < k) {
                heap_ids[pos] = vec_id;
                heap_dists[pos] = dist;
            }
        } else {
            // Find max in heap and replace if current is smaller
            float max_dist = heap_dists[0];
            int max_idx = 0;
            for (int i = 1; i < k; i++) {
                if (heap_dists[i] > max_dist) {
                    max_dist = heap_dists[i];
                    max_idx = i;
                }
            }

            if (dist < max_dist) {
                heap_ids[max_idx] = vec_id;
                heap_dists[max_idx] = dist;
            }
        }
    }
    __syncthreads();

    // Sort and write results
    if (threadIdx.x == 0) {
        // Simple bubble sort for small k
        for (int i = 0; i < k - 1; i++) {
            for (int j = 0; j < k - i - 1; j++) {
                if (heap_dists[j] > heap_dists[j + 1]) {
                    float tmp_dist = heap_dists[j];
                    heap_dists[j] = heap_dists[j + 1];
                    heap_dists[j + 1] = tmp_dist;

                    int tmp_id = heap_ids[j];
                    heap_ids[j] = heap_ids[j + 1];
                    heap_ids[j + 1] = tmp_id;
                }
            }
        }

        // Write to output
        for (int i = 0; i < k; i++) {
            results[query_id * k + i] = heap_ids[i];
            distances[query_id * k + i] = sqrtf(heap_dists[i]);
        }
    }
}

// Train product quantizer using k-means on subspaces
__global__ void pq_train_kmeans_iteration(
    const __half* embeddings,    // [num_items, embedding_dim]
    __half* centroids,           // [num_subspaces, 256, subspace_dim]
    uint8_t* assignments,        // [num_items, num_subspaces]
    int* centroid_counts,        // [num_subspaces, 256]
    int num_items,
    int num_subspaces,
    int subspace_dim
) {
    int item_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (item_id >= num_items) return;

    const __half* embedding = embeddings + item_id * num_subspaces * subspace_dim;

    // Assign each subvector to nearest centroid
    for (int s = 0; s < num_subspaces; s++) {
        const __half* subvector = embedding + s * subspace_dim;

        float min_dist = INFINITY;
        uint8_t best_centroid = 0;

        for (int c = 0; c < 256; c++) {
            const __half* centroid = centroids +
                                     s * 256 * subspace_dim +
                                     c * subspace_dim;

            float dist = 0.0f;
            for (int d = 0; d < subspace_dim; d++) {
                float diff = __half2float(subvector[d]) - __half2float(centroid[d]);
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_centroid = c;
            }
        }

        assignments[item_id * num_subspaces + s] = best_centroid;
        atomicAdd(&centroid_counts[s * 256 + best_centroid], 1);
    }
}
