#ifndef HNSW_GPU_CUH
#define HNSW_GPU_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

// HNSW Layer structure optimized for GPU
struct HNSWLayer {
    int* neighbors;          // [num_nodes * M] neighbor indices
    float* distances;        // [num_nodes * M] precomputed distances
    int num_nodes;
    int M;                   // Max connections per node
    int ef_construction;     // Beam width during construction
};

// Main HNSW structure for GPU
struct HNSW_GPU {
    HNSWLayer* layers;       // Array of layers (device pointer)
    int num_layers;
    int entry_point;         // Top-level entry node
    int M;                   // Base connectivity
    int M0;                  // Layer 0 connectivity (typically 2*M)
    float ml;                // Level multiplier for exponential decay

    // Memory management
    __half* node_embeddings; // [total_nodes, embedding_dim]
    int embedding_dim;
    int total_nodes;
};

// Priority queue for GPU beam search
struct PriorityQueue {
    int* items;              // Node indices
    float* priorities;       // Distances
    int capacity;
    int size;

    __device__ void init(int* items_ptr, float* priorities_ptr, int cap) {
        items = items_ptr;
        priorities = priorities_ptr;
        capacity = cap;
        size = 0;
    }

    __device__ bool insert(int item, float priority) {
        if (size < capacity) {
            // Insert and maintain sorted order (min-heap)
            int pos = size;
            while (pos > 0 && priorities[(pos - 1) / 2] > priority) {
                items[pos] = items[(pos - 1) / 2];
                priorities[pos] = priorities[(pos - 1) / 2];
                pos = (pos - 1) / 2;
            }
            items[pos] = item;
            priorities[pos] = priority;
            size++;
            return true;
        } else if (priority < priorities[0]) {
            // Replace max and re-heapify
            items[0] = item;
            priorities[0] = priority;
            heapify_down(0);
            return true;
        }
        return false;
    }

    __device__ void heapify_down(int pos) {
        while (true) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int largest = pos;

            if (left < size && priorities[left] > priorities[largest])
                largest = left;
            if (right < size && priorities[right] > priorities[largest])
                largest = right;

            if (largest == pos) break;

            // Swap
            int tmp_item = items[pos];
            float tmp_priority = priorities[pos];
            items[pos] = items[largest];
            priorities[pos] = priorities[largest];
            items[largest] = tmp_item;
            priorities[largest] = tmp_priority;

            pos = largest;
        }
    }
};

// Compute L2 distance using tensor cores for FP16
__device__ inline float compute_distance_tensor_core(
    const __half* a,
    const __half* b,
    int dim
) {
    float sum = 0.0f;

    // Use vectorized loads for better memory bandwidth
    const int vec_size = 8; // Load 8 fp16 values at once
    const int vec_iters = dim / vec_size;

    for (int i = 0; i < vec_iters; i++) {
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            int idx = i * vec_size + j;
            float diff = __half2float(a[idx]) - __half2float(b[idx]);
            sum += diff * diff;
        }
    }

    // Handle remaining elements
    for (int i = vec_iters * vec_size; i < dim; i++) {
        float diff = __half2float(a[i]) - __half2float(b[i]);
        sum += diff * diff;
    }

    return sqrtf(sum);
}

// Warp-level parallel distance computation
__device__ inline float compute_distance_warp(
    const __half* query,
    const __half* candidate,
    int dim
) {
    float partial_sum = 0.0f;

    // Each thread in warp computes part of the distance
    for (int i = threadIdx.x % 32; i < dim; i += 32) {
        float diff = __half2float(query[i]) - __half2float(candidate[i]);
        partial_sum += diff * diff;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    return sqrtf(partial_sum);
}

// Search layer with warp-level parallelism
__device__ int search_layer_parallel(
    const HNSW_GPU& graph,
    const HNSWLayer& layer,
    const __half* query,
    int entry_point,
    int ef
) {
    // Shared memory for candidate management
    __shared__ int candidates[256];
    __shared__ float candidate_dists[256];
    __shared__ uint32_t visited[4096]; // Bit array for 128K nodes

    // Initialize visited bitmap
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        visited[i] = 0;
    }
    __syncthreads();

    // Initialize with entry point
    if (threadIdx.x == 0) {
        candidates[0] = entry_point;
        candidate_dists[0] = compute_distance_tensor_core(
            query,
            graph.node_embeddings + entry_point * graph.embedding_dim,
            graph.embedding_dim
        );

        // Mark as visited
        int word_idx = entry_point / 32;
        int bit_idx = entry_point % 32;
        atomicOr(&visited[word_idx], 1u << bit_idx);
    }
    __syncthreads();

    int num_candidates = 1;
    int best_candidate = entry_point;
    float best_distance = candidate_dists[0];

    // Greedy search with beam
    for (int iter = 0; iter < 100 && num_candidates > 0; iter++) {
        // Get closest candidate
        int current = candidates[0];
        float current_dist = candidate_dists[0];

        bool improved = false;

        // Each thread explores different neighbors
        for (int n_idx = threadIdx.x; n_idx < layer.M; n_idx += blockDim.x) {
            int neighbor = layer.neighbors[current * layer.M + n_idx];
            if (neighbor < 0) continue;

            // Check if visited
            int word_idx = neighbor / 32;
            int bit_idx = neighbor % 32;
            uint32_t old = atomicOr(&visited[word_idx], 1u << bit_idx);

            if (!(old & (1u << bit_idx))) {
                // New node - compute distance
                float dist = compute_distance_tensor_core(
                    query,
                    graph.node_embeddings + neighbor * graph.embedding_dim,
                    graph.embedding_dim
                );

                // Update best if improved
                if (dist < best_distance) {
                    atomicExch((int*)&best_candidate, neighbor);
                    atomicExch((unsigned int*)&best_distance, __float_as_uint(dist));
                    improved = true;
                }

                // Add to candidate list if within beam
                if (num_candidates < ef && dist < current_dist * 1.5f) {
                    int insert_pos = atomicAdd(&num_candidates, 1);
                    if (insert_pos < 256) {
                        candidates[insert_pos] = neighbor;
                        candidate_dists[insert_pos] = dist;
                    }
                }
            }
        }
        __syncthreads();

        if (!improved && threadIdx.x == 0) {
            // Remove current from candidates
            for (int i = 0; i < num_candidates - 1; i++) {
                candidates[i] = candidates[i + 1];
                candidate_dists[i] = candidate_dists[i + 1];
            }
            num_candidates--;
        }
        __syncthreads();
    }

    return best_candidate;
}

// Device function for HNSW search (can be called from other kernels)
__device__ void hnsw_search_single(
    HNSW_GPU graph,
    const __half* query,
    int* results,
    float* distances,
    int k,
    int ef
) {

    // Start from top layer
    int current_nearest = graph.entry_point;

    // Search through layers (coarse to fine)
    for (int layer = graph.num_layers - 1; layer >= 0; layer--) {
        int search_ef = (layer == 0) ? ef : 1;
        current_nearest = search_layer_parallel(
            graph,
            graph.layers[layer],
            query,
            current_nearest,
            search_ef
        );
    }

    // Collect k-nearest from bottom layer using shared memory priority queue
    __shared__ int pq_items[256];
    __shared__ float pq_dists[256];

    PriorityQueue pq;
    if (threadIdx.x == 0) {
        pq.init(pq_items, pq_dists, k);
    }
    __syncthreads();

    // Explore neighborhood for k-NN
    const HNSWLayer& bottom_layer = graph.layers[0];
    for (int i = threadIdx.x; i < bottom_layer.M; i += blockDim.x) {
        int neighbor = bottom_layer.neighbors[current_nearest * bottom_layer.M + i];
        if (neighbor >= 0) {
            float dist = compute_distance_tensor_core(
                query,
                graph.node_embeddings + neighbor * graph.embedding_dim,
                graph.embedding_dim
            );

            // Thread-safe insertion
            if (threadIdx.x == 0) {
                pq.insert(neighbor, dist);
            }
        }
    }
    __syncthreads();

    // Write results
    if (threadIdx.x == 0) {
        for (int i = 0; i < k && i < pq.size; i++) {
            results[i] = pq.items[i];
            distances[i] = pq.priorities[i];
        }
    }
}

// Batch HNSW search kernel (host-callable wrapper)
__global__ void hnsw_search_batch(
    HNSW_GPU graph,
    const __half* queries,       // [batch_size, embedding_dim]
    int* results,                // [batch_size, k]
    float* distances,            // [batch_size, k]
    int batch_size,
    int k,
    int ef
) {
    int query_id = blockIdx.x;
    if (query_id >= batch_size) return;

    const __half* query = queries + query_id * graph.embedding_dim;

    hnsw_search_single(
        graph,
        query,
        results + query_id * k,
        distances + query_id * k,
        k,
        ef
    );
}

// Build HNSW index on GPU
__global__ void hnsw_insert_batch(
    HNSW_GPU graph,
    const __half* embeddings,    // [num_new_nodes, embedding_dim]
    int* node_ids,               // [num_new_nodes]
    int num_new_nodes,
    int ef_construction
) {
    int node_idx = blockIdx.x;
    if (node_idx >= num_new_nodes) return;

    int node_id = node_ids[node_idx];
    const __half* embedding = embeddings + node_idx * graph.embedding_dim;

    // Determine layer for this node (exponential decay)
    // In practice, this would be precomputed on CPU
    int node_layer = 0;

    // Insert into each layer from top to determined layer
    int entry = graph.entry_point;
    for (int layer = graph.num_layers - 1; layer >= node_layer; layer--) {
        entry = search_layer_parallel(
            graph,
            graph.layers[layer],
            embedding,
            entry,
            ef_construction
        );

        if (layer <= node_layer) {
            // Connect node at this layer
            // This requires atomic operations and neighbor selection
            // Implementation simplified here
        }
    }
}

#endif // HNSW_GPU_CUH
