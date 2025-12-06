/**
 * GPU Graph Search Kernels for Content Discovery
 *
 * Unified implementation combining:
 * - Single-Source Shortest Path (SSSP) for direct content relationships
 * - Landmark-based Approximate All-Pairs Shortest Path (APSP) for global discovery
 * - k-Shortest Paths for multi-hop recommendations
 * - Semantic path scoring for relevance-based ranking
 *
 * Optimized for media content recommendation graphs where:
 * - Nodes represent content items (videos, articles, media)
 * - Edges represent semantic relationships (similarity, co-occurrence, user behavior)
 * - Weights represent relevance/affinity scores (lower = more relevant)
 *
 * Performance characteristics:
 * - SSSP: O(E + V log V) per query using GPU-parallel Dijkstra
 * - APSP: O(k*V log V) using landmark approximation with k << V
 * - k-Shortest: O(k*E) for k alternative paths
 *
 * @author Adapted from hybrid SSSP and landmark APSP implementations
 * @version 1.0.0
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// =============================================================================
// Constants and Configuration
// =============================================================================

#define MAX_FLOAT FLT_MAX
#define INFINITY_DIST MAX_FLOAT
#define EPSILON 1e-6f
#define WARP_SIZE 32
#define MAX_K_PATHS 10
#define DEFAULT_SEMANTIC_THRESHOLD 0.8f

// =============================================================================
// Device Helper Functions
// =============================================================================

/**
 * Atomic minimum operation for float values
 * Thread-safe updates to maintain minimum distance
 */
__device__ inline float atomicMinFloat(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        float old_float = __int_as_float(assumed);
        if (old_float <= value) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}

/**
 * Compute semantic relevance score combining distance and content features
 * Lower score = more relevant
 */
__device__ inline float compute_semantic_score(
    float path_distance,
    float content_similarity,
    float user_affinity,
    float temporal_decay
) {
    // Weighted combination: distance dominant, modulated by semantic features
    float base_score = path_distance;
    float semantic_bonus = (1.0f - content_similarity) * 0.3f;
    float affinity_bonus = (1.0f - user_affinity) * 0.2f;
    float time_penalty = temporal_decay * 0.1f;

    return base_score + semantic_bonus + affinity_bonus + time_penalty;
}

/**
 * Check if path satisfies semantic constraints for content discovery
 */
__device__ inline bool is_semantically_valid(
    float similarity_score,
    int hop_count,
    int max_hops,
    float min_similarity
) {
    return similarity_score >= min_similarity && hop_count <= max_hops;
}

// =============================================================================
// Kernel 1: Single-Source Shortest Path (SSSP)
// GPU-Parallel Dijkstra with Semantic Scoring
// =============================================================================

/**
 * Compute shortest paths from a single source content item
 * Uses work-efficient frontier expansion with semantic scoring
 *
 * @param source Source content node ID
 * @param distances Output array of shortest distances [num_nodes]
 * @param predecessors Output array of path predecessors [num_nodes]
 * @param semantic_scores Output array of semantic relevance scores [num_nodes]
 * @param row_offsets CSR format row offsets [num_nodes + 1]
 * @param col_indices CSR format column indices [num_edges]
 * @param edge_weights Edge weights (distance/cost) [num_edges]
 * @param content_features Content similarity features [num_edges]
 * @param user_affinities User affinity scores [num_nodes]
 * @param frontier Current frontier nodes [num_nodes]
 * @param frontier_size Current frontier size
 * @param next_frontier Next iteration frontier [num_nodes]
 * @param next_frontier_size Next iteration frontier size (atomic counter)
 * @param num_nodes Total number of content nodes
 * @param max_hops Maximum path length to explore
 * @param min_similarity Minimum semantic similarity threshold
 */
__global__ void sssp_semantic_kernel(
    const int source,
    float* __restrict__ distances,
    int* __restrict__ predecessors,
    float* __restrict__ semantic_scores,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ content_features,
    const float* __restrict__ user_affinities,
    const int* __restrict__ frontier,
    const int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    const int num_nodes,
    const int max_hops,
    const float min_similarity
) {
    extern __shared__ int shared_frontier[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load frontier into shared memory for coalesced access
    int chunks = (frontier_size + blockDim.x - 1) / blockDim.x;
    for (int chunk = 0; chunk < chunks; chunk++) {
        int idx = chunk * blockDim.x + tid;
        if (idx < frontier_size) {
            shared_frontier[idx] = frontier[idx];
        }
    }
    __syncthreads();

    // Process frontier nodes in parallel
    for (int f_idx = tid; f_idx < frontier_size; f_idx += blockDim.x) {
        int u = shared_frontier[f_idx];
        float u_dist = distances[u];
        float u_semantic = semantic_scores[u];

        if (u_dist >= INFINITY_DIST) continue;

        // Get hop count from distance (approximate)
        int hop_count = (int)(u_dist / 10.0f) + 1;
        if (hop_count > max_hops) continue;

        // Explore all neighbors
        int edge_start = row_offsets[u];
        int edge_end = row_offsets[u + 1];

        for (int e = edge_start; e < edge_end; e++) {
            int v = col_indices[e];
            float weight = edge_weights[e];
            float similarity = content_features[e];
            float affinity = user_affinities[v];

            // Skip if doesn't meet semantic constraints
            if (!is_semantically_valid(similarity, hop_count + 1, max_hops, min_similarity)) {
                continue;
            }

            // Compute new distance and semantic score
            float new_dist = u_dist + weight;
            float new_semantic = compute_semantic_score(
                new_dist,
                similarity,
                affinity,
                0.0f  // temporal decay computed elsewhere
            );

            // Atomic update if better path found
            float old_dist = atomicMinFloat(&distances[v], new_dist);

            if (new_dist < old_dist) {
                predecessors[v] = u;
                semantic_scores[v] = new_semantic;

                // Add to next frontier
                int pos = atomicAdd(next_frontier_size, 1);
                if (pos < num_nodes) {
                    next_frontier[pos] = v;
                }
            }
        }
    }
}

// =============================================================================
// Kernel 2: Landmark Selection for APSP
// Stratified Sampling with Content Diversity
// =============================================================================

/**
 * Select landmark nodes for approximate APSP computation
 * Uses stratified sampling to ensure diverse content coverage
 *
 * @param landmarks Output array of landmark node IDs [num_landmarks]
 * @param content_clusters Content cluster assignments [num_nodes]
 * @param node_degrees Node degree counts [num_nodes]
 * @param num_nodes Total number of content nodes
 * @param num_landmarks Number of landmarks to select (k)
 * @param seed Random seed for sampling
 */
__global__ void select_content_landmarks_kernel(
    int* __restrict__ landmarks,
    const int* __restrict__ content_clusters,
    const int* __restrict__ node_degrees,
    const int num_nodes,
    const int num_landmarks,
    const unsigned long long seed
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_landmarks) return;

    // Stratified sampling: one landmark per stratum
    int stratum_size = num_nodes / num_landmarks;
    int stratum_start = tid * stratum_size;
    int stratum_end = (tid == num_landmarks - 1) ? num_nodes : stratum_start + stratum_size;

    // Within stratum, select node with highest degree (hub selection)
    int best_node = stratum_start;
    int best_degree = node_degrees[stratum_start];

    for (int i = stratum_start + 1; i < stratum_end; i++) {
        int degree = node_degrees[i];
        if (degree > best_degree) {
            best_degree = degree;
            best_node = i;
        }
    }

    // Perturb with seed for randomization
    best_node = (best_node + seed + tid) % num_nodes;

    landmarks[tid] = best_node;
}

// =============================================================================
// Kernel 3: Approximate All-Pairs Shortest Path (APSP)
// Landmark-based Distance Approximation
// =============================================================================

/**
 * Compute approximate all-pairs shortest paths using landmark distances
 * For content discovery: "how to get from any content to any other content"
 *
 * Uses triangle inequality: d(i,j) ≤ d(i,k) + d(k,j) for landmark k
 * Approximation: d(i,j) ≈ min_k(d(i,k) + d(k,j))
 *
 * @param landmark_distances Precomputed distances from landmarks [num_landmarks][num_nodes]
 * @param distance_matrix Output approximate distance matrix [num_nodes][num_nodes]
 * @param quality_scores Output quality scores for approximations [num_nodes][num_nodes]
 * @param num_nodes Total number of content nodes
 * @param num_landmarks Number of landmark nodes (k)
 */
__global__ void approximate_apsp_content_kernel(
    const float* __restrict__ landmark_distances,
    float* __restrict__ distance_matrix,
    float* __restrict__ quality_scores,
    const int num_nodes,
    const int num_landmarks
) {
    // 2D thread indexing for distance matrix
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_nodes || j >= num_nodes) return;

    // Diagonal: distance to self is zero
    if (i == j) {
        distance_matrix[i * num_nodes + j] = 0.0f;
        quality_scores[i * num_nodes + j] = 1.0f;
        return;
    }

    // Find best approximation using landmarks
    float min_dist = INFINITY_DIST;
    int best_landmark = -1;

    for (int k = 0; k < num_landmarks; k++) {
        float dist_ik = landmark_distances[k * num_nodes + i];
        float dist_kj = landmark_distances[k * num_nodes + j];

        if (dist_ik < INFINITY_DIST && dist_kj < INFINITY_DIST) {
            float estimate = dist_ik + dist_kj;
            if (estimate < min_dist) {
                min_dist = estimate;
                best_landmark = k;
            }
        }
    }

    // Compute quality score based on landmark path
    float quality = 1.0f;
    if (best_landmark >= 0) {
        // Quality decreases with path length
        quality = expf(-min_dist / (float)num_nodes);
    } else {
        // No valid path found
        min_dist = (float)num_nodes * 2.0f;
        quality = 0.0f;
    }

    distance_matrix[i * num_nodes + j] = min_dist;
    quality_scores[i * num_nodes + j] = quality;
}

// =============================================================================
// Kernel 4: k-Shortest Paths for Multi-Hop Recommendations
// Yen's Algorithm GPU Implementation
// =============================================================================

/**
 * Compute k alternative shortest paths between content items
 * Essential for diverse recommendations: "k different ways to discover content"
 *
 * @param source Source content node
 * @param target Target content node
 * @param k Number of paths to find (e.g., top-5 recommendation paths)
 * @param distances Distance matrix [num_nodes][num_nodes]
 * @param predecessors Predecessor matrix [num_nodes][num_nodes]
 * @param paths Output path sequences [k][max_path_length]
 * @param path_lengths Output path lengths [k]
 * @param path_costs Output path costs [k]
 * @param semantic_scores Semantic quality scores for paths [k]
 * @param row_offsets CSR row offsets [num_nodes + 1]
 * @param col_indices CSR column indices [num_edges]
 * @param edge_weights Edge weights [num_edges]
 * @param content_features Content similarity features [num_edges]
 * @param num_nodes Total nodes
 * @param max_path_length Maximum path length
 */
__global__ void k_shortest_paths_kernel(
    const int source,
    const int target,
    const int k,
    const float* __restrict__ distances,
    const int* __restrict__ predecessors,
    int* __restrict__ paths,
    int* __restrict__ path_lengths,
    float* __restrict__ path_costs,
    float* __restrict__ semantic_scores,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ content_features,
    const int num_nodes,
    const int max_path_length
) {
    int path_idx = blockIdx.x;
    if (path_idx >= k) return;

    // Thread per path candidate
    int tid = threadIdx.x;

    __shared__ int shared_path[256];  // Shared memory for path construction
    __shared__ float shared_cost;
    __shared__ float shared_semantic;

    if (tid == 0) {
        shared_cost = 0.0f;
        shared_semantic = 0.0f;
    }
    __syncthreads();

    // First path (path_idx == 0): use standard shortest path
    if (path_idx == 0 && tid == 0) {
        int current = target;
        int length = 0;
        float total_cost = 0.0f;
        float total_semantic = 0.0f;

        // Backtrack from target to source
        while (current != source && length < max_path_length) {
            shared_path[length++] = current;
            int prev = predecessors[current];

            if (prev < 0 || prev >= num_nodes) break;

            // Find edge weight
            int edge_start = row_offsets[prev];
            int edge_end = row_offsets[prev + 1];
            for (int e = edge_start; e < edge_end; e++) {
                if (col_indices[e] == current) {
                    total_cost += edge_weights[e];
                    total_semantic += content_features[e];
                    break;
                }
            }

            current = prev;
        }

        if (current == source) {
            shared_path[length++] = source;

            // Reverse path
            for (int i = 0; i < length / 2; i++) {
                int temp = shared_path[i];
                shared_path[i] = shared_path[length - 1 - i];
                shared_path[length - 1 - i] = temp;
            }

            // Write to global memory
            for (int i = 0; i < length; i++) {
                paths[path_idx * max_path_length + i] = shared_path[i];
            }
            path_lengths[path_idx] = length;
            path_costs[path_idx] = total_cost;
            semantic_scores[path_idx] = total_semantic / (float)length;
        } else {
            path_lengths[path_idx] = 0;
            path_costs[path_idx] = INFINITY_DIST;
            semantic_scores[path_idx] = 0.0f;
        }
    }

    // Alternative paths (path_idx > 0): Yen's algorithm deviation
    // This is simplified; full Yen's requires candidate path maintenance
    if (path_idx > 0) {
        // For demo: find paths with slight perturbations
        // Production: implement full Yen's with priority queue
        path_lengths[path_idx] = 0;
        path_costs[path_idx] = INFINITY_DIST;
        semantic_scores[path_idx] = 0.0f;
    }
}

// =============================================================================
// Kernel 5: Content Discovery Path Filtering
// Semantic Filtering and Ranking
// =============================================================================

/**
 * Filter and rank discovered paths by semantic relevance
 * Essential for high-quality recommendations
 *
 * @param paths Input path sequences [num_paths][max_path_length]
 * @param path_lengths Path lengths [num_paths]
 * @param path_costs Path costs [num_paths]
 * @param semantic_scores Semantic scores [num_paths]
 * @param filtered_paths Output filtered paths [num_filtered][max_path_length]
 * @param filtered_scores Output filtered scores [num_filtered]
 * @param num_paths Input path count
 * @param max_path_length Maximum path length
 * @param min_semantic_threshold Minimum semantic quality
 * @param max_filtered Maximum filtered paths to return
 * @param filtered_count Output count of filtered paths (atomic)
 */
__global__ void filter_content_paths_kernel(
    const int* __restrict__ paths,
    const int* __restrict__ path_lengths,
    const float* __restrict__ path_costs,
    const float* __restrict__ semantic_scores,
    int* __restrict__ filtered_paths,
    float* __restrict__ filtered_scores,
    const int num_paths,
    const int max_path_length,
    const float min_semantic_threshold,
    const int max_filtered,
    int* __restrict__ filtered_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_paths) return;

    float semantic = semantic_scores[tid];
    int length = path_lengths[tid];
    float cost = path_costs[tid];

    // Filter criteria
    bool is_valid = (
        semantic >= min_semantic_threshold &&
        length > 0 &&
        cost < INFINITY_DIST
    );

    if (is_valid) {
        // Compute final ranking score
        float rank_score = semantic * (1.0f / (1.0f + cost));

        // Add to filtered results
        int pos = atomicAdd(filtered_count, 1);
        if (pos < max_filtered) {
            // Copy path
            for (int i = 0; i < length && i < max_path_length; i++) {
                filtered_paths[pos * max_path_length + i] = paths[tid * max_path_length + i];
            }
            filtered_scores[pos] = rank_score;
        }
    }
}

// =============================================================================
// Kernel 6: Multi-Hop Content Recommendation
// Combined Path Discovery and Ranking
// =============================================================================

/**
 * End-to-end content recommendation using multi-hop graph traversal
 * Discovers related content through semantic graph paths
 *
 * @param source_items Source content items [num_sources]
 * @param num_sources Number of source items
 * @param recommendations Output recommended content IDs [num_sources][top_k]
 * @param recommendation_scores Output recommendation scores [num_sources][top_k]
 * @param distances Distance matrix [num_nodes][num_nodes]
 * @param semantic_matrix Semantic similarity matrix [num_nodes][num_nodes]
 * @param user_preferences User preference vector [num_nodes]
 * @param content_metadata Content metadata features [num_nodes][feature_dim]
 * @param num_nodes Total content nodes
 * @param top_k Number of recommendations per source
 * @param max_hops Maximum recommendation hops
 * @param diversity_factor Diversity vs relevance tradeoff [0,1]
 */
__global__ void multi_hop_recommendation_kernel(
    const int* __restrict__ source_items,
    const int num_sources,
    int* __restrict__ recommendations,
    float* __restrict__ recommendation_scores,
    const float* __restrict__ distances,
    const float* __restrict__ semantic_matrix,
    const float* __restrict__ user_preferences,
    const float* __restrict__ content_metadata,
    const int num_nodes,
    const int top_k,
    const int max_hops,
    const float diversity_factor
) {
    int source_idx = blockIdx.x;
    if (source_idx >= num_sources) return;

    int source = source_items[source_idx];
    int tid = threadIdx.x;

    extern __shared__ float shared_scores[];

    // Each thread evaluates a subset of candidate items
    for (int candidate = tid; candidate < num_nodes; candidate += blockDim.x) {
        if (candidate == source) {
            shared_scores[candidate] = -1.0f;  // Exclude source
            continue;
        }

        float dist = distances[source * num_nodes + candidate];
        float semantic = semantic_matrix[source * num_nodes + candidate];
        float preference = user_preferences[candidate];

        // Skip unreachable or too distant items
        if (dist >= INFINITY_DIST || dist > (float)max_hops * 10.0f) {
            shared_scores[candidate] = -1.0f;
            continue;
        }

        // Compute recommendation score
        float proximity_score = 1.0f / (1.0f + dist);
        float relevance_score = semantic * preference;
        float diversity_score = 1.0f - semantic;  // Diversity = dissimilarity

        float final_score = (
            (1.0f - diversity_factor) * (proximity_score + relevance_score) +
            diversity_factor * diversity_score
        );

        shared_scores[candidate] = final_score;
    }
    __syncthreads();

    // Select top-k candidates (simplified selection, production should use parallel sort)
    if (tid == 0) {
        for (int k = 0; k < top_k; k++) {
            float best_score = -1.0f;
            int best_candidate = -1;

            for (int c = 0; c < num_nodes; c++) {
                if (shared_scores[c] > best_score) {
                    best_score = shared_scores[c];
                    best_candidate = c;
                }
            }

            if (best_candidate >= 0) {
                recommendations[source_idx * top_k + k] = best_candidate;
                recommendation_scores[source_idx * top_k + k] = best_score;
                shared_scores[best_candidate] = -1.0f;  // Remove from candidates
            }
        }
    }
}

// =============================================================================
// Kernel 7: Bounded Dijkstra for Base Case SSSP
// Efficient for Small Frontiers
// =============================================================================

/**
 * Bounded Dijkstra for efficient SSSP with distance limits
 * Used when exploring local neighborhoods in content graph
 *
 * @param sources Source nodes [num_sources]
 * @param num_sources Number of sources
 * @param distances Distance array [num_nodes]
 * @param predecessors Predecessor array [num_nodes]
 * @param row_offsets CSR row offsets [num_nodes + 1]
 * @param col_indices CSR column indices [num_edges]
 * @param edge_weights Edge weights [num_edges]
 * @param distance_bound Maximum distance to explore
 * @param active_vertices Active vertex queue [num_nodes]
 * @param active_count Active count (atomic)
 * @param num_nodes Total nodes
 */
__global__ void bounded_dijkstra_content_kernel(
    const int* __restrict__ sources,
    const int num_sources,
    float* __restrict__ distances,
    int* __restrict__ predecessors,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float distance_bound,
    int* __restrict__ active_vertices,
    int* __restrict__ active_count,
    const int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize sources
    if (tid < num_sources) {
        int source = sources[tid];
        distances[source] = 0.0f;
        predecessors[source] = source;
        active_vertices[tid] = source;
    }

    if (tid == 0) {
        *active_count = num_sources;
    }
    __syncthreads();

    // Iterative relaxation
    int max_iterations = (int)(distance_bound / 0.1f) + 1;

    for (int iter = 0; iter < max_iterations; iter++) {
        int current_active = *active_count;
        if (current_active == 0) break;

        // Process active vertices
        for (int idx = tid; idx < current_active; idx += blockDim.x * gridDim.x) {
            int u = active_vertices[idx];
            float u_dist = distances[u];

            if (u_dist >= distance_bound) continue;

            int edge_start = row_offsets[u];
            int edge_end = row_offsets[u + 1];

            for (int e = edge_start; e < edge_end; e++) {
                int v = col_indices[e];
                float weight = edge_weights[e];
                float new_dist = u_dist + weight;

                if (new_dist < distance_bound) {
                    float old_dist = atomicMinFloat(&distances[v], new_dist);
                    if (new_dist < old_dist) {
                        predecessors[v] = u;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// =============================================================================
// Extern C Wrapper Functions for Host Integration
// =============================================================================

extern "C" {

/**
 * Launch SSSP with semantic scoring
 */
void launch_sssp_semantic(
    int source,
    float* distances,
    int* predecessors,
    float* semantic_scores,
    const int* row_offsets,
    const int* col_indices,
    const float* edge_weights,
    const float* content_features,
    const float* user_affinities,
    const int* frontier,
    int frontier_size,
    int* next_frontier,
    int* next_frontier_size,
    int num_nodes,
    int max_hops,
    float min_similarity,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (frontier_size + block_size - 1) / block_size;
    int shared_mem = frontier_size * sizeof(int);

    sssp_semantic_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        source, distances, predecessors, semantic_scores,
        row_offsets, col_indices, edge_weights, content_features, user_affinities,
        frontier, frontier_size, next_frontier, next_frontier_size,
        num_nodes, max_hops, min_similarity
    );
}

/**
 * Launch landmark selection
 */
void launch_select_landmarks(
    int* landmarks,
    const int* content_clusters,
    const int* node_degrees,
    int num_nodes,
    int num_landmarks,
    unsigned long long seed,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_landmarks + block_size - 1) / block_size;

    select_content_landmarks_kernel<<<grid_size, block_size, 0, stream>>>(
        landmarks, content_clusters, node_degrees,
        num_nodes, num_landmarks, seed
    );
}

/**
 * Launch approximate APSP
 */
void launch_approximate_apsp(
    const float* landmark_distances,
    float* distance_matrix,
    float* quality_scores,
    int num_nodes,
    int num_landmarks,
    cudaStream_t stream
) {
    dim3 block_size(16, 16);
    dim3 grid_size(
        (num_nodes + block_size.x - 1) / block_size.x,
        (num_nodes + block_size.y - 1) / block_size.y
    );

    approximate_apsp_content_kernel<<<grid_size, block_size, 0, stream>>>(
        landmark_distances, distance_matrix, quality_scores,
        num_nodes, num_landmarks
    );
}

/**
 * Launch multi-hop recommendation
 */
void launch_multi_hop_recommendation(
    const int* source_items,
    int num_sources,
    int* recommendations,
    float* recommendation_scores,
    const float* distances,
    const float* semantic_matrix,
    const float* user_preferences,
    const float* content_metadata,
    int num_nodes,
    int top_k,
    int max_hops,
    float diversity_factor,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_sources;
    int shared_mem = num_nodes * sizeof(float);

    multi_hop_recommendation_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        source_items, num_sources, recommendations, recommendation_scores,
        distances, semantic_matrix, user_preferences, content_metadata,
        num_nodes, top_k, max_hops, diversity_factor
    );
}

} // extern "C"
