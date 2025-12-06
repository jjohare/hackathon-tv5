/**
 * GPU Graph Search Kernels for Content Discovery - Header
 *
 * Public API for content recommendation graph algorithms
 *
 * @file graph_search.cuh
 * @author Adapted from hybrid SSSP and landmark APSP implementations
 * @version 1.0.0
 */

#ifndef GRAPH_SEARCH_CUH
#define GRAPH_SEARCH_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Configuration for graph search operations
 */
typedef struct {
    int max_hops;              // Maximum path length to explore (typical: 3-5)
    float min_similarity;      // Minimum semantic similarity threshold (0.0-1.0)
    float distance_bound;      // Maximum distance for bounded search
    int num_landmarks;         // Number of landmarks for APSP (typical: sqrt(n))
    float diversity_factor;    // Diversity vs relevance tradeoff (0.0-1.0)
} GraphSearchConfig;

/**
 * Result structure for path queries
 */
typedef struct {
    int* path;                 // Sequence of node IDs in path
    int path_length;           // Number of nodes in path
    float path_cost;           // Total path cost/distance
    float semantic_score;      // Semantic quality score
    float rank_score;          // Final ranking score
} PathResult;

/**
 * Result structure for recommendations
 */
typedef struct {
    int* item_ids;             // Recommended content item IDs
    float* scores;             // Recommendation scores
    int count;                 // Number of recommendations
} RecommendationResult;

// =============================================================================
// Core API Functions
// =============================================================================

/**
 * Launch Single-Source Shortest Path with semantic scoring
 *
 * Computes shortest paths from a source content item to all reachable items,
 * incorporating semantic relevance scoring for content discovery.
 *
 * Use case: "Find all content related to this video within 3 hops"
 *
 * @param source Source content node ID
 * @param distances [OUT] Shortest distances to all nodes [num_nodes]
 * @param predecessors [OUT] Path predecessors for reconstruction [num_nodes]
 * @param semantic_scores [OUT] Semantic relevance scores [num_nodes]
 * @param row_offsets [IN] CSR row offsets [num_nodes + 1]
 * @param col_indices [IN] CSR column indices [num_edges]
 * @param edge_weights [IN] Edge weights (distance/cost) [num_edges]
 * @param content_features [IN] Content similarity features [num_edges]
 * @param user_affinities [IN] User affinity scores [num_nodes]
 * @param frontier [IN] Current frontier nodes [num_nodes]
 * @param frontier_size [IN] Current frontier size
 * @param next_frontier [OUT] Next iteration frontier [num_nodes]
 * @param next_frontier_size [OUT] Next frontier size (atomic counter)
 * @param num_nodes Total number of content nodes
 * @param max_hops Maximum path length (typical: 3-5)
 * @param min_similarity Minimum semantic similarity (typical: 0.5-0.8)
 * @param stream CUDA stream for async execution
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
);

/**
 * Launch landmark selection for APSP approximation
 *
 * Selects diverse, high-degree nodes as landmarks for efficient
 * approximate all-pairs shortest path computation.
 *
 * Use case: "Select hub content items for global distance estimation"
 *
 * @param landmarks [OUT] Selected landmark node IDs [num_landmarks]
 * @param content_clusters [IN] Content cluster assignments [num_nodes]
 * @param node_degrees [IN] Node degree counts [num_nodes]
 * @param num_nodes Total number of nodes
 * @param num_landmarks Number of landmarks (typical: sqrt(num_nodes))
 * @param seed Random seed for sampling
 * @param stream CUDA stream
 */
void launch_select_landmarks(
    int* landmarks,
    const int* content_clusters,
    const int* node_degrees,
    int num_nodes,
    int num_landmarks,
    unsigned long long seed,
    cudaStream_t stream
);

/**
 * Launch approximate all-pairs shortest path computation
 *
 * Computes approximate distances between all content pairs using
 * landmark-based triangle inequality approximation.
 *
 * Use case: "Build global content similarity/distance matrix"
 *
 * Complexity: O(k*n) where k = num_landmarks << n
 * Approximation: d(i,j) ≈ min_k(d(i,k) + d(k,j))
 *
 * @param landmark_distances [IN] Precomputed landmark distances [num_landmarks][num_nodes]
 * @param distance_matrix [OUT] Approximate distance matrix [num_nodes][num_nodes]
 * @param quality_scores [OUT] Approximation quality scores [num_nodes][num_nodes]
 * @param num_nodes Total number of nodes
 * @param num_landmarks Number of landmarks
 * @param stream CUDA stream
 */
void launch_approximate_apsp(
    const float* landmark_distances,
    float* distance_matrix,
    float* quality_scores,
    int num_nodes,
    int num_landmarks,
    cudaStream_t stream
);

/**
 * Launch k-shortest paths computation
 *
 * Finds k alternative shortest paths between two content items,
 * enabling diverse recommendation pathways.
 *
 * Use case: "Show 5 different ways to discover related content"
 *
 * @param source Source content node
 * @param target Target content node
 * @param k Number of paths to find (typical: 3-10)
 * @param distances [IN] Distance matrix [num_nodes][num_nodes]
 * @param predecessors [IN] Predecessor matrix [num_nodes][num_nodes]
 * @param paths [OUT] Path sequences [k][max_path_length]
 * @param path_lengths [OUT] Path lengths [k]
 * @param path_costs [OUT] Path costs [k]
 * @param semantic_scores [OUT] Semantic quality scores [k]
 * @param row_offsets [IN] CSR row offsets
 * @param col_indices [IN] CSR column indices
 * @param edge_weights [IN] Edge weights
 * @param content_features [IN] Content similarity features
 * @param num_nodes Total nodes
 * @param max_path_length Maximum path length
 * @param stream CUDA stream
 */
void launch_k_shortest_paths(
    int source,
    int target,
    int k,
    const float* distances,
    const int* predecessors,
    int* paths,
    int* path_lengths,
    float* path_costs,
    float* semantic_scores,
    const int* row_offsets,
    const int* col_indices,
    const float* edge_weights,
    const float* content_features,
    int num_nodes,
    int max_path_length,
    cudaStream_t stream
);

/**
 * Launch content path filtering and ranking
 *
 * Filters discovered paths by semantic quality and ranks them
 * for final recommendation presentation.
 *
 * Use case: "Select top-quality recommendation paths from candidates"
 *
 * @param paths [IN] Input path sequences [num_paths][max_path_length]
 * @param path_lengths [IN] Path lengths [num_paths]
 * @param path_costs [IN] Path costs [num_paths]
 * @param semantic_scores [IN] Semantic scores [num_paths]
 * @param filtered_paths [OUT] Filtered paths [max_filtered][max_path_length]
 * @param filtered_scores [OUT] Ranking scores [max_filtered]
 * @param num_paths Input path count
 * @param max_path_length Maximum path length
 * @param min_semantic_threshold Minimum quality threshold
 * @param max_filtered Maximum paths to return
 * @param filtered_count [OUT] Count of filtered paths (atomic)
 * @param stream CUDA stream
 */
void launch_filter_content_paths(
    const int* paths,
    const int* path_lengths,
    const float* path_costs,
    const float* semantic_scores,
    int* filtered_paths,
    float* filtered_scores,
    int num_paths,
    int max_path_length,
    float min_semantic_threshold,
    int max_filtered,
    int* filtered_count,
    cudaStream_t stream
);

/**
 * Launch multi-hop content recommendation
 *
 * End-to-end recommendation engine combining path discovery,
 * semantic scoring, and ranking for content suggestions.
 *
 * Use case: "Recommend top-10 related videos for these 5 seed videos"
 *
 * @param source_items [IN] Source content items [num_sources]
 * @param num_sources Number of source items
 * @param recommendations [OUT] Recommended content IDs [num_sources][top_k]
 * @param recommendation_scores [OUT] Recommendation scores [num_sources][top_k]
 * @param distances [IN] Distance matrix [num_nodes][num_nodes]
 * @param semantic_matrix [IN] Semantic similarity matrix [num_nodes][num_nodes]
 * @param user_preferences [IN] User preference vector [num_nodes]
 * @param content_metadata [IN] Content metadata features [num_nodes][feature_dim]
 * @param num_nodes Total content nodes
 * @param top_k Number of recommendations per source
 * @param max_hops Maximum recommendation hops
 * @param diversity_factor Diversity vs relevance (0.0=relevant, 1.0=diverse)
 * @param stream CUDA stream
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
);

/**
 * Launch bounded Dijkstra for localized search
 *
 * Efficient SSSP for exploring local content neighborhoods
 * within a distance bound.
 *
 * Use case: "Find all content within distance 2.0 of this item"
 *
 * @param sources [IN] Source nodes [num_sources]
 * @param num_sources Number of sources
 * @param distances [OUT] Distance array [num_nodes]
 * @param predecessors [OUT] Predecessor array [num_nodes]
 * @param row_offsets [IN] CSR row offsets
 * @param col_indices [IN] CSR column indices
 * @param edge_weights [IN] Edge weights
 * @param distance_bound Maximum exploration distance
 * @param active_vertices [TEMP] Active vertex queue [num_nodes]
 * @param active_count [TEMP] Active count (atomic)
 * @param num_nodes Total nodes
 * @param stream CUDA stream
 */
void launch_bounded_dijkstra(
    const int* sources,
    int num_sources,
    float* distances,
    int* predecessors,
    const int* row_offsets,
    const int* col_indices,
    const float* edge_weights,
    float distance_bound,
    int* active_vertices,
    int* active_count,
    int num_nodes,
    cudaStream_t stream
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Create default configuration
 */
static inline GraphSearchConfig graph_search_default_config() {
    GraphSearchConfig config;
    config.max_hops = 3;
    config.min_similarity = 0.5f;
    config.distance_bound = 10.0f;
    config.num_landmarks = 0;  // Auto-compute as sqrt(n)
    config.diversity_factor = 0.3f;
    return config;
}

/**
 * Compute optimal number of landmarks for graph size
 */
static inline int compute_optimal_landmarks(int num_nodes) {
    // Rule of thumb: k ≈ sqrt(n), but clamp to reasonable range
    int k = (int)sqrtf((float)num_nodes);
    if (k < 10) k = 10;
    if (k > 1000) k = 1000;
    return k;
}

/**
 * Compute optimal grid/block dimensions
 */
static inline void compute_launch_config(
    int num_items,
    int* grid_size,
    int* block_size
) {
    *block_size = (num_items < 256) ? 128 : 256;
    *grid_size = (num_items + *block_size - 1) / *block_size;
    if (*grid_size > 65535) *grid_size = 65535;  // Hardware limit
}

#ifdef __cplusplus
}
#endif

#endif // GRAPH_SEARCH_CUH
