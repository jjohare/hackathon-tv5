/**
 * Example: Content Discovery using GPU Graph Search Kernels
 *
 * Demonstrates end-to-end content recommendation pipeline:
 * 1. Load content graph (videos, articles, media items)
 * 2. Compute recommendations using SSSP and semantic scoring
 * 3. Build global similarity matrix using landmark APSP
 * 4. Multi-hop recommendations with diversity control
 *
 * Compile:
 * nvcc -o graph_search_example graph_search_example.cu ../kernels/graph_search.cu -I../kernels -lcudart
 *
 * Run:
 * ./graph_search_example
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include "graph_search.cuh"

// =============================================================================
// Utility Functions
// =============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Generate synthetic content graph for testing
 */
void generate_test_graph(
    int num_nodes,
    int avg_degree,
    std::vector<int>& row_offsets,
    std::vector<int>& col_indices,
    std::vector<float>& edge_weights,
    std::vector<float>& content_features
) {
    row_offsets.resize(num_nodes + 1);
    row_offsets[0] = 0;

    for (int i = 0; i < num_nodes; i++) {
        // Random degree around average
        int degree = avg_degree + (rand() % (avg_degree / 2)) - (avg_degree / 4);
        degree = std::max(1, std::min(degree, num_nodes - 1));

        for (int d = 0; d < degree; d++) {
            // Random neighbor
            int neighbor = rand() % num_nodes;
            if (neighbor == i) continue;  // No self-loops

            col_indices.push_back(neighbor);

            // Random edge weight (distance): [0.1, 5.0]
            float weight = 0.1f + (rand() / (float)RAND_MAX) * 4.9f;
            edge_weights.push_back(weight);

            // Random semantic similarity: [0.3, 1.0]
            float similarity = 0.3f + (rand() / (float)RAND_MAX) * 0.7f;
            content_features.push_back(similarity);
        }

        row_offsets[i + 1] = col_indices.size();
    }

    printf("Generated graph: %d nodes, %d edges, avg degree: %.1f\n",
           num_nodes, (int)col_indices.size(), col_indices.size() / (float)num_nodes);
}

/**
 * Generate random user affinity scores
 */
void generate_user_affinities(int num_nodes, std::vector<float>& affinities) {
    affinities.resize(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        affinities[i] = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
    }
}

/**
 * Print top-k recommendations
 */
void print_recommendations(
    int source,
    const std::vector<int>& items,
    const std::vector<float>& scores,
    int top_k,
    const char* label
) {
    printf("\n%s from content item %d:\n", label, source);
    for (int i = 0; i < top_k && i < items.size(); i++) {
        printf("  %2d. Item %4d (score: %.4f)\n", i+1, items[i], scores[i]);
    }
}

// =============================================================================
// Example 1: Single-Source Recommendations
// =============================================================================

void example_sssp_recommendations(
    int num_nodes,
    const int* d_row_offsets,
    const int* d_col_indices,
    const float* d_edge_weights,
    const float* d_content_features,
    const float* d_user_affinities,
    int source_item
) {
    printf("\n=== Example 1: SSSP-based Recommendations ===\n");

    // Allocate device memory
    float *d_distances, *d_semantic_scores;
    int *d_predecessors, *d_frontier, *d_next_frontier, *d_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_distances, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_semantic_scores, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predecessors, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));

    // Initialize distances to infinity
    float* h_distances = new float[num_nodes];
    std::fill(h_distances, h_distances + num_nodes, FLT_MAX);
    h_distances[source_item] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_distances, h_distances, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_semantic_scores, 0, num_nodes * sizeof(float)));

    // Initialize frontier with source
    int h_frontier[1] = {source_item};
    CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, sizeof(int), cudaMemcpyHostToDevice));
    int frontier_size = 1;

    // Iterative SSSP with semantic scoring
    int max_hops = 3;
    float min_similarity = 0.5f;

    for (int hop = 0; hop < max_hops && frontier_size > 0; hop++) {
        // Reset next frontier size
        CUDA_CHECK(cudaMemset(d_frontier_size, 0, sizeof(int)));

        // Launch SSSP kernel
        launch_sssp_semantic(
            source_item,
            d_distances,
            d_predecessors,
            d_semantic_scores,
            d_row_offsets,
            d_col_indices,
            d_edge_weights,
            d_content_features,
            d_user_affinities,
            d_frontier,
            frontier_size,
            d_next_frontier,
            d_frontier_size,
            num_nodes,
            max_hops,
            min_similarity,
            0  // default stream
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Get next frontier size
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap frontiers
        std::swap(d_frontier, d_next_frontier);

        printf("Hop %d: Explored %d nodes\n", hop + 1, frontier_size);
    }

    // Retrieve results
    float* h_semantic = new float[num_nodes];
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_semantic, d_semantic_scores, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));

    // Extract top-20 recommendations
    std::vector<std::pair<int, float>> recommendations;
    for (int i = 0; i < num_nodes; i++) {
        if (i != source_item && h_distances[i] < FLT_MAX) {
            recommendations.push_back({i, h_semantic[i]});
        }
    }

    std::sort(recommendations.begin(), recommendations.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Print results
    std::vector<int> items;
    std::vector<float> scores;
    for (int i = 0; i < 20 && i < recommendations.size(); i++) {
        items.push_back(recommendations[i].first);
        scores.push_back(recommendations[i].second);
    }
    print_recommendations(source_item, items, scores, 20, "SSSP Recommendations");

    printf("Total reachable items: %zu\n", recommendations.size());

    // Cleanup
    delete[] h_distances;
    delete[] h_semantic;
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_semantic_scores));
    CUDA_CHECK(cudaFree(d_predecessors));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_frontier_size));
}

// =============================================================================
// Example 2: Landmark APSP for Global Similarity
// =============================================================================

void example_landmark_apsp(
    int num_nodes,
    const int* d_row_offsets,
    const int* d_col_indices,
    const float* d_edge_weights,
    const float* d_content_features,
    const float* d_user_affinities,
    const int* h_row_offsets
) {
    printf("\n=== Example 2: Landmark APSP ===\n");

    // Compute optimal landmarks
    int num_landmarks = compute_optimal_landmarks(num_nodes);
    printf("Using %d landmarks for %d nodes\n", num_landmarks, num_nodes);

    // Allocate device memory
    int *d_landmarks, *d_degrees;
    float *d_landmark_distances, *d_distance_matrix, *d_quality;

    CUDA_CHECK(cudaMalloc(&d_landmarks, num_landmarks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_degrees, num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_landmark_distances, num_landmarks * num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distance_matrix, num_nodes * num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quality, num_nodes * num_nodes * sizeof(float)));

    // Compute node degrees
    int* h_degrees = new int[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
        h_degrees[i] = h_row_offsets[i + 1] - h_row_offsets[i];
    }
    CUDA_CHECK(cudaMemcpy(d_degrees, h_degrees, num_nodes * sizeof(int), cudaMemcpyHostToDevice));

    // Select landmarks
    launch_select_landmarks(
        d_landmarks,
        nullptr,  // No cluster info in this example
        d_degrees,
        num_nodes,
        num_landmarks,
        12345ULL,  // seed
        0
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_landmarks = new int[num_landmarks];
    CUDA_CHECK(cudaMemcpy(h_landmarks, d_landmarks, num_landmarks * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Selected landmarks: ");
    for (int i = 0; i < std::min(10, num_landmarks); i++) {
        printf("%d ", h_landmarks[i]);
    }
    printf("%s\n", num_landmarks > 10 ? "..." : "");

    // Compute SSSP from each landmark (simplified - would reuse SSSP kernel)
    // For demo, just fill with random distances
    float* h_landmark_dist = new float[num_landmarks * num_nodes];
    for (int k = 0; k < num_landmarks; k++) {
        for (int i = 0; i < num_nodes; i++) {
            if (i == h_landmarks[k]) {
                h_landmark_dist[k * num_nodes + i] = 0.0f;
            } else {
                h_landmark_dist[k * num_nodes + i] = 1.0f + (rand() / (float)RAND_MAX) * 10.0f;
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(d_landmark_distances, h_landmark_dist,
                          num_landmarks * num_nodes * sizeof(float), cudaMemcpyHostToDevice));

    // Compute approximate APSP
    launch_approximate_apsp(
        d_landmark_distances,
        d_distance_matrix,
        d_quality,
        num_nodes,
        num_landmarks,
        0
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("APSP computation complete\n");

    // Sample some distances
    float* h_distance_matrix = new float[num_nodes * num_nodes];
    CUDA_CHECK(cudaMemcpy(h_distance_matrix, d_distance_matrix,
                          num_nodes * num_nodes * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nSample distances:\n");
    for (int i = 0; i < std::min(5, num_nodes); i++) {
        for (int j = 0; j < std::min(5, num_nodes); j++) {
            printf("d(%d,%d)=%.2f ", i, j, h_distance_matrix[i * num_nodes + j]);
        }
        printf("\n");
    }

    // Cleanup
    delete[] h_degrees;
    delete[] h_landmarks;
    delete[] h_landmark_dist;
    delete[] h_distance_matrix;
    CUDA_CHECK(cudaFree(d_landmarks));
    CUDA_CHECK(cudaFree(d_degrees));
    CUDA_CHECK(cudaFree(d_landmark_distances));
    CUDA_CHECK(cudaFree(d_distance_matrix));
    CUDA_CHECK(cudaFree(d_quality));
}

// =============================================================================
// Example 3: Multi-Hop Batch Recommendations
// =============================================================================

void example_multi_hop_batch(
    int num_nodes,
    const float* d_distance_matrix,
    const float* d_user_affinities
) {
    printf("\n=== Example 3: Multi-Hop Batch Recommendations ===\n");

    // Setup batch query
    int num_queries = 5;
    int top_k = 10;
    int* h_sources = new int[num_queries];
    for (int i = 0; i < num_queries; i++) {
        h_sources[i] = rand() % num_nodes;
    }

    printf("Processing %d queries for top-%d recommendations\n", num_queries, top_k);

    // Allocate device memory
    int *d_sources, *d_recommendations;
    float *d_scores, *d_semantic_matrix, *d_metadata;

    CUDA_CHECK(cudaMalloc(&d_sources, num_queries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recommendations, num_queries * top_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scores, num_queries * top_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_semantic_matrix, num_nodes * num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_metadata, num_nodes * 16 * sizeof(float)));  // 16 features

    CUDA_CHECK(cudaMemcpy(d_sources, h_sources, num_queries * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize semantic matrix (simplified - would compute from features)
    float* h_semantic = new float[num_nodes * num_nodes];
    for (int i = 0; i < num_nodes * num_nodes; i++) {
        h_semantic[i] = 0.3f + (rand() / (float)RAND_MAX) * 0.7f;
    }
    CUDA_CHECK(cudaMemcpy(d_semantic_matrix, h_semantic,
                          num_nodes * num_nodes * sizeof(float), cudaMemcpyHostToDevice));

    // Launch batch recommendations
    launch_multi_hop_recommendation(
        d_sources,
        num_queries,
        d_recommendations,
        d_scores,
        d_distance_matrix,
        d_semantic_matrix,
        d_user_affinities,
        d_metadata,
        num_nodes,
        top_k,
        3,      // max_hops
        0.3f,   // diversity_factor
        0
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve results
    int* h_recommendations = new int[num_queries * top_k];
    float* h_scores = new float[num_queries * top_k];
    CUDA_CHECK(cudaMemcpy(h_recommendations, d_recommendations,
                          num_queries * top_k * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores,
                          num_queries * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    for (int q = 0; q < num_queries; q++) {
        std::vector<int> items;
        std::vector<float> scores;
        for (int k = 0; k < top_k; k++) {
            int idx = q * top_k + k;
            items.push_back(h_recommendations[idx]);
            scores.push_back(h_scores[idx]);
        }
        char label[128];
        snprintf(label, sizeof(label), "Multi-hop recommendations");
        print_recommendations(h_sources[q], items, scores, top_k, label);
    }

    // Cleanup
    delete[] h_sources;
    delete[] h_recommendations;
    delete[] h_scores;
    delete[] h_semantic;
    CUDA_CHECK(cudaFree(d_sources));
    CUDA_CHECK(cudaFree(d_recommendations));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_semantic_matrix));
    CUDA_CHECK(cudaFree(d_metadata));
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    printf("GPU Graph Search Kernels - Content Discovery Example\n");
    printf("=====================================================\n\n");

    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s (compute capability %d.%d)\n\n",
           prop.name, prop.major, prop.minor);

    // Generate test graph
    int num_nodes = 1000;
    int avg_degree = 20;
    std::vector<int> h_row_offsets, h_col_indices;
    std::vector<float> h_edge_weights, h_content_features, h_user_affinities;

    srand(42);  // Reproducible results
    generate_test_graph(num_nodes, avg_degree, h_row_offsets, h_col_indices,
                        h_edge_weights, h_content_features);
    generate_user_affinities(num_nodes, h_user_affinities);

    // Copy graph to device
    int *d_row_offsets, *d_col_indices;
    float *d_edge_weights, *d_content_features, *d_user_affinities;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, h_row_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, h_col_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_weights, h_edge_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_content_features, h_content_features.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_user_affinities, h_user_affinities.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_offsets, h_row_offsets.data(),
                          h_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, h_col_indices.data(),
                          h_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_weights, h_edge_weights.data(),
                          h_edge_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_content_features, h_content_features.data(),
                          h_content_features.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_user_affinities, h_user_affinities.data(),
                          h_user_affinities.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Run examples
    int source_item = 42;

    // Example 1: SSSP recommendations
    example_sssp_recommendations(
        num_nodes, d_row_offsets, d_col_indices, d_edge_weights,
        d_content_features, d_user_affinities, source_item
    );

    // Example 2: Landmark APSP
    example_landmark_apsp(
        num_nodes, d_row_offsets, d_col_indices, d_edge_weights,
        d_content_features, d_user_affinities, h_row_offsets.data()
    );

    // Example 3: Multi-hop batch recommendations (requires distance matrix)
    // For demo, create a simple distance matrix
    float *d_distance_matrix;
    CUDA_CHECK(cudaMalloc(&d_distance_matrix, num_nodes * num_nodes * sizeof(float)));
    float* h_dist = new float[num_nodes * num_nodes];
    for (int i = 0; i < num_nodes * num_nodes; i++) {
        h_dist[i] = 1.0f + (rand() / (float)RAND_MAX) * 10.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_distance_matrix, h_dist,
                          num_nodes * num_nodes * sizeof(float), cudaMemcpyHostToDevice));

    example_multi_hop_batch(num_nodes, d_distance_matrix, d_user_affinities);

    // Cleanup
    delete[] h_dist;
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_edge_weights));
    CUDA_CHECK(cudaFree(d_content_features));
    CUDA_CHECK(cudaFree(d_user_affinities));
    CUDA_CHECK(cudaFree(d_distance_matrix));

    printf("\n=== All Examples Complete ===\n");

    return 0;
}
