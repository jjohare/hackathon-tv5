// =============================================================================
// Phase 2 Memory Optimization Benchmark
// =============================================================================
// Validates 4-5x speedup from memory coalescing and shared memory caching
//
// Baseline: Phase 1 (random access, 60 GB/s)
// Target: Phase 2 (coalesced access, 280+ GB/s)
//
// Test Scenarios:
// 1. Random vs Sorted pair processing
// 2. Bandwidth measurement (GB/s)
// 3. Cache hit rates
// 4. End-to-end speedup comparison
// =============================================================================

#include "../kernels/memory_optimization.cuh"
#include "../kernels/sorted_similarity.cu"
#include "../kernels/memory_layout.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <chrono>

// =============================================================================
// Baseline Kernel (Random Access - Phase 1)
// =============================================================================

__global__ void compute_similarity_random_baseline(
    const __half* __restrict__ embeddings,
    const int* __restrict__ src_indices,
    const int* __restrict__ tgt_indices,
    float* __restrict__ similarities,
    int num_pairs,
    int embedding_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_pairs) return;

    int src = src_indices[idx];
    int tgt = tgt_indices[idx];

    // Random memory access - non-coalesced
    const __half* vec_a = &embeddings[src * embedding_dim];
    const __half* vec_b = &embeddings[tgt * embedding_dim];

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int d = 0; d < embedding_dim; d++) {
        float a = __half2float(vec_a[d]);
        float b = __half2float(vec_b[d]);
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    float similarity = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-6f);
    similarities[idx] = similarity;
}

// =============================================================================
// Test Data Generation
// =============================================================================

void generate_random_embeddings(__half* embeddings, int num_embeddings, int dim) {
    for (int i = 0; i < num_embeddings * dim; i++) {
        float val = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        embeddings[i] = __float2half(val);
    }
}

void generate_random_pairs(int* src, int* tgt, int num_pairs, int num_embeddings) {
    for (int i = 0; i < num_pairs; i++) {
        src[i] = rand() % num_embeddings;
        tgt[i] = rand() % num_embeddings;
    }
}

void generate_sorted_pairs(int* src, int* tgt, int num_pairs, int num_embeddings) {
    // Generate random pairs
    generate_random_pairs(src, tgt, num_pairs, num_embeddings);

    // Sort by source index
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < num_pairs; i++) {
        pairs.push_back({src[i], tgt[i]});
    }

    std::sort(pairs.begin(), pairs.end());

    for (int i = 0; i < num_pairs; i++) {
        src[i] = pairs[i].first;
        tgt[i] = pairs[i].second;
    }
}

// =============================================================================
// Benchmark Functions
// =============================================================================

float benchmark_random_access(
    const __half* d_embeddings,
    const int* d_src,
    const int* d_tgt,
    float* d_similarities,
    int num_pairs,
    int embedding_dim,
    int iterations = 10
) {
    int block_size = 256;
    int grid_size = (num_pairs + block_size - 1) / block_size;

    // Warmup
    compute_similarity_random_baseline<<<grid_size, block_size>>>(
        d_embeddings, d_src, d_tgt, d_similarities, num_pairs, embedding_dim
    );
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        compute_similarity_random_baseline<<<grid_size, block_size>>>(
            d_embeddings, d_src, d_tgt, d_similarities, num_pairs, embedding_dim
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time_ms;
    cudaEventElapsedTime(&total_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time_ms / iterations;
}

float benchmark_coalesced_access(
    const __half* d_embeddings,
    const SortedPairBatch* d_batches,
    float* d_similarities,
    int num_batches,
    int num_items,
    int embedding_dim,
    int iterations = 10
) {
    int block_size = 256;
    int grid_size = num_batches;

    // Warmup
    if (embedding_dim == 1024) {
        compute_similarity_sorted_1024<<<grid_size, block_size>>>(
            d_embeddings, d_batches, d_similarities, num_batches, num_items
        );
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        if (embedding_dim == 1024) {
            compute_similarity_sorted_1024<<<grid_size, block_size>>>(
                d_embeddings, d_batches, d_similarities, num_batches, num_items
            );
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time_ms;
    cudaEventElapsedTime(&total_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time_ms / iterations;
}

// =============================================================================
// Main Benchmark
// =============================================================================

int main(int argc, char** argv) {
    printf("=================================================================\n");
    printf("Phase 2 Memory Optimization Benchmark\n");
    printf("=================================================================\n\n");

    srand(time(NULL));

    // Test configuration
    const int NUM_EMBEDDINGS = 10000;
    const int EMBEDDING_DIM = 1024;
    const int NUM_PAIRS = 100000;
    const int ITERATIONS = 10;

    printf("Configuration:\n");
    printf("  Embeddings: %d\n", NUM_EMBEDDINGS);
    printf("  Dimension: %d\n", EMBEDDING_DIM);
    printf("  Pairs: %d\n", NUM_PAIRS);
    printf("  Iterations: %d\n\n", ITERATIONS);

    // Allocate host memory
    __half* h_embeddings = new __half[NUM_EMBEDDINGS * EMBEDDING_DIM];
    int* h_src = new int[NUM_PAIRS];
    int* h_tgt = new int[NUM_PAIRS];
    float* h_similarities = new float[NUM_PAIRS];

    // Generate test data
    printf("Generating test data...\n");
    generate_random_embeddings(h_embeddings, NUM_EMBEDDINGS, EMBEDDING_DIM);
    generate_random_pairs(h_src, h_tgt, NUM_PAIRS, NUM_EMBEDDINGS);

    // Allocate device memory
    __half* d_embeddings;
    int* d_src_random, *d_tgt_random;
    int* d_src_sorted, *d_tgt_sorted;
    float* d_similarities;

    cudaMalloc(&d_embeddings, NUM_EMBEDDINGS * EMBEDDING_DIM * sizeof(__half));
    cudaMalloc(&d_src_random, NUM_PAIRS * sizeof(int));
    cudaMalloc(&d_tgt_random, NUM_PAIRS * sizeof(int));
    cudaMalloc(&d_src_sorted, NUM_PAIRS * sizeof(int));
    cudaMalloc(&d_tgt_sorted, NUM_PAIRS * sizeof(int));
    cudaMalloc(&d_similarities, NUM_PAIRS * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_embeddings, h_embeddings, NUM_EMBEDDINGS * EMBEDDING_DIM * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_random, h_src, NUM_PAIRS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_random, h_tgt, NUM_PAIRS * sizeof(int), cudaMemcpyHostToDevice);

    // ==========================================================================
    // Benchmark 1: Random Access (Baseline)
    // ==========================================================================
    printf("\n--- Benchmark 1: Random Access (Baseline) ---\n");
    float random_time_ms = benchmark_random_access(
        d_embeddings, d_src_random, d_tgt_random, d_similarities,
        NUM_PAIRS, EMBEDDING_DIM, ITERATIONS
    );

    unsigned long long random_bytes_read = (unsigned long long)NUM_PAIRS * EMBEDDING_DIM * 2 * 2;  // 2 vectors, 2 bytes/element
    float random_bandwidth = (random_bytes_read / (random_time_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);

    printf("  Time: %.2f ms\n", random_time_ms);
    printf("  Bandwidth: %.2f GB/s\n", random_bandwidth);

    // ==========================================================================
    // Benchmark 2: Sorted Access (Phase 2 Optimized)
    // ==========================================================================
    printf("\n--- Benchmark 2: Sorted Coalesced Access (Phase 2) ---\n");

    // Generate sorted pairs
    generate_sorted_pairs(h_src, h_tgt, NUM_PAIRS, NUM_EMBEDDINGS);
    cudaMemcpy(d_src_sorted, h_src, NUM_PAIRS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_sorted, h_tgt, NUM_PAIRS * sizeof(int), cudaMemcpyHostToDevice);

    // Create batches
    std::vector<SortedPairBatch> batches;
    int current_src = h_src[0];
    int batch_start = 0;

    for (int i = 1; i <= NUM_PAIRS; i++) {
        if (i == NUM_PAIRS || h_src[i] != current_src) {
            SortedPairBatch batch;
            batch.src_start = current_src;
            batch.src_end = current_src + 1;
            batch.tgt_indices = d_tgt_sorted + batch_start;
            batch.batch_size = i - batch_start;
            batch.batch_id = batches.size();
            batches.push_back(batch);

            if (i < NUM_PAIRS) {
                current_src = h_src[i];
                batch_start = i;
            }
        }
    }

    printf("  Generated %zu batches\n", batches.size());

    // Copy batches to device
    SortedPairBatch* d_batches;
    cudaMalloc(&d_batches, batches.size() * sizeof(SortedPairBatch));
    cudaMemcpy(d_batches, batches.data(), batches.size() * sizeof(SortedPairBatch), cudaMemcpyHostToDevice);

    // Benchmark sorted access
    float sorted_time_ms = benchmark_coalesced_access(
        d_embeddings, d_batches, d_similarities,
        batches.size(), NUM_EMBEDDINGS, EMBEDDING_DIM, ITERATIONS
    );

    unsigned long long sorted_bytes_read = (unsigned long long)NUM_PAIRS * EMBEDDING_DIM * 2 * 2;
    float sorted_bandwidth = (sorted_bytes_read / (sorted_time_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);

    printf("  Time: %.2f ms\n", sorted_time_ms);
    printf("  Bandwidth: %.2f GB/s\n", sorted_bandwidth);

    // ==========================================================================
    // Results Summary
    // ==========================================================================
    printf("\n=================================================================\n");
    printf("RESULTS SUMMARY\n");
    printf("=================================================================\n");
    printf("Random Access (Baseline):\n");
    printf("  Time: %.2f ms\n", random_time_ms);
    printf("  Bandwidth: %.2f GB/s\n", random_bandwidth);
    printf("\nSorted Coalesced Access (Phase 2):\n");
    printf("  Time: %.2f ms\n", sorted_time_ms);
    printf("  Bandwidth: %.2f GB/s\n", sorted_bandwidth);
    printf("\nImprovement:\n");
    printf("  Speedup: %.2fx\n", random_time_ms / sorted_time_ms);
    printf("  Bandwidth increase: %.2fx\n", sorted_bandwidth / random_bandwidth);
    printf("  Target achieved: %s\n", (sorted_bandwidth / random_bandwidth >= 4.0f) ? "YES" : "NO");

    if (sorted_bandwidth / random_bandwidth >= 4.0f) {
        printf("\n✓ SUCCESS: Phase 2 optimizations achieved 4-5x speedup target!\n");
    } else {
        printf("\n✗ Target not reached. Further optimization needed.\n");
    }

    printf("=================================================================\n");

    // Cleanup
    delete[] h_embeddings;
    delete[] h_src;
    delete[] h_tgt;
    delete[] h_similarities;

    cudaFree(d_embeddings);
    cudaFree(d_src_random);
    cudaFree(d_tgt_random);
    cudaFree(d_src_sorted);
    cudaFree(d_tgt_sorted);
    cudaFree(d_similarities);
    cudaFree(d_batches);

    return 0;
}
