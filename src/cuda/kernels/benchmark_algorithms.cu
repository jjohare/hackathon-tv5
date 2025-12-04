#include "hybrid_index.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <vector>

// Benchmark configuration
struct BenchmarkConfig {
    int num_items;               // Dataset size
    int embedding_dim;           // Embedding dimension
    int num_queries;             // Number of queries
    int k;                       // Number of nearest neighbors
    float recall_target;         // Target recall

    // HNSW config
    int hnsw_M;
    int hnsw_ef_construction;
    int hnsw_ef_search;

    // LSH config
    int lsh_num_tables;
    int lsh_num_projections;
    int lsh_num_buckets;

    // PQ config
    int pq_num_subspaces;
};

// Benchmark result
struct BenchmarkResult {
    std::string method;
    double build_time_ms;
    double search_time_ms;
    double throughput_qps;
    float recall;
    size_t memory_bytes;
    double speedup_vs_exact;
};

// Generate random embeddings
void generate_random_embeddings(
    __half* embeddings,
    int num_items,
    int embedding_dim,
    unsigned int seed = 42
) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    // Generate as float then convert to half
    float* temp_float;
    cudaMalloc(&temp_float, num_items * embedding_dim * sizeof(float));
    curandGenerateNormal(gen, temp_float, num_items * embedding_dim, 0.0f, 1.0f);

    // Convert to half precision
    int total = num_items * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    auto convert = [] __device__ (float* src, __half* dst, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = __float2half(src[idx]);
        }
    };

    convert<<<blocks, threads>>>(temp_float, embeddings, total);
    cudaDeviceSynchronize();

    cudaFree(temp_float);
    curandDestroyGenerator(gen);
}

// Compute exact k-NN for ground truth
__global__ void compute_exact_knn(
    const __half* queries,
    const __half* database,
    int* results,
    float* distances,
    int num_queries,
    int num_items,
    int embedding_dim,
    int k
) {
    int query_id = blockIdx.x;
    if (query_id >= num_queries) return;

    const __half* query = queries + query_id * embedding_dim;

    extern __shared__ float all_dists[];

    // Compute distances to all items
    for (int item_id = threadIdx.x; item_id < num_items; item_id += blockDim.x) {
        const __half* item = database + item_id * embedding_dim;

        float dist = 0.0f;
        for (int d = 0; d < embedding_dim; d++) {
            float diff = __half2float(query[d]) - __half2float(item[d]);
            dist += diff * diff;
        }
        all_dists[item_id] = sqrtf(dist);
    }
    __syncthreads();

    // Find k smallest (partial sort)
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            int min_idx = i;
            float min_dist = all_dists[i];

            for (int j = i + 1; j < num_items; j++) {
                if (all_dists[j] < min_dist) {
                    min_dist = all_dists[j];
                    min_idx = j;
                }
            }

            if (min_idx != i) {
                float tmp = all_dists[i];
                all_dists[i] = all_dists[min_idx];
                all_dists[min_idx] = tmp;
            }

            results[query_id * k + i] = (min_idx == i) ? i : min_idx;
            distances[query_id * k + i] = all_dists[i];
        }
    }
}

// Compute recall
float compute_recall(
    const int* results,
    const int* ground_truth,
    int num_queries,
    int k
) {
    int total_correct = 0;
    int total_possible = num_queries * k;

    for (int q = 0; q < num_queries; q++) {
        for (int i = 0; i < k; i++) {
            int result_id = results[q * k + i];

            for (int j = 0; j < k; j++) {
                if (result_id == ground_truth[q * k + j]) {
                    total_correct++;
                    break;
                }
            }
        }
    }

    return (float)total_correct / total_possible;
}

// Benchmark HNSW
BenchmarkResult benchmark_hnsw(
    const BenchmarkConfig& config,
    const __half* database,
    const __half* queries,
    const int* ground_truth
) {
    BenchmarkResult result;
    result.method = "HNSW";

    // Build index
    auto build_start = std::chrono::high_resolution_clock::now();

    // TODO: Implement HNSW build
    // hnsw_build(database, config.num_items, ...);

    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_ms = std::chrono::duration<double, std::milli>(
        build_end - build_start
    ).count();

    // Search
    int* results;
    float* distances;
    cudaMalloc(&results, config.num_queries * config.k * sizeof(int));
    cudaMalloc(&distances, config.num_queries * config.k * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // TODO: Call HNSW search kernel
    // hnsw_search_batch<<<...>>>();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    result.search_time_ms = milliseconds;
    result.throughput_qps = (config.num_queries * 1000.0) / milliseconds;

    // Compute recall
    int* h_results = new int[config.num_queries * config.k];
    cudaMemcpy(h_results, results,
               config.num_queries * config.k * sizeof(int),
               cudaMemcpyDeviceToHost);
    result.recall = compute_recall(h_results, ground_truth,
                                   config.num_queries, config.k);

    delete[] h_results;
    cudaFree(results);
    cudaFree(distances);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.memory_bytes = 0; // TODO: Compute actual memory usage
    result.speedup_vs_exact = 0.0; // Will be computed later

    return result;
}

// Main benchmark function
void run_algorithmic_benchmarks(const BenchmarkConfig& config) {
    std::cout << "=== Algorithmic Optimization Benchmarks ===" << std::endl;
    std::cout << "Dataset size: " << config.num_items << std::endl;
    std::cout << "Embedding dim: " << config.embedding_dim << std::endl;
    std::cout << "Num queries: " << config.num_queries << std::endl;
    std::cout << "k: " << config.k << std::endl;
    std::cout << std::endl;

    // Generate data
    __half *d_database, *d_queries;
    cudaMalloc(&d_database, config.num_items * config.embedding_dim * sizeof(__half));
    cudaMalloc(&d_queries, config.num_queries * config.embedding_dim * sizeof(__half));

    generate_random_embeddings(d_database, config.num_items, config.embedding_dim, 42);
    generate_random_embeddings(d_queries, config.num_queries, config.embedding_dim, 123);

    // Compute ground truth (exact k-NN)
    int *d_ground_truth, *h_ground_truth;
    float *d_gt_distances;
    cudaMalloc(&d_ground_truth, config.num_queries * config.k * sizeof(int));
    cudaMalloc(&d_gt_distances, config.num_queries * config.k * sizeof(float));
    h_ground_truth = new int[config.num_queries * config.k];

    std::cout << "Computing ground truth (exact k-NN)..." << std::endl;
    auto gt_start = std::chrono::high_resolution_clock::now();

    compute_exact_knn<<<config.num_queries, 256,
                       config.num_items * sizeof(float)>>>(
        d_queries, d_database, d_ground_truth, d_gt_distances,
        config.num_queries, config.num_items, config.embedding_dim, config.k
    );
    cudaDeviceSynchronize();

    auto gt_end = std::chrono::high_resolution_clock::now();
    double exact_time_ms = std::chrono::duration<double, std::milli>(
        gt_end - gt_start
    ).count();

    cudaMemcpy(h_ground_truth, d_ground_truth,
               config.num_queries * config.k * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::cout << "Exact search time: " << exact_time_ms << " ms" << std::endl;
    std::cout << "Exact throughput: " << (config.num_queries * 1000.0 / exact_time_ms)
              << " QPS" << std::endl;
    std::cout << std::endl;

    // Run benchmarks
    std::vector<BenchmarkResult> results;

    // Benchmark HNSW
    std::cout << "Benchmarking HNSW..." << std::endl;
    auto hnsw_result = benchmark_hnsw(config, d_database, d_queries, h_ground_truth);
    hnsw_result.speedup_vs_exact = exact_time_ms / hnsw_result.search_time_ms;
    results.push_back(hnsw_result);

    // TODO: Add LSH, PQ, and hybrid benchmarks

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed;
    std::cout << "Method\tBuild(ms)\tSearch(ms)\tQPS\tRecall\tMemory(MB)\tSpeedup" << std::endl;

    for (const auto& r : results) {
        std::cout << r.method << "\t"
                  << r.build_time_ms << "\t"
                  << r.search_time_ms << "\t"
                  << r.throughput_qps << "\t"
                  << r.recall << "\t"
                  << (r.memory_bytes / (1024.0 * 1024.0)) << "\t"
                  << r.speedup_vs_exact << "x" << std::endl;
    }

    // Complexity analysis
    std::cout << "\n=== Complexity Analysis ===" << std::endl;
    std::cout << "Exact k-NN: O(N*D) per query = O(" << config.num_items << " * "
              << config.embedding_dim << ") = O("
              << (config.num_items * config.embedding_dim) << ")" << std::endl;

    std::cout << "HNSW: O(log N * D) per query = O(log(" << config.num_items << ") * "
              << config.embedding_dim << ") ≈ O("
              << (int)(log2(config.num_items) * config.embedding_dim) << ")" << std::endl;

    std::cout << "Theoretical speedup: "
              << (config.num_items * config.embedding_dim) /
                 (log2(config.num_items) * config.embedding_dim) << "x" << std::endl;

    // Cleanup
    cudaFree(d_database);
    cudaFree(d_queries);
    cudaFree(d_ground_truth);
    cudaFree(d_gt_distances);
    delete[] h_ground_truth;
}

int main() {
    BenchmarkConfig config;
    config.num_items = 10000;
    config.embedding_dim = 1024;
    config.num_queries = 1000;
    config.k = 100;
    config.recall_target = 0.90f;

    config.hnsw_M = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 100;

    config.lsh_num_tables = 8;
    config.lsh_num_projections = 16;
    config.lsh_num_buckets = 1024;

    config.pq_num_subspaces = 64;

    run_algorithmic_benchmarks(config);

    return 0;
}
