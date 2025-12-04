#include "hnsw_gpu.cuh"
#include "lsh_gpu.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

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

    // Allocate HNSW structure
    HNSW_GPU hnsw;
    hnsw.M = config.hnsw_M;
    hnsw.M0 = config.hnsw_M * 2;
    hnsw.ml = 1.0f / log(2.0f);
    hnsw.embedding_dim = config.embedding_dim;
    hnsw.total_nodes = config.num_items;
    hnsw.entry_point = 0;

    // Calculate number of layers (logarithmic)
    hnsw.num_layers = (int)(log2(config.num_items)) + 1;

    // Allocate layers
    HNSWLayer* h_layers = new HNSWLayer[hnsw.num_layers];
    for (int i = 0; i < hnsw.num_layers; i++) {
        int layer_nodes = config.num_items >> i; // Exponential decay
        if (layer_nodes < 1) layer_nodes = 1;

        h_layers[i].num_nodes = layer_nodes;
        h_layers[i].M = (i == 0) ? hnsw.M0 : hnsw.M;
        h_layers[i].ef_construction = config.hnsw_ef_construction;

        cudaMalloc(&h_layers[i].neighbors, layer_nodes * h_layers[i].M * sizeof(int));
        cudaMalloc(&h_layers[i].distances, layer_nodes * h_layers[i].M * sizeof(float));

        // Initialize neighbors to -1 (invalid)
        cudaMemset(h_layers[i].neighbors, -1, layer_nodes * h_layers[i].M * sizeof(int));
    }

    cudaMalloc(&hnsw.layers, hnsw.num_layers * sizeof(HNSWLayer));
    cudaMemcpy(hnsw.layers, h_layers, hnsw.num_layers * sizeof(HNSWLayer), cudaMemcpyHostToDevice);

    // Copy embeddings for HNSW
    cudaMalloc(&hnsw.node_embeddings, config.num_items * config.embedding_dim * sizeof(__half));
    cudaMemcpy(hnsw.node_embeddings, database,
               config.num_items * config.embedding_dim * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    // Build graph incrementally (simplified batch insertion)
    int* d_node_ids;
    cudaMalloc(&d_node_ids, config.num_items * sizeof(int));

    // Initialize node IDs
    int* h_node_ids = new int[config.num_items];
    for (int i = 0; i < config.num_items; i++) h_node_ids[i] = i;
    cudaMemcpy(d_node_ids, h_node_ids, config.num_items * sizeof(int), cudaMemcpyHostToDevice);

    // Batch insertion (simplified - production would be incremental)
    hnsw_insert_batch<<<config.num_items, 256>>>(
        hnsw, database, d_node_ids, config.num_items, config.hnsw_ef_construction
    );
    cudaDeviceSynchronize();

    delete[] h_node_ids;
    cudaFree(d_node_ids);

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

    // Call HNSW search kernel
    hnsw_search_batch<<<config.num_queries, 256>>>(
        hnsw,
        queries,
        results,
        distances,
        config.num_queries,
        config.k,
        config.hnsw_ef_search
    );

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

    // Compute actual memory usage
    size_t total_memory = 0;

    // Node embeddings
    total_memory += config.num_items * config.embedding_dim * sizeof(__half);

    // Layers (neighbors + distances)
    for (int i = 0; i < hnsw.num_layers; i++) {
        int layer_nodes = config.num_items >> i;
        if (layer_nodes < 1) layer_nodes = 1;
        int M = (i == 0) ? hnsw.M0 : hnsw.M;

        total_memory += layer_nodes * M * sizeof(int);    // neighbors
        total_memory += layer_nodes * M * sizeof(float);  // distances
    }

    // Layer metadata
    total_memory += hnsw.num_layers * sizeof(HNSWLayer);

    result.memory_bytes = total_memory;
    result.speedup_vs_exact = 0.0; // Will be computed later

    // Cleanup
    for (int i = 0; i < hnsw.num_layers; i++) {
        cudaFree(h_layers[i].neighbors);
        cudaFree(h_layers[i].distances);
    }
    cudaFree(hnsw.layers);
    cudaFree(hnsw.node_embeddings);
    delete[] h_layers;

    return result;
}

// Benchmark LSH
BenchmarkResult benchmark_lsh(
    const BenchmarkConfig& config,
    const __half* database,
    const __half* queries,
    const int* ground_truth
) {
    BenchmarkResult result;
    result.method = "LSH";

    // Build LSH index
    auto build_start = std::chrono::high_resolution_clock::now();

    // Initialize LSH structure
    LSH_GPU lsh;
    lsh.num_tables = config.lsh_num_tables;
    lsh.num_projections = config.lsh_num_projections;
    lsh.num_buckets = config.lsh_num_buckets;
    lsh.bucket_size = (config.num_items / config.lsh_num_buckets) * 4; // 4x avg for safety
    lsh.embedding_dim = config.embedding_dim;

    // Allocate LSH data structures
    size_t proj_size = config.lsh_num_tables * config.lsh_num_projections * config.embedding_dim;
    cudaMalloc(&lsh.random_projections, proj_size * sizeof(float));

    size_t table_size = config.lsh_num_tables * config.lsh_num_buckets * lsh.bucket_size;
    cudaMalloc(&lsh.hash_tables, table_size * sizeof(int));
    cudaMemset(lsh.hash_tables, -1, table_size * sizeof(int));

    size_t count_size = config.lsh_num_tables * config.lsh_num_buckets;
    cudaMalloc(&lsh.bucket_counts, count_size * sizeof(int));
    cudaMemset(lsh.bucket_counts, 0, count_size * sizeof(int));

    // Initialize random projections
    int proj_threads = 256;
    int proj_blocks = (proj_size + proj_threads - 1) / proj_threads;
    lsh_init_projections<<<proj_blocks, proj_threads>>>(
        lsh.random_projections,
        config.lsh_num_tables,
        config.lsh_num_projections,
        config.embedding_dim,
        12345ULL
    );

    // Create item IDs
    int* d_item_ids;
    cudaMalloc(&d_item_ids, config.num_items * sizeof(int));
    int* h_item_ids = new int[config.num_items];
    for (int i = 0; i < config.num_items; i++) h_item_ids[i] = i;
    cudaMemcpy(d_item_ids, h_item_ids, config.num_items * sizeof(int), cudaMemcpyHostToDevice);

    // Insert all items into LSH tables
    lsh_insert_batch<<<config.num_items, 1>>>(
        lsh, database, d_item_ids, config.num_items
    );
    cudaDeviceSynchronize();

    delete[] h_item_ids;
    cudaFree(d_item_ids);

    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_ms = std::chrono::duration<double, std::milli>(
        build_end - build_start
    ).count();

    // Search phase
    int max_candidates = config.k * 10; // Get 10x candidates for reranking
    int* d_candidates;
    int* d_candidate_counts;
    cudaMalloc(&d_candidates, config.num_queries * max_candidates * sizeof(int));
    cudaMalloc(&d_candidate_counts, config.num_queries * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // LSH candidate generation
    lsh_search_batch<<<config.num_queries, 256>>>(
        lsh,
        queries,
        d_candidates,
        d_candidate_counts,
        config.num_queries,
        max_candidates
    );

    // Rerank candidates using exact distance computation
    int* d_results;
    float* d_distances;
    cudaMalloc(&d_results, config.num_queries * config.k * sizeof(int));
    cudaMalloc(&d_distances, config.num_queries * config.k * sizeof(float));

    // Launch reranking kernel
    auto rerank_kernel = [] __global__ (
        const __half* queries,
        const __half* database,
        const int* candidates,
        const int* candidate_counts,
        int* results,
        float* distances,
        int num_queries,
        int k,
        int embedding_dim,
        int max_candidates
    ) {
        int query_id = blockIdx.x;
        if (query_id >= num_queries) return;

        const __half* query = queries + query_id * embedding_dim;
        int num_candidates = candidate_counts[query_id];

        extern __shared__ float candidate_dists[];

        // Compute distances to all candidates
        for (int i = threadIdx.x; i < num_candidates; i += blockDim.x) {
            int candidate_id = candidates[query_id * max_candidates + i];
            if (candidate_id < 0) continue;

            const __half* candidate = database + candidate_id * embedding_dim;

            float dist = 0.0f;
            for (int d = 0; d < embedding_dim; d++) {
                float diff = __half2float(query[d]) - __half2float(candidate[d]);
                dist += diff * diff;
            }
            candidate_dists[i] = sqrtf(dist);
        }
        __syncthreads();

        // Find k smallest (partial sort)
        if (threadIdx.x == 0) {
            for (int i = 0; i < k && i < num_candidates; i++) {
                int min_idx = i;
                float min_dist = candidate_dists[i];

                for (int j = i + 1; j < num_candidates; j++) {
                    if (candidate_dists[j] < min_dist) {
                        min_dist = candidate_dists[j];
                        min_idx = j;
                    }
                }

                if (min_idx != i) {
                    float tmp_dist = candidate_dists[i];
                    candidate_dists[i] = candidate_dists[min_idx];
                    candidate_dists[min_idx] = tmp_dist;

                    int tmp_id = candidates[query_id * max_candidates + i];
                    candidates[query_id * max_candidates + i] = candidates[query_id * max_candidates + min_idx];
                    candidates[query_id * max_candidates + min_idx] = tmp_id;
                }

                results[query_id * k + i] = candidates[query_id * max_candidates + i];
                distances[query_id * k + i] = candidate_dists[i];
            }
        }
    };

    rerank_kernel<<<config.num_queries, 256, max_candidates * sizeof(float)>>>(
        queries, database, d_candidates, d_candidate_counts,
        d_results, d_distances, config.num_queries, config.k,
        config.embedding_dim, max_candidates
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    result.search_time_ms = milliseconds;
    result.throughput_qps = (config.num_queries * 1000.0) / milliseconds;

    // Compute recall
    int* h_results = new int[config.num_queries * config.k];
    cudaMemcpy(h_results, d_results,
               config.num_queries * config.k * sizeof(int),
               cudaMemcpyDeviceToHost);
    result.recall = compute_recall(h_results, ground_truth,
                                   config.num_queries, config.k);

    delete[] h_results;

    // Compute memory usage
    result.memory_bytes =
        proj_size * sizeof(float) +           // Random projections
        table_size * sizeof(int) +            // Hash tables
        count_size * sizeof(int);             // Bucket counts

    result.speedup_vs_exact = 0.0; // Will be computed later

    // Cleanup
    cudaFree(lsh.random_projections);
    cudaFree(lsh.hash_tables);
    cudaFree(lsh.bucket_counts);
    cudaFree(d_candidates);
    cudaFree(d_candidate_counts);
    cudaFree(d_results);
    cudaFree(d_distances);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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

    // Benchmark LSH
    std::cout << "Benchmarking LSH..." << std::endl;
    auto lsh_result = benchmark_lsh(config, d_database, d_queries, h_ground_truth);
    lsh_result.speedup_vs_exact = exact_time_ms / lsh_result.search_time_ms;
    results.push_back(lsh_result);

    // TODO: Add PQ and hybrid benchmarks

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

    std::cout << "HNSW Theoretical speedup: "
              << (config.num_items * config.embedding_dim) /
                 (log2(config.num_items) * config.embedding_dim) << "x" << std::endl;

    // LSH complexity
    int avg_bucket_size = config.num_items / config.lsh_num_buckets;
    int lsh_candidates = avg_bucket_size * config.lsh_num_tables;
    std::cout << "\nLSH: O(L * B * D) per query where L=" << config.lsh_num_tables
              << " tables, B≈" << avg_bucket_size << " items/bucket" << std::endl;
    std::cout << "LSH candidates: " << lsh_candidates << " vs " << config.num_items
              << " full scan" << std::endl;
    std::cout << "LSH Theoretical speedup: "
              << (float)config.num_items / (float)lsh_candidates << "x" << std::endl;

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
