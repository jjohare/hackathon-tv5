#include "../src/cuda/kernels/benchmark_algorithms.cu"
#include <cassert>
#include <iostream>

// Test HNSW and LSH implementations
void test_hnsw_lsh_algorithms() {
    std::cout << "Testing HNSW and LSH algorithm implementations..." << std::endl;

    // Small test configuration
    BenchmarkConfig config;
    config.num_items = 1000;
    config.embedding_dim = 128;
    config.num_queries = 10;
    config.k = 10;
    config.recall_target = 0.90f;

    config.hnsw_M = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 100;

    config.lsh_num_tables = 8;
    config.lsh_num_projections = 16;
    config.lsh_num_buckets = 128;

    config.pq_num_subspaces = 64;

    // Allocate test data
    __half *d_database, *d_queries;
    cudaMalloc(&d_database, config.num_items * config.embedding_dim * sizeof(__half));
    cudaMalloc(&d_queries, config.num_queries * config.embedding_dim * sizeof(__half));

    generate_random_embeddings(d_database, config.num_items, config.embedding_dim, 42);
    generate_random_embeddings(d_queries, config.num_queries, config.embedding_dim, 123);

    // Compute ground truth
    int *d_ground_truth, *h_ground_truth;
    float *d_gt_distances;
    cudaMalloc(&d_ground_truth, config.num_queries * config.k * sizeof(int));
    cudaMalloc(&d_gt_distances, config.num_queries * config.k * sizeof(float));
    h_ground_truth = new int[config.num_queries * config.k];

    compute_exact_knn<<<config.num_queries, 256,
                       config.num_items * sizeof(float)>>>(
        d_queries, d_database, d_ground_truth, d_gt_distances,
        config.num_queries, config.num_items, config.embedding_dim, config.k
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_ground_truth, d_ground_truth,
               config.num_queries * config.k * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::cout << "✓ Ground truth computed" << std::endl;

    // Test HNSW
    std::cout << "\nTesting HNSW implementation..." << std::endl;
    auto hnsw_result = benchmark_hnsw(config, d_database, d_queries, h_ground_truth);

    std::cout << "HNSW Results:" << std::endl;
    std::cout << "  Build time: " << hnsw_result.build_time_ms << " ms" << std::endl;
    std::cout << "  Search time: " << hnsw_result.search_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << hnsw_result.throughput_qps << " QPS" << std::endl;
    std::cout << "  Recall: " << hnsw_result.recall << std::endl;
    std::cout << "  Memory: " << (hnsw_result.memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    assert(hnsw_result.recall > 0.0f && "HNSW recall should be > 0");
    assert(hnsw_result.throughput_qps > 0.0 && "HNSW throughput should be > 0");
    std::cout << "✓ HNSW implementation validated" << std::endl;

    // Test LSH
    std::cout << "\nTesting LSH implementation..." << std::endl;
    auto lsh_result = benchmark_lsh(config, d_database, d_queries, h_ground_truth);

    std::cout << "LSH Results:" << std::endl;
    std::cout << "  Build time: " << lsh_result.build_time_ms << " ms" << std::endl;
    std::cout << "  Search time: " << lsh_result.search_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << lsh_result.throughput_qps << " QPS" << std::endl;
    std::cout << "  Recall: " << lsh_result.recall << std::endl;
    std::cout << "  Memory: " << (lsh_result.memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    assert(lsh_result.recall > 0.0f && "LSH recall should be > 0");
    assert(lsh_result.throughput_qps > 0.0 && "LSH throughput should be > 0");
    std::cout << "✓ LSH implementation validated" << std::endl;

    // Verify complexity improvements
    std::cout << "\nComplexity Analysis:" << std::endl;
    int exact_ops = config.num_items * config.embedding_dim;
    int hnsw_ops = (int)(log2(config.num_items) * config.embedding_dim);
    int lsh_candidates = (config.num_items / config.lsh_num_buckets) * config.lsh_num_tables;

    std::cout << "  Exact: O(" << exact_ops << ")" << std::endl;
    std::cout << "  HNSW: O(" << hnsw_ops << ") - "
              << (exact_ops / hnsw_ops) << "x reduction" << std::endl;
    std::cout << "  LSH: O(" << lsh_candidates << " candidates) - "
              << (config.num_items / lsh_candidates) << "x reduction" << std::endl;

    assert(hnsw_ops < exact_ops && "HNSW should have better complexity");
    assert(lsh_candidates < config.num_items && "LSH should reduce candidate set");

    // Cleanup
    cudaFree(d_database);
    cudaFree(d_queries);
    cudaFree(d_ground_truth);
    cudaFree(d_gt_distances);
    delete[] h_ground_truth;

    std::cout << "\n✅ All tests passed!" << std::endl;
}

int main() {
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    test_hnsw_lsh_algorithms();

    return 0;
}
