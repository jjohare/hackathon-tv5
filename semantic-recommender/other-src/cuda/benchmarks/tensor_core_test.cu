// =============================================================================
// Tensor Core Performance Benchmark
// Validates 8-10x speedup from scalar -> tensor core conversion
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>

// Original scalar implementation (for comparison)
extern "C" {
    __global__ void compute_multimodal_similarity_fp16_t4(
        const __half* visual_embeddings,
        const __half* audio_embeddings,
        const __half* text_embeddings,
        const int* item_pairs_src,
        const int* item_pairs_tgt,
        float* similarity_scores,
        int num_pairs,
        int visual_dim,
        int audio_dim,
        int text_dim,
        float visual_weight,
        float audio_weight,
        float text_weight
    );

    // New tensor core implementation
    __global__ void compute_multimodal_similarity_tensor_cores(
        const __half* visual_embeddings,
        const __half* audio_embeddings,
        const __half* text_embeddings,
        const float* visual_norms,
        const float* audio_norms,
        const float* text_norms,
        const int* item_pairs_src,
        const int* item_pairs_tgt,
        float* similarity_scores,
        int num_pairs,
        int visual_dim,
        int audio_dim,
        int text_dim,
        float visual_weight,
        float audio_weight,
        float text_weight
    );

    __global__ void precompute_norms_kernel(
        const __half* embeddings,
        float* norms,
        int num_vectors,
        int embedding_dim
    );
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start;
    cudaEvent_t cuda_start, cuda_stop;

public:
    BenchmarkTimer() {
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);
    }

    ~BenchmarkTimer() {
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
    }

    void startCPU() {
        start = std::chrono::high_resolution_clock::now();
    }

    double stopCPU() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void startGPU() {
        cudaEventRecord(cuda_start);
    }

    float stopGPU() {
        cudaEventRecord(cuda_stop);
        cudaEventSynchronize(cuda_stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, cuda_start, cuda_stop);
        return ms;
    }
};

void initialize_random_fp16(__half* data, int size) {
    float* temp = new float[size];
    for (int i = 0; i < size; i++) {
        temp[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // Convert to FP16
    for (int i = 0; i < size; i++) {
        data[i] = __float2half(temp[i]);
    }

    delete[] temp;
}

void run_benchmark() {
    printf("=============================================================================\n");
    printf("PHASE 1: Tensor Core Performance Benchmark\n");
    printf("=============================================================================\n\n");

    // Test configuration
    const int num_items = 10000;
    const int num_pairs = 50000;
    const int visual_dim = 1024;
    const int audio_dim = 512;
    const int text_dim = 768;

    printf("Configuration:\n");
    printf("  Number of items: %d\n", num_items);
    printf("  Number of pairs: %d\n", num_pairs);
    printf("  Visual dimension: %d\n", visual_dim);
    printf("  Audio dimension: %d\n", audio_dim);
    printf("  Text dimension: %d\n\n", text_dim);

    // Allocate host memory
    size_t visual_size = num_items * visual_dim * sizeof(__half);
    size_t audio_size = num_items * audio_dim * sizeof(__half);
    size_t text_size = num_items * text_dim * sizeof(__half);
    size_t pairs_size = num_pairs * sizeof(int);
    size_t scores_size = num_pairs * sizeof(float);
    size_t norms_size = num_items * sizeof(float);

    __half* h_visual = (__half*)malloc(visual_size);
    __half* h_audio = (__half*)malloc(audio_size);
    __half* h_text = (__half*)malloc(text_size);
    int* h_src = (int*)malloc(pairs_size);
    int* h_tgt = (int*)malloc(pairs_size);
    float* h_scores_scalar = (float*)malloc(scores_size);
    float* h_scores_tensor = (float*)malloc(scores_size);

    // Initialize data
    printf("Initializing random data...\n");
    srand(42);
    initialize_random_fp16(h_visual, num_items * visual_dim);
    initialize_random_fp16(h_audio, num_items * audio_dim);
    initialize_random_fp16(h_text, num_items * text_dim);

    for (int i = 0; i < num_pairs; i++) {
        h_src[i] = rand() % num_items;
        h_tgt[i] = rand() % num_items;
    }

    // Allocate device memory
    __half *d_visual, *d_audio, *d_text;
    int *d_src, *d_tgt;
    float *d_scores_scalar, *d_scores_tensor;
    float *d_visual_norms, *d_audio_norms, *d_text_norms;

    CUDA_CHECK(cudaMalloc(&d_visual, visual_size));
    CUDA_CHECK(cudaMalloc(&d_audio, audio_size));
    CUDA_CHECK(cudaMalloc(&d_text, text_size));
    CUDA_CHECK(cudaMalloc(&d_src, pairs_size));
    CUDA_CHECK(cudaMalloc(&d_tgt, pairs_size));
    CUDA_CHECK(cudaMalloc(&d_scores_scalar, scores_size));
    CUDA_CHECK(cudaMalloc(&d_scores_tensor, scores_size));
    CUDA_CHECK(cudaMalloc(&d_visual_norms, norms_size));
    CUDA_CHECK(cudaMalloc(&d_audio_norms, norms_size));
    CUDA_CHECK(cudaMalloc(&d_text_norms, norms_size));

    // Copy to device
    printf("Copying data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_visual, h_visual, visual_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_audio, h_audio, audio_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_text, h_text, text_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, pairs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt, pairs_size, cudaMemcpyHostToDevice));

    // Precompute norms for tensor core version
    printf("Precomputing norms...\n");
    dim3 norm_block(256);
    dim3 norm_grid((num_items + 255) / 256);

    precompute_norms_kernel<<<norm_grid, norm_block>>>(d_visual, d_visual_norms, num_items, visual_dim);
    precompute_norms_kernel<<<norm_grid, norm_block>>>(d_audio, d_audio_norms, num_items, audio_dim);
    precompute_norms_kernel<<<norm_grid, norm_block>>>(d_text, d_text_norms, num_items, text_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch configuration
    dim3 block_size(256);
    dim3 grid_size((num_pairs + 255) / 256);

    float visual_weight = 0.5f;
    float audio_weight = 0.3f;
    float text_weight = 0.2f;

    BenchmarkTimer timer;

    printf("\n=============================================================================\n");
    printf("BENCHMARK: Scalar Operations (Original)\n");
    printf("=============================================================================\n");

    // Warmup
    compute_multimodal_similarity_fp16_t4<<<grid_size, block_size>>>(
        d_visual, d_audio, d_text, d_src, d_tgt, d_scores_scalar,
        num_pairs, visual_dim, audio_dim, text_dim,
        visual_weight, audio_weight, text_weight
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark scalar version
    const int num_iterations = 100;
    timer.startGPU();
    for (int i = 0; i < num_iterations; i++) {
        compute_multimodal_similarity_fp16_t4<<<grid_size, block_size>>>(
            d_visual, d_audio, d_text, d_src, d_tgt, d_scores_scalar,
            num_pairs, visual_dim, audio_dim, text_dim,
            visual_weight, audio_weight, text_weight
        );
    }
    float scalar_time = timer.stopGPU() / num_iterations;

    printf("Average time: %.3f ms\n", scalar_time);
    printf("Throughput: %.2f million pairs/sec\n", (num_pairs / 1e6) / (scalar_time / 1000.0));

    printf("\n=============================================================================\n");
    printf("BENCHMARK: Tensor Core Operations (Optimized)\n");
    printf("=============================================================================\n");

    // Warmup
    compute_multimodal_similarity_tensor_cores<<<grid_size, block_size, 2048 * sizeof(__half)>>>(
        d_visual, d_audio, d_text, d_visual_norms, d_audio_norms, d_text_norms,
        d_src, d_tgt, d_scores_tensor,
        num_pairs, visual_dim, audio_dim, text_dim,
        visual_weight, audio_weight, text_weight
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark tensor core version
    timer.startGPU();
    for (int i = 0; i < num_iterations; i++) {
        compute_multimodal_similarity_tensor_cores<<<grid_size, block_size, 2048 * sizeof(__half)>>>(
            d_visual, d_audio, d_text, d_visual_norms, d_audio_norms, d_text_norms,
            d_src, d_tgt, d_scores_tensor,
            num_pairs, visual_dim, audio_dim, text_dim,
            visual_weight, audio_weight, text_weight
        );
    }
    float tensor_time = timer.stopGPU() / num_iterations;

    printf("Average time: %.3f ms\n", tensor_time);
    printf("Throughput: %.2f million pairs/sec\n", (num_pairs / 1e6) / (tensor_time / 1000.0));

    printf("\n=============================================================================\n");
    printf("SPEEDUP ANALYSIS\n");
    printf("=============================================================================\n");

    float speedup = scalar_time / tensor_time;
    printf("Speedup: %.2fx\n", speedup);
    printf("Time reduction: %.1f%%\n", (1.0 - tensor_time / scalar_time) * 100.0);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_scores_scalar, d_scores_scalar, scores_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores_tensor, d_scores_tensor, scores_size, cudaMemcpyDeviceToHost));

    // Validate accuracy
    printf("\n=============================================================================\n");
    printf("ACCURACY VALIDATION\n");
    printf("=============================================================================\n");

    double max_error = 0.0;
    double avg_error = 0.0;
    int num_samples = std::min(1000, num_pairs);

    for (int i = 0; i < num_samples; i++) {
        double error = fabs(h_scores_scalar[i] - h_scores_tensor[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= num_samples;

    printf("Maximum error: %.6f\n", max_error);
    printf("Average error: %.6f\n", avg_error);
    printf("Relative error: %.4f%%\n", (avg_error * 100.0));

    if (avg_error < 0.01) {
        printf("✓ PASSED: Results are accurate\n");
    } else {
        printf("✗ WARNING: High error detected\n");
    }

    printf("\n=============================================================================\n");
    printf("EXPECTED vs ACTUAL\n");
    printf("=============================================================================\n");
    printf("Expected speedup: 8-10x\n");
    printf("Actual speedup: %.2fx\n", speedup);

    if (speedup >= 8.0) {
        printf("✓ SUCCESS: Target achieved!\n");
    } else if (speedup >= 5.0) {
        printf("⚠ PARTIAL: Good improvement, but below target\n");
    } else {
        printf("✗ FAILED: Speedup below expectations\n");
    }

    // Cleanup
    free(h_visual);
    free(h_audio);
    free(h_text);
    free(h_src);
    free(h_tgt);
    free(h_scores_scalar);
    free(h_scores_tensor);

    cudaFree(d_visual);
    cudaFree(d_audio);
    cudaFree(d_text);
    cudaFree(d_src);
    cudaFree(d_tgt);
    cudaFree(d_scores_scalar);
    cudaFree(d_scores_tensor);
    cudaFree(d_visual_norms);
    cudaFree(d_audio_norms);
    cudaFree(d_text_norms);
}

int main() {
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=============================================================================\n");
    printf("GPU Information\n");
    printf("=============================================================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 64);
    printf("Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("\n");

    // Run benchmark
    run_benchmark();

    return 0;
}
