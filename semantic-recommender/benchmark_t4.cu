#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Simple semantic similarity kernel for benchmarking
__global__ void semantic_similarity_benchmark(
    const float* embeddings_a,
    const float* embeddings_b,
    float* similarities,
    int num_pairs,
    int embedding_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    const float* a = embeddings_a + idx * embedding_dim;
    const float* b = embeddings_b + idx * embedding_dim;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    similarities[idx] = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

// CPU baseline for comparison
void cpu_similarity(const float* a, const float* b, float* sim, int n, int dim) {
    for (int i = 0; i < n; i++) {
        float dot = 0, na = 0, nb = 0;
        for (int j = 0; j < dim; j++) {
            dot += a[i*dim+j] * b[i*dim+j];
            na += a[i*dim+j] * a[i*dim+j];
            nb += b[i*dim+j] * b[i*dim+j];
        }
        sim[i] = dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);
    }
}

int main() {
    // Benchmark parameters
    const int NUM_PAIRS = 1000000;  // 1M similarity computations
    const int EMBEDDING_DIM = 384;  // Typical embedding size
    const int NUM_WARMUP = 3;
    const int NUM_RUNS = 10;

    printf("=== T4 GPU Semantic Similarity Benchmark ===\n");
    printf("Pairs: %d, Embedding dim: %d\n\n", NUM_PAIRS, EMBEDDING_DIM);

    // Allocate host memory
    size_t embed_size = NUM_PAIRS * EMBEDDING_DIM * sizeof(float);
    size_t sim_size = NUM_PAIRS * sizeof(float);

    float* h_a = (float*)malloc(embed_size);
    float* h_b = (float*)malloc(embed_size);
    float* h_sim_gpu = (float*)malloc(sim_size);
    float* h_sim_cpu = (float*)malloc(sim_size);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < NUM_PAIRS * EMBEDDING_DIM; i++) {
        h_a[i] = (float)rand() / RAND_MAX - 0.5f;
        h_b[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_sim;
    cudaMalloc(&d_a, embed_size);
    cudaMalloc(&d_b, embed_size);
    cudaMalloc(&d_sim, sim_size);

    // Copy to device
    cudaMemcpy(d_a, h_a, embed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, embed_size, cudaMemcpyHostToDevice);

    // GPU benchmark
    int threads = 256;
    int blocks = (NUM_PAIRS + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        semantic_similarity_benchmark<<<blocks, threads>>>(d_a, d_b, d_sim, NUM_PAIRS, EMBEDDING_DIM);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {
        semantic_similarity_benchmark<<<blocks, threads>>>(d_a, d_b, d_sim, NUM_PAIRS, EMBEDDING_DIM);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    float gpu_avg_ms = gpu_time_ms / NUM_RUNS;

    // Copy results back
    cudaMemcpy(h_sim_gpu, d_sim, sim_size, cudaMemcpyDeviceToHost);

    // CPU benchmark (smaller sample for time)
    const int CPU_SAMPLE = 10000;
    clock_t cpu_start = clock();
    cpu_similarity(h_a, h_b, h_sim_cpu, CPU_SAMPLE, EMBEDDING_DIM);
    clock_t cpu_end = clock();
    float cpu_time_ms = ((float)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0f;
    float cpu_projected_ms = cpu_time_ms * (NUM_PAIRS / (float)CPU_SAMPLE);

    // Validate results
    float max_error = 0;
    for (int i = 0; i < CPU_SAMPLE; i++) {
        float err = fabsf(h_sim_gpu[i] - h_sim_cpu[i]);
        if (err > max_error) max_error = err;
    }

    // Report results
    printf("GPU (T4) Performance:\n");
    printf("  Average kernel time: %.3f ms\n", gpu_avg_ms);
    printf("  Throughput: %.2f M pairs/sec\n", NUM_PAIRS / gpu_avg_ms / 1000.0f);
    printf("  GFLOPS: %.2f\n", (NUM_PAIRS * EMBEDDING_DIM * 4.0f) / gpu_avg_ms / 1e6f);
    printf("\n");

    printf("CPU Baseline (projected from %d samples):\n", CPU_SAMPLE);
    printf("  Projected time: %.1f ms\n", cpu_projected_ms);
    printf("\n");

    printf("Speedup: %.1fx faster on T4\n", cpu_projected_ms / gpu_avg_ms);
    printf("Max error: %.2e (validation: %s)\n", max_error, max_error < 1e-5 ? "PASSED" : "CHECK");

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_sim);
    free(h_a); free(h_b); free(h_sim_gpu); free(h_sim_cpu);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
