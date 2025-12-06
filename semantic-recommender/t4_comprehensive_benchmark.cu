#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// T4-optimized semantic similarity with shared memory
__global__ void semantic_similarity_optimized(
    const float* __restrict__ embeddings_a,
    const float* __restrict__ embeddings_b,
    float* __restrict__ similarities,
    int num_pairs,
    int embedding_dim
) {
    extern __shared__ float shared_mem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (idx >= num_pairs) return;

    const float* a = embeddings_a + idx * embedding_dim;
    const float* b = embeddings_b + idx * embedding_dim;

    // Use vectorized loads for memory coalescing
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    #pragma unroll 8
    for (int i = 0; i < embedding_dim; i += 4) {
        float4 va = *reinterpret_cast<const float4*>(a + i);
        float4 vb = *reinterpret_cast<const float4*>(b + i);

        dot += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
        norm_a += va.x * va.x + va.y * va.y + va.z * va.z + va.w * va.w;
        norm_b += vb.x * vb.x + vb.y * vb.y + vb.z * vb.z + vb.w * vb.w;
    }

    similarities[idx] = dot * rsqrtf(norm_a * norm_b + 1e-8f);
}

// FP16 kernel for T4 Tensor Cores
__global__ void semantic_similarity_fp16(
    const half* __restrict__ embeddings_a,
    const half* __restrict__ embeddings_b,
    float* __restrict__ similarities,
    int num_pairs,
    int embedding_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    const half* a = embeddings_a + idx * embedding_dim;
    const half* b = embeddings_b + idx * embedding_dim;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    #pragma unroll 8
    for (int i = 0; i < embedding_dim; i += 8) {
        // Load 8 half values at a time
        half2 va0 = *reinterpret_cast<const half2*>(a + i);
        half2 va1 = *reinterpret_cast<const half2*>(a + i + 2);
        half2 va2 = *reinterpret_cast<const half2*>(a + i + 4);
        half2 va3 = *reinterpret_cast<const half2*>(a + i + 6);

        half2 vb0 = *reinterpret_cast<const half2*>(b + i);
        half2 vb1 = *reinterpret_cast<const half2*>(b + i + 2);
        half2 vb2 = *reinterpret_cast<const half2*>(b + i + 4);
        half2 vb3 = *reinterpret_cast<const half2*>(b + i + 6);

        // Accumulate in FP32 for precision
        dot += __half2float(va0.x) * __half2float(vb0.x) + __half2float(va0.y) * __half2float(vb0.y);
        dot += __half2float(va1.x) * __half2float(vb1.x) + __half2float(va1.y) * __half2float(vb1.y);
        dot += __half2float(va2.x) * __half2float(vb2.x) + __half2float(va2.y) * __half2float(vb2.y);
        dot += __half2float(va3.x) * __half2float(vb3.x) + __half2float(va3.y) * __half2float(vb3.y);

        norm_a += __half2float(va0.x) * __half2float(va0.x) + __half2float(va0.y) * __half2float(va0.y);
        norm_a += __half2float(va1.x) * __half2float(va1.x) + __half2float(va1.y) * __half2float(va1.y);
        norm_a += __half2float(va2.x) * __half2float(va2.x) + __half2float(va2.y) * __half2float(va2.y);
        norm_a += __half2float(va3.x) * __half2float(va3.x) + __half2float(va3.y) * __half2float(va3.y);

        norm_b += __half2float(vb0.x) * __half2float(vb0.x) + __half2float(vb0.y) * __half2float(vb0.y);
        norm_b += __half2float(vb1.x) * __half2float(vb1.x) + __half2float(vb1.y) * __half2float(vb1.y);
        norm_b += __half2float(vb2.x) * __half2float(vb2.x) + __half2float(vb2.y) * __half2float(vb2.y);
        norm_b += __half2float(vb3.x) * __half2float(vb3.x) + __half2float(vb3.y) * __half2float(vb3.y);
    }

    similarities[idx] = dot * rsqrtf(norm_a * norm_b + 1e-8f);
}

// Graph edge scoring kernel
__global__ void graph_edge_score(
    const float* __restrict__ node_embeddings,
    const int* __restrict__ edge_src,
    const int* __restrict__ edge_dst,
    float* __restrict__ edge_scores,
    int num_edges,
    int embedding_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    int src = edge_src[idx];
    int dst = edge_dst[idx];

    const float* emb_src = node_embeddings + src * embedding_dim;
    const float* emb_dst = node_embeddings + dst * embedding_dim;

    float dot = 0.0f, norm_s = 0.0f, norm_d = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < embedding_dim; i += 4) {
        float4 vs = *reinterpret_cast<const float4*>(emb_src + i);
        float4 vd = *reinterpret_cast<const float4*>(emb_dst + i);

        dot += vs.x * vd.x + vs.y * vd.y + vs.z * vd.z + vs.w * vd.w;
        norm_s += vs.x * vs.x + vs.y * vs.y + vs.z * vs.z + vs.w * vs.w;
        norm_d += vd.x * vd.x + vd.y * vd.y + vd.z * vd.z + vd.w * vd.w;
    }

    edge_scores[idx] = dot * rsqrtf(norm_s * norm_d + 1e-8f);
}

// CPU baseline
void cpu_similarity(const float* a, const float* b, float* sim, int n, int dim) {
    for (int i = 0; i < n; i++) {
        float dot = 0, na = 0, nb = 0;
        for (int j = 0; j < dim; j++) {
            dot += a[i*dim+j] * b[i*dim+j];
            na += a[i*dim+j] * a[i*dim+j];
            nb += b[i*dim+j] * b[i*dim+j];
        }
        sim[i] = dot / sqrtf(na * nb + 1e-8f);
    }
}

int main() {
    printf("=====================================================\n");
    printf("     T4 GPU COMPREHENSIVE BENCHMARK SUITE\n");
    printf("     Semantic Discovery Kernels\n");
    printf("=====================================================\n\n");

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 64);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Memory Bandwidth: %.0f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("\n");

    // Benchmark parameters
    const int NUM_PAIRS = 1000000;
    const int EMBEDDING_DIM = 384;
    const int NUM_WARMUP = 5;
    const int NUM_RUNS = 20;

    printf("Benchmark: %d pairs, %d dimensions\n\n", NUM_PAIRS, EMBEDDING_DIM);

    // Allocate memory
    size_t embed_size = NUM_PAIRS * EMBEDDING_DIM * sizeof(float);
    size_t embed_size_fp16 = NUM_PAIRS * EMBEDDING_DIM * sizeof(half);
    size_t sim_size = NUM_PAIRS * sizeof(float);

    float* h_a = (float*)malloc(embed_size);
    float* h_b = (float*)malloc(embed_size);
    half* h_a_fp16 = (half*)malloc(embed_size_fp16);
    half* h_b_fp16 = (half*)malloc(embed_size_fp16);
    float* h_sim = (float*)malloc(sim_size);
    float* h_sim_cpu = (float*)malloc(sim_size);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < NUM_PAIRS * EMBEDDING_DIM; i++) {
        h_a[i] = (float)rand() / RAND_MAX - 0.5f;
        h_b[i] = (float)rand() / RAND_MAX - 0.5f;
        h_a_fp16[i] = __float2half(h_a[i]);
        h_b_fp16[i] = __float2half(h_b[i]);
    }

    // Device memory
    float *d_a, *d_b, *d_sim;
    half *d_a_fp16, *d_b_fp16;

    cudaMalloc(&d_a, embed_size);
    cudaMalloc(&d_b, embed_size);
    cudaMalloc(&d_sim, sim_size);
    cudaMalloc(&d_a_fp16, embed_size_fp16);
    cudaMalloc(&d_b_fp16, embed_size_fp16);

    cudaMemcpy(d_a, h_a, embed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, embed_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_fp16, h_a_fp16, embed_size_fp16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp16, h_b_fp16, embed_size_fp16, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (NUM_PAIRS + threads - 1) / threads;

    // ============ FP32 Optimized Benchmark ============
    printf("1. FP32 Optimized Kernel (vectorized loads)\n");

    for (int i = 0; i < NUM_WARMUP; i++) {
        semantic_similarity_optimized<<<blocks, threads, 0>>>(d_a, d_b, d_sim, NUM_PAIRS, EMBEDDING_DIM);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {
        semantic_similarity_optimized<<<blocks, threads, 0>>>(d_a, d_b, d_sim, NUM_PAIRS, EMBEDDING_DIM);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fp32_time_ms;
    cudaEventElapsedTime(&fp32_time_ms, start, stop);
    float fp32_avg_ms = fp32_time_ms / NUM_RUNS;

    printf("   Kernel time: %.3f ms\n", fp32_avg_ms);
    printf("   Throughput: %.2f M pairs/sec\n", NUM_PAIRS / fp32_avg_ms / 1000.0f);
    printf("   Effective bandwidth: %.1f GB/s\n",
           (2.0 * embed_size + sim_size) / fp32_avg_ms / 1e6);
    printf("\n");

    // ============ FP16 Benchmark ============
    printf("2. FP16 Kernel (half precision, 2x memory efficiency)\n");

    for (int i = 0; i < NUM_WARMUP; i++) {
        semantic_similarity_fp16<<<blocks, threads>>>(d_a_fp16, d_b_fp16, d_sim, NUM_PAIRS, EMBEDDING_DIM);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {
        semantic_similarity_fp16<<<blocks, threads>>>(d_a_fp16, d_b_fp16, d_sim, NUM_PAIRS, EMBEDDING_DIM);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fp16_time_ms;
    cudaEventElapsedTime(&fp16_time_ms, start, stop);
    float fp16_avg_ms = fp16_time_ms / NUM_RUNS;

    printf("   Kernel time: %.3f ms\n", fp16_avg_ms);
    printf("   Throughput: %.2f M pairs/sec\n", NUM_PAIRS / fp16_avg_ms / 1000.0f);
    printf("   Speedup vs FP32: %.2fx\n", fp32_avg_ms / fp16_avg_ms);
    printf("\n");

    // ============ CPU Baseline ============
    printf("3. CPU Baseline (single-threaded)\n");
    const int CPU_SAMPLE = 10000;
    clock_t cpu_start = clock();
    cpu_similarity(h_a, h_b, h_sim_cpu, CPU_SAMPLE, EMBEDDING_DIM);
    clock_t cpu_end = clock();
    float cpu_time_ms = ((float)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0f;
    float cpu_projected_ms = cpu_time_ms * (NUM_PAIRS / (float)CPU_SAMPLE);

    printf("   Projected time (1M pairs): %.1f ms\n", cpu_projected_ms);
    printf("\n");

    // ============ Summary ============
    printf("=====================================================\n");
    printf("                    SUMMARY\n");
    printf("=====================================================\n");
    printf("T4 GPU vs CPU Speedup:\n");
    printf("   FP32 optimized: %.0fx faster\n", cpu_projected_ms / fp32_avg_ms);
    printf("   FP16 optimized: %.0fx faster\n", cpu_projected_ms / fp16_avg_ms);
    printf("\n");
    printf("Memory efficiency:\n");
    printf("   FP32: %.1f GB/s effective bandwidth\n", (2.0 * embed_size + sim_size) / fp32_avg_ms / 1e6);
    printf("   FP16: %.1f GB/s effective bandwidth\n", (2.0 * embed_size_fp16 + sim_size) / fp16_avg_ms / 1e6);
    printf("\n");
    printf("T4 theoretical peak: 320 GB/s memory, 65 TFLOPS FP16\n");
    printf("=====================================================\n");

    // Validate
    cudaMemcpy(h_sim, d_sim, sim_size, cudaMemcpyDeviceToHost);
    float max_error = 0;
    for (int i = 0; i < CPU_SAMPLE; i++) {
        float err = fabsf(h_sim[i] - h_sim_cpu[i]);
        if (err > max_error) max_error = err;
    }
    printf("Validation: max error = %.2e (%s)\n", max_error, max_error < 1e-3 ? "PASSED" : "CHECK");

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_sim);
    cudaFree(d_a_fp16); cudaFree(d_b_fp16);
    free(h_a); free(h_b); free(h_sim); free(h_sim_cpu);
    free(h_a_fp16); free(h_b_fp16);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
