# GPU Setup and CUDA Development Guide

**Target Audience**: DevOps Engineers, CUDA Developers
**Prerequisites**: Linux server access, NVIDIA GPU (A100/H100 recommended)
**Estimated Setup Time**: 2-4 hours

---

## Table of Contents

1. [CUDA Toolkit Installation](#1-cuda-toolkit-installation)
2. [Rust CUDA Bindings Setup](#2-rust-cuda-bindings-setup)
3. [Testing GPU Kernels](#3-testing-gpu-kernels)
4. [Performance Optimization](#4-performance-optimization)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. CUDA Toolkit Installation

### 1.1 System Requirements

**Hardware:**
- NVIDIA GPU: A100 (40/80GB), H100 (80GB), or V100 (32GB)
- System RAM: 64GB minimum, 128GB recommended
- Storage: 500GB NVMe SSD for temporary processing

**Software:**
- OS: Ubuntu 22.04 LTS (recommended) or RHEL 8/9
- Kernel: 5.15+ with NVIDIA driver support
- GCC: 11.x or 12.x

### 1.2 Driver Installation

```bash
# Remove existing NVIDIA drivers
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge '^libnvidia-.*'
sudo apt-get autoremove

# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NVIDIA driver (version 535+ for A100/H100)
sudo apt-get install -y nvidia-driver-535

# Reboot to load driver
sudo reboot
```

**Verify Installation:**
```bash
nvidia-smi
# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   32C    P0    49W / 400W |      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 1.3 CUDA Toolkit 12.2+ Installation

```bash
# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-2

# Install cuDNN (for deep learning)
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y cudnn

# Install NVIDIA cuVS (Vector Search)
sudo apt-get install -y libcuvs-dev
```

### 1.4 Environment Configuration

Add to `~/.bashrc`:
```bash
# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# cuDNN paths
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

# Compiler settings
export CC=gcc-12
export CXX=g++-12
```

Apply changes:
```bash
source ~/.bashrc
```

### 1.5 Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version
# Expected: Cuda compilation tools, release 12.2, V12.2.140

# Compile sample program
cd $CUDA_HOME/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
# Expected: Result = PASS

# Test CUDA runtime
cat > test_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello_cuda<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

nvcc test_cuda.cu -o test_cuda
./test_cuda
# Expected:
# Hello from GPU thread 0
# Hello from GPU thread 1
# Hello from GPU thread 2
# Hello from GPU thread 3
```

---

## 2. Rust CUDA Bindings Setup

### 2.1 Install Rust Toolchain

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install nightly toolchain (required for CUDA)
rustup toolchain install nightly
rustup default nightly

# Verify installation
rustc --version
cargo --version
```

### 2.2 Install Rust-CUDA

```bash
# Install LLVM (required for ptx compilation)
sudo apt-get install -y llvm-14 llvm-14-dev clang-14

# Install rust-cuda toolchain
cargo install cargo-make

# Create project with CUDA support
cargo new --lib semantic-gpu
cd semantic-gpu
```

### 2.3 Configure Cargo.toml

```toml
[package]
name = "semantic-gpu"
version = "0.1.0"
edition = "2021"

[dependencies]
cuda-sys = "0.3"
cudarc = { version = "0.10", features = ["cuda-12"] }
half = "2.3"  # FP16 support
ndarray = "0.15"

[dev-dependencies]
criterion = "0.5"

[lib]
crate-type = ["cdylib", "rlib"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### 2.4 Create CUDA Kernel (semantic_forces.cu)

```cuda
// src/kernels/semantic_forces.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Cosine similarity using tensor cores (FP16)
__device__ float cosine_similarity_tensorcore(
    const half* vec_a,
    const half* vec_b,
    int dim
) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    // Use warp-level primitives for efficient reduction
    for (int i = threadIdx.x; i < dim; i += warpSize) {
        float a = __half2float(vec_a[i]);
        float b = __half2float(vec_b[i]);

        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    // Warp shuffle reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_a += __shfl_down_sync(0xffffffff, norm_a, offset);
        norm_b += __shfl_down_sync(0xffffffff, norm_b, offset);
    }

    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

// Compute semantic forces between embeddings
__global__ void semantic_forces_kernel(
    const half* embeddings,      // [N, D] embeddings (FP16)
    const float* color_vectors,  // [N, 64] color features
    float* forces,               // [N, N] output force matrix
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N || i == j) return;

    // Shared memory for coalesced access
    __shared__ half embedding_i[1024];
    __shared__ half embedding_j[1024];

    // Load embeddings into shared memory
    if (threadIdx.y == 0 && threadIdx.x < D) {
        embedding_i[threadIdx.x] = embeddings[i * D + threadIdx.x];
    }
    if (threadIdx.x == 0 && threadIdx.y < D) {
        embedding_j[threadIdx.y] = embeddings[j * D + threadIdx.y];
    }
    __syncthreads();

    // Compute semantic similarity
    float sim = cosine_similarity_tensorcore(embedding_i, embedding_j, D);

    // Compute color distance
    float color_dist = 0.0f;
    for (int k = 0; k < 64; k++) {
        float diff = color_vectors[i * 64 + k] - color_vectors[j * 64 + k];
        color_dist += diff * diff;
    }
    color_dist = sqrtf(color_dist);

    // Semantic force: similarity / (distance^2 + epsilon)
    // Attractive force if similar, repulsive if different
    forces[i * N + j] = sim / (color_dist * color_dist + 1e-6f);
}

// Host function to launch kernel
extern "C" void compute_semantic_forces(
    const half* embeddings,
    const float* color_vectors,
    float* forces,
    int N,
    int D,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    semantic_forces_kernel<<<grid, block, 0, stream>>>(
        embeddings, color_vectors, forces, N, D
    );
}
```

### 2.5 Rust Wrapper (lib.rs)

```rust
// src/lib.rs
use cudarc::driver::*;
use cudarc::nvrtc::*;
use half::f16;
use std::sync::Arc;

pub struct SemanticGPU {
    device: Arc<CudaDevice>,
    module: CudaModule,
}

impl SemanticGPU {
    pub fn new() -> Result<Self, DriverError> {
        // Initialize CUDA device
        let device = CudaDevice::new(0)?;

        // Load PTX module
        let ptx = compile_ptx(include_str!("kernels/semantic_forces.cu"))?;
        let module = device.load_ptx(ptx, "semantic_forces", &["compute_semantic_forces"])?;

        Ok(Self { device, module })
    }

    pub fn compute_forces(
        &self,
        embeddings: &[f16],  // [N, D] flattened
        color_vectors: &[f32],  // [N, 64] flattened
        n: usize,
        d: usize,
    ) -> Result<Vec<f32>, DriverError> {
        // Allocate device memory
        let d_embeddings = self.device.htod_copy(embeddings)?;
        let d_colors = self.device.htod_copy(color_vectors)?;
        let d_forces = self.device.alloc_zeros::<f32>(n * n)?;

        // Get kernel function
        let func = self.module.get_func("compute_semantic_forces")?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((n + 15) / 16, (n + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_embeddings,
                &d_colors,
                &d_forces,
                n as i32,
                d as i32,
                0u64,  // default stream
            ))?;
        }

        // Copy result back to host
        let forces = self.device.dtoh_sync_copy(&d_forces)?;
        Ok(forces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_forces() {
        let gpu = SemanticGPU::new().expect("Failed to initialize GPU");

        // Create test data
        let n = 100;
        let d = 1024;
        let embeddings: Vec<f16> = (0..n*d).map(|i| f16::from_f32(i as f32 / (n*d) as f32)).collect();
        let colors: Vec<f32> = (0..n*64).map(|i| i as f32 / 64.0).collect();

        // Compute forces
        let forces = gpu.compute_forces(&embeddings, &colors, n, d).expect("Kernel failed");

        // Verify output
        assert_eq!(forces.len(), n * n);
        println!("Force[0,1] = {}", forces[1]);
    }
}
```

### 2.6 Compile PTX Module

Create `build.rs`:
```rust
// build.rs
use std::process::Command;

fn main() {
    // Compile CUDA kernel to PTX
    let output = Command::new("nvcc")
        .args(&[
            "--ptx",
            "-O3",
            "-arch=sm_80",  // A100
            "--use_fast_math",
            "-o",
            "target/semantic_forces.ptx",
            "src/kernels/semantic_forces.cu",
        ])
        .output()
        .expect("Failed to compile CUDA kernel");

    if !output.status.success() {
        panic!("CUDA compilation failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    println!("cargo:rerun-if-changed=src/kernels/semantic_forces.cu");
}
```

---

## 3. Testing GPU Kernels

### 3.1 Unit Tests

```rust
// tests/kernel_tests.rs
use semantic_gpu::*;

#[test]
fn test_kernel_correctness() {
    let gpu = SemanticGPU::new().unwrap();

    // Simple test case: 2 identical embeddings
    let n = 2;
    let d = 1024;
    let mut embeddings = vec![f16::from_f32(0.0); n * d];

    // Embedding 0: all 0.5
    for i in 0..d {
        embeddings[i] = f16::from_f32(0.5);
    }

    // Embedding 1: all 0.5 (identical)
    for i in 0..d {
        embeddings[d + i] = f16::from_f32(0.5);
    }

    let colors = vec![1.0; n * 64];

    let forces = gpu.compute_forces(&embeddings, &colors, n, d).unwrap();

    // Identical embeddings should have max similarity (force → ∞)
    assert!(forces[1] > 100.0, "Expected high force for identical embeddings");
}
```

### 3.2 Performance Benchmarks

```rust
// benches/kernel_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use semantic_gpu::*;

fn bench_semantic_forces(c: &mut Criterion) {
    let gpu = SemanticGPU::new().unwrap();

    let mut group = c.benchmark_group("semantic_forces");
    for n in [100, 1000, 10000] {
        let d = 1024;
        let embeddings: Vec<f16> = (0..n*d).map(|i| f16::from_f32(i as f32 / (n*d) as f32)).collect();
        let colors: Vec<f32> = (0..n*64).map(|i| i as f32 / 64.0).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                gpu.compute_forces(black_box(&embeddings), black_box(&colors), n, d).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_semantic_forces);
criterion_main!(benches);
```

Run benchmarks:
```bash
cargo bench
# Expected output:
# semantic_forces/100     time: [245.2 µs 247.8 µs 250.6 µs]
# semantic_forces/1000    time: [2.458 ms 2.471 ms 2.485 ms]
# semantic_forces/10000   time: [245.1 ms 247.5 ms 250.2 ms]
```

### 3.3 Profiling with NVIDIA Nsight

```bash
# Profile kernel execution
nsys profile --stats=true ./target/release/semantic-gpu-bench

# Analyze GPU utilization
ncu --set full ./target/release/semantic-gpu-bench
```

**Key Metrics:**
- **GPU Utilization**: Target >85%
- **Memory Bandwidth**: Target >80% of peak (1.6 TB/s on A100)
- **Occupancy**: Target >75% active warps
- **Kernel Duration**: <5ms per kernel

---

## 4. Performance Optimization

### 4.1 Tensor Core Utilization

```cuda
// Use WMMA API for tensor core acceleration
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensor_core_similarity(
    const half* queries,    // [M, D]
    const half* embeddings, // [N, D]
    float* output,          // [M, N]
    int M, int N, int D
) {
    // 16x16x16 matrix multiply (tensor core)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y) / 32;

    load_matrix_sync(a_frag, queries + warp_m * 16 * D, D);
    load_matrix_sync(b_frag, embeddings + warp_n * 16 * D, D);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(output + warp_m * 16 * N + warp_n * 16, c_frag, N, mem_row_major);
}
```

**Expected Speedup**: 16x vs CUDA cores (312 TFLOPS FP16 on A100)

### 4.2 Memory Coalescing

```cuda
// BAD: Strided access pattern
for (int i = 0; i < N; i++) {
    result += data[i * stride];  // Poor coalescing
}

// GOOD: Contiguous access pattern
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    result += data[i];  // Coalesced access
}
```

### 4.3 Occupancy Optimization

```bash
# Check occupancy with Nsight Compute
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./kernel

# Optimize via:
# 1. Reduce register usage (compile with -maxrregcount 64)
# 2. Reduce shared memory usage (<48KB per block)
# 3. Increase threads per block (256-512 recommended)
```

### 4.4 Multi-Stream Processing

```rust
// Process batches in parallel streams
pub fn compute_forces_batched(
    &self,
    embeddings: &[f16],
    colors: &[f32],
    batch_size: usize,
) -> Result<Vec<f32>, DriverError> {
    let num_batches = (embeddings.len() / (batch_size * 1024) + 1) as usize;
    let streams: Vec<_> = (0..num_batches)
        .map(|_| self.device.fork_default_stream())
        .collect::<Result<_, _>>()?;

    let mut results = Vec::new();
    for (i, stream) in streams.iter().enumerate() {
        let start = i * batch_size * 1024;
        let end = ((i + 1) * batch_size * 1024).min(embeddings.len());

        let result = self.compute_forces_async(
            &embeddings[start..end],
            &colors[start/16..end/16],
            stream,
        )?;
        results.push(result);
    }

    // Synchronize all streams
    for stream in streams {
        stream.synchronize()?;
    }

    Ok(results.concat())
}
```

---

## 5. Troubleshooting

### 5.1 Common Errors

#### Error: `CUDA driver version insufficient`
**Cause**: Mismatched driver/toolkit versions
**Solution**:
```bash
# Check versions
nvidia-smi  # Driver version
nvcc --version  # Toolkit version

# Upgrade driver if needed
sudo apt-get install --only-upgrade nvidia-driver-535
```

#### Error: `PTX compilation failed`
**Cause**: Missing LLVM or incompatible CUDA version
**Solution**:
```bash
# Install LLVM 14
sudo apt-get install llvm-14-dev clang-14

# Set environment variables
export LLVM_CONFIG=/usr/bin/llvm-config-14
```

#### Error: `Out of memory`
**Cause**: Insufficient GPU memory
**Solution**:
```rust
// Use streaming to process in chunks
let chunk_size = 10000;
for chunk in embeddings.chunks(chunk_size) {
    gpu.compute_forces(chunk, colors, chunk_size / 1024, 1024)?;
}
```

### 5.2 Performance Issues

#### Low GPU Utilization (<50%)
**Diagnosis**:
```bash
nvidia-smi dmon -s pucvmet
```

**Potential Causes**:
1. **Small batch size**: Increase to 256-512
2. **CPU bottleneck**: Profile host code
3. **Memory bandwidth**: Use texture memory for read-only data

#### High Latency (>10ms per kernel)
**Diagnosis**:
```bash
ncu --metrics gpu__time_duration.sum ./kernel
```

**Potential Causes**:
1. **Poor occupancy**: Reduce register usage
2. **Branch divergence**: Minimize conditional code
3. **Uncoalesced memory**: Reorder data layout

### 5.3 Debugging Tools

```bash
# CUDA-GDB for kernel debugging
cuda-gdb ./target/debug/semantic-gpu
(cuda-gdb) break semantic_forces_kernel
(cuda-gdb) run
(cuda-gdb) info cuda threads

# CUDA-MEMCHECK for memory errors
cuda-memcheck --leak-check full ./kernel

# Nsight Systems for timeline profiling
nsys profile --trace=cuda,nvtx --output=report.nsys-rep ./kernel
# Open in Nsight Systems GUI
```

---

## Performance Targets

**Target Metrics (A100 40GB):**
- Semantic forces (N=10K, D=1024): <50ms
- Tensor core utilization: >80%
- Memory bandwidth: >1.2 TB/s (>75% of peak)
- GPU utilization: >85%
- Kernel occupancy: >75%

**Optimization Checklist:**
- [ ] Tensor cores enabled (WMMA/MMA API)
- [ ] Memory access coalesced (128-byte aligned)
- [ ] Shared memory usage <48KB per block
- [ ] Occupancy >75% (check with Nsight Compute)
- [ ] Multi-stream processing implemented
- [ ] FP16 precision used where applicable
- [ ] Batch size optimized (256-512 items)

---

**Next Steps:**
1. Implement `ontology_constraints.cu` kernel
2. Integrate with vector search pipeline
3. Deploy to production GPU cluster
4. Set up monitoring (Prometheus + Grafana)

**Related Guides:**
- [Vector Search Implementation Guide](vector-search-implementation.md)
- [Ontology Reasoning Guide](ontology-reasoning-guide.md)
