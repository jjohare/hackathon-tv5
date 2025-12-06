# CUDA Kernel Build System

## Overview

This build system compiles CUDA kernels to PTX format for runtime loading in Rust. PTX (Parallel Thread Execution) is NVIDIA's intermediate assembly language that can be loaded dynamically without requiring recompilation.

## Architecture

- **Target**: T4 GPU (sm_75 - Turing architecture)
- **Features**: FP16 Tensor Cores, 2560 CUDA cores, 16GB GDDR6
- **Output**: PTX files in `target/ptx/` directory

## Build Components

### 1. Makefile (`src/cuda/kernels/Makefile`)

**Key Features:**
- Compiles 7 critical kernels to PTX format
- T4-optimized flags: `-arch=sm_75` with Tensor Core support
- Extended lambda support for modern CUDA code
- Automatic dependency management

**PTX Compilation Flags:**
```makefile
NVCC_PTX_FLAGS := -ptx -O3 -std=c++17 -arch=sm_75 \
                  --use_fast_math --expt-relaxed-constexpr --extended-lambda \
                  -Xptxas -v -Xcompiler -fPIC \
                  -DUSE_TENSOR_CORES=1 -DUSE_FP16=1
```

### 2. Build Script (`scripts/build-cuda-kernels.sh`)

Automated build script with comprehensive validation:

**Features:**
- CUDA installation verification
- GPU detection
- Parallel PTX compilation
- Size analysis
- Manifest generation

**Usage:**
```bash
./scripts/build-cuda-kernels.sh
```

### 3. Cargo Integration (`build.rs`)

Integrates PTX compilation into Rust build process:

**Features:**
- Automatic PTX compilation during `cargo build`
- Environment variable setup (`PTX_DIR`)
- CUDA library linking
- Graceful fallback if nvcc unavailable

**Usage:**
```bash
cargo build          # Automatically compiles PTX
cargo build --release
```

## Critical Kernels

### Successfully Compiled (7 kernels):

1. **semantic_similarity_fp16_tensor_cores.ptx** (38KB)
   - FP16 tensor core acceleration
   - Primary kernel for semantic search
   - 65 TFLOPS FP16 performance on T4

2. **graph_search.ptx** (42KB)
   - GPU-accelerated graph traversal
   - Semantic-aware pathfinding

3. **ontology_reasoning.ptx** (39KB)
   - Knowledge graph reasoning
   - Relationship inference

4. **hybrid_sssp.ptx** (22KB)
   - Hybrid Single-Source Shortest Path
   - CPU/GPU coordination

5. **semantic_similarity.ptx** (94KB)
   - FP32 semantic similarity
   - Fallback for non-tensor-core GPUs

6. **semantic_similarity_fp16.ptx** (53KB)
   - FP16 without tensor cores
   - Memory-optimized variant

7. **product_quantization.ptx** (41KB)
   - Vector compression
   - Reduced memory footprint

### Total Size: 348KB

## Usage in Rust

### Method 1: Include PTX at Compile Time

```rust
use cuda_driver_sys::*;
use std::ffi::CString;

// Include PTX directly in binary
const PTX: &str = include_str!("../target/ptx/semantic_similarity_fp16_tensor_cores.ptx");

fn load_kernel() -> Result<CUmodule, CUresult> {
    let mut module: CUmodule = std::ptr::null_mut();

    unsafe {
        // Load PTX module
        let ptx_cstr = CString::new(PTX).unwrap();
        cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _)?;

        // Get kernel function
        let mut kernel: CUfunction = std::ptr::null_mut();
        let name = CString::new("compute_semantic_similarity_fp16_tensor_cores").unwrap();
        cuModuleGetFunction(&mut kernel, module, name.as_ptr())?;
    }

    Ok(module)
}
```

### Method 2: Load PTX at Runtime

```rust
use std::fs;
use std::path::PathBuf;

fn load_kernel_runtime() -> Result<CUmodule, Box<dyn std::error::Error>> {
    let ptx_path = PathBuf::from(env!("PTX_DIR"))
        .join("semantic_similarity_fp16_tensor_cores.ptx");

    let ptx = fs::read_to_string(ptx_path)?;
    let ptx_cstr = CString::new(ptx)?;

    let mut module: CUmodule = std::ptr::null_mut();
    unsafe {
        cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _)?;
    }

    Ok(module)
}
```

### Method 3: Use cuModuleLoadDataEx with Options

```rust
fn load_kernel_optimized() -> Result<CUmodule, CUresult> {
    let ptx = include_str!("../target/ptx/semantic_similarity_fp16_tensor_cores.ptx");
    let ptx_cstr = CString::new(ptx).unwrap();

    let mut module: CUmodule = std::ptr::null_mut();

    // JIT options for optimization
    let options = [
        CUjit_option::CU_JIT_MAX_REGISTERS,
        CUjit_option::CU_JIT_TARGET,
    ];
    let values = [128, CUjit_target::CU_TARGET_COMPUTE_75 as u32];

    unsafe {
        cuModuleLoadDataEx(
            &mut module,
            ptx_cstr.as_ptr() as *const _,
            options.len() as u32,
            options.as_ptr() as *mut _,
            values.as_ptr() as *mut _,
        )?;
    }

    Ok(module)
}
```

## Build Commands

### Makefile Targets

```bash
# Build PTX for Rust FFI
make ptx

# Build complete kernel library
make all

# Generate SASS assembly (inspect optimizations)
make sass

# Check register usage (occupancy analysis)
make check-registers

# Profile with Nsight Compute
make profile

# Clean build artifacts
make clean

# Run T4 validation tests
make test-t4

# Memory analysis
make memory-check
```

### Direct Build

```bash
# Navigate to kernel directory
cd src/cuda/kernels

# Compile single kernel
nvcc -ptx -O3 -std=c++17 -arch=sm_75 \
     --use_fast_math --expt-relaxed-constexpr --extended-lambda \
     -Xptxas -v -Xcompiler -fPIC \
     -DUSE_TENSOR_CORES=1 -DUSE_FP16=1 \
     semantic_similarity_fp16_tensor_cores.cu \
     -o ../../target/ptx/semantic_similarity_fp16_tensor_cores.ptx
```

## Performance Characteristics

### T4 GPU Specifications
- **Architecture**: Turing (sm_75)
- **CUDA Cores**: 2560 (40 SMs × 64 cores/SM)
- **Tensor Cores**: 320 (FP16 only)
- **Memory**: 16GB GDDR6 @ 320 GB/s
- **FP16 Peak**: 65 TFLOPS
- **FP32 Peak**: 8.1 TFLOPS

### Optimization Flags Explained

- `-arch=sm_75`: Target T4 architecture
- `-use_fast_math`: Enable fast math approximations
- `--expt-relaxed-constexpr`: Allow complex constexpr
- `--extended-lambda`: Support `__device__` lambdas
- `-Xptxas -v`: Verbose PTX assembly info
- `-Xcompiler -fPIC`: Position-independent code
- `-DUSE_TENSOR_CORES=1`: Enable tensor core paths
- `-DUSE_FP16=1`: Enable FP16 optimizations

## Troubleshooting

### CUDA Not Found

```bash
# Set CUDA_PATH environment variable
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### PTX Compilation Errors

**Symptom**: "extern C linkage" errors
**Solution**: Exclude problematic kernels from PTX_KERNELS in Makefile

**Symptom**: "cannot perform wmma load on local memory"
**Solution**: Warnings only - safe to ignore for local memory operations

**Symptom**: "kernel launch requires separate compilation"
**Solution**: Use device-side kernel compilation or exclude from PTX

### Cargo Build Issues

```bash
# Skip CUDA compilation
cargo build --no-default-features

# Force rebuild
cargo clean
./scripts/build-cuda-kernels.sh
cargo build
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: CUDA Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:13.0-devel-ubuntu22.04

    steps:
      - uses: actions/checkout@v4

      - name: Build CUDA Kernels
        run: ./scripts/build-cuda-kernels.sh

      - name: Verify PTX
        run: |
          test -f target/ptx/semantic_similarity_fp16_tensor_cores.ptx
          test -f target/ptx/graph_search.ptx
          test -f target/ptx/ontology_reasoning.ptx
          test -f target/ptx/hybrid_sssp.ptx

      - name: Build Rust
        run: cargo build --release
```

## Manifest File

Generated at `target/ptx/manifest.txt`:

```
# PTX Manifest
# Generated: 2025-12-04 16:39:45 UTC
# Target: T4 GPU (sm_75) with Tensor Core support
# Compiler: 13.0

graph_search.ptx 42809
hybrid_sssp.ptx 22281
ontology_reasoning.ptx 39477
product_quantization.ptx 41085
semantic_similarity.ptx 95378
semantic_similarity_fp16.ptx 53349
semantic_similarity_fp16_tensor_cores.ptx 38527
```

## Advanced Usage

### Profile PTX Compilation

```bash
nvcc -ptx -arch=sm_75 kernel.cu -o kernel.ptx --ptxas-options=-v
```

**Output Analysis:**
- Register usage per thread
- Shared memory usage
- Constant memory usage
- Theoretical occupancy

### Inspect Generated PTX

```bash
less target/ptx/semantic_similarity_fp16_tensor_cores.ptx
```

**Key Sections:**
- `.version`: PTX version
- `.target`: Target architecture
- `.visible .entry`: Kernel entry points
- `wmma.*`: Tensor core instructions

### Generate CUBIN (Binary)

```bash
make sass
# Creates .cubin files for direct binary loading (faster than PTX)
```

## References

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [T4 GPU Specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
