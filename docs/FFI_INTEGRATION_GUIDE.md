# FFI Integration Guide: CUDA ↔ Rust Ontology Structures

**Purpose**: Step-by-step guide to integrate FFI-safe structures into the build system
**Status**: Ready for implementation
**Last Updated**: 2025-12-04

---

## Overview

This guide provides instructions for integrating the newly created FFI-safe structures between CUDA and Rust, ensuring binary compatibility and compile-time safety guarantees.

---

## Files Created

### Rust Layer
1. **`src/rust/models/ontology_ffi.rs`** (529 lines)
   - FFI-safe struct definitions
   - Compile-time size/alignment assertions
   - Helper methods and constants

2. **`src/rust/models/ontology_ffi_tests.rs`** (447 lines)
   - Comprehensive test suite
   - Serialization tests
   - GPU transfer simulations
   - Platform verification

### CUDA Layer
3. **`src/cuda/kernels/ontology_ffi_check.cuh`** (154 lines)
   - Static assertions for CUDA structs
   - Field offset verification
   - Platform compatibility checks

### Documentation
4. **`docs/FFI_ALIGNMENT_REPORT.md`** (Comprehensive audit)
5. **`docs/FFI_INTEGRATION_GUIDE.md`** (This file)

---

## Quick Start

### Prerequisites

```bash
# Rust toolchain
rustc --version  # Should be 1.70+

# CUDA toolkit
nvcc --version   # Should be 11.0+

# Build tools
cargo --version
cmake --version  # Optional, if using CMake
```

---

## Step 1: Update Rust Build

### 1.1 Module Integration (✅ COMPLETED)

The following has already been added to `src/rust/models/mod.rs`:

```rust
pub mod ontology_ffi;

#[cfg(test)]
mod ontology_ffi_tests;
```

### 1.2 Run Rust Tests

```bash
cd /home/devuser/workspace/hackathon-tv5/src/rust

# Run all FFI tests
cargo test ontology_ffi -- --nocapture

# Expected output:
# running 12 tests
# test ontology_ffi::tests::test_media_ontology_node_size ... ok
# test ontology_ffi::tests::test_field_offsets ... ok
# ... (all 12 tests should pass)
```

### 1.3 Verify Compilation

```bash
# Build Rust library
cargo build --release

# Check for warnings
cargo clippy -- -D warnings
```

**Expected Result**: ✅ No compilation errors or warnings

---

## Step 2: Update CUDA Build

### 2.1 Include FFI Check Header

Edit `src/cuda/kernels/ontology_reasoning.cu`:

**Add at line 785 (after struct definitions, before extern "C"):**

```cpp
// ============================================================================
// FFI SAFETY VERIFICATION
// ============================================================================

// Verify struct layout matches Rust FFI definitions
// This will cause compilation to FAIL if there's any mismatch
#include "ontology_ffi_check.cuh"

// ============================================================================
// HOST FUNCTIONS FOR KERNEL LAUNCH
// ============================================================================

extern "C" {
    // ... existing host functions ...
}
```

### 2.2 Update CUDA Compilation

**Option A: Using nvcc directly**

```bash
cd /home/devuser/workspace/hackathon-tv5/src/cuda

# Compile with FFI checks
nvcc -c kernels/ontology_reasoning.cu \
     -o build/ontology_reasoning.o \
     -I./kernels \
     --std=c++17 \
     -arch=sm_80 \
     --Werror all-warnings

# Expected output:
# (no errors - static assertions pass)
```

**Option B: Using CMakeLists.txt**

Add to `src/cuda/CMakeLists.txt`:

```cmake
# Add FFI check header to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# Compile ontology reasoning kernel with FFI checks
cuda_add_library(ontology_reasoning
    kernels/ontology_reasoning.cu
    OPTIONS -std=c++17 -Werror all-warnings
)
```

Then build:

```bash
cd /home/devuser/workspace/hackathon-tv5
mkdir -p build && cd build
cmake ../src/cuda
make
```

### 2.3 Verify CUDA Compilation

**If compilation succeeds**: ✅ FFI structs are binary-compatible
**If compilation fails**: ❌ Struct layout mismatch detected

Example failure message:
```
error: static assertion failed: "MediaOntologyNode size mismatch! Expected 80 bytes"
```

---

## Step 3: Integration Testing

### 3.1 Create Integration Test

Create `tests/gpu_integration_test.rs`:

```rust
use hackathon_tv5::models::{MediaOntologyNode, MediaOntologyConstraint};

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_transfer() {
    // Create test data
    let mut nodes = vec![MediaOntologyNode::default(); 100];
    for (i, node) in nodes.iter_mut().enumerate() {
        node.node_id = i as u32;
    }

    // Simulate CUDA transfer (replace with actual CUDA calls)
    let gpu_ptr = cuda_malloc(nodes.len() * std::mem::size_of::<MediaOntologyNode>());
    cuda_memcpy_htod(gpu_ptr, nodes.as_ptr(), nodes.len());

    // Transfer back
    let mut result = vec![MediaOntologyNode::default(); 100];
    cuda_memcpy_dtoh(result.as_mut_ptr(), gpu_ptr, nodes.len());

    // Verify
    for (orig, res) in nodes.iter().zip(result.iter()) {
        assert_eq!(orig.node_id, res.node_id);
    }

    cuda_free(gpu_ptr);
}
```

### 3.2 Run Integration Tests

```bash
# With CUDA feature enabled
cargo test --features cuda gpu_integration -- --nocapture

# Expected: All tests pass
```

---

## Step 4: Validation Checklist

### Compilation Checks

- [ ] Rust builds without warnings: `cargo build --release`
- [ ] CUDA compiles with FFI checks: `nvcc -c ontology_reasoning.cu`
- [ ] Static assertions pass in both Rust and CUDA
- [ ] No alignment warnings from compiler

### Runtime Checks

- [ ] Rust FFI tests pass: `cargo test ontology_ffi`
- [ ] 10K node serialization test passes
- [ ] Extreme value tests pass
- [ ] GPU transfer simulation passes

### Platform Verification

- [ ] Tested on Linux x86_64
- [ ] Tested on target GPU (A100 / RTX 4090)
- [ ] Endianness verified (little-endian)
- [ ] 64-byte alignment confirmed

---

## Step 5: Performance Validation

### 5.1 Benchmark Transfer Speed

```rust
use std::time::Instant;

#[test]
fn benchmark_transfer() {
    const COUNT: usize = 10_000;
    let nodes = vec![MediaOntologyNode::default(); COUNT];

    let start = Instant::now();

    // Host → Device
    let gpu_ptr = cuda_malloc(COUNT * 80);
    cuda_memcpy_htod(gpu_ptr, nodes.as_ptr(), COUNT);

    // Device → Host
    let mut result = vec![MediaOntologyNode::default(); COUNT];
    cuda_memcpy_dtoh(result.as_mut_ptr(), gpu_ptr, COUNT);

    let elapsed = start.elapsed();

    println!("Transfer time for 10K nodes: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    assert!(elapsed.as_millis() < 5, "Transfer too slow!");

    cuda_free(gpu_ptr);
}
```

**Expected Performance (NVIDIA A100)**:
- 10K nodes (800 KB)
- Host→Device: ~0.3 ms
- Device→Host: ~0.3 ms
- Total: <1 ms

### 5.2 Verify Memory Coalescing

Use NVIDIA Nsight Compute:

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu \
    --kernel-name apply_disjoint_genres_kernel \
    ./your_binary

# Expected: 0 bank conflicts (perfectly aligned)
```

---

## Step 6: Troubleshooting

### Issue: Compilation Fails with Size Mismatch

**Error**:
```
static assertion failed: "MediaOntologyNode size mismatch!"
```

**Solution**:
1. Check struct field order matches exactly
2. Verify padding arrays are correct size
3. Ensure `repr(C, align(64))` is present in Rust
4. Check CUDA struct doesn't have unexpected alignment

### Issue: Tests Pass but GPU Transfer Corrupts Data

**Symptoms**:
- Rust tests pass
- CUDA compiles successfully
- Data corruption after GPU transfer

**Diagnosis**:
```rust
// Add debug prints
println!("Before transfer: {:?}", nodes[0]);
cuda_memcpy_htod(gpu_ptr, nodes.as_ptr(), count);
cuda_memcpy_dtoh(result.as_mut_ptr(), gpu_ptr, count);
println!("After transfer: {:?}", result[0]);
```

**Common Causes**:
1. Pointer arithmetic error (wrong offset calculation)
2. Asynchronous transfer without synchronization
3. GPU memory not initialized properly

**Solution**:
```rust
// Add explicit synchronization
cuda_device_synchronize();

// Verify transfer size
assert_eq!(count * size_of::<MediaOntologyNode>(), transfer_bytes);
```

### Issue: Performance Lower Than Expected

**Symptoms**:
- Transfer takes >5ms for 10K nodes
- GPU utilization low

**Diagnosis**:
```bash
# Profile with Nsight Systems
nsys profile --stats=true ./your_binary

# Check for:
# - Memory copy events
# - Kernel execution time
# - PCIe bandwidth utilization
```

**Optimization**:
1. Use pinned memory: `cudaMallocHost` instead of `malloc`
2. Enable asynchronous transfers with streams
3. Batch multiple transfers together

---

## Step 7: Production Deployment

### Pre-Deployment Checklist

- [ ] All tests pass on production hardware
- [ ] Performance benchmarks meet requirements (<3ms for 10K nodes)
- [ ] Memory leaks checked with `valgrind` or CUDA-MEMCHECK
- [ ] Static assertions integrated into CI/CD pipeline
- [ ] Documentation updated

### CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
name: FFI Safety Checks

on: [push, pull_request]

jobs:
  test-ffi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run FFI tests
        run: |
          cd src/rust
          cargo test ontology_ffi -- --nocapture

      - name: Check CUDA compilation
        run: |
          cd src/cuda
          nvcc -c kernels/ontology_reasoning.cu -I./kernels
```

---

## Step 8: Maintenance

### When to Update FFI Structs

**Add a field to CUDA struct**:
1. Update `ontology_reasoning.cu` struct definition
2. Update `ontology_ffi.rs` Rust definition
3. Update `ontology_ffi_check.cuh` assertions
4. Run all tests to verify compatibility

**Change alignment**:
1. Update both `repr(C, align(N))` and CUDA struct
2. Re-run memory coalescing benchmarks
3. Update documentation

### Versioning

Track FFI compatibility versions:

```rust
pub const FFI_VERSION: u32 = 1;

pub struct MediaOntologyNode {
    pub ffi_version: u32,  // Always 1 for this version
    // ... other fields
}
```

---

## Appendix A: Quick Reference

### Struct Sizes
- `MediaOntologyNode`: 80 bytes (64-byte aligned)
- `MediaOntologyConstraint`: 64 bytes (64-byte aligned)
- `Float3`: 12 bytes (4-byte aligned)

### Key Files
- Rust FFI: `src/rust/models/ontology_ffi.rs`
- Rust Tests: `src/rust/models/ontology_ffi_tests.rs`
- CUDA Check: `src/cuda/kernels/ontology_ffi_check.cuh`
- CUDA Kernel: `src/cuda/kernels/ontology_reasoning.cu`

### Useful Commands
```bash
# Rust: Run FFI tests
cargo test ontology_ffi

# CUDA: Compile with checks
nvcc -c kernels/ontology_reasoning.cu -I./kernels

# Check struct layout
cargo rustc -- --print=type-sizes

# Profile GPU transfers
nsys profile ./binary
```

---

## Appendix B: Common Patterns

### Pattern 1: Batch Transfer

```rust
fn transfer_batch(nodes: &[MediaOntologyNode]) -> Result<(), CudaError> {
    let size_bytes = nodes.len() * std::mem::size_of::<MediaOntologyNode>();

    let mut gpu_ptr = cuda_malloc(size_bytes)?;
    cuda_memcpy_htod(gpu_ptr, nodes.as_ptr(), nodes.len())?;

    // Process on GPU...

    let mut result = vec![MediaOntologyNode::default(); nodes.len()];
    cuda_memcpy_dtoh(result.as_mut_ptr(), gpu_ptr, nodes.len())?;

    cuda_free(gpu_ptr)?;
    Ok(())
}
```

### Pattern 2: Zero-Copy Pinned Memory

```rust
fn transfer_with_pinned_memory(nodes: &[MediaOntologyNode]) {
    // Allocate pinned host memory
    let pinned_ptr = cuda_malloc_host(nodes.len() * 80)?;

    // Copy to pinned memory
    unsafe {
        std::ptr::copy_nonoverlapping(
            nodes.as_ptr(),
            pinned_ptr,
            nodes.len()
        );
    }

    // Transfer to GPU (faster due to pinned memory)
    let gpu_ptr = cuda_malloc(nodes.len() * 80)?;
    cuda_memcpy_htod_async(gpu_ptr, pinned_ptr, nodes.len(), stream)?;

    // ... GPU processing ...

    cuda_free_host(pinned_ptr)?;
}
```

---

## Support

For issues with FFI integration:
1. Check `/docs/FFI_ALIGNMENT_REPORT.md` for detailed analysis
2. Run test suite: `cargo test ontology_ffi`
3. Verify CUDA compilation: `nvcc -c ontology_reasoning.cu -I./kernels`
4. Review static assertion errors carefully

**Report Issues**: Include output from:
- `cargo test ontology_ffi`
- `nvcc --version`
- `rustc --version`
- Target GPU architecture
