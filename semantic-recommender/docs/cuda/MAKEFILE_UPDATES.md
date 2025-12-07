# CUDA Makefile Updates - Kernel Reorganization

## Summary
Updated CUDA Makefiles to reference new kernel locations after reorganization into `variants/` subdirectory.

## Files Updated

### 1. `/src/cuda/kernels/Makefile` (T4-Optimized)
- Updated PTX_KERNELS to reference variant kernels in `variants/` subdirectory
- Added separate handling for variant kernel object compilation
- Updated benchmark algorithm paths to use `variants/`

### 2. `/src/cuda/kernels/Makefile.a100` (A100-Optimized)
- Updated PTX_KERNELS to reference variant kernels in `variants/` subdirectory
- Added separate handling for variant kernel object compilation
- Updated benchmark algorithm paths to use `variants/`

## Kernel Reorganization Details

### Moved to `variants/` Subdirectory:
- `semantic_similarity_fp16.cu`
- `semantic_similarity_fp16_tensor_cores.cu`
- `semantic_similarity_tf32.cu` (A100-only)
- `product_quantization.cu`
- `memory_layout.cu`
- `sorted_similarity.cu`
- `lsh_gpu.cu`
- `hybrid_index.cu`
- `benchmark_algorithms.cu`
- `hnsw_gpu.cuh`
- `unified_pipeline.cu`

### Remaining in Root `kernels/` Directory:
- `graph_search.cu`
- `ontology_reasoning.cu`
- `hybrid_sssp.cu`
- `semantic_similarity.cu` (core kernel)

## Changes Made

### Makefile Pattern Matching
Added dual-path object file compilation handling:
- `MAIN_OBJECTS`: Non-variant kernels (root directory)
- `VARIANT_OBJECTS`: Variant kernels (variants/ subdirectory)
- Build rules for both paths: `$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu` and `$(BUILD_DIR)/%.o: $(KERNEL_DIR)/variants/%.cu`

### PTX Compilation
Updated PTX kernel lists to include variant paths:
- **T4 (sm_75)**: 7 kernels including 3 from variants/
- **A100 (sm_80)**: 8 kernels including 4 from variants/

### Benchmark Targets
Updated benchmark dependencies:
- `benchmark_algorithms.cu` → `variants/benchmark_algorithms.cu`
- `hnsw_gpu.cuh` → `variants/hnsw_gpu.cuh`
- `lsh_gpu.cu` → `variants/lsh_gpu.cu`

## Build Verification

### T4 Makefile (sm_75)
```bash
cd src/cuda/kernels
make -n ptx  # Dry-run PTX compilation
```

PTX Kernels (7 total):
- `./variants/semantic_similarity_fp16_tensor_cores.cu`
- `./graph_search.cu`
- `./ontology_reasoning.cu`
- `./hybrid_sssp.cu`
- `./semantic_similarity.cu`
- `./variants/semantic_similarity_fp16.cu`
- `./variants/product_quantization.cu`

### A100 Makefile (sm_80)
```bash
cd src/cuda/kernels
make -n -f Makefile.a100 ptx  # Dry-run PTX compilation
```

PTX Kernels (8 total):
- `./semantic_similarity.cu`
- `./variants/semantic_similarity_fp16.cu`
- `./variants/semantic_similarity_tf32.cu`
- `./variants/semantic_similarity_fp16_tensor_cores.cu`
- `./graph_search.cu`
- `./ontology_reasoning.cu`
- `./hybrid_sssp.cu`
- `./variants/product_quantization.cu`

## Compilation Paths Verified
- All variant paths correctly resolve to `variants/` subdirectory
- Object file rules handle both root and variant kernels
- PTX compilation includes correct architecture flags (sm_75 for T4, sm_80 for A100)
- Benchmark targets properly reference variant kernel dependencies

## Build Targets Preserved

### T4 Targets
- `all` - Build T4-optimized kernel library (default)
- `ptx` - Generate PTX assembly for Rust FFI
- `test-t4` - Run T4 validation tests
- `benchmark` - Run T4 benchmarks
- `benchmark-algorithms` - Build HNSW/LSH benchmark suite

### A100 Targets
- `all` - Build A100-optimized kernel library (default)
- `ptx` - Generate PTX assembly for Rust FFI
- `benchmark-a100` - Build A100 benchmark binary
- `run-benchmark` - Run A100 performance benchmarks
- `profile` - Full Nsight Compute analysis

## Next Steps
1. Verify compilation by running: `make clean && make ptx`
2. Test complete builds: `make all`
3. Validate Rust FFI integration with updated PTX files
4. Run performance benchmarks to confirm kernel variants are accessible

## Date Updated
2025-12-07

## Status
✓ All kernel paths updated
✓ Object file compilation rules added for variant kernels
✓ PTX compilation verified with dry-run
✓ Build targets preserved
✓ Documentation updated
