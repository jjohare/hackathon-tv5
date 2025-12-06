# CUDA Build Quick Reference

## Build PTX Kernels

```bash
# Automated build
./scripts/build-cuda-kernels.sh

# Via Makefile
cd src/cuda/kernels && make ptx

# Via Cargo (automatic)
cargo build
```

## PTX Output Location

```
target/ptx/
├── semantic_similarity_fp16_tensor_cores.ptx  (38KB)
├── graph_search.ptx                           (42KB)
├── ontology_reasoning.ptx                     (39KB)
├── hybrid_sssp.ptx                            (22KB)
├── semantic_similarity.ptx                    (94KB)
├── semantic_similarity_fp16.ptx               (53KB)
├── product_quantization.ptx                   (41KB)
└── manifest.txt
```

## Rust Integration (Minimal)

```rust
use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr;

const PTX: &str = include_str!("../target/ptx/semantic_similarity_fp16_tensor_cores.ptx");

unsafe fn load_kernel() -> Result<(CUmodule, CUfunction), CUresult> {
    // Init CUDA
    cuInit(0)?;
    let mut device = 0;
    cuDeviceGet(&mut device, 0)?;
    let mut ctx = ptr::null_mut();
    cuCtxCreate_v2(&mut ctx, 0, device)?;

    // Load PTX
    let mut module = ptr::null_mut();
    let ptx = CString::new(PTX).unwrap();
    cuModuleLoadData(&mut module, ptx.as_ptr() as *const _)?;

    // Get kernel
    let mut kernel = ptr::null_mut();
    let name = CString::new("compute_semantic_similarity_fp16_tensor_cores").unwrap();
    cuModuleGetFunction(&mut kernel, module, name.as_ptr())?;

    Ok((module, kernel))
}
```

## Critical Kernel Entry Points

| PTX File | Entry Point Function |
|----------|---------------------|
| `semantic_similarity_fp16_tensor_cores.ptx` | `compute_semantic_similarity_fp16_tensor_cores` |
| `graph_search.ptx` | `semantic_graph_search` |
| `ontology_reasoning.ptx` | `process_ontology_rules` |
| `hybrid_sssp.ptx` | `hybrid_sssp_kernel` |

## Makefile Targets

```bash
make ptx              # Build PTX (primary)
make all              # Build everything
make clean            # Clean artifacts
make check-registers  # Register analysis
make sass             # SASS assembly
make profile          # Nsight profile
```

## Verification

```bash
# Check PTX exists
ls -lh target/ptx/*.ptx

# Verify PTX format
grep ".version" target/ptx/semantic_similarity_fp16_tensor_cores.ptx

# Check tensor cores
grep "wmma" target/ptx/semantic_similarity_fp16_tensor_cores.ptx
```

## Troubleshooting

**PTX not found**: Run `./scripts/build-cuda-kernels.sh`
**CUDA error**: Check `nvcc --version` and GPU availability
**Link error**: Ensure `LD_LIBRARY_PATH` includes CUDA libs

## Documentation

- **Build System**: `docs/cuda-build-system.md`
- **Rust Integration**: `docs/rust-cuda-integration.md`
- **Summary**: `docs/BUILD_SUMMARY.md`
