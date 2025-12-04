# Rust + CUDA PTX Integration Guide

## Quick Start

All PTX kernels are available in `target/ptx/` after running the build script.

```bash
# Build PTX kernels
./scripts/build-cuda-kernels.sh

# Or let cargo build them automatically
cargo build
```

## Critical Kernels Available

| Kernel | Size | Purpose | Entry Point |
|--------|------|---------|-------------|
| `semantic_similarity_fp16_tensor_cores.ptx` | 38KB | FP16 tensor core semantic search | `compute_semantic_similarity_fp16_tensor_cores` |
| `graph_search.ptx` | 42KB | GPU graph traversal | `semantic_graph_search` |
| `ontology_reasoning.ptx` | 39KB | Knowledge graph reasoning | `process_ontology_rules` |
| `hybrid_sssp.ptx` | 22KB | Shortest path algorithm | `hybrid_sssp_kernel` |
| `semantic_similarity.ptx` | 94KB | FP32 semantic similarity | `compute_similarity` |
| `semantic_similarity_fp16.ptx` | 53KB | FP16 semantic (no tensor cores) | `compute_similarity_fp16` |
| `product_quantization.ptx` | 41KB | Vector compression | `pq_encode_kernel` |

## Rust FFI Setup

### Dependencies

Add to `Cargo.toml`:

```toml
[dependencies]
cuda-driver-sys = "0.3"
cuda-runtime-sys = "0.3"

[build-dependencies]
# None needed - build.rs handles PTX compilation
```

### Environment Setup

```rust
// The PTX_DIR environment variable is set by build.rs
let ptx_dir = env!("PTX_DIR");
```

## Basic Usage Pattern

### 1. Initialize CUDA

```rust
use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr;

unsafe fn init_cuda() -> Result<CUcontext, CUresult> {
    // Initialize CUDA driver API
    cuInit(0)?;

    // Get device
    let mut device: CUdevice = 0;
    cuDeviceGet(&mut device, 0)?;

    // Create context
    let mut context: CUcontext = ptr::null_mut();
    cuCtxCreate_v2(&mut context, 0, device)?;

    Ok(context)
}
```

### 2. Load PTX Module

```rust
const PTX: &str = include_str!("../target/ptx/semantic_similarity_fp16_tensor_cores.ptx");

unsafe fn load_module() -> Result<(CUmodule, CUfunction), CUresult> {
    let mut module: CUmodule = ptr::null_mut();

    // Load PTX
    let ptx_cstr = CString::new(PTX).unwrap();
    cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _)?;

    // Get kernel function
    let mut kernel: CUfunction = ptr::null_mut();
    let name = CString::new("compute_semantic_similarity_fp16_tensor_cores").unwrap();
    cuModuleGetFunction(&mut kernel, module, name.as_ptr())?;

    Ok((module, kernel))
}
```

### 3. Allocate GPU Memory

```rust
unsafe fn allocate_gpu_memory(size: usize) -> Result<CUdeviceptr, CUresult> {
    let mut d_ptr: CUdeviceptr = 0;
    cuMemAlloc_v2(&mut d_ptr, size)?;
    Ok(d_ptr)
}

unsafe fn copy_to_gpu(d_ptr: CUdeviceptr, h_data: &[f32]) -> Result<(), CUresult> {
    cuMemcpyHtoD_v2(
        d_ptr,
        h_data.as_ptr() as *const _,
        h_data.len() * std::mem::size_of::<f32>(),
    )?;
    Ok(())
}
```

### 4. Launch Kernel

```rust
unsafe fn launch_kernel(
    kernel: CUfunction,
    d_query: CUdeviceptr,
    d_embeddings: CUdeviceptr,
    d_results: CUdeviceptr,
    num_embeddings: u32,
    embedding_dim: u32,
) -> Result<(), CUresult> {
    // Kernel parameters
    let mut params = [
        &d_query as *const _ as *mut _,
        &d_embeddings as *const _ as *mut _,
        &d_results as *const _ as *mut _,
        &num_embeddings as *const _ as *mut _,
        &embedding_dim as *const _ as *mut _,
    ];

    // Grid/block dimensions
    let grid_dim = (num_embeddings + 255) / 256;
    let block_dim = 256;

    // Launch
    cuLaunchKernel(
        kernel,
        grid_dim, 1, 1,        // gridDim
        block_dim, 1, 1,       // blockDim
        0,                     // sharedMem
        ptr::null_mut(),       // stream
        params.as_mut_ptr(),   // kernel params
        ptr::null_mut(),       // extra
    )?;

    // Wait for completion
    cuCtxSynchronize()?;

    Ok(())
}
```

### 5. Copy Results Back

```rust
unsafe fn copy_from_gpu(h_data: &mut [f32], d_ptr: CUdeviceptr) -> Result<(), CUresult> {
    cuMemcpyDtoH_v2(
        h_data.as_mut_ptr() as *mut _,
        d_ptr,
        h_data.len() * std::mem::size_of::<f32>(),
    )?;
    Ok(())
}
```

### 6. Cleanup

```rust
unsafe fn cleanup(
    context: CUcontext,
    module: CUmodule,
    d_ptrs: &[CUdeviceptr],
) -> Result<(), CUresult> {
    // Free GPU memory
    for &ptr in d_ptrs {
        cuMemFree_v2(ptr)?;
    }

    // Unload module
    cuModuleUnload(module)?;

    // Destroy context
    cuCtxDestroy_v2(context)?;

    Ok(())
}
```

## Complete Example

```rust
use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr;

const PTX: &str = include_str!("../target/ptx/semantic_similarity_fp16_tensor_cores.ptx");

pub struct CudaSemanticSearch {
    context: CUcontext,
    module: CUmodule,
    kernel: CUfunction,
}

impl CudaSemanticSearch {
    pub unsafe fn new() -> Result<Self, CUresult> {
        // Initialize CUDA
        cuInit(0)?;

        let mut device: CUdevice = 0;
        cuDeviceGet(&mut device, 0)?;

        let mut context: CUcontext = ptr::null_mut();
        cuCtxCreate_v2(&mut context, 0, device)?;

        // Load module
        let mut module: CUmodule = ptr::null_mut();
        let ptx_cstr = CString::new(PTX).unwrap();
        cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _)?;

        // Get kernel
        let mut kernel: CUfunction = ptr::null_mut();
        let name = CString::new("compute_semantic_similarity_fp16_tensor_cores").unwrap();
        cuModuleGetFunction(&mut kernel, module, name.as_ptr())?;

        Ok(Self {
            context,
            module,
            kernel,
        })
    }

    pub unsafe fn search(
        &self,
        query: &[f32],
        embeddings: &[f32],
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> Result<Vec<f32>, CUresult> {
        // Allocate GPU memory
        let query_size = embedding_dim * std::mem::size_of::<f32>();
        let embeddings_size = num_embeddings * embedding_dim * std::mem::size_of::<f32>();
        let results_size = num_embeddings * std::mem::size_of::<f32>();

        let mut d_query: CUdeviceptr = 0;
        let mut d_embeddings: CUdeviceptr = 0;
        let mut d_results: CUdeviceptr = 0;

        cuMemAlloc_v2(&mut d_query, query_size)?;
        cuMemAlloc_v2(&mut d_embeddings, embeddings_size)?;
        cuMemAlloc_v2(&mut d_results, results_size)?;

        // Copy data to GPU
        cuMemcpyHtoD_v2(d_query, query.as_ptr() as *const _, query_size)?;
        cuMemcpyHtoD_v2(d_embeddings, embeddings.as_ptr() as *const _, embeddings_size)?;

        // Launch kernel
        let mut params = [
            &d_query as *const _ as *mut _,
            &d_embeddings as *const _ as *mut _,
            &d_results as *const _ as *mut _,
            &(num_embeddings as u32) as *const _ as *mut _,
            &(embedding_dim as u32) as *const _ as *mut _,
        ];

        let grid_dim = ((num_embeddings + 255) / 256) as u32;
        cuLaunchKernel(
            self.kernel,
            grid_dim, 1, 1,
            256, 1, 1,
            0,
            ptr::null_mut(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        )?;

        cuCtxSynchronize()?;

        // Copy results back
        let mut results = vec![0.0f32; num_embeddings];
        cuMemcpyDtoH_v2(
            results.as_mut_ptr() as *mut _,
            d_results,
            results_size,
        )?;

        // Cleanup
        cuMemFree_v2(d_query)?;
        cuMemFree_v2(d_embeddings)?;
        cuMemFree_v2(d_results)?;

        Ok(results)
    }
}

impl Drop for CudaSemanticSearch {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module).ok();
            cuCtxDestroy_v2(self.context).ok();
        }
    }
}

// Usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        let searcher = CudaSemanticSearch::new()?;

        let query = vec![0.1; 1024];
        let embeddings = vec![0.5; 1000 * 1024];

        let results = searcher.search(&query, &embeddings, 1000, 1024)?;

        println!("Top result: {}", results[0]);
    }

    Ok(())
}
```

## Kernel-Specific Integration

### Graph Search

```rust
const GRAPH_PTX: &str = include_str!("../target/ptx/graph_search.ptx");

// Entry point: semantic_graph_search
// Parameters: (nodes, edges, query, results, num_nodes, num_edges)
```

### Ontology Reasoning

```rust
const ONTOLOGY_PTX: &str = include_str!("../target/ptx/ontology_reasoning.ptx");

// Entry point: process_ontology_rules
// Parameters: (entities, relationships, rules, results, num_entities)
```

### Hybrid SSSP

```rust
const SSSP_PTX: &str = include_str!("../target/ptx/hybrid_sssp.ptx");

// Entry point: hybrid_sssp_kernel
// Parameters: (graph, distances, predecessors, source, num_vertices)
```

## Error Handling

```rust
fn cuda_result_to_error(result: CUresult) -> Option<String> {
    match result {
        CUresult::CUDA_SUCCESS => None,
        CUresult::CUDA_ERROR_INVALID_VALUE => Some("Invalid value".to_string()),
        CUresult::CUDA_ERROR_OUT_OF_MEMORY => Some("Out of memory".to_string()),
        CUresult::CUDA_ERROR_NOT_INITIALIZED => Some("CUDA not initialized".to_string()),
        _ => Some(format!("CUDA error: {:?}", result)),
    }
}

// Use with ? operator
impl From<CUresult> for Box<dyn std::error::Error> {
    fn from(result: CUresult) -> Self {
        cuda_result_to_error(result)
            .unwrap_or_else(|| "Unknown CUDA error".to_string())
            .into()
    }
}
```

## Performance Tips

### 1. Stream Management

```rust
unsafe fn create_stream() -> Result<CUstream, CUresult> {
    let mut stream: CUstream = ptr::null_mut();
    cuStreamCreate(&mut stream, CU_STREAM_DEFAULT)?;
    Ok(stream)
}

// Launch kernel asynchronously
cuLaunchKernel(
    kernel,
    grid, 1, 1,
    block, 1, 1,
    0,
    stream,  // Use stream instead of null_mut()
    params.as_mut_ptr(),
    ptr::null_mut(),
)?;
```

### 2. Pinned Memory

```rust
unsafe fn allocate_pinned_memory(size: usize) -> Result<*mut f32, CUresult> {
    let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
    cuMemAllocHost_v2(&mut ptr, size)?;
    Ok(ptr as *mut f32)
}

// Faster H2D/D2H transfers
```

### 3. Unified Memory

```rust
unsafe fn allocate_unified_memory(size: usize) -> Result<CUdeviceptr, CUresult> {
    let mut ptr: CUdeviceptr = 0;
    cuMemAllocManaged(&mut ptr, size, CU_MEM_ATTACH_GLOBAL)?;
    Ok(ptr)
}

// Accessible from both CPU and GPU
```

### 4. Batch Processing

```rust
// Process multiple queries in parallel
for chunk in queries.chunks(batch_size) {
    // Launch kernel for chunk
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_loading() {
        unsafe {
            let searcher = CudaSemanticSearch::new();
            assert!(searcher.is_ok());
        }
    }

    #[test]
    fn test_semantic_search() {
        unsafe {
            let searcher = CudaSemanticSearch::new().unwrap();
            let query = vec![1.0; 128];
            let embeddings = vec![0.5; 100 * 128];
            let results = searcher.search(&query, &embeddings, 100, 128);
            assert!(results.is_ok());
        }
    }
}
```

## Benchmarking

```rust
use std::time::Instant;

fn benchmark_kernel(iterations: usize) {
    unsafe {
        let searcher = CudaSemanticSearch::new().unwrap();
        let query = vec![1.0; 1024];
        let embeddings = vec![0.5; 10000 * 1024];

        let start = Instant::now();
        for _ in 0..iterations {
            searcher.search(&query, &embeddings, 10000, 1024).unwrap();
        }
        let duration = start.elapsed();

        println!("Average: {:?}", duration / iterations as u32);
    }
}
```

## Common Issues

### Issue: PTX not found

**Solution**: Run `./scripts/build-cuda-kernels.sh` first

### Issue: CUDA_ERROR_INVALID_PTX

**Solution**: Ensure sm_75 compatibility (T4 GPU)

### Issue: Out of memory

**Solution**: Reduce batch size or use streaming

### Issue: Kernel launch timeout

**Solution**: Break work into smaller chunks

## References

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [rust-cuda Project](https://github.com/Rust-GPU/Rust-CUDA)
