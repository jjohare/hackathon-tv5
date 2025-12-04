// Build script for linking CUDA unified pipeline library

use std::env;
use std::path::PathBuf;

fn main() {
    // Get the project directory
    let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = PathBuf::from(&project_dir)
        .join("target")
        .join("release");

    // Tell cargo to look for libraries in target/release
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link the CUDA unified pipeline library
    println!("cargo:rustc-link-lib=dylib=unified_gpu");

    // Link CUDA runtime
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Link cuBLAS for tensor core operations
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Rerun if library changes
    println!("cargo:rerun-if-changed=src/cuda/kernels/unified_pipeline.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/semantic_similarity_fp16_tensor_cores.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/sorted_similarity.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/hnsw_gpu.cuh");
    println!("cargo:rerun-if-changed=src/cuda/kernels/lsh_gpu.cu");

    // Check if CUDA is available
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(cuda_path).join("lib64");
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    } else if cfg!(target_os = "linux") {
        // Common CUDA installation paths on Linux
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }

    // Enable FP16 features if available
    if cfg!(feature = "fp16") {
        println!("cargo:rustc-cfg=feature=\"fp16\"");
    }
}
