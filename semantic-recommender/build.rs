// Build script for CUDA kernel compilation and linking
// Compiles PTX kernels during `cargo build` and links CUDA libraries

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=src/cuda/kernels");
    println!("cargo:rerun-if-changed=src/cuda/kernels/Makefile");
    println!("cargo:rerun-if-changed=src/cuda/kernels/unified_pipeline.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/semantic_similarity_fp16_tensor_cores.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/graph_search.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/ontology_reasoning.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/hybrid_sssp.cu");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernel_dir = PathBuf::from(&manifest_dir).join("src/cuda/kernels");
    let target_dir = PathBuf::from(&manifest_dir).join("target");
    let lib_dir = target_dir.join("release");

    println!("cargo:warning=Building CUDA kernels to PTX...");

    // Check if nvcc is available
    let nvcc_available = Command::new("nvcc").arg("--version").output().is_ok();

    if nvcc_available {
        // Create PTX output directory
        let ptx_dir = target_dir.join("ptx");
        std::fs::create_dir_all(&ptx_dir).expect("Failed to create PTX directory");

        // Compile PTX kernels using Makefile
        let status = Command::new("make")
            .arg("ptx")
            .current_dir(&kernel_dir)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=✓ CUDA kernels compiled to PTX successfully");
                println!("cargo:rustc-env=PTX_DIR={}", ptx_dir.display());
                println!("cargo:rustc-link-search=native={}", ptx_dir.display());
            }
            Ok(s) => {
                println!("cargo:warning=⚠ PTX compilation failed with status: {}", s);
            }
            Err(e) => {
                println!("cargo:warning=⚠ Failed to run make: {}", e);
            }
        }
    } else {
        println!("cargo:warning=⚠ nvcc not found - skipping PTX compilation");
        println!("cargo:warning=  Install CUDA toolkit or set CUDA_PATH environment variable");
    }

    // Link directories
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link CUDA libraries if available
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(cuda_path).join("lib64");
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }

    // Link CUDA runtime and libraries
    if nvcc_available {
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cublas");
    }

    // Link unified pipeline library if it exists
    let unified_lib = lib_dir.join("libunified_gpu.so");
    if unified_lib.exists() {
        println!("cargo:rustc-link-lib=dylib=unified_gpu");
    }

    // Enable FP16 features if available
    if cfg!(feature = "fp16") {
        println!("cargo:rustc-cfg=feature=\"fp16\"");
    }
}
