/// CUDA Kernel FFI Bindings
///
/// Provides safe Rust wrappers around compiled CUDA kernels with
/// proper error handling and type safety.

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::path::Path;
use std::sync::Arc;

use super::*;

// Type alias for FP16 - using u16 as raw representation
// The CUDA kernels expect __half which is internally u16
pub type GpuHalf = u16;

/// Error types for kernel operations
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("Failed to load PTX module: {0}")]
    ModuleLoad(String),

    #[error("Kernel function not found: {0}")]
    FunctionNotFound(String),

    #[error("Invalid kernel parameters: {0}")]
    InvalidParameters(String),

    #[error("Kernel launch failed: {0}")]
    LaunchFailed(String),
}

/// Collection of loaded CUDA kernel modules
pub struct KernelModules {
    device: Arc<CudaDevice>,

    // Tensor Core optimized kernels
    batch_cosine_similarity: Option<CudaFunction>,
    batch_dot_product: Option<CudaFunction>,
    precompute_norms: Option<CudaFunction>,
    compute_multimodal_similarity: Option<CudaFunction>,
}

impl KernelModules {
    /// Load all kernel modules from PTX directory
    ///
    /// PTX files should be located at:
    /// - `<ptx_path>/semantic_similarity_fp16_tensor_cores.ptx`
    ///
    /// Expected kernels:
    /// - `batch_cosine_similarity_tensor_cores`
    /// - `batch_dot_product_tensor_cores`
    /// - `precompute_norms_kernel`
    /// - `compute_multimodal_similarity_tensor_cores`
    pub fn load(device: Arc<CudaDevice>, ptx_path: &str) -> GpuResult<Self> {
        let ptx_dir = Path::new(ptx_path);

        if !ptx_dir.exists() {
            return Err(KernelError::ModuleLoad(
                format!("PTX directory not found: {}", ptx_path)
            ).into());
        }

        // Load main tensor core kernel module
        let tensor_core_ptx_path = ptx_dir.join("semantic_similarity_fp16_tensor_cores.ptx");

        let mut batch_cosine_similarity = None;
        let mut batch_dot_product = None;
        let mut precompute_norms = None;
        let mut compute_multimodal_similarity = None;

        // Attempt to load PTX module
        if tensor_core_ptx_path.exists() {
            match std::fs::read_to_string(&tensor_core_ptx_path) {
                Ok(ptx_src) => {
                    // Load PTX module into device
                    match device.load_ptx(
                        ptx_src.into(),
                        "semantic_similarity_fp16_tensor_cores",
                        &[
                            "batch_cosine_similarity_tensor_cores",
                            "batch_dot_product_tensor_cores",
                            "precompute_norms_kernel",
                            "compute_multimodal_similarity_tensor_cores",
                        ]
                    ) {
                        Ok(module) => {
                            // Extract function references
                            batch_cosine_similarity = Some(
                                module.get_func("batch_cosine_similarity_tensor_cores")
                                    .map_err(|e| KernelError::FunctionNotFound(
                                        format!("batch_cosine_similarity_tensor_cores: {}", e)
                                    ))?
                            );

                            batch_dot_product = Some(
                                module.get_func("batch_dot_product_tensor_cores")
                                    .map_err(|e| KernelError::FunctionNotFound(
                                        format!("batch_dot_product_tensor_cores: {}", e)
                                    ))?
                            );

                            precompute_norms = Some(
                                module.get_func("precompute_norms_kernel")
                                    .map_err(|e| KernelError::FunctionNotFound(
                                        format!("precompute_norms_kernel: {}", e)
                                    ))?
                            );

                            compute_multimodal_similarity = Some(
                                module.get_func("compute_multimodal_similarity_tensor_cores")
                                    .map_err(|e| KernelError::FunctionNotFound(
                                        format!("compute_multimodal_similarity_tensor_cores: {}", e)
                                    ))?
                            );
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load PTX module: {}", e);
                        }
                    }
                }
                Err(e) => {
                    return Err(KernelError::ModuleLoad(
                        format!("Failed to read PTX file {}: {}",
                            tensor_core_ptx_path.display(), e)
                    ).into());
                }
            }
        } else {
            eprintln!(
                "Warning: PTX file not found at {}. Kernels will not be available.",
                tensor_core_ptx_path.display()
            );
        }

        Ok(Self {
            device,
            batch_cosine_similarity,
            batch_dot_product,
            precompute_norms,
            compute_multimodal_similarity,
        })
    }

    /// Launch batch cosine similarity kernel with tensor cores
    ///
    /// # Arguments
    /// * `embeddings` - All embeddings in FP16 format
    /// * `precomputed_norms` - Precomputed vector norms
    /// * `src_indices` - Source embedding indices
    /// * `tgt_indices` - Target embedding indices
    /// * `similarities` - Output similarity scores
    /// * `batch_size` - Number of pairs to compute
    /// * `embedding_dim` - Dimension of embeddings
    pub fn launch_batch_cosine_similarity(
        &self,
        embeddings: &cudarc::driver::CudaSlice<GpuHalf>,
        precomputed_norms: &cudarc::driver::CudaSlice<f32>,
        src_indices: &cudarc::driver::CudaSlice<i32>,
        tgt_indices: &cudarc::driver::CudaSlice<i32>,
        similarities: &mut cudarc::driver::CudaSlice<f32>,
        batch_size: u32,
        embedding_dim: u32,
    ) -> GpuResult<()> {
        let func = self.batch_cosine_similarity
            .as_ref()
            .ok_or_else(|| KernelError::LaunchFailed(
                "batch_cosine_similarity kernel not loaded".to_string()
            ))?;

        // Calculate launch configuration
        // Process 16 pairs per block (optimized for tensor cores)
        let blocks = (batch_size + 15) / 16;
        let threads_per_block = 512; // 16 warps * 32 threads

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    embeddings,
                    precomputed_norms,
                    src_indices,
                    tgt_indices,
                    similarities,
                    batch_size,
                    embedding_dim,
                )
            ).map_err(|e| KernelError::LaunchFailed(format!("{}", e)))?;
        }

        Ok(())
    }

    /// Launch precompute norms kernel
    ///
    /// Computes L2 norms for all embeddings to avoid recomputation
    pub fn launch_precompute_norms(
        &self,
        embeddings: &cudarc::driver::CudaSlice<GpuHalf>,
        norms: &mut cudarc::driver::CudaSlice<f32>,
        num_vectors: u32,
        embedding_dim: u32,
    ) -> GpuResult<()> {
        let func = self.precompute_norms
            .as_ref()
            .ok_or_else(|| KernelError::LaunchFailed(
                "precompute_norms kernel not loaded".to_string()
            ))?;

        let cfg = self.optimal_launch_config(num_vectors);

        unsafe {
            func.launch(
                cfg,
                (embeddings, norms, num_vectors, embedding_dim)
            ).map_err(|e| KernelError::LaunchFailed(format!("{}", e)))?;
        }

        Ok(())
    }

    /// Launch multimodal similarity kernel with tensor cores
    ///
    /// Computes weighted similarity across visual, audio, and text modalities
    pub fn launch_multimodal_similarity(
        &self,
        visual_embeddings: &cudarc::driver::CudaSlice<GpuHalf>,
        audio_embeddings: &cudarc::driver::CudaSlice<GpuHalf>,
        text_embeddings: &cudarc::driver::CudaSlice<GpuHalf>,
        visual_norms: &cudarc::driver::CudaSlice<f32>,
        audio_norms: &cudarc::driver::CudaSlice<f32>,
        text_norms: &cudarc::driver::CudaSlice<f32>,
        src_indices: &cudarc::driver::CudaSlice<i32>,
        tgt_indices: &cudarc::driver::CudaSlice<i32>,
        similarities: &mut cudarc::driver::CudaSlice<f32>,
        num_pairs: u32,
        visual_dim: u32,
        audio_dim: u32,
        text_dim: u32,
        visual_weight: f32,
        audio_weight: f32,
        text_weight: f32,
    ) -> GpuResult<()> {
        let func = self.compute_multimodal_similarity
            .as_ref()
            .ok_or_else(|| KernelError::LaunchFailed(
                "compute_multimodal_similarity kernel not loaded".to_string()
            ))?;

        let cfg = self.optimal_launch_config(num_pairs);

        // Calculate required shared memory (2048 half elements)
        let shared_mem_bytes = 2048 * std::mem::size_of::<GpuHalf>() as u32;

        let cfg = LaunchConfig {
            grid_dim: cfg.grid_dim,
            block_dim: cfg.block_dim,
            shared_mem_bytes,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    visual_embeddings,
                    audio_embeddings,
                    text_embeddings,
                    visual_norms,
                    audio_norms,
                    text_norms,
                    src_indices,
                    tgt_indices,
                    similarities,
                    num_pairs,
                    visual_dim,
                    audio_dim,
                    text_dim,
                    visual_weight,
                    audio_weight,
                    text_weight,
                )
            ).map_err(|e| KernelError::LaunchFailed(format!("{}", e)))?;
        }

        Ok(())
    }

    /// Legacy compatibility: Launch cosine similarity kernel
    pub fn launch_cosine_similarity(
        &self,
        embeddings1: &cudarc::driver::CudaSlice<f32>,
        embeddings2: &cudarc::driver::CudaSlice<f32>,
        output: &mut cudarc::driver::CudaSlice<f32>,
        dim: u32,
        batch_size: u32,
    ) -> GpuResult<()> {
        // Note: This requires FP32->FP16 conversion
        // For now, return error directing to use FP16 version
        Err(KernelError::LaunchFailed(
            "Use launch_batch_cosine_similarity with FP16 embeddings for tensor core optimization".to_string()
        ).into())
    }

    /// Launch batch dot product kernel with tensor cores
    ///
    /// Computes dot products between pairs of embeddings using tensor cores
    pub fn launch_batch_dot_product(
        &self,
        embeddings: &cudarc::driver::CudaSlice<GpuHalf>,
        src_indices: &cudarc::driver::CudaSlice<i32>,
        tgt_indices: &cudarc::driver::CudaSlice<i32>,
        dot_products: &mut cudarc::driver::CudaSlice<f32>,
        batch_size: u32,
        embedding_dim: u32,
    ) -> GpuResult<()> {
        let func = self.batch_dot_product
            .as_ref()
            .ok_or_else(|| KernelError::LaunchFailed(
                "batch_dot_product kernel not loaded".to_string()
            ))?;

        // Each warp processes one dot product
        let warps_needed = batch_size;
        let threads_per_block = 256;
        let blocks = (warps_needed * 32 + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    embeddings,
                    src_indices,
                    tgt_indices,
                    dot_products,
                    batch_size,
                    embedding_dim,
                )
            ).map_err(|e| KernelError::LaunchFailed(format!("{}", e)))?;
        }

        Ok(())
    }

    /// Legacy batch similarity (redirects to tensor core version)
    pub fn launch_batch_similarity(
        &self,
        _embeddings: &cudarc::driver::CudaSlice<f32>,
        _output: &mut cudarc::driver::CudaSlice<f32>,
        _dim: u32,
        _batch_size: u32,
    ) -> GpuResult<()> {
        Err(KernelError::LaunchFailed(
            "Use launch_batch_dot_product with FP16 embeddings for tensor core optimization".to_string()
        ).into())
    }

    /// Launch constraint checking kernel
    pub fn launch_constraint_check(
        &self,
        _entities: &cudarc::driver::CudaSlice<u32>,
        _constraints: &cudarc::driver::CudaSlice<u32>,
        _violations: &mut cudarc::driver::CudaSlice<u32>,
        _num_entities: u32,
        _num_constraints: u32,
    ) -> GpuResult<()> {
        Err(KernelError::LaunchFailed("PTX modules not loaded".to_string()).into())
    }

    /// Launch reasoning inference kernel
    pub fn launch_reasoning_inference(
        &self,
        _facts: &cudarc::driver::CudaSlice<u32>,
        _rules: &cudarc::driver::CudaSlice<u32>,
        _inferred: &mut cudarc::driver::CudaSlice<u32>,
        _num_facts: u32,
        _num_rules: u32,
    ) -> GpuResult<()> {
        Err(KernelError::LaunchFailed("PTX modules not loaded".to_string()).into())
    }

    /// Launch BFS kernel
    pub fn launch_bfs(
        &self,
        _graph: &cudarc::driver::CudaSlice<u32>,
        _sources: &cudarc::driver::CudaSlice<u32>,
        _distances: &mut cudarc::driver::CudaSlice<u32>,
        _predecessors: &mut cudarc::driver::CudaSlice<u32>,
        _num_nodes: u32,
        _num_edges: u32,
    ) -> GpuResult<()> {
        Err(KernelError::LaunchFailed("PTX modules not loaded".to_string()).into())
    }

    /// Launch Dijkstra's algorithm kernel
    pub fn launch_dijkstra(
        &self,
        _graph: &cudarc::driver::CudaSlice<u32>,
        _weights: &cudarc::driver::CudaSlice<f32>,
        _sources: &cudarc::driver::CudaSlice<u32>,
        _distances: &mut cudarc::driver::CudaSlice<f32>,
        _predecessors: &mut cudarc::driver::CudaSlice<u32>,
        _num_nodes: u32,
        _num_edges: u32,
    ) -> GpuResult<()> {
        Err(KernelError::LaunchFailed("PTX modules not loaded".to_string()).into())
    }

    /// Calculate optimal launch configuration for given problem size
    pub fn optimal_launch_config(&self, problem_size: u32) -> LaunchConfig {
        let device_props = self.device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        ).unwrap_or(1024) as u32;

        let threads_per_block = device_props.min(256);
        let blocks = (problem_size + threads_per_block - 1) / threads_per_block;

        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Check if kernels are loaded and available
    pub fn is_loaded(&self) -> bool {
        self.batch_cosine_similarity.is_some()
            && self.batch_dot_product.is_some()
            && self.precompute_norms.is_some()
            && self.compute_multimodal_similarity.is_some()
    }

    /// Get list of loaded kernels
    pub fn loaded_kernels(&self) -> Vec<&str> {
        let mut kernels = Vec::new();
        if self.batch_cosine_similarity.is_some() {
            kernels.push("batch_cosine_similarity_tensor_cores");
        }
        if self.batch_dot_product.is_some() {
            kernels.push("batch_dot_product_tensor_cores");
        }
        if self.precompute_norms.is_some() {
            kernels.push("precompute_norms_kernel");
        }
        if self.compute_multimodal_similarity.is_some() {
            kernels.push("compute_multimodal_similarity_tensor_cores");
        }
        kernels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA device and compiled PTX files
    fn test_load_modules() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let modules = KernelModules::load(device, "./cuda_kernels/build");
        assert!(modules.is_ok());
    }
}
