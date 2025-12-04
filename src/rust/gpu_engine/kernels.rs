/// CUDA Kernel FFI Bindings
///
/// Provides safe Rust wrappers around compiled CUDA kernels with
/// proper error handling and type safety.

use cudarc::driver::{CudaDevice, CudaFunction, CudaModule, LaunchAsync, LaunchConfig};
use std::path::Path;
use std::sync::Arc;

use super::*;

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

    // Semantic similarity kernels
    similarity_module: CudaModule,
    cosine_similarity_fn: CudaFunction,
    batch_similarity_fn: CudaFunction,

    // Ontology reasoning kernels
    reasoning_module: CudaModule,
    constraint_check_fn: CudaFunction,
    inference_fn: CudaFunction,

    // Graph search kernels
    graph_module: CudaModule,
    bfs_fn: CudaFunction,
    dijkstra_fn: CudaFunction,
}

impl KernelModules {
    /// Load all kernel modules from PTX directory
    pub fn load(device: Arc<CudaDevice>, ptx_path: &str) -> GpuResult<Self> {
        let ptx_dir = Path::new(ptx_path);

        // Load semantic similarity module
        let similarity_ptx = ptx_dir.join("semantic_similarity.ptx");
        let similarity_module = Self::load_module(&device, &similarity_ptx)?;
        let cosine_similarity_fn = Self::get_function(&similarity_module, "cosine_similarity_kernel")?;
        let batch_similarity_fn = Self::get_function(&similarity_module, "batch_similarity_kernel")?;

        // Load ontology reasoning module
        let reasoning_ptx = ptx_dir.join("ontology_reasoning.ptx");
        let reasoning_module = Self::load_module(&device, &reasoning_ptx)?;
        let constraint_check_fn = Self::get_function(&reasoning_module, "check_ontology_constraints")?;
        let inference_fn = Self::get_function(&reasoning_module, "reasoning_inference_kernel")?;

        // Load graph search module
        let graph_ptx = ptx_dir.join("graph_search.ptx");
        let graph_module = Self::load_module(&device, &graph_ptx)?;
        let bfs_fn = Self::get_function(&graph_module, "bfs_kernel")?;
        let dijkstra_fn = Self::get_function(&graph_module, "dijkstra_kernel")?;

        Ok(Self {
            device,
            similarity_module,
            cosine_similarity_fn,
            batch_similarity_fn,
            reasoning_module,
            constraint_check_fn,
            inference_fn,
            graph_module,
            bfs_fn,
            dijkstra_fn,
        })
    }

    /// Load a PTX module from file
    fn load_module(device: &CudaDevice, path: &Path) -> GpuResult<CudaModule> {
        let ptx_data = std::fs::read_to_string(path)
            .map_err(|e| KernelError::ModuleLoad(format!("{}: {}", path.display(), e)))?;

        device.load_ptx(ptx_data.into(), "", &[])
            .map_err(|e| KernelError::ModuleLoad(format!("{}: {}", path.display(), e)).into())
    }

    /// Get a kernel function from a module
    fn get_function(module: &CudaModule, name: &str) -> GpuResult<CudaFunction> {
        module.get_func(name)
            .ok_or_else(|| KernelError::FunctionNotFound(name.to_string()).into())
    }

    /// Launch cosine similarity kernel
    pub fn launch_cosine_similarity(
        &self,
        embeddings1: &cudarc::driver::CudaSlice<f32>,
        embeddings2: &cudarc::driver::CudaSlice<f32>,
        output: &mut cudarc::driver::CudaSlice<f32>,
        dim: u32,
        batch_size: u32,
    ) -> GpuResult<()> {
        let threads_per_block = 256;
        let blocks = (batch_size + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.cosine_similarity_fn.launch(
                config,
                (embeddings1, embeddings2, output, dim, batch_size),
            )?;
        }

        Ok(())
    }

    /// Launch batch similarity kernel
    pub fn launch_batch_similarity(
        &self,
        embeddings: &cudarc::driver::CudaSlice<f32>,
        output: &mut cudarc::driver::CudaSlice<f32>,
        dim: u32,
        batch_size: u32,
    ) -> GpuResult<()> {
        // Use 2D grid for matrix computation
        let threads_per_block = 16; // 16x16 = 256 threads
        let blocks_x = (batch_size + threads_per_block - 1) / threads_per_block;
        let blocks_y = (batch_size + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks_x, blocks_y, 1),
            block_dim: (threads_per_block, threads_per_block, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.batch_similarity_fn.launch(
                config,
                (embeddings, output, dim, batch_size),
            )?;
        }

        Ok(())
    }

    /// Launch constraint checking kernel
    pub fn launch_constraint_check(
        &self,
        entities: &cudarc::driver::CudaSlice<u32>,
        constraints: &cudarc::driver::CudaSlice<u32>,
        violations: &mut cudarc::driver::CudaSlice<u32>,
        num_entities: u32,
        num_constraints: u32,
    ) -> GpuResult<()> {
        let threads_per_block = 256;
        let blocks = (num_entities + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.constraint_check_fn.launch(
                config,
                (entities, constraints, violations, num_entities, num_constraints),
            )?;
        }

        Ok(())
    }

    /// Launch reasoning inference kernel
    pub fn launch_reasoning_inference(
        &self,
        facts: &cudarc::driver::CudaSlice<u32>,
        rules: &cudarc::driver::CudaSlice<u32>,
        inferred: &mut cudarc::driver::CudaSlice<u32>,
        num_facts: u32,
        num_rules: u32,
    ) -> GpuResult<()> {
        let threads_per_block = 256;
        let blocks = (num_facts + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.inference_fn.launch(
                config,
                (facts, rules, inferred, num_facts, num_rules),
            )?;
        }

        Ok(())
    }

    /// Launch BFS kernel
    pub fn launch_bfs(
        &self,
        graph: &cudarc::driver::CudaSlice<u32>,
        sources: &cudarc::driver::CudaSlice<u32>,
        distances: &mut cudarc::driver::CudaSlice<u32>,
        predecessors: &mut cudarc::driver::CudaSlice<u32>,
        num_nodes: u32,
        num_edges: u32,
    ) -> GpuResult<()> {
        let threads_per_block = 256;
        let blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.bfs_fn.launch(
                config,
                (graph, sources, distances, predecessors, num_nodes, num_edges),
            )?;
        }

        Ok(())
    }

    /// Launch Dijkstra's algorithm kernel
    pub fn launch_dijkstra(
        &self,
        graph: &cudarc::driver::CudaSlice<u32>,
        weights: &cudarc::driver::CudaSlice<f32>,
        sources: &cudarc::driver::CudaSlice<u32>,
        distances: &mut cudarc::driver::CudaSlice<f32>,
        predecessors: &mut cudarc::driver::CudaSlice<u32>,
        num_nodes: u32,
        num_edges: u32,
    ) -> GpuResult<()> {
        let threads_per_block = 256;
        let blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.dijkstra_fn.launch(
                config,
                (graph, weights, sources, distances, predecessors, num_nodes, num_edges),
            )?;
        }

        Ok(())
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
