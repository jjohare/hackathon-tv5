// T4 GPU Configuration and Management
// Google T4 (Turing sm_75) specific optimizations

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// T4 GPU specifications and runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T4Config {
    /// Device ID (0-based)
    pub device_id: i32,

    /// Number of streaming multiprocessors (40 for T4)
    pub sm_count: u32,

    /// CUDA cores per SM (64 for Turing)
    pub cores_per_sm: u32,

    /// Total CUDA cores (2560)
    pub total_cores: u32,

    /// Tensor cores (320 for T4, FP16 only)
    pub tensor_cores: u32,

    /// VRAM size in GB (16 for T4)
    pub vram_gb: u32,

    /// Memory bandwidth in GB/s (320 for T4)
    pub memory_bandwidth_gbs: u32,

    /// PCIe generation (3 for T4)
    pub pcie_gen: u32,

    /// Use FP16 tensor cores
    pub use_fp16: bool,

    /// Maximum batch size for memory constraints
    pub max_batch_size: usize,

    /// Occupancy target (threads per SM)
    pub target_occupancy: u32,
}

impl Default for T4Config {
    fn default() -> Self {
        Self {
            device_id: 0,
            sm_count: 40,
            cores_per_sm: 64,
            total_cores: 2560,
            tensor_cores: 320,
            vram_gb: 16,
            memory_bandwidth_gbs: 320,
            pcie_gen: 3,
            use_fp16: true,
            max_batch_size: 131072, // 128K vectors
            target_occupancy: 2048,  // 32 warps per SM
        }
    }
}

impl T4Config {
    /// Create T4 configuration for specific device
    pub fn new(device_id: i32) -> Self {
        Self {
            device_id,
            ..Default::default()
        }
    }

    /// Calculate optimal block size for kernel launch
    pub fn optimal_block_size(&self, workload_type: WorkloadType) -> u32 {
        match workload_type {
            WorkloadType::MemoryBound => 256,    // Memory-bound operations
            WorkloadType::ComputeBound => 256,   // Compute-bound (tensor cores)
            WorkloadType::MatrixOps => 256,      // 16×16 blocks for tensor cores
            WorkloadType::Reduction => 512,      // Higher occupancy for reductions
        }
    }

    /// Calculate optimal grid size for kernel launch
    pub fn optimal_grid_size(&self, num_elements: usize, block_size: u32) -> u32 {
        let blocks_needed = (num_elements as u32 + block_size - 1) / block_size;

        // Limit to 2-4× SM count for good occupancy
        let max_blocks = self.sm_count * 4;
        blocks_needed.min(max_blocks)
    }

    /// Calculate memory budget for embeddings
    pub fn memory_budget(&self, embedding_dim: usize, safety_margin: f32) -> MemoryBudget {
        let total_vram_bytes = (self.vram_gb as u64) * 1024 * 1024 * 1024;
        let available_bytes = (total_vram_bytes as f32 * safety_margin) as u64;

        // FP16: 2 bytes per element, FP32: 4 bytes
        let bytes_per_vector = if self.use_fp16 {
            embedding_dim * 2
        } else {
            embedding_dim * 4
        };

        let max_vectors = (available_bytes / bytes_per_vector as u64) as usize;

        MemoryBudget {
            total_vram_bytes,
            available_bytes,
            bytes_per_vector,
            max_vectors,
            num_batches: 1,
        }
    }

    /// Calculate expected throughput (queries/sec)
    pub fn expected_throughput(&self, embedding_dim: usize, batch_size: usize) -> f64 {
        // T4 FP16 tensor core peak: 65 TFLOPS
        let peak_tflops = if self.use_fp16 { 65.0 } else { 8.1 };

        // Cosine similarity: 2×dim FLOPs (dot + norms)
        let flops_per_comparison = (2 * embedding_dim) as f64;
        let flops_per_batch = flops_per_comparison * batch_size as f64;

        // Memory bandwidth limit: 320 GB/s
        let bytes_per_batch = if self.use_fp16 {
            (embedding_dim * 2 * batch_size * 2) as f64  // Read 2 vectors per comparison
        } else {
            (embedding_dim * 4 * batch_size * 2) as f64
        };

        let memory_time_sec = bytes_per_batch / (self.memory_bandwidth_gbs as f64 * 1e9);
        let compute_time_sec = flops_per_batch / (peak_tflops * 1e12);

        // Bottleneck is max of compute and memory time
        let time_per_batch = memory_time_sec.max(compute_time_sec);

        batch_size as f64 / time_per_batch
    }

    /// Check if multi-GPU is available
    pub fn multi_gpu_available(&self) -> bool {
        // Check CUDA runtime for peer access
        // T4 uses PCIe Gen3, so peer-to-peer is over PCIe
        unsafe {
            let mut can_access = 0i32;
            cuda_sys::cudaDeviceCanAccessPeer(&mut can_access, self.device_id, self.device_id + 1);
            can_access == 1
        }
    }
}

/// Workload type for kernel optimization
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    MemoryBound,   // Bandwidth-limited operations
    ComputeBound,  // Compute-limited (tensor cores)
    MatrixOps,     // Matrix operations (WMMA)
    Reduction,     // Reduction operations
}

/// Memory budget calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub total_vram_bytes: u64,
    pub available_bytes: u64,
    pub bytes_per_vector: usize,
    pub max_vectors: usize,
    pub num_batches: usize,
}

impl MemoryBudget {
    /// Calculate number of batches needed for dataset
    pub fn batches_needed(&self, total_vectors: usize) -> usize {
        (total_vectors + self.max_vectors - 1) / self.max_vectors
    }

    /// Get batch size for specific batch index
    pub fn batch_size(&self, total_vectors: usize, batch_idx: usize) -> usize {
        let remaining = total_vectors - (batch_idx * self.max_vectors);
        remaining.min(self.max_vectors)
    }

    /// Print memory budget summary
    pub fn print_summary(&self) {
        println!("T4 Memory Budget:");
        println!("  Total VRAM: {:.2} GB", self.total_vram_bytes as f64 / 1e9);
        println!("  Available: {:.2} GB", self.available_bytes as f64 / 1e9);
        println!("  Bytes per vector: {}", self.bytes_per_vector);
        println!("  Max vectors per batch: {}", self.max_vectors);
        println!("  Batches needed: {}", self.num_batches);
    }
}

/// Multi-GPU configuration for T4 clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGPUT4Config {
    /// Number of GPUs
    pub num_gpus: usize,

    /// Individual GPU configurations
    pub gpu_configs: Vec<T4Config>,

    /// Use NCCL for communication
    pub use_nccl: bool,

    /// Async copy streams
    pub num_streams: usize,
}

impl MultiGPUT4Config {
    /// Create multi-GPU configuration
    pub fn new(num_gpus: usize) -> Self {
        let gpu_configs = (0..num_gpus)
            .map(|i| T4Config::new(i as i32))
            .collect();

        Self {
            num_gpus,
            gpu_configs,
            use_nccl: true,
            num_streams: 4,
        }
    }

    /// Calculate total throughput for all GPUs
    pub fn total_throughput(&self, embedding_dim: usize, batch_size: usize) -> f64 {
        self.gpu_configs
            .iter()
            .map(|cfg| cfg.expected_throughput(embedding_dim, batch_size))
            .sum()
    }

    /// Distribute workload across GPUs
    pub fn distribute_workload(&self, total_vectors: usize) -> Vec<WorkloadDistribution> {
        let vectors_per_gpu = (total_vectors + self.num_gpus - 1) / self.num_gpus;

        self.gpu_configs
            .iter()
            .enumerate()
            .map(|(i, config)| {
                let start_idx = i * vectors_per_gpu;
                let end_idx = ((i + 1) * vectors_per_gpu).min(total_vectors);
                let count = end_idx - start_idx;

                WorkloadDistribution {
                    device_id: config.device_id,
                    start_idx,
                    end_idx,
                    count,
                }
            })
            .collect()
    }

    /// Print multi-GPU configuration summary
    pub fn print_summary(&self) {
        println!("Multi-GPU T4 Configuration:");
        println!("  Number of GPUs: {}", self.num_gpus);
        println!("  Total CUDA cores: {}", self.num_gpus * 2560);
        println!("  Total tensor cores: {}", self.num_gpus * 320);
        println!("  Total VRAM: {} GB", self.num_gpus * 16);
        println!("  Total memory bandwidth: {} GB/s", self.num_gpus * 320);
        println!("  NCCL enabled: {}", self.use_nccl);
        println!("  Async streams: {}", self.num_streams);
    }
}

/// Workload distribution for multi-GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadDistribution {
    pub device_id: i32,
    pub start_idx: usize,
    pub end_idx: usize,
    pub count: usize,
}

/// Runtime T4 GPU manager
pub struct T4Manager {
    config: Arc<RwLock<T4Config>>,
    multi_gpu: Option<Arc<RwLock<MultiGPUT4Config>>>,
}

impl T4Manager {
    /// Create new T4 manager
    pub fn new(device_id: i32) -> Self {
        Self {
            config: Arc::new(RwLock::new(T4Config::new(device_id))),
            multi_gpu: None,
        }
    }

    /// Create multi-GPU manager
    pub fn new_multi_gpu(num_gpus: usize) -> Self {
        Self {
            config: Arc::new(RwLock::new(T4Config::new(0))),
            multi_gpu: Some(Arc::new(RwLock::new(MultiGPUT4Config::new(num_gpus)))),
        }
    }

    /// Get single GPU configuration
    pub async fn get_config(&self) -> T4Config {
        self.config.read().await.clone()
    }

    /// Get multi-GPU configuration
    pub async fn get_multi_gpu_config(&self) -> Option<MultiGPUT4Config> {
        if let Some(ref multi_gpu) = self.multi_gpu {
            Some(multi_gpu.read().await.clone())
        } else {
            None
        }
    }

    /// Update configuration
    pub async fn update_config<F>(&self, f: F)
    where
        F: FnOnce(&mut T4Config),
    {
        let mut config = self.config.write().await;
        f(&mut config);
    }

    /// Check GPU health and availability
    pub async fn health_check(&self) -> Result<HealthStatus, String> {
        let config = self.config.read().await;

        // Check CUDA runtime
        unsafe {
            let mut props: cuda_sys::cudaDeviceProp = std::mem::zeroed();
            let result = cuda_sys::cudaGetDeviceProperties(&mut props, config.device_id);

            if result != cuda_sys::cudaError::cudaSuccess {
                return Err(format!("CUDA error: {:?}", result));
            }

            // Check memory
            let mut free_mem = 0u64;
            let mut total_mem = 0u64;
            cuda_sys::cudaMemGetInfo(&mut free_mem, &mut total_mem);

            let free_gb = free_mem as f64 / 1e9;
            let total_gb = total_mem as f64 / 1e9;
            let utilization = (total_mem - free_mem) as f64 / total_mem as f64;

            Ok(HealthStatus {
                device_id: config.device_id,
                available: true,
                free_memory_gb: free_gb,
                total_memory_gb: total_gb,
                memory_utilization: utilization,
                compute_capability: format!("{}.{}", props.major, props.minor),
            })
        }
    }
}

/// GPU health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub device_id: i32,
    pub available: bool,
    pub free_memory_gb: f64,
    pub total_memory_gb: f64,
    pub memory_utilization: f64,
    pub compute_capability: String,
}

// CUDA FFI bindings
#[allow(dead_code)]
mod cuda_sys {
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub enum cudaError {
        cudaSuccess = 0,
    }

    #[repr(C)]
    pub struct cudaDeviceProp {
        pub name: [u8; 256],
        pub major: i32,
        pub minor: i32,
        pub totalGlobalMem: usize,
        pub sharedMemPerBlock: usize,
        pub regsPerBlock: i32,
        pub warpSize: i32,
        pub memPitch: usize,
        pub maxThreadsPerBlock: i32,
        pub maxThreadsDim: [i32; 3],
        pub maxGridSize: [i32; 3],
        pub clockRate: i32,
        pub totalConstMem: usize,
    }

    extern "C" {
        pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> cudaError;
        pub fn cudaMemGetInfo(free: *mut u64, total: *mut u64) -> cudaError;
        pub fn cudaDeviceCanAccessPeer(can_access: *mut i32, device: i32, peer_device: i32) -> cudaError;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t4_config_defaults() {
        let config = T4Config::default();
        assert_eq!(config.sm_count, 40);
        assert_eq!(config.total_cores, 2560);
        assert_eq!(config.tensor_cores, 320);
        assert_eq!(config.vram_gb, 16);
    }

    #[test]
    fn test_memory_budget() {
        let config = T4Config::default();
        let budget = config.memory_budget(768, 0.8);

        assert!(budget.max_vectors > 0);
        assert!(budget.available_bytes < budget.total_vram_bytes);

        budget.print_summary();
    }

    #[test]
    fn test_throughput_estimation() {
        let config = T4Config::default();
        let throughput = config.expected_throughput(768, 10000);

        println!("Expected throughput: {:.2} queries/sec", throughput);
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_multi_gpu_distribution() {
        let multi_config = MultiGPUT4Config::new(4);
        let distribution = multi_config.distribute_workload(1_000_000);

        assert_eq!(distribution.len(), 4);

        let total_vectors: usize = distribution.iter().map(|d| d.count).sum();
        assert_eq!(total_vectors, 1_000_000);

        multi_config.print_summary();
    }
}
