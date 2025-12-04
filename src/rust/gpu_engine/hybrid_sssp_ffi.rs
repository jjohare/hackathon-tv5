//! Hybrid SSSP CUDA Kernel FFI Bindings
//!
//! Safe Rust wrappers for hybrid Single-Source Shortest Path (SSSP) CUDA kernels.
//! Provides k-step relaxation, pivot detection, bounded Dijkstra, and frontier partitioning
//! with comprehensive error handling and memory safety guarantees.
//!
//! # Architecture
//!
//! - **Safe wrappers**: Zero-cost abstractions over raw FFI
//! - **Memory safety**: Automatic device memory management with cudarc
//! - **Stream support**: Asynchronous kernel launches
//! - **Error handling**: Result-based error propagation
//!
//! # Example
//!
//! ```rust
//! use gpu_engine::hybrid_sssp_ffi::HybridSSSPKernels;
//!
//! let kernels = HybridSSSPKernels::new(device)?;
//! let results = kernels.k_step_relaxation(
//!     &d_distances,
//!     &d_graph,
//!     &d_weights,
//!     num_nodes,
//!     k_steps,
//!     stream,
//! ).await?;
//! ```

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchConfig};
use std::sync::Arc;

use super::{GpuError, GpuResult};

// =============================================================================
// FFI Declarations
// =============================================================================

/// Raw FFI bindings to CUDA hybrid SSSP kernels
#[link(name = "graph_search", kind = "static")]
extern "C" {
    /// K-step relaxation kernel launch
    ///
    /// Performs k rounds of edge relaxation for approximate SSSP.
    /// More efficient than full Dijkstra for large sparse graphs.
    fn k_step_relaxation_launch(
        distances: *mut f32,
        predecessors: *mut i32,
        row_offsets: *const i32,
        col_indices: *const i32,
        edge_weights: *const f32,
        num_nodes: i32,
        num_edges: i32,
        k_steps: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Detect pivot nodes for hybrid algorithm
    ///
    /// Identifies critical nodes where exact Dijkstra is needed.
    /// Returns indices of pivot nodes detected.
    fn detect_pivots_launch(
        distances: *const f32,
        frontier: *const i32,
        frontier_size: i32,
        pivots: *mut i32,
        pivot_count: *mut i32,
        convergence_threshold: f32,
        degree_threshold: i32,
        row_offsets: *const i32,
        num_nodes: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Bounded Dijkstra around pivots
    ///
    /// Runs exact Dijkstra within radius around pivot nodes.
    /// Combines with k-step relaxation for hybrid approach.
    fn bounded_dijkstra_launch(
        distances: *mut f32,
        predecessors: *mut i32,
        pivots: *const i32,
        pivot_count: i32,
        row_offsets: *const i32,
        col_indices: *const i32,
        edge_weights: *const f32,
        num_nodes: i32,
        num_edges: i32,
        radius: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Partition frontier into active/inactive
    ///
    /// Filters frontier nodes based on distance changes.
    /// Compacts frontier for more efficient iteration.
    fn partition_frontier_launch(
        frontier: *const i32,
        frontier_size: i32,
        distances: *const f32,
        old_distances: *const f32,
        active_frontier: *mut i32,
        active_count: *mut i32,
        epsilon: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Compact frontier on GPU
    ///
    /// Removes invalid/duplicate entries from frontier.
    /// Uses stream compaction for efficiency.
    fn compact_frontier_gpu(
        frontier: *const i32,
        frontier_size: i32,
        valid_flags: *const i32,
        compacted_frontier: *mut i32,
        compacted_size: *mut i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Initialize distances array
    ///
    /// Sets source distance to 0, all others to infinity.
    fn initialize_distances_launch(
        distances: *mut f32,
        predecessors: *mut i32,
        source: i32,
        num_nodes: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// SSSP semantic kernel for content discovery
    ///
    /// Main frontier-based SSSP with semantic scoring for media graphs.
    fn sssp_semantic_kernel_launch(
        source: i32,
        distances: *mut f32,
        predecessors: *mut i32,
        semantic_scores: *mut f32,
        row_offsets: *const i32,
        col_indices: *const i32,
        edge_weights: *const f32,
        content_features: *const f32,
        user_affinities: *const f32,
        frontier: *const i32,
        frontier_size: i32,
        next_frontier: *mut i32,
        next_frontier_size: *mut i32,
        num_nodes: i32,
        max_hops: i32,
        min_similarity: f32,
        blocks: i32,
        threads: i32,
        shared_mem: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Select content landmarks for APSP
    ///
    /// Stratified sampling with content diversity for landmark selection.
    fn select_content_landmarks_kernel_launch(
        landmarks: *mut i32,
        content_clusters: *const i32,
        node_degrees: *const i32,
        num_nodes: i32,
        num_landmarks: i32,
        seed: u64,
        blocks: i32,
        threads: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Approximate APSP using landmarks
    ///
    /// Triangle inequality-based distance approximation.
    fn approximate_apsp_content_kernel_launch(
        landmark_distances: *const f32,
        distance_matrix: *mut f32,
        quality_scores: *mut f32,
        num_nodes: i32,
        num_landmarks: i32,
        blocks_x: i32,
        blocks_y: i32,
        threads_x: i32,
        threads_y: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

// =============================================================================
// Safe Rust Wrappers
// =============================================================================

/// Hybrid SSSP kernel collection with safe Rust API
pub struct HybridSSSPKernels {
    device: Arc<CudaDevice>,
}

impl HybridSSSPKernels {
    /// Create new kernel wrapper for device
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Perform k-step edge relaxation
    ///
    /// Approximates SSSP by relaxing all edges k times in parallel.
    /// More efficient than Dijkstra for large sparse graphs with small k.
    ///
    /// # Arguments
    ///
    /// * `distances` - Distance array to update [num_nodes]
    /// * `predecessors` - Predecessor array [num_nodes]
    /// * `row_offsets` - CSR row offsets [num_nodes + 1]
    /// * `col_indices` - CSR column indices [num_edges]
    /// * `edge_weights` - Edge weights [num_edges]
    /// * `k_steps` - Number of relaxation rounds
    /// * `stream` - CUDA stream for async execution
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn k_step_relaxation(
        &self,
        distances: &mut CudaSlice<f32>,
        predecessors: &mut CudaSlice<i32>,
        row_offsets: &CudaSlice<i32>,
        col_indices: &CudaSlice<i32>,
        edge_weights: &CudaSlice<f32>,
        k_steps: u32,
        stream: &CudaStream,
    ) -> GpuResult<()> {
        let num_nodes = distances.len() as i32;
        let num_edges = col_indices.len() as i32;

        if num_nodes == 0 || num_edges == 0 {
            return Err(GpuError::Config("Empty graph".to_string()));
        }

        if row_offsets.len() != (num_nodes + 1) as usize {
            return Err(GpuError::Config(format!(
                "Invalid CSR format: row_offsets length {} != num_nodes + 1 {}",
                row_offsets.len(),
                num_nodes + 1
            )));
        }

        let result = unsafe {
            k_step_relaxation_launch(
                distances.as_mut_ptr(),
                predecessors.as_mut_ptr(),
                row_offsets.as_ptr(),
                col_indices.as_ptr(),
                edge_weights.as_ptr(),
                num_nodes,
                num_edges,
                k_steps as i32,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "k_step_relaxation_launch failed with code {}",
                result
            ).into()));
        }

        Ok(())
    }

    /// Detect pivot nodes for hybrid algorithm
    ///
    /// Identifies nodes where distance estimates have poor convergence
    /// or high degree (hubs). These receive exact Dijkstra treatment.
    ///
    /// # Arguments
    ///
    /// * `distances` - Current distance estimates [num_nodes]
    /// * `frontier` - Current frontier nodes [frontier_size]
    /// * `row_offsets` - CSR row offsets for degree computation [num_nodes + 1]
    /// * `convergence_threshold` - Distance change threshold for pivot detection
    /// * `degree_threshold` - Minimum degree for hub detection
    ///
    /// # Returns
    ///
    /// Vector of pivot node indices
    pub fn detect_pivots(
        &self,
        distances: &CudaSlice<f32>,
        frontier: &CudaSlice<i32>,
        row_offsets: &CudaSlice<i32>,
        convergence_threshold: f32,
        degree_threshold: i32,
        stream: &CudaStream,
    ) -> GpuResult<Vec<u32>> {
        let num_nodes = distances.len() as i32;
        let frontier_size = frontier.len() as i32;

        // Allocate output buffers
        let mut d_pivots = self.device.alloc_zeros::<i32>(num_nodes as usize)?;
        let mut d_pivot_count = self.device.alloc_zeros::<i32>(1)?;

        let result = unsafe {
            detect_pivots_launch(
                distances.as_ptr(),
                frontier.as_ptr(),
                frontier_size,
                d_pivots.as_mut_ptr(),
                d_pivot_count.as_mut_ptr(),
                convergence_threshold,
                degree_threshold,
                row_offsets.as_ptr(),
                num_nodes,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "detect_pivots_launch failed with code {}",
                result
            ).into()));
        }

        // Copy pivot count back
        let pivot_count = self.device.dtoh_sync_copy(&d_pivot_count)?[0] as usize;

        // Copy pivots back
        let pivots_i32 = self.device.dtoh_sync_copy(&d_pivots)?;
        let pivots: Vec<u32> = pivots_i32[..pivot_count]
            .iter()
            .map(|&x| x as u32)
            .collect();

        Ok(pivots)
    }

    /// Run bounded Dijkstra around pivot nodes
    ///
    /// Performs exact Dijkstra within specified radius of each pivot.
    /// Combines with k-step relaxation for hybrid SSSP approach.
    ///
    /// # Arguments
    ///
    /// * `distances` - Distance array to refine [num_nodes]
    /// * `predecessors` - Predecessor array [num_nodes]
    /// * `pivots` - Pivot node indices to process
    /// * `row_offsets` - CSR row offsets [num_nodes + 1]
    /// * `col_indices` - CSR column indices [num_edges]
    /// * `edge_weights` - Edge weights [num_edges]
    /// * `radius` - Search radius around each pivot
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn bounded_dijkstra(
        &self,
        distances: &mut CudaSlice<f32>,
        predecessors: &mut CudaSlice<i32>,
        pivots: &[u32],
        row_offsets: &CudaSlice<i32>,
        col_indices: &CudaSlice<i32>,
        edge_weights: &CudaSlice<f32>,
        radius: u32,
        stream: &CudaStream,
    ) -> GpuResult<()> {
        if pivots.is_empty() {
            return Ok(());
        }

        let num_nodes = distances.len() as i32;
        let num_edges = col_indices.len() as i32;

        // Convert pivots to i32 and upload
        let pivots_i32: Vec<i32> = pivots.iter().map(|&x| x as i32).collect();
        let d_pivots = self.device.htod_sync_copy(&pivots_i32)?;

        let result = unsafe {
            bounded_dijkstra_launch(
                distances.as_mut_ptr(),
                predecessors.as_mut_ptr(),
                d_pivots.as_ptr(),
                pivots.len() as i32,
                row_offsets.as_ptr(),
                col_indices.as_ptr(),
                edge_weights.as_ptr(),
                num_nodes,
                num_edges,
                radius as i32,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "bounded_dijkstra_launch failed with code {}",
                result
            ).into()));
        }

        Ok(())
    }

    /// Partition frontier into active and inactive nodes
    ///
    /// Filters frontier based on distance changes since last iteration.
    /// Produces compacted frontier for next iteration.
    ///
    /// # Arguments
    ///
    /// * `frontier` - Current frontier [frontier_size]
    /// * `distances` - Current distances [num_nodes]
    /// * `old_distances` - Previous distances [num_nodes]
    /// * `epsilon` - Minimum distance change to be considered active
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    ///
    /// Active frontier indices
    pub fn partition_frontier(
        &self,
        frontier: &CudaSlice<i32>,
        distances: &CudaSlice<f32>,
        old_distances: &CudaSlice<f32>,
        epsilon: f32,
        stream: &CudaStream,
    ) -> GpuResult<Vec<u32>> {
        let frontier_size = frontier.len() as i32;

        // Allocate output buffers
        let mut d_active = self.device.alloc_zeros::<i32>(frontier_size as usize)?;
        let mut d_active_count = self.device.alloc_zeros::<i32>(1)?;

        let result = unsafe {
            partition_frontier_launch(
                frontier.as_ptr(),
                frontier_size,
                distances.as_ptr(),
                old_distances.as_ptr(),
                d_active.as_mut_ptr(),
                d_active_count.as_mut_ptr(),
                epsilon,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "partition_frontier_launch failed with code {}",
                result
            ).into()));
        }

        // Copy active count
        let active_count = self.device.dtoh_sync_copy(&d_active_count)?[0] as usize;

        // Copy active frontier
        let active_i32 = self.device.dtoh_sync_copy(&d_active)?;
        let active: Vec<u32> = active_i32[..active_count]
            .iter()
            .map(|&x| x as u32)
            .collect();

        Ok(active)
    }

    /// Compact frontier by removing invalid entries
    ///
    /// Uses stream compaction to remove nodes marked as invalid.
    /// Produces dense frontier array for next iteration.
    ///
    /// # Arguments
    ///
    /// * `frontier` - Input frontier [frontier_size]
    /// * `valid_flags` - Validity flags (1=valid, 0=invalid) [frontier_size]
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    ///
    /// Compacted frontier indices
    pub fn compact_frontier(
        &self,
        frontier: &CudaSlice<i32>,
        valid_flags: &CudaSlice<i32>,
        stream: &CudaStream,
    ) -> GpuResult<Vec<u32>> {
        let frontier_size = frontier.len() as i32;

        if frontier_size == 0 {
            return Ok(Vec::new());
        }

        // Allocate output buffers
        let mut d_compacted = self.device.alloc_zeros::<i32>(frontier_size as usize)?;
        let mut d_compacted_size = self.device.alloc_zeros::<i32>(1)?;

        let result = unsafe {
            compact_frontier_gpu(
                frontier.as_ptr(),
                frontier_size,
                valid_flags.as_ptr(),
                d_compacted.as_mut_ptr(),
                d_compacted_size.as_mut_ptr(),
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "compact_frontier_gpu failed with code {}",
                result
            ).into()));
        }

        // Copy compacted size
        let compacted_size = self.device.dtoh_sync_copy(&d_compacted_size)?[0] as usize;

        // Copy compacted frontier
        let compacted_i32 = self.device.dtoh_sync_copy(&d_compacted)?;
        let compacted: Vec<u32> = compacted_i32[..compacted_size]
            .iter()
            .map(|&x| x as u32)
            .collect();

        Ok(compacted)
    }

    /// Initialize distances for SSSP
    ///
    /// Sets source distance to 0, all others to infinity.
    /// Sets all predecessors to -1 (invalid).
    ///
    /// # Arguments
    ///
    /// * `distances` - Distance array [num_nodes]
    /// * `predecessors` - Predecessor array [num_nodes]
    /// * `source` - Source node index
    /// * `stream` - CUDA stream
    pub fn initialize_distances(
        &self,
        distances: &mut CudaSlice<f32>,
        predecessors: &mut CudaSlice<i32>,
        source: u32,
        stream: &CudaStream,
    ) -> GpuResult<()> {
        let num_nodes = distances.len() as i32;

        let result = unsafe {
            initialize_distances_launch(
                distances.as_mut_ptr(),
                predecessors.as_mut_ptr(),
                source as i32,
                num_nodes,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "initialize_distances_launch failed with code {}",
                result
            ).into()));
        }

        Ok(())
    }

    /// Launch SSSP semantic kernel for content discovery
    ///
    /// Performs frontier-based SSSP with semantic scoring for media recommendation graphs.
    /// Combines graph distance with content similarity and user affinity.
    ///
    /// # Arguments
    ///
    /// * `source` - Source content node
    /// * `distances` - Distance array [num_nodes]
    /// * `predecessors` - Predecessor array [num_nodes]
    /// * `semantic_scores` - Semantic quality scores [num_nodes]
    /// * `row_offsets` - CSR row offsets [num_nodes + 1]
    /// * `col_indices` - CSR column indices [num_edges]
    /// * `edge_weights` - Edge weights [num_edges]
    /// * `content_features` - Content similarity scores [num_edges]
    /// * `user_affinities` - User affinity scores [num_nodes]
    /// * `frontier` - Current frontier [frontier_size]
    /// * `next_frontier` - Output next frontier [num_nodes]
    /// * `next_frontier_size` - Output next frontier size [1]
    /// * `max_hops` - Maximum path length
    /// * `min_similarity` - Minimum semantic similarity threshold
    /// * `stream` - CUDA stream
    pub fn sssp_semantic(
        &self,
        source: u32,
        distances: &mut CudaSlice<f32>,
        predecessors: &mut CudaSlice<i32>,
        semantic_scores: &mut CudaSlice<f32>,
        row_offsets: &CudaSlice<i32>,
        col_indices: &CudaSlice<i32>,
        edge_weights: &CudaSlice<f32>,
        content_features: &CudaSlice<f32>,
        user_affinities: &CudaSlice<f32>,
        frontier: &CudaSlice<i32>,
        next_frontier: &mut CudaSlice<i32>,
        next_frontier_size: &mut CudaSlice<i32>,
        max_hops: u32,
        min_similarity: f32,
        stream: &CudaStream,
    ) -> GpuResult<()> {
        let num_nodes = distances.len() as i32;
        let frontier_size = frontier.len() as i32;

        // Calculate launch configuration
        let threads = 256;
        let blocks = (frontier_size + threads - 1) / threads;
        let shared_mem = (frontier_size * std::mem::size_of::<i32>() as i32).min(48 * 1024);

        let result = unsafe {
            sssp_semantic_kernel_launch(
                source as i32,
                distances.as_mut_ptr(),
                predecessors.as_mut_ptr(),
                semantic_scores.as_mut_ptr(),
                row_offsets.as_ptr(),
                col_indices.as_ptr(),
                edge_weights.as_ptr(),
                content_features.as_ptr(),
                user_affinities.as_ptr(),
                frontier.as_ptr(),
                frontier_size,
                next_frontier.as_mut_ptr(),
                next_frontier_size.as_mut_ptr(),
                num_nodes,
                max_hops as i32,
                min_similarity,
                blocks,
                threads,
                shared_mem,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "sssp_semantic_kernel_launch failed with code {}",
                result
            ).into()));
        }

        Ok(())
    }

    /// Select content landmarks for approximate APSP
    ///
    /// Uses stratified sampling with hub detection to select diverse landmarks.
    ///
    /// # Arguments
    ///
    /// * `content_clusters` - Cluster assignments [num_nodes]
    /// * `node_degrees` - Node degree counts [num_nodes]
    /// * `num_landmarks` - Number of landmarks to select
    /// * `seed` - Random seed for sampling
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    ///
    /// Vector of landmark node indices
    pub fn select_landmarks(
        &self,
        content_clusters: &CudaSlice<i32>,
        node_degrees: &CudaSlice<i32>,
        num_landmarks: u32,
        seed: u64,
        stream: &CudaStream,
    ) -> GpuResult<Vec<u32>> {
        let num_nodes = content_clusters.len() as i32;

        // Allocate output
        let mut d_landmarks = self.device.alloc_zeros::<i32>(num_landmarks as usize)?;

        // Launch configuration
        let threads = 256;
        let blocks = (num_landmarks as i32 + threads - 1) / threads;

        let result = unsafe {
            select_content_landmarks_kernel_launch(
                d_landmarks.as_mut_ptr(),
                content_clusters.as_ptr(),
                node_degrees.as_ptr(),
                num_nodes,
                num_landmarks as i32,
                seed,
                blocks,
                threads,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "select_content_landmarks_kernel_launch failed with code {}",
                result
            ).into()));
        }

        // Copy landmarks back
        let landmarks_i32 = self.device.dtoh_sync_copy(&d_landmarks)?;
        let landmarks: Vec<u32> = landmarks_i32.iter().map(|&x| x as u32).collect();

        Ok(landmarks)
    }

    /// Approximate all-pairs shortest paths using landmarks
    ///
    /// Uses triangle inequality with landmark distances to approximate
    /// distances between all node pairs.
    ///
    /// # Arguments
    ///
    /// * `landmark_distances` - Distances from landmarks [num_landmarks * num_nodes]
    /// * `distance_matrix` - Output distance matrix [num_nodes * num_nodes]
    /// * `quality_scores` - Output quality scores [num_nodes * num_nodes]
    /// * `num_nodes` - Number of nodes
    /// * `num_landmarks` - Number of landmarks
    /// * `stream` - CUDA stream
    pub fn approximate_apsp(
        &self,
        landmark_distances: &CudaSlice<f32>,
        distance_matrix: &mut CudaSlice<f32>,
        quality_scores: &mut CudaSlice<f32>,
        num_nodes: u32,
        num_landmarks: u32,
        stream: &CudaStream,
    ) -> GpuResult<()> {
        // 2D launch configuration for distance matrix
        let threads_x = 16;
        let threads_y = 16;
        let blocks_x = (num_nodes as i32 + threads_x - 1) / threads_x;
        let blocks_y = (num_nodes as i32 + threads_y - 1) / threads_y;

        let result = unsafe {
            approximate_apsp_content_kernel_launch(
                landmark_distances.as_ptr(),
                distance_matrix.as_mut_ptr(),
                quality_scores.as_mut_ptr(),
                num_nodes as i32,
                num_landmarks as i32,
                blocks_x,
                blocks_y,
                threads_x,
                threads_y,
                stream.stream_ptr() as *mut _,
            )
        };

        if result != 0 {
            return Err(GpuError::Kernel(format!(
                "approximate_apsp_content_kernel_launch failed with code {}",
                result
            ).into()));
        }

        Ok(())
    }
}

// =============================================================================
// Configuration and Helper Types
// =============================================================================

/// Configuration for hybrid SSSP algorithm
#[derive(Debug, Clone)]
pub struct HybridSSSPConfig {
    /// Number of k-step relaxation rounds
    pub k_steps: u32,

    /// Convergence threshold for pivot detection
    pub convergence_threshold: f32,

    /// Minimum degree for hub detection
    pub degree_threshold: u32,

    /// Radius for bounded Dijkstra around pivots
    pub dijkstra_radius: u32,

    /// Epsilon for frontier partitioning
    pub frontier_epsilon: f32,

    /// Maximum hops for semantic search
    pub max_hops: u32,

    /// Minimum similarity threshold
    pub min_similarity: f32,
}

impl Default for HybridSSSPConfig {
    fn default() -> Self {
        Self {
            k_steps: 10,
            convergence_threshold: 0.01,
            degree_threshold: 100,
            dijkstra_radius: 3,
            frontier_epsilon: 1e-6,
            max_hops: 10,
            min_similarity: 0.5,
        }
    }
}

/// Results from SSSP computation
#[derive(Debug, Clone)]
pub struct SSSPResult {
    /// Shortest distances from source [num_nodes]
    pub distances: Vec<f32>,

    /// Predecessor nodes in shortest paths [num_nodes]
    pub predecessors: Vec<i32>,

    /// Semantic quality scores [num_nodes]
    pub semantic_scores: Option<Vec<f32>>,
}

impl SSSPResult {
    /// Reconstruct path from source to target
    pub fn reconstruct_path(&self, target: u32) -> Option<Vec<u32>> {
        let target_idx = target as usize;

        if target_idx >= self.distances.len() {
            return None;
        }

        if self.distances[target_idx].is_infinite() {
            return None; // No path exists
        }

        let mut path = Vec::new();
        let mut current = target_idx;

        path.push(current as u32);

        while self.predecessors[current] != -1 {
            current = self.predecessors[current] as usize;
            path.push(current as u32);

            if path.len() > self.distances.len() {
                return None; // Cycle detected
            }
        }

        path.reverse();
        Some(path)
    }

    /// Get distance to target
    pub fn distance_to(&self, target: u32) -> Option<f32> {
        let target_idx = target as usize;
        if target_idx < self.distances.len() {
            let dist = self.distances[target_idx];
            if dist.is_finite() {
                return Some(dist);
            }
        }
        None
    }

    /// Get semantic score for target
    pub fn semantic_score(&self, target: u32) -> Option<f32> {
        if let Some(ref scores) = self.semantic_scores {
            let target_idx = target as usize;
            if target_idx < scores.len() {
                return Some(scores[target_idx]);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HybridSSSPConfig::default();
        assert_eq!(config.k_steps, 10);
        assert_eq!(config.degree_threshold, 100);
        assert!(config.convergence_threshold > 0.0);
    }

    #[test]
    fn test_sssp_result_reconstruction() {
        let result = SSSPResult {
            distances: vec![0.0, 1.0, 2.0, 3.0],
            predecessors: vec![-1, 0, 1, 2],
            semantic_scores: None,
        };

        let path = result.reconstruct_path(3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);

        assert_eq!(result.distance_to(3), Some(3.0));
    }

    #[test]
    fn test_no_path() {
        let result = SSSPResult {
            distances: vec![0.0, f32::INFINITY, f32::INFINITY],
            predecessors: vec![-1, -1, -1],
            semantic_scores: None,
        };

        assert!(result.reconstruct_path(2).is_none());
        assert!(result.distance_to(2).is_none());
    }
}
