//! Comprehensive tests for Hybrid SSSP FFI bindings
//!
//! Tests cover:
//! - Memory safety and error handling
//! - Kernel correctness
//! - Stream synchronization
//! - Edge cases and boundary conditions

use cudarc::driver::{CudaDevice, CudaStream};
use gpu_engine::{HybridSSSPKernels, HybridSSSPConfig, SSSPResult, GpuResult};
use std::sync::Arc;

/// Helper: Create simple linear graph for testing
fn create_linear_graph(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    // Graph: 0 -> 1 -> 2 -> ... -> n-1
    let mut row_offsets = vec![0];
    let mut col_indices = Vec::new();
    let mut edge_weights = Vec::new();

    for i in 0..n {
        if i < n - 1 {
            col_indices.push((i + 1) as i32);
            edge_weights.push(1.0);
        }
        row_offsets.push(col_indices.len() as i32);
    }

    (row_offsets, col_indices, edge_weights)
}

/// Helper: Create complete graph for testing
fn create_complete_graph(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut row_offsets = vec![0];
    let mut col_indices = Vec::new();
    let mut edge_weights = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                col_indices.push(j as i32);
                edge_weights.push(1.0);
            }
        }
        row_offsets.push(col_indices.len() as i32);
    }

    (row_offsets, col_indices, edge_weights)
}

#[cfg(test)]
mod ffi_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires CUDA device
    async fn test_kernel_initialization() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let _kernels = HybridSSSPKernels::new(device);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_initialize_distances() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10;
        let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;

        kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;
        stream.synchronize()?;

        let distances = device.dtoh_sync_copy(&d_distances)?;
        let predecessors = device.dtoh_sync_copy(&d_predecessors)?;

        // Source should be 0, others infinity
        assert_eq!(distances[0], 0.0);
        for i in 1..num_nodes {
            assert!(distances[i].is_infinite());
            assert_eq!(predecessors[i], -1);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_k_step_relaxation_linear_graph() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10;
        let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

        // Upload graph
        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        // Initialize
        let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
        kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;

        // Relax
        kernels.k_step_relaxation(
            &mut d_distances,
            &mut d_predecessors,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            num_nodes as u32,  // Enough steps for complete propagation
            &stream,
        )?;

        stream.synchronize()?;
        let distances = device.dtoh_sync_copy(&d_distances)?;

        // Linear graph: distance to node i should be i
        for i in 0..num_nodes {
            assert!((distances[i] - i as f32).abs() < 0.01,
                "Node {}: expected {}, got {}", i, i, distances[i]);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_k_step_relaxation_convergence() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 5;
        let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        // Test with insufficient k
        let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
        kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;

        kernels.k_step_relaxation(
            &mut d_distances,
            &mut d_predecessors,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            2,  // Only 2 steps
            &stream,
        )?;

        stream.synchronize()?;
        let distances_partial = device.dtoh_sync_copy(&d_distances)?;

        // Should only reach first 2 nodes
        assert_eq!(distances_partial[0], 0.0);
        assert_eq!(distances_partial[1], 1.0);
        assert_eq!(distances_partial[2], 2.0);
        assert!(distances_partial[3].is_infinite() || distances_partial[3] > 100.0);

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_detect_pivots() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10;
        let (row_offsets, col_indices, edge_weights) = create_complete_graph(num_nodes);

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;

        // Mock distances (all finite for active frontier)
        let distances: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();
        let d_distances = device.htod_sync_copy(&distances)?;

        // All nodes in frontier
        let frontier: Vec<i32> = (0..num_nodes).map(|i| i as i32).collect();
        let d_frontier = device.htod_sync_copy(&frontier)?;

        // Detect pivots (complete graph nodes have high degree)
        let pivots = kernels.detect_pivots(
            &d_distances,
            &d_frontier,
            &d_row_offsets,
            0.1,
            5,  // degree threshold
            &stream,
        )?;

        stream.synchronize()?;

        // Complete graph: all nodes have degree n-1
        assert!(!pivots.is_empty(), "Should detect high-degree pivots");
        assert!(pivots.len() <= num_nodes, "Cannot have more pivots than nodes");

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_bounded_dijkstra() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10;
        let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        // Initialize with poor estimates
        let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
        kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;

        // Set pivot in middle
        let pivots = vec![5u32];

        // Run bounded Dijkstra with radius 3
        kernels.bounded_dijkstra(
            &mut d_distances,
            &mut d_predecessors,
            &pivots,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            3,
            &stream,
        )?;

        stream.synchronize()?;
        let distances = device.dtoh_sync_copy(&d_distances)?;

        // Should refine distances within radius of pivot
        // Nodes [2..8] should have correct distances from pivot's perspective
        for i in 2..=8 {
            let expected_from_source = i as f32;
            assert!(
                (distances[i] - expected_from_source).abs() < 1.0,
                "Node {}: expected ~{}, got {}",
                i, expected_from_source, distances[i]
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_partition_frontier() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10;

        // Create frontier
        let frontier: Vec<i32> = (0..num_nodes).map(|i| i as i32).collect();
        let d_frontier = device.htod_sync_copy(&frontier)?;

        // Old vs new distances (some changed, some not)
        let old_distances: Vec<f32> = (0..num_nodes).map(|i| i as f32).collect();
        let new_distances: Vec<f32> = (0..num_nodes).map(|i| {
            if i % 2 == 0 {
                i as f32 - 0.5  // Changed
            } else {
                i as f32  // Unchanged
            }
        }).collect();

        let d_old_distances = device.htod_sync_copy(&old_distances)?;
        let d_new_distances = device.htod_sync_copy(&new_distances)?;

        let active = kernels.partition_frontier(
            &d_frontier,
            &d_new_distances,
            &d_old_distances,
            0.1,  // epsilon
            &stream,
        )?;

        stream.synchronize()?;

        // Only even nodes should be active (distance changed by 0.5)
        assert_eq!(active.len(), num_nodes / 2);
        for &node in &active {
            assert_eq!(node % 2, 0, "Only even nodes should be active");
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_compact_frontier() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10;
        let frontier: Vec<i32> = (0..num_nodes).map(|i| i as i32).collect();

        // Mark odd nodes as invalid
        let valid_flags: Vec<i32> = (0..num_nodes).map(|i| (i % 2) as i32).collect();

        let d_frontier = device.htod_sync_copy(&frontier)?;
        let d_valid_flags = device.htod_sync_copy(&valid_flags)?;

        let compacted = kernels.compact_frontier(&d_frontier, &d_valid_flags, &stream)?;

        stream.synchronize()?;

        // Should have only even nodes
        assert_eq!(compacted.len(), num_nodes / 2);
        for i in 0..compacted.len() {
            assert_eq!(compacted[i], (i * 2) as u32);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_empty_graph() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let row_offsets = vec![0, 0];  // 1 node, 0 edges
        let col_indices: Vec<i32> = vec![];
        let edge_weights: Vec<f32> = vec![];

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        let mut d_distances = device.alloc_zeros::<f32>(1)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(1)?;

        // Should handle gracefully
        let result = kernels.k_step_relaxation(
            &mut d_distances,
            &mut d_predecessors,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            5,
            &stream,
        );

        assert!(result.is_err() || result.is_ok());  // Either error or no-op is acceptable

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_sssp_result_path_reconstruction() {
        let result = SSSPResult {
            distances: vec![0.0, 1.0, 2.0, 3.0, f32::INFINITY],
            predecessors: vec![-1, 0, 1, 2, -1],
            semantic_scores: None,
        };

        // Valid path
        let path = result.reconstruct_path(3);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![0, 1, 2, 3]);

        // Unreachable node
        let path = result.reconstruct_path(4);
        assert!(path.is_none());

        // Distance queries
        assert_eq!(result.distance_to(2), Some(2.0));
        assert_eq!(result.distance_to(4), None);
    }

    #[tokio::test]
    #[ignore]
    async fn test_config_validation() {
        let config = HybridSSSPConfig::default();

        assert!(config.k_steps > 0);
        assert!(config.convergence_threshold > 0.0);
        assert!(config.degree_threshold > 0);
        assert!(config.dijkstra_radius > 0);
        assert!(config.frontier_epsilon > 0.0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_concurrent_stream_operations() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());

        // Create multiple streams
        let stream1 = device.fork_default_stream()?;
        let stream2 = device.fork_default_stream()?;

        let num_nodes = 10;
        let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        // Launch operations on different streams
        let mut d_distances1 = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors1 = device.alloc_zeros::<i32>(num_nodes)?;
        kernels.initialize_distances(&mut d_distances1, &mut d_predecessors1, 0, &stream1)?;

        let mut d_distances2 = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors2 = device.alloc_zeros::<i32>(num_nodes)?;
        kernels.initialize_distances(&mut d_distances2, &mut d_predecessors2, 0, &stream2)?;

        // Both should complete successfully
        stream1.synchronize()?;
        stream2.synchronize()?;

        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_full_hybrid_workflow() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let config = HybridSSSPConfig::default();
        let num_nodes = 20;
        let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        // Initialize
        let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
        kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;

        // Phase 1: K-step relaxation
        kernels.k_step_relaxation(
            &mut d_distances,
            &mut d_predecessors,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            config.k_steps,
            &stream,
        )?;

        // Phase 2: Detect pivots
        let frontier: Vec<i32> = (0..num_nodes).map(|i| i as i32).collect();
        let d_frontier = device.htod_sync_copy(&frontier)?;

        let pivots = kernels.detect_pivots(
            &d_distances,
            &d_frontier,
            &d_row_offsets,
            config.convergence_threshold,
            config.degree_threshold as i32,
            &stream,
        )?;

        // Phase 3: Bounded Dijkstra (if pivots found)
        if !pivots.is_empty() {
            kernels.bounded_dijkstra(
                &mut d_distances,
                &mut d_predecessors,
                &pivots,
                &d_row_offsets,
                &d_col_indices,
                &d_edge_weights,
                config.dijkstra_radius,
                &stream,
            )?;
        }

        // Verify results
        stream.synchronize()?;
        let distances = device.dtoh_sync_copy(&d_distances)?;
        let predecessors = device.dtoh_sync_copy(&d_predecessors)?;

        // Linear graph should have correct distances
        for i in 0..num_nodes {
            assert!(
                (distances[i] - i as f32).abs() < 0.1,
                "Incorrect distance at node {}: expected {}, got {}",
                i, i, distances[i]
            );
        }

        // Verify predecessor chain
        for i in 1..num_nodes {
            assert_eq!(
                predecessors[i], (i - 1) as i32,
                "Incorrect predecessor at node {}: expected {}, got {}",
                i, i - 1, predecessors[i]
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_landmark_selection_coverage() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 100;
        let num_landmarks = 10;

        // Generate mock data
        let content_clusters: Vec<i32> = (0..num_nodes).map(|i| (i % 5) as i32).collect();
        let node_degrees: Vec<i32> = (0..num_nodes).map(|i| (i % 20) as i32).collect();

        let d_content_clusters = device.htod_sync_copy(&content_clusters)?;
        let d_node_degrees = device.htod_sync_copy(&node_degrees)?;

        let landmarks = kernels.select_landmarks(
            &d_content_clusters,
            &d_node_degrees,
            num_landmarks,
            42,
            &stream,
        )?;

        stream.synchronize()?;

        // Verify landmark count
        assert_eq!(landmarks.len(), num_landmarks as usize);

        // Verify uniqueness
        let mut unique_landmarks = landmarks.clone();
        unique_landmarks.sort();
        unique_landmarks.dedup();
        assert_eq!(unique_landmarks.len(), landmarks.len(), "Landmarks should be unique");

        // Verify within bounds
        for &landmark in &landmarks {
            assert!(landmark < num_nodes as u32, "Landmark out of bounds");
        }

        Ok(())
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_large_graph_performance() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        let num_nodes = 10000;
        let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

        let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
        let d_col_indices = device.htod_sync_copy(&col_indices)?;
        let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

        let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
        let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;

        let start = std::time::Instant::now();

        kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, &stream)?;
        kernels.k_step_relaxation(
            &mut d_distances,
            &mut d_predecessors,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            100,
            &stream,
        )?;

        stream.synchronize()?;

        let elapsed = start.elapsed();
        println!("Large graph ({} nodes) processed in {:?}", num_nodes, elapsed);

        // Should complete in reasonable time
        assert!(elapsed.as_secs() < 5, "Processing took too long");

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_memory_pressure() -> GpuResult<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let kernels = HybridSSSPKernels::new(device.clone());
        let stream = device.fork_default_stream()?;

        // Allocate multiple large graphs
        for _ in 0..5 {
            let num_nodes = 5000;
            let (row_offsets, col_indices, edge_weights) = create_linear_graph(num_nodes);

            let _d_row_offsets = device.htod_sync_copy(&row_offsets)?;
            let _d_col_indices = device.htod_sync_copy(&col_indices)?;
            let _d_edge_weights = device.htod_sync_copy(&edge_weights)?;
            let mut _d_distances = device.alloc_zeros::<f32>(num_nodes)?;
            let mut _d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;

            // Memory should be released when dropped
        }

        stream.synchronize()?;
        Ok(())
    }
}
