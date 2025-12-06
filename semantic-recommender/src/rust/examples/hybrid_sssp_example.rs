//! Example: Hybrid SSSP with CUDA FFI Bindings
//!
//! Demonstrates usage of the hybrid Single-Source Shortest Path (SSSP) algorithm
//! combining k-step relaxation with pivot-based bounded Dijkstra.
//!
//! Run with: cargo run --example hybrid_sssp_example --features cuda

use cudarc::driver::{CudaDevice, CudaStream};
use gpu_engine::{HybridSSSPKernels, HybridSSSPConfig, GpuResult};
use std::sync::Arc;

/// Build a sample CSR graph for testing
fn build_sample_graph() -> (Vec<i32>, Vec<i32>, Vec<f32>, usize) {
    // Simple graph with 10 nodes:
    //   0 -> 1 (1.0)
    //   0 -> 2 (4.0)
    //   1 -> 2 (2.0)
    //   1 -> 3 (5.0)
    //   2 -> 3 (1.0)
    //   3 -> 4 (3.0)
    //   4 -> 5 (2.0)
    //   5 -> 6 (1.0)
    //   6 -> 7 (2.0)
    //   7 -> 8 (1.0)
    //   8 -> 9 (3.0)

    let num_nodes = 10;

    // CSR format: row_offsets, col_indices, edge_weights
    let row_offsets = vec![
        0,  // node 0: edges [0..2)
        2,  // node 1: edges [2..4)
        4,  // node 2: edges [4..5)
        5,  // node 3: edges [5..6)
        6,  // node 4: edges [6..7)
        7,  // node 5: edges [7..8)
        8,  // node 6: edges [8..9)
        9,  // node 7: edges [9..10)
        10, // node 8: edges [10..11)
        11, // node 9: edges [11..11)
        11, // end marker
    ];

    let col_indices = vec![1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9];
    let edge_weights = vec![1.0, 4.0, 2.0, 5.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0, 3.0];

    (row_offsets, col_indices, edge_weights, num_nodes)
}

/// Example 1: Basic k-step relaxation
async fn example_k_step_relaxation(
    kernels: &HybridSSSPKernels,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> GpuResult<()> {
    println!("\n=== Example 1: K-Step Relaxation ===");

    let (row_offsets, col_indices, edge_weights, num_nodes) = build_sample_graph();

    // Upload graph to GPU
    let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
    let d_col_indices = device.htod_sync_copy(&col_indices)?;
    let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

    // Allocate distance and predecessor arrays
    let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
    let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;

    // Initialize distances (source = 0)
    kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, stream)?;

    // Perform k-step relaxation
    let k_steps = 5;
    kernels.k_step_relaxation(
        &mut d_distances,
        &mut d_predecessors,
        &d_row_offsets,
        &d_col_indices,
        &d_edge_weights,
        k_steps,
        stream,
    )?;

    // Synchronize and retrieve results
    stream.synchronize()?;
    let distances = device.dtoh_sync_copy(&d_distances)?;
    let predecessors = device.dtoh_sync_copy(&d_predecessors)?;

    println!("After {} k-step relaxation rounds:", k_steps);
    for (i, (&dist, &pred)) in distances.iter().zip(predecessors.iter()).enumerate() {
        println!("  Node {}: distance = {:.2}, predecessor = {}", i, dist, pred);
    }

    Ok(())
}

/// Example 2: Pivot detection and bounded Dijkstra
async fn example_pivot_detection(
    kernels: &HybridSSSPKernels,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> GpuResult<()> {
    println!("\n=== Example 2: Pivot Detection ===");

    let (row_offsets, col_indices, edge_weights, num_nodes) = build_sample_graph();

    // Upload graph
    let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
    let d_col_indices = device.htod_sync_copy(&col_indices)?;
    let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

    // Initialize distances
    let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
    let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
    kernels.initialize_distances(&mut d_distances, &mut d_predecessors, 0, stream)?;

    // Run initial k-step relaxation
    kernels.k_step_relaxation(
        &mut d_distances,
        &mut d_predecessors,
        &d_row_offsets,
        &d_col_indices,
        &d_edge_weights,
        3,
        stream,
    )?;

    // Create frontier (all nodes except source)
    let frontier: Vec<i32> = (1..num_nodes as i32).collect();
    let d_frontier = device.htod_sync_copy(&frontier)?;

    // Detect pivots (high-degree nodes or poor convergence)
    let pivots = kernels.detect_pivots(
        &d_distances,
        &d_frontier,
        &d_row_offsets,
        0.1,  // convergence_threshold
        2,    // degree_threshold
        stream,
    )?;

    println!("Detected {} pivot nodes: {:?}", pivots.len(), pivots);

    if !pivots.is_empty() {
        // Run bounded Dijkstra around pivots
        println!("Running bounded Dijkstra with radius=2 around pivots...");
        kernels.bounded_dijkstra(
            &mut d_distances,
            &mut d_predecessors,
            &pivots,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            2,  // radius
            stream,
        )?;

        stream.synchronize()?;
        let refined_distances = device.dtoh_sync_copy(&d_distances)?;

        println!("Refined distances after bounded Dijkstra:");
        for (i, &dist) in refined_distances.iter().enumerate() {
            println!("  Node {}: distance = {:.2}", i, dist);
        }
    }

    Ok(())
}

/// Example 3: Complete hybrid SSSP workflow
async fn example_hybrid_workflow(
    kernels: &HybridSSSPKernels,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> GpuResult<()> {
    println!("\n=== Example 3: Complete Hybrid SSSP Workflow ===");

    let config = HybridSSSPConfig {
        k_steps: 5,
        convergence_threshold: 0.01,
        degree_threshold: 2,
        dijkstra_radius: 3,
        frontier_epsilon: 1e-6,
        ..Default::default()
    };

    let (row_offsets, col_indices, edge_weights, num_nodes) = build_sample_graph();

    // Upload graph
    let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
    let d_col_indices = device.htod_sync_copy(&col_indices)?;
    let d_edge_weights = device.htod_sync_copy(&edge_weights)?;

    // Initialize
    let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
    let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
    let source = 0u32;
    kernels.initialize_distances(&mut d_distances, &mut d_predecessors, source, stream)?;

    println!("Configuration: {:?}", config);
    println!("Running hybrid SSSP from source {}...\n", source);

    // Phase 1: K-step relaxation
    println!("Phase 1: K-step relaxation ({} steps)", config.k_steps);
    kernels.k_step_relaxation(
        &mut d_distances,
        &mut d_predecessors,
        &d_row_offsets,
        &d_col_indices,
        &d_edge_weights,
        config.k_steps,
        stream,
    )?;

    // Phase 2: Detect pivots
    println!("Phase 2: Detecting pivots");
    let frontier: Vec<i32> = (0..num_nodes as i32).collect();
    let d_frontier = device.htod_sync_copy(&frontier)?;

    let pivots = kernels.detect_pivots(
        &d_distances,
        &d_frontier,
        &d_row_offsets,
        config.convergence_threshold,
        config.degree_threshold as i32,
        stream,
    )?;

    println!("  Found {} pivots: {:?}", pivots.len(), pivots);

    // Phase 3: Bounded Dijkstra around pivots
    if !pivots.is_empty() {
        println!("Phase 3: Bounded Dijkstra (radius={})", config.dijkstra_radius);
        kernels.bounded_dijkstra(
            &mut d_distances,
            &mut d_predecessors,
            &pivots,
            &d_row_offsets,
            &d_col_indices,
            &d_edge_weights,
            config.dijkstra_radius,
            stream,
        )?;
    }

    // Retrieve final results
    stream.synchronize()?;
    let distances = device.dtoh_sync_copy(&d_distances)?;
    let predecessors = device.dtoh_sync_copy(&d_predecessors)?;

    println!("\nFinal shortest paths from source {}:", source);
    for target in 0..num_nodes {
        let dist = distances[target];
        if dist.is_finite() {
            // Reconstruct path
            let mut path = vec![target as u32];
            let mut current = target;
            while predecessors[current] != -1 {
                current = predecessors[current] as usize;
                path.push(current as u32);
            }
            path.reverse();

            println!(
                "  To node {}: distance = {:.2}, path = {:?}",
                target, dist, path
            );
        } else {
            println!("  To node {}: unreachable", target);
        }
    }

    Ok(())
}

/// Example 4: Frontier partitioning and compaction
async fn example_frontier_operations(
    kernels: &HybridSSSPKernels,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> GpuResult<()> {
    println!("\n=== Example 4: Frontier Operations ===");

    let num_nodes = 10;

    // Simulate frontier with some nodes
    let frontier = vec![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let d_frontier = device.htod_sync_copy(&frontier)?;

    // Simulate distances (some changed, some not)
    let old_distances = vec![0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
    let new_distances = vec![0.0, 1.0, 2.5, 6.0, 8.0, 12.0, 14.0, 18.0, 20.0, 24.0];

    let d_old_distances = device.htod_sync_copy(&old_distances)?;
    let d_new_distances = device.htod_sync_copy(&new_distances)?;

    // Partition frontier (identify nodes with significant distance changes)
    let active_frontier = kernels.partition_frontier(
        &d_frontier,
        &d_new_distances,
        &d_old_distances,
        0.1,  // epsilon
        stream,
    )?;

    println!("Original frontier: {:?}", frontier);
    println!("Active nodes (distance changed > 0.1): {:?}", active_frontier);

    // Test frontier compaction
    let valid_flags = vec![1i32, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let d_valid_flags = device.htod_sync_copy(&valid_flags)?;

    let compacted = kernels.compact_frontier(&d_frontier, &d_valid_flags, stream)?;

    println!("\nFrontier compaction:");
    println!("  Original: {:?}", frontier);
    println!("  Valid flags: {:?}", valid_flags);
    println!("  Compacted: {:?}", compacted);

    Ok(())
}

/// Example 5: Semantic SSSP for content discovery
async fn example_semantic_sssp(
    kernels: &HybridSSSPKernels,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> GpuResult<()> {
    println!("\n=== Example 5: Semantic SSSP for Content Discovery ===");

    let (row_offsets, col_indices, edge_weights, num_nodes) = build_sample_graph();
    let num_edges = col_indices.len();

    // Generate mock semantic features
    let content_features: Vec<f32> = (0..num_edges).map(|i| 0.8 - (i as f32) * 0.02).collect();
    let user_affinities: Vec<f32> = (0..num_nodes).map(|i| 0.9 - (i as f32) * 0.05).collect();

    // Upload to GPU
    let d_row_offsets = device.htod_sync_copy(&row_offsets)?;
    let d_col_indices = device.htod_sync_copy(&col_indices)?;
    let d_edge_weights = device.htod_sync_copy(&edge_weights)?;
    let d_content_features = device.htod_sync_copy(&content_features)?;
    let d_user_affinities = device.htod_sync_copy(&user_affinities)?;

    // Allocate outputs
    let mut d_distances = device.alloc_zeros::<f32>(num_nodes)?;
    let mut d_predecessors = device.alloc_zeros::<i32>(num_nodes)?;
    let mut d_semantic_scores = device.alloc_zeros::<f32>(num_nodes)?;
    let mut d_next_frontier = device.alloc_zeros::<i32>(num_nodes)?;
    let mut d_next_frontier_size = device.alloc_zeros::<i32>(1)?;

    // Initialize
    let source = 0u32;
    kernels.initialize_distances(&mut d_distances, &mut d_predecessors, source, stream)?;

    // Create initial frontier (just source)
    let frontier = vec![source as i32];
    let d_frontier = device.htod_sync_copy(&frontier)?;

    println!("Running semantic SSSP from source {}...", source);

    // Launch semantic kernel
    kernels.sssp_semantic(
        source,
        &mut d_distances,
        &mut d_predecessors,
        &mut d_semantic_scores,
        &d_row_offsets,
        &d_col_indices,
        &d_edge_weights,
        &d_content_features,
        &d_user_affinities,
        &d_frontier,
        &mut d_next_frontier,
        &mut d_next_frontier_size,
        10,   // max_hops
        0.5,  // min_similarity
        stream,
    )?;

    // Retrieve results
    stream.synchronize()?;
    let distances = device.dtoh_sync_copy(&d_distances)?;
    let semantic_scores = device.dtoh_sync_copy(&d_semantic_scores)?;
    let next_frontier_size = device.dtoh_sync_copy(&d_next_frontier_size)?[0];

    println!("\nSemantic SSSP results:");
    for i in 0..num_nodes {
        println!(
            "  Node {}: distance = {:.2}, semantic_score = {:.3}",
            i, distances[i], semantic_scores[i]
        );
    }
    println!("Next frontier size: {}", next_frontier_size);

    Ok(())
}

/// Example 6: Landmark-based APSP
async fn example_landmark_apsp(
    kernels: &HybridSSSPKernels,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> GpuResult<()> {
    println!("\n=== Example 6: Landmark-based APSP ===");

    let num_nodes = 10;
    let num_landmarks = 3;

    // Mock data
    let content_clusters: Vec<i32> = (0..num_nodes).map(|i| i as i32 % 3).collect();
    let node_degrees: Vec<i32> = vec![2, 2, 1, 1, 1, 1, 1, 1, 1, 0];

    let d_content_clusters = device.htod_sync_copy(&content_clusters)?;
    let d_node_degrees = device.htod_sync_copy(&node_degrees)?;

    // Select landmarks
    let landmarks = kernels.select_landmarks(
        &d_content_clusters,
        &d_node_degrees,
        num_landmarks,
        12345,  // seed
        stream,
    )?;

    println!("Selected {} landmarks: {:?}", landmarks.len(), landmarks);

    // Mock landmark distances (would be computed by running SSSP from each landmark)
    let landmark_distances: Vec<f32> = (0..num_landmarks * num_nodes)
        .map(|i| ((i / num_nodes) + (i % num_nodes)) as f32)
        .collect();

    let d_landmark_distances = device.htod_sync_copy(&landmark_distances)?;

    // Allocate output matrices
    let mut d_distance_matrix = device.alloc_zeros::<f32>(num_nodes * num_nodes)?;
    let mut d_quality_scores = device.alloc_zeros::<f32>(num_nodes * num_nodes)?;

    // Compute approximate APSP
    println!("Computing approximate APSP using {} landmarks...", num_landmarks);
    kernels.approximate_apsp(
        &d_landmark_distances,
        &mut d_distance_matrix,
        &mut d_quality_scores,
        num_nodes as u32,
        num_landmarks,
        stream,
    )?;

    // Retrieve results
    stream.synchronize()?;
    let distance_matrix = device.dtoh_sync_copy(&d_distance_matrix)?;
    let quality_scores = device.dtoh_sync_copy(&d_quality_scores)?;

    println!("\nApproximate all-pairs distances:");
    for i in 0..num_nodes.min(5) {
        print!("  From {}: ", i);
        for j in 0..num_nodes {
            let idx = i * num_nodes + j;
            print!("{:.1} ", distance_matrix[idx]);
        }
        println!();
    }

    println!("\nQuality scores (first 5x5 block):");
    for i in 0..num_nodes.min(5) {
        print!("  Row {}: ", i);
        for j in 0..num_nodes.min(5) {
            let idx = i * num_nodes + j;
            print!("{:.2} ", quality_scores[idx]);
        }
        println!();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> GpuResult<()> {
    println!("Hybrid SSSP CUDA FFI Examples");
    println!("==============================\n");

    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("Using device: {:?}\n", device.ordinal());

    // Create stream
    let stream = device.fork_default_stream()?;

    // Create kernel wrapper
    let kernels = HybridSSSPKernels::new(device.clone());

    // Run examples
    if let Err(e) = example_k_step_relaxation(&kernels, &device, &stream).await {
        eprintln!("Example 1 failed: {}", e);
    }

    if let Err(e) = example_pivot_detection(&kernels, &device, &stream).await {
        eprintln!("Example 2 failed: {}", e);
    }

    if let Err(e) = example_hybrid_workflow(&kernels, &device, &stream).await {
        eprintln!("Example 3 failed: {}", e);
    }

    if let Err(e) = example_frontier_operations(&kernels, &device, &stream).await {
        eprintln!("Example 4 failed: {}", e);
    }

    if let Err(e) = example_semantic_sssp(&kernels, &device, &stream).await {
        eprintln!("Example 5 failed: {}", e);
    }

    if let Err(e) = example_landmark_apsp(&kernels, &device, &stream).await {
        eprintln!("Example 6 failed: {}", e);
    }

    println!("\n=============================");
    println!("All examples completed!");

    Ok(())
}
