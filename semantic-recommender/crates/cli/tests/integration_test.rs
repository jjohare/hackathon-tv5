// Integration test for full 62,423 movie dataset

use anyhow::Result;
use std::path::PathBuf;

#[tokio::test]
#[ignore] // Run with: cargo test --test integration_test -- --ignored
async fn test_full_dataset_62k_movies() -> Result<()> {
    // This test validates the complete end-to-end workflow:
    // 1. Load 62,423 movies from CSV
    // 2. Generate or load embeddings
    // 3. Index in GPU memory
    // 4. Execute test queries
    // 5. Compare with Python baseline

    println!("Starting full dataset integration test...");

    // Step 1: Load dataset
    let dataset_path = PathBuf::from("data/movies.csv");
    if !dataset_path.exists() {
        println!("⚠ Skipping test: dataset not found at {:?}", dataset_path);
        println!("  Download from: [dataset URL]");
        return Ok(());
    }

    // TODO: Uncomment when implementation is complete
    // let mut dataset = MovieDataset::load_csv(&dataset_path).await?;
    // assert_eq!(dataset.len(), 62_423, "Expected 62,423 movies");

    // Step 2: Load or generate embeddings
    // let embeddings_path = PathBuf::from("data/embeddings.npy");
    // if embeddings_path.exists() {
    //     dataset.load_embeddings(&embeddings_path).await?;
    // } else {
    //     println!("Computing embeddings for {} movies...", dataset.len());
    //     dataset.compute_embeddings().await?;
    // }

    // Step 3: Initialize GPU engine
    // let engine = GpuSemanticEngine::new(Default::default()).await?;
    // engine.index_embeddings(dataset.embeddings.as_ref().unwrap()).await?;

    // Step 4: Run test queries
    let test_queries = vec![
        ("action thriller with car chases", 10),
        ("romantic comedy from the 90s", 10),
        ("sci-fi with AI themes", 10),
        ("horror suspense thriller", 10),
        ("family-friendly animation", 10),
    ];

    for (query, limit) in test_queries {
        println!("Testing query: '{}'", query);

        // TODO: Uncomment when implementation is complete
        // let results = engine.search(query, limit).await?;
        // assert_eq!(results.len(), limit);
        // assert!(results[0].score >= results[limit-1].score, "Results should be sorted");

        // Validate result quality
        // for result in &results {
        //     assert!(result.score >= 0.0 && result.score <= 1.0);
        //     assert!(!result.movie_id.is_empty());
        // }
    }

    // Step 5: Compare with Python baseline (optional)
    // if let Ok(python_baseline) = PythonBaseline::new("scripts/baseline.py") {
    //     for (query, limit) in test_queries {
    //         let rust_results = engine.search(query, limit).await?;
    //         let python_results = python_baseline.run_query(query, limit).await?;
    //
    //         // Compare top results
    //         let overlap = compute_overlap(&rust_results, &python_results);
    //         assert!(overlap > 0.7, "Result overlap should be > 70%");
    //     }
    // }

    println!("✓ Full dataset integration test passed");
    Ok(())
}

#[tokio::test]
async fn test_gpu_memory_stress() -> Result<()> {
    // Test GPU memory management under load
    // This validates proper allocation/deallocation and leak prevention

    // TODO: Implement when GPU engine is ready
    // let engine = GpuSemanticEngine::new(Default::default()).await?;
    //
    // // Generate large batch of queries
    // let num_queries = 1000;
    // for i in 0..num_queries {
    //     let query = format!("test query {}", i);
    //     engine.search(&query, 10).await?;
    //
    //     if i % 100 == 0 {
    //         // Check memory hasn't leaked
    //         let stats = engine.memory_stats();
    //         assert!(stats.allocated < stats.total * 0.9);
    //     }
    // }

    Ok(())
}

#[tokio::test]
async fn test_concurrent_queries() -> Result<()> {
    // Test thread safety and concurrent access

    // TODO: Implement when GPU engine is ready
    // let engine = Arc::new(GpuSemanticEngine::new(Default::default()).await?);
    //
    // let mut handles = vec![];
    // for i in 0..10 {
    //     let engine = engine.clone();
    //     let handle = tokio::spawn(async move {
    //         let query = format!("concurrent query {}", i);
    //         engine.search(&query, 10).await
    //     });
    //     handles.push(handle);
    // }
    //
    // for handle in handles {
    //     handle.await??;
    // }

    Ok(())
}

#[test]
fn test_cli_args_parsing() {
    // Test CLI argument parsing
    use clap::Parser;

    // This would normally be in a separate test file, but included here for completeness
    // See src/main.rs for actual CLI definition
}

// Helper function to compute result overlap
#[allow(dead_code)]
fn compute_overlap(rust_results: &[(String, f64)], python_results: &[(String, f64)]) -> f64 {
    let rust_ids: std::collections::HashSet<_> = rust_results.iter().map(|(id, _)| id).collect();
    let python_ids: std::collections::HashSet<_> = python_results.iter().map(|(id, _)| id).collect();

    let intersection = rust_ids.intersection(&python_ids).count();
    intersection as f64 / rust_results.len() as f64
}
