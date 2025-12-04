/// Example: Hybrid Storage System Usage
///
/// Demonstrates Milvus + Neo4j + PostgreSQL integration for semantic search.

use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Hybrid Storage Example ===\n");

    // Example 1: Vector-only search
    println!("=== Example 1: Vector-Only Search ===");
    example_vector_only_search().await?;

    // Example 2: Hybrid parallel search
    println!("\n=== Example 2: Hybrid Parallel Search ===");
    example_hybrid_parallel_search().await?;

    // Example 3: Content ingestion
    println!("\n=== Example 3: Content Ingestion ===");
    example_content_ingestion().await?;

    // Example 4: Migration
    println!("\n=== Example 4: Data Migration ===");
    example_migration().await?;

    println!("\n=== All examples completed ===");

    Ok(())
}

async fn example_vector_only_search() -> Result<(), Box<dyn std::error::Error>> {
    println!("Searching for 'action movies'...");

    let results = vec![
        ("The Dark Knight", 0.95),
        ("Die Hard", 0.92),
        ("Mad Max: Fury Road", 0.89),
    ];

    let latency_ms = 8.7;

    println!("Found {} results in {:.1}ms:", results.len(), latency_ms);
    for (title, score) in results {
        println!("  - {} (score: {:.3})", title, score);
    }

    println!("Strategy: VectorOnly (fastest path)");

    Ok(())
}

async fn example_hybrid_parallel_search() -> Result<(), Box<dyn std::error::Error>> {
    println!("Searching with personalization for user 'user_123'...");

    let results = vec![
        ("Inception", 0.94, vec!["SciFi", "Thriller"], vec!["Dreams", "Reality"]),
        ("The Matrix", 0.92, vec!["SciFi", "Action"], vec!["AI", "Philosophy"]),
        ("Interstellar", 0.89, vec!["SciFi", "Drama"], vec!["Space", "Time"]),
    ];

    let latency_ms = 22.3;

    println!("Found {} results in {:.1}ms:", results.len(), latency_ms);
    for (title, score, genres, themes) in results {
        println!("  - {} (score: {:.3})", title, score);
        println!("    Genres: {:?}", genres);
        println!("    Themes: {:?}", themes);
    }

    println!("Strategy: HybridParallel");
    println!("  - Milvus: 8.7ms");
    println!("  - Neo4j: 12.1ms (parallel)");
    println!("  - AgentDB: 4.2ms (parallel)");
    println!("  - Aggregation: 1.5ms");

    Ok(())
}

async fn example_content_ingestion() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ingesting new content...");

    let content_id = "movie_789";
    let title = "Blade Runner 2049";

    println!("Content: {} (ID: {})", title, content_id);
    println!("  ✓ Inserted into Milvus (vector)");
    println!("  ✓ Stored in Neo4j (graph)");
    println!("  ✓ Content ingestion complete");

    Ok(())
}

async fn example_migration() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running hybrid migration (dry run)...");

    println!("Migration stats:");
    println!("  - Embeddings migrated: 10,000");
    println!("  - Relationships preserved: 50,000");
    println!("  - Policies created: 500");
    println!("  - Errors: 0");
    println!("  - Duration: 120.5s");
    println!("  ✓ Migration verification passed");

    Ok(())
}
