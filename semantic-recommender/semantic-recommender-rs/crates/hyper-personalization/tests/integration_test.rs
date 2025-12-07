//! Integration tests for hyper-personalization system
//!
//! These tests verify end-to-end functionality of the complete pipeline.

use hyper_personalization::{HyperPersonalizationSystem, SystemConfig};
use std::path::PathBuf;

fn get_test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_data")
}

#[test]
#[ignore] // Requires test data files
fn test_system_initialization() {
    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        max_query_length: 128,
        cache_size: 100,
        cache_ttl_secs: 60,
        use_gpu: false, // Use CPU for CI
    };

    let result = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    );

    assert!(result.is_ok(), "System initialization failed: {:?}", result.err());
}

#[test]
#[ignore] // Requires test data files
fn test_personalized_search_no_context() {
    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        use_gpu: false,
        ..Default::default()
    };

    let mut system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    let result = system
        .personalized_search("user123", "action movie", 10, None)
        .expect("Search failed");

    assert!(
        result.items.len() <= 10,
        "Expected at most 10 results, got {}",
        result.items.len()
    );

    assert!(
        result.timing.total_ms > 0.0,
        "Total time should be positive"
    );

    println!("Search timing breakdown:");
    println!("  Query encoding: {:.3}ms", result.timing.query_encoding_ms);
    println!("  User fusion: {:.3}ms", result.timing.user_fusion_ms);
    println!("  Similarity: {:.3}ms", result.timing.similarity_ms);
    println!("  Attention rerank: {:.3}ms", result.timing.attention_rerank_ms);
    println!("  Top-K: {:.3}ms", result.timing.topk_ms);
    println!("  TOTAL: {:.3}ms", result.timing.total_ms);
}

#[test]
#[ignore] // Requires test data files
fn test_personalized_search_with_context() {
    use attention::ContextFeatures;

    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        use_gpu: false,
        ..Default::default()
    };

    let mut system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    let context = ContextFeatures::new(
        [1.0, 0.0, 0.0], // Morning
        [0.0, 1.0, 0.0], // Drama
        [0.5, 0.5],       // Mixed social
    );

    let result = system
        .personalized_search("user123", "romantic comedy", 10, Some(&context))
        .expect("Search with context failed");

    assert!(result.items.len() <= 10);
    assert!(result.timing.total_ms > 0.0);
}

#[test]
#[ignore] // Requires test data files
fn test_cache_hit() {
    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        cache_size: 100,
        use_gpu: false,
        ..Default::default()
    };

    let mut system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    // First query (cache miss)
    let result1 = system
        .personalized_search("user123", "action movie", 10, None)
        .expect("First search failed");
    assert!(!result1.from_cache, "First query should be cache miss");

    // Same query again (should be cache hit)
    let result2 = system
        .personalized_search("user123", "action movie", 10, None)
        .expect("Second search failed");
    assert!(result2.from_cache, "Second query should be cache hit");

    // Cache hit should be faster
    assert!(
        result2.timing.total_ms < result1.timing.total_ms,
        "Cache hit should be faster: {} vs {}",
        result2.timing.total_ms,
        result1.timing.total_ms
    );
}

#[test]
#[ignore] // Requires test data files
fn test_user_preference_update() {
    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        use_gpu: false,
        ..Default::default()
    };

    let mut system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    // Get initial results
    let result1 = system
        .personalized_search("user_new", "thriller movie", 10, None)
        .expect("Initial search failed");

    // Update user preferences (user liked item 42)
    system
        .update_user_preferences("user_new", 42, 0.9)
        .expect("Preference update failed");

    // Search again - results may be different now
    let result2 = system
        .personalized_search("user_new", "thriller movie", 10, None)
        .expect("Search after update failed");

    assert!(result1.items.len() > 0);
    assert!(result2.items.len() > 0);

    // User stats should show embedding exists now
    assert!(result2.user_stats.is_some());
    if let Some(stats) = result2.user_stats {
        assert!(stats.has_embedding, "User should have embedding after update");
    }
}

#[test]
#[ignore] // Requires test data files
fn test_performance_metrics() {
    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        use_gpu: false,
        ..Default::default()
    };

    let mut system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    // Run multiple queries
    for i in 0..10 {
        let query = format!("test query {}", i);
        system
            .personalized_search("user123", &query, 5, None)
            .expect("Search failed");
    }

    let metrics = system.metrics();

    assert_eq!(metrics.total_queries, 10);
    assert!(metrics.avg_latency_ms > 0.0);

    println!("Performance metrics after 10 queries:");
    println!("  Total queries: {}", metrics.total_queries);
    println!("  Cache hits: {}", metrics.cache_hits);
    println!("  Cache misses: {}", metrics.cache_misses);
    println!("  Hit rate: {:.2}%", metrics.cache_hit_rate() * 100.0);
    println!("  Avg latency: {:.3}ms", metrics.avg_latency_ms);
    println!("  P95 latency: {:.3}ms", metrics.p95_latency_ms);
    println!("  P99 latency: {:.3}ms", metrics.p99_latency_ms);
}

#[test]
#[ignore] // Requires test data files
fn test_concurrent_searches() {
    use std::thread;

    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        use_gpu: false,
        ..Default::default()
    };

    let system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    // Note: This test would require Arc<Mutex<System>> or similar for true concurrency
    // For now, we test sequential access which is still valid

    let queries = vec!["action", "comedy", "drama", "thriller"];

    for query in queries {
        let _result = system
            .personalized_search("user123", query, 5, None)
            .expect("Concurrent search failed");
    }
}

#[test]
#[ignore] // Requires test data files
fn test_latency_target() {
    let data_dir = get_test_data_dir();

    let config = SystemConfig {
        embedding_dim: 384,
        use_gpu: true, // Use GPU for performance testing
        ..Default::default()
    };

    let mut system = HyperPersonalizationSystem::new(
        data_dir.join("models/semantic_model.onnx"),
        data_dir.join("models/tokenizer.json"),
        data_dir.join("embeddings/item_embeddings.bin"),
        data_dir.join("embeddings/user_embeddings.bin"),
        config,
    )
    .expect("Failed to initialize system");

    // Warm up
    for _ in 0..5 {
        system
            .personalized_search("user123", "warmup query", 10, None)
            .ok();
    }

    // Measure latency
    let result = system
        .personalized_search("user123", "action movie", 10, None)
        .expect("Performance test failed");

    println!("End-to-end latency: {:.3}ms", result.timing.total_ms);

    // Target: <10ms for GPU
    if cfg!(feature = "cuda") {
        assert!(
            result.timing.total_ms < 10.0,
            "Expected <10ms latency, got {:.3}ms",
            result.timing.total_ms
        );
    }
}
