use temporal_cache::{TemporalGPUCache, CacheError, CacheStats};
use std::time::{Duration, Instant};

fn create_test_embeddings(num_items: usize, embed_dim: usize) -> Vec<f32> {
    let mut embeddings = vec![0.0f32; num_items * embed_dim];
    for i in 0..num_items {
        for j in 0..embed_dim {
            // Create normalized embeddings with some structure
            let val = ((i + j) as f32 / 100.0).sin();
            embeddings[i * embed_dim + j] = val;
        }
    }

    // L2 normalize each embedding
    for i in 0..num_items {
        let mut norm = 0.0f32;
        for j in 0..embed_dim {
            let val = embeddings[i * embed_dim + j];
            norm += val * val;
        }
        norm = norm.sqrt();

        if norm > 1e-6 {
            for j in 0..embed_dim {
                embeddings[i * embed_dim + j] /= norm;
            }
        }
    }

    embeddings
}

#[test]
fn test_initialization_various_sizes() {
    let test_cases = vec![
        (100, 64, 10),
        (1000, 128, 100),
        (5000, 256, 500),
    ];

    for (num_items, embed_dim, num_popular) in test_cases {
        let embeddings = create_test_embeddings(num_items, embed_dim);

        let cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        );

        assert!(cache.is_ok(), "Failed for size: {}x{}, popular: {}",
                num_items, embed_dim, num_popular);
    }
}

#[test]
fn test_invalid_dimensions() {
    let num_items = 100;
    let embed_dim = 64;

    // Wrong size embeddings
    let embeddings = vec![0.0f32; 50]; // Too small

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(10),
        Some(0.1),
    );

    assert!(cache.is_err());
}

#[test]
fn test_popular_indices_exceed_items() {
    let num_items = 100;
    let embed_dim = 64;
    let num_popular = 200; // More than num_items

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    );

    assert!(cache.is_err());
}

#[test]
fn test_cache_hit_rate() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Query only popular items (100% hit rate)
    for i in 0..num_popular {
        let result = cache.get_similar_items(i).unwrap();
        assert!(result.from_cache, "Expected cache hit for item {}", i);
    }

    let stats = cache.cache_stats();
    assert_eq!(stats.total_hits, num_popular);
    assert_eq!(stats.total_misses, 0);
    assert_eq!(stats.hit_rate, 1.0);

    // Query non-popular items (0% hit rate for new queries)
    for i in num_popular..num_popular + 50 {
        let result = cache.get_similar_items(i).unwrap();
        assert!(!result.from_cache, "Expected cache miss for item {}", i);
    }

    let stats = cache.cache_stats();
    assert_eq!(stats.total_hits, num_popular);
    assert_eq!(stats.total_misses, 50);
    assert!((stats.hit_rate - 0.666).abs() < 0.01); // ~66.7%
}

#[test]
fn test_similarity_consistency() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Query same item multiple times
    let item_id = 50;
    let result1 = cache.get_similar_items(item_id).unwrap();
    let result2 = cache.get_similar_items(item_id).unwrap();

    // Results should be identical
    assert_eq!(result1.similarities.len(), result2.similarities.len());

    for (s1, s2) in result1.similarities.iter().zip(result2.similarities.iter()) {
        assert!((s1 - s2).abs() < 1e-5, "Similarity inconsistency: {} vs {}", s1, s2);
    }
}

#[test]
fn test_similarity_bounds() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    let result = cache.get_similar_items(50).unwrap();

    // All similarities should be in valid range [-1, 1] for normalized embeddings
    for (idx, &sim) in result.similarities.iter().enumerate() {
        assert!(sim >= -1.0 && sim <= 1.0,
                "Similarity out of bounds at {}: {}", idx, sim);
    }

    // Self-similarity should be ~1.0
    let self_sim = result.similarities[50];
    assert!((self_sim - 1.0).abs() < 0.1, "Self-similarity should be ~1.0, got {}", self_sim);
}

#[test]
fn test_rebuild_cache() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let mut cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Get initial similarities
    let result1 = cache.get_similar_items(50).unwrap();

    // Rebuild cache
    cache.rebuild_cache().unwrap();

    // Get similarities after rebuild
    let result2 = cache.get_similar_items(50).unwrap();

    // Results should be identical
    for (s1, s2) in result1.similarities.iter().zip(result2.similarities.iter()) {
        assert!((s1 - s2).abs() < 1e-5);
    }

    // Stats should be reset
    let stats = cache.cache_stats();
    assert_eq!(stats.cache_age_secs, 0.0);
}

#[test]
fn test_temporal_decay() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;
    let decay_rate = 0.5;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let mut cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(decay_rate),
    ).unwrap();

    // Initial weights should be 1.0
    let stats1 = cache.cache_stats();
    assert_eq!(stats1.cache_age_secs, 0.0);

    // Simulate time passage
    std::thread::sleep(Duration::from_millis(200));

    // Update weights
    cache.update_temporal_weights().unwrap();

    let stats2 = cache.cache_stats();
    assert!(stats2.cache_age_secs > 0.0);
    assert!(stats2.cache_age_secs < 1.0);
}

#[test]
fn test_latency_measurement() {
    let num_items = 10_000;
    let embed_dim = 256;
    let num_popular = 1000;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Warm up
    for i in 0..10 {
        let _ = cache.get_similar_items(i);
    }

    // Measure cache hit latency
    let mut hit_latencies = Vec::new();
    for i in 0..100 {
        let result = cache.get_similar_items(i % num_popular).unwrap();
        hit_latencies.push(result.latency_ms);
    }

    let avg_hit_latency = hit_latencies.iter().sum::<f64>() / hit_latencies.len() as f64;
    println!("Average cache hit latency: {:.3}ms", avg_hit_latency);

    // Measure cache miss latency
    let mut miss_latencies = Vec::new();
    for i in 0..20 {
        let result = cache.get_similar_items(num_popular + i * 100).unwrap();
        miss_latencies.push(result.latency_ms);
    }

    let avg_miss_latency = miss_latencies.iter().sum::<f64>() / miss_latencies.len() as f64;
    println!("Average cache miss latency: {:.3}ms", avg_miss_latency);

    // Cache hits should be faster than misses
    assert!(avg_hit_latency < avg_miss_latency);

    // Verify stats
    let stats = cache.cache_stats();
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
    println!("Stats avg hit latency: {:.3}ms", stats.avg_hit_latency_ms);
}

#[test]
fn test_query_custom_embeddings() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Create custom query
    let mut query: Vec<f32> = (0..embed_dim).map(|i| (i as f32).sin()).collect();

    // Normalize
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut query {
        *x /= norm;
    }

    let result = cache.get_similarities(&query, Instant::now()).unwrap();

    assert_eq!(result.similarities.len(), num_items);
    assert!(!result.from_cache);

    // Check bounds
    for &sim in &result.similarities {
        assert!(sim >= -1.0 && sim <= 1.0);
    }
}

#[test]
fn test_update_popular_indices() {
    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let mut cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Initial query
    let result1 = cache.get_similar_items(50).unwrap();
    assert!(result1.from_cache);

    // Update popular indices to different items
    let new_indices: Vec<usize> = (500..600).collect();
    cache.update_popular_indices(new_indices).unwrap();

    // Old item should now be cache miss
    let result2 = cache.get_similar_items(50).unwrap();
    assert!(!result2.from_cache);

    // New item should be cache hit
    let result3 = cache.get_similar_items(550).unwrap();
    assert!(result3.from_cache);
}

#[test]
fn test_large_scale() {
    // Test with realistic production sizes
    let num_items = 62_000;
    let embed_dim = 256;
    let num_popular = 10_000;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let start = Instant::now();
    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    );
    let init_time = start.elapsed();

    assert!(cache.is_ok());
    println!("Large-scale initialization: {:.2}ms", init_time.as_secs_f64() * 1000.0);

    let cache = cache.unwrap();

    // Test queries
    for i in 0..100 {
        let result = cache.get_similar_items(i).unwrap();
        assert!(result.from_cache);
    }

    let stats = cache.cache_stats();
    println!("Large-scale stats: hit_rate={:.2}%, avg_latency={:.3}ms",
             stats.hit_rate * 100.0, stats.avg_hit_latency_ms);
}

#[test]
fn test_concurrent_queries() {
    use std::sync::Arc;
    use std::thread;

    let num_items = 1000;
    let embed_dim = 128;
    let num_popular = 100;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = Arc::new(TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap());

    let mut handles = Vec::new();

    // Spawn multiple threads
    for thread_id in 0..4 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            for i in 0..25 {
                let item_id = (thread_id * 25 + i) % num_popular;
                let result = cache_clone.get_similar_items(item_id).unwrap();
                assert!(result.from_cache);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let stats = cache.cache_stats();
    assert_eq!(stats.total_hits, 100);
    assert_eq!(stats.hit_rate, 1.0);
}

#[test]
fn test_memory_efficiency() {
    let num_items = 10_000;
    let embed_dim = 256;
    let num_popular = 1000;

    let embeddings = create_test_embeddings(num_items, embed_dim);

    let cache = TemporalGPUCache::new(
        &embeddings,
        num_items,
        embed_dim,
        Some(num_popular),
        Some(0.1),
    ).unwrap();

    // Expected GPU memory usage (rough estimate):
    // - item_embeddings: num_items * embed_dim * 4 bytes
    // - popular_similarities: num_popular * num_items * 4 bytes
    // - temporal_weights: num_popular * 4 bytes

    let embedding_size = num_items * embed_dim * 4;
    let similarity_size = num_popular * num_items * 4;
    let weights_size = num_popular * 4;
    let total_bytes = embedding_size + similarity_size + weights_size;
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

    println!("Estimated GPU memory usage: {:.2} MB", total_mb);
    println!("  - Embeddings: {:.2} MB", embedding_size as f64 / (1024.0 * 1024.0));
    println!("  - Similarities: {:.2} MB", similarity_size as f64 / (1024.0 * 1024.0));
    println!("  - Weights: {:.2} KB", weights_size as f64 / 1024.0);

    // For production (62K items, 10K popular, 256 dim):
    // - Embeddings: 62,000 * 256 * 4 = 63.5 MB
    // - Similarities: 10,000 * 62,000 * 4 = 2.48 GB
    // - Weights: 10,000 * 4 = 40 KB
    // Total: ~2.54 GB
}
