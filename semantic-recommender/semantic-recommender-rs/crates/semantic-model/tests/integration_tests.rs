//! Integration tests for semantic model
//!
//! These tests verify the complete functionality of the ONNX-based semantic model.
//! Note: These tests require actual model files to run.

use semantic_model::{ModelConfig, ModelError, PoolingStrategy, SemanticModel, cosine_similarity};

/// Mock test data for validation
const TEST_SENTENCES: &[&str] = &[
    "The cat sat on the mat",
    "A feline was resting on a rug",
    "The weather is nice today",
    "Machine learning is fascinating",
];

#[test]
#[ignore] // Requires actual model files
fn test_model_loading() {
    let result = SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    );

    match result {
        Ok(model) => {
            assert_eq!(model.config().embedding_dim, 384);
            assert_eq!(model.config().max_length, 512);
        }
        Err(ModelError::IoError(_)) => {
            // Expected if model files don't exist
            println!("Model files not found - skipping test");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
#[ignore] // Requires actual model files
fn test_single_encoding() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    let embedding = model.encode(TEST_SENTENCES[0]).unwrap();

    // Verify embedding properties
    assert_eq!(embedding.len(), 384, "Embedding dimension should be 384");

    // Check normalization (if enabled)
    if model.config().normalize {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Normalized embedding should have unit norm");
    }

    // Verify values are reasonable
    for &val in &embedding {
        assert!(val.is_finite(), "All values should be finite");
    }
}

#[test]
#[ignore] // Requires actual model files
fn test_batch_encoding() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    let texts: Vec<String> = TEST_SENTENCES.iter().map(|s| s.to_string()).collect();
    let embeddings = model.encode_batch(&texts).unwrap();

    assert_eq!(embeddings.len(), TEST_SENTENCES.len());

    for embedding in &embeddings {
        assert_eq!(embedding.len(), 384);
    }
}

#[test]
#[ignore] // Requires actual model files
fn test_semantic_similarity() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    // Similar sentences
    let emb1 = model.encode(TEST_SENTENCES[0]).unwrap();
    let emb2 = model.encode(TEST_SENTENCES[1]).unwrap();

    // Dissimilar sentences
    let emb3 = model.encode(TEST_SENTENCES[2]).unwrap();

    let similarity_similar = cosine_similarity(&emb1, &emb2);
    let similarity_dissimilar = cosine_similarity(&emb1, &emb3);

    println!("Similar sentences similarity: {}", similarity_similar);
    println!("Dissimilar sentences similarity: {}", similarity_dissimilar);

    // Similar sentences should have higher similarity
    assert!(
        similarity_similar > similarity_dissimilar,
        "Similar sentences should have higher cosine similarity"
    );
}

#[test]
#[ignore] // Requires actual model files
fn test_consistency() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    let text = TEST_SENTENCES[0];

    // Encode the same text multiple times
    let emb1 = model.encode(text).unwrap();
    let emb2 = model.encode(text).unwrap();

    // Results should be identical
    for (a, b) in emb1.iter().zip(emb2.iter()) {
        assert!((a - b).abs() < 1e-6, "Encodings should be deterministic");
    }
}

#[test]
#[ignore] // Requires actual model files
fn test_batch_vs_single_consistency() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    let texts: Vec<String> = TEST_SENTENCES.iter().map(|s| s.to_string()).collect();

    // Encode individually
    let mut single_embeddings = Vec::new();
    for text in &texts {
        single_embeddings.push(model.encode(text).unwrap());
    }

    // Encode as batch
    let batch_embeddings = model.encode_batch(&texts).unwrap();

    // Results should match
    assert_eq!(single_embeddings.len(), batch_embeddings.len());

    for (single, batch) in single_embeddings.iter().zip(batch_embeddings.iter()) {
        for (a, b) in single.iter().zip(batch.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Batch and single encodings should match"
            );
        }
    }
}

#[test]
#[ignore] // Requires actual model files
fn test_different_pooling_strategies() {
    // Test with mean pooling
    let config_mean = ModelConfig {
        pooling_strategy: PoolingStrategy::Mean,
        ..Default::default()
    };

    let model_mean = match SemanticModel::with_config(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json",
        config_mean,
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    let emb_mean = model_mean.encode(TEST_SENTENCES[0]).unwrap();

    // Test with max pooling
    let config_max = ModelConfig {
        pooling_strategy: PoolingStrategy::Max,
        ..Default::default()
    };

    let model_max = match SemanticModel::with_config(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json",
        config_max,
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    let emb_max = model_max.encode(TEST_SENTENCES[0]).unwrap();

    // Embeddings should be different with different pooling strategies
    let similarity = cosine_similarity(&emb_mean, &emb_max);
    println!("Similarity between mean and max pooling: {}", similarity);

    assert_eq!(emb_mean.len(), emb_max.len());
    // They should be somewhat similar but not identical
    assert!(similarity > 0.5 && similarity < 0.99);
}

#[test]
#[ignore] // Requires actual model files
fn test_empty_and_edge_cases() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    // Empty string
    let emb_empty = model.encode("").unwrap();
    assert_eq!(emb_empty.len(), 384);

    // Single character
    let emb_single = model.encode("a").unwrap();
    assert_eq!(emb_single.len(), 384);

    // Very long text (should be truncated)
    let long_text = "word ".repeat(1000);
    let emb_long = model.encode(&long_text).unwrap();
    assert_eq!(emb_long.len(), 384);

    // Empty batch
    let empty_batch: Vec<String> = vec![];
    let emb_batch_empty = model.encode_batch(&empty_batch).unwrap();
    assert_eq!(emb_batch_empty.len(), 0);
}

#[test]
#[ignore] // Requires actual model files
fn test_model_stats() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    model.reset_stats();

    // Initial stats should be zero
    let initial_stats = model.stats();
    assert_eq!(initial_stats.total_encodings, 0);

    // Encode some texts
    let _ = model.encode(TEST_SENTENCES[0]).unwrap();
    let texts: Vec<String> = vec![
        TEST_SENTENCES[1].to_string(),
        TEST_SENTENCES[2].to_string(),
    ];
    let _ = model.encode_batch(&texts).unwrap();

    // Check updated stats
    let stats = model.stats();
    assert_eq!(stats.total_encodings, 3);
    assert!(stats.total_tokens_processed > 0);
    assert!(stats.avg_encoding_time_ms >= 0.0);
}

#[test]
#[ignore] // Performance test - requires actual model files
fn test_performance_target() {
    let model = match SemanticModel::new(
        "tests/fixtures/model.onnx",
        "tests/fixtures/tokenizer.json"
    ) {
        Ok(m) => m,
        Err(_) => {
            println!("Model files not found - skipping test");
            return;
        }
    };

    model.reset_stats();

    // Warm-up
    for _ in 0..5 {
        let _ = model.encode(TEST_SENTENCES[0]).unwrap();
    }

    // Measure performance
    let iterations = 100;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let _ = model.encode(TEST_SENTENCES[0]).unwrap();
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!("Average encoding time: {:.2}ms", avg_ms);

    // Target: <5ms per query (should be faster than Python's 11ms)
    assert!(
        avg_ms < 10.0,
        "Encoding should be faster than 10ms (target <5ms), got {:.2}ms",
        avg_ms
    );
}

#[test]
fn test_cosine_similarity_edge_cases() {
    // Identical vectors
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

    // Orthogonal vectors
    let c = vec![1.0, 0.0, 0.0];
    let d = vec![0.0, 1.0, 0.0];
    assert!(cosine_similarity(&c, &d).abs() < 1e-6);

    // Opposite vectors
    let e = vec![1.0, 0.0, 0.0];
    let f = vec![-1.0, 0.0, 0.0];
    assert!((cosine_similarity(&e, &f) + 1.0).abs() < 1e-6);

    // Different lengths (should return 0)
    let g = vec![1.0, 2.0];
    let h = vec![1.0, 2.0, 3.0];
    assert_eq!(cosine_similarity(&g, &h), 0.0);

    // Zero vectors
    let zero1 = vec![0.0, 0.0, 0.0];
    let zero2 = vec![0.0, 0.0, 0.0];
    assert_eq!(cosine_similarity(&zero1, &zero2), 0.0);
}
