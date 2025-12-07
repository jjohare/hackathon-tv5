//! Basic usage example for the semantic model
//!
//! This example demonstrates:
//! - Loading a model
//! - Encoding single texts
//! - Encoding batches
//! - Computing similarity
//! - Using different pooling strategies

use semantic_model::{ModelConfig, PoolingStrategy, SemanticModel, cosine_similarity};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("=== Semantic Model Example ===\n");

    // Example 1: Load model with default configuration
    println!("1. Loading model...");
    let model = SemanticModel::new(
        "models/all-MiniLM-L6-v2.onnx",
        "models/tokenizer.json"
    )?;

    println!("   Model loaded successfully!");
    println!("   Embedding dimension: {}", model.config().embedding_dim);
    println!("   Max sequence length: {}\n", model.config().max_length);

    // Example 2: Encode a single text
    println!("2. Encoding single text...");
    let text = "The cat sat on the mat";
    let embedding = model.encode(text)?;

    println!("   Text: \"{}\"", text);
    println!("   Embedding length: {}", embedding.len());
    println!("   First 5 values: {:?}\n", &embedding[..5]);

    // Example 3: Encode multiple texts
    println!("3. Encoding batch of texts...");
    let texts = vec![
        "The cat sat on the mat".to_string(),
        "A feline was resting on a rug".to_string(),
        "The weather is nice today".to_string(),
        "Machine learning is fascinating".to_string(),
    ];

    let embeddings = model.encode_batch(&texts)?;
    println!("   Encoded {} texts successfully\n", embeddings.len());

    // Example 4: Compute semantic similarity
    println!("4. Computing semantic similarity...");
    let text1 = "The cat sat on the mat";
    let text2 = "A feline was resting on a rug";
    let text3 = "The weather is nice today";

    let emb1 = model.encode(text1)?;
    let emb2 = model.encode(text2)?;
    let emb3 = model.encode(text3)?;

    let similarity_similar = cosine_similarity(&emb1, &emb2);
    let similarity_dissimilar = cosine_similarity(&emb1, &emb3);

    println!("   Text 1: \"{}\"", text1);
    println!("   Text 2: \"{}\"", text2);
    println!("   Similarity: {:.4}", similarity_similar);
    println!();
    println!("   Text 1: \"{}\"", text1);
    println!("   Text 3: \"{}\"", text3);
    println!("   Similarity: {:.4}\n", similarity_dissimilar);

    // Example 5: Different pooling strategies
    println!("5. Testing different pooling strategies...");

    for strategy in [PoolingStrategy::Mean, PoolingStrategy::Max, PoolingStrategy::Cls] {
        let config = ModelConfig {
            pooling_strategy: strategy,
            ..Default::default()
        };

        let model_custom = SemanticModel::with_config(
            "models/all-MiniLM-L6-v2.onnx",
            "models/tokenizer.json",
            config,
        )?;

        let emb = model_custom.encode("Test sentence")?;
        println!("   {} pooling: first 3 values: {:?}", strategy, &emb[..3]);
    }
    println!();

    // Example 6: Performance statistics
    println!("6. Model statistics...");
    let stats = model.stats();
    println!("   Total encodings: {}", stats.total_encodings);
    println!("   Total tokens processed: {}", stats.total_tokens_processed);
    println!("   Average encoding time: {:.2}ms\n", stats.avg_encoding_time_ms);

    // Example 7: Benchmark encoding speed
    println!("7. Benchmarking encoding speed...");
    let iterations = 100;
    let test_text = "This is a test sentence for benchmarking purposes";

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = model.encode(test_text)?;
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    let qps = 1000.0 / avg_ms;

    println!("   Iterations: {}", iterations);
    println!("   Average time per encoding: {:.2}ms", avg_ms);
    println!("   Queries per second: {:.0}\n", qps);

    // Example 8: Semantic search
    println!("8. Semantic search example...");
    let query = "cute animals";
    let documents = vec![
        "Dogs and cats are popular pets".to_string(),
        "Python is a programming language".to_string(),
        "Pandas are adorable bears".to_string(),
        "JavaScript frameworks are numerous".to_string(),
        "Kittens and puppies are very cute".to_string(),
    ];

    let query_emb = model.encode(query)?;
    let doc_embs = model.encode_batch(&documents)?;

    let mut similarities: Vec<(usize, f32)> = doc_embs
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(&query_emb, emb)))
        .collect();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("   Query: \"{}\"", query);
    println!("   Top 3 results:");
    for (idx, (doc_idx, score)) in similarities.iter().take(3).enumerate() {
        println!("      {}. [Score: {:.4}] {}", idx + 1, score, documents[*doc_idx]);
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
