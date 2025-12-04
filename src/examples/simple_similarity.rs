/// Example: Simple Content Similarity
///
/// Demonstrates:
/// - Creating MediaContent instances
/// - Computing semantic similarity
/// - Basic vector operations

use hackathon_tv5::models::{
    MediaContent, ContentId, ContentType, Genre, EmbeddingVector
};

fn main() {
    println!("=== Simple Content Similarity Example ===\n");

    // Create two film content items
    let film1 = create_sample_film(
        "Inception",
        vec![Genre::SciFi, Genre::Thriller],
        generate_embedding(vec![0.8, 0.2, 0.1, 0.5]), // Sci-fi weighted
    );

    let film2 = create_sample_film(
        "Interstellar",
        vec![Genre::SciFi, Genre::Drama],
        generate_embedding(vec![0.7, 0.3, 0.15, 0.4]), // Similar to film1
    );

    let film3 = create_sample_film(
        "The Notebook",
        vec![Genre::Romance, Genre::Drama],
        generate_embedding(vec![0.1, 0.1, 0.9, 0.2]), // Romance weighted
    );

    println!("Film 1: {} ({:?})", film1.metadata.title, film1.genres);
    println!("Film 2: {} ({:?})", film2.metadata.title, film2.genres);
    println!("Film 3: {} ({:?})\n", film3.metadata.title, film3.genres);

    // Compute similarities
    let sim_1_2 = film1.similarity(&film2);
    let sim_1_3 = film1.similarity(&film3);
    let sim_2_3 = film2.similarity(&film3);

    println!("Similarity Scores:");
    println!("  {} ↔ {}: {:.4}", film1.metadata.title, film2.metadata.title, sim_1_2);
    println!("  {} ↔ {}: {:.4}", film1.metadata.title, film3.metadata.title, sim_1_3);
    println!("  {} ↔ {}: {:.4}\n", film2.metadata.title, film3.metadata.title, sim_2_3);

    // Interpret results
    println!("Analysis:");
    if sim_1_2 > 0.7 {
        println!("  ✓ {} and {} are highly similar (both Sci-Fi)",
                 film1.metadata.title, film2.metadata.title);
    }
    if sim_1_3 < 0.5 {
        println!("  ✓ {} and {} are quite different (Sci-Fi vs Romance)",
                 film1.metadata.title, film3.metadata.title);
    }

    // Find most similar pair
    let max_sim = sim_1_2.max(sim_1_3).max(sim_2_3);
    println!("\n  Most similar pair has similarity: {:.4}", max_sim);

    // Demonstrate embedding properties
    println!("\n=== Embedding Properties ===");

    let embedding = EmbeddingVector::new(
        vec![3.0, 4.0, 0.0, 0.0],
        "test-model"
    );

    println!("Original embedding: {:?}", embedding.data);
    println!("Is normalized: {}", embedding.is_normalized());

    let mut normalized = embedding.clone();
    normalized.normalize();
    println!("After normalization: {:?}", normalized.data);
    println!("Is normalized: {}", normalized.is_normalized());

    // Demonstrate cosine similarity
    let e1 = EmbeddingVector::new(vec![1.0, 0.0, 0.0, 0.0], "model");
    let e2 = EmbeddingVector::new(vec![0.0, 1.0, 0.0, 0.0], "model");
    let e3 = EmbeddingVector::new(vec![1.0, 0.0, 0.0, 0.0], "model");

    println!("\nCosine Similarity Examples:");
    println!("  Orthogonal vectors: {:.4}", e1.cosine_similarity(&e2));
    println!("  Identical vectors: {:.4}", e1.cosine_similarity(&e3));
}

/// Helper: Create sample film with embeddings
fn create_sample_film(
    title: &str,
    genres: Vec<Genre>,
    embedding: Vec<f32>,
) -> MediaContent {
    let mut content = MediaContent::new(
        ContentId::new(format!("film:{}", title.to_lowercase())),
        ContentType::Film,
        title.to_string(),
    );

    content.genres = genres;
    content.unified_embedding = embedding;
    content
}

/// Helper: Generate normalized embedding from feature vector
fn generate_embedding(features: Vec<f32>) -> Vec<f32> {
    // Expand to 1024 dimensions (simplified)
    let mut embedding = vec![0.0; 1024];

    // Fill with pattern based on features
    for (i, chunk) in embedding.chunks_mut(256).enumerate() {
        let feature_val = features.get(i).copied().unwrap_or(0.1);
        for val in chunk {
            *val = feature_val + (rand::random::<f32>() - 0.5) * 0.1;
        }
    }

    // Normalize (L2 norm = 1)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }

    embedding
}

// To run: cargo run --example simple_similarity
