/// Simple embedding service for converting text to vectors
/// In production, this would call a proper embedding model (OpenAI, Cohere, etc.)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct EmbeddingService {
    dimension: usize,
}

impl EmbeddingService {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a deterministic embedding from text
    /// This is a MOCK implementation for demonstration
    /// In production, use real embedding models like:
    /// - OpenAI text-embedding-ada-002
    /// - Cohere embed-multilingual-v3.0
    /// - sentence-transformers
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        // Create deterministic hash-based embedding
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate pseudo-random but deterministic vector
        let mut embedding = Vec::with_capacity(self.dimension);
        let mut seed = hash;

        for _ in 0..self.dimension {
            // Simple LCG (Linear Congruential Generator)
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((seed / 65536) % 2048) as f32 / 2048.0;
            embedding.push(value);
        }

        // Normalize to unit vector
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        embedding
    }

    /// Batch embed multiple texts
    pub fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let service = EmbeddingService::new(384);
        let embedding = service.embed_text("test query");
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_embedding_normalized() {
        let service = EmbeddingService::new(128);
        let embedding = service.embed_text("test");
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_deterministic() {
        let service = EmbeddingService::new(256);
        let emb1 = service.embed_text("same text");
        let emb2 = service.embed_text("same text");
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let service = EmbeddingService::new(256);
        let emb1 = service.embed_text("text one");
        let emb2 = service.embed_text("text two");
        assert_ne!(emb1, emb2);
    }
}
