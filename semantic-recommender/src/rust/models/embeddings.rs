/// Multi-modal embedding types for semantic representation
///
/// Embeddings represent content in high-dimensional semantic spaces:
/// - Visual: 768-dim (CLIP ViT-L/14)
/// - Audio: 512-dim (CLAP)
/// - Text: 1024-dim (text-embedding-3)
/// - Unified: 1024-dim (fused multi-modal)

use serde::{Deserialize, Serialize};

/// Generic embedding vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVector {
    /// Dimensionality of the vector
    pub dimensions: usize,

    /// Embedding data (normalized, L2 norm = 1)
    pub data: Vec<f32>,

    /// Model used to generate embedding
    pub embedding_model: String,

    /// Timestamp when generated
    pub generated_at: chrono::DateTime<chrono::Utc>,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

impl EmbeddingVector {
    /// Create new embedding vector
    pub fn new(data: Vec<f32>, model: impl Into<String>) -> Self {
        let dimensions = data.len();
        Self {
            dimensions,
            data,
            embedding_model: model.into(),
            generated_at: chrono::Utc::now(),
            confidence: 1.0,
        }
    }

    /// Normalize the embedding vector (L2 norm = 1)
    pub fn normalize(&mut self) {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut self.data {
                *value /= norm;
            }
        }
    }

    /// Check if vector is normalized
    pub fn is_normalized(&self) -> bool {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        (norm - 1.0).abs() < 1e-5
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &EmbeddingVector) -> f32 {
        if self.dimensions != other.dimensions {
            return 0.0;
        }

        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute Euclidean distance
    pub fn euclidean_distance(&self, other: &EmbeddingVector) -> f32 {
        if self.dimensions != other.dimensions {
            return f32::INFINITY;
        }

        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Visual embedding from image/video analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEmbedding {
    /// Base embedding vector (768-dim, CLIP)
    pub embedding: EmbeddingVector,

    /// Color palette features (64-dim)
    pub color_palette: Vec<f32>,

    /// Motion vectors (32-dim)
    pub motion_features: Vec<f32>,

    /// Number of frames analyzed
    pub frame_count: usize,

    /// Dominant colors (hex codes)
    pub dominant_colors: Vec<String>,
}

impl VisualEmbedding {
    /// Create from raw frame embeddings with aggregation
    pub fn from_frames(
        frame_embeddings: Vec<Vec<f32>>,
        model: impl Into<String>,
    ) -> Self {
        let dimensions = frame_embeddings.first().map(|v| v.len()).unwrap_or(768);
        let frame_count = frame_embeddings.len();

        // Aggregate by averaging (attention pooling in production)
        let mut aggregated = vec![0.0; dimensions];
        for frame in &frame_embeddings {
            for (i, value) in frame.iter().enumerate() {
                if i < aggregated.len() {
                    aggregated[i] += value;
                }
            }
        }

        for value in &mut aggregated {
            *value /= frame_count as f32;
        }

        let mut embedding = EmbeddingVector::new(aggregated, model);
        embedding.normalize();

        Self {
            embedding,
            color_palette: vec![0.0; 64],
            motion_features: vec![0.0; 32],
            frame_count,
            dominant_colors: Vec::new(),
        }
    }

    /// Get visual aesthetic score
    pub fn aesthetic_score(&self) -> f32 {
        // Placeholder: compute from color harmony and composition
        self.embedding.confidence
    }
}

/// Audio embedding from soundtrack/dialogue analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedding {
    /// Base embedding vector (512-dim, CLAP)
    pub embedding: EmbeddingVector,

    /// Music features (tempo, key, intensity) (64-dim)
    pub music_features: Vec<f32>,

    /// Spectral features (MFCC, chroma) (128-dim)
    pub spectral_features: Vec<f32>,

    /// Detected tempo (BPM)
    pub tempo_bpm: Option<f32>,

    /// Musical key (C, C#, D, etc.)
    pub musical_key: Option<String>,

    /// Major/Minor classification
    pub key_mode: Option<KeyMode>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KeyMode {
    Major,
    Minor,
}

impl AudioEmbedding {
    /// Create from audio analysis results
    pub fn new(
        audio_data: Vec<f32>,
        tempo_bpm: Option<f32>,
        key: Option<String>,
        mode: Option<KeyMode>,
        model: impl Into<String>,
    ) -> Self {
        let embedding = EmbeddingVector::new(audio_data, model);

        Self {
            embedding,
            music_features: vec![0.0; 64],
            spectral_features: vec![0.0; 128],
            tempo_bpm,
            musical_key: key,
            key_mode: mode,
        }
    }

    /// Get mood intensity (0.0 = calm, 1.0 = intense)
    pub fn mood_intensity(&self) -> f32 {
        self.tempo_bpm.unwrap_or(100.0) / 200.0
    }
}

/// Text embedding from script/subtitle analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedding {
    /// Base embedding vector (1024-dim, text-embedding-3)
    pub embedding: EmbeddingVector,

    /// Extracted themes
    pub themes: Vec<String>,

    /// Identified tropes
    pub tropes: Vec<String>,

    /// Emotional arc (sentiment over time)
    pub emotional_arc: Vec<f32>,

    /// Dialogue complexity (Flesch-Kincaid score)
    pub complexity_score: f32,

    /// Language (ISO 639-1)
    pub language: String,
}

impl TextEmbedding {
    /// Create from text analysis
    pub fn new(
        text_data: Vec<f32>,
        themes: Vec<String>,
        language: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let embedding = EmbeddingVector::new(text_data, model);

        Self {
            embedding,
            themes,
            tropes: Vec::new(),
            emotional_arc: Vec::new(),
            complexity_score: 0.0,
            language: language.into(),
        }
    }

    /// Check if text is complex (requires high reading level)
    pub fn is_complex(&self) -> bool {
        self.complexity_score > 12.0 // College level
    }
}

/// Unified multi-modal embedding (fusion of visual+audio+text)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalEmbedding {
    /// Unified embedding (1024-dim, projected from 2304-dim concat)
    pub unified: EmbeddingVector,

    /// Source embeddings (preserved for debugging)
    pub visual: Option<VisualEmbedding>,
    pub audio: Option<AudioEmbedding>,
    pub text: Option<TextEmbedding>,

    /// Fusion weights (attention scores)
    pub fusion_weights: FusionWeights,

    /// Quality score (0.0-1.0)
    pub quality_score: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FusionWeights {
    pub visual_weight: f32,
    pub audio_weight: f32,
    pub text_weight: f32,
}

impl FusionWeights {
    /// Default balanced weights
    pub fn balanced() -> Self {
        Self {
            visual_weight: 0.333,
            audio_weight: 0.333,
            text_weight: 0.334,
        }
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.visual_weight + self.audio_weight + self.text_weight;
        if sum > 0.0 {
            self.visual_weight /= sum;
            self.audio_weight /= sum;
            self.text_weight /= sum;
        }
    }
}

impl MultiModalEmbedding {
    /// Create from individual modality embeddings with GPU fusion
    pub fn fuse(
        visual: VisualEmbedding,
        audio: AudioEmbedding,
        text: TextEmbedding,
        weights: FusionWeights,
    ) -> Self {
        // Concatenate embeddings: [visual (768) | audio (512) | text (1024)] = 2304
        let mut concatenated = Vec::with_capacity(2304);
        concatenated.extend(&visual.embedding.data);
        concatenated.extend(&audio.embedding.data);
        concatenated.extend(&text.embedding.data);

        // Project to unified 1024-dim (using learned projection matrix in production)
        let unified_data = project_to_1024(&concatenated);

        let mut unified = EmbeddingVector::new(unified_data, "multi-modal-fusion-v1");
        unified.normalize();

        // Compute quality score based on completeness and confidence
        let quality_score = (visual.embedding.confidence
            + audio.embedding.confidence
            + text.embedding.confidence) / 3.0;

        Self {
            unified,
            visual: Some(visual),
            audio: Some(audio),
            text: Some(text),
            fusion_weights: weights,
            quality_score,
        }
    }

    /// Check if all modalities are present
    pub fn is_complete(&self) -> bool {
        self.visual.is_some() && self.audio.is_some() && self.text.is_some()
    }
}

/// Project high-dimensional concatenated embedding to 1024-dim
/// (Simplified version - production uses learned neural projection)
fn project_to_1024(concatenated: &[f32]) -> Vec<f32> {
    let mut projected = vec![0.0; 1024];

    // Simple linear projection (averaging blocks)
    let block_size = concatenated.len() / 1024;
    for (i, chunk) in concatenated.chunks(block_size).enumerate() {
        if i < 1024 {
            projected[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
        }
    }

    projected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_normalization() {
        let data = vec![3.0, 4.0, 0.0];
        let mut embedding = EmbeddingVector::new(data, "test-model");
        embedding.normalize();

        assert!((embedding.data[0] - 0.6).abs() < 1e-5);
        assert!((embedding.data[1] - 0.8).abs() < 1e-5);
        assert!(embedding.is_normalized());
    }

    #[test]
    fn test_cosine_similarity() {
        let e1 = EmbeddingVector::new(vec![1.0, 0.0, 0.0], "model");
        let e2 = EmbeddingVector::new(vec![1.0, 0.0, 0.0], "model");
        assert!((e1.cosine_similarity(&e2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fusion_weights_normalize() {
        let mut weights = FusionWeights {
            visual_weight: 2.0,
            audio_weight: 3.0,
            text_weight: 5.0,
        };
        weights.normalize();

        let sum = weights.visual_weight + weights.audio_weight + weights.text_weight;
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
