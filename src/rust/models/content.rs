/// Media content types and metadata structures
///
/// Represents all content types (films, series, episodes) with:
/// - Unique identifiers (EIDR standard)
/// - Multi-modal embeddings
/// - Rich metadata (cast, director, awards)
/// - Semantic classifications

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique content identifier using EIDR standard
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentId(pub String);

impl ContentId {
    /// Create new content ID from EIDR identifier
    pub fn new(eidr: impl Into<String>) -> Self {
        Self(eidr.into())
    }

    /// Parse EIDR format: 10.5240/XXXX-XXXX-XXXX-XXXX-XXXX-C
    pub fn from_eidr(eidr: &str) -> Result<Self, String> {
        if eidr.starts_with("10.5240/") && eidr.len() == 36 {
            Ok(Self(eidr.to_string()))
        } else {
            Err(format!("Invalid EIDR format: {}", eidr))
        }
    }

    /// Get the ID as string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Content type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    /// Feature film (theatrical release)
    Film,
    /// TV series (multiple seasons)
    Series,
    /// Single episode of a series
    Episode,
    /// Short film (<40 minutes)
    ShortFilm,
    /// Documentary
    Documentary,
    /// Miniseries (limited series)
    Miniseries,
}

// Genre is now auto-generated from ontology in models/generated.rs
pub use super::generated::Genre;

// VisualAesthetic is now auto-generated from ontology in models/generated.rs
pub use super::generated::VisualAesthetic;

// NarrativeStructure is now auto-generated from ontology in models/generated.rs
pub use super::generated::NarrativeStructure;

/// Pacing metrics (GPU motion analysis)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PacingMetrics {
    /// Average shot length in seconds
    pub avg_shot_length: f32,
    /// Cuts per minute
    pub cuts_per_minute: f32,
    /// Motion intensity (0.0-1.0)
    pub motion_intensity: f32,
    /// Dialogue density (words per minute)
    pub dialogue_density: f32,
}

/// Content metadata (EIDR + enriched data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    /// Primary title
    pub title: String,
    /// Original language title
    pub original_title: Option<String>,
    /// Release year
    pub year: u16,
    /// Runtime in minutes
    pub duration_minutes: u16,
    /// Primary language (ISO 639-1)
    pub language: String,
    /// Available subtitle languages
    pub subtitle_languages: Vec<String>,
    /// Director(s)
    pub directors: Vec<String>,
    /// Main cast (up to 10)
    pub cast: Vec<String>,
    /// Production companies
    pub production_companies: Vec<String>,
    /// Critical ratings (RT, Metacritic, IMDb)
    pub ratings: Ratings,
    /// Awards and nominations
    pub awards: Vec<Award>,
    /// Content rating (G, PG, PG-13, R, etc.)
    pub content_rating: String,
    /// Keywords/tags
    pub keywords: Vec<String>,
}

/// Critical ratings from multiple sources
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Ratings {
    /// Rotten Tomatoes critic score (0-100)
    pub rotten_tomatoes_critic: Option<u8>,
    /// Rotten Tomatoes audience score (0-100)
    pub rotten_tomatoes_audience: Option<u8>,
    /// Metacritic score (0-100)
    pub metacritic: Option<u8>,
    /// IMDb rating (0.0-10.0)
    pub imdb: Option<f32>,
}

/// Award information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Award {
    /// Award name (e.g., "Academy Award")
    pub name: String,
    /// Category (e.g., "Best Picture")
    pub category: String,
    /// Year awarded
    pub year: u16,
    /// Won or nominated
    pub won: bool,
}

/// Complete media content representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaContent {
    /// Unique identifier (EIDR)
    pub id: ContentId,

    /// Content type (film, series, episode)
    pub content_type: ContentType,

    /// Primary genres (1-3 typically)
    pub genres: Vec<Genre>,

    /// Visual aesthetic classification
    pub visual_aesthetic: Option<VisualAesthetic>,

    /// Narrative structure
    pub narrative_structure: Option<NarrativeStructure>,

    /// Pacing metrics (from motion analysis)
    pub pacing: Option<PacingMetrics>,

    /// Unified multi-modal embedding (1024-dim)
    pub unified_embedding: Vec<f32>,

    /// Visual embedding (768-dim, CLIP)
    pub visual_embedding: Option<Vec<f32>>,

    /// Audio embedding (512-dim, CLAP)
    pub audio_embedding: Option<Vec<f32>>,

    /// Text embedding (1024-dim, text-embedding-3)
    pub text_embedding: Option<Vec<f32>>,

    /// Rich metadata
    pub metadata: ContentMetadata,

    /// Confidence scores for classifications
    pub confidence_scores: HashMap<String, f32>,

    /// Timestamp when processed
    pub processed_at: chrono::DateTime<chrono::Utc>,
}

impl MediaContent {
    /// Create new media content with minimal required fields
    pub fn new(id: ContentId, content_type: ContentType, title: String) -> Self {
        Self {
            id,
            content_type,
            genres: Vec::new(),
            visual_aesthetic: None,
            narrative_structure: None,
            pacing: None,
            unified_embedding: vec![0.0; 1024],
            visual_embedding: None,
            audio_embedding: None,
            text_embedding: None,
            metadata: ContentMetadata {
                title,
                original_title: None,
                year: 2024,
                duration_minutes: 0,
                language: "en".to_string(),
                subtitle_languages: Vec::new(),
                directors: Vec::new(),
                cast: Vec::new(),
                production_companies: Vec::new(),
                ratings: Ratings::default(),
                awards: Vec::new(),
                content_rating: "NR".to_string(),
                keywords: Vec::new(),
            },
            confidence_scores: HashMap::new(),
            processed_at: chrono::Utc::now(),
        }
    }

    /// Check if embeddings are complete
    pub fn has_complete_embeddings(&self) -> bool {
        self.visual_embedding.is_some()
            && self.audio_embedding.is_some()
            && self.text_embedding.is_some()
            && !self.unified_embedding.is_empty()
    }

    /// Get primary genre
    pub fn primary_genre(&self) -> Option<&Genre> {
        self.genres.first()
    }

    /// Calculate semantic similarity with another content
    pub fn similarity(&self, other: &MediaContent) -> f32 {
        cosine_similarity(&self.unified_embedding, &other.unified_embedding)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_id_creation() {
        let id = ContentId::new("10.5240/1234-5678-9ABC-DEF0-1234-5");
        assert_eq!(id.as_str(), "10.5240/1234-5678-9ABC-DEF0-1234-5");
    }

    #[test]
    fn test_eidr_validation() {
        assert!(ContentId::from_eidr("10.5240/1234-5678-9ABC-DEF0-1234-5").is_ok());
        assert!(ContentId::from_eidr("invalid").is_err());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&c, &d).abs() < 1e-6);
    }
}
