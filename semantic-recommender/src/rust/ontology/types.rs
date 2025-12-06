/// GMC-O (Generic Media Content Ontology) Data Structures
///
/// This module defines the core data structures for media ontology reasoning,
/// including media entities, user profiles, context information, and relationships.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Media entity representing any piece of content in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaEntity {
    pub id: String,
    pub title: String,
    pub media_type: MediaType,
    pub genres: Vec<String>,
    pub moods: Vec<String>,
    pub themes: Vec<String>,
    pub cultural_context: Vec<String>,
    pub technical_metadata: TechnicalMetadata,
    pub semantic_tags: Vec<String>,
}

/// Types of media content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MediaType {
    Video,
    Audio,
    Image,
    Text,
    Interactive,
    Mixed,
}

/// Technical metadata for media content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalMetadata {
    pub duration_seconds: Option<f32>,
    pub resolution: Option<String>,
    pub format: String,
    pub bitrate: Option<u32>,
    pub file_size_bytes: Option<u64>,
}

/// User profile with preferences and interaction history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub preferred_genres: HashMap<String, f32>,
    pub preferred_moods: HashMap<String, f32>,
    pub cultural_background: Vec<String>,
    pub language_preferences: Vec<String>,
    pub interaction_history: Vec<InteractionRecord>,
    pub demographic: Demographic,
}

/// User demographic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographic {
    pub age_range: Option<AgeRange>,
    pub location: Option<String>,
    pub timezone: Option<String>,
}

/// Age range categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgeRange {
    Child,      // 0-12
    Teen,       // 13-17
    YoungAdult, // 18-24
    Adult,      // 25-54
    Senior,     // 55+
}

/// Record of user interaction with media
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub media_id: String,
    pub interaction_type: InteractionType,
    pub timestamp: u64,
    pub duration_seconds: Option<f32>,
    pub completion_rate: Option<f32>,
    pub rating: Option<f32>,
}

/// Types of user interactions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionType {
    View,
    Like,
    Dislike,
    Share,
    Comment,
    Skip,
    Complete,
}

/// Context information for content delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryContext {
    pub device_type: DeviceType,
    pub network_quality: NetworkQuality,
    pub time_of_day: u8, // Hour 0-23
    pub location: Option<String>,
    pub social_context: SocialContext,
}

/// Device categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceType {
    Mobile,
    Tablet,
    Desktop,
    TV,
    SmartDisplay,
}

/// Network quality levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum NetworkQuality {
    Poor,
    Fair,
    Good,
    Excellent,
}

/// Social context for content consumption
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SocialContext {
    Alone,
    WithFamily,
    WithFriends,
    Public,
}

/// Media relationship types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MediaRelation {
    SequelOf,
    PrequelOf,
    SimilarTo,
    RelatedTo,
    PartOf,
    BasedOn,
    RemakeOf,
    InspiredBy,
}

/// Genre hierarchy node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreNode {
    pub name: String,
    pub parent_genres: Vec<String>,
    pub child_genres: Vec<String>,
    pub characteristics: Vec<String>,
    pub typical_moods: Vec<String>,
}

/// Mood definition with emotional dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mood {
    pub name: String,
    pub valence: f32,    // Positive/negative: -1.0 to 1.0
    pub arousal: f32,    // Energy level: 0.0 to 1.0
    pub dominance: f32,  // Control/power: 0.0 to 1.0
    pub related_moods: Vec<String>,
}

/// Cultural context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalContext {
    pub region: String,
    pub language: String,
    pub cultural_themes: Vec<String>,
    pub taboos: Vec<String>,
    pub preferences: HashMap<String, f32>,
}

/// Recommendation result with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationResult {
    pub media_id: String,
    pub score: f32,
    pub reasoning: Vec<ReasoningFactor>,
    pub confidence: f32,
}

/// Factor contributing to recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningFactor {
    pub factor_type: FactorType,
    pub weight: f32,
    pub explanation: String,
}

/// Types of reasoning factors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FactorType {
    GenreMatch,
    MoodAlignment,
    CulturalRelevance,
    ContextualFit,
    SimilarityToHistory,
    PopularityTrend,
    SemanticSimilarity,
    TechnicalCompatibility,
}

/// Error types for ontology operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OntologyError {
    EntityNotFound(String),
    InvalidRelation(String),
    CircularDependency(String),
    ConstraintViolation(String),
    InferenceFailure(String),
}

impl std::fmt::Display for OntologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntityNotFound(id) => write!(f, "Entity not found: {}", id),
            Self::InvalidRelation(msg) => write!(f, "Invalid relation: {}", msg),
            Self::CircularDependency(msg) => write!(f, "Circular dependency: {}", msg),
            Self::ConstraintViolation(msg) => write!(f, "Constraint violation: {}", msg),
            Self::InferenceFailure(msg) => write!(f, "Inference failure: {}", msg),
        }
    }
}

impl std::error::Error for OntologyError {}

/// Result type for ontology operations
pub type OntologyResult<T> = Result<T, OntologyError>;
