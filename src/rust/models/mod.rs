/// Data models for the GPU-Accelerated Semantic Recommendation Engine
///
/// This module provides comprehensive type definitions for:
/// - Media content (films, series, episodes)
/// - Multi-modal embeddings (visual, audio, text)
/// - Ontology classes and semantic relationships
/// - User profiles and psychographic states
/// - Recommendation results with semantic paths
/// - GPU-optimized memory layouts

pub mod content;
pub mod embeddings;
pub mod ontology;
pub mod user;
pub mod recommendation;
pub mod gpu_types;
pub mod ontology_ffi;

#[cfg(test)]
mod ontology_ffi_tests;
pub mod generated;

// Re-export commonly used types
pub use content::{MediaContent, ContentType, ContentMetadata, ContentId, Genre, VisualAesthetic, NarrativeStructure};
pub use embeddings::{EmbeddingVector, VisualEmbedding, AudioEmbedding, TextEmbedding, MultiModalEmbedding};
pub use ontology::{OntologyClass, OntologyProperty, SemanticTriple, OWLAxiom};
pub use user::{UserProfile, PsychographicState, TasteCluster, ViewingContext};
pub use recommendation::{Recommendation, RecommendationScore, SemanticPath, ExplanationReason};
pub use gpu_types::{GPUEmbedding, GPUBatch, GPUEmbeddingBatch};
pub use ontology_ffi::{MediaOntologyNode, MediaOntologyConstraint, Float3, constraint_types, ontology_types};

// Re-export auto-generated ontology types
pub use generated::{
    Genre, VisualAesthetic, NarrativeStructure, Mood, Pacing,
    OntologyMappable
};
