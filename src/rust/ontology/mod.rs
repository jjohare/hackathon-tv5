/// GMC-O (Generic Media Content Ontology) Module
///
/// Production-ready Rust implementation of media ontology reasoning
/// with GPU acceleration hooks and Neo4j integration.
///
/// # Architecture
///
/// - `types`: Core data structures for media entities, user profiles, and contexts
/// - `reasoner`: OWL-based reasoning engine adapted for media domain
/// - `loader`: Neo4j integration for loading and storing ontology data
///
/// # Features
///
/// - Genre hierarchy reasoning with transitive closure
/// - Mood inference using VAD (Valence-Arousal-Dominance) model
/// - Cultural context matching
/// - User preference reasoning
/// - GPU acceleration hooks for batch operations
/// - Neo4j graph database integration
///
/// # Example Usage
///
/// ```rust
/// use ontology::{
///     reasoner::{MediaReasoner, ProductionMediaReasoner},
///     loader::create_test_ontology,
///     types::{UserProfile, DeliveryContext},
/// };
///
/// // Create ontology
/// let ontology = create_test_ontology();
///
/// // Initialize reasoner
/// let reasoner = ProductionMediaReasoner::new();
///
/// // Infer new relationships
/// let axioms = reasoner.infer_axioms(&ontology).unwrap();
///
/// // Check genre hierarchy
/// assert!(reasoner.is_subgenre_of("Thriller", "Drama", &ontology));
///
/// // Generate recommendations
/// let user_profile = UserProfile { /* ... */ };
/// let context = DeliveryContext { /* ... */ };
/// let recommendations = reasoner.recommend_for_user(&user_profile, &context, &ontology);
/// ```
///
/// # GPU Integration
///
/// When compiled with `gpu` feature, batch similarity computations
/// can be accelerated using CUDA kernels:
///
/// ```rust
/// let reasoner = ProductionMediaReasoner::new()
///     .with_gpu(1024); // Batch size
/// ```
///
/// # Neo4j Integration
///
/// Load ontology from Neo4j graph database:
///
/// ```rust
/// use ontology::loader::{OntologyLoader, Neo4jConfig};
///
/// let config = Neo4jConfig {
///     uri: "bolt://localhost:7687".to_string(),
///     username: "neo4j".to_string(),
///     password: "password".to_string(),
///     database: "neo4j".to_string(),
/// };
///
/// let loader = OntologyLoader::new(config);
/// let ontology = loader.load_ontology().await?;
/// ```

pub mod types;
pub mod reasoner;
pub mod loader;

// Re-export commonly used types
pub use types::{
    MediaEntity,
    MediaType,
    UserProfile,
    DeliveryContext,
    RecommendationResult,
    OntologyError,
    OntologyResult,
};

pub use reasoner::{
    MediaOntology,
    MediaReasoner,
    ProductionMediaReasoner,
    InferredMediaAxiom,
    MediaAxiomType,
};

pub use loader::{
    OntologyLoader,
    Neo4jConfig,
    create_test_ontology,
};

/// Version of the ontology module
pub const VERSION: &str = "1.0.0";

/// Supported OWL reasoning capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningCapability {
    /// Transitive closure of hierarchies
    TransitiveClosure,

    /// Disjoint class inference
    DisjointInference,

    /// Equivalent class inference
    EquivalenceInference,

    /// Media-specific mood inference
    MoodInference,

    /// Cultural context matching
    CulturalMatching,

    /// User preference reasoning
    PreferenceReasoning,
}

impl ReasoningCapability {
    /// Get all supported capabilities
    pub fn all() -> Vec<Self> {
        vec![
            Self::TransitiveClosure,
            Self::DisjointInference,
            Self::EquivalenceInference,
            Self::MoodInference,
            Self::CulturalMatching,
            Self::PreferenceReasoning,
        ]
    }
}

/// Module metadata and capabilities
pub struct ModuleInfo {
    pub version: &'static str,
    pub capabilities: Vec<ReasoningCapability>,
    pub gpu_enabled: bool,
}

impl ModuleInfo {
    /// Get module information
    pub fn get() -> Self {
        Self {
            version: VERSION,
            capabilities: ReasoningCapability::all(),
            gpu_enabled: cfg!(feature = "gpu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_info() {
        let info = ModuleInfo::get();
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.capabilities.len(), 6);
    }

    #[test]
    fn test_reasoning_capabilities() {
        let caps = ReasoningCapability::all();
        assert!(caps.contains(&ReasoningCapability::TransitiveClosure));
        assert!(caps.contains(&ReasoningCapability::MoodInference));
    }
}
