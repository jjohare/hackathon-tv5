# Media Ontology Reasoner

Production-ready Rust implementation of OWL reasoning for GMC-O (Generic Media Content Ontology).

## Overview

This module adapts generic OWL reasoning for the media domain, providing:

- **Genre Hierarchy Reasoning**: Transitive closure, subgenre inference
- **Mood Inference**: VAD (Valence-Arousal-Dominance) model for emotional similarity
- **Cultural Context Matching**: Region, language, theme, and taboo awareness
- **User Preference Reasoning**: Personalized recommendations with explainability
- **GPU Acceleration**: Batch similarity computation hooks
- **Neo4j Integration**: Load/store ontology from graph database

## Architecture

```
ontology/
├── mod.rs          # Module entry point with re-exports
├── types.rs        # GMC-O data structures (MediaEntity, UserProfile, etc.)
├── reasoner.rs     # Core reasoning engine with media-specific logic
├── loader.rs       # Neo4j integration and test data generation
└── README.md       # This file
```

## Quick Start

### Basic Reasoning

```rust
use ontology::{
    reasoner::{MediaReasoner, ProductionMediaReasoner},
    loader::create_test_ontology,
};

// Create or load ontology
let ontology = create_test_ontology();

// Initialize reasoner
let reasoner = ProductionMediaReasoner::new();

// Infer new relationships
let axioms = reasoner.infer_axioms(&ontology)?;

// Check genre relationships
assert!(reasoner.is_subgenre_of("Thriller", "Drama", &ontology));
assert!(reasoner.are_disjoint_genres("Comedy", "Horror", &ontology));

// Infer moods for media
let media = ontology.media.get("media_001").unwrap();
let moods = reasoner.infer_mood(media, &ontology);
```

### Recommendations

```rust
use ontology::types::{UserProfile, DeliveryContext};

let user_profile = UserProfile {
    user_id: "user_123".to_string(),
    preferred_genres: [
        ("Drama".to_string(), 0.8),
        ("Thriller".to_string(), 0.9),
    ].into_iter().collect(),
    preferred_moods: [
        ("Tense".to_string(), 0.7),
    ].into_iter().collect(),
    // ... other fields
};

let context = DeliveryContext {
    device_type: DeviceType::Mobile,
    network_quality: NetworkQuality::Good,
    time_of_day: 20, // 8 PM
    // ... other fields
};

let recommendations = reasoner.recommend_for_user(
    &user_profile,
    &context,
    &ontology,
);

for rec in recommendations.iter().take(10) {
    println!("Media: {}, Score: {:.2}", rec.media_id, rec.score);
    for factor in &rec.reasoning {
        println!("  - {} (weight: {:.2}): {}",
            factor.factor_type, factor.weight, factor.explanation);
    }
}
```

### GPU Acceleration

```rust
// Enable GPU for batch operations
let reasoner = ProductionMediaReasoner::new()
    .with_gpu(1024); // Batch size

// Batch operations automatically use GPU when available
let axioms = reasoner.infer_axioms(&ontology)?;
```

### Neo4j Integration

```rust
use ontology::loader::{OntologyLoader, Neo4jConfig};

let config = Neo4jConfig {
    uri: "bolt://localhost:7687".to_string(),
    username: "neo4j".to_string(),
    password: std::env::var("NEO4J_PASSWORD").unwrap(),
    database: "media_ontology".to_string(),
};

let loader = OntologyLoader::new(config);

// Load ontology from Neo4j
let ontology = loader.load_ontology().await?;

// Load user profile
let user = loader.load_user_profile("user_123").await?;

// Store inferred axioms back to Neo4j
let axioms = reasoner.infer_axioms(&ontology)?;
loader.store_inferred_axioms(&axioms).await?;
```

## Data Structures

### MediaEntity

Represents any piece of content:

```rust
pub struct MediaEntity {
    pub id: String,
    pub title: String,
    pub media_type: MediaType,  // Video, Audio, Image, etc.
    pub genres: Vec<String>,
    pub moods: Vec<String>,
    pub themes: Vec<String>,
    pub cultural_context: Vec<String>,
    pub technical_metadata: TechnicalMetadata,
    pub semantic_tags: Vec<String>,
}
```

### UserProfile

User preferences and history:

```rust
pub struct UserProfile {
    pub user_id: String,
    pub preferred_genres: HashMap<String, f32>,  // Genre -> score
    pub preferred_moods: HashMap<String, f32>,   // Mood -> score
    pub cultural_background: Vec<String>,
    pub language_preferences: Vec<String>,
    pub interaction_history: Vec<InteractionRecord>,
    pub demographic: Demographic,
}
```

### DeliveryContext

Context for content delivery:

```rust
pub struct DeliveryContext {
    pub device_type: DeviceType,         // Mobile, TV, etc.
    pub network_quality: NetworkQuality, // Poor to Excellent
    pub time_of_day: u8,                 // Hour (0-23)
    pub location: Option<String>,
    pub social_context: SocialContext,   // Alone, WithFamily, etc.
}
```

### Mood (VAD Model)

Emotional dimensions:

```rust
pub struct Mood {
    pub name: String,
    pub valence: f32,    // Positive/negative: -1.0 to 1.0
    pub arousal: f32,    // Energy level: 0.0 to 1.0
    pub dominance: f32,  // Control/power: 0.0 to 1.0
    pub related_moods: Vec<String>,
}
```

## Reasoning Algorithms

### 1. Genre Hierarchy (Transitive Closure)

```
Drama
  └── Thriller
        └── PsychologicalThriller

Inferred: PsychologicalThriller ⊑ Drama
```

### 2. Mood Similarity (VAD Distance)

```rust
// Euclidean distance in VAD space
fn similarity(mood_a, mood_b) -> f32 {
    let d_valence = (mood_a.valence - mood_b.valence).abs();
    let d_arousal = (mood_a.arousal - mood_b.arousal).abs();
    let d_dominance = (mood_a.dominance - mood_b.dominance).abs();

    let distance = sqrt((d_v² + d_a² + d_d²) / 3.0);
    1.0 - distance  // Convert to similarity
}
```

### 3. Cultural Context Matching

Factors:
- Language match
- Regional match
- Theme overlap
- Taboo checking (negative)

### 4. Recommendation Scoring

Weighted combination:
- Genre preference: 30%
- Mood alignment: 25%
- Cultural relevance: 20%
- Contextual fit: 15%
- History similarity: 10%

## Performance

### Caching Strategy

- **Genre closure cache**: Precompute transitive closure
- **Mood similarity cache**: Store pairwise similarities
- **User profile cache**: Cache frequently accessed profiles

### GPU Acceleration

Batch operations eligible for GPU:
- Mood similarity matrix computation
- Semantic similarity (when using embeddings)
- Large-scale constraint checking

### Complexity

- Genre hierarchy inference: O(V² + E) where V = genres, E = edges
- Mood similarity: O(M²) where M = number of moods
- Recommendation generation: O(U × M × F) where U = users, M = media, F = factors

## Testing

Run tests:

```bash
cargo test --package ontology
```

Test with GPU:

```bash
cargo test --package ontology --features gpu
```

Integration tests with Neo4j:

```bash
# Start Neo4j
docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Run tests
cargo test --package ontology --features neo4j-integration
```

## Neo4j Schema

### Node Labels

- `Media`: Content entities
- `Genre`: Genre definitions
- `Mood`: Mood definitions
- `User`: User profiles
- `CulturalContext`: Cultural context definitions
- `Tag`: Semantic tags

### Relationships

- `SUBGENRE_OF`: Genre hierarchy
- `HAS_GENRE`: Media → Genre
- `HAS_MOOD`: Media → Mood
- `PREFERS`: User → Genre/Mood (with score)
- `INTERACTED_WITH`: User → Media (with metadata)
- `SIMILAR_TO`: Mood → Mood / Media → Media
- `DISJOINT_WITH`: Genre → Genre
- `EQUIVALENT_TO`: Genre → Genre
- `RELATED_TO`: Media → Media

### Example Cypher Queries

Load genre hierarchy:
```cypher
MATCH (child:Genre)-[:SUBGENRE_OF]->(parent:Genre)
RETURN child.name as child, collect(parent.name) as parents
```

Load user preferences:
```cypher
MATCH (u:User {id: $userId})-[r:PREFERS]->(g:Genre)
RETURN g.name as genre, r.score as score
```

Store inferred axiom:
```cypher
MATCH (child:Genre {name: $child}), (parent:Genre {name: $parent})
MERGE (child)-[r:SUBGENRE_OF]->(parent)
SET r.inferred = true, r.confidence = $confidence
```

## GPU Kernel Integration

When `gpu` feature is enabled, implement custom kernels:

```rust
// Hook for GPU batch similarity
#[cfg(feature = "gpu")]
fn gpu_batch_similarity(&self, pairs: Vec<(String, String)>) -> Vec<f32> {
    // 1. Transfer mood data to GPU
    // 2. Launch CUDA kernel for pairwise VAD distance
    // 3. Transfer results back
    // See: src/rust/gpu/kernels.cu
    unimplemented!()
}
```

CUDA kernel signature:
```cuda
__global__ void compute_vad_similarity(
    float* valence,
    float* arousal,
    float* dominance,
    int* pairs,
    float* results,
    int num_pairs
);
```

## Error Handling

All operations return `OntologyResult<T>`:

```rust
pub enum OntologyError {
    EntityNotFound(String),
    InvalidRelation(String),
    CircularDependency(String),
    ConstraintViolation(String),
    InferenceFailure(String),
}
```

## Integration with Hackathon System

This module integrates with:

1. **Neo4j Database** (`/home/devuser/workspace/hackathon-tv5/docs/neo4j_schema.md`)
   - Load GMC-O ontology
   - Store inferred relationships

2. **GPU Kernels** (`/home/devuser/workspace/hackathon-tv5/src/rust/gpu/`)
   - Batch similarity computation
   - Constraint enforcement

3. **Python Bindings** (via PyO3)
   - Expose reasoner to Python inference pipeline
   - Integration with LangGraph orchestration

4. **REST API**
   - Serve recommendations
   - Real-time reasoning endpoints

## Future Enhancements

- [ ] Temporal reasoning (time-based recommendations)
- [ ] Social network analysis (friend preferences)
- [ ] Multi-modal embedding similarity
- [ ] Probabilistic reasoning (uncertainty handling)
- [ ] Explanation generation (natural language)
- [ ] Online learning (update from interactions)

## License

See project root LICENSE file.

## References

- [OWL 2 Specification](https://www.w3.org/TR/owl2-overview/)
- [VAD Model](https://en.wikipedia.org/wiki/PAD_emotional_state_model)
- [Neo4j Graph Database](https://neo4j.com/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
