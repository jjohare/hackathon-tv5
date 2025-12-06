# API Reference: Data Models & Types

**Version:** 1.0
**Date:** 2025-12-04

Complete API documentation for all data models, types, and core functions.

---

## Table of Contents

1. [Content Models](#content-models)
2. [Embedding Types](#embedding-types)
3. [Ontology Models](#ontology-models)
4. [User Models](#user-models)
5. [Recommendation Models](#recommendation-models)
6. [GPU Types](#gpu-types)

---

## Content Models

### `MediaContent`

Core representation of media assets (films, series, episodes).

```rust
pub struct MediaContent {
    pub id: ContentId,
    pub content_type: ContentType,
    pub genres: Vec<Genre>,
    pub visual_aesthetic: Option<VisualAesthetic>,
    pub narrative_structure: Option<NarrativeStructure>,
    pub pacing: Option<PacingMetrics>,
    pub unified_embedding: Vec<f32>,
    pub visual_embedding: Option<Vec<f32>>,
    pub audio_embedding: Option<Vec<f32>>,
    pub text_embedding: Option<Vec<f32>>,
    pub metadata: ContentMetadata,
    pub confidence_scores: HashMap<String, f32>,
    pub processed_at: DateTime<Utc>,
}
```

**Methods**:

- `new(id, content_type, title)` - Create new media content
- `has_complete_embeddings() -> bool` - Check if all embeddings are present
- `primary_genre() -> Option<&Genre>` - Get primary genre
- `similarity(&self, other) -> f32` - Compute cosine similarity

**Example**:
```rust
let film = MediaContent::new(
    ContentId::new("film:inception"),
    ContentType::Film,
    "Inception".to_string(),
);

film.genres = vec![Genre::SciFi, Genre::Thriller];
assert!(film.has_complete_embeddings());
```

---

### `ContentId`

Unique content identifier using EIDR standard.

```rust
pub struct ContentId(pub String);
```

**Methods**:

- `new(id: impl Into<String>) -> Self` - Create from string
- `from_eidr(eidr: &str) -> Result<Self, String>` - Parse EIDR format
- `as_str(&self) -> &str` - Get ID as string

**EIDR Format**: `10.5240/XXXX-XXXX-XXXX-XXXX-XXXX-C`

**Example**:
```rust
let id = ContentId::from_eidr("10.5240/1234-5678-9ABC-DEF0-1234-5")?;
```

---

### `ContentType`

Classification of content types.

```rust
pub enum ContentType {
    Film,         // Feature film
    Series,       // TV series
    Episode,      // Single episode
    ShortFilm,    // <40 minutes
    Documentary,  // Documentary
    Miniseries,   // Limited series
}
```

---

### `Genre`

Genre classification aligned with GMC-O ontology.

```rust
pub enum Genre {
    Action, Adventure, Animation, Biography, Comedy, Crime,
    Documentary, Drama, Family, Fantasy, Horror, Mystery,
    Romance, SciFi, Thriller, War, Western,
    Custom(String),
}
```

---

### `VisualAesthetic`

GPU-derived visual style classification.

```rust
pub enum VisualAesthetic {
    Noir,           // Dark, high-contrast
    Neon,           // Bright neon colors
    Pastel,         // Soft, muted colors
    Desaturated,    // Low saturation, gritty
    Naturalistic,   // Natural, documentary style
    Vibrant,        // High saturation
}
```

---

### `NarrativeStructure`

LLM-derived narrative classification.

```rust
pub enum NarrativeStructure {
    Linear,         // Chronological
    NonLinear,      // Non-chronological
    HerosJourney,   // Hero archetype
    EnsembleCast,   // Multiple storylines
    Circular,       // Circular narrative
    FrameStory,     // Story within story
}
```

---

### `PacingMetrics`

GPU motion analysis results.

```rust
pub struct PacingMetrics {
    pub avg_shot_length: f32,      // Seconds
    pub cuts_per_minute: f32,      // Cuts/min
    pub motion_intensity: f32,     // 0.0-1.0
    pub dialogue_density: f32,     // Words/min
}
```

---

## Embedding Types

### `EmbeddingVector`

Generic embedding with metadata.

```rust
pub struct EmbeddingVector {
    pub dimensions: usize,
    pub data: Vec<f32>,
    pub embedding_model: String,
    pub generated_at: DateTime<Utc>,
    pub confidence: f32,
}
```

**Methods**:

- `new(data, model) -> Self` - Create embedding
- `normalize(&mut self)` - Normalize to L2 norm = 1
- `is_normalized() -> bool` - Check normalization
- `cosine_similarity(&self, other) -> f32` - Compute similarity
- `euclidean_distance(&self, other) -> f32` - Compute distance

**Example**:
```rust
let mut embedding = EmbeddingVector::new(vec![3.0, 4.0], "model");
embedding.normalize();
assert!(embedding.is_normalized());
```

---

### `VisualEmbedding`

Visual features from image/video analysis.

```rust
pub struct VisualEmbedding {
    pub embedding: EmbeddingVector,    // 768-dim (CLIP)
    pub color_palette: Vec<f32>,       // 64-dim
    pub motion_features: Vec<f32>,     // 32-dim
    pub frame_count: usize,
    pub dominant_colors: Vec<String>,  // Hex codes
}
```

**Methods**:

- `from_frames(embeddings, model) -> Self` - Aggregate frame embeddings
- `aesthetic_score() -> f32` - Get visual aesthetic score

---

### `AudioEmbedding`

Audio features from soundtrack/dialogue.

```rust
pub struct AudioEmbedding {
    pub embedding: EmbeddingVector,       // 512-dim (CLAP)
    pub music_features: Vec<f32>,         // 64-dim
    pub spectral_features: Vec<f32>,      // 128-dim
    pub tempo_bpm: Option<f32>,
    pub musical_key: Option<String>,
    pub key_mode: Option<KeyMode>,        // Major/Minor
}
```

**Methods**:

- `new(data, tempo, key, mode, model) -> Self` - Create audio embedding
- `mood_intensity() -> f32` - Get mood intensity (0.0-1.0)

---

### `TextEmbedding`

Text features from script/subtitle analysis.

```rust
pub struct TextEmbedding {
    pub embedding: EmbeddingVector,    // 1024-dim
    pub themes: Vec<String>,
    pub tropes: Vec<String>,
    pub emotional_arc: Vec<f32>,
    pub complexity_score: f32,         // Flesch-Kincaid
    pub language: String,              // ISO 639-1
}
```

**Methods**:

- `new(data, themes, language, model) -> Self` - Create text embedding
- `is_complex() -> bool` - Check if requires high reading level (>12.0)

---

### `MultiModalEmbedding`

Fused multi-modal embedding.

```rust
pub struct MultiModalEmbedding {
    pub unified: EmbeddingVector,        // 1024-dim
    pub visual: Option<VisualEmbedding>,
    pub audio: Option<AudioEmbedding>,
    pub text: Option<TextEmbedding>,
    pub fusion_weights: FusionWeights,
    pub quality_score: f32,
}
```

**Methods**:

- `fuse(visual, audio, text, weights) -> Self` - GPU tensor fusion
- `is_complete() -> bool` - Check if all modalities present

**Example**:
```rust
let fused = MultiModalEmbedding::fuse(
    visual_emb,
    audio_emb,
    text_emb,
    FusionWeights::balanced(),
);
```

---

## Ontology Models

### `OntologyClass`

OWL class representation (GMC-O).

```rust
pub struct OntologyClass {
    pub uri: String,
    pub label: String,
    pub comment: Option<String>,
    pub parent: Option<Box<OntologyClass>>,
    pub equivalent_to: Vec<String>,
}
```

**Methods**:

- `new(uri, label) -> Self` - Create class
- `with_parent(self, parent) -> Self` - Set parent class
- `with_comment(self, comment) -> Self` - Set description

---

### `OntologyProperty`

OWL property (relationships between classes).

```rust
pub struct OntologyProperty {
    pub uri: String,
    pub label: String,
    pub domain: String,
    pub range: String,
    pub property_type: PropertyType,
    pub comment: Option<String>,
}
```

**Methods**:

- `new_object_property(uri, label, domain, range) -> Self`
- `new_datatype_property(uri, label, domain, range) -> Self`

---

### `SemanticTriple`

RDF triple (subject-predicate-object).

```rust
pub struct SemanticTriple {
    pub subject: String,
    pub predicate: String,
    pub object: TripleObject,
    pub confidence: f32,
}
```

**Methods**:

- `new_resource(subject, predicate, object) -> Self`
- `new_literal(subject, predicate, literal) -> Self`
- `with_confidence(self, confidence) -> Self`
- `meets_threshold(&self, threshold) -> bool`

**Example**:
```rust
let triple = SemanticTriple::new_resource(
    "film:inception",
    "media:hasGenre",
    "genre:SciFi",
).with_confidence(0.95);
```

---

### `OWLAxiom`

Semantic constraints and rules.

```rust
pub struct OWLAxiom {
    pub axiom_type: AxiomType,
    pub components: Vec<String>,
    pub description: String,
}
```

**Axiom Types**:
- `SubClassOf` - A is subclass of B
- `EquivalentClasses` - A and B are equivalent
- `DisjointClasses` - A and B are disjoint
- `TransitiveProperty` - P is transitive
- `SymmetricProperty` - P is symmetric
- `InverseProperties` - P and Q are inverses

**Methods**:

- `subclass_of(subclass, superclass) -> Self`
- `transitive_property(property) -> Self`

---

### `SemanticConstraint`

IF-THEN rules for embeddings.

```rust
pub struct SemanticConstraint {
    pub id: String,
    pub condition: ConstraintCondition,
    pub action: ConstraintAction,
    pub weight: f32,
}
```

**Conditions**:
- `HasProperty { property, value }` - Has specific property
- `SimilarTo { target, threshold }` - Above similarity threshold
- `And(Vec<Condition>)` - All conditions must hold
- `Or(Vec<Condition>)` - Any condition must hold

**Actions**:
- `AdjustEmbedding { direction, magnitude }` - Modify embedding
- `AddTriple(SemanticTriple)` - Add semantic fact
- `ModifyScore { delta }` - Boost/penalize score

---

## User Models

### `UserProfile`

Complete user representation.

```rust
pub struct UserProfile {
    pub user_id: UserId,
    pub user_embedding: Vec<f32>,           // 1024-dim
    pub watch_history: Vec<Interaction>,    // Last 100
    pub preferences: UserPreferences,
    pub current_state: Option<PsychographicState>,
    pub taste_cluster: Option<TasteCluster>,
    pub tolerances: ToleranceLevels,
    pub metadata: UserMetadata,
    pub last_updated: DateTime<Utc>,
}
```

**Methods**:

- `new(user_id) -> Self` - Create new profile
- `update_embedding(&mut self, embeddings)` - Update from interactions
- `add_interaction(&mut self, interaction)` - Add to history
- `avg_completion_rate() -> f32` - Average watch completion

---

### `Interaction`

User interaction record.

```rust
pub struct Interaction {
    pub content_id: String,
    pub interaction_type: InteractionType,
    pub timestamp: DateTime<Utc>,
    pub watch_duration: Option<u32>,
    pub content_duration: Option<u32>,
    pub watch_completion_rate: Option<f32>,
    pub rating: Option<u8>,
    pub device: DeviceType,
    pub context: Option<ViewingContext>,
}
```

**Interaction Types**:
- `Click` - User clicked
- `Watch` - User started watching
- `Skip` - User abandoned
- `Complete` - User completed
- `Rate` - User rated
- `Watchlist` - Added to watchlist

---

### `PsychographicState`

Current mood/mindset.

```rust
pub struct PsychographicState {
    pub state: StateType,
    pub intensity: f32,                // 0.0-1.0
    pub duration: Duration,
    pub state_embedding: Option<Vec<f32>>,  // 128-dim
    pub detected_at: DateTime<Utc>,
}
```

**State Types**:
- `SeekingComfort` - Seeking familiarity
- `SeekingChallenge` - Seeking mental stimulation
- `Nostalgic` - Nostalgic mood
- `Energetic` - Excited state
- `Relaxed` - Winding down
- `Stressed` - Anxious state
- `Social` - Social viewing
- `Focused` - Solo focused

**Methods**:

- `new(state, intensity) -> Self` - Create state
- `is_current() -> bool` - Check if still valid (<2 hours)

---

### `TasteCluster`

Collaborative filtering group.

```rust
pub struct TasteCluster {
    pub cluster_id: u32,
    pub centroid: Vec<f32>,           // 1024-dim
    pub size: usize,
    pub characteristics: Vec<String>,
    pub typical_genres: Vec<String>,
}
```

**Methods**:

- `new(cluster_id, centroid) -> Self` - Create cluster
- `distance_from_centroid(&self, embedding) -> f32` - Compute distance

---

### `ViewingContext`

Current viewing situation.

```rust
pub struct ViewingContext {
    pub time_of_day: TimeOfDay,       // Morning/Afternoon/Evening/Night
    pub day_of_week: Weekday,
    pub device: DeviceType,           // Mobile/Tablet/Desktop/TV
    pub network_speed: Option<f32>,   // Mbps
    pub location: Option<String>,
    pub social_setting: SocialSetting, // Solo/DateNight/Family
    pub ambient: AmbientConditions,
}
```

---

## Recommendation Models

### `Recommendation`

Recommendation result with score and explanation.

```rust
pub struct Recommendation {
    pub content: MediaContent,
    pub score: RecommendationScore,
    pub explanation: String,
    pub semantic_path: Option<SemanticPath>,
    pub ranking_factors: RankingFactors,
    pub rank: usize,
    pub generated_at: DateTime<Utc>,
}
```

**Methods**:

- `new(content, score, explanation) -> Self` - Create recommendation
- `with_path(self, path) -> Self` - Set semantic path
- `with_factors(self, factors) -> Self` - Set ranking factors
- `with_rank(self, rank) -> Self` - Set position
- `is_high_confidence() -> bool` - Check if confidence >0.8

---

### `RecommendationScore`

Multi-component scoring.

```rust
pub struct RecommendationScore {
    pub total: f32,              // 0.0-1.0
    pub relevance: f32,          // Semantic match
    pub personalization: f32,    // User fit
    pub quality: f32,            // Content quality
    pub diversity: f32,          // Serendipity
    pub confidence: f32,         // Prediction certainty
}
```

**Methods**:

- `new(total) -> Self` - Create with total score
- `from_components(relevance, personalization, quality, diversity) -> Self`
- `with_confidence(self, confidence) -> Self`

**Default Weights**: 0.4 relevance + 0.4 personalization + 0.15 quality + 0.05 diversity

---

### `SemanticPath`

Explanation trail from user to content.

```rust
pub struct SemanticPath {
    pub nodes: Vec<PathNode>,
    pub length: usize,
    pub strength: f32,           // Weakest link
    pub path_type: PathType,
}
```

**Path Types**:
- `DirectSimilarity` - Content similarity
- `GenrePath` - Genre-based
- `AestheticPath` - Visual aesthetic
- `NarrativePath` - Narrative structure
- `CollaborativePath` - Collaborative filtering
- `HybridPath` - Multi-hop path

**Methods**:

- `new(nodes, path_type) -> Self` - Create path
- `explain() -> String` - Generate explanation
- `is_strong() -> bool` - Check if strength >0.7

---

### `RankingFactors`

Detailed score breakdown.

```rust
pub struct RankingFactors {
    pub vector_similarity: f32,
    pub graph_distance: f32,
    pub mf_score: f32,                 // Matrix factorization
    pub neural_score: f32,
    pub llm_score: Option<f32>,
    pub rl_score: f32,                 // Reinforcement learning
    pub temporal_boost: f32,
    pub cultural_boost: f32,
    pub reasons: Vec<ExplanationReason>,
}
```

**Methods**:

- `add_reason(&mut self, reason)` - Add explanation
- `top_reasons(&self, n) -> Vec<&ExplanationReason>` - Get top N reasons

---

### `ExplanationReason`

Human-readable explanation.

```rust
pub struct ExplanationReason {
    pub reason_type: ReasonType,
    pub explanation: String,
    pub importance: f32,
    pub evidence: Option<String>,
}
```

**Reason Types**:
- `SimilarTo` - Similar to watched content
- `MatchesPreferences` - Matches user preferences
- `PopularInCluster` - Popular in taste cluster
- `Trending` - Trending globally
- `CulturallyRelevant` - Culturally relevant
- `DiversityPick` - Fills diversity gap
- `CriticalAcclaim` - High critical acclaim
- `AwardWinning` - Award-winning
- `MatchesMood` - Matches current mood
- `CollaborativeFiltering` - Recommended by similar users

---

## GPU Types

### `GPUEmbedding`

GPU-aligned embedding (32-byte aligned).

```rust
#[repr(C, align(32))]
pub struct GPUEmbedding {
    pub dims: u32,
    pub data: Vec<f32>,
}
```

**Methods**:

- `new(data) -> Self` - Create aligned embedding
- `as_ptr() -> *const f32` - Get data pointer
- `is_aligned() -> bool` - Check alignment
- `size_bytes() -> usize` - Size in bytes

---

### `GPUBatch<T>`

Batch container for parallel processing.

```rust
pub struct GPUBatch<T> {
    pub items: Vec<T>,
    pub size: usize,
    pub capacity: usize,
    pub batch_id: u64,
}
```

**Methods**:

- `with_capacity(capacity) -> Self` - Create batch
- `push(&mut self, item) -> Result<(), String>` - Add item
- `is_full() -> bool` - Check if full
- `is_empty() -> bool` - Check if empty
- `clear(&mut self)` - Clear batch
- `fill_rate() -> f32` - Get fill rate (0.0-1.0)

---

### `GPUEmbeddingBatch`

Flattened embeddings for GPU transfer.

```rust
#[repr(C)]
pub struct GPUEmbeddingBatch {
    pub count: u32,
    pub dims: u32,
    pub data: Vec<f32>,    // [count × dims]
}
```

**Methods**:

- `from_embeddings(embeddings) -> Self` - Create from vectors
- `get(&self, index) -> Option<&[f32]>` - Get embedding at index
- `size_bytes() -> usize` - Total size
- `fits_in_gpu(&self, gpu_memory) -> bool` - Check GPU fit

---

### GPU Kernel Inputs

#### `SemanticForcesInput`

Input for `semantic_forces.cu` kernel.

```rust
#[repr(C)]
pub struct SemanticForcesInput {
    pub n: u32,                      // Number of items
    pub dims: u32,                   // Embedding dimensions
    pub embeddings: Vec<f32>,        // [n × dims]
    pub color_vectors: Vec<f32>,     // [n × 64]
    pub forces: Vec<f32>,            // [n × n] output
}
```

#### `OntologyConstraintsInput`

Input for `ontology_constraints.cu` kernel.

```rust
#[repr(C)]
pub struct OntologyConstraintsInput {
    pub n: u32,                          // Number of items
    pub m: u32,                          // Number of constraints
    pub dims: u32,
    pub embeddings: Vec<f32>,            // [n × dims]
    pub constraint_graph: Vec<i32>,      // [m × 3]
    pub constraint_weights: Vec<f32>,    // [m]
}
```

#### `TensorFusionInput`

Input for GPU tensor fusion kernel.

```rust
#[repr(C)]
pub struct TensorFusionInput {
    pub batch_size: u32,
    pub visual_dims: u32,        // 768
    pub audio_dims: u32,         // 512
    pub text_dims: u32,          // 1024
    pub output_dims: u32,        // 1024
    pub visual: Vec<f32>,
    pub audio: Vec<f32>,
    pub text: Vec<f32>,
    pub weights: [f32; 3],
    pub output: Vec<f32>,
}
```

---

### `GPUBatchStats`

Batch processing statistics.

```rust
pub struct GPUBatchStats {
    pub total_batches: u64,
    pub total_items: u64,
    pub avg_batch_size: f32,
    pub avg_batch_time_ms: f32,
    pub gpu_utilization: f32,
    pub memory_usage_bytes: usize,
    pub throughput: f32,           // Items/sec
}
```

**Methods**:

- `new() -> Self` - Create stats tracker
- `update(&mut self, batch_size, processing_time)` - Update with new batch

---

**Last Updated**: 2025-12-04
**Version**: 1.0
