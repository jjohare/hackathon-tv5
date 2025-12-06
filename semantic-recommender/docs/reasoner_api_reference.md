# MediaReasoner API Reference

## ProductionMediaReasoner

Production-ready media ontology reasoner with caching, parallelization, and GPU hooks.

### Constructor

```rust
pub fn new() -> Self
```

Creates a new reasoner instance with default configuration.

**Example:**
```rust
let reasoner = ProductionMediaReasoner::new();
```

### Configuration

```rust
pub fn with_gpu(self, batch_size: usize) -> Self
```

Enables GPU acceleration with specified batch size.

**Parameters:**
- `batch_size`: Number of items to process in GPU batches (typically 1024-4096)

**Example:**
```rust
let reasoner = ProductionMediaReasoner::new()
    .with_gpu(2048);
```

## Core Methods

### Transitive Closure

```rust
fn compute_genre_closure(&mut self, ontology: &MediaOntology)
```

Computes transitive closure of genre hierarchy using parallel graph algorithms.

**Time Complexity:** O((V + E) × V) with parallelization
**Space Complexity:** O(V²)

**Performance:**
- 10K nodes: ~32ms
- 100K nodes: ~285ms

### Subgenre Checking

```rust
fn is_subgenre_of(&self, child: &str, parent: &str, ontology: &MediaOntology) -> bool
```

Checks if `child` is a subgenre of `parent` (including transitive relationships).

**Example:**
```rust
if reasoner.is_subgenre_of("PsychologicalThriller", "Drama", &ontology) {
    println!("PsychologicalThriller is under Drama hierarchy");
}
```

### Disjoint Genre Checking

```rust
fn are_disjoint_genres(&self, genre_a: &str, genre_b: &str, ontology: &MediaOntology) -> bool
```

Checks if two genres are mutually exclusive.

**Example:**
```rust
if reasoner.are_disjoint_genres("Comedy", "Horror", &ontology) {
    println!("These genres cannot coexist");
}
```

## Constraint Checking

### Circular Dependency Detection

```rust
pub fn detect_circular_dependencies(&mut self, ontology: &MediaOntology) -> Vec<ConstraintViolation>
```

Detects circular relationships in genre hierarchy using Tarjan's SCC algorithm.

**Returns:** Vector of violations with severity `Critical`

**Example:**
```rust
let violations = reasoner.detect_circular_dependencies(&ontology);
for v in violations {
    eprintln!("Circular: {}", v.explanation);
}
```

### Disjoint Violation Check

```rust
pub fn check_disjoint_violations(
    &self,
    media: &MediaEntity,
    ontology: &MediaOntology,
) -> Vec<ConstraintViolation>
```

Checks if media has conflicting disjoint genres.

**Returns:** Vector of violations with severity `Error`

**Example:**
```rust
let violations = reasoner.check_disjoint_violations(&media, &ontology);
if !violations.is_empty() {
    println!("Media has {} disjoint conflicts", violations.len());
}
```

### Paradox Detection

```rust
pub fn check_paradoxical_properties(&self, media: &MediaEntity) -> Vec<ConstraintViolation>
```

Detects contradictory property combinations.

**Detected Paradoxes:**
1. FamilyFriendly + Rated-R
2. Educational + Exploitation
3. Peaceful + Horror/Action

**Returns:** Vector of violations with severity `Warning`

### Comprehensive Check

```rust
pub fn check_all_constraints(&mut self, ontology: &MediaOntology) -> Vec<ConstraintViolation>
```

Runs all constraint checks and returns sorted list (by severity).

**Example:**
```rust
let mut reasoner = ProductionMediaReasoner::new();
let violations = reasoner.check_all_constraints(&ontology);

println!("Found {} violations:", violations.len());
for v in violations {
    println!("[{:?}] {}: {}", v.severity, v.violation_type, v.explanation);
}
```

## Cultural Matching

### Cultural Context Scoring

```rust
fn match_cultural_context(&self, media: &MediaEntity, context: &CulturalContext) -> f32
```

Scores media-context cultural fit using weighted factors.

**Returns:** Score between 0.0 and 1.0

**Scoring Weights:**
- Language match: 30%
- Regional match: 25%
- Theme overlap: 25%
- Regional preferences: 15%
- Taboo penalties: -50% per violation

**Example:**
```rust
let context = CulturalContext {
    region: "US".to_string(),
    language: "en-US".to_string(),
    cultural_themes: vec!["family".to_string()],
    taboos: vec!["violence".to_string()],
    preferences: HashMap::from([
        ("Drama".to_string(), 0.9),
    ]),
};

let score = reasoner.match_cultural_context(&media, &context);

match score {
    s if s > 0.8 => println!("Excellent cultural fit"),
    s if s > 0.6 => println!("Good cultural fit"),
    s if s > 0.4 => println!("Moderate cultural fit"),
    _ => println!("Poor cultural fit"),
}
```

## Mood Analysis

### Mood Similarity (VAD Model)

```rust
fn calculate_mood_similarity(&self, mood_a: &Mood, mood_b: &Mood) -> f32
```

Calculates similarity between moods using Valence-Arousal-Dominance model.

**Formula:**
```
distance = sqrt((V_a - V_b)² + (A_a - A_b)² + (D_a - D_b)²) / sqrt(3)
similarity = 1 - distance
```

**Example:**
```rust
let tense = Mood {
    name: "Tense".to_string(),
    valence: -0.3,  // Negative emotion
    arousal: 0.8,   // High energy
    dominance: 0.4, // Moderate control
    related_moods: vec![],
};

let anxious = Mood {
    name: "Anxious".to_string(),
    valence: -0.4,
    arousal: 0.7,
    dominance: 0.3,
    related_moods: vec![],
};

let similarity = reasoner.calculate_mood_similarity(&tense, &anxious);
// Returns ~0.85 (very similar)
```

### Mood Inference

```rust
fn infer_mood(&self, media: &MediaEntity, ontology: &MediaOntology) -> Vec<String>
```

Infers additional moods from existing moods and genre patterns.

## Recommendation

### User Recommendation

```rust
fn recommend_for_user(
    &self,
    user: &UserProfile,
    context: &DeliveryContext,
    ontology: &MediaOntology,
) -> Vec<RecommendationResult>
```

Generates ranked media recommendations for user.

**Scoring Factors:**
- Genre preference: 30%
- Mood alignment: 25%
- Cultural relevance: 20%
- Contextual fit: 15%
- History similarity: 10%

**Example:**
```rust
let recommendations = reasoner.recommend_for_user(&user, &context, &ontology);

for rec in recommendations.iter().take(10) {
    println!("{}: Score {:.2} (confidence: {:.2})",
        rec.media_id,
        rec.score,
        rec.confidence
    );

    for factor in &rec.reasoning {
        println!("  - {}: {}", factor.factor_type, factor.explanation);
    }
}
```

## Inference

### Full Axiom Inference

```rust
fn infer_axioms(&self, ontology: &MediaOntology) -> OntologyResult<Vec<InferredMediaAxiom>>
```

Infers all possible axioms from ontology.

**Inferred Types:**
- SubGenreOf (transitive)
- SimilarMood
- EquivalentGenre
- DisjointGenre

**Example:**
```rust
let inferred = reasoner.infer_axioms(&ontology)?;

let subgenre_axioms: Vec<_> = inferred.iter()
    .filter(|a| a.axiom_type == MediaAxiomType::SubGenreOf)
    .collect();

println!("Inferred {} subgenre relationships", subgenre_axioms.len());
```

## Data Structures

### ConstraintViolation

```rust
pub struct ConstraintViolation {
    pub violation_type: ViolationType,
    pub subject: String,
    pub object: Option<String>,
    pub severity: ViolationSeverity,
    pub explanation: String,
}
```

**ViolationType:**
- `DisjointGenreConflict`
- `CircularHierarchy`
- `CardinalityViolation`
- `DomainRangeViolation`
- `ParadoxicalProperty`
- `MutuallyExclusiveMood`

**ViolationSeverity:**
- `Warning`: Informational, may be acceptable
- `Error`: Should be fixed, affects quality
- `Critical`: Must be fixed, breaks consistency

### InferredMediaAxiom

```rust
pub struct InferredMediaAxiom {
    pub axiom_type: MediaAxiomType,
    pub subject: String,
    pub object: Option<String>,
    pub confidence: f32,
    pub reasoning: String,
}
```

### RecommendationResult

```rust
pub struct RecommendationResult {
    pub media_id: String,
    pub score: f32,
    pub reasoning: Vec<ReasoningFactor>,
    pub confidence: f32,
}

pub struct ReasoningFactor {
    pub factor_type: FactorType,
    pub weight: f32,
    pub explanation: String,
}
```

## Error Handling

### OntologyError

```rust
pub enum OntologyError {
    EntityNotFound(String),
    InvalidRelation(String),
    CircularDependency(String),
    ConstraintViolation(String),
    InferenceFailure(String),
}
```

**Example:**
```rust
match reasoner.infer_axioms(&ontology) {
    Ok(axioms) => process_axioms(axioms),
    Err(OntologyError::CircularDependency(msg)) => {
        eprintln!("Circular dependency detected: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Performance Tips

### 1. Cache Transitive Closure

```rust
// Compute once, reuse many times
reasoner.compute_genre_closure(&ontology);

// Now these are O(1) lookups
for child in children {
    if reasoner.is_subgenre_of(&child, "Drama", &ontology) {
        // ...
    }
}
```

### 2. Batch Cultural Matching

```rust
use rayon::prelude::*;

let scores: Vec<f32> = media_batch.par_iter()
    .map(|m| reasoner.match_cultural_context(m, &context))
    .collect();
```

### 3. Enable GPU for Large Batches

```rust
let reasoner = ProductionMediaReasoner::new()
    .with_gpu(4096);  // Best for >10K items

// GPU automatically used for:
// - Mood similarity matrices
// - Cultural matching batches
// - Embedding comparisons
```

### 4. Incremental Constraint Checking

```rust
// Check only new media
for new_media in &new_entities {
    let violations = reasoner.check_disjoint_violations(new_media, &ontology);
    violations.extend(reasoner.check_paradoxical_properties(new_media));
}
```

## Thread Safety

All reasoner operations are thread-safe:
- `DashMap` for concurrent cache access
- Immutable borrows for read operations
- Safe to share across threads with `Arc`

**Example:**
```rust
use std::sync::Arc;

let reasoner = Arc::new(ProductionMediaReasoner::new());

let handles: Vec<_> = media_chunks.iter().map(|chunk| {
    let reasoner = Arc::clone(&reasoner);
    thread::spawn(move || {
        chunk.iter()
            .map(|m| reasoner.match_cultural_context(m, &context))
            .collect::<Vec<_>>()
    })
}).collect();
```

## Feature Flags

```toml
[features]
default = []
gpu = ["cudarc"]  # Enable GPU acceleration
```

**Build with GPU:**
```bash
cargo build --features gpu
```

---

**Version:** 1.0.0
**Stability:** Production
**Thread Safety:** Yes
**GPU Support:** Optional (feature: `gpu`)
