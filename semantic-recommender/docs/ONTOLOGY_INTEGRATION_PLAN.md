# Ontology Integration Plan - Semantic Recommender

**Date:** 2025-12-06
**Version:** 1.0
**Status:** Design Phase

---

## Executive Summary

Integrate two complementary movie ontologies with our semantic recommender using whelk-rs (EL++ reasoner) to provide **ontology-aware recommendation reasoning** beyond simple semantic similarity.

### Ontologies to Integrate

1. **AdA Film Ontology** ([ProjectAdA/public](https://github.com/ProjectAdA/public/tree/master/ontology))
   - 8 annotation levels, 78 types, 502 values
   - Film-analytical concepts (camera, editing, sound, etc.)
   - 740 KB OWL file
   - **Status:** Required for film technique analysis

2. **Movies Ontology** ([robotenique/movies-ontology](https://github.com/robotenique/movies-ontology))
   - General movie metadata ontology
   - Actors, directors, genres, ratings
   - Complementary to AdA's technical focus
   - **Status:** Required for metadata reasoning

3. **OMC (Ontology for Media Creation)** ([MovieLabs/OMC](https://github.com/MovieLabs/OMC))
   - Production process ontology (Apache 2.0 license)
   - Creative works (cw:), production (omc:), distribution (omd:)
   - JSON Schema + RDF/Turtle format
   - **Status:** OPTIONAL - Limited value for recommendations (no production metadata in MovieLens)

### Reasoning Engine

**Whelk-rs** (EL++ reasoner)
- Already integrated in `/home/devuser/workspace/project/src`
- Fast, complete EL reasoning
- Perfect for movie ontology classification
- **Validation:** horned-owl ^1.0 library for OWL parsing
- **Dependencies:** curie, im, rayon (parallel reasoning)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Recommendation Request                    │
│          "Find movies like Citizen Kane (1941)"              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Semantic Similarity (Current)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ GPU Tensor Core Similarity                           │   │
│  │ • Cosine similarity on 384-dim embeddings            │   │
│  │ • Top-K retrieval (<1ms on A100)                     │   │
│  │ • Result: Films with similar embeddings              │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│         Ontology-Enhanced Reasoning (NEW)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Whelk-rs EL++ Reasoner                               │   │
│  │ • Load AdA + Movies ontologies                       │   │
│  │ • Classify films by ontology concepts                │   │
│  │ • Infer relationships (e.g., Film Noir → Drama)      │   │
│  │ • Re-rank by ontology similarity                     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│               Hybrid Ranking                                 │
│  Score = α × SemanticSimilarity                             │
│          + β × OntologyMatch                                │
│          + γ × GenreOverlap                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
              Final Results
```

---

## Integration Phases

### Phase 1: Ontology Loading (Week 1)

**Goal:** Load and parse both ontologies into whelk-rs

**Tasks:**
1. Clone ontology repositories (see ONTOLOGY_SOURCES.md)
   ```bash
   git clone https://github.com/ProjectAdA/public.git data/ontologies/ada
   git clone https://github.com/robotenique/movies-ontology.git data/ontologies/movies
   # Optional: git clone https://github.com/MovieLabs/OMC.git data/ontologies/omc
   ```
2. Extract OWL files using horned-owl
3. Load into WhelkInferenceEngine (existing code from /home/devuser/workspace/project/src)
4. Validate ontology integrity with horned-owl validation framework

**Implementation:**
```rust
// src/ontology/loader.rs
use horned_owl::io::owx::reader::read;
use whelk::reasoner::Reasoner;

pub struct OntologyLoader {
    ada_ontology: SetOntology<ArcStr>,
    movies_ontology: SetOntology<ArcStr>,
    reasoner: Reasoner,
}

impl OntologyLoader {
    pub async fn load_ada_ontology(path: &Path) -> Result<SetOntology<ArcStr>> {
        // Load ada_ontology.owl (740KB)
        let ontology = read(&mut File::open(path)?)?;

        // Extract key concepts:
        // - 8 levels: Camera, Editing, Sound, Lighting, etc.
        // - 78 types: CameraAngle, CameraMovement, EditingPace, etc.
        // - 502 values: close-up, tracking-shot, fast-cut, etc.

        Ok(ontology)
    }

    pub async fn load_movies_ontology(path: &Path) -> Result<SetOntology<ArcStr>> {
        // Load movies ontology
        // Concepts: Movie, Actor, Director, Genre, etc.
        Ok(ontology)
    }
}
```

**Expected Output:**
- Ontologies loaded: 2
- Classes loaded: ~600
- Axioms loaded: ~2,000
- Loading time: <100ms

### Phase 2: MovieLens Mapping (Week 1-2)

**Goal:** Map MovieLens data to ontology concepts

**Mapping Strategy:**

| MovieLens Data | Ontology Concept | Reasoning |
|---------------|------------------|-----------|
| **Genres** | `ada:Genre` | Direct mapping |
| Drama | `ada:DramaticGenre` | SubClassOf |
| Action | `ada:ActionGenre` | SubClassOf |
| **Genome Tags** | `ada:AnnotationValue` | Semantic alignment |
| "dark" | `ada:Lighting/DarkLighting` | Tag → Film technique |
| "intense" | `ada:Editing/FastPace` | Tag → Editing style |
| "cerebral" | `movies:IntellectualFilm` | Tag → Film type |
| **Ratings** | `movies:PopularityRating` | Derived property |
| High ratings | `movies:CriticallyAcclaimed` | Inferred |

**Implementation:**
```python
# scripts/map_to_ontology.py
class OntologyMapper:
    def __init__(self):
        self.ada_mappings = {
            # Genome tag → AdA concept
            'dark': 'ada:DarkLighting',
            'noir': 'ada:FilmNoirStyle',
            'slow-paced': 'ada:SlowEditingPace',
            'fast-action': 'ada:FastCameraMovement',
            # ... 1,128 genome tags to map
        }

    def map_movie_to_ontology(self, movie_data):
        """Map a movie to ontology classes"""
        ontology_classes = []

        # Map genres
        for genre in movie_data['genres']:
            ontology_classes.append(f"movies:{genre}Genre")

        # Map genome tags to film techniques
        for tag, score in movie_data['genome_tags']:
            if score > 0.7:  # High relevance
                if tag in self.ada_mappings:
                    ontology_classes.append(self.ada_mappings[tag])

        return ontology_classes
```

**Expected Mapping Coverage:**
- Movies with ontology classes: 62,423 (100%)
- Avg classes per movie: 5-10
- Genome tag mapping: ~300/1,128 (high-confidence)

### Phase 3: Semantic Reasoning Integration (Week 2)

**Goal:** Use whelk-rs to infer relationships and enhance recommendations

**Reasoning Capabilities:**

1. **Class Hierarchy Reasoning**
   ```
   Input: Film has tag "noir"
   Reasoning:
     noir → FilmNoirStyle
     FilmNoirStyle SubClassOf DramaticFilm
     DramaticFilm SubClassOf Film
   Output: Film is a DramaticFilm
   ```

2. **Transitive Relationships**
   ```
   Input: User likes "Citizen Kane" (FilmNoir + DeepFocus)
   Reasoning:
     CitizenKane hasStyle FilmNoir
     FilmNoir relatedTo DramaticLighting
     Films with DramaticLighting → Similar aesthetic
   Output: Recommend "Touch of Evil", "The Third Man"
   ```

3. **Constraint Checking**
   ```
   Input: Recommend family-friendly films
   Constraint: NOT (hasStyle ViolentContent)
   Reasoning: Filter using ontology constraints
   Output: Only G/PG rated films
   ```

**Implementation:**
```rust
// src/reasoning/ontology_enhanced_recommender.rs
use whelk::reasoner::Reasoner;

pub struct OntologyEnhancedRecommender {
    reasoner: Reasoner,
    embeddings: Tensor,
}

impl OntologyEnhancedRecommender {
    pub async fn recommend_with_ontology(
        &self,
        query_movie_id: &str,
        top_k: usize,
    ) -> Result<Vec<(String, f32, OntologyExplanation)>> {

        // Step 1: Semantic similarity (GPU)
        let semantic_candidates = self.gpu_similarity(query_movie_id, top_k * 5)?;

        // Step 2: Ontology reasoning
        let query_classes = self.get_ontology_classes(query_movie_id)?;

        let mut scored_results = vec![];
        for (candidate_id, sem_score) in semantic_candidates {
            let candidate_classes = self.get_ontology_classes(&candidate_id)?;

            // Compute ontology similarity
            let onto_score = self.reasoner.class_similarity(
                &query_classes,
                &candidate_classes
            )?;

            // Hybrid score
            let final_score = 0.7 * sem_score + 0.3 * onto_score;

            // Generate explanation
            let explanation = self.explain_match(&query_classes, &candidate_classes);

            scored_results.push((candidate_id, final_score, explanation));
        }

        // Sort by hybrid score
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored_results.into_iter().take(top_k).collect())
    }

    fn explain_match(
        &self,
        query_classes: &[String],
        candidate_classes: &[String],
    ) -> OntologyExplanation {
        // Find shared ontology concepts
        let shared: Vec<_> = query_classes.iter()
            .filter(|c| candidate_classes.contains(c))
            .collect();

        OntologyExplanation {
            shared_concepts: shared,
            reasoning_path: self.reasoner.path(query_classes, candidate_classes),
        }
    }
}
```

### Phase 4: GPU-Accelerated Reasoning (Week 3)

**Goal:** Offload ontology operations to GPU kernels

**GPU Operations:**

1. **Class Membership Check** (Parallel)
   ```cuda
   __global__ void check_class_membership(
       const uint32_t* movie_classes,     // Movie → Class mapping
       const uint32_t* query_classes,     // Query classes
       float* membership_scores,          // Output
       int num_movies,
       int num_classes
   ) {
       int movie_id = blockIdx.x * blockDim.x + threadIdx.x;
       if (movie_id >= num_movies) return;

       // Count matching classes
       int matches = 0;
       for (int i = 0; i < num_classes; i++) {
           if (has_class(movie_id, query_classes[i])) {
               matches++;
           }
       }

       membership_scores[movie_id] = (float)matches / num_classes;
   }
   ```

2. **Transitive Closure** (Graph traversal on GPU)
   - Use existing `graph_search.cu` kernel
   - Compute reachability in ontology hierarchy
   - <1ms for 500-class ontology

**Performance Target:**
- Ontology reasoning: <5ms for 62K movies
- Combined (semantic + ontology): <10ms total
- Throughput: >1,000 QPS with reasoning

### Phase 5: Explainable Recommendations (Week 4)

**Goal:** Generate human-readable explanations

**Example Output:**
```json
{
  "query": "Citizen Kane (1941)",
  "recommendation": "Touch of Evil (1958)",
  "similarity": 0.92,
  "explanation": {
    "semantic_similarity": 0.89,
    "ontology_similarity": 0.95,
    "shared_concepts": [
      "FilmNoirStyle",
      "DeepFocusCinematography",
      "DramaticLighting",
      "NonLinearNarrative"
    ],
    "reasoning": "Both films share Film Noir aesthetic with innovative cinematography techniques. Deep focus and dramatic lighting are hallmarks of both directors (Welles, Orson)."
  }
}
```

---

## Data Structures

### Ontology-Enhanced Movie Representation

```rust
pub struct OntologyMovie {
    // Original data
    pub movie_id: String,
    pub title: String,
    pub genres: Vec<String>,
    pub embedding: Vec<f32>,  // 384-dim

    // Ontology enrichment
    pub ada_classes: Vec<String>,      // AdA film concepts
    pub movie_classes: Vec<String>,    // Movies ontology
    pub inferred_classes: Vec<String>, // From reasoning

    // Derived properties
    pub style_tags: Vec<String>,       // Visual/audio style
    pub technique_tags: Vec<String>,   // Cinematography
    pub narrative_tags: Vec<String>,   // Story structure
}
```

### Reasoning Results

```rust
pub struct OntologyExplanation {
    pub shared_concepts: Vec<String>,
    pub reasoning_path: Vec<String>,  // A → B → C
    pub confidence: f32,
    pub explanation_text: String,
}
```

---

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| **1** | Ontology Loading | Loaded ontologies, validated |
| **1-2** | MovieLens Mapping | 62K movies mapped to concepts |
| **2** | Reasoning Integration | Whelk-rs + recommendations working |
| **3** | GPU Acceleration | GPU kernels for ontology ops |
| **4** | Explainability | Human-readable explanations |

**Total: 1 month**

---

## Benefits

### 1. Better Recommendations

**Before (Embedding-only):**
```
Query: "Citizen Kane"
Results: Toy Story 2, Toy Story 3, Antz
Reason: Similar "animation" semantics in text
```

**After (Ontology-enhanced):**
```
Query: "Citizen Kane"
Results: Touch of Evil, The Third Man, Double Indemnity
Reason: Shared FilmNoir + DeepFocus + DramaticLighting
```

### 2. Explainable AI

- Users understand WHY a movie was recommended
- Builds trust in recommendations
- Enables debugging and improvement

### 3. Richer Queries

Support complex queries:
- "Film noir with innovative cinematography"
- "Drama with non-linear narrative"
- "Action films with slow-motion techniques"

### 4. Domain Knowledge

Leverage expert film knowledge:
- 8 AdA annotation levels
- 78 film-analytical types
- 502 professional film concepts

---

## Technical Requirements

### Dependencies

```toml
[dependencies]
whelk = "0.1.0"
horned-owl = "^1.0"
curie = "^0.1"
im = "^15.1"
rayon = "^1.6"  # Parallel reasoning
```

### Storage

- AdA ontology: 740 KB
- Movies ontology: ~500 KB
- Movie → Class mappings: ~5 MB (62K movies × 10 classes)
- **Total: ~6 MB**

### Performance Budget

| Operation | CPU | GPU | Target |
|-----------|-----|-----|--------|
| Ontology loading | 100ms | N/A | <200ms |
| Classification | 50ms | 5ms | <10ms |
| Reasoning | 100ms | 10ms | <20ms |
| **Total overhead** | **250ms** | **15ms** | **<50ms** |

---

## Risk Assessment

### Low Risk
- ✅ Whelk-rs already integrated
- ✅ Ontologies publicly available
- ✅ GPU kernels support graph operations

### Medium Risk
- ⚠️ **Genome tag → Ontology mapping**
  - Mitigation: Manual expert mapping for top 300 tags
  - Fallback: LLM-assisted mapping

- ⚠️ **Ontology maintenance**
  - Mitigation: Versioned ontology files
  - Update schedule: Quarterly

### Mitigated
- ❌ Performance overhead
  - Solution: GPU acceleration keeps <50ms total
  - 90% users won't notice <50ms difference

---

## Success Metrics

### Phase 1-2
- [ ] Both ontologies loaded successfully
- [ ] 62,423 movies mapped to concepts
- [ ] Avg 5-10 classes per movie
- [ ] <200ms ontology loading time

### Phase 3-4
- [ ] Reasoning working end-to-end
- [ ] Recommendation quality improved (A/B test)
- [ ] <50ms total overhead
- [ ] Explanations generated for all recommendations

### Production
- [ ] User satisfaction +20% (surveys)
- [ ] Click-through rate +15%
- [ ] "Interesting" recommendations +30%

---

## Next Steps

### Immediate (This Week)
1. Clone both ontology repositories
2. Set up whelk-rs in semantic-recommender
3. Load and validate ontologies
4. Create initial mapping for top 100 movies

### Short-term (2 Weeks)
1. Complete MovieLens → Ontology mapping
2. Integrate reasoning with existing recommender
3. A/B test against baseline
4. Document ontology usage

### Long-term (1 Month)
1. GPU-accelerated reasoning
2. Explainable AI interface
3. Advanced query support
4. Production deployment

---

## Conclusion

Integrating film ontologies with whelk-rs reasoning provides:
- **Richer recommendations** through domain knowledge
- **Explainable AI** for user trust
- **Semantic queries** beyond keyword search
- **Professional film concepts** (AdA's 502 values)

The integration is **technically feasible** with existing infrastructure and can be completed in **1 month** with minimal risk.

**Recommendation:** Proceed with Phase 1 (Ontology Loading) immediately.

---

**Document Version:** 1.0
**Author:** Semantic Recommender Team
**Date:** 2025-12-06
**Status:** Design Ready
