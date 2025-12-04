# Data Models & Type Definitions

Complete implementation of production-ready data models for the GPU-Accelerated Semantic Recommendation Engine.

## Overview

This module provides comprehensive type definitions for all system components:

- **Content Models**: Media representation with multi-modal embeddings
- **Embedding Types**: Visual, audio, text, and fused embeddings
- **Ontology Models**: OWL classes, properties, and semantic reasoning
- **User Models**: Profiles, psychographics, and behavioral tracking
- **Recommendation Models**: Scoring, ranking, and explanation generation
- **GPU Types**: Memory-aligned structures for CUDA processing

## Module Structure

```
models/
├── mod.rs                  # Module entry point with re-exports
├── content.rs              # MediaContent, ContentId, Genre, etc.
├── embeddings.rs           # EmbeddingVector, VisualEmbedding, etc.
├── ontology.rs             # OWL classes, triples, axioms
├── user.rs                 # UserProfile, Interaction, PsychographicState
├── recommendation.rs       # Recommendation, Score, SemanticPath
└── gpu_types.rs            # GPU-aligned memory layouts
```

## Quick Start

### Create Media Content

```rust
use hackathon_tv5::models::{MediaContent, ContentId, ContentType, Genre};

let film = MediaContent::new(
    ContentId::new("film:inception"),
    ContentType::Film,
    "Inception".to_string(),
);

film.genres = vec![Genre::SciFi, Genre::Thriller];
film.unified_embedding = vec![0.5; 1024];
```

### Compute Similarity

```rust
let similarity = film1.similarity(&film2);
println!("Similarity: {:.3}", similarity);
```

### Generate Recommendation

```rust
use hackathon_tv5::models::{Recommendation, RecommendationScore};

let rec = Recommendation::new(
    film,
    0.85,
    "Similar to films you've enjoyed"
);
```

## Features

### Multi-Modal Embeddings

Supports fusion of visual (768-dim), audio (512-dim), and text (1024-dim) embeddings into unified 1024-dim representation:

```rust
let fused = MultiModalEmbedding::fuse(
    visual_embedding,
    audio_embedding,
    text_embedding,
    FusionWeights::balanced(),
);
```

### OWL Ontology Reasoning

GMC-O (Global Media & Context Ontology) with semantic reasoning:

```rust
let triple = SemanticTriple::new_resource(
    "film:inception",
    "media:hasGenre",
    "genre:SciFi",
).with_confidence(0.95);
```

### User Psychographics

Context-aware user modeling with mood detection:

```rust
let state = PsychographicState::new(
    StateType::Relaxed,
    0.7  // intensity
);

user.current_state = Some(state);
```

### GPU-Optimized Memory

32-byte aligned structures for optimal GPU transfer:

```rust
let gpu_embedding = GPUEmbedding::new(data);
assert!(gpu_embedding.is_aligned());
```

## Documentation

- **Getting Started**: `/src/docs/GETTING_STARTED.md`
- **API Reference**: `/src/docs/API_REFERENCE.md`
- **Performance Guide**: `/src/docs/PERFORMANCE.md`

## Examples

Complete working examples in `/src/examples/`:

1. `simple_similarity.rs` - Basic content similarity
2. `ontology_reasoning.rs` - OWL reasoning demo
3. `full_recommendation.rs` - Complete recommendation pipeline
4. `batch_processing.rs` - GPU batch processing

Run examples:
```bash
cargo run --example simple_similarity
cargo run --example ontology_reasoning
cargo run --example full_recommendation
cargo run --example batch_processing
```

## Testing

All models include comprehensive unit tests:

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test --lib models::content
cargo test --lib models::embeddings
cargo test --lib models::ontology
```

## Performance

Key performance characteristics:

- **Vector Similarity**: <1μs (1024-dim cosine similarity)
- **Embedding Normalization**: <10μs (1024-dim L2 norm)
- **GPU Batch Transfer**: <5ms (256 embeddings)
- **Memory Footprint**: 4KB per content item (1024-dim FP32)

## Design Principles

1. **Type Safety**: Leverage Rust's type system for correctness
2. **Zero-Copy**: Minimize allocations and copying
3. **GPU-Friendly**: Memory layouts optimized for CUDA
4. **Extensibility**: Easy to add new features and models
5. **Documentation**: Every public API is documented

## Integration

These models integrate with:

- **Vector Database**: RuVector HNSW indexing
- **Knowledge Graph**: Neo4j with GMC-O ontology
- **User Database**: ScyllaDB for profiles
- **GPU Pipeline**: CUDA kernels for processing
- **RL Engine**: AgentDB for personalization

## Version

**Version**: 1.0.0
**Last Updated**: 2025-12-04
**Rust Edition**: 2021
**MSRV**: 1.70+

## License

See project LICENSE file.

---

For detailed API documentation, see `/src/docs/API_REFERENCE.md`
