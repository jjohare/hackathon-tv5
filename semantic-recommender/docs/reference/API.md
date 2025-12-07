# API Reference

Complete specification for REST API, MCP server interfaces, data models, and adaptive SSSP functionality.

**Version**: 1.2.0
**Last Updated**: 2025-12-07

---

## Table of Contents

1. [REST API Overview](#rest-api-overview)
2. [REST API Endpoints](#rest-api-endpoints)
3. [Data Models & Types](#data-models--types)
4. [Adaptive SSSP API](#adaptive-sssp-api)
5. [MCP Server Interface](#mcp-server-interface)
6. [Authentication & Rate Limiting](#authentication--rate-limiting)
7. [Error Handling](#error-handling)
8. [Code Examples](#code-examples)

---

## REST API Overview

**Base URL**: `http://localhost:8080/api/v1`

**Authentication**: All endpoints require `Authorization: Bearer <api_key>` header (development: optional)

**Response Format**: JSON

**Latency SLA**: P50 <15ms, P99 <50ms

### Performance SLAs

| Operation | P50 | P99 | Expected |
|-----------|-----|-----|----------|
| Search (100M entities) | 12ms | 45ms | <15ms |
| Batch Search (10 queries) | 120ms | 400ms | <150ms |
| Recommendation (cold-start) | 45ms | 150ms | <100ms |
| Ontology Query (depth 2) | 28ms | 100ms | <50ms |

---

## REST API Endpoints

### Health Check

**GET** `/health` (no auth required)

Check system status.

```bash
curl http://localhost:8080/api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.2.0",
  "components": {
    "gpu": "operational",
    "vector_db": "operational",
    "knowledge_graph": "operational"
  },
  "timestamp": "2025-12-07T10:30:00Z"
}
```

---

### Semantic Search

**POST** `/search`

Search for semantically similar media entities.

**Request**:
```json
{
  "query": "string",
  "query_type": "text|image|audio|video",
  "limit": 10,
  "threshold": 0.85,
  "filters": {
    "language": "fr",
    "genre": "Documentary",
    "year_min": 2020,
    "year_max": 2025
  },
  "execution_hint": "auto|gpu|vector_db"
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "doc_12345",
      "title": "Climat: l'Urgence d'Agir",
      "description": "Documentary on climate change",
      "similarity": 0.94,
      "metadata": {
        "language": "fr",
        "genre": "Documentary",
        "duration": 52,
        "year": 2023,
        "provider": "TV5Monde"
      },
      "explanation": {
        "matched_fields": ["title", "description"],
        "related_entities": ["Environment", "Climate"]
      }
    }
  ],
  "query_time_ms": 12,
  "total_entities_searched": 100000000,
  "metadata": {
    "execution_path": "gpu",
    "gpu_utilization": 0.92,
    "cache_hit_rate": 0.85
  }
}
```

---

### Batch Search

**POST** `/batch-search`

Execute multiple queries in parallel.

**Request**:
```json
{
  "queries": [
    "French documentary climate change",
    "Spanish thriller series",
    "Japanese anime movies"
  ],
  "limit": 10,
  "threshold": 0.85
}
```

**Response**:
```json
{
  "results": [
    {
      "query_index": 0,
      "query": "French documentary climate change",
      "results": [],
      "query_time_ms": 12
    }
  ],
  "batch_time_ms": 45,
  "metadata": {
    "parallel_execution": true,
    "total_queries": 3,
    "avg_query_time_ms": 13
  }
}
```

---

### Recommendations

**POST** `/recommend`

Generate personalized recommendations for a user.

**Request**:
```json
{
  "user_id": "user_abc123",
  "limit": 10,
  "context": {
    "last_watched": ["doc_123", "doc_456"],
    "preferences": {
      "genres": ["Documentary", "Drama"],
      "languages": ["fr", "en"],
      "max_duration": 120
    },
    "temporal": {
      "time_of_day": "evening",
      "day_of_week": "Friday"
    }
  },
  "exploration_rate": 0.1
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "rank": 1,
      "id": "doc_789",
      "title": "La Nature Sauvage",
      "score": 0.92,
      "reason": "Similar to 'Climat: l'Urgence d'Agir'",
      "confidence": 0.85,
      "cold_start": false
    }
  ],
  "query_time_ms": 45,
  "metadata": {
    "algorithm": "thompson_sampling",
    "interactions_seen": 127,
    "convergence": 0.92
  }
}
```

---

### Ontology Query

**POST** `/ontology/query`

Traverse and reason over the knowledge graph.

**Request**:
```json
{
  "entity": "Documentary",
  "depth": 2,
  "direction": "outbound|inbound|both",
  "edge_types": ["subClassOf", "hasGenre"],
  "limit": 50
}
```

**Response**:
```json
{
  "entity": "Documentary",
  "paths": [
    {
      "path": ["Documentary", "NonFiction", "Content"],
      "relations": ["subClassOf", "subClassOf"],
      "confidence": 0.98
    }
  ],
  "related_entities": [
    {
      "name": "Drama",
      "relation": "hasGenre",
      "confidence": 0.92
    }
  ],
  "inferred_properties": {
    "is_factual": true,
    "typical_duration": 60,
    "typical_languages": ["en", "fr", "de"]
  },
  "query_time_ms": 28
}
```

---

### Get Similar

**GET** `/similar/<entity_id>`

Find entities similar to a known entity.

**Request**:
```bash
curl "http://localhost:8080/api/v1/similar/doc_12345?limit=10&threshold=0.80"
```

**Response**:
```json
{
  "query_id": "doc_12345",
  "similar": [
    {
      "id": "doc_12346",
      "similarity": 0.91,
      "title": "La Planète Brûle"
    }
  ],
  "metadata": {
    "query_time_ms": 10
  }
}
```

---

### Statistics

**GET** `/stats` (no auth required)

System performance and usage statistics.

**Response**:
```json
{
  "uptime_seconds": 86400,
  "total_queries": 1234567,
  "queries_per_second": 1250,
  "average_latency_ms": 12.5,
  "gpu": {
    "utilization": 0.92,
    "memory_used_gb": 14.2,
    "temperature_celsius": 65
  },
  "cache": {
    "hit_rate": 0.85,
    "size_mb": 2048
  },
  "entities": {
    "total": 100000000,
    "indexed": 99999999
  }
}
```

---

## Data Models & Types

### Content Models

#### MediaContent

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

#### ContentType

```rust
pub enum ContentType {
    Film,
    Series,
    Episode,
    ShortFilm,
    Documentary,
    Miniseries,
}
```

#### Genre

```rust
pub enum Genre {
    Action, Adventure, Animation, Biography, Comedy, Crime,
    Documentary, Drama, Family, Fantasy, Horror, Mystery,
    Romance, SciFi, Thriller, War, Western,
    Custom(String),
}
```

#### VisualAesthetic

```rust
pub enum VisualAesthetic {
    Noir,
    Neon,
    Pastel,
    Desaturated,
    Naturalistic,
    Vibrant,
}
```

#### NarrativeStructure

```rust
pub enum NarrativeStructure {
    Linear,
    NonLinear,
    HerosJourney,
    EnsembleCast,
    Circular,
    FrameStory,
}
```

---

### Embedding Types

#### EmbeddingVector

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
- `new(data, model) -> Self`
- `normalize(&mut self)`
- `is_normalized() -> bool`
- `cosine_similarity(&self, other) -> f32`
- `euclidean_distance(&self, other) -> f32`

#### MultiModalEmbedding

```rust
pub struct MultiModalEmbedding {
    pub unified: EmbeddingVector,
    pub visual: Option<VisualEmbedding>,
    pub audio: Option<AudioEmbedding>,
    pub text: Option<TextEmbedding>,
    pub fusion_weights: FusionWeights,
    pub quality_score: f32,
}
```

**Methods**:
- `fuse(visual, audio, text, weights) -> Self`
- `is_complete() -> bool`

---

### User Models

#### UserProfile

```rust
pub struct UserProfile {
    pub user_id: UserId,
    pub user_embedding: Vec<f32>,
    pub watch_history: Vec<Interaction>,
    pub preferences: UserPreferences,
    pub current_state: Option<PsychographicState>,
    pub taste_cluster: Option<TasteCluster>,
    pub tolerances: ToleranceLevels,
    pub metadata: UserMetadata,
    pub last_updated: DateTime<Utc>,
}
```

#### Interaction

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
- `Click`, `Watch`, `Skip`, `Complete`, `Rate`, `Watchlist`

---

### Recommendation Models

#### Recommendation

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

#### RecommendationScore

```rust
pub struct RecommendationScore {
    pub total: f32,
    pub relevance: f32,
    pub personalization: f32,
    pub quality: f32,
    pub diversity: f32,
    pub confidence: f32,
}
```

**Default Weights**: 0.4 relevance + 0.4 personalization + 0.15 quality + 0.05 diversity

---

## Adaptive SSSP API

### Configuration

#### AlgorithmMode

```rust
pub enum AlgorithmMode {
    Auto,
    GpuDijkstra,
    LandmarkApsp,
    Duan,
}
```

#### AdaptiveSsspConfig

```rust
pub struct AdaptiveSsspConfig {
    pub mode: AlgorithmMode,
    pub landmark_count: usize,
    pub large_graph_threshold: usize,
    pub collect_metrics: bool,
}
```

**Default Configuration**:
```rust
let config = AdaptiveSsspConfig::default();
```

**Custom Configuration**:
```rust
let config = AdaptiveSsspConfig {
    mode: AlgorithmMode::Auto,
    landmark_count: 64,
    large_graph_threshold: 5_000_000,
    collect_metrics: true,
};
```

#### SsspMetrics

```rust
pub struct SsspMetrics {
    pub algorithm_used: String,
    pub total_time_ms: f32,
    pub gpu_time_ms: Option<f32>,
    pub nodes_processed: usize,
    pub edges_relaxed: usize,
    pub landmarks_used: Option<usize>,
    pub complexity_factor: Option<f32>,
}
```

---

### RecommendationEngine SSSP Integration

#### Constructor with SSSP Config

```rust
pub async fn with_sssp_config(
    embeddings: Vec<f32>,
    embedding_dim: usize,
    metadata: Vec<ContentMetadata>,
    sssp_config: AdaptiveSsspConfig,
) -> Result<Self>
```

**Example**:
```rust
let config = AdaptiveSsspConfig {
    mode: AlgorithmMode::Auto,
    landmark_count: 32,
    large_graph_threshold: 10_000_000,
    collect_metrics: true,
};

let engine = RecommendationEngine::with_sssp_config(
    embeddings,
    1024,
    metadata,
    config,
).await?;
```

#### Get Metrics

```rust
pub async fn get_sssp_metrics(&self) -> Option<SsspMetrics>
```

**Example**:
```rust
if let Some(metrics) = engine.get_sssp_metrics().await {
    println!("Algorithm: {}", metrics.algorithm_used);
    println!("Latency: {:.2}ms", metrics.total_time_ms);
}
```

---

### GpuSemanticEngine API

#### Find Shortest Paths (Auto-Select)

```rust
pub async fn find_shortest_paths(
    &self,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
) -> GpuResult<Vec<Path>>
```

#### Find Shortest Paths (Explicit Algorithm)

```rust
pub async fn find_shortest_paths_with_algorithm(
    &self,
    graph: &[u32],
    sources: &[u32],
    targets: &[u32],
    config: &PathfindingConfig,
    algorithm: Option<AlgorithmMode>,
) -> GpuResult<Vec<Path>>
```

**Example**:
```rust
// Force GPU Dijkstra
let paths = engine.find_shortest_paths_with_algorithm(
    &graph_data,
    &sources,
    &targets,
    &PathfindingConfig::default(),
    Some(AlgorithmMode::GpuDijkstra),
).await?;
```

---

### Algorithm Selection Guidelines

| Graph Size | Queries/sec | Recommended Algorithm |
|-----------|-------------|----------------------|
| <100K nodes | Any | GPU Dijkstra |
| 100K-1M nodes | <100 | GPU Dijkstra |
| 100K-1M nodes | >100 | Landmark APSP |
| 1M-10M nodes | <50 | GPU Dijkstra |
| 1M-10M nodes | >50 | Landmark APSP |
| >10M nodes | Any | Landmark APSP |

### Expected Latencies

```
Algorithm         | 10K nodes | 1M nodes | 10M nodes | 100M nodes
-----------------|-----------|----------|-----------|------------
GPU Dijkstra     | 0.5ms     | 15ms     | 180ms     | 2000ms
Landmark APSP    | 2ms       | 25ms     | 120ms     | 500ms
Duan (future)    | 0.4ms     | 12ms     | 100ms     | 400ms
```

---

## MCP Server Interface

**Start**: `cargo run --release --bin mcp-server`

**Protocol**: JSON-RPC 2.0 over stdio or SSE

### Available Tools

#### semantic_search

```json
{
  "query": "French documentary",
  "query_type": "text",
  "limit": 10,
  "filters": {}
}
```

#### batch_search

```json
{
  "queries": ["query1", "query2"],
  "limit": 10
}
```

#### recommend

```json
{
  "user_id": "user_123",
  "context": {},
  "limit": 10
}
```

#### ontology_query

```json
{
  "entity": "Documentary",
  "depth": 2,
  "direction": "outbound"
}
```

#### get_similar

```json
{
  "entity_id": "doc_12345",
  "limit": 10
}
```

---

## Authentication & Rate Limiting

### Rate Limiting

**Default**: 1000 requests/second per API key

**Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1670323800
```

### API Key Authentication

```bash
Authorization: Bearer ag_xxxxxxxxxxxxxxxx
```

### OAuth 2.0

```bash
curl -X POST http://localhost:8080/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=...&client_secret=..."
```

---

## Error Handling

### Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `INVALID_REQUEST` | 400 | Bad request syntax |
| `INVALID_QUERY` | 400 | Query validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `GPU_UNAVAILABLE` | 503 | GPU not responding |
| `DB_UNAVAILABLE` | 503 | Vector DB offline |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

**Error Response**:
```json
{
  "error": "rate_limit_exceeded",
  "message": "Exceeded 1000 requests/second limit",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after_seconds": 60
}
```

---

## Code Examples

### cURL

```bash
# Simple search
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "documentary", "limit": 5}'

# Batch search
curl -X POST http://localhost:8080/api/v1/batch-search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["doc", "thriller"], "limit": 5}'

# With filters
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change",
    "filters": {"language": "fr", "year_min": 2020},
    "limit": 10
  }'
```

### Python

```python
import requests

api_url = "http://localhost:8080/api/v1"

# Search
response = requests.post(f"{api_url}/search", json={
    "query": "French documentary",
    "limit": 10
})
results = response.json()["results"]

# Recommend
response = requests.post(f"{api_url}/recommend", json={
    "user_id": "user_123",
    "limit": 10
})
recommendations = response.json()["recommendations"]
```

### JavaScript/TypeScript

```typescript
const api = "http://localhost:8080/api/v1";

// Search
const response = await fetch(`${api}/search`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "French documentary",
    limit: 10
  })
});

const { results } = await response.json();
```

### Rust (Complete Example with SSSP)

```rust
use crate::adaptive_sssp::{AdaptiveSsspConfig, AlgorithmMode};
use crate::semantic_search::unified_engine::RecommendationEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create engine with custom SSSP config
    let sssp_config = AdaptiveSsspConfig {
        mode: AlgorithmMode::Auto,
        landmark_count: 64,
        large_graph_threshold: 5_000_000,
        collect_metrics: true,
    };

    let engine = RecommendationEngine::with_sssp_config(
        embeddings,
        1024,
        metadata,
        sssp_config,
    ).await?;

    // Get recommendations
    let recommendations = engine.recommend(
        "user123",
        &user_context,
        10,
    ).await?;

    // Check SSSP metrics
    if let Some(metrics) = engine.get_sssp_metrics().await {
        println!("Algorithm: {}", metrics.algorithm_used);
        println!("Latency: {:.2}ms", metrics.total_time_ms);
    }

    Ok(())
}
```

---

## Request/Response Specification

### Common Fields

#### Filters Object
```json
{
  "language": "fr",
  "genre": "Documentary",
  "year_min": 2020,
  "year_max": 2025,
  "duration_min": 0,
  "duration_max": 120,
  "provider": "TV5Monde",
  "content_type": "video|audio|text|image",
  "rating_min": 7.0,
  "rating_max": 10.0
}
```

#### Metadata Object
```json
{
  "language": "fr",
  "genre": "Documentary",
  "duration": 52,
  "year": 2023,
  "provider": "TV5Monde",
  "rating": 8.5,
  "views": 1234567,
  "verified": true
}
```

---

## Best Practices

### SSSP Configuration
1. **Use Auto mode by default** - Let the system choose
2. **Monitor metrics** - Track algorithm selection and performance
3. **Update graph stats** - Keep statistics current for optimal selection
4. **Benchmark before forcing** - Only override if you have data
5. **Enable metrics collection** - Essential for performance tuning

### API Usage
1. Use batch endpoints for multiple queries
2. Implement client-side rate limiting
3. Cache responses when appropriate
4. Handle errors gracefully with retry logic
5. Monitor latency metrics

---

**See Also**:
- [INTEGRATION.md](../INTEGRATION.md) - Integration patterns
- [GPU_ACCELERATION.md](../GPU_ACCELERATION.md) - GPU optimization
- [ONTOLOGY_GUIDE.md](../ONTOLOGY_GUIDE.md) - Knowledge graph usage

---

**API Version**: 1.2.0
**Status**: Production Ready
**Last Updated**: 2025-12-07
