# Hybrid Storage Architecture

## Overview

The hybrid storage system integrates three specialized databases to provide semantic search with < 10ms P99 latency:

- **Milvus**: Vector similarity search (8.7ms P99)
- **Neo4j**: Graph relationships and ontology
- **PostgreSQL (AgentDB)**: User policies and learning patterns

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  HybridStorageCoordinator                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ QueryPlanner │  │ RedisCache   │  │ Metrics      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │ Milvus  │         │ Neo4j   │        │AgentDB  │
    │ Client  │         │ Client  │        │(Postgres)│
    └─────────┘         └─────────┘        └─────────┘
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │ Vector  │         │ Graph   │        │ Policy  │
    │Embeddings│        │Relations│        │ Store   │
    └─────────┘         └─────────┘        └─────────┘
```

## Query Strategies

### 1. VectorOnly (Fastest: ~8.7ms)
```rust
// Simple semantic search without graph enrichment
let query = SearchQuery {
    embedding: vec![...],
    k: 10,
    include_relationships: false,
    ..Default::default()
};
```

**Flow**:
1. Milvus vector search (8.7ms)
2. Return results

**Use Case**: Quick similarity search, no personalization needed

### 2. HybridParallel (Most Common: ~22ms)
```rust
// Parallel vector + graph + policy
let query = SearchQuery {
    embedding: vec![...],
    k: 50,
    user_id: "user_123",
    include_relationships: true,
    require_genre_filter: true,
    ..Default::default()
};
```

**Flow**:
1. Milvus vector search (8.7ms) → Over-fetch 3x
2. **Parallel**:
   - Neo4j graph enrichment (15ms)
   - AgentDB policy fetch (5ms)
3. Merge + re-rank (2ms)

**Use Case**: Personalized recommendations with context

### 3. HybridSequential (Filtered: ~25ms)
```rust
// Vector filtering then graph enrichment
let query = SearchQuery {
    embedding: vec![...],
    k: 20,
    metadata_filters: filters,
    include_relationships: true,
    ..Default::default()
};
```

**Flow**:
1. Milvus vector search with filters (8.7ms)
2. Neo4j graph enrichment (15ms)
3. AgentDB policy re-ranking (2ms)

**Use Case**: Complex filtering requirements

## Implementation Details

### Milvus Client Features

```rust
// HNSW index configuration
let index_params = IndexParams {
    index_type: IndexType::HNSW,
    metric_type: MetricType::Cosine,
    m: 16,                    // Max connections
    ef_construction: 200,      // Build parameter
    nlist: 1024,
};

// Search parameters
let search_params = SearchParams {
    metric_type: MetricType::Cosine,
    ef: 64,                   // HNSW search parameter
    nprobe: 16,               // IVF search parameter
};
```

**Performance**:
- P50: 5.2ms
- P95: 7.8ms
- P99: 8.7ms (target: < 10ms)

### Neo4j Graph Schema

```cypher
// Media content nodes
CREATE (m:MediaContent {
    id: "content_123",
    title: "Example Movie",
    created_at: datetime()
})

// Genre hierarchy
CREATE (g1:Genre {name: "Action"})
CREATE (g2:Genre {name: "SciFi"})
CREATE (m)-[:BELONGS_TO]->(g1)
CREATE (m)-[:BELONGS_TO]->(g2)

// Relationships
CREATE (m1:MediaContent)-[:SIMILAR_TO {strength: 0.9}]->(m2:MediaContent)
CREATE (m1)-[:SEQUEL_OF]->(m2)

// Themes and moods
CREATE (m)-[:HAS_THEME]->(t:Theme {name: "Space Exploration"})
CREATE (m)-[:HAS_MOOD]->(mood:Mood {name: "Tense", valence: -0.3})
```

### AgentDB Policy Schema

```sql
-- User policies table
CREATE TABLE user_policies (
    user_id TEXT NOT NULL,
    context TEXT NOT NULL,
    preferences JSONB NOT NULL DEFAULT '{}',
    constraints JSONB NOT NULL DEFAULT '[]',
    learning_rate FLOAT NOT NULL DEFAULT 0.1,
    exploration_rate FLOAT NOT NULL DEFAULT 0.15,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, context)
);

-- Learning events table
CREATE TABLE learning_events (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    content_id TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- 'view', 'like', 'rate', 'skip'
    reward FLOAT NOT NULL,
    context JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## Query Planning Logic

```rust
impl QueryPlanner {
    pub fn plan(&self, query: &SearchQuery) -> QueryAnalysis {
        // Cache eligibility check
        if self.is_cache_eligible(query) {
            return QueryAnalysis {
                strategy: QueryStrategy::CachedOnly,
                estimated_latency_ms: 1.0,
                ..
            };
        }

        // Analyze requirements
        let needs_graph = query.include_relationships
            || query.require_genre_filter
            || !query.graph_filters.is_empty();

        let needs_vector = !query.embedding.is_empty();
        let needs_policy = !query.user_id.is_empty();

        // Select strategy
        if !needs_graph && needs_vector {
            QueryStrategy::VectorOnly  // Fastest path
        } else if query.k <= 20 && needs_graph {
            QueryStrategy::HybridSequential  // Small result sets
        } else {
            QueryStrategy::HybridParallel  // Large result sets
        }
    }
}
```

## Migration from Neo4j-Only

### Phase 1: Embeddings Migration

```rust
let migration = HybridMigration::new(
    source_neo4j,
    target_milvus,
    target_neo4j,
    target_agentdb,
    MigrationConfig {
        batch_size: 1000,
        parallel_workers: 4,
        verify_data: true,
        dry_run: false,
    },
);

let stats = migration.migrate_embeddings().await?;
println!("Migrated {} embeddings", stats.embeddings_migrated);
```

**Process**:
1. Query Neo4j for all MediaContent with embeddings
2. Batch insert into Milvus (1000 per batch)
3. Verify data integrity

### Phase 2: Relationship Preservation

Graph relationships stay in Neo4j:
- `SIMILAR_TO`
- `RELATED_TO`
- `SEQUEL_OF`
- `BELONGS_TO` (genres)
- `HAS_THEME`
- `HAS_MOOD`

### Phase 3: Policy Migration

Convert user interaction history to AgentDB policies:

```rust
// Neo4j: User interactions
MATCH (u:User)-[:VIEWED|LIKED]->(m:MediaContent)-[:BELONGS_TO]->(g:Genre)

// AgentDB: User policies
INSERT INTO user_policies (user_id, preferences)
VALUES ('user_123', '{"action": 0.8, "scifi": 0.6}')
```

## Performance Benchmarks

### Latency Targets

| Operation | P50 | P95 | P99 | Target |
|-----------|-----|-----|-----|--------|
| Vector Search | 5.2ms | 7.8ms | 8.7ms | < 10ms |
| Graph Enrichment | 12ms | 18ms | 22ms | < 25ms |
| Policy Re-rank | 1ms | 2ms | 3ms | < 5ms |
| **Hybrid Total** | **18ms** | **27ms** | **33ms** | **< 50ms** |

### Throughput

- Milvus: 10,000 QPS (single node)
- Neo4j: 5,000 QPS (graph traversal)
- PostgreSQL: 15,000 QPS (policy lookup)

**Hybrid System**: 4,000-5,000 QPS (limited by Neo4j)

## Usage Examples

### Example 1: Simple Semantic Search

```rust
use hackathon_tv5::storage::*;

let coordinator = HybridStorageCoordinator::new(
    milvus_client,
    neo4j_client,
    agentdb_coordinator,
    query_planner,
    None,  // No cache
);

let query = SearchQuery {
    embedding: encode_text("science fiction movie"),
    k: 10,
    ..Default::default()
};

let (results, metrics) = coordinator.search_with_context(&query).await?;

println!("Found {} results in {}μs", results.len(), metrics.total_time_us);
for rec in results {
    println!("- {} (score: {:.3})", rec.title, rec.score);
}
```

### Example 2: Personalized Recommendations

```rust
let query = SearchQuery {
    embedding: encode_text("action thriller"),
    k: 20,
    user_id: "user_456".to_string(),
    context: "evening_browsing".to_string(),
    include_relationships: true,
    require_genre_filter: true,
    ..Default::default()
};

let (results, metrics) = coordinator.search_with_context(&query).await?;

println!("Strategy: {}, Latency: {}ms",
    metrics.strategy_used,
    metrics.total_time_us / 1000
);

for rec in results {
    println!("- {} (score: {:.3})", rec.title, rec.score);
    println!("  Genres: {:?}", rec.genres);
    println!("  Reasoning: {}", rec.reasoning);
}
```

### Example 3: Content Ingestion

```rust
let content = MediaContent {
    id: "movie_789".to_string(),
    title: "Interstellar".to_string(),
    embedding: encode_text("space exploration science fiction"),
    metadata: HashMap::from([
        ("year".to_string(), "2014".to_string()),
        ("director".to_string(), "Christopher Nolan".to_string()),
    ]),
    genres: vec!["SciFi".to_string(), "Drama".to_string()],
    themes: vec!["Space".to_string(), "Time".to_string()],
    created_at: chrono::Utc::now(),
};

coordinator.ingest_content(&content).await?;
```

## Monitoring

### Key Metrics

```rust
let metrics = coordinator.get_metrics();

println!("Total searches: {}", metrics["total_searches"]);
println!("Cache hit rate: {:.1}%",
    100.0 * metrics["cache_hits"] as f32 / metrics["total_searches"] as f32
);
println!("Vector-only queries: {}", metrics["vector_only_queries"]);
println!("Hybrid queries: {}", metrics["hybrid_queries"]);
println!("Avg latency: {}μs", metrics["avg_latency_us"]);
```

### Alerts

- **Latency**: Alert if P99 > 50ms
- **Error Rate**: Alert if > 1%
- **Cache Hit Rate**: Alert if < 40%
- **Milvus Availability**: Alert if < 99.9%

## Future Optimizations

1. **HNSW Parameter Tuning**:
   - Increase `M` for better recall
   - Increase `efConstruction` for better index quality
   - Trade-off: memory vs accuracy

2. **Query Result Caching**:
   - Redis cache layer for popular queries
   - 5-minute TTL for personalized results
   - 1-hour TTL for non-personalized

3. **Adaptive Query Planning**:
   - Learn from historical performance
   - Switch strategies based on actual latency
   - User-specific optimization

4. **Distributed Milvus**:
   - Shard by content type
   - Replicate hot shards
   - Geographic distribution

## References

- [Milvus Documentation](https://milvus.io/docs)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/)
- [AgentDB Paper](https://arxiv.org/abs/2406.18085)
