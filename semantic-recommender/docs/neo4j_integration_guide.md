# Neo4j Integration Guide

## Overview

This guide covers the complete Neo4j integration for GMC-O (Generic Media Content Ontology) persistence, including bulk loading, schema setup, and incremental updates.

## Architecture

### Components

1. **OntologyLoader** (`src/rust/ontology/loader.rs`)
   - Connection pooling with configurable pool size
   - Retry logic with exponential backoff
   - Query timeout management
   - Batch operations for efficiency

2. **Schema Setup** (`scripts/setup_neo4j_schema.cypher`)
   - Constraints for data integrity
   - Indexes for query optimization
   - Full-text search capabilities

3. **Integration Tests** (`tests/neo4j_integration_tests.rs`)
   - Connection testing
   - Bulk load performance
   - Incremental updates
   - Concurrent operations

## Features

### 1. Bulk Loading Pipeline

**Performance Target**: 100K triples in <10 seconds

```rust
use hackathon_tv5::ontology::loader::{OntologyLoader, Neo4jConfig};

let config = Neo4jConfig {
    uri: "bolt://localhost:7687".to_string(),
    username: "neo4j".to_string(),
    password: "password".to_string(),
    database: "neo4j".to_string(),
    max_connections: 10,
    connection_timeout_secs: 10,
};

let loader = OntologyLoader::new(config).await?;

// Parse RDF/Turtle file
let triples = OntologyLoader::parse_ttl_file("ontology.ttl")?;

// Bulk load with batching (1000 nodes per transaction)
loader.bulk_load_triples(triples).await?;
```

**Batching Strategy**:
- Nodes: 1000 per batch via `UNWIND`
- Relationships: 1000 per batch via `UNWIND`
- Automatic type detection from RDF URIs
- Transaction-level atomicity

### 2. Schema Setup

Run the schema setup script before bulk loading:

```bash
# From Neo4j Browser or cypher-shell
cypher-shell -u neo4j -p password -f scripts/setup_neo4j_schema.cypher
```

**Created Constraints**:
- `media_id_unique`: Unique media IDs
- `genre_name_unique`: Unique genre names
- `mood_name_unique`: Unique mood names
- `user_id_unique`: Unique user IDs
- `cultural_context_unique`: Unique cultural regions

**Created Indexes**:
- Single-property indexes on frequently queried fields
- Composite indexes for multi-property queries
- Full-text search on title, themes, semantic tags

### 3. Incremental Updates

Use MERGE strategy for upserts:

```rust
use std::collections::HashMap;

let mut properties = HashMap::new();
properties.insert("title".to_string(), "Updated Title".to_string());
properties.insert("duration_seconds".to_string(), "7200".to_string());

loader.incremental_update("media:001", properties).await?;
```

**Features**:
- Automatic timestamp updates (`updated_at`)
- Property merging (existing properties preserved)
- Atomic operations

### 4. Loading Ontology

Load complete ontology structure:

```rust
let ontology = loader.load_ontology().await?;

println!("Loaded {} media entities", ontology.media.len());
println!("Genre hierarchy: {} entries", ontology.genre_hierarchy.len());
println!("Mood relations: {} entries", ontology.mood_relations.len());
```

**Loaded Data**:
- Media entities with genres, moods, themes
- Genre hierarchy (parent-child relationships)
- Mood relations with emotional dimensions
- Cultural contexts
- Media relationships (sequel, similar, etc.)
- Disjoint and equivalent genre sets
- Semantic tag hierarchy

### 5. Storing Inferred Axioms

Store reasoning results back to Neo4j:

```rust
use hackathon_tv5::ontology::reasoner::{InferredMediaAxiom, MediaAxiomType};

let axioms = vec![
    InferredMediaAxiom {
        axiom_type: MediaAxiomType::SubGenreOf,
        subject: "SciFi".to_string(),
        object: Some("Action".to_string()),
        confidence: 0.85,
        reasoning: "Inferred from co-occurrence patterns".to_string(),
    }
];

loader.store_inferred_axioms(&axioms).await?;
```

## RDF/Turtle Parsing

### Supported Formats

- Turtle (.ttl)
- RDF triple structure
- URI-based entity references
- Literal values with optional datatypes

### Example Ontology File

```turtle
@prefix media: <http://recommendation.org/ontology/media#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

media:movie001 rdf:type media:Media .
media:movie001 media:title "The Matrix" .
media:movie001 media:mediaType "Video" .
media:movie001 media:hasGenre media:SciFi .
media:movie001 media:hasGenre media:Action .
media:movie001 media:hasMood media:Intense .
media:movie001 media:durationSeconds "8160"^^xsd:integer .

media:SciFi rdf:type media:Genre .
media:SciFi media:subGenreOf media:Fiction .

media:Intense rdf:type media:Mood .
media:Intense media:valence "-0.2"^^xsd:float .
media:Intense media:arousal "0.9"^^xsd:float .
media:Intense media:dominance "0.7"^^xsd:float .
```

## Performance Optimization

### Connection Pooling

Configure pool size based on workload:

```rust
let config = Neo4jConfig {
    max_connections: 20, // For high-concurrency scenarios
    connection_timeout_secs: 15,
    ..Default::default()
};
```

### Batch Size Tuning

Adjust `BATCH_SIZE` constant in `loader.rs`:

```rust
const BATCH_SIZE: usize = 1000; // Default
// Increase for faster bulk loads: 5000-10000
// Decrease for memory-constrained environments: 100-500
```

### Query Performance

Monitor with Neo4j's query profiler:

```cypher
PROFILE MATCH (m:Media)-[:HAS_GENRE]->(g:Genre)
WHERE g.name = 'Action'
RETURN m, g
LIMIT 100;
```

## Error Handling

### Retry Logic

Automatic retries with exponential backoff:

- **Max Retries**: 3
- **Initial Delay**: 1000ms
- **Timeout**: 30 seconds per query

### Common Errors

1. **Connection Timeout**
   - Increase `connection_timeout_secs`
   - Check network connectivity
   - Verify Neo4j is running

2. **Query Timeout**
   - Reduce batch size
   - Optimize Cypher queries
   - Add missing indexes

3. **Constraint Violations**
   - Ensure unique IDs in source data
   - Check for duplicate entries
   - Run schema setup before loading

## Testing

### Running Integration Tests

**Prerequisites**:
- Neo4j instance running on localhost:7687
- Credentials: neo4j/password (or set via env vars)

```bash
# Set environment variables (optional)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"

# Run tests
cargo test --test neo4j_integration_tests -- --ignored
```

### Test Coverage

- `test_connection`: Basic connectivity
- `test_schema_setup`: Constraint/index creation
- `test_bulk_load_small_dataset`: Small data load
- `test_bulk_load_performance`: 30K triples load test
- `test_incremental_update`: MERGE operations
- `test_load_media_entities`: Ontology loading
- `test_load_genre_hierarchy`: Hierarchy queries
- `test_store_inferred_axioms`: Reasoning results
- `test_parse_ttl_file`: RDF parsing
- `test_concurrent_operations`: Parallel updates

## Production Deployment

### Docker Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.14-enterprise
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/production_password
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
      NEO4J_dbms_memory_heap_max__size: 4G
      NEO4J_dbms_memory_pagecache_size: 2G
      NEO4J_apoc_export_file_enabled: "true"
      NEO4J_apoc_import_file_enabled: "true"
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
      - ./scripts:/scripts

volumes:
  neo4j-data:
  neo4j-logs:
```

### Initial Data Load

```bash
# Start Neo4j
docker-compose up -d neo4j

# Wait for startup
sleep 10

# Setup schema
docker exec neo4j cypher-shell -u neo4j -p production_password \
  -f /scripts/setup_neo4j_schema.cypher

# Load ontology
cargo run --release --bin load_ontology -- \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password production_password \
  --file design/ontology/expanded-media-ontology.ttl
```

### Monitoring

```cypher
// Query statistics
CALL db.indexes() YIELD name, type, state, populationPercent
RETURN name, type, state, populationPercent
ORDER BY populationPercent;

// Data counts
MATCH (n) RETURN labels(n) AS label, count(n) AS count
ORDER BY count DESC;

// Relationship counts
MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count
ORDER BY count DESC;
```

## Troubleshooting

### Slow Bulk Loads

1. Disable constraints temporarily:
   ```cypher
   DROP CONSTRAINT media_id_unique;
   // Perform bulk load
   CREATE CONSTRAINT media_id_unique FOR (m:Media) REQUIRE m.id IS UNIQUE;
   ```

2. Use APOC procedures:
   ```cypher
   CALL apoc.periodic.iterate(
     "UNWIND $nodes AS node RETURN node",
     "MERGE (m:Media {id: node.id}) SET m += node",
     {batchSize: 1000, params: {nodes: $nodes}}
   );
   ```

### Memory Issues

Increase heap and pagecache:
```properties
# neo4j.conf
dbms.memory.heap.max_size=8G
dbms.memory.pagecache.size=4G
```

### Connection Pool Exhaustion

Monitor active connections:
```cypher
CALL dbms.listConnections();
```

Increase pool size:
```rust
max_connections: 50
```

## References

- [Neo4j Rust Driver (neo4rs)](https://github.com/neo4j-labs/neo4rs)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [RDF Turtle Format](https://www.w3.org/TR/turtle/)
- [APOC Procedures](https://neo4j.com/labs/apoc/)
