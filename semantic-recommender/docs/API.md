# API Reference

Complete specification for REST API and MCP server interfaces.

## REST API Overview

**Base URL**: `http://localhost:8080/api/v1`

**Authentication**: All endpoints require `Authorization: Bearer <api_key>` header (development: optional)

**Response Format**: JSON

**Latency SLA**: P50 <15ms, P99 <50ms

---

## Endpoints

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
  "timestamp": "2025-12-06T10:30:00Z"
}
```

---

### Semantic Search

**POST** `/search`

Search for semantically similar media entities.

**Request**:
```json
{
  "query": "string",                    // Required: search query
  "query_type": "text|image|audio|video", // Optional: default "text"
  "limit": 10,                          // Optional: result count (1-1000)
  "threshold": 0.85,                    // Optional: min similarity (0-1.0)
  "filters": {                          // Optional: metadata filters
    "language": "fr",
    "genre": "Documentary",
    "year_min": 2020,
    "year_max": 2025
  },
  "execution_hint": "auto|gpu|vector_db" // Optional: routing hint
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
    "execution_path": "gpu",           // "gpu" | "vector_db" | "hybrid"
    "gpu_utilization": 0.92,
    "cache_hit_rate": 0.85
  }
}
```

**Error Cases**:
```json
{
  "error": "invalid_query",
  "message": "Query must be non-empty",
  "code": "INVALID_REQUEST"
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
      "results": [...],  // Same as /search response
      "query_time_ms": 12
    },
    {
      "query_index": 1,
      "query": "Spanish thriller series",
      "results": [...],
      "query_time_ms": 14
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
  "exploration_rate": 0.1  // 10% exploration vs 90% exploitation
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
  "entity": "Documentary",           // Required: entity name
  "depth": 2,                        // Optional: traversal depth
  "direction": "outbound|inbound|both", // Optional: edge direction
  "edge_types": ["subClassOf", "hasGenre"], // Optional: filter edges
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

## Request/Response Specification

### Common Fields

#### Filters Object
```json
{
  "language": "fr",              // ISO 639-1
  "genre": "Documentary",        // Exact or partial match
  "year_min": 2020,
  "year_max": 2025,
  "duration_min": 0,
  "duration_max": 120,
  "provider": "TV5Monde",
  "content_type": "video|audio|text|image",
  "rating_min": 7.0,             // 0-10 scale
  "rating_max": 10.0
}
```

#### Metadata Object
```json
{
  "language": "fr",
  "genre": "Documentary",
  "duration": 52,                // minutes
  "year": 2023,
  "provider": "TV5Monde",
  "rating": 8.5,
  "views": 1234567,
  "verified": true
}
```

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

## MCP Server Interface

**Start**: `cargo run --release --bin mcp-server`

**Protocol**: JSON-RPC 2.0 over stdio or SSE

### Available Tools

#### semantic_search

Multi-modal semantic search.

**Arguments**:
```json
{
  "query": "French documentary",
  "query_type": "text",
  "limit": 10,
  "filters": {}
}
```

**Result**:
```json
{
  "results": [...],
  "query_time_ms": 12
}
```

#### batch_search

Execute multiple queries.

**Arguments**:
```json
{
  "queries": ["query1", "query2"],
  "limit": 10
}
```

#### recommend

Generate recommendations.

**Arguments**:
```json
{
  "user_id": "user_123",
  "context": {},
  "limit": 10
}
```

#### ontology_query

Graph traversal.

**Arguments**:
```json
{
  "entity": "Documentary",
  "depth": 2,
  "direction": "outbound"
}
```

#### get_similar

Find similar items.

**Arguments**:
```json
{
  "entity_id": "doc_12345",
  "limit": 10
}
```

---

## Rate Limiting

**Default**: 1000 requests/second per API key

**Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1670323800
```

When limit exceeded:
```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

---

## Authentication (Production)

### API Key
```bash
Authorization: Bearer ag_xxxxxxxxxxxxxxxx
```

### OAuth 2.0
```bash
# Get token
curl -X POST http://localhost:8080/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=...&client_secret=..."

# Use token
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/v1/search
```

---

## Examples

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

---

## Performance SLAs

| Operation | P50 | P99 | Expected |
|-----------|-----|-----|----------|
| Search (100M entities) | 12ms | 45ms | <15ms |
| Batch Search (10 queries) | 120ms | 400ms | <150ms |
| Recommendation (cold-start) | 45ms | 150ms | <100ms |
| Ontology Query (depth 2) | 28ms | 100ms | <50ms |

---

See [INTEGRATION.md](INTEGRATION.md) for integration patterns with the hackathon.
