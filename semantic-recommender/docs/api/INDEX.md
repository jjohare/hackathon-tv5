# API Documentation

**Version:** 1.0
**Date:** 2025-12-07
**Protocol:** REST over HTTPS
**Base URL:** `http://localhost:5000` (development)

---

## Overview

Complete interface specifications for the semantic recommender system, including REST API endpoints, MCP server integration, wire formats, and authentication mechanisms.

---

## Documents

### Core API Reference

1. **[REST_API.md](./REST_API.md)** - HTTP API complete reference
   - `/api/query` - Single semantic query
   - `/api/query/batch` - Batch query processing
   - Request/response schemas (JSON)
   - Error handling and status codes
   - Rate limiting

2. **[MCP_SERVER.md](./MCP_SERVER.md)** - MCP integration specification
   - MCP protocol overview
   - Tool definitions
   - Message formats
   - Integration patterns

3. **[PROTOCOLS.md](./PROTOCOLS.md)** - Wire format specifications
   - JSON schema definitions
   - Binary embedding formats (NumPy)
   - Vector serialisation
   - Metadata structures

4. **[AUTHENTICATION.md](./AUTHENTICATION.md)** - Auth mechanisms
   - Bearer token authentication
   - API key management
   - Rate limiting policies
   - Security best practices

---

## Quick Examples

### Single Query

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "dark psychological thriller",
    "limit": 5
  }'
```

### Batch Query

```bash
curl -X POST http://localhost:5000/api/query/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "dark thriller",
      "romantic comedy",
      "sci-fi adventure"
    ],
    "limit": 10
  }'
```

See [examples/](./examples/) for more request/response examples.

---

## Response Format

All successful queries return:

```json
{
  "results": [
    {
      "rank": 1,
      "title": "The Prestige (2006)",
      "score": 0.8734,
      "similarity_score": 0.8421,
      "ontology": {
        "ontology_score": 0.91,
        "shared_classes": ["thriller", "mystery"]
      }
    }
  ],
  "performance": {
    "total_time_ms": 26.9,
    "encoding_time_ms": 24.0
  }
}
```

---

## Performance Characteristics

| Endpoint | Latency (P50) | Latency (P95) | Throughput |
|----------|---------------|---------------|------------|
| `/api/query` | 27ms | 50ms | 270 QPS |
| `/api/query/batch` (10) | 120ms | 250ms | 83 QPS |

Hardware: RTX A6000, 1.3M movie dataset

---

## Error Handling

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_QUERY` | Query validation failed |
| 401 | `INVALID_API_KEY` | Authentication failed |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_SERVER_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Temporary unavailability |

See [REST_API.md](./REST_API.md) for detailed error schemas.

---

## Related Documentation

### Prerequisites
- [Architecture Overview](../architecture/SYSTEM_OVERVIEW.md)
- [Quick Start](../guides/QUICKSTART.md)

### Implementation
- [Query Interface Source](../../scripts/server/query_interface.py)
- [MCP Server Source](../../scripts/server/mcp_server_websocket.py)

### Guides
- [Deployment Guide](../guides/DEPLOYMENT_GUIDE.md)
- [Performance Tuning](../guides/PERFORMANCE_TUNING.md)

---

**Last Updated:** 2025-12-07
