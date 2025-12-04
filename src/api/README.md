# TV5 Media Gateway API

GPU-accelerated semantic media recommendation system with AI agent integration.

## Features

- **REST API** - RESTful endpoints with OpenAPI 3.1 documentation
- **GraphQL API** - Flexible query interface for complex data needs
- **MCP Integration** - Model Context Protocol for AI agent compatibility
- **HATEOAS** - Hypermedia-driven navigation for discoverability
- **JSON-LD** - Semantic web integration with Schema.org vocabulary
- **High Performance** - <10ms overhead, 7,000+ QPS capability

## Quick Start

```bash
# Build the API
cargo build --release

# Run the server
cargo run --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## API Endpoints

### REST API

- `POST /api/v1/search` - Semantic media search
- `GET /api/v1/recommendations/:user_id` - Personalized recommendations
- `GET /api/v1/mcp/manifest` - MCP tool manifest for AI agents
- `GET /health` - Health check

### GraphQL

- `POST /graphql` - GraphQL queries and mutations

### Documentation

- `/swagger-ui` - Interactive OpenAPI documentation
- `/api-docs/openapi.json` - OpenAPI 3.1 specification

## Example Requests

### Search Media (REST)

```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "French noir films with existential themes",
    "filters": {
      "language": "fr",
      "min_rating": 7.0
    },
    "limit": 5
  }'
```

### Get Recommendations (REST)

```bash
curl http://localhost:3000/api/v1/recommendations/user_123?limit=10&explain=true
```

### Search Media (GraphQL)

```graphql
query {
  searchMedia(
    query: "French films"
    limit: 5
    genres: ["drama", "noir"]
  ) {
    results {
      id
      title
      similarityScore
      explanation
    }
    queryTimeMs
  }
}
```

### Get MCP Manifest

```bash
curl http://localhost:3000/api/v1/mcp/manifest
```

## Response Formats

### HATEOAS Response

```json
{
  "data": {
    "results": [...],
    "total": 42
  },
  "_links": {
    "self": {
      "href": "/api/v1/search",
      "rel": "self",
      "method": "POST"
    },
    "actions": [
      {
        "href": "/api/v1/search/refine",
        "rel": "refine",
        "method": "POST",
        "title": "Refine search results"
      }
    ]
  }
}
```

### JSON-LD Response

```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "tv5": "https://tv5monde.com/vocab/",
    "similarity": "tv5:similarityScore"
  },
  "@type": "RecommendationList",
  "data": {
    "user_id": "user_123",
    "recommendations": [...]
  }
}
```

## Performance

Target metrics:
- **Latency**: <10ms API overhead
- **Throughput**: 7,000+ QPS
- **Concurrency**: 100+ concurrent connections

Run benchmarks:
```bash
cargo bench
```

## AI Agent Integration

### MCP Protocol

The API exposes MCP-compatible tool definitions for seamless AI agent integration:

```json
{
  "name": "tv5-media-gateway",
  "version": "1.0.0",
  "tools": [
    {
      "name": "search_media",
      "description": "Search for media using semantic similarity",
      "input_schema": {...},
      "output_schema": {...},
      "examples": [...]
    }
  ]
}
```

### Example Agent Usage

```python
# Claude Desktop MCP integration
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    tools=[{
        "name": "search_media",
        "description": "Search TV5 media catalog",
        "input_schema": {...}
    }],
    messages=[{
        "role": "user",
        "content": "Find French films about existentialism"
    }]
)
```

## Architecture

```
┌─────────────────┐
│   REST/GraphQL  │
└────────┬────────┘
         │
┌────────▼────────┐
│  Axum Router    │
└────────┬────────┘
         │
┌────────▼────────────┐
│ Recommendation      │
│ Engine (GPU)        │
└─────────────────────┘
```

## Development

### Project Structure

```
src/api/
├── main.rs              # Application entry point
├── models.rs            # Data models and schemas
├── mcp.rs              # MCP manifest generation
├── hateoas.rs          # HATEOAS response helpers
├── jsonld.rs           # JSON-LD context generation
├── error.rs            # Error handling
├── recommendation.rs   # Recommendation engine
├── graphql.rs          # GraphQL schema
├── benches/            # Performance benchmarks
└── tests/              # Integration tests
```

### Adding New Endpoints

1. Define models in `models.rs`
2. Add route handler in `main.rs`
3. Document with `#[utoipa::path]` macro
4. Update OpenAPI schema
5. Add tests

## License

MIT
