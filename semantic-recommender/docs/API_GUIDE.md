# API Guide

Complete API reference for the Media Gateway Hackathon system.

## Table of Contents

1. [Authentication](#authentication)
2. [API Endpoints](#api-endpoints)
3. [Request/Response Examples](#requestresponse-examples)
4. [MCP Integration](#mcp-integration)
5. [Rate Limiting](#rate-limiting)
6. [Error Handling](#error-handling)
7. [SDKs and Client Libraries](#sdks-and-client-libraries)

## Base URL

**Production**: `https://api.media-gateway.example.com`
**Development**: `http://localhost:8080`

## Authentication

### API Key Authentication

All requests require an API key in the header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Obtaining an API Key

```bash
# Request via CLI
npx agentics-hackathon auth register \
  --email your@email.com \
  --project "My Project"

# Or via API
curl -X POST https://api.media-gateway.example.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "project": "My Project"
  }'

# Response
{
  "api_key": "ag_1234567890abcdef",
  "project_id": "proj_abc123",
  "quota": {
    "requests_per_day": 10000,
    "embeddings_per_month": 1000000
  }
}
```

### OAuth 2.0 (Optional)

For user-facing applications:

```bash
# Authorization URL
https://api.media-gateway.example.com/oauth/authorize?
  client_id=YOUR_CLIENT_ID&
  redirect_uri=YOUR_REDIRECT_URI&
  response_type=code&
  scope=read write

# Token exchange
curl -X POST https://api.media-gateway.example.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code&code=AUTH_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET"
```

## API Endpoints

### Health and Status

#### GET /health

Health check endpoint (no authentication required).

```bash
curl https://api.media-gateway.example.com/health
```

**Response**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-12-04T10:30:00Z",
  "components": {
    "gpu": "operational",
    "vector_db": "operational",
    "knowledge_graph": "operational",
    "cache": "operational"
  }
}
```

#### GET /ready

Readiness check for Kubernetes.

```bash
curl https://api.media-gateway.example.com/ready
```

**Response**
```json
{
  "ready": true,
  "checks": {
    "gpu_initialized": true,
    "models_loaded": true,
    "databases_connected": true
  }
}
```

#### GET /metrics

Prometheus metrics endpoint.

```bash
curl https://api.media-gateway.example.com/metrics
```

**Response** (Prometheus format)
```
# HELP request_duration_seconds Request duration
# TYPE request_duration_seconds histogram
request_duration_seconds_bucket{le="0.01"} 1000
request_duration_seconds_bucket{le="0.05"} 5000
request_duration_seconds_sum 125.5
request_duration_seconds_count 10000

# HELP gpu_utilization_percent GPU utilization percentage
# TYPE gpu_utilization_percent gauge
gpu_utilization_percent{gpu="0"} 87.5
```

### Search API

#### POST /v1/search

Semantic search across media content.

**Request**
```bash
curl -X POST https://api.media-gateway.example.com/v1/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "French documentary about climate change",
    "limit": 10,
    "threshold": 0.85,
    "modalities": ["text", "visual"],
    "filters": {
      "language": "fr",
      "content_type": "documentary",
      "year_min": 2020
    }
  }'
```

**Response**
```json
{
  "results": [
    {
      "id": "doc_123456",
      "title": "Le Climat en Péril",
      "description": "Documentary about climate change impacts in France",
      "score": 0.94,
      "metadata": {
        "duration": 5400,
        "language": "fr",
        "year": 2023,
        "genres": ["documentary", "environmental"]
      },
      "embeddings": {
        "visual_similarity": 0.92,
        "text_similarity": 0.96,
        "audio_similarity": 0.89
      },
      "ontology": {
        "categories": ["Environment", "Science", "Documentary"],
        "entities": ["Climate", "France", "Environment"]
      }
    }
  ],
  "query_time_ms": 45,
  "total_results": 127,
  "has_more": true
}
```

#### POST /v1/search/multimodal

Multi-modal search with image/video/text inputs.

**Request**
```bash
curl -X POST https://api.media-gateway.example.com/v1/search/multimodal \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "text=Documentary about art" \
  -F "image=@reference_image.jpg" \
  -F "limit=5"
```

**Response**
```json
{
  "results": [
    {
      "id": "doc_789012",
      "title": "L'Art Moderne",
      "score": 0.91,
      "match_breakdown": {
        "text_contribution": 0.35,
        "visual_contribution": 0.65
      }
    }
  ],
  "query_time_ms": 78
}
```

### Recommendation API

#### POST /v1/recommend

Get personalized recommendations.

**Request**
```bash
curl -X POST https://api.media-gateway.example.com/v1/recommend \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "history": [
      {"item_id": "doc_456", "rating": 5},
      {"item_id": "doc_789", "rating": 4}
    ],
    "limit": 20,
    "diversify": true,
    "explain": true
  }'
```

**Response**
```json
{
  "recommendations": [
    {
      "id": "doc_345",
      "title": "Histoire de France",
      "score": 0.89,
      "reason": "Similar to 'doc_456' (French history documentaries)",
      "explanation": {
        "collaborative_filtering": 0.35,
        "content_based": 0.45,
        "ontology_reasoning": 0.20
      }
    }
  ],
  "user_profile": {
    "top_genres": ["documentary", "history", "science"],
    "languages": ["fr", "en"],
    "avg_rating": 4.5
  },
  "query_time_ms": 125
}
```

#### GET /v1/recommend/trending

Get trending content.

**Request**
```bash
curl https://api.media-gateway.example.com/v1/recommend/trending \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -G -d "category=documentary" \
  -d "time_window=7d" \
  -d "limit=10"
```

**Response**
```json
{
  "trending": [
    {
      "id": "doc_111",
      "title": "Intelligence Artificielle",
      "views_7d": 125000,
      "growth_rate": 3.5,
      "score": 0.95
    }
  ],
  "period": "2025-11-27T00:00:00Z/2025-12-04T00:00:00Z"
}
```

### Embeddings API

#### POST /v1/embeddings

Generate embeddings for text, images, or audio.

**Request**
```bash
curl -X POST https://api.media-gateway.example.com/v1/embeddings \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "type": "text",
        "content": "French cinema history"
      },
      {
        "type": "image_url",
        "url": "https://example.com/image.jpg"
      }
    ],
    "model": "multimodal-1024",
    "normalize": true
  }'
```

**Response**
```json
{
  "embeddings": [
    {
      "index": 0,
      "embedding": [0.123, -0.456, 0.789, ...],
      "dimensions": 1024,
      "model": "multimodal-1024"
    },
    {
      "index": 1,
      "embedding": [0.234, -0.567, 0.890, ...],
      "dimensions": 1024,
      "model": "multimodal-1024"
    }
  ],
  "usage": {
    "total_tokens": 15,
    "compute_time_ms": 12
  }
}
```

### Ontology API

#### POST /v1/ontology/infer

Apply ontology reasoning to content.

**Request**
```bash
curl -X POST https://api.media-gateway.example.com/v1/ontology/infer \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "doc_123",
    "inference_rules": ["transitive_closure", "disjoint_check"],
    "depth": 3
  }'
```

**Response**
```json
{
  "entity_id": "doc_123",
  "inferred_relationships": [
    {
      "relation": "isPartOf",
      "target": "series_456",
      "confidence": 0.98,
      "rule": "transitive_closure"
    },
    {
      "relation": "hasGenre",
      "target": "Documentary",
      "confidence": 1.0,
      "rule": "direct"
    }
  ],
  "categories": [
    "Documentary",
    "Science",
    "Educational"
  ],
  "reasoning_time_ms": 23
}
```

#### GET /v1/ontology/graph

Query the knowledge graph.

**Request**
```bash
curl -X GET https://api.media-gateway.example.com/v1/ontology/graph \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -G -d "query=MATCH (d:Documentary)-[:ABOUT]->(t:Topic {name: 'Climate'}) RETURN d" \
  -d "limit=10"
```

**Response**
```json
{
  "nodes": [
    {
      "id": "doc_123",
      "labels": ["Documentary", "Content"],
      "properties": {
        "title": "Le Climat en Péril",
        "year": 2023
      }
    }
  ],
  "relationships": [
    {
      "id": "rel_456",
      "type": "ABOUT",
      "start_node": "doc_123",
      "end_node": "topic_789"
    }
  ],
  "query_time_ms": 34
}
```

### Analytics API

#### GET /v1/analytics/usage

Get API usage statistics.

**Request**
```bash
curl https://api.media-gateway.example.com/v1/analytics/usage \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -G -d "start_date=2025-11-01" \
  -d "end_date=2025-12-01"
```

**Response**
```json
{
  "period": "2025-11-01/2025-12-01",
  "total_requests": 250000,
  "successful_requests": 248500,
  "error_rate": 0.006,
  "avg_latency_ms": 67,
  "p99_latency_ms": 145,
  "endpoints": {
    "/v1/search": 150000,
    "/v1/recommend": 80000,
    "/v1/embeddings": 20000
  },
  "quota_used": {
    "requests": 250000,
    "limit": 1000000,
    "percentage": 25.0
  }
}
```

#### GET /v1/analytics/performance

GPU and system performance metrics.

**Request**
```bash
curl https://api.media-gateway.example.com/v1/analytics/performance \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response**
```json
{
  "timestamp": "2025-12-04T10:30:00Z",
  "gpu": {
    "utilization_percent": 87.5,
    "memory_used_gb": 12.3,
    "memory_total_gb": 16.0,
    "temperature_c": 68,
    "power_usage_w": 145
  },
  "throughput": {
    "queries_per_second": 127,
    "embeddings_per_second": 450
  },
  "latency": {
    "search_p50_ms": 35,
    "search_p99_ms": 89,
    "embedding_p50_ms": 8,
    "embedding_p99_ms": 15
  }
}
```

## MCP Integration

### MCP Server Setup

The system includes a Model Context Protocol (MCP) server for AI agent integration.

#### STDIO Transport

```json
// claude_desktop_config.json or cline_mcp_settings.json
{
  "mcpServers": {
    "media-gateway": {
      "command": "npx",
      "args": ["agentics-hackathon", "mcp", "stdio"],
      "env": {
        "API_KEY": "YOUR_API_KEY",
        "API_URL": "https://api.media-gateway.example.com"
      }
    }
  }
}
```

#### SSE Transport

```bash
# Start SSE server
npx agentics-hackathon mcp sse --port 3000

# Connect from client
const eventSource = new EventSource('http://localhost:3000/sse');
eventSource.onmessage = (event) => {
  console.log('MCP message:', JSON.parse(event.data));
};
```

### MCP Tools

**Available MCP Tools**

1. **search_content** - Semantic search
2. **get_recommendations** - Personalized recommendations
3. **generate_embeddings** - Create embeddings
4. **query_ontology** - Knowledge graph queries
5. **get_analytics** - Usage statistics

**Example MCP Tool Call**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_content",
    "arguments": {
      "query": "French documentary about climate change",
      "limit": 5,
      "threshold": 0.85
    }
  },
  "id": 1
}
```

**Response**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 5 results:\n1. Le Climat en Péril (score: 0.94)\n2. Environnement Urgence (score: 0.91)\n..."
      }
    ],
    "isError": false
  },
  "id": 1
}
```

### MCP Resources

**Accessing Pre-indexed Content**

```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "media-gateway://catalog/documentaries/climate"
  },
  "id": 1
}
```

## Rate Limiting

### Default Limits

| Tier | Requests/Day | Embeddings/Month | Burst |
|------|-------------|------------------|-------|
| Free | 1,000 | 10,000 | 10/sec |
| Starter | 10,000 | 1,000,000 | 50/sec |
| Pro | 100,000 | 10,000,000 | 200/sec |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers

```http
X-RateLimit-Limit: 10000
X-RateLimit-Remaining: 9500
X-RateLimit-Reset: 1733328000
Retry-After: 60
```

### Handling Rate Limits

```bash
# Response when rate limited
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Retry after 60 seconds.",
    "retry_after_seconds": 60
  }
}
```

**Exponential Backoff Example**

```python
import time
import requests

def api_call_with_retry(url, headers, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            time.sleep(retry_after * (2 ** attempt))
        else:
            raise Exception(f"API error: {response.status_code}")

    raise Exception("Max retries exceeded")
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Missing required parameter: query",
    "details": {
      "parameter": "query",
      "expected_type": "string"
    },
    "request_id": "req_abc123xyz"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Invalid request parameters |
| `authentication_failed` | 401 | Invalid or missing API key |
| `authorization_failed` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limit_exceeded` | 429 | Rate limit exceeded |
| `internal_error` | 500 | Internal server error |
| `service_unavailable` | 503 | Service temporarily unavailable |
| `gpu_error` | 500 | GPU computation error |
| `database_error` | 500 | Database connection error |

### Handling Errors

```javascript
const axios = require('axios');

async function searchContent(query) {
  try {
    const response = await axios.post(
      'https://api.media-gateway.example.com/v1/search',
      { query, limit: 10 },
      { headers: { 'Authorization': `Bearer ${API_KEY}` } }
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      const { code, message, request_id } = error.response.data.error;

      switch (code) {
        case 'rate_limit_exceeded':
          console.error('Rate limited. Request ID:', request_id);
          // Implement retry logic
          break;
        case 'authentication_failed':
          console.error('Invalid API key');
          // Refresh credentials
          break;
        default:
          console.error(`API error: ${message} (${request_id})`);
      }
    } else {
      console.error('Network error:', error.message);
    }
    throw error;
  }
}
```

## SDKs and Client Libraries

### Official SDKs

**Node.js/TypeScript**

```bash
npm install @media-gateway/sdk
```

```typescript
import { MediaGatewayClient } from '@media-gateway/sdk';

const client = new MediaGatewayClient({
  apiKey: process.env.MEDIA_GATEWAY_API_KEY,
  baseUrl: 'https://api.media-gateway.example.com'
});

// Search
const results = await client.search({
  query: 'French documentary',
  limit: 10
});

// Recommendations
const recommendations = await client.recommend({
  userId: 'user_123',
  limit: 20
});
```

**Python**

```bash
pip install media-gateway-sdk
```

```python
from media_gateway import MediaGatewayClient

client = MediaGatewayClient(
    api_key=os.environ['MEDIA_GATEWAY_API_KEY'],
    base_url='https://api.media-gateway.example.com'
)

# Search
results = client.search(
    query='French documentary',
    limit=10
)

# Embeddings
embeddings = client.embeddings.create(
    inputs=[
        {'type': 'text', 'content': 'Cinema history'},
        {'type': 'image_url', 'url': 'https://example.com/image.jpg'}
    ]
)
```

**Rust**

```bash
cargo add media-gateway-sdk
```

```rust
use media_gateway_sdk::{MediaGatewayClient, SearchRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = MediaGatewayClient::new(
        std::env::var("MEDIA_GATEWAY_API_KEY")?
    );

    let results = client.search(SearchRequest {
        query: "French documentary".to_string(),
        limit: Some(10),
        ..Default::default()
    }).await?;

    for result in results.results {
        println!("{}: {:.3}", result.title, result.score);
    }

    Ok(())
}
```

### Community Libraries

- **Go**: `github.com/example/media-gateway-go`
- **Ruby**: `gem install media_gateway`
- **PHP**: `composer require media-gateway/sdk`
- **Java**: `com.media-gateway:sdk:1.0.0`

## Webhooks

### Configuring Webhooks

```bash
curl -X POST https://api.media-gateway.example.com/v1/webhooks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/webhook",
    "events": ["recommendation.completed", "search.anomaly"],
    "secret": "whsec_your_webhook_secret"
  }'
```

### Webhook Events

```json
{
  "id": "evt_abc123",
  "type": "recommendation.completed",
  "created": 1733328000,
  "data": {
    "user_id": "user_123",
    "recommendations": [
      {"id": "doc_456", "score": 0.92}
    ]
  }
}
```

### Verifying Webhooks

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)
```

## Additional Resources

- **OpenAPI Spec**: [https://api.media-gateway.example.com/openapi.json](https://api.media-gateway.example.com/openapi.json)
- **Postman Collection**: [Download](https://api.media-gateway.example.com/postman.json)
- **API Status**: [status.media-gateway.example.com](https://status.media-gateway.example.com)
- **Developer Portal**: [developers.media-gateway.example.com](https://developers.media-gateway.example.com)

## Support

- **Documentation**: [docs.media-gateway.example.com](https://docs.media-gateway.example.com)
- **Discord**: [discord.agentics.org](https://discord.agentics.org)
- **Email**: support@agentics.org
- **GitHub**: [github.com/agenticsorg/hackathon-tv5/issues](https://github.com/agenticsorg/hackathon-tv5/issues)
