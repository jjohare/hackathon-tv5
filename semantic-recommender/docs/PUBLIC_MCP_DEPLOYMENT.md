# Public MCP Server Deployment Guide

## Overview

Deploy the GPU-accelerated Semantic Recommender as a **public MCP server** accessible via HTTP/REST API for AI agents, applications, and services.

**Performance:** 316K QPS on A100, <1ms latency

---

## 🚀 Quick Start (Local Testing)

```bash
# 1. Install dependencies
cd semantic-recommender
source venv/bin/activate
pip install aiohttp

# 2. Start HTTP MCP server
python scripts/mcp_server_http.py

# Server starts on http://localhost:8888
```

**Test the server:**
```bash
# Health check
curl http://localhost:8888/health

# Get MCP manifest
curl http://localhost:8888/mcp/manifest

# Search for movies
curl -X POST http://localhost:8888/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search_media",
    "arguments": {
      "query": "animated family movies",
      "limit": 5
    }
  }'
```

---

## 🔒 Production Deployment (with API Key)

### Environment Variables

```bash
export MCP_PORT=8888
export MCP_HOST=0.0.0.0
export MCP_API_KEY="your-secret-api-key-here"
```

### Start with Authentication

```bash
python scripts/mcp_server_http.py \
  --port 8888 \
  --host 0.0.0.0 \
  --api-key "your-secret-api-key"
```

### Authenticated Requests

```bash
curl -X POST http://your-server:8888/mcp/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "tool": "search_media",
    "arguments": {"query": "sci-fi movies", "limit": 10}
  }'
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
cd semantic-recommender

# Build image
docker build -f docker/Dockerfile.mcp -t semantic-recommender-mcp:latest .

# Run container
docker run -d \
  -p 8888:8888 \
  --name mcp-server \
  semantic-recommender-mcp:latest
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.mcp
    ports:
      - "8888:8888"
    environment:
      - MCP_API_KEY=${MCP_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8888/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

**Deploy:**
```bash
export MCP_API_KEY="your-secret-key"
docker-compose up -d
```

---

## ☁️ Cloud Deployment Options

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/mcp-server

# Deploy
gcloud run deploy mcp-server \
  --image gcr.io/PROJECT_ID/mcp-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8888 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars MCP_API_KEY=your-secret-key
```

**With A100 GPU (GCE):**

```bash
# Deploy to Compute Engine with A100
gcloud compute instances create mcp-server-a100 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --metadata=startup-script='#!/bin/bash
    cd /opt
    git clone https://github.com/jjohare/hackathon-tv5.git
    cd hackathon-tv5/semantic-recommender
    pip install aiohttp torch sentence-transformers numpy
    python scripts/mcp_server_http.py --host 0.0.0.0 --port 8888
  '

# Open firewall
gcloud compute firewall-rules create allow-mcp \
  --allow tcp:8888 \
  --source-ranges 0.0.0.0/0
```

### AWS ECS

```bash
# Create task definition
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster mcp-cluster \
  --service-name mcp-server \
  --task-definition mcp-server:1 \
  --desired-count 1 \
  --launch-type FARGATE
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: semantic-recommender-mcp:latest
        ports:
        - containerPort: 8888
        env:
        - name: MCP_API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"  # For A100
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8888
  type: LoadBalancer
```

---

## 📋 API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "gpu_available": true,
  "movies_loaded": 62423
}
```

### GET /mcp/manifest

Returns MCP tool manifest.

**Response:**
```json
{
  "name": "semantic-recommender",
  "version": "1.0.0",
  "description": "GPU-accelerated hybrid semantic + ontology reasoning",
  "tools": [
    {
      "name": "search_media",
      "description": "Search for media using GPU-accelerated semantic similarity (316K QPS on A100)",
      "input_schema": {...}
    },
    {
      "name": "get_recommendations",
      "description": "Get personalized recommendations with hybrid reasoning",
      "input_schema": {...}
    }
  ],
  "capabilities": {
    "gpu_accelerated": true,
    "device": "cuda:0",
    "throughput_qps": 316000
  }
}
```

### POST /mcp/call

Execute MCP tool call.

**Request:**
```json
{
  "tool": "search_media",
  "arguments": {
    "query": "French science fiction films",
    "filters": {
      "language": "fr",
      "year_range": [2000, 2020],
      "min_rating": 7.0
    },
    "limit": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "results": [
      {
        "id": "movie_123",
        "title": "La Jetée",
        "similarity_score": 0.91,
        "explanation": "La Jetée (1962) - Sci-Fi, Short - Excellent match (91% similarity)",
        "metadata": {
          "genres": ["Sci-Fi", "Short"],
          "year": 1962,
          "language": "fr",
          "rating": 8.3
        }
      }
    ],
    "total": 1,
    "query_time_ms": 0.8,
    "device": "cuda:0",
    "gpu_accelerated": true
  }
}
```

---

## 🔒 Security Best Practices

### 1. API Key Authentication

**Always use API keys in production:**

```bash
# Generate strong API key
openssl rand -base64 32

# Start server with API key
python scripts/mcp_server_http.py --api-key "$(cat api_key.txt)"
```

### 2. HTTPS/TLS

Use a reverse proxy (nginx, Caddy) for TLS:

```nginx
server {
    listen 443 ssl http2;
    server_name mcp.example.com;

    ssl_certificate /etc/letsencrypt/live/mcp.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8888;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Rate Limiting

Use nginx rate limiting:

```nginx
limit_req_zone $binary_remote_addr zone=mcp_limit:10m rate=100r/s;

location /mcp/ {
    limit_req zone=mcp_limit burst=200 nodelay;
    proxy_pass http://localhost:8888;
}
```

### 4. CORS Configuration

The MCP server includes CORS headers by default. To restrict origins:

Edit `scripts/mcp_server_http.py`:
```python
response.headers['Access-Control-Allow-Origin'] = 'https://your-app.com'
```

---

## 📊 Monitoring

### Prometheus Metrics (Optional Enhancement)

Add to `scripts/mcp_server_http.py`:

```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('mcp_requests_total', 'Total requests')
request_duration = Histogram('mcp_request_duration_seconds', 'Request duration')

@app.route('/metrics')
async def metrics(request):
    return web.Response(text=generate_latest(), content_type='text/plain')
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mcp-server.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🧪 Testing Public Endpoint

```bash
# Test from remote machine
curl -X POST https://your-server.com/mcp/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "tool": "search_media",
    "arguments": {
      "query": "thriller movies",
      "limit": 5
    }
  }'

# Expected response time: <1ms on A100, <50ms on CPU
```

---

## 🌐 Public MCP Registry (Future)

Register your MCP server with AI agent marketplaces:

```json
{
  "name": "Semantic Recommender MCP",
  "url": "https://your-server.com/mcp",
  "description": "GPU-accelerated semantic search and recommendations",
  "performance": {
    "throughput_qps": 316000,
    "latency_ms": 1
  },
  "pricing": "free" | "paid",
  "authentication": "api-key"
}
```

---

## 📖 Integration Examples

### Claude Code

```json
{
  "mcpServers": {
    "semantic-recommender": {
      "command": "curl",
      "args": [
        "-X", "POST",
        "https://your-server.com/mcp/call",
        "-H", "Authorization: Bearer YOUR_API_KEY",
        "-H", "Content-Type: application/json"
      ]
    }
  }
}
```

### Python Client

```python
import requests

def search_movies(query: str, limit: int = 10):
    response = requests.post(
        'https://your-server.com/mcp/call',
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_API_KEY'
        },
        json={
            'tool': 'search_media',
            'arguments': {
                'query': query,
                'limit': limit
            }
        }
    )
    return response.json()['result']

# Usage
results = search_movies('sci-fi movies')
for movie in results['results']:
    print(f"{movie['title']} ({movie['similarity_score']:.2f})")
```

### JavaScript Client

```javascript
async function searchMovies(query, limit = 10) {
  const response = await fetch('https://your-server.com/mcp/call', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      tool: 'search_media',
      arguments: { query, limit }
    })
  });

  const data = await response.json();
  return data.result;
}

// Usage
const results = await searchMovies('action movies');
console.log(results.results);
```

---

## 🚀 Performance Tuning

### A100 GPU Optimization

```bash
# Enable TF32 for 5-10x speedup
export NVIDIA_TF32_OVERRIDE=1

# Optimize CUDA
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Batch Processing

For high-throughput scenarios, implement request batching:

```python
# Process multiple queries in parallel
batch = [query1, query2, query3]
results = await asyncio.gather(*[search_media(q) for q in batch])
```

---

## 📄 License

Apache 2.0 - See LICENSE file

---

**Documentation Version:** 1.0.0
**Last Updated:** 2025-12-06
**Maintainer:** Semantic Recommender Team
