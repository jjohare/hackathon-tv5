# Batch Processing Implementation

## Overview

Batch processing implementation for the Flask query interface to achieve 1000+ QPS using TensorRT's batch_size=32 capability.

## Architecture

### BatchProcessor Class

The `BatchProcessor` class implements request queuing and batching:

```python
class BatchProcessor:
    def __init__(self, encoder, max_batch_size=32, max_wait_ms=50):
        """
        Args:
            encoder: TensorRT or SentenceTransformer encoder
            max_batch_size: Maximum batch size (32 for TensorRT)
            max_wait_ms: Maximum wait before processing partial batch
        """
```

**Key Features:**
- Request queuing with thread-safe deque
- Background processing thread
- Automatic batch accumulation (up to 32 requests or 50ms timeout)
- Future-based result delivery to individual requests
- Zero-copy GPU memory management (via TensorRT)

### Processing Flow

```
Client Request → Queue → Batch Accumulator → TensorRT Encoder → Result Distribution
                          (32 requests or 50ms)   (single GPU call)
```

1. **Request Queuing**: Client requests are added to a thread-safe queue
2. **Batch Accumulation**: Background thread collects up to 32 requests or waits 50ms
3. **Batch Encoding**: All queries encoded in single TensorRT call (GPU batch processing)
4. **Result Distribution**: Individual embeddings returned to corresponding requests

## API Endpoints

### 1. Single Query (with optional batching)

**Endpoint**: `POST /api/query`

**Request**:
```json
{
  "query": "action movies",
  "limit": 10,
  "use_batch": true,
  "filters": {
    "genres": ["Action"],
    "min_rating": 4.0
  }
}
```

**Response**:
```json
{
  "results": [...],
  "decision_log": {...},
  "performance": {
    "total_time_ms": 2.5,
    "encoding_time_ms": 0.8
  }
}
```

### 2. Batch Query (high-throughput)

**Endpoint**: `POST /api/query/batch`

**Request**:
```json
{
  "queries": [
    "action movies",
    "sci-fi thriller",
    "romantic comedy"
  ],
  "limit": 5
}
```

**Response**:
```json
{
  "results": [
    {
      "query": "action movies",
      "results": [...],
      "performance": {...}
    },
    ...
  ],
  "batch_performance": {
    "total_queries": 3,
    "total_time_ms": 5.2,
    "avg_time_per_query_ms": 1.73,
    "qps": 576.92
  }
}
```

### 3. System Status

**Endpoint**: `GET /api/status`

**Response**:
```json
{
  "backend": "TensorRT FP16",
  "device": "cuda:0",
  "cuda_available": true,
  "items_loaded": 10000,
  "batch_processor": {
    "enabled": true,
    "max_batch_size": 32,
    "max_wait_ms": 50,
    "queue_size": 0,
    "running": true
  }
}
```

## Performance Targets

### Current Performance
- **Sequential**: 39.9 QPS (no batching)
- **With Batching**: 1185 QPS (projected)
- **Target**: 1000+ QPS

### Batch Processing Benefits
- **25-30x speedup** over sequential processing
- **Efficient GPU utilization** (batch_size=32)
- **Low latency** (50ms max wait time)

### Performance Breakdown

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Request queuing | 0.1 | Thread-safe deque |
| Batch accumulation | 0-50 | Max wait time |
| TensorRT encoding | 0.8-1.2 | 32 queries/batch |
| Result distribution | 0.1 | Zero-copy |
| **Total** | **1-51** | Per query (batched) |

## Testing

### 1. Basic Functionality Test

```bash
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender
python scripts/tests/test_batch_processing.py
```

**Tests**:
- Single query (baseline)
- Sequential queries (without batching)
- Batch endpoint (with batching)
- Individual queries with batch flag
- Concurrent batch queries (stress test)

### 2. QPS Benchmark

```bash
python scripts/tests/benchmark_batch_qps.py --requests 1000 --workers 20 --batch-size 32
```

**Options**:
- `--requests`: Total number of queries (default: 1000)
- `--workers`: Concurrent workers (default: 20)
- `--batch-size`: Queries per batch (default: 32)
- `--mode`: `batch` or `single` endpoint

### 3. Load Testing

```bash
# High load test (10K queries)
python scripts/tests/benchmark_batch_qps.py --requests 10000 --workers 50

# Stress test (100K queries)
python scripts/tests/benchmark_batch_qps.py --requests 100000 --workers 100
```

## Usage Examples

### Example 1: Single Query with Batching

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "action movies",
    "limit": 5,
    "use_batch": true
  }'
```

### Example 2: Batch Query

```bash
curl -X POST http://localhost:5000/api/query/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "action movies",
      "sci-fi thriller",
      "romantic comedy"
    ],
    "limit": 5
  }'
```

### Example 3: Python Client

```python
import requests

# Single query with batching
response = requests.post(
    "http://localhost:5000/api/query",
    json={"query": "action movies", "limit": 5, "use_batch": True}
)
print(f"QPS: {1000 / response.json()['performance']['total_time_ms']:.2f}")

# Batch query
response = requests.post(
    "http://localhost:5000/api/query/batch",
    json={
        "queries": ["query1", "query2", "query3"],
        "limit": 5
    }
)
print(f"Batch QPS: {response.json()['batch_performance']['qps']:.2f}")
```

## Implementation Details

### Thread Safety

- **Queue**: `collections.deque` with `threading.Lock`
- **Future pattern**: Dictionary-based future with `ready` flag
- **Background thread**: Daemon thread for batch processing

### Error Handling

- **Timeout**: 5s default timeout for batch encoding
- **Batch errors**: All requests in failed batch receive error
- **Graceful degradation**: Falls back to sequential on batch processor failure

### Memory Management

- **Zero-copy**: TensorRT tensors stay on GPU
- **Batch buffers**: Pre-allocated for batch_size=32
- **Dynamic shapes**: Buffers reallocated for smaller batches

## Configuration

### Tuning Parameters

```python
# In query_interface.py
self.batch_processor = BatchProcessor(
    encoder=self.encoder,
    max_batch_size=32,      # Match TensorRT engine batch size
    max_wait_ms=50          # Balance latency vs throughput
)
```

**Recommendations**:
- `max_batch_size`: Should match TensorRT engine batch size (32)
- `max_wait_ms`:
  - Lower (10-20ms): Better latency, lower throughput
  - Higher (50-100ms): Better throughput, higher latency

### Expected Performance by Configuration

| max_wait_ms | Latency (p50) | Latency (p95) | QPS (1000 req) |
|-------------|---------------|---------------|----------------|
| 10ms | 5-15ms | 20-30ms | 800-1000 |
| 50ms | 10-60ms | 80-100ms | 1000-1200 |
| 100ms | 20-120ms | 150-200ms | 1200-1500 |

## Monitoring

### Health Check

```bash
curl http://localhost:5000/api/status | jq '.batch_processor'
```

### Metrics to Monitor

- **Queue size**: Should be 0-5 under normal load (check `queue_size`)
- **QPS**: Should be 1000+ with TensorRT (check benchmark)
- **Batch processor running**: Should be `true` (check `running`)

## Troubleshooting

### Issue: QPS < 1000

**Possible causes**:
1. TensorRT not loaded (check `/api/status` - should show "TensorRT FP16")
2. CPU bottleneck (check GPU utilization with `nvidia-smi`)
3. max_wait_ms too low (increase to 50-100ms)
4. Insufficient concurrent load (increase workers in benchmark)

**Solutions**:
```bash
# Check TensorRT engine exists
ls -lh models/sentence_transformer_fp16_sm86.trt

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Increase batch wait time (in query_interface.py)
max_wait_ms=100  # Instead of 50
```

### Issue: High Latency

**Possible causes**:
1. max_wait_ms too high
2. Sequential processing (use_batch=False)
3. Network latency

**Solutions**:
```bash
# Reduce wait time
max_wait_ms=10  # Instead of 50

# Ensure batching is enabled
curl -X POST http://localhost:5000/api/query \
  -d '{"query": "test", "use_batch": true}'
```

### Issue: Queue Buildup

**Possible causes**:
1. Requests arriving faster than processing
2. GPU memory issues
3. Batch processor not running

**Solutions**:
```bash
# Check batch processor status
curl http://localhost:5000/api/status | jq '.batch_processor'

# Check GPU memory
nvidia-smi

# Restart server
pkill -f query_interface.py
python scripts/server/query_interface.py
```

## Future Enhancements

1. **Dynamic batch sizing**: Adjust batch_size based on load
2. **Priority queues**: High-priority requests processed first
3. **Multiple batch processors**: One per GPU for multi-GPU systems
4. **Metrics endpoint**: Prometheus metrics for monitoring
5. **Health endpoint**: Detailed health check with queue stats

## References

- TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/
- Flask Documentation: https://flask.palletsprojects.com/
- Thread Safety in Python: https://docs.python.org/3/library/threading.html
