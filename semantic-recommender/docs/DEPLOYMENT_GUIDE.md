# Production Deployment Guide

**GPU-Accelerated Semantic Recommender - Production Deployment**

This guide covers deployment from development testing to production-ready multi-worker configurations with load balancing, monitoring, and performance tuning.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Development Deployment](#development-deployment)
4. [Production Deployment](#production-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Minimum (Development):**
- GPU: NVIDIA T4 or better
- VRAM: 16 GB
- RAM: 32 GB
- Storage: 100 GB SSD

**Recommended (Production):**
- GPU: NVIDIA RTX A6000 or A100
- VRAM: 40-48 GB
- RAM: 64-128 GB
- Storage: 500 GB NVMe SSD
- Network: 10 Gbps

**Current Test Environment:**
- Primary GPU: NVIDIA RTX A6000 (48 GB VRAM)
- Secondary GPUs: 2x Quadro RTX 6000 (24 GB each)
- Driver: 580.105.08
- Total GPU Memory: 96 GB

### Software Requirements

**Operating System:**
- Ubuntu 22.04 LTS (recommended)
- RHEL 8.x / Rocky Linux 8.x
- Debian 11+

**Core Dependencies:**
```bash
Python 3.13+
CUDA 13.0+
cuDNN 9.x
TensorRT 10.14+ (optional, 3-5x speedup)
```

**Python Libraries:**
```bash
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
flask>=3.0.0
gunicorn>=21.0.0
```

**Optional Production Components:**
```bash
nginx>=1.18.0          # Load balancer
redis>=7.0.0           # Caching layer
prometheus>=2.40.0     # Metrics
grafana>=9.0.0         # Dashboards
```

---

## Installation

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential git wget curl
sudo apt install -y python3.13 python3.13-venv python3.13-dev
```

### 2. NVIDIA Drivers and CUDA

```bash
# Install NVIDIA drivers (if not present)
sudo apt install -y nvidia-driver-580

# Verify GPU detection
nvidia-smi

# Install CUDA Toolkit 13.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-13-0

# Add to PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
```

### 3. TensorRT Installation (Optional but Recommended)

**Option 1: Debian Package (Recommended)**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install TensorRT
sudo apt install -y tensorrt python3-libnvinfer-dev

# Verify installation
dpkg -l | grep tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

**Option 2: NVIDIA Container (Production)**
```bash
# Use NVIDIA PyTorch container with TensorRT
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

### 4. Python Environment

```bash
# Clone repository
git clone https://github.com/jjohare/hackathon-tv5.git
cd hackathon-tv5/semantic-recommender

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install project dependencies
pip install -r scripts/requirements.txt

# Install TensorRT Python bindings (if using TensorRT)
pip install tensorrt pycuda onnx
```

### 5. Data Preparation

```bash
# Create data directories
mkdir -p data/{raw,embeddings/media,models}

# Download MovieLens dataset
cd data/raw
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
cd ../..

# Process dataset
python scripts/data_pipeline/parse_movielens.py

# Generate embeddings (10-15 minutes on RTX A6000)
python scripts/ops/generate_embeddings.py

# Expected output:
# ✅ Generated embeddings for 62,423 movies
# 📁 Saved to: data/embeddings/media/content_vectors.npy
# 📋 Metadata: data/embeddings/media/metadata.jsonl
```

### 6. TensorRT Engine Build (Optional)

```bash
# Convert model to ONNX
python scripts/ops/convert_to_onnx.py

# Build TensorRT engine (3-5x speedup)
python scripts/ops/build_trt_engine.py

# Expected output:
# ✅ Engine saved: data/models/minilm_l12_v2_fp16.plan (28 MB)
# ⚡ Expected speedup: 3-5x vs PyTorch
```

---

## Development Deployment

### Flask Development Server

**For local testing and development only. NOT for production.**

#### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Run development server
python scripts/utils/gpu_hyper_personalization.py --test

# Test with curl
curl http://localhost:5000/health
```

#### Configuration

Create `config/development.env`:
```bash
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# GPU Settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model Settings
USE_TENSORRT=false
BATCH_SIZE=1
TOP_K=10
```

Load configuration:
```bash
export $(cat config/development.env | xargs)
python scripts/utils/gpu_hyper_personalization.py --test
```

#### Limitations

- ❌ Single process (no parallelism)
- ❌ No load balancing
- ❌ Low throughput (~100 req/s)
- ❌ No automatic restart on crash
- ✅ Easy debugging with Flask debug mode
- ✅ Fast iteration for development

---

## Production Deployment

### Architecture Overview

```
Internet
   ↓
[Nginx Load Balancer]
   ↓
[Gunicorn Workers] × N
   ↓
[GPU Inference Engine]
   ↓
[Data Layer: Embeddings + Metadata]
```

### 1. Gunicorn Multi-Worker Setup

#### Configuration File

Create `config/gunicorn.conf.py`:
```python
# gunicorn.conf.py
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 4  # Adjust based on GPU memory
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Resource limits
max_requests = 1000  # Restart workers after N requests (prevents memory leaks)
max_requests_jitter = 50
graceful_timeout = 30

# Logging
accesslog = "/var/log/semantic-recommender/access.log"
errorlog = "/var/log/semantic-recommender/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "semantic-recommender"

# Server mechanics
daemon = False
pidfile = "/var/run/semantic-recommender.pid"
user = None
group = None
tmp_upload_dir = None

# Preload app (faster worker startup)
preload_app = True

# Worker lifecycle hooks
def on_starting(server):
    """Called before master process is initialized"""
    import torch
    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

def when_ready(server):
    """Called after server is started"""
    server.log.info("Server is ready. GPU workers starting...")

def worker_init(worker):
    """Called when a worker is initialized"""
    import torch
    worker.log.info(f"Worker {worker.pid} initialized on GPU")

def worker_exit(server, worker):
    """Called when a worker exits"""
    import torch
    torch.cuda.empty_cache()
    server.log.info(f"Worker {worker.pid} cleaned up GPU memory")
```

#### Worker Count Tuning

**Formula:**
```
workers = min(GPU_memory_GB / model_memory_GB, CPU_cores / 2)
```

**Examples:**

| GPU | VRAM | Model Size | Workers | Throughput |
|-----|------|------------|---------|------------|
| RTX A6000 | 48 GB | 2 GB | 8-12 | ~200K QPS |
| A100 | 40 GB | 2 GB | 8-10 | ~316K QPS |
| T4 | 16 GB | 2 GB | 4-6 | ~50K QPS |

**RTX A6000 Configuration:**
```python
# For RTX A6000 (48 GB VRAM)
workers = 8  # Conservative (6 GB per worker)
worker_class = "sync"
max_requests = 2000
```

#### Start Gunicorn

```bash
# Create log directory
sudo mkdir -p /var/log/semantic-recommender
sudo chown $USER:$USER /var/log/semantic-recommender

# Start Gunicorn
gunicorn \
  --config config/gunicorn.conf.py \
  --chdir /path/to/semantic-recommender \
  scripts.utils.gpu_hyper_personalization:app

# Or use wrapper script (recommended)
./scripts/ops/start_gunicorn.sh
```

#### Systemd Service

Create `/etc/systemd/system/semantic-recommender.service`:
```ini
[Unit]
Description=Semantic Recommender GPU Service
After=network.target

[Service]
Type=notify
User=recommender
Group=recommender
WorkingDirectory=/opt/semantic-recommender
Environment="PATH=/opt/semantic-recommender/venv/bin:/usr/local/cuda-13.0/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/semantic-recommender/venv/bin/gunicorn \
  --config /opt/semantic-recommender/config/gunicorn.conf.py \
  scripts.utils.gpu_hyper_personalization:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable semantic-recommender
sudo systemctl start semantic-recommender
sudo systemctl status semantic-recommender

# View logs
sudo journalctl -u semantic-recommender -f
```

### 2. Nginx Load Balancer

#### Installation

```bash
sudo apt install -y nginx
```

#### Configuration

Create `/etc/nginx/sites-available/semantic-recommender`:
```nginx
upstream semantic_backend {
    # Gunicorn workers
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;

    # For multi-GPU setup (optional)
    # server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    # server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;

    keepalive 64;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    listen 80;
    server_name semantic-recommender.example.com;

    # Request limits
    client_max_body_size 1M;
    client_body_timeout 10s;
    client_header_timeout 10s;

    # Logging
    access_log /var/log/nginx/semantic-recommender-access.log combined;
    error_log /var/log/nginx/semantic-recommender-error.log warn;

    # Health check endpoint (no rate limit)
    location /health {
        proxy_pass http://semantic_backend;
        proxy_set_header Host $host;
        access_log off;
    }

    # API endpoints
    location /api/ {
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
        limit_conn conn_limit 10;

        # Proxy settings
        proxy_pass http://semantic_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;

        # HTTP 1.1 keepalive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # Metrics endpoint (Prometheus)
    location /metrics {
        proxy_pass http://semantic_backend;
        access_log off;
        allow 10.0.0.0/8;  # Internal only
        deny all;
    }
}
```

Enable and test:
```bash
# Create symlink
sudo ln -s /etc/nginx/sites-available/semantic-recommender /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx

# Test endpoint
curl http://localhost/health
```

#### SSL/TLS (Production)

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d semantic-recommender.example.com

# Auto-renewal test
sudo certbot renew --dry-run
```

Updated nginx config with SSL:
```nginx
server {
    listen 443 ssl http2;
    server_name semantic-recommender.example.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/semantic-recommender.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/semantic-recommender.example.com/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # ... rest of config
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name semantic-recommender.example.com;
    return 301 https://$host$request_uri;
}
```

### 3. Multi-GPU Setup

#### GPU Sharding Configuration

For multiple GPUs, run separate Gunicorn instances per GPU:

**GPU 0 (RTX A6000):**
```bash
CUDA_VISIBLE_DEVICES=0 gunicorn \
  --bind 127.0.0.1:8000 \
  --workers 8 \
  --config config/gunicorn.conf.py \
  scripts.utils.gpu_hyper_personalization:app
```

**GPU 1 (Quadro RTX 6000):**
```bash
CUDA_VISIBLE_DEVICES=1 gunicorn \
  --bind 127.0.0.1:8001 \
  --workers 4 \
  --config config/gunicorn.conf.py \
  scripts.utils.gpu_hyper_personalization:app
```

**GPU 2 (Quadro RTX 6000):**
```bash
CUDA_VISIBLE_DEVICES=2 gunicorn \
  --bind 127.0.0.1:8002 \
  --workers 4 \
  --config config/gunicorn.conf.py \
  scripts.utils.gpu_hyper_personalization:app
```

#### Nginx Load Balancing

Update upstream block:
```nginx
upstream semantic_backend {
    least_conn;  # Route to least-busy worker

    server 127.0.0.1:8000 weight=2 max_fails=3;  # RTX A6000 (48GB)
    server 127.0.0.1:8001 weight=1 max_fails=3;  # Quadro RTX 6000 (24GB)
    server 127.0.0.1:8002 weight=1 max_fails=3;  # Quadro RTX 6000 (24GB)

    keepalive 128;
}
```

**Expected Throughput:**
- GPU 0 (RTX A6000): ~200K QPS
- GPU 1 (Quadro RTX 6000): ~100K QPS
- GPU 2 (Quadro RTX 6000): ~100K QPS
- **Total: ~400K QPS**

### 4. Environment Variables

Create `/opt/semantic-recommender/.env`:
```bash
# Application
APP_NAME=semantic-recommender
APP_VERSION=1.0.0
ENVIRONMENT=production

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model Configuration
USE_TENSORRT=true
MODEL_PATH=data/models/minilm_l12_v2_fp16.plan
EMBEDDINGS_PATH=data/embeddings/media/content_vectors.npy
METADATA_PATH=data/embeddings/media/metadata.jsonl

# Inference Settings
BATCH_SIZE=100
TOP_K=10
QUERY_WEIGHT=0.7

# Performance
MAX_BATCH_SIZE=1000
WORKER_THREADS=4

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

---

## Performance Tuning

### 1. Batch Size Optimization

**Trade-off:** Latency vs Throughput

| Batch Size | Latency | Throughput | Use Case |
|------------|---------|------------|----------|
| 1 | 0.5ms | 2K QPS | Real-time API |
| 10 | 0.8ms | 12K QPS | Web applications |
| 100 | 3ms | 123K QPS | Batch processing |
| 1000 | 15ms | 316K QPS | Offline analytics |

**Configuration:**
```python
# In gpu_hyper_personalization.py
BATCH_SIZE = 100  # Optimal for web apps

# Dynamic batching (advanced)
def adaptive_batch_size(queue_depth):
    if queue_depth < 10:
        return 1  # Low latency
    elif queue_depth < 100:
        return 10
    else:
        return 100  # High throughput
```

### 2. Worker Count Tuning

**Formula:**
```python
# CPU-bound tasks
workers = min(CPU_cores, 2 * CPU_cores + 1)

# GPU-bound tasks (our case)
workers = GPU_memory_GB // model_memory_per_worker_GB

# Example: RTX A6000 (48 GB VRAM)
# Model uses ~2 GB per worker
workers = 48 // 2 = 24  # Theoretical max
workers = 12  # Practical (leaves headroom)
```

**Monitoring Worker Load:**
```bash
# Check Gunicorn workers
ps aux | grep gunicorn

# Monitor GPU memory per process
nvidia-smi pmon -c 1

# Check worker utilization
curl http://localhost:8000/metrics | grep worker
```

### 3. GPU Memory Management

**PyTorch Settings:**
```python
import torch

# Enable TF32 for A100/RTX 3090+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Memory allocation strategy
torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # 80% max per worker

# Empty cache between requests (if needed)
torch.cuda.empty_cache()

# Gradients not needed for inference
with torch.no_grad():
    results = model(input_tensors)
```

**Environment Variables:**
```bash
# Fragment memory less aggressively
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For debugging memory issues
export CUDA_LAUNCH_BLOCKING=1
```

### 4. TensorRT Optimization

**Build Engine with Optimal Profile:**
```python
# In build_trt_engine.py
PROFILE_CONFIG = {
    'min': (1, 1),      # Single query
    'opt': (1, 32),     # Typical web query
    'max': (100, 128)   # Batch processing
}

# Precision
USE_FP16 = True  # 2x speedup on A100/RTX A6000
USE_INT8 = False # 4x speedup but requires calibration
```

**Expected Performance:**
- PyTorch: 5-10ms per query
- TensorRT FP16: 1-2ms per query (3-5x faster)
- TensorRT INT8: 0.5-1ms per query (5-10x faster)

### 5. Connection Pooling

**Nginx Keepalive:**
```nginx
upstream semantic_backend {
    server 127.0.0.1:8000;
    keepalive 128;  # Connection pool
}

# In location block
proxy_http_version 1.1;
proxy_set_header Connection "";
```

**Gunicorn Keepalive:**
```python
# gunicorn.conf.py
keepalive = 5  # Keep connections alive for 5s
worker_connections = 1000  # Max connections per worker
```

---

## Monitoring

### 1. Metrics to Track

**System Metrics:**
- GPU utilization (target: 80-95%)
- GPU memory usage (target: 60-80%)
- GPU temperature (target: <80°C)
- PCIe bandwidth utilization
- CPU load average
- RAM usage

**Application Metrics:**
- Requests per second (QPS)
- Latency (p50, p95, p99)
- Error rate (target: <0.1%)
- Worker restarts
- Queue depth

**Business Metrics:**
- Recommendation quality (CTR)
- User engagement
- Cost per query

### 2. Prometheus Integration

**Install Prometheus:**
```bash
# Download and install
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar -xvf prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64
```

**Configure Prometheus (`prometheus.yml`):**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Application metrics
  - job_name: 'semantic-recommender'
    static_configs:
      - targets: ['localhost:9090']

  # NVIDIA GPU metrics
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9400']

  # Node exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

**Start Prometheus:**
```bash
./prometheus --config.file=prometheus.yml
```

**Add Metrics to Application:**
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request counter
request_count = Counter('requests_total', 'Total requests', ['method', 'endpoint'])

# Latency histogram
request_latency = Histogram('request_duration_seconds', 'Request latency')

# GPU memory gauge
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used')

# Endpoint
@app.route('/metrics')
def metrics():
    return generate_latest()
```

### 3. Grafana Dashboards

**Install Grafana:**
```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y grafana

sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

**Access:** http://localhost:3000 (default: admin/admin)

**Create Dashboard:**
1. Add Prometheus data source
2. Import dashboard ID: 12239 (NVIDIA GPU metrics)
3. Create custom panels:

**Throughput Panel:**
```promql
rate(requests_total[5m])
```

**Latency Panel:**
```promql
histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))
```

**GPU Utilization Panel:**
```promql
nvidia_gpu_utilization_percent{gpu="0"}
```

### 4. Log Analysis

**Structured Logging:**
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }
        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger('semantic-recommender')
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

**Log Aggregation (ELK Stack):**
```bash
# Install Filebeat
curl -L -O https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.8.0-amd64.deb
sudo dpkg -i filebeat-8.8.0-amd64.deb

# Configure filebeat.yml
filebeat.inputs:
  - type: log
    paths:
      - /var/log/semantic-recommender/*.log
    json.keys_under_root: true

# Start filebeat
sudo systemctl enable filebeat
sudo systemctl start filebeat
```

### 5. Alerting

**Prometheus Alert Rules (`alerts.yml`):**
```yaml
groups:
  - name: semantic-recommender
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(requests_total{status="500"}[5m]) > 0.01
        for: 5m
        annotations:
          summary: "High error rate detected"

      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        annotations:
          summary: "P95 latency > 100ms"

      # GPU memory
      - alert: GPUMemoryHigh
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
        for: 5m
        annotations:
          summary: "GPU memory usage > 90%"

      # GPU temperature
      - alert: GPUTemperatureHigh
        expr: nvidia_gpu_temperature_celsius > 85
        for: 5m
        annotations:
          summary: "GPU temperature > 85°C"
```

---

## Troubleshooting

### 1. TensorRT Errors

#### Error: "Module 'tensorrt' not found"

**Symptoms:**
```
ImportError: No module named 'tensorrt'
```

**Solution 1: Install TensorRT Python package**
```bash
pip install tensorrt

# Verify
python -c "import tensorrt; print(tensorrt.__version__)"
```

**Solution 2: Use NVIDIA container**
```bash
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
# TensorRT is pre-installed
```

**Solution 3: Install from .whl file**
```bash
# Download from NVIDIA Developer
# https://developer.nvidia.com/tensorrt

pip install tensorrt-10.14.0-cp313-none-linux_x86_64.whl
pip install tensorrt-10.14.0/python/tensorrt-10.14.0-cp313-none-linux_x86_64.whl
```

#### Error: "Failed to parse ONNX model"

**Symptoms:**
```
[TensorRT] ERROR: Failed to parse onnx file
```

**Diagnosis:**
```bash
# Check ONNX validity
python -c "import onnx; onnx.checker.check_model('data/models/minilm_l12_v2.onnx')"

# Check opset version
python -c "import onnx; model = onnx.load('data/models/minilm_l12_v2.onnx'); print(model.opset_import)"
```

**Solution:**
```bash
# Rebuild ONNX with compatible opset
python scripts/ops/convert_to_onnx.py --opset 14

# Verify file integrity
ls -lh data/models/minilm_l12_v2.onnx
```

#### Error: "FP16 not supported"

**Symptoms:**
```
[TensorRT] WARNING: FP16 is not supported on this platform
```

**Cause:** GPU compute capability < 7.0 (Pascal or older)

**Check GPU:**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Solution:**
```python
# In build_trt_engine.py, disable FP16
TRT_CONFIG = {
    'fp16': False,  # Use FP32 instead
    'workspace_size': 2048
}
```

### 2. Memory Issues

#### Error: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnosis:**
```bash
# Check GPU memory
nvidia-smi

# Check per-process memory
nvidia-smi pmon -c 1
```

**Solution 1: Reduce worker count**
```python
# gunicorn.conf.py
workers = 4  # Reduce from 8
```

**Solution 2: Reduce batch size**
```python
# In application
BATCH_SIZE = 10  # Reduce from 100
```

**Solution 3: Clear cache**
```python
import torch
torch.cuda.empty_cache()
```

**Solution 4: Limit memory per process**
```python
torch.cuda.set_per_process_memory_fraction(0.8, device=0)
```

#### Error: "Too many active users"

**Symptoms:**
```
RuntimeError: Exceeded max active users (100000)
```

**Solution:**
```python
# In GPUUserEmbeddings class
self.max_active_users = 1_000_000  # Increase limit

# Or implement LRU eviction
from collections import OrderedDict
self.user_cache = OrderedDict()
if len(self.user_cache) > MAX_USERS:
    self.user_cache.popitem(last=False)  # Remove oldest
```

### 3. Performance Issues

#### Issue: Low GPU utilization (<50%)

**Diagnosis:**
```bash
# Monitor GPU utilization
nvidia-smi dmon -s u -c 10

# Check batch size
curl http://localhost:8000/metrics | grep batch_size
```

**Solution:**
```python
# Increase batch size
BATCH_SIZE = 100

# Increase workers
workers = 8

# Enable TensorRT
USE_TENSORRT = True
```

#### Issue: High latency (>10ms)

**Diagnosis:**
```bash
# Check query latency distribution
curl http://localhost:8000/metrics | grep request_duration

# Profile with PyTorch profiler
python -m torch.utils.bottleneck scripts/utils/gpu_hyper_personalization.py
```

**Solution:**
```python
# Enable TF32 (A100/RTX 3090+)
torch.backends.cuda.matmul.allow_tf32 = True

# Use TensorRT
USE_TENSORRT = True

# Preload embeddings to GPU
embeddings = embeddings.to('cuda')
```

#### Issue: Worker crashes/restarts

**Diagnosis:**
```bash
# Check Gunicorn logs
tail -f /var/log/semantic-recommender/error.log

# Check system logs
sudo journalctl -u semantic-recommender -f
```

**Common Causes:**
1. **Memory leaks:** Set `max_requests = 1000`
2. **GPU memory fragmentation:** Clear cache periodically
3. **Timeout:** Increase `timeout = 60` in gunicorn.conf.py

### 4. Network Issues

#### Issue: 502 Bad Gateway (Nginx)

**Diagnosis:**
```bash
# Check Gunicorn status
systemctl status semantic-recommender

# Check Gunicorn logs
tail -f /var/log/semantic-recommender/error.log

# Test backend directly
curl http://127.0.0.1:8000/health
```

**Solution:**
```bash
# Restart Gunicorn
sudo systemctl restart semantic-recommender

# Check nginx logs
tail -f /var/log/nginx/error.log
```

#### Issue: Connection timeouts

**Diagnosis:**
```bash
# Check nginx timeouts
grep timeout /etc/nginx/sites-enabled/semantic-recommender

# Check Gunicorn timeout
grep timeout config/gunicorn.conf.py
```

**Solution:**
```nginx
# Increase nginx timeouts
proxy_connect_timeout 10s;
proxy_send_timeout 60s;
proxy_read_timeout 60s;
```

```python
# Increase Gunicorn timeout
timeout = 60
graceful_timeout = 60
```

### 5. Debugging Tools

**GPU Debugging:**
```bash
# Detailed GPU info
nvidia-smi -q

# Monitor GPU continuously
watch -n 1 nvidia-smi

# Check CUDA errors
cuda-gdb python script.py
```

**Application Debugging:**
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Profile inference
python -m cProfile -o profile.stats scripts/utils/gpu_hyper_personalization.py

# Analyze profile
python -m pstats profile.stats
```

**Network Debugging:**
```bash
# Check open connections
netstat -an | grep 8000

# Monitor requests
tcpdump -i any port 8000

# Load testing
ab -n 10000 -c 100 http://localhost/api/search
```

---

## Performance Checklist

**Before Production:**
- [ ] GPU drivers updated (580.105.08+)
- [ ] CUDA Toolkit installed (13.0+)
- [ ] TensorRT engine built and validated
- [ ] Worker count optimized for GPU memory
- [ ] Batch size tuned for latency/throughput trade-off
- [ ] Nginx load balancer configured
- [ ] SSL/TLS certificates installed
- [ ] Monitoring (Prometheus + Grafana) deployed
- [ ] Log aggregation configured
- [ ] Alerts configured for critical metrics
- [ ] Backup and disaster recovery plan
- [ ] Load testing completed (target: 100K QPS)

**Production Optimization:**
- [ ] TF32 enabled for A100/RTX 3090+
- [ ] Connection pooling (nginx keepalive)
- [ ] HTTP/2 enabled
- [ ] Gzip compression for API responses
- [ ] CDN for static assets
- [ ] Rate limiting configured
- [ ] DDoS protection enabled
- [ ] Auto-scaling rules defined

---

## Quick Reference

**Start Services:**
```bash
# Development
source venv/bin/activate
python scripts/utils/gpu_hyper_personalization.py --test

# Production
sudo systemctl start semantic-recommender
sudo systemctl start nginx
```

**Check Status:**
```bash
# Application
curl http://localhost/health

# GPU
nvidia-smi

# Logs
tail -f /var/log/semantic-recommender/error.log
```

**Performance Tests:**
```bash
# Single query
time curl http://localhost/api/search?q=inception

# Load test
ab -n 10000 -c 100 http://localhost/api/search?q=inception
```

---

**Deployment Complete! Expected Performance:**
- Latency: <1ms (warm), 90ms (cold)
- Throughput: 200K+ QPS (RTX A6000)
- GPU Utilization: 80-95%
- Availability: 99.9%+
