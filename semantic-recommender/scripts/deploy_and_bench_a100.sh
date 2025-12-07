#!/bin/bash
# Complete A100 Deployment and Benchmarking Script

set -e

echo "================================================================================"
echo "GPU HYPER-PERSONALIZATION - A100 DEPLOYMENT & BENCHMARKING"
echo "================================================================================"
echo ""

# Configuration
VM_NAME="semantic-recommender-a100"
ZONE="us-central1-a"
PROJECT_DIR="/opt/hackathon-tv5/semantic-recommender"

echo "Step 1: Package hyper-personalization system"
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender

tar -czf /tmp/hyper_personalization_full.tar.gz \
  scripts/gpu_hyper_personalization.py \
  scripts/benchmark_hyper_personalization.py \
  data/embeddings/media/content_vectors.npy \
  data/embeddings/media/metadata.jsonl \
  docs/BREAKTHROUGH_ARCHITECTURE.md \
  docs/HYPER_PERSONALIZATION_DEPLOYMENT.md

PACKAGE_SIZE=$(du -h /tmp/hyper_personalization_full.tar.gz | cut -f1)
echo "✓ Package created: ${PACKAGE_SIZE}"
echo ""

echo "Step 2: Transfer to A100"
export PATH="/home/devuser/google-cloud-sdk/bin:$PATH"

gcloud compute scp /tmp/hyper_personalization_full.tar.gz \
  ${VM_NAME}:/tmp/ --zone=${ZONE}

echo "✓ Transfer complete"
echo ""

echo "Step 3: Deploy and run benchmarks on A100"
gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
set -e

echo '=== Extracting package ==='
cd ${PROJECT_DIR}
tar -xzf /tmp/hyper_personalization_full.tar.gz

echo ''
echo '=== Activating GPU environment ==='
source /opt/conda/bin/activate pytorch_gpu

echo ''
echo '=== Verifying CUDA ==='
python -c \"import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\"

echo ''
echo '=== Running Demo Test ==='
timeout 120 python scripts/gpu_hyper_personalization.py --test

echo ''
echo '=== Running Comprehensive Benchmarks ==='
timeout 600 python scripts/benchmark_hyper_personalization.py

echo ''
echo '=== Benchmark Results ==='
if [ -f docs/HYPER_PERSONALIZATION_RESULTS.json ]; then
  cat docs/HYPER_PERSONALIZATION_RESULTS.json
else
  echo 'Warning: Results file not found'
fi
"

echo ""
echo "Step 4: Retrieve benchmark results"
gcloud compute scp ${VM_NAME}:${PROJECT_DIR}/docs/HYPER_PERSONALIZATION_RESULTS.json \
  /tmp/ --zone=${ZONE} 2>/dev/null || echo "Note: Results file may not exist yet"

echo ""
echo "================================================================================"
echo "✓ DEPLOYMENT AND BENCHMARKING COMPLETE"
echo "================================================================================"
echo ""
echo "Results available at: /tmp/HYPER_PERSONALIZATION_RESULTS.json"
