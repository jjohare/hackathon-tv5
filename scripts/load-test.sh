#!/bin/bash
set -e

# Load Test Orchestration Script
# Runs comprehensive load tests against hybrid architecture

# Configuration
TARGET_QPS=${TARGET_QPS:-7000}
DURATION_SECONDS=${DURATION_SECONDS:-600}
WORKERS=${WORKERS:-100}
PROFILE=${PROFILE:-mixed}
OUTPUT_DIR="results"

echo "==================================="
echo "    Hybrid Architecture Load Test   "
echo "==================================="
echo "Target QPS:  ${TARGET_QPS}"
echo "Duration:    ${DURATION_SECONDS}s"
echo "Workers:     ${WORKERS}"
echo "Profile:     ${PROFILE}"
echo "==================================="
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Build load generator
echo "[1/5] Building load generator..."
cargo build --release --bin load-generator

# Start metrics collection
echo "[2/5] Starting metrics collection..."
if command -v prometheus &> /dev/null; then
    ./scripts/start-prometheus.sh &
    PROMETHEUS_PID=$!
    echo "Prometheus started (PID: ${PROMETHEUS_PID})"
else
    echo "Prometheus not found - skipping metrics collection"
    PROMETHEUS_PID=""
fi

# Wait for services to be ready
echo "[3/5] Waiting for services..."
sleep 5

# Run load test
echo "[4/5] Running load test..."
TIMESTAMP=$(date +%s)
OUTPUT_FILE="${OUTPUT_DIR}/load-test-${TIMESTAMP}.json"

./target/release/load-generator \
    --qps ${TARGET_QPS} \
    --duration ${DURATION_SECONDS} \
    --workers ${WORKERS} \
    --profile ${PROFILE} \
    --output ${OUTPUT_FILE}

# Analyze results
echo "[5/5] Analyzing results..."
if command -v python3 &> /dev/null; then
    python3 scripts/analyze-results.py ${OUTPUT_FILE}
else
    echo "Python not found - skipping detailed analysis"
    echo "Results saved to: ${OUTPUT_FILE}"
fi

# Stop Prometheus
if [ -n "${PROMETHEUS_PID}" ]; then
    echo "Stopping Prometheus..."
    kill ${PROMETHEUS_PID} 2>/dev/null || true
fi

# Generate summary
echo ""
echo "==================================="
echo "    Load Test Complete             "
echo "==================================="
echo "Results: ${OUTPUT_FILE}"
echo "Plots:   ${OUTPUT_DIR}/latency-analysis.png"
echo "==================================="
