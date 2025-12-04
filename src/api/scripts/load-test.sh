#!/bin/bash

# Load test script for API performance validation
# Target: 7,000 QPS with <10ms overhead

API_URL="http://localhost:3000"
DURATION=30
CONNECTIONS=100
THREADS=8

echo "Starting API load test..."
echo "Target: 7,000 QPS"
echo "Duration: ${DURATION}s"
echo "Connections: ${CONNECTIONS}"
echo ""

# Test 1: Health endpoint baseline
echo "=== Test 1: Health endpoint baseline ==="
wrk -t${THREADS} -c${CONNECTIONS} -d${DURATION}s --latency \
  "${API_URL}/health"

echo ""
echo "=== Test 2: Search endpoint ==="
wrk -t${THREADS} -c${CONNECTIONS} -d${DURATION}s --latency \
  -s search-payload.lua "${API_URL}/api/v1/search"

echo ""
echo "=== Test 3: Recommendations endpoint ==="
wrk -t${THREADS} -c${CONNECTIONS} -d${DURATION}s --latency \
  "${API_URL}/api/v1/recommendations/user_123?limit=10"

echo ""
echo "=== Test 4: MCP manifest endpoint ==="
wrk -t${THREADS} -c${CONNECTIONS} -d${DURATION}s --latency \
  "${API_URL}/api/v1/mcp/manifest"

echo ""
echo "Load test complete!"
