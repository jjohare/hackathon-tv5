#!/bin/bash
# Start Prometheus for performance monitoring

PROMETHEUS_CONFIG="/home/devuser/workspace/hackathon-tv5/config/prometheus.yml"
PROMETHEUS_DATA="/home/devuser/workspace/hackathon-tv5/data/prometheus"

mkdir -p $(dirname ${PROMETHEUS_DATA})

# Create Prometheus config if it doesn't exist
if [ ! -f "${PROMETHEUS_CONFIG}" ]; then
    mkdir -p $(dirname ${PROMETHEUS_CONFIG})
    cat > ${PROMETHEUS_CONFIG} <<EOF
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'hybrid-coordinator'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'dcgm-exporter'
    static_configs:
      - targets: ['localhost:9400']

  - job_name: 'milvus'
    static_configs:
      - targets: ['localhost:9091']

  - job_name: 'neo4j'
    static_configs:
      - targets: ['localhost:2004']
EOF
fi

echo "Starting Prometheus..."
prometheus \
    --config.file=${PROMETHEUS_CONFIG} \
    --storage.tsdb.path=${PROMETHEUS_DATA} \
    --web.listen-address=:9090 \
    --web.enable-lifecycle \
    &

echo "Prometheus started on http://localhost:9090"
