#!/bin/bash
set -euo pipefail

# Scale Milvus Query Nodes Horizontally
# Automatically adjusts based on GPU node availability

NAMESPACE="milvus-prod"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Get current GPU node count
get_gpu_nodes() {
    kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-Tesla-T4 --no-headers | wc -l
}

# Get current query node count
get_query_nodes() {
    kubectl get ds milvus-querynode -n ${NAMESPACE} -o jsonpath='{.status.numberReady}' 2>/dev/null || echo 0
}

# Main
main() {
    GPU_NODES=$(get_gpu_nodes)
    QUERY_NODES=$(get_query_nodes)

    log_info "Current state:"
    echo "  GPU nodes: ${GPU_NODES}"
    echo "  Query nodes: ${QUERY_NODES}"

    if [ "${QUERY_NODES}" -lt "${GPU_NODES}" ]; then
        log_warn "Query nodes (${QUERY_NODES}) < GPU nodes (${GPU_NODES})"
        log_info "DaemonSet will automatically schedule missing query nodes"

        # Trigger rollout restart to accelerate scheduling
        kubectl rollout restart daemonset/milvus-querynode -n ${NAMESPACE}

        log_info "Waiting for query nodes to reach target..."
        kubectl rollout status daemonset/milvus-querynode -n ${NAMESPACE} --timeout=10m
    else
        log_info "Query nodes are properly scaled (${QUERY_NODES}/${GPU_NODES})"
    fi

    # Show final status
    echo ""
    kubectl get ds milvus-querynode -n ${NAMESPACE}
    echo ""
    kubectl get pods -n ${NAMESPACE} -l component=querynode -o wide
}

main "$@"
