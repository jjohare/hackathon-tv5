#!/bin/bash
set -euo pipefail

# Milvus Cluster Health Check Script
# Validates all components and reports issues

NAMESPACE="milvus-prod"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

ISSUES=0

# Check etcd cluster
check_etcd() {
    echo "=== etcd Health ==="
    ETCD_READY=$(kubectl get statefulset etcd -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    ETCD_DESIRED=$(kubectl get statefulset etcd -n ${NAMESPACE} -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)

    if [ "${ETCD_READY}" -eq "${ETCD_DESIRED}" ] && [ "${ETCD_READY}" -ge 2 ]; then
        log_info "etcd cluster healthy (${ETCD_READY}/${ETCD_DESIRED})"
    else
        log_error "etcd cluster unhealthy (${ETCD_READY}/${ETCD_DESIRED})"
        ((ISSUES++))
    fi
}

# Check MinIO
check_minio() {
    echo "=== MinIO Health ==="
    MINIO_READY=$(kubectl get statefulset minio -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    MINIO_DESIRED=$(kubectl get statefulset minio -n ${NAMESPACE} -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)

    if [ "${MINIO_READY}" -eq "${MINIO_DESIRED}" ] && [ "${MINIO_READY}" -ge 3 ]; then
        log_info "MinIO cluster healthy (${MINIO_READY}/${MINIO_DESIRED})"
    else
        log_error "MinIO cluster unhealthy (${MINIO_READY}/${MINIO_DESIRED})"
        ((ISSUES++))
    fi
}

# Check Pulsar
check_pulsar() {
    echo "=== Pulsar Health ==="
    PULSAR_READY=$(kubectl get statefulset pulsar -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    PULSAR_DESIRED=$(kubectl get statefulset pulsar -n ${NAMESPACE} -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)

    if [ "${PULSAR_READY}" -eq "${PULSAR_DESIRED}" ] && [ "${PULSAR_READY}" -ge 2 ]; then
        log_info "Pulsar cluster healthy (${PULSAR_READY}/${PULSAR_DESIRED})"
    else
        log_error "Pulsar cluster unhealthy (${PULSAR_READY}/${PULSAR_DESIRED})"
        ((ISSUES++))
    fi
}

# Check coordinators
check_coordinators() {
    echo "=== Milvus Coordinators ==="
    for COORD in rootcoord datacoord querycoord indexcoord; do
        READY=$(kubectl get deployment ${COORD} -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
        if [ "${READY}" -ge 1 ]; then
            log_info "${COORD} ready"
        else
            log_error "${COORD} not ready"
            ((ISSUES++))
        fi
    done
}

# Check query nodes
check_query_nodes() {
    echo "=== Query Nodes (GPU) ==="
    READY=$(kubectl get ds milvus-querynode -n ${NAMESPACE} -o jsonpath='{.status.numberReady}' 2>/dev/null || echo 0)
    DESIRED=$(kubectl get ds milvus-querynode -n ${NAMESPACE} -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo 0)

    if [ "${READY}" -ge 90 ]; then
        log_info "Query nodes healthy (${READY}/${DESIRED})"
    elif [ "${READY}" -ge 70 ]; then
        log_warn "Query nodes degraded (${READY}/${DESIRED})"
        ((ISSUES++))
    else
        log_error "Query nodes critical (${READY}/${DESIRED})"
        ((ISSUES+=2))
    fi

    # Check GPU allocation
    GPU_PODS=$(kubectl get pods -n ${NAMESPACE} -l component=querynode --field-selector=status.phase=Running --no-headers | wc -l)
    log_info "Query nodes with GPUs: ${GPU_PODS}"
}

# Check other workers
check_workers() {
    echo "=== Worker Nodes ==="

    # Data nodes
    DATA_READY=$(kubectl get deployment datanode -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    if [ "${DATA_READY}" -ge 15 ]; then
        log_info "Data nodes healthy (${DATA_READY})"
    else
        log_warn "Data nodes degraded (${DATA_READY})"
        ((ISSUES++))
    fi

    # Index nodes
    INDEX_READY=$(kubectl get deployment indexnode -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    if [ "${INDEX_READY}" -ge 7 ]; then
        log_info "Index nodes healthy (${INDEX_READY})"
    else
        log_warn "Index nodes degraded (${INDEX_READY})"
        ((ISSUES++))
    fi
}

# Check proxy
check_proxy() {
    echo "=== Milvus Proxy ==="
    PROXY_READY=$(kubectl get deployment milvus-proxy -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    PROXY_DESIRED=$(kubectl get deployment milvus-proxy -n ${NAMESPACE} -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)

    if [ "${PROXY_READY}" -eq "${PROXY_DESIRED}" ] && [ "${PROXY_READY}" -ge 8 ]; then
        log_info "Proxy healthy (${PROXY_READY}/${PROXY_DESIRED})"
    else
        log_error "Proxy unhealthy (${PROXY_READY}/${PROXY_DESIRED})"
        ((ISSUES++))
    fi
}

# Check connectivity
check_connectivity() {
    echo "=== Connectivity Test ==="
    if kubectl run -n ${NAMESPACE} milvus-health-test --image=milvusdb/milvus:v2.4.0 --rm --restart=Never --timeout=30s -- \
        milvus-cli -h milvus-proxy -p 19530 -c "show collections" &>/dev/null; then
        log_info "Milvus proxy reachable"
    else
        log_warn "Milvus proxy unreachable (may not have collections yet)"
    fi
}

# Check storage
check_storage() {
    echo "=== Storage Health ==="
    PVC_BOUND=$(kubectl get pvc -n ${NAMESPACE} --field-selector=status.phase=Bound --no-headers | wc -l)
    PVC_TOTAL=$(kubectl get pvc -n ${NAMESPACE} --no-headers | wc -l)

    if [ "${PVC_BOUND}" -eq "${PVC_TOTAL}" ]; then
        log_info "All PVCs bound (${PVC_BOUND}/${PVC_TOTAL})"
    else
        log_error "Some PVCs not bound (${PVC_BOUND}/${PVC_TOTAL})"
        ((ISSUES++))
    fi
}

# Main
main() {
    echo "======================================"
    echo "Milvus Cluster Health Check"
    echo "Namespace: ${NAMESPACE}"
    echo "======================================"
    echo ""

    check_etcd
    echo ""
    check_minio
    echo ""
    check_pulsar
    echo ""
    check_coordinators
    echo ""
    check_query_nodes
    echo ""
    check_workers
    echo ""
    check_proxy
    echo ""
    check_storage
    echo ""
    check_connectivity

    echo ""
    echo "======================================"
    if [ ${ISSUES} -eq 0 ]; then
        log_info "Cluster health: HEALTHY"
        exit 0
    elif [ ${ISSUES} -le 3 ]; then
        log_warn "Cluster health: DEGRADED (${ISSUES} issues)"
        exit 1
    else
        log_error "Cluster health: CRITICAL (${ISSUES} issues)"
        exit 2
    fi
}

main "$@"
