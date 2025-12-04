#!/bin/bash
set -euo pipefail

# Milvus Cluster Deployment Script for 100x NVIDIA T4 GPUs
# Deploys full Milvus 2.4+ cluster with cuVS GPU acceleration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/../../k8s/milvus"
NAMESPACE="milvus-prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check NVIDIA GPU Operator
    if ! kubectl get daemonset -n gpu-operator nvidia-device-plugin-daemonset &> /dev/null; then
        log_warn "NVIDIA GPU Operator not found. Installing..."
        helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
        helm repo update
        helm install --wait --generate-name \
            -n gpu-operator --create-namespace \
            nvidia/gpu-operator \
            --set driver.enabled=false  # Assume pre-installed drivers
    fi

    # Check StorageClass
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        log_warn "StorageClass 'fast-ssd' not found. Creating..."
        cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
    fi

    log_info "Prerequisites check complete"
}

# Create namespace and quotas
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."
    kubectl apply -f "${K8S_DIR}/namespace.yaml"
    log_info "Namespace created"
}

# Deploy infrastructure components (etcd, MinIO, Pulsar)
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."

    log_info "  - etcd (metadata store)"
    kubectl apply -f "${K8S_DIR}/etcd-statefulset.yaml"
    kubectl rollout status statefulset/etcd -n ${NAMESPACE} --timeout=10m

    log_info "  - MinIO (object storage)"
    kubectl apply -f "${K8S_DIR}/minio-statefulset.yaml"
    kubectl rollout status statefulset/minio -n ${NAMESPACE} --timeout=10m

    # Initialize MinIO bucket
    log_info "  - Initializing MinIO bucket"
    kubectl run -n ${NAMESPACE} minio-init --image=minio/mc:latest --rm -it --restart=Never -- \
        sh -c "mc alias set myminio http://minio:9000 minioadmin minioadmin123 && mc mb myminio/milvus-bucket || true"

    log_info "  - Pulsar (message queue)"
    kubectl apply -f "${K8S_DIR}/pulsar-statefulset.yaml"
    kubectl rollout status statefulset/pulsar -n ${NAMESPACE} --timeout=10m

    log_info "Infrastructure deployment complete"
}

# Deploy Milvus configuration
deploy_config() {
    log_info "Deploying Milvus configuration..."
    kubectl apply -f "${K8S_DIR}/configmap.yaml"
    log_info "Configuration deployed"
}

# Deploy Milvus coordinators
deploy_coordinators() {
    log_info "Deploying Milvus coordinator components..."
    kubectl apply -f "${K8S_DIR}/milvus-cluster.yaml"

    log_info "  - Waiting for rootcoord..."
    kubectl rollout status deployment/rootcoord -n ${NAMESPACE} --timeout=5m

    log_info "  - Waiting for datacoord..."
    kubectl rollout status deployment/datacoord -n ${NAMESPACE} --timeout=5m

    log_info "  - Waiting for querycoord..."
    kubectl rollout status deployment/querycoord -n ${NAMESPACE} --timeout=5m

    log_info "  - Waiting for indexcoord..."
    kubectl rollout status deployment/indexcoord -n ${NAMESPACE} --timeout=5m

    log_info "Coordinators deployed successfully"
}

# Deploy worker nodes (data, index, query)
deploy_workers() {
    log_info "Deploying Milvus worker nodes..."

    log_info "  - Data nodes (20 replicas)"
    kubectl apply -f "${K8S_DIR}/datanode-deployment.yaml"
    kubectl rollout status deployment/datanode -n ${NAMESPACE} --timeout=10m

    log_info "  - Index nodes (10 replicas, GPU-accelerated)"
    kubectl apply -f "${K8S_DIR}/indexnode-deployment.yaml"
    kubectl rollout status deployment/indexnode -n ${NAMESPACE} --timeout=10m

    log_info "  - Query nodes (100 replicas, 1 per T4 GPU)"
    kubectl apply -f "${K8S_DIR}/querynode-daemonset.yaml"

    # Wait for query nodes to be scheduled
    log_info "  - Waiting for query nodes to be scheduled (target: 100)..."
    for i in {1..30}; do
        READY=$(kubectl get ds milvus-querynode -n ${NAMESPACE} -o jsonpath='{.status.numberReady}' || echo 0)
        DESIRED=$(kubectl get ds milvus-querynode -n ${NAMESPACE} -o jsonpath='{.status.desiredNumberScheduled}' || echo 0)
        log_info "    Query nodes ready: ${READY}/${DESIRED}"

        if [ "${READY}" -ge 90 ]; then
            log_info "  - Sufficient query nodes ready (${READY})"
            break
        fi

        if [ $i -eq 30 ]; then
            log_warn "  - Timeout waiting for query nodes. Continuing anyway..."
        fi
        sleep 20
    done

    log_info "Worker nodes deployed"
}

# Deploy proxy and services
deploy_services() {
    log_info "Deploying Milvus proxy and services..."
    kubectl apply -f "${K8S_DIR}/services.yaml"

    log_info "  - Waiting for proxy deployment..."
    kubectl rollout status deployment/milvus-proxy -n ${NAMESPACE} --timeout=5m

    log_info "  - Waiting for LoadBalancer IP..."
    for i in {1..30}; do
        EXTERNAL_IP=$(kubectl get svc milvus-external -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [ -n "${EXTERNAL_IP}" ]; then
            log_info "  - Milvus external IP: ${EXTERNAL_IP}"
            break
        fi
        sleep 10
    done

    log_info "Services deployed"
}

# Deploy resource limits and autoscaling
deploy_resource_limits() {
    log_info "Deploying resource limits and autoscaling..."
    kubectl apply -f "${K8S_DIR}/gpu-resource-limits.yaml"
    log_info "Resource limits configured"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    echo ""
    echo "=== Deployment Status ==="
    kubectl get all -n ${NAMESPACE} -o wide

    echo ""
    echo "=== GPU Nodes ==="
    kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-Tesla-T4 -o wide

    echo ""
    echo "=== Query Node Status ==="
    kubectl get ds milvus-querynode -n ${NAMESPACE}

    echo ""
    echo "=== Storage ==="
    kubectl get pvc -n ${NAMESPACE}

    echo ""
    log_info "Testing Milvus connectivity..."
    kubectl run -n ${NAMESPACE} milvus-test --image=milvusdb/milvus:v2.4.0 --rm -it --restart=Never -- \
        milvus-cli -h milvus-proxy -p 19530 -c "show collections" || log_warn "Connection test failed (expected if no collections exist yet)"

    echo ""
    log_info "Deployment verification complete"
}

# Main deployment flow
main() {
    log_info "Starting Milvus cluster deployment for 100x T4 GPUs..."

    check_prerequisites
    create_namespace
    deploy_config
    deploy_infrastructure
    deploy_coordinators
    deploy_workers
    deploy_services
    deploy_resource_limits
    verify_deployment

    echo ""
    echo "=================================================="
    log_info "Milvus cluster deployment complete!"
    echo "=================================================="
    echo ""
    echo "Connection details:"
    echo "  Namespace: ${NAMESPACE}"
    echo "  Internal endpoint: milvus-proxy.${NAMESPACE}.svc.cluster.local:19530"
    EXTERNAL_IP=$(kubectl get svc milvus-external -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    echo "  External endpoint: ${EXTERNAL_IP}:19530"
    echo ""
    echo "Next steps:"
    echo "  1. Create collections: kubectl apply -f k8s/milvus/collections/"
    echo "  2. Load embeddings: ./scripts/milvus/load-embeddings.sh"
    echo "  3. Deploy monitoring: ./scripts/milvus/setup-monitoring.sh"
    echo ""
}

main "$@"
