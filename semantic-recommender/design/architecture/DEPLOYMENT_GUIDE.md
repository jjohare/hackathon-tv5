# GMC-O T4 Cluster Deployment Guide

## Prerequisites

- GCP Project with billing enabled
- `gcloud` CLI configured
- `kubectl` v1.28+
- Terraform v1.5+ (optional, for IaC)
- 100M embeddings exported from prototype

## Phase 1: Infrastructure Provisioning (Day 1)

### 1.1 Create GKE Cluster

```bash
# Set environment variables
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"
export CLUSTER_NAME="gmc-o-production"

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com

# Create GPU node pool cluster
gcloud container clusters create $CLUSTER_NAME \
  --region=$REGION \
  --num-nodes=0 \
  --machine-type=n1-standard-8 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=150 \
  --enable-stackdriver-kubernetes \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# Create GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=100 \
  --machine-type=n1-standard-8 \
  --disk-size=200 \
  --disk-type=pd-ssd \
  --enable-autoscaling \
  --min-nodes=100 \
  --max-nodes=200

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
```

### 1.2 Install GPU Drivers

```bash
# Install NVIDIA GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Verify GPU nodes
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-tesla-t4
```

## Phase 2: Deploy Core Services (Day 2)

### 2.1 Create Namespace and Apply Manifests

```bash
cd /home/devuser/workspace/hackathon-tv5/design/architecture/kubernetes

# Apply in order
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f services.yaml

# Wait for services to be ready
kubectl get svc -n gmc-o-prod
```

## Phase 3: Data Migration (Day 3-5)

See migration scripts in the main architecture document.

## Phase 4: Load Testing (Day 6-7)

Use Locust or K6 as documented in the main architecture specification.

## Phase 5: Production Traffic Migration (Day 8-10)

Gradual rollout from 1% to 100% over 3 days with monitoring.

## Rollback Plan

Emergency rollback procedures are documented in the main architecture specification.

---

**Last Updated**: 2025-12-04
