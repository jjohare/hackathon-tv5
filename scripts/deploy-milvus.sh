#!/bin/bash
set -e

# Deploy Milvus cluster with Helm
# Requires: Kubernetes cluster with 3+ nodes, 50GB+ storage

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Deploying Milvus cluster...${NC}"

# Add Milvus Helm repo
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update

# Create namespace
kubectl create namespace milvus || true

# Deploy Milvus with custom values
cat > /tmp/milvus-values.yaml <<EOF
cluster:
  enabled: true

etcd:
  replicaCount: 3
  persistence:
    storageClass: "standard"
    size: 10Gi

minio:
  mode: distributed
  replicas: 4
  persistence:
    storageClass: "standard"
    size: 50Gi

pulsar:
  enabled: true
  replicaCount: 3

queryNode:
  replicas: 2
  resources:
    limits:
      cpu: "2"
      memory: 8Gi

indexNode:
  replicas: 1
  resources:
    limits:
      cpu: "2"
      memory: 8Gi

dataNode:
  replicas: 2

rootCoordinator:
  resources:
    limits:
      cpu: "1"
      memory: 2Gi

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
EOF

helm upgrade --install milvus milvus/milvus \
  --namespace milvus \
  --values /tmp/milvus-values.yaml \
  --wait \
  --timeout 10m

echo -e "${GREEN}✅ Milvus cluster deployed${NC}"

# Create collection
echo -e "${BLUE}Creating media_embeddings collection...${NC}"

# Port-forward to access Milvus
kubectl port-forward -n milvus svc/milvus 19530:19530 &
PF_PID=$!
sleep 5

# Create collection using Python script
python3 - <<'PYTHON'
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

connections.connect(host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]

schema = CollectionSchema(fields=fields, description="Media content embeddings")

# Create collection
if utility.has_collection("media_embeddings"):
    utility.drop_collection("media_embeddings")

collection = Collection(name="media_embeddings", schema=schema)

# Create HNSW index for fast similarity search
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256}
}

collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

print("Collection created and indexed")
PYTHON

# Cleanup port-forward
kill $PF_PID

echo -e "${GREEN}✅ Collection created with HNSW index${NC}"

# Create Kubernetes service for external access
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: milvus-external
  namespace: milvus
spec:
  type: LoadBalancer
  ports:
  - port: 19530
    targetPort: 19530
    protocol: TCP
    name: grpc
  selector:
    app.kubernetes.io/name: milvus
    component: standalone
EOF

echo -e "${GREEN}✅ Milvus deployment complete${NC}"
echo ""
echo "Milvus endpoints:"
echo "  • Internal: milvus.milvus.svc.cluster.local:19530"
echo "  • External: $(kubectl get svc -n milvus milvus-external -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):19530"
echo ""
