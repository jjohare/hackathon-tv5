# Deployment Guide

Production deployment guide for the GPU-accelerated Media Gateway system.

## Table of Contents

1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Kubernetes Deployment](#kubernetes-deployment)
3. [Scaling Strategies](#scaling-strategies)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Backup and Recovery](#backup-and-recovery)
6. [Security Best Practices](#security-best-practices)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

## Infrastructure Requirements

### Minimum Production Setup

**Compute Nodes**
- 3x GPU nodes (NVIDIA T4 or better)
- 16 cores CPU per node
- 64GB RAM per node
- 16GB VRAM per GPU
- 1TB NVMe SSD per node

**Storage**
- 5TB persistent volume for embeddings
- 1TB for knowledge graph (Neo4j)
- 500GB for metrics and logs
- S3-compatible object storage (optional)

**Network**
- 10 Gbps internal network
- Load balancer with SSL termination
- VPN or private network for management

### Recommended Production Setup

**Compute Nodes**
- 6x GPU nodes (NVIDIA A100 80GB)
- 32 cores CPU per node
- 256GB RAM per node
- 80GB VRAM per GPU
- 2TB NVMe SSD per node

**High Availability**
- Multi-region deployment
- Active-active configuration
- Automated failover
- Geographic load balancing

### Cloud Platform Costs

**AWS Pricing (us-east-1)**
```
p3.2xlarge (1x V100):   $3.06/hr  × 3 nodes  = $9.18/hr  ($220/day)
p4d.24xlarge (8x A100): $32.77/hr × 1 node   = $32.77/hr ($786/day)
g5.xlarge (1x A10G):    $1.01/hr  × 3 nodes  = $3.03/hr  ($73/day)

Storage (EBS gp3): $0.08/GB/month × 5TB = $400/month
Transfer out: $0.09/GB (first 10TB)
```

**Google Cloud Pricing (us-central1)**
```
a2-highgpu-1g (1x A100): $3.67/hr × 3 nodes = $11.01/hr ($264/day)
a2-ultragpu-8g (8x A100): $31.22/hr × 1 node = $31.22/hr ($749/day)
g2-standard-4 (1x L4): $0.87/hr × 3 nodes = $2.61/hr ($63/day)

Persistent Disk SSD: $0.17/GB/month × 5TB = $850/month
Network egress: $0.12/GB
```

**Cost Optimization Tips**
- Use spot/preemptible instances (60-90% savings)
- Reserved instances for baseline load (40-60% savings)
- Autoscaling for burst capacity
- S3/GCS lifecycle policies for cold data

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -o json | jq '.items[].status.capacity."nvidia.com/gpu"'
```

### Namespace and Configuration

```bash
# Create namespace
kubectl create namespace media-gateway

# Create secrets
kubectl create secret generic db-credentials \
  --from-literal=neo4j-password='YOUR_SECURE_PASSWORD' \
  --from-literal=redis-password='YOUR_SECURE_PASSWORD' \
  -n media-gateway

# Create ConfigMap
kubectl create configmap app-config \
  --from-file=config/production.yaml \
  -n media-gateway
```

### Deploy Core Services

**1. Storage Layer**

```bash
# Deploy Neo4j (Knowledge Graph)
kubectl apply -f k8s/milvus/namespace.yaml
kubectl apply -f k8s/milvus/etcd-statefulset.yaml
kubectl apply -f k8s/milvus/minio-statefulset.yaml
kubectl apply -f k8s/milvus/pulsar-statefulset.yaml

# Deploy Redis (Caching)
kubectl apply -f k8s/agentdb/redis-statefulset.yaml

# Deploy PostgreSQL with pgvector
kubectl apply -f k8s/agentdb/postgres-statefulset.yaml
kubectl apply -f k8s/agentdb/pgvector-init.yaml
```

**2. Vector Database (Milvus)**

```bash
# Deploy Milvus cluster components
kubectl apply -f k8s/milvus/configmap.yaml
kubectl apply -f k8s/milvus/services.yaml
kubectl apply -f k8s/milvus/datanode-deployment.yaml
kubectl apply -f k8s/milvus/querynode-daemonset.yaml
kubectl apply -f k8s/milvus/indexnode-deployment.yaml

# Apply GPU resource limits
kubectl apply -f k8s/milvus/gpu-resource-limits.yaml
```

**3. Application Services**

```yaml
# k8s/app/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: media-gateway-api
  namespace: media-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: media-gateway-api
  template:
    metadata:
      labels:
        app: media-gateway-api
    spec:
      containers:
      - name: api
        image: media-gateway:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: RUST_LOG
          value: "info"
        - name: NEO4J_URI
          value: "bolt://neo4j:7687"
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: MILVUS_HOST
          value: "milvus-service"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: media-gateway-api
  namespace: media-gateway
spec:
  selector:
    app: media-gateway-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

```bash
kubectl apply -f k8s/app/deployment.yaml
```

**4. Load Balancer and Ingress**

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: media-gateway-ingress
  namespace: media-gateway
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.media-gateway.example.com
    secretName: media-gateway-tls
  rules:
  - host: api.media-gateway.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: media-gateway-api
            port:
              number: 80
```

```bash
kubectl apply -f k8s/ingress.yaml
```

### Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n media-gateway

# Expected output:
# NAME                                 READY   STATUS    RESTARTS   AGE
# media-gateway-api-6c4d8f9b5-abc12    1/1     Running   0          5m
# media-gateway-api-6c4d8f9b5-def34    1/1     Running   0          5m
# media-gateway-api-6c4d8f9b5-ghi56    1/1     Running   0          5m
# milvus-querynode-0                   1/1     Running   0          10m
# milvus-datanode-0                    1/1     Running   0          10m
# redis-0                              1/1     Running   0          15m

# Check GPU allocation
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Test API endpoint
kubectl port-forward svc/media-gateway-api 8080:80 -n media-gateway
curl http://localhost:8080/health
```

## Scaling Strategies

### Horizontal Scaling

**Auto-scaling based on GPU utilization**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: media-gateway-hpa
  namespace: media-gateway
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: media-gateway-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

```bash
kubectl apply -f k8s/hpa.yaml
```

### Vertical Scaling

**Cluster Autoscaler**

```yaml
# k8s/cluster-autoscaler.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  min-nodes: "3"
  max-nodes: "20"
  scale-down-delay: "10m"
  scale-down-unneeded-time: "10m"
```

### Database Scaling

**Milvus Sharding**

```yaml
# k8s/milvus/sharding-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: milvus-config
  namespace: media-gateway
data:
  milvus.yaml: |
    queryNode:
      replicas: 6  # One per GPU node
    dataNode:
      replicas: 3
    indexNode:
      replicas: 3
    rootCoord:
      dmlChannelNum: 16  # Parallel write channels
    queryCoord:
      balanceIntervalSeconds: 60
```

**Neo4j Clustering (Causal Cluster)**

```bash
# Deploy 3-node Neo4j cluster
kubectl apply -f k8s/neo4j/statefulset-cluster.yaml

# Configure read replicas
kubectl scale statefulset neo4j --replicas=5 -n media-gateway
```

## Monitoring and Alerting

### Prometheus Setup

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Apply ServiceMonitor for custom metrics
kubectl apply -f k8s/monitoring.yaml
```

**Custom Metrics**

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: media-gateway-metrics
  namespace: media-gateway
spec:
  selector:
    matchLabels:
      app: media-gateway-api
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

### Grafana Dashboards

**Import Pre-built Dashboards**

```bash
# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring

# Import dashboards from grafana/
# - performance-dashboard.json (GPU metrics, latency, throughput)
# - migration-dashboard.json (data pipeline monitoring)
```

**Key Metrics to Monitor**

```yaml
# Application Metrics
- request_duration_seconds (histogram)
- request_count_total (counter)
- gpu_utilization_percent (gauge)
- gpu_memory_used_bytes (gauge)
- vector_search_duration_seconds (histogram)
- ontology_reasoning_duration_seconds (histogram)

# Infrastructure Metrics
- node_cpu_utilization
- node_memory_utilization
- pod_restarts_total
- nvidia_gpu_duty_cycle
- nvidia_gpu_memory_total_bytes
- nvidia_gpu_power_usage_watts

# Database Metrics
- milvus_search_latency_ms
- milvus_insert_rate
- neo4j_transaction_duration_ms
- redis_memory_used_bytes
- redis_ops_per_sec
```

### Alert Rules

```yaml
# k8s/prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: media-gateway-alerts
  namespace: media-gateway
spec:
  groups:
  - name: media-gateway
    interval: 30s
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.99, rate(request_duration_seconds_bucket[5m])) > 0.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High API latency detected"
        description: "P99 latency is {{ $value }}s (threshold: 0.5s)"

    - alert: GPUOutOfMemory
      expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "GPU out of memory"
        description: "GPU memory usage is {{ $value }}%"

    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Pod is crash looping"
        description: "Pod {{ $labels.pod }} is restarting frequently"

    - alert: MilvusDown
      expr: up{job="milvus"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Milvus is down"
        description: "Milvus instance {{ $labels.instance }} is unavailable"
```

```bash
kubectl apply -f k8s/prometheus-rules.yaml
```

### Log Aggregation

**EFK Stack (Elasticsearch, Fluentd, Kibana)**

```bash
# Install ECK (Elastic Cloud on Kubernetes)
kubectl create -f https://download.elastic.co/downloads/eck/2.10.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.10.0/operator.yaml

# Deploy Elasticsearch
kubectl apply -f k8s/logging/elasticsearch.yaml

# Deploy Kibana
kubectl apply -f k8s/logging/kibana.yaml

# Deploy Fluentd
kubectl apply -f k8s/logging/fluentd-daemonset.yaml
```

## Backup and Recovery

### Database Backups

**Neo4j Backup**

```bash
# Create backup CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: neo4j-backup
  namespace: media-gateway
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: neo4j:5.13
            command:
            - /bin/bash
            - -c
            - |
              neo4j-admin database dump \
                --to-path=/backups/neo4j-$(date +%Y%m%d).dump \
                neo4j

              # Upload to S3
              aws s3 cp /backups/neo4j-$(date +%Y%m%d).dump \
                s3://backups/neo4j/
          restartPolicy: OnFailure
```

**Milvus Backup**

```bash
# Milvus supports snapshots
kubectl exec -it milvus-standalone-0 -n media-gateway -- \
  milvusctl backup create --collection media_embeddings

# Export to object storage
kubectl exec -it milvus-standalone-0 -n media-gateway -- \
  milvusctl backup export --backup-id <backup-id> \
  --destination s3://backups/milvus/
```

**Redis Backup**

```bash
# RDB snapshots
kubectl exec -it redis-0 -n media-gateway -- redis-cli BGSAVE

# AOF for point-in-time recovery
kubectl exec -it redis-0 -n media-gateway -- redis-cli CONFIG SET appendonly yes
```

### Disaster Recovery

**Recovery Time Objective (RTO): < 1 hour**
**Recovery Point Objective (RPO): < 15 minutes**

**Recovery Procedure**

```bash
# 1. Restore from backup
kubectl create job restore-neo4j --from=cronjob/neo4j-backup

# 2. Verify data integrity
kubectl exec -it neo4j-0 -n media-gateway -- \
  cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN count(n);"

# 3. Restore application state
kubectl rollout restart deployment/media-gateway-api -n media-gateway

# 4. Run smoke tests
./scripts/smoke-test.sh

# 5. Monitor metrics
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
```

## Security Best Practices

### Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: media-gateway-netpol
  namespace: media-gateway
spec:
  podSelector:
    matchLabels:
      app: media-gateway-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
  - to:
    - podSelector:
        matchLabels:
          app: milvus
    ports:
    - protocol: TCP
      port: 19530
```

### RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind:ole
metadata:
  name: media-gateway-reader
  namespace: media-gateway
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: media-gateway-reader-binding
  namespace: media-gateway
subjects:
- kind: ServiceAccount
  name: media-gateway-sa
  namespace: media-gateway
roleRef:
  kind: Role
  name: media-gateway-reader
  apiGroup: rbac.authorization.k8s.io
```

### Secrets Management

```bash
# Use external secrets operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets-system \
  --create-namespace

# Integrate with AWS Secrets Manager, GCP Secret Manager, or Vault
```

## Performance Tuning

### GPU Optimization

```yaml
# Enable GPU time-slicing for cost savings
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-sharing-config
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        replicas: 4  # 4 pods per GPU
```

### Kernel Parameters

```yaml
# k8s/daemonset-sysctl.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: sysctl-tuning
spec:
  template:
    spec:
      hostPID: true
      hostNetwork: true
      initContainers:
      - name: sysctl
        image: busybox
        command:
        - sh
        - -c
        - |
          sysctl -w net.core.somaxconn=65535
          sysctl -w net.ipv4.tcp_max_syn_backlog=8192
          sysctl -w net.core.netdev_max_backlog=5000
        securityContext:
          privileged: true
```

## Troubleshooting

### Common Issues

**Issue: Pods stuck in Pending**
```bash
kubectl describe pod <pod-name> -n media-gateway
# Check: Insufficient GPU resources, PVC not bound, Node selector mismatch
```

**Issue: High latency**
```bash
# Check GPU utilization
kubectl exec -it <pod-name> -n media-gateway -- nvidia-smi

# Profile application
kubectl exec -it <pod-name> -n media-gateway -- \
  nsys profile -o profile ./app
```

**Issue: OOM errors**
```bash
# Increase memory limits
kubectl set resources deployment media-gateway-api \
  --limits=memory=64Gi -n media-gateway
```

Continue to [API_GUIDE.md](API_GUIDE.md) for API reference.
