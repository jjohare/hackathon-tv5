# Production Deployment Guide

**Target Audience**: DevOps Engineers, SREs
**Prerequisites**: Kubernetes knowledge, cloud platform access
**Estimated Deployment Time**: 1-2 weeks

---

## Table of Contents

1. [Multi-Region Architecture](#1-multi-region-architecture)
2. [Kubernetes Deployment](#2-kubernetes-deployment)
3. [Monitoring & Alerting](#3-monitoring--alerting)
4. [Scaling Strategies](#4-scaling-strategies)
5. [Disaster Recovery](#5-disaster-recovery)
6. [Cost Optimization](#6-cost-optimization)

---

## 1. Multi-Region Architecture

### 1.1 Geographic Distribution

```yaml
# Global architecture overview
regions:
  primary:
    name: us-east-1
    provider: AWS
    services:
      - hot_path_api
      - vector_search
      - knowledge_graph
      - gpu_cluster
    capacity:
      api_servers: 80
      gpu_nodes: 20

  secondary:
    name: eu-west-1
    provider: GCP
    services:
      - hot_path_api
      - vector_search (read replica)
      - knowledge_graph (read replica)
    capacity:
      api_servers: 40

  tertiary:
    name: ap-southeast-1
    provider: AWS
    services:
      - hot_path_api
      - vector_search (read replica)
    capacity:
      api_servers: 20

data_replication:
  vector_db:
    strategy: async_replication
    lag_target: 5_seconds

  knowledge_graph:
    strategy: async_replication
    lag_target: 10_seconds

  user_profiles:
    strategy: multi_master
    consistency: eventual
```

### 1.2 DNS-Based Routing

```yaml
# Route53 configuration (AWS)
hosted_zone:
  name: api.tv5monde.com
  type: public

routing_policies:
  - type: geolocation
    records:
      - geo: NA
        value: us-east-1-lb.tv5monde.com
        weight: 100

      - geo: EU
        value: eu-west-1-lb.tv5monde.com
        weight: 100

      - geo: AP
        value: ap-southeast-1-lb.tv5monde.com
        weight: 100

      - geo: default
        value: us-east-1-lb.tv5monde.com
        weight: 100

health_checks:
  - endpoint: /health
    interval: 30s
    timeout: 10s
    failure_threshold: 3
```

### 1.3 CDN Integration

```yaml
# Cloudflare configuration
cloudflare:
  zone: tv5monde.com

  caching:
    popular_queries:
      ttl: 300  # 5 minutes
      key_format: "query:{query_hash}"

    user_sessions:
      ttl: 3600  # 1 hour
      key_format: "session:{user_id}"

  edge_rules:
    - name: cache_popular_recommendations
      match: "path('/api/recommendations') AND header('Cache-Control', 'public')"
      action:
        cache_level: standard
        cache_ttl: 300

    - name: bypass_cache_personalized
      match: "path('/api/recommendations') AND header('X-Personalized', 'true')"
      action:
        cache_level: bypass
```

---

## 2. Kubernetes Deployment

### 2.1 Hot Path API Deployment

```yaml
# k8s/hot-path-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hot-path-api
  namespace: recommendation
spec:
  replicas: 20
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 5
      maxUnavailable: 2

  selector:
    matchLabels:
      app: hot-path-api

  template:
    metadata:
      labels:
        app: hot-path-api
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: tv5monde/recommendation-api:1.0.0
        ports:
        - containerPort: 8080
          name: http

        resources:
          requests:
            cpu: "2"
            memory: 4Gi
          limits:
            cpu: "4"
            memory: 8Gi

        env:
        - name: VECTOR_DB_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: service-endpoints
              key: vector_db_url

        - name: GRAPH_DB_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: service-endpoints
              key: neo4j_url

        - name: AGENTDB_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: service-endpoints
              key: agentdb_url

        - name: LOG_LEVEL
          value: "info"

        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - hot-path-api
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: hot-path-api
  namespace: recommendation
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  selector:
    app: hot-path-api

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hot-path-api-hpa
  namespace: recommendation
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hot-path-api
  minReplicas: 20
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### 2.2 GPU Cluster Deployment

```yaml
# k8s/gpu-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: gpu-cluster
  namespace: recommendation
spec:
  serviceName: gpu-cluster
  replicas: 6

  selector:
    matchLabels:
      app: gpu-cluster

  template:
    metadata:
      labels:
        app: gpu-cluster
    spec:
      nodeSelector:
        gpu: "nvidia-a100"

      containers:
      - name: gpu-worker
        image: tv5monde/gpu-processor:1.0.0

        resources:
          requests:
            nvidia.com/gpu: 4  # 4x A100 per node
            cpu: "16"
            memory: 64Gi
          limits:
            nvidia.com/gpu: 4
            cpu: "32"
            memory: 128Gi

        volumeMounts:
        - name: models
          mountPath: /models
        - name: processing-queue
          mountPath: /queue

        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"

  volumeClaimTemplates:
  - metadata:
      name: processing-queue
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### 2.3 Vector Database Cluster

```yaml
# k8s/qdrant-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: recommendation
spec:
  serviceName: qdrant
  replicas: 10

  selector:
    matchLabels:
      app: qdrant

  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.7.0

        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc

        resources:
          requests:
            cpu: "4"
            memory: 16Gi
          limits:
            cpu: "8"
            memory: 32Gi

        env:
        - name: QDRANT__CLUSTER__ENABLED
          value: "true"
        - name: QDRANT__CLUSTER__NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

        volumeMounts:
        - name: data
          mountPath: /qdrant/storage

  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### 2.4 ConfigMaps and Secrets

```yaml
# k8s/config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: service-endpoints
  namespace: recommendation
data:
  vector_db_url: "http://qdrant-service:6333"
  neo4j_url: "bolt://neo4j-service:7687"
  agentdb_url: "http://agentdb-service:8888"
  kafka_brokers: "kafka-broker-1:9092,kafka-broker-2:9092,kafka-broker-3:9092"

---
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
  namespace: recommendation
type: Opaque
stringData:
  neo4j_username: "neo4j"
  neo4j_password: "<REPLACE_WITH_SECRET>"
  scylladb_username: "admin"
  scylladb_password: "<REPLACE_WITH_SECRET>"
```

---

## 3. Monitoring & Alerting

### 3.1 Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: tv5monde-prod
    region: us-east-1

scrape_configs:
  - job_name: 'hot-path-api'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - recommendation
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: hot-path-api
      action: keep
    - source_labels: [__meta_kubernetes_pod_name]
      target_label: pod

  - job_name: 'qdrant'
    static_configs:
    - targets:
      - qdrant-0:6333
      - qdrant-1:6333
      - qdrant-2:6333

  - job_name: 'gpu-cluster'
    static_configs:
    - targets:
      - gpu-cluster-0:9090
      - gpu-cluster-1:9090

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093
```

### 3.2 Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Recommendation System Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(recommendations_total[5m])",
            "legendFormat": "{{variant}}"
          }
        ]
      },
      {
        "title": "P99 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(recommendation_latency_seconds_bucket[5m]))",
            "legendFormat": "{{variant}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "avg(gpu_utilization_percent) by (gpu_id)",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      },
      {
        "title": "Vector Search Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(vector_search_latency_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      }
    ]
  }
}
```

### 3.3 Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: recommendation_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(recommendation_latency_seconds_bucket[5m])) > 0.15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High recommendation latency (p99 > 150ms)"
          description: "p99 latency is {{ $value | humanizeDuration }}"

      - alert: LowCacheHitRate
        expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.4
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate (< 40%)"

      - alert: GPUUtilizationLow
        expr: avg(gpu_utilization_percent) < 50
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "GPU utilization below 50%"

      - alert: VectorDBDown
        expr: up{job="qdrant"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Vector database instance down"
```

---

## 4. Scaling Strategies

### 4.1 Horizontal Pod Autoscaling

```yaml
# k8s/hpa-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hot-path-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hot-path-api

  minReplicas: 20
  maxReplicas: 200

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 10
        periodSeconds: 60
      selectPolicy: Max

    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      selectPolicy: Min

  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"

  - type: External
    external:
      metric:
        name: cloudwatch_alb_request_count
      target:
        type: AverageValue
        averageValue: "10000"
```

### 4.2 Cluster Autoscaling

```yaml
# k8s/cluster-autoscaler.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  cluster-autoscaler.yaml: |
    apiVersion: v1
    kind: Pod
    spec:
      containers:
      - name: cluster-autoscaler
        image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        command:
        - ./cluster-autoscaler
        - --cloud-provider=aws
        - --namespace=kube-system
        - --nodes=20:100:recommendation-node-group
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
```

### 4.3 Database Scaling

```yaml
# Neo4j scaling strategy
neo4j:
  core_servers: 3
  read_replicas: 5

  scaling_policy:
    metric: query_latency_p99
    target: 5ms
    scale_up_threshold: 10ms
    scale_down_threshold: 2ms
    cooldown: 300s

# Qdrant sharding strategy
qdrant:
  shards: 20
  replicas_per_shard: 2

  rebalancing:
    enabled: true
    trigger: shard_size_imbalance
    threshold: 20%  # Rebalance if shard differs by >20%
```

---

## 5. Disaster Recovery

### 5.1 Backup Strategy

```yaml
# backup/backup-config.yaml
backups:
  vector_db:
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: 30d
    storage: s3://tv5monde-backups/qdrant/
    compression: gzip

  knowledge_graph:
    schedule: "0 3 * * *"
    retention: 30d
    storage: s3://tv5monde-backups/neo4j/
    type: full

  user_profiles:
    schedule: "0 */6 * * *"  # Every 6 hours
    retention: 7d
    storage: s3://tv5monde-backups/scylladb/
    type: incremental
```

### 5.2 Disaster Recovery Procedures

```bash
#!/bin/bash
# scripts/disaster-recovery.sh

# Restore from backup
restore_vector_db() {
    BACKUP_DATE=$1
    aws s3 cp s3://tv5monde-backups/qdrant/snapshot-$BACKUP_DATE.tar.gz /tmp/

    kubectl exec -n recommendation qdrant-0 -- tar -xzf /tmp/snapshot-$BACKUP_DATE.tar.gz -C /qdrant/storage

    kubectl rollout restart statefulset/qdrant -n recommendation
}

# Failover to secondary region
failover_to_secondary() {
    # Update Route53 to point to EU region
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z1234567890ABC \
        --change-batch file://failover-eu.json

    # Scale up EU cluster
    kubectl scale deployment/hot-path-api --replicas=80 -n recommendation --context=gcp-eu-west-1

    echo "Failover complete. Monitor metrics at https://grafana.tv5monde.com"
}

# Health check
check_system_health() {
    # Check API health
    curl -f http://api.tv5monde.com/health || echo "API DOWN"

    # Check Vector DB
    curl -f http://qdrant-service:6333/collections || echo "Vector DB DOWN"

    # Check Neo4j
    cypher-shell -a bolt://neo4j-service:7687 -u neo4j -p $NEO4J_PASSWORD "RETURN 1" || echo "Neo4j DOWN"
}
```

### 5.3 RPO/RTO Targets

```yaml
recovery_objectives:
  hot_path_api:
    rpo: 0  # Stateless, no data loss
    rto: 5m  # Restore service in 5 minutes

  vector_db:
    rpo: 1h  # Last backup + replication lag
    rto: 30m  # Restore from backup

  knowledge_graph:
    rpo: 24h  # Daily backups
    rto: 2h  # Full restore from backup

  user_profiles:
    rpo: 6h  # Incremental backups every 6 hours
    rto: 1h
```

---

## 6. Cost Optimization

### 6.1 Resource Optimization

```yaml
# cost_optimization.yml
strategies:
  compute:
    - use_spot_instances:
        workloads: [gpu_cluster, batch_processing]
        savings: 70%

    - rightsize_instances:
        target_cpu_utilization: 70%
        target_memory_utilization: 80%
        savings: 30%

  storage:
    - use_lifecycle_policies:
        move_to_glacier_after: 90d
        delete_after: 365d
        savings: 80%

    - compress_vectors:
        quantization: int8
        compression_ratio: 4x
        savings: 75%

  network:
    - use_cdn_caching:
        cache_hit_rate_target: 60%
        bandwidth_savings: 50%

    - regional_data_egress:
        replicate_to_edge: true
        cross_region_traffic_reduction: 80%
```

### 6.2 Cost Monitoring

```python
# monitoring/cost_tracking.py
from prometheus_client import Gauge

class CostMetrics:
    def __init__(self):
        self.compute_cost = Gauge(
            'compute_cost_usd_per_hour',
            'Compute cost',
            ['region', 'service']
        )

        self.storage_cost = Gauge(
            'storage_cost_usd_per_month',
            'Storage cost',
            ['region', 'storage_type']
        )

        self.total_monthly_cost = Gauge(
            'total_monthly_cost_usd',
            'Projected monthly cost'
        )

    def update_costs(self):
        # Compute costs
        self.compute_cost.labels(region='us-east-1', service='api').set(800)
        self.compute_cost.labels(region='us-east-1', service='gpu').set(500)

        # Storage costs
        self.storage_cost.labels(region='us-east-1', storage_type='vector_db').set(5000)

        # Total projection
        self.total_monthly_cost.set(85000)
```

### 6.3 Cost Breakdown (Projected)

```yaml
monthly_costs:
  compute:
    api_servers: $30,000
    gpu_cluster: $50,000
    database_instances: $20,000
    total: $100,000

  storage:
    vector_db: $5,000
    knowledge_graph: $3,000
    user_profiles: $2,000
    backups: $1,000
    total: $11,000

  network:
    data_transfer: $8,000
    cdn: $5,000
    load_balancers: $2,000
    total: $15,000

  total_monthly: $126,000
  total_annual: $1,512,000

cost_optimization_opportunities:
  - use_reserved_instances: save $20,000/month
  - use_spot_for_gpu: save $35,000/month
  - optimize_storage: save $3,000/month
  - total_potential_savings: $58,000/month (46%)
```

---

## Deployment Checklist

**Pre-Deployment:**
- [ ] Kubernetes cluster provisioned
- [ ] GPU nodes configured
- [ ] Storage classes created
- [ ] Secrets and ConfigMaps deployed
- [ ] DNS records configured
- [ ] CDN setup complete

**Deployment:**
- [ ] Vector DB cluster deployed
- [ ] Knowledge Graph cluster deployed
- [ ] GPU cluster deployed
- [ ] Hot Path API deployed
- [ ] Autoscaling configured

**Post-Deployment:**
- [ ] Monitoring dashboards live
- [ ] Alerting rules active
- [ ] Backup jobs scheduled
- [ ] Disaster recovery tested
- [ ] Cost tracking enabled
- [ ] Load testing passed

---

## Performance Targets

**Production SLOs:**
- **Availability**: 99.9% (43 minutes downtime/month)
- **Latency (p99)**: <100ms
- **Throughput**: >166K req/sec
- **Error Rate**: <0.1%
- **Data Loss**: RPO <1 hour

---

**Next Steps:**
1. Run load tests
2. Chaos engineering tests
3. Security audit
4. Cost optimization review

**Related Guides:**
- [GPU Setup Guide](gpu-setup-guide.md)
- [Vector Search Implementation](vector-search-implementation.md)
- [Learning Pipeline Guide](learning-pipeline-guide.md)
