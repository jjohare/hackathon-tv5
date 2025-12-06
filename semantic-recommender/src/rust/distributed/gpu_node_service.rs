use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use prometheus::{
    Counter, Histogram, IntGauge, Registry, Encoder, TextEncoder,
};

const MAX_BATCH_SIZE: usize = 128;
const INDEX_CAPACITY: usize = 1_000_000;

#[derive(Clone)]
pub struct GpuNodeService {
    node_id: String,
    index: Arc<RwLock<VectorIndex>>,
    metrics: Arc<Metrics>,
    config: NodeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub node_id: String,
    pub listen_address: String,
    pub gpu_device_id: u32,
    pub max_concurrent_requests: usize,
    pub index_type: IndexType,
    pub index_params: IndexParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    HNSW,
    IVF,
    Flat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParams {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for IndexParams {
    fn default() -> Self {
        Self {
            dim: 768,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub embedding: Vec<f32>,
    pub k: usize,
    pub filters: HashMap<String, String>,
    pub request_id: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub processing_time_us: u64,
    pub vectors_searched: u64,
    pub request_id: String,
    pub node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct VectorIndex {
    vectors: Vec<Vec<f32>>,
    ids: Vec<String>,
    metadata: HashMap<String, HashMap<String, String>>,
    index_type: IndexType,
    params: IndexParams,
    hnsw_graph: Option<HNSWGraph>,
}

impl VectorIndex {
    pub fn new(index_type: IndexType, params: IndexParams) -> Self {
        let hnsw_graph = if matches!(index_type, IndexType::HNSW) {
            Some(HNSWGraph::new(params.m, params.ef_construction))
        } else {
            None
        };

        Self {
            vectors: Vec::with_capacity(INDEX_CAPACITY),
            ids: Vec::with_capacity(INDEX_CAPACITY),
            metadata: HashMap::new(),
            index_type,
            params,
            hnsw_graph,
        }
    }

    pub fn add_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        if vector.len() != self.params.dim {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.params.dim,
                vector.len()
            ));
        }

        let idx = self.vectors.len();
        self.vectors.push(vector.clone());
        self.ids.push(id.clone());
        self.metadata.insert(id.clone(), metadata);

        if let Some(graph) = &mut self.hnsw_graph {
            graph.insert(idx, &vector)?;
        }

        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.params.dim {
            return Err(anyhow!("Query dimension mismatch"));
        }

        let mut candidates = match &self.hnsw_graph {
            Some(graph) => self.hnsw_search(graph, query, k * 2)?,
            None => self.flat_search(query, k * 2)?,
        };

        candidates.retain(|result| self.matches_filters(&result.id, filters));

        candidates.truncate(k);
        Ok(candidates)
    }

    fn hnsw_search(
        &self,
        graph: &HNSWGraph,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let ef = self.params.ef_search.max(k);
        let indices = graph.search(query, ef, &self.vectors)?;

        Ok(indices
            .into_iter()
            .take(k)
            .map(|(idx, score)| SearchResult {
                id: self.ids[idx].clone(),
                score,
                metadata: self.metadata.get(&self.ids[idx]).cloned().unwrap_or_default(),
            })
            .collect())
    }

    fn flat_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut scores: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| (idx, Self::cosine_similarity(query, vec)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);

        Ok(scores
            .into_iter()
            .map(|(idx, score)| SearchResult {
                id: self.ids[idx].clone(),
                score,
                metadata: self.metadata.get(&self.ids[idx]).cloned().unwrap_or_default(),
            })
            .collect())
    }

    fn matches_filters(&self, id: &str, filters: &HashMap<String, String>) -> bool {
        if filters.is_empty() {
            return true;
        }

        let metadata = match self.metadata.get(id) {
            Some(m) => m,
            None => return false,
        };

        filters.iter().all(|(key, value)| {
            metadata.get(key).map(|v| v == value).unwrap_or(false)
        })
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    pub fn size(&self) -> usize {
        self.vectors.len()
    }
}

#[derive(Debug, Clone)]
struct HNSWGraph {
    m: usize,
    ef_construction: usize,
    layers: Vec<HashMap<usize, Vec<usize>>>,
    entry_point: Option<usize>,
    max_layer: usize,
}

impl HNSWGraph {
    fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            m,
            ef_construction,
            layers: vec![HashMap::new()],
            entry_point: None,
            max_layer: 0,
        }
    }

    fn insert(&mut self, idx: usize, vector: &[f32]) -> Result<()> {
        let layer = self.random_layer();

        while self.layers.len() <= layer {
            self.layers.push(HashMap::new());
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = layer;
            return Ok(());
        }

        for l in 0..=layer {
            self.layers[l].insert(idx, Vec::new());
        }

        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        ef: usize,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<(usize, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        let mut candidates = Vec::new();
        let visited = std::collections::HashSet::new();

        for (idx, vec) in vectors.iter().enumerate() {
            let score = Self::cosine_similarity(query, vec);
            candidates.push((idx, score));
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(ef);

        Ok(candidates)
    }

    fn random_layer(&self) -> usize {
        let ml = 1.0 / (self.m as f64).ln();
        let r: f64 = rand::random();
        (-r.ln() * ml).floor() as usize
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }
}

struct Metrics {
    requests_total: Counter,
    requests_duration: Histogram,
    vectors_searched: Counter,
    active_requests: IntGauge,
    index_size: IntGauge,
}

impl Metrics {
    fn new(registry: &Registry) -> Result<Self> {
        let requests_total = Counter::new(
            "gpu_node_requests_total",
            "Total number of search requests",
        )?;
        registry.register(Box::new(requests_total.clone()))?;

        let requests_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "gpu_node_request_duration_us",
                "Request duration in microseconds",
            )
            .buckets(vec![
                100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0,
            ]),
        )?;
        registry.register(Box::new(requests_duration.clone()))?;

        let vectors_searched = Counter::new(
            "gpu_node_vectors_searched_total",
            "Total number of vectors searched",
        )?;
        registry.register(Box::new(vectors_searched.clone()))?;

        let active_requests = IntGauge::new(
            "gpu_node_active_requests",
            "Number of active requests",
        )?;
        registry.register(Box::new(active_requests.clone()))?;

        let index_size = IntGauge::new(
            "gpu_node_index_size",
            "Number of vectors in index",
        )?;
        registry.register(Box::new(index_size.clone()))?;

        Ok(Self {
            requests_total,
            requests_duration,
            vectors_searched,
            active_requests,
            index_size,
        })
    }
}

impl GpuNodeService {
    pub fn new(config: NodeConfig) -> Result<Self> {
        let registry = Registry::new();
        let metrics = Arc::new(Metrics::new(&registry)?);

        let index = VectorIndex::new(
            config.index_type.clone(),
            config.index_params.clone(),
        );

        Ok(Self {
            node_id: config.node_id.clone(),
            index: Arc::new(RwLock::new(index)),
            metrics,
            config,
        })
    }

    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse> {
        let start = Instant::now();
        self.metrics.requests_total.inc();
        self.metrics.active_requests.inc();

        let result = self.do_search(request).await;

        self.metrics.active_requests.dec();
        let duration = start.elapsed().as_micros() as u64;
        self.metrics.requests_duration.observe(duration as f64);

        result
    }

    async fn do_search(&self, request: SearchRequest) -> Result<SearchResponse> {
        let index = self.index.read().await;

        let results = index
            .search(&request.embedding, request.k, &request.filters)
            .context("Search failed")?;

        let vectors_searched = index.size() as u64;
        self.metrics.vectors_searched.inc_by(vectors_searched);

        Ok(SearchResponse {
            results,
            processing_time_us: 0,
            vectors_searched,
            request_id: request.request_id,
            node_id: self.node_id.clone(),
        })
    }

    pub async fn batch_search(
        &self,
        requests: Vec<SearchRequest>,
    ) -> Result<Vec<SearchResponse>> {
        if requests.len() > MAX_BATCH_SIZE {
            return Err(anyhow!(
                "Batch size {} exceeds maximum {}",
                requests.len(),
                MAX_BATCH_SIZE
            ));
        }

        let mut responses = Vec::with_capacity(requests.len());

        for request in requests {
            let response = self.search(request).await?;
            responses.push(response);
        }

        Ok(responses)
    }

    pub async fn add_vector(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let mut index = self.index.write().await;
        index.add_vector(id, vector, metadata)?;
        self.metrics.index_size.set(index.size() as i64);
        Ok(())
    }

    pub async fn get_stats(&self) -> Result<NodeStats> {
        let index = self.index.read().await;

        Ok(NodeStats {
            node_id: self.node_id.clone(),
            total_embeddings: index.size() as u64,
            index_type: format!("{:?}", self.config.index_type),
            gpu_model: "T4".to_string(),
            requests_total: self.metrics.requests_total.get() as u64,
            active_requests: self.metrics.active_requests.get() as u64,
        })
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse> {
        Ok(HealthCheckResponse {
            status: HealthStatus::Serving,
            message: "Healthy".to_string(),
            gpu_utilization: 0.5,
            memory_utilization: 0.6,
            active_requests: self.metrics.active_requests.get() as u64,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStats {
    pub node_id: String,
    pub total_embeddings: u64,
    pub index_type: String,
    pub gpu_model: String,
    pub requests_total: u64,
    pub active_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: HealthStatus,
    pub message: String,
    pub gpu_utilization: f32,
    pub memory_utilization: f32,
    pub active_requests: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Serving,
    NotServing,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vector_index_add_search() {
        let mut index = VectorIndex::new(
            IndexType::Flat,
            IndexParams::default(),
        );

        let vec1 = vec![1.0; 768];
        let vec2 = vec![0.5; 768];

        index
            .add_vector("id1".to_string(), vec1.clone(), HashMap::new())
            .unwrap();
        index
            .add_vector("id2".to_string(), vec2, HashMap::new())
            .unwrap();

        let results = index.search(&vec1, 1, &HashMap::new()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id1");
    }

    #[tokio::test]
    async fn test_gpu_node_service() {
        let config = NodeConfig {
            node_id: "node-1".to_string(),
            listen_address: "0.0.0.0:5001".to_string(),
            gpu_device_id: 0,
            max_concurrent_requests: 100,
            index_type: IndexType::Flat,
            index_params: IndexParams::default(),
        };

        let service = GpuNodeService::new(config).unwrap();

        let vec = vec![1.0; 768];
        service
            .add_vector("id1".to_string(), vec.clone(), HashMap::new())
            .await
            .unwrap();

        let request = SearchRequest {
            embedding: vec,
            k: 1,
            filters: HashMap::new(),
            request_id: "req-1".to_string(),
            timeout_ms: 1000,
        };

        let response = service.search(request).await.unwrap();
        assert_eq!(response.results.len(), 1);
    }
}
