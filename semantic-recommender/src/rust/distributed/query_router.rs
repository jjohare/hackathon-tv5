use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;
use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use prometheus::{Counter, Histogram, IntGauge, Registry};

use super::shard_manager::{ShardManager, ShardInfo};
use super::gpu_node_service::{SearchRequest, SearchResponse, SearchResult};
use super::result_aggregator::ResultAggregator;

const DEFAULT_TIMEOUT_MS: u64 = 100;
const HEDGING_DELAY_MS: u64 = 50;
const MAX_CONCURRENT_SHARD_QUERIES: usize = 100;
const CIRCUIT_BREAKER_THRESHOLD: u32 = 5;
const CIRCUIT_BREAKER_RESET_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    Exhaustive,
    LshFiltered,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSearchRequest {
    pub embedding: Vec<f32>,
    pub k: usize,
    pub filters: HashMap<String, String>,
    pub request_id: String,
    pub strategy: SearchStrategy,
    pub enable_hedging: bool,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSearchResponse {
    pub results: Vec<SearchResult>,
    pub total_time_us: u64,
    pub shard_timings: Vec<ShardTiming>,
    pub shards_queried: u32,
    pub shards_succeeded: u32,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardTiming {
    pub shard_id: String,
    pub latency_us: u64,
    pub vectors_searched: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

pub struct QueryRouter {
    shard_manager: Arc<ShardManager>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    aggregator: Arc<ResultAggregator>,
    semaphore: Arc<Semaphore>,
    metrics: Arc<RouterMetrics>,
    cache: Arc<QueryCache>,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    failures: u32,
    last_failure: Instant,
    state: CircuitState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            failures: 0,
            last_failure: Instant::now(),
            state: CircuitState::Closed,
        }
    }

    fn record_success(&mut self) {
        self.failures = 0;
        self.state = CircuitState::Closed;
    }

    fn record_failure(&mut self) {
        self.failures += 1;
        self.last_failure = Instant::now();

        if self.failures >= CIRCUIT_BREAKER_THRESHOLD {
            self.state = CircuitState::Open;
        }
    }

    fn should_allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if self.last_failure.elapsed() >= CIRCUIT_BREAKER_RESET_INTERVAL {
                    self.state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
}

struct RouterMetrics {
    requests_total: Counter,
    requests_duration: Histogram,
    shard_queries_total: Counter,
    shard_errors_total: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
    active_requests: IntGauge,
}

impl RouterMetrics {
    fn new(registry: &Registry) -> Result<Self> {
        let requests_total = Counter::new(
            "router_requests_total",
            "Total distributed search requests",
        )?;
        registry.register(Box::new(requests_total.clone()))?;

        let requests_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "router_request_duration_us",
                "Request duration in microseconds",
            )
            .buckets(vec![
                1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 200000.0,
            ]),
        )?;
        registry.register(Box::new(requests_duration.clone()))?;

        let shard_queries_total = Counter::new(
            "router_shard_queries_total",
            "Total shard queries",
        )?;
        registry.register(Box::new(shard_queries_total.clone()))?;

        let shard_errors_total = Counter::new(
            "router_shard_errors_total",
            "Total shard query errors",
        )?;
        registry.register(Box::new(shard_errors_total.clone()))?;

        let cache_hits = Counter::new("router_cache_hits", "Cache hits")?;
        registry.register(Box::new(cache_hits.clone()))?;

        let cache_misses = Counter::new("router_cache_misses", "Cache misses")?;
        registry.register(Box::new(cache_misses.clone()))?;

        let active_requests = IntGauge::new(
            "router_active_requests",
            "Active requests",
        )?;
        registry.register(Box::new(active_requests.clone()))?;

        Ok(Self {
            requests_total,
            requests_duration,
            shard_queries_total,
            shard_errors_total,
            cache_hits,
            cache_misses,
            active_requests,
        })
    }
}

struct QueryCache {
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    ttl: Duration,
}

#[derive(Clone)]
struct CachedResult {
    results: Vec<SearchResult>,
    timestamp: Instant,
}

impl QueryCache {
    fn new(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl,
        }
    }

    async fn get(&self, key: &str) -> Option<Vec<SearchResult>> {
        let cache = self.cache.read().await;
        let cached = cache.get(key)?;

        if cached.timestamp.elapsed() < self.ttl {
            Some(cached.results.clone())
        } else {
            None
        }
    }

    async fn set(&self, key: String, results: Vec<SearchResult>) {
        let mut cache = self.cache.write().await;
        cache.insert(
            key,
            CachedResult {
                results,
                timestamp: Instant::now(),
            },
        );
    }

    fn generate_key(request: &DistributedSearchRequest) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        request.embedding.iter().for_each(|f| {
            f.to_bits().hash(&mut hasher);
        });
        request.k.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl QueryRouter {
    pub fn new(shard_manager: Arc<ShardManager>) -> Result<Self> {
        let registry = Registry::new();
        let metrics = Arc::new(RouterMetrics::new(&registry)?);

        Ok(Self {
            shard_manager,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            aggregator: Arc::new(ResultAggregator::new()),
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_SHARD_QUERIES)),
            metrics,
            cache: Arc::new(QueryCache::new(Duration::from_secs(60))),
        })
    }

    pub async fn distributed_search(
        &self,
        request: DistributedSearchRequest,
    ) -> Result<DistributedSearchResponse> {
        let start = Instant::now();
        self.metrics.requests_total.inc();
        self.metrics.active_requests.inc();

        let cache_key = QueryCache::generate_key(&request);
        if let Some(cached_results) = self.cache.get(&cache_key).await {
            self.metrics.cache_hits.inc();
            self.metrics.active_requests.dec();

            return Ok(DistributedSearchResponse {
                results: cached_results,
                total_time_us: start.elapsed().as_micros() as u64,
                shard_timings: Vec::new(),
                shards_queried: 0,
                shards_succeeded: 0,
                request_id: request.request_id,
            });
        }

        self.metrics.cache_misses.inc();

        let result = self.do_distributed_search(request, start).await;

        self.metrics.active_requests.dec();
        let duration = start.elapsed().as_micros() as u64;
        self.metrics.requests_duration.observe(duration as f64);

        result
    }

    async fn do_distributed_search(
        &self,
        request: DistributedSearchRequest,
        start: Instant,
    ) -> Result<DistributedSearchResponse> {
        let shards = match request.strategy {
            SearchStrategy::Exhaustive => self.shard_manager.get_healthy_shards().await,
            SearchStrategy::LshFiltered => {
                self.get_filtered_shards(&request.embedding).await?
            }
            SearchStrategy::Adaptive => self.get_adaptive_shards(&request).await?,
        };

        if shards.is_empty() {
            return Err(anyhow!("No healthy shards available"));
        }

        let timeout_duration = Duration::from_millis(request.timeout_ms);
        let shard_responses = self
            .query_shards_parallel(&shards, &request, timeout_duration)
            .await;

        let mut shard_timings = Vec::new();
        let mut all_results = Vec::new();
        let mut succeeded = 0u32;

        for (shard_id, result) in shard_responses {
            match result {
                Ok(response) => {
                    succeeded += 1;
                    all_results.extend(response.results);
                    shard_timings.push(ShardTiming {
                        shard_id,
                        latency_us: response.processing_time_us,
                        vectors_searched: response.vectors_searched,
                        success: true,
                        error_message: None,
                    });
                }
                Err(e) => {
                    self.metrics.shard_errors_total.inc();
                    shard_timings.push(ShardTiming {
                        shard_id,
                        latency_us: 0,
                        vectors_searched: 0,
                        success: false,
                        error_message: Some(e.to_string()),
                    });
                }
            }
        }

        if succeeded == 0 {
            return Err(anyhow!("All shard queries failed"));
        }

        let aggregated_results = self
            .aggregator
            .aggregate(all_results, request.k, &request.filters)
            .await?;

        let cache_key = QueryCache::generate_key(&request);
        self.cache.set(cache_key, aggregated_results.clone()).await;

        Ok(DistributedSearchResponse {
            results: aggregated_results,
            total_time_us: start.elapsed().as_micros() as u64,
            shard_timings,
            shards_queried: shards.len() as u32,
            shards_succeeded: succeeded,
            request_id: request.request_id,
        })
    }

    async fn query_shards_parallel(
        &self,
        shards: &[ShardInfo],
        request: &DistributedSearchRequest,
        timeout_duration: Duration,
    ) -> Vec<(String, Result<SearchResponse>)> {
        let mut tasks = Vec::new();

        for shard in shards {
            let shard_id = shard.id.clone();
            let shard_address = shard.address.clone();
            let request = request.clone();
            let semaphore = self.semaphore.clone();
            let circuit_breakers = self.circuit_breakers.clone();
            let metrics = self.metrics.clone();
            let enable_hedging = request.enable_hedging;

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.ok();

                let mut breakers = circuit_breakers.write().await;
                let breaker = breakers.entry(shard_id.clone()).or_insert_with(CircuitBreaker::new);

                if !breaker.should_allow_request() {
                    drop(breakers);
                    return (
                        shard_id,
                        Err(anyhow!("Circuit breaker open")),
                    );
                }
                drop(breakers);

                metrics.shard_queries_total.inc();

                let result = if enable_hedging {
                    Self::query_with_hedging(&shard_address, request, timeout_duration).await
                } else {
                    Self::query_shard(&shard_address, request, timeout_duration).await
                };

                let mut breakers = circuit_breakers.write().await;
                let breaker = breakers.get_mut(&shard_id).unwrap();

                match &result {
                    Ok(_) => breaker.record_success(),
                    Err(_) => breaker.record_failure(),
                }

                (shard_id, result)
            });

            tasks.push(task);
        }

        let mut results = Vec::new();
        for task in tasks {
            if let Ok(result) = task.await {
                results.push(result);
            }
        }

        results
    }

    async fn query_shard(
        address: &str,
        request: DistributedSearchRequest,
        timeout_duration: Duration,
    ) -> Result<SearchResponse> {
        let shard_request = SearchRequest {
            embedding: request.embedding,
            k: request.k,
            filters: request.filters,
            request_id: request.request_id,
            timeout_ms: timeout_duration.as_millis() as u64,
        };

        timeout(timeout_duration, async {
            Ok(SearchResponse {
                results: Vec::new(),
                processing_time_us: 0,
                vectors_searched: 0,
                request_id: shard_request.request_id,
                node_id: address.to_string(),
            })
        })
        .await
        .context("Shard query timeout")?
    }

    async fn query_with_hedging(
        address: &str,
        request: DistributedSearchRequest,
        timeout_duration: Duration,
    ) -> Result<SearchResponse> {
        let primary = Self::query_shard(address, request.clone(), timeout_duration);

        tokio::select! {
            result = primary => result,
            _ = tokio::time::sleep(Duration::from_millis(HEDGING_DELAY_MS)) => {
                let hedge = Self::query_shard(address, request, timeout_duration);

                tokio::select! {
                    result = hedge => result,
                }
            }
        }
    }

    async fn get_filtered_shards(&self, _embedding: &[f32]) -> Result<Vec<ShardInfo>> {
        Ok(self.shard_manager.get_healthy_shards().await)
    }

    async fn get_adaptive_shards(
        &self,
        _request: &DistributedSearchRequest,
    ) -> Result<Vec<ShardInfo>> {
        Ok(self.shard_manager.get_healthy_shards().await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new();

        assert!(breaker.should_allow_request());

        for _ in 0..CIRCUIT_BREAKER_THRESHOLD {
            breaker.record_failure();
        }

        assert!(!breaker.should_allow_request());

        breaker.record_success();
        assert!(breaker.should_allow_request());
    }

    #[tokio::test]
    async fn test_query_cache() {
        let cache = QueryCache::new(Duration::from_secs(1));

        let results = vec![SearchResult {
            id: "id1".to_string(),
            score: 0.9,
            metadata: HashMap::new(),
        }];

        cache.set("key1".to_string(), results.clone()).await;

        let cached = cache.get("key1").await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);

        tokio::time::sleep(Duration::from_secs(2)).await;

        let expired = cache.get("key1").await;
        assert!(expired.is_none());
    }
}
