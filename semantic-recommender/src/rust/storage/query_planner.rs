/// Query Planner
///
/// Determines optimal query execution strategy based on query characteristics.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryStrategy {
    /// Only use Milvus for vector search (fastest, 8.7ms)
    VectorOnly,

    /// Only use Neo4j for graph traversal (rare)
    GraphOnly,

    /// Execute Milvus and Neo4j in parallel (most common)
    HybridParallel,

    /// Execute Milvus first, then Neo4j with filtered results
    HybridSequential,

    /// Use cached results only
    CachedOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    pub strategy: QueryStrategy,
    pub estimated_latency_ms: f32,
    pub requires_milvus: bool,
    pub requires_neo4j: bool,
    pub requires_agentdb: bool,
    pub cache_eligible: bool,
    pub reason: String,
}

/// Query planner with optimization heuristics
pub struct QueryPlanner {
    config: PlannerConfig,
}

#[derive(Debug, Clone)]
pub struct PlannerConfig {
    pub vector_only_threshold: f32,
    pub graph_only_threshold: f32,
    pub cache_ttl_secs: u64,
    pub enable_adaptive: bool,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            vector_only_threshold: 0.8,
            graph_only_threshold: 0.2,
            cache_ttl_secs: 300,
            enable_adaptive: true,
        }
    }
}

impl QueryPlanner {
    pub fn new(config: PlannerConfig) -> Self {
        Self { config }
    }

    /// Plan query execution strategy
    pub fn plan(&self, query: &super::hybrid_coordinator::SearchQuery) -> QueryAnalysis {
        // Check if query can be served from cache
        if self.is_cache_eligible(query) {
            return QueryAnalysis {
                strategy: QueryStrategy::CachedOnly,
                estimated_latency_ms: 1.0,
                requires_milvus: false,
                requires_neo4j: false,
                requires_agentdb: false,
                cache_eligible: true,
                reason: "Query matches cache criteria".to_string(),
            };
        }

        // Analyze query characteristics
        let needs_graph = self.requires_graph_traversal(query);
        let needs_vector = self.requires_vector_search(query);
        let needs_policy = !query.user_id.is_empty();

        let strategy = if !needs_graph && needs_vector {
            QueryStrategy::VectorOnly
        } else if needs_graph && !needs_vector {
            QueryStrategy::GraphOnly
        } else if needs_graph && needs_vector {
            if query.k <= 20 {
                // Small result sets benefit from sequential
                QueryStrategy::HybridSequential
            } else {
                // Large result sets benefit from parallel
                QueryStrategy::HybridParallel
            }
        } else {
            QueryStrategy::VectorOnly
        };

        let estimated_latency_ms = self.estimate_latency(&strategy, query);

        QueryAnalysis {
            strategy,
            estimated_latency_ms,
            requires_milvus: needs_vector,
            requires_neo4j: needs_graph,
            requires_agentdb: needs_policy,
            cache_eligible: self.is_cache_eligible(query),
            reason: self.explain_strategy(&strategy, query),
        }
    }

    fn requires_graph_traversal(&self, query: &super::hybrid_coordinator::SearchQuery) -> bool {
        // Heuristics for graph traversal requirement
        query.include_relationships
            || query.require_genre_filter
            || query.require_cultural_context
            || !query.graph_filters.is_empty()
    }

    fn requires_vector_search(&self, query: &super::hybrid_coordinator::SearchQuery) -> bool {
        // Always need vector search unless explicitly graph-only
        !query.embedding.is_empty()
    }

    fn is_cache_eligible(&self, query: &super::hybrid_coordinator::SearchQuery) -> bool {
        // Cache eligibility criteria
        query.user_id.is_empty() // No personalization
            && query.metadata_filters.is_empty() // No complex filters
            && query.k <= 50 // Reasonable result size
            && !query.include_relationships // No graph traversal
    }

    fn estimate_latency(&self, strategy: &QueryStrategy, query: &super::hybrid_coordinator::SearchQuery) -> f32 {
        match strategy {
            QueryStrategy::VectorOnly => {
                // Milvus P99: 8.7ms
                8.7 + (query.k as f32 * 0.01)
            }
            QueryStrategy::GraphOnly => {
                // Neo4j graph traversal: 15-30ms
                20.0 + (query.k as f32 * 0.05)
            }
            QueryStrategy::HybridParallel => {
                // Max of parallel operations + aggregation
                f32::max(8.7, 20.0) + 2.0
            }
            QueryStrategy::HybridSequential => {
                // Sum of sequential operations
                8.7 + 15.0 + 2.0
            }
            QueryStrategy::CachedOnly => 1.0,
        }
    }

    fn explain_strategy(&self, strategy: &QueryStrategy, query: &super::hybrid_coordinator::SearchQuery) -> String {
        match strategy {
            QueryStrategy::VectorOnly => format!(
                "Vector-only search for {} results with no graph requirements",
                query.k
            ),
            QueryStrategy::GraphOnly => "Graph-only traversal for relationship-focused query".to_string(),
            QueryStrategy::HybridParallel => format!(
                "Parallel hybrid search for {} results with graph enrichment",
                query.k
            ),
            QueryStrategy::HybridSequential => format!(
                "Sequential hybrid search: vector filtering then graph enrichment for {} results",
                query.k
            ),
            QueryStrategy::CachedOnly => "Serving from cache".to_string(),
        }
    }

    /// Adaptive planning based on historical performance
    pub fn plan_adaptive(
        &self,
        query: &super::hybrid_coordinator::SearchQuery,
        historical_metrics: &HashMap<QueryStrategy, f32>,
    ) -> QueryAnalysis {
        if !self.config.enable_adaptive {
            return self.plan(query);
        }

        // Get base plan
        let mut analysis = self.plan(query);

        // Adjust based on historical performance
        if let Some(&actual_latency) = historical_metrics.get(&analysis.strategy) {
            if actual_latency > analysis.estimated_latency_ms * 1.5 {
                // Performance degraded, try alternative strategy
                analysis = self.try_alternative_strategy(query, &analysis);
            }
        }

        analysis
    }

    fn try_alternative_strategy(
        &self,
        query: &super::hybrid_coordinator::SearchQuery,
        current: &QueryAnalysis,
    ) -> QueryAnalysis {
        // Try alternative strategy
        let alternative_strategy = match current.strategy {
            QueryStrategy::HybridParallel => QueryStrategy::HybridSequential,
            QueryStrategy::HybridSequential => QueryStrategy::HybridParallel,
            _ => return current.clone(),
        };

        let estimated_latency_ms = self.estimate_latency(&alternative_strategy, query);

        QueryAnalysis {
            strategy: alternative_strategy,
            estimated_latency_ms,
            requires_milvus: current.requires_milvus,
            requires_neo4j: current.requires_neo4j,
            requires_agentdb: current.requires_agentdb,
            cache_eligible: current.cache_eligible,
            reason: format!("Adaptive fallback: {}", self.explain_strategy(&alternative_strategy, query)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_planner_vector_only() {
        let planner = QueryPlanner::new(PlannerConfig::default());

        let query = super::super::hybrid_coordinator::SearchQuery {
            embedding: vec![0.1; 768],
            k: 10,
            user_id: String::new(),
            context: String::new(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: false,
            require_genre_filter: false,
            require_cultural_context: false,
            timeout_ms: 100,
        };

        let analysis = planner.plan(&query);
        assert_eq!(analysis.strategy, QueryStrategy::VectorOnly);
        assert!(analysis.estimated_latency_ms < 10.0);
    }

    #[test]
    fn test_query_planner_hybrid_parallel() {
        let planner = QueryPlanner::new(PlannerConfig::default());

        let query = super::super::hybrid_coordinator::SearchQuery {
            embedding: vec![0.1; 768],
            k: 50,
            user_id: "user1".to_string(),
            context: "browsing".to_string(),
            metadata_filters: HashMap::new(),
            graph_filters: HashMap::new(),
            include_relationships: true,
            require_genre_filter: true,
            require_cultural_context: false,
            timeout_ms: 100,
        };

        let analysis = planner.plan(&query);
        assert_eq!(analysis.strategy, QueryStrategy::HybridParallel);
        assert!(analysis.requires_milvus);
        assert!(analysis.requires_neo4j);
        assert!(analysis.requires_agentdb);
    }
}
