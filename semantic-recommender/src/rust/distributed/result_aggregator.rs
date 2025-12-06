use std::collections::{HashMap, HashSet};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

use super::gpu_node_service::SearchResult;

const DIVERSITY_THRESHOLD: f32 = 0.95;
const MIN_SCORE_THRESHOLD: f32 = 0.1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub enable_diversity: bool,
    pub enable_deduplication: bool,
    pub enable_score_normalization: bool,
    pub diversity_lambda: f32,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            enable_diversity: true,
            enable_deduplication: true,
            enable_score_normalization: true,
            diversity_lambda: 0.5,
        }
    }
}

pub struct ResultAggregator {
    config: AggregationConfig,
}

impl ResultAggregator {
    pub fn new() -> Self {
        Self {
            config: AggregationConfig::default(),
        }
    }

    pub fn with_config(config: AggregationConfig) -> Self {
        Self { config }
    }

    pub async fn aggregate(
        &self,
        mut results: Vec<SearchResult>,
        k: usize,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<SearchResult>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        if self.config.enable_deduplication {
            results = self.deduplicate(results);
        }

        if self.config.enable_score_normalization {
            self.normalize_scores(&mut results);
        }

        results = self.apply_filters(results, filters);

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if self.config.enable_diversity {
            results = self.enforce_diversity(results, k);
        } else {
            results.truncate(k);
        }

        Ok(results)
    }

    fn deduplicate(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut seen = HashMap::new();
        let mut deduped = Vec::new();

        for result in results {
            seen.entry(result.id.clone())
                .and_modify(|existing: &mut SearchResult| {
                    if result.score > existing.score {
                        *existing = result.clone();
                    }
                })
                .or_insert_with(|| {
                    deduped.push(result.clone());
                    result
                });
        }

        deduped
    }

    fn normalize_scores(&self, results: &mut [SearchResult]) {
        if results.is_empty() {
            return;
        }

        let max_score = results
            .iter()
            .map(|r| r.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        let min_score = results
            .iter()
            .map(|r| r.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let range = max_score - min_score;
        if range > 0.0 {
            for result in results.iter_mut() {
                result.score = (result.score - min_score) / range;
            }
        }
    }

    fn apply_filters(
        &self,
        results: Vec<SearchResult>,
        filters: &HashMap<String, String>,
    ) -> Vec<SearchResult> {
        if filters.is_empty() {
            return results;
        }

        results
            .into_iter()
            .filter(|result| {
                filters.iter().all(|(key, value)| {
                    result
                        .metadata
                        .get(key)
                        .map(|v| v == value)
                        .unwrap_or(false)
                })
            })
            .collect()
    }

    fn enforce_diversity(&self, results: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
        if results.len() <= k {
            return results;
        }

        let mut selected = Vec::new();
        let mut remaining = results;

        if let Some(first) = remaining.first() {
            selected.push(first.clone());
        }

        remaining.remove(0);

        while selected.len() < k && !remaining.is_empty() {
            let next_idx = self.find_most_diverse(&selected, &remaining);
            selected.push(remaining[next_idx].clone());
            remaining.remove(next_idx);
        }

        selected
    }

    fn find_most_diverse(
        &self,
        selected: &[SearchResult],
        candidates: &[SearchResult],
    ) -> usize {
        let lambda = self.config.diversity_lambda;

        candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = self.diversity_score(a, selected, lambda);
                let score_b = self.diversity_score(b, selected, lambda);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn diversity_score(
        &self,
        candidate: &SearchResult,
        selected: &[SearchResult],
        lambda: f32,
    ) -> f32 {
        let relevance = candidate.score;

        let max_similarity = selected
            .iter()
            .map(|s| self.metadata_similarity(candidate, s))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let diversity = 1.0 - max_similarity;

        lambda * relevance + (1.0 - lambda) * diversity
    }

    fn metadata_similarity(&self, a: &SearchResult, b: &SearchResult) -> f32 {
        let keys_a: HashSet<_> = a.metadata.keys().collect();
        let keys_b: HashSet<_> = b.metadata.keys().collect();

        let intersection = keys_a.intersection(&keys_b).count();
        let union = keys_a.union(&keys_b).count();

        if union == 0 {
            return 0.0;
        }

        let mut common_values = 0;
        for key in keys_a.intersection(&keys_b) {
            if a.metadata.get(*key) == b.metadata.get(*key) {
                common_values += 1;
            }
        }

        (common_values as f32) / (union as f32)
    }

    pub async fn merge_with_reranking(
        &self,
        shard_results: Vec<Vec<SearchResult>>,
        k: usize,
        rerank_fn: impl Fn(&SearchResult) -> f32,
    ) -> Result<Vec<SearchResult>> {
        let mut all_results: Vec<SearchResult> = shard_results
            .into_iter()
            .flatten()
            .collect();

        for result in &mut all_results {
            let rerank_score = rerank_fn(result);
            result.score = 0.7 * result.score + 0.3 * rerank_score;
        }

        self.aggregate(all_results, k, &HashMap::new()).await
    }

    pub fn compute_mrr(&self, results: &[SearchResult], relevant_ids: &HashSet<String>) -> f32 {
        for (idx, result) in results.iter().enumerate() {
            if relevant_ids.contains(&result.id) {
                return 1.0 / (idx + 1) as f32;
            }
        }
        0.0
    }

    pub fn compute_ndcg(
        &self,
        results: &[SearchResult],
        relevance_scores: &HashMap<String, f32>,
    ) -> f32 {
        let dcg: f32 = results
            .iter()
            .enumerate()
            .map(|(idx, result)| {
                let relevance = relevance_scores.get(&result.id).unwrap_or(&0.0);
                relevance / ((idx + 2) as f32).log2()
            })
            .sum();

        let mut ideal_relevances: Vec<f32> = relevance_scores.values().copied().collect();
        ideal_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let idcg: f32 = ideal_relevances
            .iter()
            .take(results.len())
            .enumerate()
            .map(|(idx, rel)| rel / ((idx + 2) as f32).log2())
            .sum();

        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }

    pub fn compute_precision_at_k(
        &self,
        results: &[SearchResult],
        relevant_ids: &HashSet<String>,
        k: usize,
    ) -> f32 {
        let relevant_in_topk = results
            .iter()
            .take(k)
            .filter(|r| relevant_ids.contains(&r.id))
            .count();

        (relevant_in_topk as f32) / (k as f32)
    }

    pub fn compute_recall_at_k(
        &self,
        results: &[SearchResult],
        relevant_ids: &HashSet<String>,
        k: usize,
    ) -> f32 {
        if relevant_ids.is_empty() {
            return 0.0;
        }

        let relevant_in_topk = results
            .iter()
            .take(k)
            .filter(|r| relevant_ids.contains(&r.id))
            .count();

        (relevant_in_topk as f32) / (relevant_ids.len() as f32)
    }
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_deduplication() {
        let aggregator = ResultAggregator::new();

        let results = vec![
            create_test_result("id1", 0.9),
            create_test_result("id1", 0.8),
            create_test_result("id2", 0.7),
        ];

        let aggregated = aggregator
            .aggregate(results, 10, &HashMap::new())
            .await
            .unwrap();

        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].id, "id1");
        assert_eq!(aggregated[0].score, 0.9);
    }

    #[tokio::test]
    async fn test_score_normalization() {
        let aggregator = ResultAggregator::new();

        let results = vec![
            create_test_result("id1", 100.0),
            create_test_result("id2", 50.0),
            create_test_result("id3", 0.0),
        ];

        let aggregated = aggregator
            .aggregate(results, 10, &HashMap::new())
            .await
            .unwrap();

        assert_eq!(aggregated[0].score, 1.0);
        assert_eq!(aggregated[1].score, 0.5);
        assert_eq!(aggregated[2].score, 0.0);
    }

    #[tokio::test]
    async fn test_filtering() {
        let aggregator = ResultAggregator::new();

        let mut metadata1 = HashMap::new();
        metadata1.insert("category".to_string(), "sports".to_string());

        let mut metadata2 = HashMap::new();
        metadata2.insert("category".to_string(), "news".to_string());

        let results = vec![
            SearchResult {
                id: "id1".to_string(),
                score: 0.9,
                metadata: metadata1,
            },
            SearchResult {
                id: "id2".to_string(),
                score: 0.8,
                metadata: metadata2,
            },
        ];

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), "sports".to_string());

        let aggregated = aggregator.aggregate(results, 10, &filters).await.unwrap();

        assert_eq!(aggregated.len(), 1);
        assert_eq!(aggregated[0].id, "id1");
    }

    #[test]
    fn test_mrr_calculation() {
        let aggregator = ResultAggregator::new();

        let results = vec![
            create_test_result("id1", 0.9),
            create_test_result("id2", 0.8),
            create_test_result("id3", 0.7),
        ];

        let mut relevant = HashSet::new();
        relevant.insert("id2".to_string());

        let mrr = aggregator.compute_mrr(&results, &relevant);
        assert_eq!(mrr, 0.5);
    }

    #[test]
    fn test_precision_recall() {
        let aggregator = ResultAggregator::new();

        let results = vec![
            create_test_result("id1", 0.9),
            create_test_result("id2", 0.8),
            create_test_result("id3", 0.7),
            create_test_result("id4", 0.6),
        ];

        let mut relevant = HashSet::new();
        relevant.insert("id1".to_string());
        relevant.insert("id3".to_string());
        relevant.insert("id5".to_string());

        let precision = aggregator.compute_precision_at_k(&results, &relevant, 3);
        assert_eq!(precision, 2.0 / 3.0);

        let recall = aggregator.compute_recall_at_k(&results, &relevant, 3);
        assert_eq!(recall, 2.0 / 3.0);
    }

    #[test]
    fn test_ndcg() {
        let aggregator = ResultAggregator::new();

        let results = vec![
            create_test_result("id1", 0.9),
            create_test_result("id2", 0.8),
            create_test_result("id3", 0.7),
        ];

        let mut relevance = HashMap::new();
        relevance.insert("id1".to_string(), 3.0);
        relevance.insert("id2".to_string(), 2.0);
        relevance.insert("id3".to_string(), 1.0);

        let ndcg = aggregator.compute_ndcg(&results, &relevance);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
    }
}
