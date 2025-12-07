//! GPU-accelerated hyper-personalization system for semantic search
//!
//! This crate integrates all components of the hyper-personalization pipeline:
//! - Semantic model for query encoding (ONNX)
//! - User embeddings with real-time updates (GPU)
//! - Temporal cache for hot queries (GPU)
//! - Attention-based reranking (PyTorch/libtorch)
//!
//! # Architecture
//!
//! ```text
//! Query ’ SemanticModel ’ User Fusion ’ Cache/GPU Similarity ’ Attention Reranking ’ Top-K
//! ```
//!
//! # Performance Targets
//!
//! - End-to-end latency: <10ms (p95)
//! - GPU memory: ~4GB for 100K items
//! - Throughput: >1000 QPS

use anyhow::{Context, Result};
use attention::{AttentionReranker, ContextFeatures};
use cudarc::driver::CudaDevice;
use gpu_embeddings::GPUUserEmbeddings;
use semantic_model::{ModelConfig, SemanticModel};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tch::{nn, Device as TchDevice, Kind, Tensor};
use temporal_cache::TemporalGPUCache;
use tracing::{debug, info, warn};

mod error;
mod metrics;

pub use error::HyperPersonalizationError;
pub use metrics::PerformanceMetrics;

/// Complete hyper-personalization search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Top-K item indices with scores
    pub items: Vec<(usize, f32)>,

    /// Detailed timing breakdown
    pub timing: TimingBreakdown,

    /// Whether result came from cache
    pub from_cache: bool,

    /// User embedding stats
    pub user_stats: Option<UserStats>,
}

/// Detailed timing breakdown for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    /// Query encoding time (ONNX inference)
    pub query_encoding_ms: f64,

    /// User embedding fusion time
    pub user_fusion_ms: f64,

    /// Similarity computation time (cache or GPU)
    pub similarity_ms: f64,

    /// Attention reranking time
    pub attention_rerank_ms: f64,

    /// Top-K selection time
    pub topk_ms: f64,

    /// Total end-to-end time
    pub total_ms: f64,
}

/// User-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStats {
    /// Whether user has embedding
    pub has_embedding: bool,

    /// User embedding dimension
    pub embedding_dim: usize,

    /// Fusion weight applied
    pub fusion_weight: f32,
}

/// Main hyper-personalization system
pub struct HyperPersonalizationSystem {
    /// Semantic model for query encoding
    semantic_model: SemanticModel,

    /// GPU user embeddings
    user_embeddings: GPUUserEmbeddings,

    /// Temporal GPU cache
    temporal_cache: TemporalGPUCache,

    /// Attention reranker
    attention: AttentionReranker,

    /// Item embeddings on GPU (for similarity)
    item_embeddings_gpu: Arc<cudarc::driver::CudaSlice<f32>>,

    /// Item embeddings for PyTorch (for attention)
    item_embeddings_torch: Tensor,

    /// Number of items in catalog
    num_items: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// CUDA device
    cuda_device: Arc<CudaDevice>,

    /// PyTorch device
    torch_device: TchDevice,

    /// Performance metrics
    metrics: Arc<parking_lot::RwLock<PerformanceMetrics>>,
}

impl HyperPersonalizationSystem {
    /// Create a new hyper-personalization system
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to ONNX model file
    /// * `tokenizer_path` - Path to tokenizer.json
    /// * `embeddings_path` - Path to item embeddings file
    /// * `user_embeddings_path` - Path to user embeddings file
    /// * `config` - System configuration
    ///
    /// # Returns
    ///
    /// Initialized system ready for inference
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        embeddings_path: impl AsRef<Path>,
        user_embeddings_path: impl AsRef<Path>,
        config: SystemConfig,
    ) -> Result<Self> {
        info!("Initializing HyperPersonalizationSystem");
        let start = Instant::now();

        // 1. Initialize CUDA device
        let cuda_device = CudaDevice::new(0)
            .context("Failed to initialize CUDA device")?;
        let cuda_device = Arc::new(cuda_device);

        // 2. Initialize semantic model
        let model_config = ModelConfig {
            max_length: config.max_query_length,
            embedding_dim: config.embedding_dim,
            normalize: true,
            ..Default::default()
        };

        let semantic_model = SemanticModel::with_config(
            model_path,
            tokenizer_path,
            model_config,
        ).context("Failed to initialize semantic model")?;

        info!("Semantic model loaded");

        // 3. Load item embeddings
        let (item_embeddings_cpu, num_items, embedding_dim) =
            Self::load_item_embeddings(embeddings_path.as_ref())?;

        if embedding_dim != config.embedding_dim {
            anyhow::bail!(
                "Embedding dimension mismatch: config={}, loaded={}",
                config.embedding_dim,
                embedding_dim
            );
        }

        info!("Loaded {} item embeddings (dim={})", num_items, embedding_dim);

        // 4. Initialize GPU user embeddings
        let mut user_embeddings = GPUUserEmbeddings::new(
            cuda_device.clone(),
            embedding_dim,
        )?;

        user_embeddings
            .load_embeddings(user_embeddings_path.as_ref().to_str().unwrap())
            .context("Failed to load user embeddings")?;

        info!("Loaded {} user embeddings", user_embeddings.num_users());

        // 5. Initialize temporal cache
        let temporal_cache = TemporalGPUCache::new(
            &item_embeddings_cpu,
            num_items,
            embedding_dim,
            Some(config.cache_size),
            Some(config.cache_ttl_secs),
        )?;

        info!("Temporal cache initialized (size={})", config.cache_size);

        // 6. Copy item embeddings to GPU for CUDA operations
        let item_embeddings_gpu = cuda_device
            .htod_sync_copy(&item_embeddings_cpu)
            .context("Failed to copy item embeddings to GPU")?;
        let item_embeddings_gpu = Arc::new(item_embeddings_gpu);

        // 7. Initialize PyTorch device and convert embeddings for attention
        let torch_device = if config.use_gpu {
            TchDevice::Cuda(0)
        } else {
            TchDevice::Cpu
        };

        let item_embeddings_torch = Tensor::from_slice(&item_embeddings_cpu)
            .reshape(&[num_items as i64, embedding_dim as i64])
            .to_device(torch_device);

        info!("Item embeddings copied to GPU (torch device: {:?})", torch_device);

        // 8. Initialize attention reranker
        let vs = nn::VarStore::new(torch_device);
        let attention = AttentionReranker::new(&vs.root(), embedding_dim as i64);

        info!("Attention reranker initialized");

        let elapsed = start.elapsed();
        info!(
            "HyperPersonalizationSystem initialized in {:.2}ms",
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(Self {
            semantic_model,
            user_embeddings,
            temporal_cache,
            attention,
            item_embeddings_gpu,
            item_embeddings_torch,
            num_items,
            embedding_dim,
            cuda_device,
            torch_device,
            metrics: Arc::new(parking_lot::RwLock::new(PerformanceMetrics::default())),
        })
    }

    /// Perform personalized semantic search
    ///
    /// # Arguments
    ///
    /// * `user_id` - User identifier for personalization
    /// * `query` - Search query text
    /// * `top_k` - Number of results to return
    /// * `context` - Optional contextual features
    ///
    /// # Returns
    ///
    /// Search result with top-K items and detailed timing
    pub fn personalized_search(
        &mut self,
        user_id: &str,
        query: &str,
        top_k: usize,
        context: Option<&ContextFeatures>,
    ) -> Result<SearchResult> {
        let overall_start = Instant::now();
        let mut timing = TimingBreakdown {
            query_encoding_ms: 0.0,
            user_fusion_ms: 0.0,
            similarity_ms: 0.0,
            attention_rerank_ms: 0.0,
            topk_ms: 0.0,
            total_ms: 0.0,
        };

        // 1. Encode query
        let query_start = Instant::now();
        let query_embedding = self
            .semantic_model
            .encode(query)
            .context("Query encoding failed")?;
        timing.query_encoding_ms = query_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Query encoded in {:.3}ms: '{}'",
            timing.query_encoding_ms, query
        );

        // 2. Fuse with user embedding
        let fusion_start = Instant::now();
        let fused_embedding = self
            .user_embeddings
            .fuse_embeddings(user_id, &query_embedding)?;
        timing.user_fusion_ms = fusion_start.elapsed().as_secs_f64() * 1000.0;

        let has_user_embedding = self.user_embeddings.get_embedding(user_id).is_some();

        debug!(
            "User fusion in {:.3}ms (has_embedding={})",
            timing.user_fusion_ms, has_user_embedding
        );

        // 3. Check cache or compute similarities
        let sim_start = Instant::now();
        let cache_key = format!("{}:{}", user_id, query);

        let (similarities, from_cache) = if let Some(cached) = self.temporal_cache.get(&cache_key) {
            debug!("Cache HIT for query: '{}'", query);
            (
                cached.iter().map(|(idx, score)| (*idx, *score)).collect::<Vec<_>>(),
                true,
            )
        } else {
            debug!("Cache MISS for query: '{}'", query);

            // Compute similarities on GPU
            let sim_result = self.temporal_cache.get_similarities(
                &fused_embedding,
                Instant::now(),
            )?;

            let similarities_with_idx: Vec<(usize, f32)> = sim_result
                .similarities
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();

            // Cache the result
            self.temporal_cache
                .put(cache_key, fused_embedding.clone(), similarities_with_idx.clone())?;

            (similarities_with_idx, false)
        };

        timing.similarity_ms = sim_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Similarities computed in {:.3}ms (from_cache={})",
            timing.similarity_ms, from_cache
        );

        // 4. Get top candidates for reranking (2x top_k)
        let topk_start = Instant::now();
        let mut candidates = similarities.clone();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(top_k * 2);
        timing.topk_ms = topk_start.elapsed().as_secs_f64() * 1000.0;

        debug!("Top-{} candidates selected in {:.3}ms", top_k * 2, timing.topk_ms);

        // 5. Attention reranking
        let rerank_start = Instant::now();

        // Convert query to tensor
        let query_tensor = Tensor::from_slice(&fused_embedding)
            .to_device(self.torch_device);

        // Get candidate embeddings
        let candidate_indices: Vec<i64> = candidates
            .iter()
            .map(|(idx, _)| *idx as i64)
            .collect();

        let candidate_indices_tensor = Tensor::from_slice(&candidate_indices)
            .to_device(self.torch_device);

        let candidate_embeddings = self
            .item_embeddings_torch
            .index_select(0, &candidate_indices_tensor);

        // Apply attention
        let attention_scores = self.attention.forward(
            &query_tensor,
            &candidate_embeddings,
            context,
        );

        let attention_vec: Vec<f32> = attention_scores.try_into()?;

        // Combine attention scores with base similarities
        let mut reranked: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, &(idx, base_score))| {
                let attn_score = attention_vec.get(i).copied().unwrap_or(0.0);
                // Weighted combination: 0.6 attention + 0.4 base similarity
                let final_score = 0.6 * attn_score + 0.4 * base_score;
                (idx, final_score)
            })
            .collect();

        // Sort by final score
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        reranked.truncate(top_k);

        timing.attention_rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Attention reranking in {:.3}ms",
            timing.attention_rerank_ms
        );

        // 6. Finalize timing
        timing.total_ms = overall_start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Search completed in {:.3}ms (query={}, user={}, results={})",
            timing.total_ms, query, user_id, reranked.len()
        );

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.record_query(timing.total_ms, from_cache);
        }

        Ok(SearchResult {
            items: reranked,
            timing,
            from_cache,
            user_stats: Some(UserStats {
                has_embedding: has_user_embedding,
                embedding_dim: self.embedding_dim,
                fusion_weight: if has_user_embedding { 0.3 } else { 0.0 },
            }),
        })
    }

    /// Update user preferences based on interaction
    ///
    /// # Arguments
    ///
    /// * `user_id` - User identifier
    /// * `item_id` - Item that was interacted with
    /// * `rating` - Rating/score (0-1 scale)
    ///
    /// # Returns
    ///
    /// Updated user embedding
    pub fn update_user_preferences(
        &mut self,
        user_id: &str,
        item_id: usize,
        rating: f64,
    ) -> Result<()> {
        if item_id >= self.num_items {
            anyhow::bail!("Item ID {} out of bounds (max: {})", item_id, self.num_items);
        }

        // Get current user embedding or initialize
        let mut user_emb = self
            .user_embeddings
            .get_embedding(user_id)
            .unwrap_or_else(|| vec![0.0; self.embedding_dim]);

        // Get item embedding from GPU
        let offset = item_id * self.embedding_dim;
        let item_emb = self.cuda_device.dtoh_sync_copy_range(
            &self.item_embeddings_gpu,
            offset..offset + self.embedding_dim,
        )?;

        // Update user embedding: user' = user + lr * rating * item
        let learning_rate = 0.01;
        for i in 0..self.embedding_dim {
            user_emb[i] += (learning_rate * rating as f32) * item_emb[i];
        }

        // Normalize
        let norm: f32 = user_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut user_emb {
                *x /= norm;
            }
        }

        // Update in GPU memory
        self.user_embeddings.update_embedding(user_id, user_emb)?;

        debug!(
            "Updated user embedding for {} (item={}, rating={})",
            user_id, item_id, rating
        );

        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> temporal_cache::CacheStats {
        self.temporal_cache.cache_stats()
    }

    /// Get performance metrics
    pub fn metrics(&self) -> PerformanceMetrics {
        self.metrics.read().clone()
    }

    /// Reset performance metrics
    pub fn reset_metrics(&self) {
        *self.metrics.write() = PerformanceMetrics::default();
    }

    /// Load item embeddings from file
    fn load_item_embeddings(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
        info!("Loading item embeddings from {:?}", path);

        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read embeddings from {:?}", path))?;

        // Try deserializing as bincode
        if let Ok(embeddings) = bincode::deserialize::<Vec<f32>>(&data) {
            // Infer dimensions (assume square-ish matrix)
            let total = embeddings.len();
            let dim = 384; // Common dimension
            let num_items = total / dim;

            if num_items * dim != total {
                anyhow::bail!(
                    "Cannot infer dimensions: total={}, dim={}, remainder={}",
                    total,
                    dim,
                    total % dim
                );
            }

            return Ok((embeddings, num_items, dim));
        }

        // Try as JSON
        if let Ok(json_data) = serde_json::from_slice::<serde_json::Value>(&data) {
            if let Some(arr) = json_data.as_array() {
                let num_items = arr.len();
                let dim = arr
                    .get(0)
                    .and_then(|v| v.as_array())
                    .map(|v| v.len())
                    .unwrap_or(0);

                let mut embeddings = Vec::with_capacity(num_items * dim);
                for item in arr {
                    if let Some(item_arr) = item.as_array() {
                        for val in item_arr {
                            embeddings.push(val.as_f64().unwrap_or(0.0) as f32);
                        }
                    }
                }

                return Ok((embeddings, num_items, dim));
            }
        }

        anyhow::bail!("Unsupported embeddings format")
    }
}

/// System configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Maximum query length (tokens)
    pub max_query_length: usize,

    /// Cache size (number of popular items)
    pub cache_size: usize,

    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,

    /// Whether to use GPU for PyTorch
    pub use_gpu: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            max_query_length: 512,
            cache_size: 10_000,
            cache_ttl_secs: 3600, // 1 hour
            use_gpu: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model files
    fn test_system_initialization() {
        let config = SystemConfig::default();

        let system = HyperPersonalizationSystem::new(
            "models/semantic_model.onnx",
            "models/tokenizer.json",
            "data/item_embeddings.bin",
            "data/user_embeddings.bin",
            config,
        );

        assert!(system.is_ok());
    }

    #[test]
    #[ignore] // Requires model files
    fn test_personalized_search() {
        let config = SystemConfig::default();

        let mut system = HyperPersonalizationSystem::new(
            "models/semantic_model.onnx",
            "models/tokenizer.json",
            "data/item_embeddings.bin",
            "data/user_embeddings.bin",
            config,
        )
        .unwrap();

        let result = system
            .personalized_search("user123", "action movie", 10, None)
            .unwrap();

        assert!(result.items.len() <= 10);
        assert!(result.timing.total_ms > 0.0);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_user_preference_update() {
        let config = SystemConfig::default();

        let mut system = HyperPersonalizationSystem::new(
            "models/semantic_model.onnx",
            "models/tokenizer.json",
            "data/item_embeddings.bin",
            "data/user_embeddings.bin",
            config,
        )
        .unwrap();

        let result = system.update_user_preferences("user123", 42, 0.9);
        assert!(result.is_ok());
    }
}
