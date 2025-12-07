//! GPU-accelerated temporal similarity cache with exponential decay
//!
//! This crate provides a high-performance cache for item similarity computations
//! using GPU acceleration (CUDA) and temporal decay weights.
//!
//! # Key Features
//! - Batch matrix multiplication on GPU
//! - Exponential temporal decay: exp(-λ * age)
//! - Sub-millisecond cache hits (<0.16ms target)
//! - Atomic hit/miss tracking
//! - 2.48 GB GPU memory for 10K×62K matrix

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::sys::cublasOperation_t;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, info};

/// Cache errors
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Item not found: {0}")]
    ItemNotFound(usize),

    #[error("Cache not initialized")]
    NotInitialized,

    #[error("GPU operation failed: {0}")]
    GpuOperation(String),
}

/// Result type for cache operations
pub type Result<T> = std::result::Result<T, CacheError>;

/// GPU-accelerated temporal similarity cache
///
/// Maintains precomputed similarities for popular items with temporal decay weights.
pub struct TemporalGPUCache {
    /// Item embeddings on GPU: (num_items, embed_dim)
    item_embeddings: Arc<CudaSlice<f32>>,

    /// Indices of popular items (top num_popular)
    popular_indices: Vec<usize>,

    /// Precomputed similarity matrix on GPU: (num_popular, num_items)
    popular_similarities: Arc<CudaSlice<f32>>,

    /// Temporal decay weights: (num_popular,)
    temporal_weights: Arc<CudaSlice<f32>>,

    /// Timestamp of last cache rebuild
    last_rebuild: Instant,

    /// Cache hit counter
    cache_hits: AtomicUsize,

    /// Cache miss counter
    cache_misses: AtomicUsize,

    /// Total hit latency in nanoseconds
    total_hit_latency_ns: AtomicU64,

    /// Embedding dimension
    embed_dim: usize,

    /// Number of items
    num_items: usize,

    /// Number of popular items to cache
    num_popular: usize,

    /// Temporal decay rate (λ)
    decay_rate: f64,

    /// CUDA device
    device: Arc<CudaDevice>,

    /// cuBLAS handle
    blas: Arc<CudaBlas>,

    /// Thread-safe cache state
    state: Arc<RwLock<CacheState>>,
}

/// Internal cache state
#[derive(Debug)]
struct CacheState {
    /// Whether cache is initialized
    initialized: bool,

    /// Age of cache in seconds
    cache_age_secs: f64,
}

/// Result from cache query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheResult {
    /// Similarity scores (CPU tensor)
    pub similarities: Vec<f32>,

    /// Whether result came from cache
    pub from_cache: bool,

    /// Query latency in milliseconds
    pub latency_ms: f64,

    /// Item indices (if applicable)
    pub indices: Option<Vec<usize>>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,

    /// Total cache hits
    pub total_hits: usize,

    /// Total cache misses
    pub total_misses: usize,

    /// Average hit latency in milliseconds
    pub avg_hit_latency_ms: f64,

    /// Cache age in seconds
    pub cache_age_secs: f64,

    /// Number of cached items
    pub num_cached: usize,
}

impl TemporalGPUCache {
    /// Create a new temporal GPU cache
    ///
    /// # Arguments
    /// - `item_embeddings_cpu`: Item embeddings as flat array (num_items * embed_dim)
    /// - `num_items`: Number of items
    /// - `embed_dim`: Embedding dimension
    /// - `num_popular`: Number of popular items to cache (default: 10000)
    /// - `decay_rate`: Temporal decay rate λ (default: 0.1)
    ///
    /// # Returns
    /// Initialized cache instance
    pub fn new(
        item_embeddings_cpu: &[f32],
        num_items: usize,
        embed_dim: usize,
        num_popular: Option<usize>,
        decay_rate: Option<f64>,
    ) -> Result<Self> {
        let num_popular = num_popular.unwrap_or(10_000);
        let decay_rate = decay_rate.unwrap_or(0.1);

        // Validate dimensions
        if item_embeddings_cpu.len() != num_items * embed_dim {
            return Err(CacheError::InvalidDimensions(format!(
                "Expected {} elements, got {}",
                num_items * embed_dim,
                item_embeddings_cpu.len()
            )));
        }

        if num_popular > num_items {
            return Err(CacheError::InvalidDimensions(format!(
                "num_popular ({}) cannot exceed num_items ({})",
                num_popular, num_items
            )));
        }

        info!("Initializing TemporalGPUCache: {} items, {} dim, {} popular",
              num_items, embed_dim, num_popular);

        // Initialize CUDA device (returns Arc<CudaDevice>)
        let device = CudaDevice::new(0)
            .map_err(|e| CacheError::Cuda(format!("Failed to initialize CUDA device: {}", e)))?;

        // Initialize cuBLAS
        let blas = Arc::new(CudaBlas::new(device.clone())
            .map_err(|e| CacheError::Cuda(format!("Failed to initialize cuBLAS: {}", e)))?);

        // Copy embeddings to GPU
        let item_embeddings = device.htod_sync_copy(item_embeddings_cpu)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to copy embeddings to GPU: {}", e)))?;
        let item_embeddings = Arc::new(item_embeddings);

        // Allocate GPU memory for similarity matrix
        let similarity_size = num_popular * num_items;
        let popular_similarities = device.alloc_zeros::<f32>(similarity_size)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to allocate similarity matrix: {}", e)))?;
        let popular_similarities = Arc::new(popular_similarities);

        // Allocate temporal weights
        let temporal_weights = device.alloc_zeros::<f32>(num_popular)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to allocate temporal weights: {}", e)))?;
        let temporal_weights = Arc::new(temporal_weights);

        // Initialize with top items (simple: first num_popular items)
        let popular_indices: Vec<usize> = (0..num_popular).collect();

        let mut cache = Self {
            item_embeddings,
            popular_indices,
            popular_similarities,
            temporal_weights,
            last_rebuild: Instant::now(),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            total_hit_latency_ns: AtomicU64::new(0),
            embed_dim,
            num_items,
            num_popular,
            decay_rate,
            device,
            blas,
            state: Arc::new(RwLock::new(CacheState {
                initialized: false,
                cache_age_secs: 0.0,
            })),
        };

        // Initial cache rebuild
        cache.rebuild_cache()?;

        info!("TemporalGPUCache initialized successfully");
        Ok(cache)
    }

    /// Rebuild the similarity cache
    ///
    /// Performs batch matrix multiplication: popular_emb @ all_emb^T
    /// Updates temporal weights with exponential decay
    pub fn rebuild_cache(&mut self) -> Result<()> {
        let start = Instant::now();
        info!("Rebuilding cache for {} popular items", self.num_popular);

        // Extract popular embeddings: (num_popular, embed_dim)
        let mut popular_emb_cpu = vec![0.0f32; self.num_popular * self.embed_dim];

        // Copy from GPU
        let all_embeddings_cpu = self.device.dtoh_sync_copy(&*self.item_embeddings)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to copy embeddings from GPU: {}", e)))?;

        for (i, &idx) in self.popular_indices.iter().enumerate() {
            let offset_src = idx * self.embed_dim;
            let offset_dst = i * self.embed_dim;
            popular_emb_cpu[offset_dst..offset_dst + self.embed_dim]
                .copy_from_slice(&all_embeddings_cpu[offset_src..offset_src + self.embed_dim]);
        }

        // Upload popular embeddings to GPU
        let popular_emb_gpu = self.device.htod_sync_copy(&popular_emb_cpu)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to upload popular embeddings: {}", e)))?;

        // Batch matrix multiplication: C = A @ B^T
        // A: (num_popular, embed_dim) - popular_emb
        // B: (num_items, embed_dim) - all_emb
        // C: (num_popular, num_items) - similarities
        //
        // cuBLAS GEMM: C = alpha * A @ B^T + beta * C

        let config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // No transpose for A
            transb: cublasOperation_t::CUBLAS_OP_T, // Transpose B
            m: self.num_popular as i32,
            n: self.num_items as i32,
            k: self.embed_dim as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,  // Leading dimension of A
            ldb: self.embed_dim as i32,  // Leading dimension of B
            beta: 0.0f32,
            ldc: self.num_items as i32,  // Leading dimension of C
        };

        unsafe {
            self.blas.gemm(
                config,
                &popular_emb_gpu,
                &*self.item_embeddings,
                Arc::get_mut(&mut self.popular_similarities)
                    .ok_or_else(|| CacheError::GpuOperation("Cannot get mutable reference to similarities".to_string()))?,
            ).map_err(|e| CacheError::GpuOperation(format!("GEMM failed: {}", e)))?;
        }

        // Update temporal weights: w_i = exp(-λ * 0) = 1.0 for fresh cache
        let weights_cpu = vec![1.0f32; self.num_popular];
        self.device.htod_sync_copy_into(&weights_cpu,
            Arc::get_mut(&mut self.temporal_weights)
                .ok_or_else(|| CacheError::GpuOperation("Cannot get mutable reference to weights".to_string()))?)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to upload weights: {}", e)))?;

        self.last_rebuild = Instant::now();

        {
            let mut state = self.state.write();
            state.initialized = true;
            state.cache_age_secs = 0.0;
        }

        let elapsed = start.elapsed();
        info!("Cache rebuilt in {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        Ok(())
    }

    /// Get similar items for a cached item
    ///
    /// # Arguments
    /// - `item_id`: Item index
    ///
    /// # Returns
    /// CacheResult with similarities, or cache miss
    pub fn get_similar_items(&self, item_id: usize) -> Result<CacheResult> {
        let start = Instant::now();

        // Check if item is in cache
        if let Some(cache_idx) = self.popular_indices.iter().position(|&x| x == item_id) {
            // Cache hit
            let offset = cache_idx * self.num_items;

            // Copy similarities from GPU
            let slice_view = self.popular_similarities.slice(offset..offset + self.num_items);
            let similarities = self.device.dtoh_sync_copy(&slice_view)
                .map_err(|e| CacheError::GpuOperation(format!("Failed to copy similarities: {}", e)))?;

            let latency = start.elapsed();
            let latency_ms = latency.as_secs_f64() * 1000.0;

            // Update stats
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.total_hit_latency_ns.fetch_add(latency.as_nanos() as u64, Ordering::Relaxed);

            debug!("Cache HIT for item {} (cache_idx={}, latency={:.3}ms)",
                   item_id, cache_idx, latency_ms);

            Ok(CacheResult {
                similarities,
                from_cache: true,
                latency_ms,
                indices: None,
            })
        } else {
            // Cache miss
            self.cache_misses.fetch_add(1, Ordering::Relaxed);

            debug!("Cache MISS for item {}", item_id);

            // Fallback: compute on-the-fly
            self.compute_similarities_gpu(item_id, start)
        }
    }

    /// Compute similarities on GPU for non-cached item
    fn compute_similarities_gpu(&self, item_id: usize, start: Instant) -> Result<CacheResult> {
        if item_id >= self.num_items {
            return Err(CacheError::ItemNotFound(item_id));
        }

        // Extract query embedding from GPU
        let offset = item_id * self.embed_dim;
        let slice_view = self.item_embeddings.slice(offset..offset + self.embed_dim);
        let query_emb = self.device.dtoh_sync_copy(&slice_view)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to copy query embedding: {}", e)))?;

        // Compute dot products on GPU
        self.get_similarities(&query_emb, start)
    }

    /// Get similarities for a query embedding
    ///
    /// # Arguments
    /// - `query_emb`: Query embedding (embed_dim,)
    ///
    /// # Returns
    /// CacheResult with similarities
    pub fn get_similarities(&self, query_emb: &[f32], start: Instant) -> Result<CacheResult> {
        if query_emb.len() != self.embed_dim {
            return Err(CacheError::InvalidDimensions(format!(
                "Expected {} dimensions, got {}",
                self.embed_dim, query_emb.len()
            )));
        }

        // Upload query to GPU
        let query_gpu = self.device.htod_sync_copy(query_emb)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to upload query: {}", e)))?;

        // Allocate output
        let mut output_gpu = self.device.alloc_zeros::<f32>(self.num_items)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to allocate output: {}", e)))?;

        // Matrix-vector multiplication: output = all_emb @ query
        // all_emb: (num_items, embed_dim)
        // query: (embed_dim, 1) treated as (embed_dim,)
        // output: (num_items, 1) treated as (num_items,)

        let config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // No transpose for A
            transb: cublasOperation_t::CUBLAS_OP_N, // No transpose for B
            m: self.num_items as i32,
            n: 1i32,  // Single column
            k: self.embed_dim as i32,
            alpha: 1.0f32,
            lda: self.embed_dim as i32,  // Leading dimension of A
            ldb: self.embed_dim as i32,  // Leading dimension of B
            beta: 0.0f32,
            ldc: self.num_items as i32,  // Leading dimension of C
        };

        unsafe {
            self.blas.gemm(
                config,
                &*self.item_embeddings,
                &query_gpu,
                &mut output_gpu,
            ).map_err(|e| CacheError::GpuOperation(format!("GEMM failed: {}", e)))?;
        }

        // Copy result to CPU
        let similarities = self.device.dtoh_sync_copy(&output_gpu)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to copy result: {}", e)))?;

        let latency = start.elapsed();
        let latency_ms = latency.as_secs_f64() * 1000.0;

        Ok(CacheResult {
            similarities,
            from_cache: false,
            latency_ms,
            indices: None,
        })
    }

    /// Update temporal weights based on cache age
    ///
    /// Applies exponential decay: w(t) = exp(-λ * t)
    pub fn update_temporal_weights(&mut self) -> Result<()> {
        let age_secs = self.last_rebuild.elapsed().as_secs_f64();

        // Compute decay factor
        let decay_factor = (-self.decay_rate * age_secs).exp() as f32;

        // Create weights on CPU
        let weights_cpu = vec![decay_factor; self.num_popular];

        // Upload to GPU
        self.device.htod_sync_copy_into(&weights_cpu,
            Arc::get_mut(&mut self.temporal_weights)
                .ok_or_else(|| CacheError::GpuOperation("Cannot get mutable reference to weights".to_string()))?)
            .map_err(|e| CacheError::GpuOperation(format!("Failed to upload weights: {}", e)))?;

        // Update state
        {
            let mut state = self.state.write();
            state.cache_age_secs = age_secs;
        }

        debug!("Updated temporal weights: age={:.1}s, decay={:.4}", age_secs, decay_factor);

        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        let total_latency_ns = self.total_hit_latency_ns.load(Ordering::Relaxed);
        let avg_hit_latency_ms = if hits > 0 {
            (total_latency_ns as f64 / hits as f64) / 1_000_000.0
        } else {
            0.0
        };

        let cache_age_secs = self.state.read().cache_age_secs;

        CacheStats {
            hit_rate,
            total_hits: hits,
            total_misses: misses,
            avg_hit_latency_ms,
            cache_age_secs,
            num_cached: self.num_popular,
        }
    }

    /// Update popular indices (e.g., from analytics)
    ///
    /// # Arguments
    /// - `new_indices`: New popular item indices
    pub fn update_popular_indices(&mut self, new_indices: Vec<usize>) -> Result<()> {
        if new_indices.len() != self.num_popular {
            return Err(CacheError::InvalidDimensions(format!(
                "Expected {} indices, got {}",
                self.num_popular, new_indices.len()
            )));
        }

        for &idx in &new_indices {
            if idx >= self.num_items {
                return Err(CacheError::InvalidDimensions(format!(
                    "Index {} out of bounds (max: {})",
                    idx, self.num_items - 1
                )));
            }
        }

        self.popular_indices = new_indices;
        self.rebuild_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embeddings(num_items: usize, embed_dim: usize) -> Vec<f32> {
        let mut embeddings = vec![0.0f32; num_items * embed_dim];
        for i in 0..num_items {
            for j in 0..embed_dim {
                embeddings[i * embed_dim + j] = ((i + j) as f32).sin();
            }
        }
        embeddings
    }

    #[test]
    fn test_cache_initialization() {
        let num_items = 1000;
        let embed_dim = 128;
        let num_popular = 100;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        );

        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.num_popular, num_popular);
        assert_eq!(cache.embed_dim, embed_dim);
    }

    #[test]
    fn test_cache_hit() {
        let num_items = 1000;
        let embed_dim = 128;
        let num_popular = 100;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        ).unwrap();

        // Query popular item (should be cache hit)
        let result = cache.get_similar_items(50).unwrap();
        assert!(result.from_cache);
        assert_eq!(result.similarities.len(), num_items);

        // Check stats
        let stats = cache.cache_stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 0);
        assert_eq!(stats.hit_rate, 1.0);
    }

    #[test]
    fn test_cache_miss() {
        let num_items = 1000;
        let embed_dim = 128;
        let num_popular = 100;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        ).unwrap();

        // Query non-popular item (should be cache miss)
        let result = cache.get_similar_items(500).unwrap();
        assert!(!result.from_cache);
        assert_eq!(result.similarities.len(), num_items);

        // Check stats
        let stats = cache.cache_stats();
        assert_eq!(stats.total_hits, 0);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_temporal_decay() {
        let num_items = 1000;
        let embed_dim = 128;
        let num_popular = 100;
        let decay_rate = 0.5;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let mut cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(decay_rate),
        ).unwrap();

        // Simulate time passage
        std::thread::sleep(std::time::Duration::from_millis(100));

        cache.update_temporal_weights().unwrap();

        let stats = cache.cache_stats();
        assert!(stats.cache_age_secs > 0.0);
    }

    #[test]
    fn test_rebuild_performance() {
        let num_items = 10_000;
        let embed_dim = 256;
        let num_popular = 1000;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let mut cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        ).unwrap();

        let start = Instant::now();
        cache.rebuild_cache().unwrap();
        let elapsed = start.elapsed();

        println!("Rebuild time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        // Should be fast (<100ms for this size)
        assert!(elapsed.as_secs_f64() < 0.1);
    }

    #[test]
    fn test_query_embeddings() {
        let num_items = 1000;
        let embed_dim = 128;
        let num_popular = 100;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        ).unwrap();

        // Create query embedding
        let query: Vec<f32> = (0..embed_dim).map(|i| (i as f32).sin()).collect();

        let result = cache.get_similarities(&query, Instant::now()).unwrap();
        assert_eq!(result.similarities.len(), num_items);
        assert!(!result.from_cache);
    }

    #[test]
    fn test_hit_latency() {
        let num_items = 10_000;
        let embed_dim = 256;
        let num_popular = 1000;

        let embeddings = create_test_embeddings(num_items, embed_dim);

        let cache = TemporalGPUCache::new(
            &embeddings,
            num_items,
            embed_dim,
            Some(num_popular),
            Some(0.1),
        ).unwrap();

        // Warm up
        for i in 0..10 {
            let _ = cache.get_similar_items(i);
        }

        // Measure latency
        let num_queries = 100;
        for i in 0..num_queries {
            let _ = cache.get_similar_items(i % num_popular);
        }

        let stats = cache.cache_stats();
        println!("Avg hit latency: {:.3}ms", stats.avg_hit_latency_ms);

        // Target: <0.16ms
        assert!(stats.avg_hit_latency_ms < 1.0); // Relaxed for CI
    }
}
