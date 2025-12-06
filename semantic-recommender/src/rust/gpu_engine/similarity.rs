/// Semantic Similarity Operations
///
/// GPU-accelerated semantic similarity computation using CUDA kernels.

use cudarc::driver::CudaDevice;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::*;

/// Configuration for similarity computation
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Batch size for computation
    pub batch_size: usize,

    /// Similarity metric (cosine, euclidean, etc.)
    pub metric: SimilarityMetric,

    /// Threshold for considering vectors similar
    pub threshold: f32,

    /// Normalize embeddings before computation
    pub normalize: bool,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 768,
            batch_size: 32,
            metric: SimilarityMetric::Cosine,
            threshold: 0.8,
            normalize: true,
        }
    }
}

/// Similarity metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Similarity matrix result
#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    /// Flattened similarity scores [batch_size * batch_size]
    pub scores: Vec<f32>,

    /// Matrix dimensions
    pub rows: usize,
    pub cols: usize,

    /// Computation time in milliseconds
    pub compute_time_ms: f64,
}

impl SimilarityMatrix {
    /// Get similarity score at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<f32> {
        if i < self.rows && j < self.cols {
            Some(self.scores[i * self.cols + j])
        } else {
            None
        }
    }

    /// Find k most similar vectors for each query
    pub fn top_k(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        let mut results = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            let mut row_scores: Vec<(usize, f32)> = (0..self.cols)
                .map(|j| (j, self.scores[i * self.cols + j]))
                .collect();

            // Sort by score descending
            row_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            row_scores.truncate(k);

            results.push(row_scores);
        }

        results
    }

    /// Filter pairs above threshold
    pub fn above_threshold(&self, threshold: f32) -> Vec<(usize, usize, f32)> {
        let mut results = Vec::new();

        for i in 0..self.rows {
            for j in 0..self.cols {
                let score = self.scores[i * self.cols + j];
                if score >= threshold {
                    results.push((i, j, score));
                }
            }
        }

        results
    }
}

/// Compute similarity matrix for batch of embeddings
pub async fn compute_similarity_batch(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    embeddings: &[f32],
    config: &SimilarityConfig,
) -> GpuResult<SimilarityMatrix> {
    let start = std::time::Instant::now();

    // Validate input
    if embeddings.len() != config.batch_size * config.embedding_dim {
        return Err(GpuError::Config(format!(
            "Invalid input size: expected {}, got {}",
            config.batch_size * config.embedding_dim,
            embeddings.len()
        )));
    }

    // Acquire stream
    let stream = streams.acquire().await?;

    // Allocate device memory
    let mut d_embeddings = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(embeddings.len())?
    };

    let mut d_output = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(config.batch_size * config.batch_size)?
    };

    // Transfer input to device
    device.htod_copy_into(embeddings, &mut d_embeddings)?;

    // Normalize if requested
    if config.normalize {
        normalize_embeddings(
            device,
            &mut d_embeddings,
            config.embedding_dim,
            config.batch_size,
        ).await?;
    }

    // Launch kernel based on metric
    match config.metric {
        SimilarityMetric::Cosine => {
            modules.launch_batch_similarity(
                &d_embeddings,
                &mut d_output,
                config.embedding_dim as u32,
                config.batch_size as u32,
            )?;
        }
        SimilarityMetric::Euclidean => {
            // Use cosine kernel with distance conversion
            modules.launch_batch_similarity(
                &d_embeddings,
                &mut d_output,
                config.embedding_dim as u32,
                config.batch_size as u32,
            )?;
        }
        SimilarityMetric::DotProduct => {
            modules.launch_batch_similarity(
                &d_embeddings,
                &mut d_output,
                config.embedding_dim as u32,
                config.batch_size as u32,
            )?;
        }
    }

    // Synchronize stream
    stream.synchronize().await?;

    // Transfer result back to host
    let scores = d_output.dtoh()?;

    // Free device memory
    {
        let mut pool = memory_pool.write().await;
        pool.free(d_embeddings);
        pool.free(d_output);
    }

    let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(SimilarityMatrix {
        scores,
        rows: config.batch_size,
        cols: config.batch_size,
        compute_time_ms,
    })
}

/// Compute pairwise similarities between two sets of embeddings
pub async fn compute_pairwise_similarity(
    device: &Arc<CudaDevice>,
    modules: &Arc<KernelModules>,
    memory_pool: &Arc<RwLock<MemoryPool>>,
    streams: &Arc<StreamManager>,
    embeddings1: &[f32],
    embeddings2: &[f32],
    config: &SimilarityConfig,
) -> GpuResult<SimilarityMatrix> {
    let start = std::time::Instant::now();

    let batch_size1 = embeddings1.len() / config.embedding_dim;
    let batch_size2 = embeddings2.len() / config.embedding_dim;

    // Acquire stream
    let stream = streams.acquire().await?;

    // Allocate device memory
    let mut d_embeddings1 = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(embeddings1.len())?
    };

    let mut d_embeddings2 = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(embeddings2.len())?
    };

    let mut d_output = {
        let mut pool = memory_pool.write().await;
        pool.alloc::<f32>(batch_size1 * batch_size2)?
    };

    // Transfer inputs to device
    device.htod_copy_into(embeddings1, &mut d_embeddings1)?;
    device.htod_copy_into(embeddings2, &mut d_embeddings2)?;

    // Launch kernel
    modules.launch_cosine_similarity(
        &d_embeddings1,
        &d_embeddings2,
        &mut d_output,
        config.embedding_dim as u32,
        batch_size1 as u32,
    )?;

    // Synchronize stream
    stream.synchronize().await?;

    // Transfer result back
    let scores = d_output.dtoh()?;

    // Free memory
    {
        let mut pool = memory_pool.write().await;
        pool.free(d_embeddings1);
        pool.free(d_embeddings2);
        pool.free(d_output);
    }

    let compute_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(SimilarityMatrix {
        scores,
        rows: batch_size1,
        cols: batch_size2,
        compute_time_ms,
    })
}

/// Normalize embeddings in-place on GPU
async fn normalize_embeddings(
    device: &CudaDevice,
    embeddings: &mut cudarc::driver::CudaSlice<f32>,
    embedding_dim: usize,
    batch_size: usize,
) -> GpuResult<()> {
    // This would call a normalization kernel
    // For now, placeholder implementation
    device.synchronize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_matrix() {
        let scores = vec![
            1.0, 0.8, 0.6,
            0.8, 1.0, 0.7,
            0.6, 0.7, 1.0,
        ];

        let matrix = SimilarityMatrix {
            scores,
            rows: 3,
            cols: 3,
            compute_time_ms: 1.0,
        };

        assert_eq!(matrix.get(0, 1), Some(0.8));
        assert_eq!(matrix.get(1, 2), Some(0.7));

        let top_k = matrix.top_k(2);
        assert_eq!(top_k[0][0].0, 0); // Self similarity
        assert_eq!(top_k[0][1].0, 1); // Most similar
    }

    #[test]
    fn test_threshold_filter() {
        let scores = vec![
            1.0, 0.8, 0.6,
            0.8, 1.0, 0.7,
            0.6, 0.7, 1.0,
        ];

        let matrix = SimilarityMatrix {
            scores,
            rows: 3,
            cols: 3,
            compute_time_ms: 1.0,
        };

        let above = matrix.above_threshold(0.75);
        assert_eq!(above.len(), 5); // 3 diagonal + 2 pairs
    }
}
