//! Semantic model using ONNX Runtime for optimized inference
//!
//! This crate provides a high-performance implementation of sentence transformer models
//! using ONNX Runtime as the inference backend. It supports:
//!
//! - Fast single-text and batch encoding
//! - GPU acceleration via CUDA (when available)
//! - Mean pooling for sentence embeddings
//! - Efficient tokenization with Hugging Face tokenizers

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

mod pooling;
mod preprocessing;

pub use pooling::PoolingStrategy;
pub use preprocessing::TextPreprocessor;

/// Errors that can occur during model operations
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("ONNX error: {0}")]
    OnnxError(String),

    #[error("Tokenization error: {0}")]
    TokenError(String),

    #[error("Invalid dimensions: expected {expected}, got {actual}")]
    DimensionError { expected: usize, actual: usize },

    #[error("Model not initialized")]
    NotInitialized,

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Inference error: {0}")]
    InferenceError(String),
}

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

/// Configuration for the semantic model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Maximum sequence length for tokenization
    pub max_length: usize,

    /// Whether to normalize embeddings to unit length
    pub normalize: bool,

    /// Pooling strategy to use
    pub pooling_strategy: PoolingStrategy,

    /// Whether to use GPU if available
    pub use_gpu: bool,

    /// Embedding dimension (384 for MiniLM)
    pub embedding_dim: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean,
            use_gpu: true,
            embedding_dim: 384,
        }
    }
}

/// Statistics about model performance
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelStats {
    pub total_encodings: u64,
    pub total_tokens_processed: u64,
    pub avg_encoding_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Main semantic model for generating embeddings
pub struct SemanticModel {
    /// ONNX Runtime session (wrapped in Arc for interior mutability)
    session: Arc<parking_lot::Mutex<Session>>,

    /// Tokenizer for text preprocessing
    tokenizer: Tokenizer,

    /// Model configuration
    config: ModelConfig,

    /// Text preprocessor
    preprocessor: TextPreprocessor,

    /// Model statistics
    stats: Arc<RwLock<ModelStats>>,
}

impl SemanticModel {
    /// Create a new semantic model from ONNX model and tokenizer paths
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use semantic_model::SemanticModel;
    ///
    /// let model = SemanticModel::new(
    ///     "models/model.onnx",
    ///     "models/tokenizer.json"
    /// ).unwrap();
    /// ```
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(model_path, tokenizer_path, ModelConfig::default())
    }

    /// Create a new semantic model with custom configuration
    pub fn with_config(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();

        info!(
            "Loading semantic model from {} with tokenizer {}",
            model_path.display(),
            tokenizer_path.display()
        );

        // Initialize ONNX Runtime session
        let session = Self::create_session(model_path, config.use_gpu)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::TokenError(e.to_string()))?;

        info!(
            "Model loaded successfully. Embedding dim: {}, Max length: {}",
            config.embedding_dim, config.max_length
        );

        Ok(Self {
            session: Arc::new(parking_lot::Mutex::new(session)),
            tokenizer,
            config,
            preprocessor: TextPreprocessor::default(),
            stats: Arc::new(RwLock::new(ModelStats::default())),
        })
    }

    /// Create ONNX Runtime session with appropriate configuration
    fn create_session(model_path: &Path, use_gpu: bool) -> Result<Session> {
        let session_builder = Session::builder()
            .map_err(|e| ModelError::OnnxError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ModelError::OnnxError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| ModelError::OnnxError(e.to_string()))?;

        // Enable CUDA if requested and available
        #[cfg(feature = "cuda")]
        let session_builder = if use_gpu {
            session_builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .unwrap_or_else(|e| {
                    warn!("Failed to enable CUDA, falling back to CPU: {}", e);
                    Session::builder()
                        .unwrap()
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .unwrap()
                        .with_intra_threads(4)
                        .unwrap()
                })
        } else {
            session_builder
        };

        session_builder
            .commit_from_file(model_path)
            .map_err(|e| ModelError::OnnxError(e.to_string()))
    }

    /// Encode a single text into an embedding vector
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    ///
    /// # Returns
    ///
    /// A vector of f32 values representing the embedding (length = embedding_dim)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use semantic_model::SemanticModel;
    /// # let model = SemanticModel::new("model.onnx", "tokenizer.json").unwrap();
    /// let embedding = model.encode("Hello, world!").unwrap();
    /// assert_eq!(embedding.len(), 384);
    /// ```
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Preprocess text
        let processed = self.preprocessor.preprocess(text);

        // Tokenize
        let (input_ids, attention_mask) = self.tokenize(&processed)?;

        // Run inference
        let embeddings = self.run_inference(&[input_ids], &[attention_mask])?;

        // Extract first (and only) embedding
        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| ModelError::InferenceError("No embeddings returned".to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_encodings += 1;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            stats.avg_encoding_time_ms =
                (stats.avg_encoding_time_ms * (stats.total_encodings - 1) as f64 + elapsed)
                / stats.total_encodings as f64;
        }

        Ok(embedding)
    }

    /// Encode multiple texts into embedding vectors
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    ///
    /// A vector of embedding vectors
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use semantic_model::SemanticModel;
    /// # let model = SemanticModel::new("model.onnx", "tokenizer.json").unwrap();
    /// let texts = vec!["Hello".to_string(), "World".to_string()];
    /// let embeddings = model.encode_batch(&texts).unwrap();
    /// assert_eq!(embeddings.len(), 2);
    /// ```
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();

        // Preprocess all texts
        let processed: Vec<_> = texts
            .iter()
            .map(|t| self.preprocessor.preprocess(t))
            .collect();

        // Tokenize all texts
        let mut all_input_ids = Vec::with_capacity(texts.len());
        let mut all_attention_masks = Vec::with_capacity(texts.len());

        for text in &processed {
            let (input_ids, attention_mask) = self.tokenize(text)?;
            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
        }

        // Run inference on batch
        let embeddings = self.run_inference(&all_input_ids, &all_attention_masks)?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_encodings += texts.len() as u64;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            stats.avg_encoding_time_ms =
                (stats.avg_encoding_time_ms * (stats.total_encodings - texts.len() as u64) as f64
                + elapsed) / stats.total_encodings as f64;
        }

        Ok(embeddings)
    }

    /// Tokenize input text into input IDs and attention mask
    fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ModelError::TokenError(e.to_string()))?;

        let mut input_ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();

        // Pad or truncate to max_length
        let current_len = input_ids.len();
        if current_len < self.config.max_length {
            // Pad
            input_ids.resize(self.config.max_length, 0);
            attention_mask.resize(self.config.max_length, 0);
        } else if current_len > self.config.max_length {
            // Truncate
            input_ids.truncate(self.config.max_length);
            attention_mask.truncate(self.config.max_length);
        }

        // Update token stats
        {
            let mut stats = self.stats.write();
            stats.total_tokens_processed += current_len.min(self.config.max_length) as u64;
        }

        Ok((input_ids, attention_mask))
    }

    /// Run ONNX inference on tokenized inputs
    fn run_inference(
        &self,
        input_ids_batch: &[Vec<i64>],
        attention_mask_batch: &[Vec<i64>],
    ) -> Result<Vec<Vec<f32>>> {
        if input_ids_batch.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = input_ids_batch.len();
        let seq_len = self.config.max_length;

        // Flatten batch into single arrays
        let input_ids_flat: Vec<i64> = input_ids_batch.iter().flatten().copied().collect();
        let attention_mask_flat: Vec<i64> = attention_mask_batch.iter().flatten().copied().collect();

        // Create ONNX input tensors
        let input_ids_array = ndarray::Array2::from_shape_vec(
            (batch_size, seq_len),
            input_ids_flat,
        )
        .map_err(|e| ModelError::InferenceError(e.to_string()))?;

        let attention_mask_array = ndarray::Array2::from_shape_vec(
            (batch_size, seq_len),
            attention_mask_flat,
        )
        .map_err(|e| ModelError::InferenceError(e.to_string()))?;

        // Run inference
        // Convert ndarray to (shape, data) tuple format expected by ort
        let input_ids_shape = [batch_size, seq_len];
        let input_ids_data = input_ids_array.into_raw_vec();
        let input_ids_value = Value::from_array((input_ids_shape, input_ids_data))
            .map_err(|e| ModelError::OnnxError(e.to_string()))?;

        let attention_mask_shape = [batch_size, seq_len];
        let attention_mask_data = attention_mask_array.into_raw_vec();
        let attention_mask_value = Value::from_array((attention_mask_shape, attention_mask_data))
            .map_err(|e| ModelError::OnnxError(e.to_string()))?;

        use std::borrow::Cow;

        let inputs_vec = vec![
            (Cow::from("input_ids"), input_ids_value.into_dyn()),
            (Cow::from("attention_mask"), attention_mask_value.into_dyn()),
        ];

        // Run inference and extract data while holding the lock
        let (output_shape, output_data_vec) = {
            let mut session = self.session.lock();
            let outputs = session.run(inputs_vec)
                .map_err(|e| ModelError::OnnxError(e.to_string()))?;

            // Extract embeddings from output
            let output_tensor = outputs
                .get("last_hidden_state")
                .ok_or_else(|| ModelError::InferenceError("No output tensor found".to_string()))?;

            let (shape, data) = output_tensor
                .try_extract_tensor::<f32>()
                .map_err(|e| ModelError::OnnxError(e.to_string()))?;

            // Copy data out before lock is dropped
            (shape.to_vec(), data.to_vec())
        };

        debug!("Output shape: {:?}", output_shape);
        let output_data = &output_data_vec;

        // Apply pooling strategy
        let mut embeddings = Vec::with_capacity(batch_size);
        let embedding_dim = self.config.embedding_dim;

        for i in 0..batch_size {
            // Extract token embeddings for this batch item
            let start_idx = i * seq_len * embedding_dim;
            let end_idx = start_idx + (seq_len * embedding_dim);
            let token_embeddings = &output_data[start_idx..end_idx];
            let attention_mask = &attention_mask_batch[i];

            let pooled = self.pool_embeddings(
                token_embeddings,
                attention_mask,
                seq_len,
            )?;

            embeddings.push(pooled);
        }

        Ok(embeddings)
    }

    /// Apply pooling strategy to token embeddings
    fn pool_embeddings(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let embedding_dim = self.config.embedding_dim;

        let pooled = match self.config.pooling_strategy {
            PoolingStrategy::Mean => {
                self.mean_pooling(token_embeddings, attention_mask, seq_len, embedding_dim)
            }
            PoolingStrategy::Max => {
                self.max_pooling(token_embeddings, seq_len, embedding_dim)
            }
            PoolingStrategy::Cls => {
                self.cls_pooling(token_embeddings, embedding_dim)
            }
        }?;

        // Normalize if configured
        if self.config.normalize {
            Ok(normalize_vector(&pooled))
        } else {
            Ok(pooled)
        }
    }

    /// Mean pooling over token embeddings
    fn mean_pooling(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        seq_len: usize,
        embedding_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut pooled = vec![0.0; embedding_dim];
        let mut mask_sum = 0.0;

        for i in 0..seq_len {
            if attention_mask[i] == 1 {
                let start_idx = i * embedding_dim;
                let end_idx = start_idx + embedding_dim;

                if end_idx <= token_embeddings.len() {
                    for (j, value) in token_embeddings[start_idx..end_idx].iter().enumerate() {
                        pooled[j] += value;
                    }
                    mask_sum += 1.0;
                }
            }
        }

        if mask_sum > 0.0 {
            for value in &mut pooled {
                *value /= mask_sum;
            }
        }

        Ok(pooled)
    }

    /// Max pooling over token embeddings
    fn max_pooling(
        &self,
        token_embeddings: &[f32],
        seq_len: usize,
        embedding_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut pooled = vec![f32::NEG_INFINITY; embedding_dim];

        for i in 0..seq_len {
            let start_idx = i * embedding_dim;
            let end_idx = start_idx + embedding_dim;

            if end_idx <= token_embeddings.len() {
                for (j, value) in token_embeddings[start_idx..end_idx].iter().enumerate() {
                    pooled[j] = pooled[j].max(*value);
                }
            }
        }

        Ok(pooled)
    }

    /// CLS token pooling (use first token embedding)
    fn cls_pooling(&self, token_embeddings: &[f32], embedding_dim: usize) -> Result<Vec<f32>> {
        if token_embeddings.len() < embedding_dim {
            return Err(ModelError::DimensionError {
                expected: embedding_dim,
                actual: token_embeddings.len(),
            });
        }

        Ok(token_embeddings[..embedding_dim].to_vec())
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get model statistics
    pub fn stats(&self) -> ModelStats {
        self.stats.read().clone()
    }

    /// Reset model statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = ModelStats::default();
    }
}

/// Normalize a vector to unit length
fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

/// Calculate cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vector() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize_vector(&vec);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&c, &d).abs() < 1e-6);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.max_length, 512);
        assert_eq!(config.embedding_dim, 384);
        assert!(config.normalize);
    }
}
