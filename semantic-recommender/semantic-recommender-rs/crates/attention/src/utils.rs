//! Utility functions for attention operations

use tch::{Device, Tensor, Kind};

/// Check if CUDA is available and return appropriate device
pub fn get_device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

/// Normalize embeddings to unit length
pub fn normalize_embeddings(embeddings: &Tensor) -> Tensor {
    let norms = embeddings.norm_scalaropt_dim(2.0, &[-1], true);
    embeddings / (norms + 1e-8)
}

/// Compute cosine similarity between query and candidates
pub fn cosine_similarity(query: &Tensor, candidates: &Tensor) -> Tensor {
    let query_norm = normalize_embeddings(query);
    let candidates_norm = normalize_embeddings(candidates);

    // Handle both single and batched queries
    if query_norm.dim() == 1 {
        // [embed_dim] @ [num_candidates, embed_dim]^T
        candidates_norm.matmul(&query_norm.unsqueeze(1)).squeeze_dim(1)
    } else {
        // [batch, embed_dim] @ [embed_dim, num_candidates]
        query_norm.matmul(&candidates_norm.transpose(0, 1))
    }
}

/// Convert vector of floats to tensor
pub fn vec_to_tensor(vec: &[f32], device: Device) -> Tensor {
    Tensor::from_slice(vec)
        .to_device(device)
        .to_kind(Kind::Float)
}

/// Convert 2D vector to tensor
pub fn vec2d_to_tensor(vecs: &[Vec<f32>], device: Device) -> Tensor {
    if vecs.is_empty() {
        return Tensor::zeros(&[0, 0], (Kind::Float, device));
    }

    let num_rows = vecs.len() as i64;
    let num_cols = vecs[0].len() as i64;

    let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter()).copied().collect();

    Tensor::from_slice(&flat)
        .view([num_rows, num_cols])
        .to_device(device)
        .to_kind(Kind::Float)
}

/// Apply temperature scaling to logits
pub fn temperature_scale(logits: &Tensor, temperature: f64) -> Tensor {
    logits / temperature
}

/// Top-k sampling from scores
pub fn top_k_indices(scores: &Tensor, k: i64) -> Vec<i64> {
    let (_, indices) = scores.topk(k, -1, true, true);
    let indices_vec: Vec<i64> = indices.try_into().unwrap_or_default();
    indices_vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_get_device() {
        let device = get_device();
        // Should return either CPU or CUDA(0)
        assert!(matches!(device, Device::Cpu | Device::Cuda(_)));
    }

    #[test]
    fn test_normalize_embeddings() {
        let device = Device::Cpu;
        let embeddings = Tensor::from_slice(&[3.0_f32, 4.0])
            .to_device(device);

        let normalized = normalize_embeddings(&embeddings);
        let norm: f32 = normalized.norm_scalaropt_dim(2.0, &[-1], false)
            .try_into()
            .unwrap();

        assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_similarity_single() {
        let device = Device::Cpu;

        // Query: [1, 0, 0]
        let query = Tensor::from_slice(&[1.0_f32, 0.0, 0.0])
            .to_device(device);

        // Candidates: [[1,0,0], [0,1,0], [1,1,0]]
        let candidates = Tensor::from_slice(&[
            1.0_f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
        ])
        .view([3, 3])
        .to_device(device);

        let sim = cosine_similarity(&query, &candidates);
        let sim_vec: Vec<f32> = sim.try_into().unwrap();

        // Should be [1.0, 0.0, ~0.707]
        assert_relative_eq!(sim_vec[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sim_vec[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(sim_vec[2], 0.707, epsilon = 1e-2);
    }

    #[test]
    fn test_vec_to_tensor() {
        let vec = vec![1.0_f32, 2.0, 3.0, 4.0];
        let tensor = vec_to_tensor(&vec, Device::Cpu);

        assert_eq!(tensor.size(), vec![4]);

        let result: Vec<f32> = tensor.try_into().unwrap();
        assert_eq!(result, vec);
    }

    #[test]
    fn test_vec2d_to_tensor() {
        let vecs = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let tensor = vec2d_to_tensor(&vecs, Device::Cpu);
        assert_eq!(tensor.size(), vec![2, 3]);

        let flat: Vec<f32> = tensor.flatten(0, 1).try_into().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vec2d_to_tensor_empty() {
        let vecs: Vec<Vec<f32>> = vec![];
        let tensor = vec2d_to_tensor(&vecs, Device::Cpu);
        assert_eq!(tensor.size(), vec![0, 0]);
    }

    #[test]
    fn test_temperature_scale() {
        let logits = Tensor::from_slice(&[1.0_f32, 2.0, 3.0])
            .to_device(Device::Cpu);

        let scaled = temperature_scale(&logits, 2.0);
        let result: Vec<f32> = scaled.try_into().unwrap();

        assert_relative_eq!(result[0], 0.5, epsilon = 1e-5);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[2], 1.5, epsilon = 1e-5);
    }

    #[test]
    fn test_top_k_indices() {
        let scores = Tensor::from_slice(&[0.1_f32, 0.5, 0.3, 0.9, 0.2])
            .to_device(Device::Cpu);

        let top_3 = top_k_indices(&scores, 3);

        // Should return indices [3, 1, 2] (sorted by score)
        assert_eq!(top_3.len(), 3);
        assert_eq!(top_3[0], 3); // 0.9
        assert_eq!(top_3[1], 1); // 0.5
        assert_eq!(top_3[2], 2); // 0.3
    }
}
