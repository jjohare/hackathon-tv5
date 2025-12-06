/// GPU-optimized memory layouts and batch processing types
///
/// Provides GPU-compatible data structures for:
/// - Aligned memory layouts (CUDA requirements)
/// - Batch processing (maximize GPU utilization)
/// - Zero-copy transfers (host ↔ device)
/// - SIMD-friendly representations

use serde::{Deserialize, Serialize};

/// GPU-aligned embedding vector (32-byte aligned for optimal GPU access)
#[repr(C, align(32))]
#[derive(Debug, Clone)]
pub struct GPUEmbedding {
    /// Embedding dimensions
    pub dims: u32,

    /// Padding for alignment
    _padding: u32,

    /// Embedding data (must be aligned)
    pub data: Vec<f32>,
}

impl GPUEmbedding {
    /// Create new GPU-aligned embedding
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            dims: data.len() as u32,
            _padding: 0,
            data,
        }
    }

    /// Get data pointer for GPU transfer
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Get mutable data pointer for GPU transfer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    /// Check if properly aligned for GPU
    pub fn is_aligned(&self) -> bool {
        (self.data.as_ptr() as usize) % 32 == 0
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// GPU batch for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUBatch<T> {
    /// Batch items
    pub items: Vec<T>,

    /// Batch size
    pub size: usize,

    /// Max batch capacity (GPU memory limit)
    pub capacity: usize,

    /// Batch ID
    pub batch_id: u64,
}

impl<T> GPUBatch<T> {
    /// Create new batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            size: 0,
            capacity,
            batch_id: 0,
        }
    }

    /// Add item to batch
    pub fn push(&mut self, item: T) -> Result<(), String> {
        if self.size >= self.capacity {
            return Err("Batch full".to_string());
        }

        self.items.push(item);
        self.size += 1;
        Ok(())
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.size >= self.capacity
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear batch
    pub fn clear(&mut self) {
        self.items.clear();
        self.size = 0;
    }

    /// Get fill rate (0.0-1.0)
    pub fn fill_rate(&self) -> f32 {
        self.size as f32 / self.capacity as f32
    }
}

/// GPU memory layout for embeddings batch
#[repr(C)]
#[derive(Debug, Clone)]
pub struct GPUEmbeddingBatch {
    /// Number of embeddings in batch
    pub count: u32,

    /// Embedding dimensionality
    pub dims: u32,

    /// Flattened embedding data [count × dims]
    pub data: Vec<f32>,
}

impl GPUEmbeddingBatch {
    /// Create batch from embeddings
    pub fn from_embeddings(embeddings: Vec<Vec<f32>>) -> Self {
        let count = embeddings.len() as u32;
        let dims = embeddings.first().map(|e| e.len()).unwrap_or(0) as u32;

        // Flatten into contiguous memory
        let mut data = Vec::with_capacity((count * dims) as usize);
        for embedding in embeddings {
            data.extend(embedding);
        }

        Self { count, dims, data }
    }

    /// Get embedding at index
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        let start = index * self.dims as usize;
        let end = start + self.dims as usize;

        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Check if batch fits in GPU memory
    pub fn fits_in_gpu(&self, gpu_memory_bytes: usize) -> bool {
        self.size_bytes() <= gpu_memory_bytes
    }
}

/// GPU memory layout for semantic forces kernel
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SemanticForcesInput {
    /// Number of items
    pub n: u32,

    /// Embedding dimensions
    pub dims: u32,

    /// Content embeddings [n × dims]
    pub embeddings: Vec<f32>,

    /// Color palette vectors [n × 64]
    pub color_vectors: Vec<f32>,

    /// Output force matrix [n × n]
    pub forces: Vec<f32>,
}

impl SemanticForcesInput {
    /// Create input for semantic forces kernel
    pub fn new(
        embeddings: Vec<Vec<f32>>,
        color_vectors: Vec<Vec<f32>>,
    ) -> Self {
        let n = embeddings.len() as u32;
        let dims = embeddings.first().map(|e| e.len()).unwrap_or(1024) as u32;

        // Flatten embeddings
        let embeddings_flat = embeddings.into_iter().flatten().collect();

        // Flatten color vectors
        let color_flat = color_vectors.into_iter().flatten().collect();

        // Allocate output matrix
        let forces = vec![0.0; (n * n) as usize];

        Self {
            n,
            dims,
            embeddings: embeddings_flat,
            color_vectors: color_flat,
            forces,
        }
    }

    /// Get force between items i and j
    pub fn get_force(&self, i: usize, j: usize) -> f32 {
        let idx = i * self.n as usize + j;
        self.forces.get(idx).copied().unwrap_or(0.0)
    }

    /// Memory requirements (bytes)
    pub fn memory_required(&self) -> usize {
        let embeddings_size = self.n as usize * self.dims as usize * 4;
        let colors_size = self.n as usize * 64 * 4;
        let forces_size = self.n as usize * self.n as usize * 4;

        embeddings_size + colors_size + forces_size
    }
}

/// OWL constraint type IDs for GPU processing
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintType {
    DisjointWith = 1,
    SubClassOf = 2,
    EquivalentClass = 3,
    InverseOf = 4,
    FunctionalProperty = 5,
    InverseFunctionalProperty = 6,
    TransitiveProperty = 7,
    SymmetricProperty = 8,
    AsymmetricProperty = 9,
    ReflexiveProperty = 10,
    IrreflexiveProperty = 11,
    ObjectPropertyDomain = 12,
    ObjectPropertyRange = 13,
    DataPropertyDomain = 14,
    DataPropertyRange = 15,
    AllValuesFrom = 16,
    SomeValuesFrom = 17,
    HasValue = 18,
    MinCardinality = 19,
    MaxCardinality = 20,
    ExactCardinality = 21,
}

impl ConstraintType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::DisjointWith),
            2 => Some(Self::SubClassOf),
            3 => Some(Self::EquivalentClass),
            4 => Some(Self::InverseOf),
            5 => Some(Self::FunctionalProperty),
            6 => Some(Self::InverseFunctionalProperty),
            7 => Some(Self::TransitiveProperty),
            8 => Some(Self::SymmetricProperty),
            9 => Some(Self::AsymmetricProperty),
            10 => Some(Self::ReflexiveProperty),
            11 => Some(Self::IrreflexiveProperty),
            12 => Some(Self::ObjectPropertyDomain),
            13 => Some(Self::ObjectPropertyRange),
            14 => Some(Self::DataPropertyDomain),
            15 => Some(Self::DataPropertyRange),
            16 => Some(Self::AllValuesFrom),
            17 => Some(Self::SomeValuesFrom),
            18 => Some(Self::HasValue),
            19 => Some(Self::MinCardinality),
            20 => Some(Self::MaxCardinality),
            21 => Some(Self::ExactCardinality),
            _ => None,
        }
    }

    pub fn to_u32(self) -> u32 {
        self as u32
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::DisjointWith => "DisjointWith",
            Self::SubClassOf => "SubClassOf",
            Self::EquivalentClass => "EquivalentClass",
            Self::InverseOf => "InverseOf",
            Self::FunctionalProperty => "FunctionalProperty",
            Self::InverseFunctionalProperty => "InverseFunctionalProperty",
            Self::TransitiveProperty => "TransitiveProperty",
            Self::SymmetricProperty => "SymmetricProperty",
            Self::AsymmetricProperty => "AsymmetricProperty",
            Self::ReflexiveProperty => "ReflexiveProperty",
            Self::IrreflexiveProperty => "IrreflexiveProperty",
            Self::ObjectPropertyDomain => "ObjectPropertyDomain",
            Self::ObjectPropertyRange => "ObjectPropertyRange",
            Self::DataPropertyDomain => "DataPropertyDomain",
            Self::DataPropertyRange => "DataPropertyRange",
            Self::AllValuesFrom => "AllValuesFrom",
            Self::SomeValuesFrom => "SomeValuesFrom",
            Self::HasValue => "HasValue",
            Self::MinCardinality => "MinCardinality",
            Self::MaxCardinality => "MaxCardinality",
            Self::ExactCardinality => "ExactCardinality",
        }
    }
}

/// GPU constraint representation as flat integer array
/// Format: [subject_id, predicate_id, object_id, constraint_type, ...]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConstraintGraph {
    /// Flat array of constraint quads: [subj, pred, obj, type, subj, pred, obj, type, ...]
    pub constraint_data: Vec<u32>,
    /// Constraint weights/strengths [0.0, 1.0]
    pub constraint_weights: Vec<f32>,
    /// Number of constraints
    pub constraint_count: usize,
    /// Elements per constraint (4: subject, predicate, object, type)
    pub elements_per_constraint: usize,
}

impl GpuConstraintGraph {
    pub fn new() -> Self {
        Self {
            constraint_data: Vec::new(),
            constraint_weights: Vec::new(),
            constraint_count: 0,
            elements_per_constraint: 4,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            constraint_data: Vec::with_capacity(capacity * 4),
            constraint_weights: Vec::with_capacity(capacity),
            constraint_count: 0,
            elements_per_constraint: 4,
        }
    }

    pub fn add_constraint(
        &mut self,
        subject_id: u32,
        predicate_id: u32,
        object_id: u32,
        constraint_type: ConstraintType,
        weight: f32,
    ) {
        self.constraint_data.push(subject_id);
        self.constraint_data.push(predicate_id);
        self.constraint_data.push(object_id);
        self.constraint_data.push(constraint_type.to_u32());
        self.constraint_weights.push(weight.clamp(0.0, 1.0));
        self.constraint_count += 1;
    }

    pub fn size_bytes(&self) -> usize {
        self.constraint_data.len() * std::mem::size_of::<u32>()
            + self.constraint_weights.len() * std::mem::size_of::<f32>()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.constraint_data.len() != self.constraint_count * self.elements_per_constraint {
            return Err(format!(
                "Invalid constraint data length: expected {}, got {}",
                self.constraint_count * self.elements_per_constraint,
                self.constraint_data.len()
            ));
        }

        if self.constraint_weights.len() != self.constraint_count {
            return Err(format!(
                "Invalid weights length: expected {}, got {}",
                self.constraint_count,
                self.constraint_weights.len()
            ));
        }

        for (i, &weight) in self.constraint_weights.iter().enumerate() {
            if !(0.0..=1.0).contains(&weight) {
                return Err(format!("Weight at index {} out of range: {}", i, weight));
            }
        }

        Ok(())
    }
}

impl Default for GpuConstraintGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU violation result from kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuViolationResult {
    /// Constraint indices that were violated
    pub violated_constraint_indices: Vec<u32>,
    /// Severity scores for each violation [0.0, 1.0]
    pub violation_severities: Vec<f32>,
    /// Total number of violations detected
    pub total_violations: usize,
}

impl GpuViolationResult {
    pub fn new() -> Self {
        Self {
            violated_constraint_indices: Vec::new(),
            violation_severities: Vec::new(),
            total_violations: 0,
        }
    }

    pub fn add_violation(&mut self, constraint_index: u32, severity: f32) {
        self.violated_constraint_indices.push(constraint_index);
        self.violation_severities.push(severity.clamp(0.0, 1.0));
        self.total_violations += 1;
    }

    pub fn merge(&mut self, other: GpuViolationResult) {
        self.violated_constraint_indices
            .extend(other.violated_constraint_indices);
        self.violation_severities
            .extend(other.violation_severities);
        self.total_violations += other.total_violations;
    }
}

impl Default for GpuViolationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Entity ID mapping for GPU serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityIdMap {
    /// Map from entity IRI to GPU ID
    pub iri_to_id: std::collections::HashMap<String, u32>,
    /// Map from GPU ID to entity IRI
    pub id_to_iri: std::collections::HashMap<u32, String>,
    /// Next available ID
    pub next_id: u32,
}

impl EntityIdMap {
    pub fn new() -> Self {
        Self {
            iri_to_id: std::collections::HashMap::new(),
            id_to_iri: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    pub fn get_or_create_id(&mut self, iri: &str) -> u32 {
        if let Some(&id) = self.iri_to_id.get(iri) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.iri_to_id.insert(iri.to_string(), id);
            self.id_to_iri.insert(id, iri.to_string());
            id
        }
    }

    pub fn get_id(&self, iri: &str) -> Option<u32> {
        self.iri_to_id.get(iri).copied()
    }

    pub fn get_iri(&self, id: u32) -> Option<&str> {
        self.id_to_iri.get(&id).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.iri_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.iri_to_id.is_empty()
    }
}

impl Default for EntityIdMap {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU memory layout for ontology constraints kernel (legacy compatibility)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct OntologyConstraintsInput {
    /// Number of items
    pub n: u32,

    /// Number of constraints
    pub m: u32,

    /// Embedding dimensions
    pub dims: u32,

    /// Padding for alignment
    _padding: u32,

    /// Fused embeddings [n × dims]
    pub embeddings: Vec<f32>,

    /// Constraint graph [m × 4] (subject, predicate, object, type)
    pub constraint_graph: Vec<u32>,

    /// Constraint weights [m]
    pub constraint_weights: Vec<f32>,
}

impl OntologyConstraintsInput {
    /// Create input for ontology constraints kernel (legacy format)
    pub fn new(
        embeddings: Vec<Vec<f32>>,
        constraints: Vec<(i32, i32, i32)>,
        weights: Vec<f32>,
    ) -> Self {
        let n = embeddings.len() as u32;
        let m = constraints.len() as u32;
        let dims = embeddings.first().map(|e| e.len()).unwrap_or(1024) as u32;

        let embeddings_flat = embeddings.into_iter().flatten().collect();

        let mut constraint_graph = Vec::with_capacity((m * 4) as usize);
        for (s, p, o) in constraints {
            constraint_graph.push(s as u32);
            constraint_graph.push(p as u32);
            constraint_graph.push(o as u32);
            constraint_graph.push(ConstraintType::SubClassOf.to_u32());
        }

        Self {
            n,
            m,
            dims,
            _padding: 0,
            embeddings: embeddings_flat,
            constraint_graph,
            constraint_weights: weights,
        }
    }

    /// Create from GpuConstraintGraph
    pub fn from_gpu_graph(
        embeddings: Vec<Vec<f32>>,
        graph: &GpuConstraintGraph,
    ) -> Self {
        let n = embeddings.len() as u32;
        let m = graph.constraint_count as u32;
        let dims = embeddings.first().map(|e| e.len()).unwrap_or(1024) as u32;

        let embeddings_flat = embeddings.into_iter().flatten().collect();

        Self {
            n,
            m,
            dims,
            _padding: 0,
            embeddings: embeddings_flat,
            constraint_graph: graph.constraint_data.clone(),
            constraint_weights: graph.constraint_weights.clone(),
        }
    }

    /// Memory requirements (bytes)
    pub fn memory_required(&self) -> usize {
        let embeddings_size = self.n as usize * self.dims as usize * 4;
        let constraints_size = self.m as usize * 4 * 4;
        let weights_size = self.m as usize * 4;

        embeddings_size + constraints_size + weights_size
    }
}

/// GPU memory layout for APSP landmark kernel
#[repr(C)]
#[derive(Debug, Clone)]
pub struct APSPLandmarkInput {
    /// Number of nodes
    pub n: u32,

    /// Number of edges
    pub e: u32,

    /// Number of landmarks
    pub l: u32,

    /// Padding for alignment
    _padding: u32,

    /// Edge list [e × 2] (source, target)
    pub edges: Vec<i32>,

    /// Edge weights [e]
    pub weights: Vec<f32>,

    /// Landmark node IDs [l]
    pub landmarks: Vec<i32>,

    /// Output distances [n × l]
    pub distances: Vec<f32>,
}

impl APSPLandmarkInput {
    /// Create input for APSP landmark kernel
    pub fn new(
        n: u32,
        edges: Vec<(i32, i32, f32)>,
        landmarks: Vec<i32>,
    ) -> Self {
        let e = edges.len() as u32;
        let l = landmarks.len() as u32;

        let mut edge_list = Vec::with_capacity((e * 2) as usize);
        let mut edge_weights = Vec::with_capacity(e as usize);

        for (src, dst, weight) in edges {
            edge_list.push(src);
            edge_list.push(dst);
            edge_weights.push(weight);
        }

        let distances = vec![f32::INFINITY; (n * l) as usize];

        Self {
            n,
            e,
            l,
            _padding: 0,
            edges: edge_list,
            weights: edge_weights,
            landmarks,
            distances,
        }
    }

    /// Get distance from node to landmark
    pub fn get_distance(&self, node: usize, landmark: usize) -> f32 {
        let idx = node * self.l as usize + landmark;
        self.distances.get(idx).copied().unwrap_or(f32::INFINITY)
    }

    /// Memory requirements (bytes)
    pub fn memory_required(&self) -> usize {
        let edges_size = self.e as usize * 2 * 4;
        let weights_size = self.e as usize * 4;
        let landmarks_size = self.l as usize * 4;
        let distances_size = self.n as usize * self.l as usize * 4;

        edges_size + weights_size + landmarks_size + distances_size
    }
}

/// GPU memory layout for tensor fusion kernel
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TensorFusionInput {
    /// Batch size
    pub batch_size: u32,

    /// Visual embedding dims (768)
    pub visual_dims: u32,

    /// Audio embedding dims (512)
    pub audio_dims: u32,

    /// Text embedding dims (1024)
    pub text_dims: u32,

    /// Output dims (1024)
    pub output_dims: u32,

    /// Padding for alignment
    _padding: [u32; 3],

    /// Visual embeddings [batch_size × visual_dims]
    pub visual: Vec<f32>,

    /// Audio embeddings [batch_size × audio_dims]
    pub audio: Vec<f32>,

    /// Text embeddings [batch_size × text_dims]
    pub text: Vec<f32>,

    /// Fusion weights [3]
    pub weights: [f32; 3],

    /// Output unified embeddings [batch_size × output_dims]
    pub output: Vec<f32>,
}

impl TensorFusionInput {
    /// Create input for tensor fusion kernel
    pub fn new(
        visual: Vec<Vec<f32>>,
        audio: Vec<Vec<f32>>,
        text: Vec<Vec<f32>>,
        weights: [f32; 3],
    ) -> Self {
        let batch_size = visual.len() as u32;
        let output_dims = 1024;

        let visual_flat = visual.into_iter().flatten().collect();
        let audio_flat = audio.into_iter().flatten().collect();
        let text_flat = text.into_iter().flatten().collect();

        let output = vec![0.0; (batch_size * output_dims) as usize];

        Self {
            batch_size,
            visual_dims: 768,
            audio_dims: 512,
            text_dims: 1024,
            output_dims,
            _padding: [0; 3],
            visual: visual_flat,
            audio: audio_flat,
            text: text_flat,
            weights,
            output,
        }
    }

    /// Get unified embedding at index
    pub fn get_output(&self, index: usize) -> Option<&[f32]> {
        let start = index * self.output_dims as usize;
        let end = start + self.output_dims as usize;

        if end <= self.output.len() {
            Some(&self.output[start..end])
        } else {
            None
        }
    }

    /// Memory requirements (bytes)
    pub fn memory_required(&self) -> usize {
        let visual_size = self.batch_size as usize * self.visual_dims as usize * 4;
        let audio_size = self.batch_size as usize * self.audio_dims as usize * 4;
        let text_size = self.batch_size as usize * self.text_dims as usize * 4;
        let output_size = self.batch_size as usize * self.output_dims as usize * 4;

        visual_size + audio_size + text_size + output_size
    }
}

/// GPU batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUBatchStats {
    /// Total batches processed
    pub total_batches: u64,

    /// Total items processed
    pub total_items: u64,

    /// Average batch size
    pub avg_batch_size: f32,

    /// Average processing time per batch (ms)
    pub avg_batch_time_ms: f32,

    /// GPU utilization (0.0-1.0)
    pub gpu_utilization: f32,

    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,

    /// Throughput (items/sec)
    pub throughput: f32,
}

impl GPUBatchStats {
    /// Create new stats
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_items: 0,
            avg_batch_size: 0.0,
            avg_batch_time_ms: 0.0,
            gpu_utilization: 0.0,
            memory_usage_bytes: 0,
            throughput: 0.0,
        }
    }

    /// Update stats with new batch
    pub fn update(&mut self, batch_size: usize, processing_time_ms: f32) {
        self.total_batches += 1;
        self.total_items += batch_size as u64;

        // Update moving average
        let alpha = 0.1;
        self.avg_batch_size = (1.0 - alpha) * self.avg_batch_size + alpha * batch_size as f32;
        self.avg_batch_time_ms =
            (1.0 - alpha) * self.avg_batch_time_ms + alpha * processing_time_ms;

        // Calculate throughput
        if self.avg_batch_time_ms > 0.0 {
            self.throughput = (self.avg_batch_size * 1000.0) / self.avg_batch_time_ms;
        }
    }
}

impl Default for GPUBatchStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_embedding_alignment() {
        let data = vec![1.0; 1024];
        let embedding = GPUEmbedding::new(data);
        assert_eq!(embedding.dims, 1024);
    }

    #[test]
    fn test_gpu_batch() {
        let mut batch = GPUBatch::<i32>::with_capacity(10);
        assert_eq!(batch.capacity, 10);

        batch.push(1).unwrap();
        batch.push(2).unwrap();
        assert_eq!(batch.size, 2);
        assert_eq!(batch.fill_rate(), 0.2);
    }

    #[test]
    fn test_embedding_batch() {
        let embeddings = vec![
            vec![1.0; 1024],
            vec![2.0; 1024],
            vec![3.0; 1024],
        ];

        let batch = GPUEmbeddingBatch::from_embeddings(embeddings);
        assert_eq!(batch.count, 3);
        assert_eq!(batch.dims, 1024);

        let first = batch.get(0).unwrap();
        assert_eq!(first.len(), 1024);
        assert_eq!(first[0], 1.0);
    }

    #[test]
    fn test_batch_stats() {
        let mut stats = GPUBatchStats::new();
        stats.update(32, 10.0);
        stats.update(32, 12.0);

        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_items, 64);
        assert!(stats.throughput > 0.0);
    }
}
