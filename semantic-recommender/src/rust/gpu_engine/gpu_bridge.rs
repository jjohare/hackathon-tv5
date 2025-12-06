/// GPU Bridge: OWL Axiom Serialization and GPU Transfer
///
/// Converts OWL ontology constraints to GPU-compatible integer arrays
/// and manages async data transfer between CPU and GPU memory.

use crate::models::gpu_types::{
    ConstraintType, EntityIdMap, GpuConstraintGraph, GpuViolationResult,
};
use std::collections::HashMap;
use std::time::Instant;

/// Error types for GPU bridge operations
#[derive(Debug, Clone)]
pub enum GpuBridgeError {
    SerializationError(String),
    MemoryAllocationError(String),
    TransferError(String),
    ValidationError(String),
    OntologyParseError(String),
}

impl std::fmt::Display for GpuBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::MemoryAllocationError(msg) => write!(f, "Memory allocation error: {}", msg),
            Self::TransferError(msg) => write!(f, "Transfer error: {}", msg),
            Self::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            Self::OntologyParseError(msg) => write!(f, "Ontology parse error: {}", msg),
        }
    }
}

impl std::error::Error for GpuBridgeError {}

/// Represents a parsed OWL axiom ready for serialization
#[derive(Debug, Clone)]
pub struct ParsedAxiom {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub constraint_type: ConstraintType,
    pub weight: f32,
}

/// GPU Bridge for ontology constraint serialization
pub struct GpuBridge {
    /// Entity ID mapping for serialization
    entity_map: EntityIdMap,
    /// Parsed axioms cache
    axioms_cache: Vec<ParsedAxiom>,
    /// Serialization statistics
    stats: SerializationStats,
}

#[derive(Debug, Clone, Default)]
pub struct SerializationStats {
    pub total_axioms: usize,
    pub total_entities: usize,
    pub serialization_time_ms: f64,
    pub memory_bytes: usize,
}

impl GpuBridge {
    pub fn new() -> Self {
        Self {
            entity_map: EntityIdMap::new(),
            axioms_cache: Vec::new(),
            stats: SerializationStats::default(),
        }
    }

    /// Parse OWL axioms from text format
    /// Format: "subject predicate object constraint_type weight"
    pub fn parse_axioms(&mut self, axiom_lines: &[String]) -> Result<(), GpuBridgeError> {
        let start = Instant::now();
        self.axioms_cache.clear();

        for (line_num, line) in axiom_lines.iter().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(GpuBridgeError::OntologyParseError(format!(
                    "Invalid axiom at line {}: expected at least 4 fields",
                    line_num
                )));
            }

            let subject = parts[0].to_string();
            let predicate = parts[1].to_string();
            let object = parts[2].to_string();

            let constraint_type = self.parse_constraint_type(parts[3])?;
            let weight = if parts.len() > 4 {
                parts[4].parse::<f32>().unwrap_or(1.0)
            } else {
                1.0
            };

            self.axioms_cache.push(ParsedAxiom {
                subject,
                predicate,
                object,
                constraint_type,
                weight,
            });
        }

        self.stats.total_axioms = self.axioms_cache.len();
        self.stats.serialization_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }

    /// Parse constraint type from string
    fn parse_constraint_type(&self, type_str: &str) -> Result<ConstraintType, GpuBridgeError> {
        match type_str.to_lowercase().as_str() {
            "disjointwith" | "disjoint" => Ok(ConstraintType::DisjointWith),
            "subclassof" | "subclass" => Ok(ConstraintType::SubClassOf),
            "equivalentclass" | "equivalent" => Ok(ConstraintType::EquivalentClass),
            "inverseof" | "inverse" => Ok(ConstraintType::InverseOf),
            "functionalproperty" | "functional" => Ok(ConstraintType::FunctionalProperty),
            "inversefunctionalproperty" => Ok(ConstraintType::InverseFunctionalProperty),
            "transitiveproperty" | "transitive" => Ok(ConstraintType::TransitiveProperty),
            "symmetricproperty" | "symmetric" => Ok(ConstraintType::SymmetricProperty),
            "asymmetricproperty" | "asymmetric" => Ok(ConstraintType::AsymmetricProperty),
            "reflexiveproperty" | "reflexive" => Ok(ConstraintType::ReflexiveProperty),
            "irreflexiveproperty" | "irreflexive" => Ok(ConstraintType::IrreflexiveProperty),
            "objectpropertydomain" => Ok(ConstraintType::ObjectPropertyDomain),
            "objectpropertyrange" => Ok(ConstraintType::ObjectPropertyRange),
            "datapropertydomain" => Ok(ConstraintType::DataPropertyDomain),
            "datapropertyrange" => Ok(ConstraintType::DataPropertyRange),
            "allvaluesfrom" => Ok(ConstraintType::AllValuesFrom),
            "somevaluesfrom" => Ok(ConstraintType::SomeValuesFrom),
            "hasvalue" => Ok(ConstraintType::HasValue),
            "mincardinality" => Ok(ConstraintType::MinCardinality),
            "maxcardinality" => Ok(ConstraintType::MaxCardinality),
            "exactcardinality" => Ok(ConstraintType::ExactCardinality),
            _ => Err(GpuBridgeError::OntologyParseError(format!(
                "Unknown constraint type: {}",
                type_str
            ))),
        }
    }

    /// Serialize axioms to GPU constraint graph
    pub fn serialize_to_gpu(&mut self) -> Result<GpuConstraintGraph, GpuBridgeError> {
        let start = Instant::now();

        let mut graph = GpuConstraintGraph::with_capacity(self.axioms_cache.len());

        for axiom in &self.axioms_cache {
            let subject_id = self.entity_map.get_or_create_id(&axiom.subject);
            let predicate_id = self.entity_map.get_or_create_id(&axiom.predicate);
            let object_id = self.entity_map.get_or_create_id(&axiom.object);

            graph.add_constraint(
                subject_id,
                predicate_id,
                object_id,
                axiom.constraint_type,
                axiom.weight,
            );
        }

        graph
            .validate()
            .map_err(|e| GpuBridgeError::ValidationError(e))?;

        self.stats.total_entities = self.entity_map.len();
        self.stats.memory_bytes = graph.size_bytes();
        self.stats.serialization_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(graph)
    }

    /// Parse and serialize in one step
    pub fn parse_and_serialize(
        &mut self,
        axiom_lines: &[String],
    ) -> Result<GpuConstraintGraph, GpuBridgeError> {
        self.parse_axioms(axiom_lines)?;
        self.serialize_to_gpu()
    }

    /// Get serialization statistics
    pub fn get_stats(&self) -> &SerializationStats {
        &self.stats
    }

    /// Get entity ID map (for result interpretation)
    pub fn get_entity_map(&self) -> &EntityIdMap {
        &self.entity_map
    }

    /// Deserialize violation results back to IRIs
    pub fn deserialize_violations(
        &self,
        violations: &GpuViolationResult,
        graph: &GpuConstraintGraph,
    ) -> Vec<ViolationReport> {
        let mut reports = Vec::new();

        for (i, &constraint_idx) in violations.violated_constraint_indices.iter().enumerate() {
            let offset = (constraint_idx as usize) * 4;
            if offset + 3 < graph.constraint_data.len() {
                let subject_id = graph.constraint_data[offset];
                let predicate_id = graph.constraint_data[offset + 1];
                let object_id = graph.constraint_data[offset + 2];
                let constraint_type_id = graph.constraint_data[offset + 3];

                let subject = self
                    .entity_map
                    .get_iri(subject_id)
                    .unwrap_or("<unknown>")
                    .to_string();
                let predicate = self
                    .entity_map
                    .get_iri(predicate_id)
                    .unwrap_or("<unknown>")
                    .to_string();
                let object = self
                    .entity_map
                    .get_iri(object_id)
                    .unwrap_or("<unknown>")
                    .to_string();
                let constraint_type = ConstraintType::from_u32(constraint_type_id)
                    .unwrap_or(ConstraintType::SubClassOf);
                let severity = violations.violation_severities[i];

                reports.push(ViolationReport {
                    constraint_index: constraint_idx,
                    subject,
                    predicate,
                    object,
                    constraint_type,
                    severity,
                });
            }
        }

        reports
    }
}

impl Default for GpuBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Human-readable violation report
#[derive(Debug, Clone)]
pub struct ViolationReport {
    pub constraint_index: u32,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub constraint_type: ConstraintType,
    pub severity: f32,
}

impl ViolationReport {
    pub fn format(&self) -> String {
        format!(
            "[{}] Severity: {:.2} | {} {} {} ({})",
            self.constraint_index,
            self.severity,
            self.subject,
            self.predicate,
            self.object,
            self.constraint_type.name()
        )
    }
}

/// GPU data transfer manager (placeholder for cudarc integration)
pub struct GpuTransferManager {
    max_gpu_memory: usize,
}

impl GpuTransferManager {
    pub fn new(max_gpu_memory: usize) -> Self {
        Self { max_gpu_memory }
    }

    /// Upload constraint graph to GPU
    pub fn upload_to_gpu(
        &self,
        graph: &GpuConstraintGraph,
    ) -> Result<GpuHandle, GpuBridgeError> {
        let required_memory = graph.size_bytes();

        if required_memory > self.max_gpu_memory {
            return Err(GpuBridgeError::MemoryAllocationError(format!(
                "Required {} bytes exceeds GPU memory limit {} bytes",
                required_memory, self.max_gpu_memory
            )));
        }

        // Placeholder: In real implementation, use cudarc to transfer data
        // Example:
        // let device = CudaDevice::new(0)?;
        // let constraint_data_gpu = device.htod_copy(graph.constraint_data.as_slice())?;
        // let weights_gpu = device.htod_copy(graph.constraint_weights.as_slice())?;

        Ok(GpuHandle {
            constraint_data_size: graph.constraint_data.len(),
            weights_size: graph.constraint_weights.len(),
            memory_bytes: required_memory,
        })
    }

    /// Download violation results from GPU
    pub fn download_results(
        &self,
        handle: &GpuHandle,
        max_violations: usize,
    ) -> Result<GpuViolationResult, GpuBridgeError> {
        // Placeholder: In real implementation, use cudarc to download data
        // Example:
        // let violations_gpu = device.dtoh_sync_copy(&violations_buffer)?;

        Ok(GpuViolationResult::new())
    }

    /// Async upload with stream
    pub fn upload_async(
        &self,
        graph: &GpuConstraintGraph,
        stream_id: u32,
    ) -> Result<GpuHandle, GpuBridgeError> {
        // Placeholder: Real implementation would use CUDA streams
        self.upload_to_gpu(graph)
    }

    /// Async download with stream
    pub fn download_async(
        &self,
        handle: &GpuHandle,
        max_violations: usize,
        stream_id: u32,
    ) -> Result<GpuViolationResult, GpuBridgeError> {
        // Placeholder: Real implementation would use CUDA streams
        self.download_results(handle, max_violations)
    }
}

/// Handle to GPU memory allocation
#[derive(Debug, Clone)]
pub struct GpuHandle {
    pub constraint_data_size: usize,
    pub weights_size: usize,
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constraint_type() {
        let bridge = GpuBridge::new();
        assert!(matches!(
            bridge.parse_constraint_type("subclassof"),
            Ok(ConstraintType::SubClassOf)
        ));
        assert!(matches!(
            bridge.parse_constraint_type("disjoint"),
            Ok(ConstraintType::DisjointWith)
        ));
        assert!(bridge.parse_constraint_type("invalid").is_err());
    }

    #[test]
    fn test_parse_axioms() {
        let mut bridge = GpuBridge::new();
        let axioms = vec![
            "Class1 subClassOf Class2 subclassof 1.0".to_string(),
            "Class3 disjointWith Class4 disjoint 0.9".to_string(),
        ];

        bridge.parse_axioms(&axioms).unwrap();
        assert_eq!(bridge.axioms_cache.len(), 2);
    }

    #[test]
    fn test_serialize_to_gpu() {
        let mut bridge = GpuBridge::new();
        let axioms = vec![
            "ex:Dog subClassOf ex:Animal subclassof 1.0".to_string(),
            "ex:Cat subClassOf ex:Animal subclassof 1.0".to_string(),
            "ex:Dog disjointWith ex:Cat disjoint 1.0".to_string(),
        ];

        bridge.parse_axioms(&axioms).unwrap();
        let graph = bridge.serialize_to_gpu().unwrap();

        assert_eq!(graph.constraint_count, 3);
        assert_eq!(graph.constraint_data.len(), 12); // 3 constraints × 4 elements
        assert_eq!(graph.constraint_weights.len(), 3);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_entity_id_mapping() {
        let mut bridge = GpuBridge::new();
        let axioms = vec!["ex:A subClassOf ex:B subclassof 1.0".to_string()];

        bridge.parse_axioms(&axioms).unwrap();
        let graph = bridge.serialize_to_gpu().unwrap();

        let entity_map = bridge.get_entity_map();
        assert_eq!(entity_map.len(), 3); // A, B, subClassOf
        assert!(entity_map.get_id("ex:A").is_some());
        assert!(entity_map.get_id("ex:B").is_some());
    }

    #[test]
    fn test_serialization_performance() {
        let mut bridge = GpuBridge::new();
        let mut axioms = Vec::new();

        // Generate 50K constraints
        for i in 0..50_000 {
            axioms.push(format!(
                "ex:Class{} subClassOf ex:Class{} subclassof 1.0",
                i,
                (i + 1) % 10000
            ));
        }

        let start = Instant::now();
        bridge.parse_axioms(&axioms).unwrap();
        let graph = bridge.serialize_to_gpu().unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Target: <100ms for 50K constraints
        println!("Serialized 50K constraints in {:.2}ms", elapsed_ms);
        assert_eq!(graph.constraint_count, 50_000);
        assert!(elapsed_ms < 100.0, "Serialization too slow: {}ms", elapsed_ms);
    }

    #[test]
    fn test_gpu_transfer_manager() {
        let mut bridge = GpuBridge::new();
        let axioms = vec![
            "ex:Dog subClassOf ex:Animal subclassof 1.0".to_string(),
            "ex:Cat subClassOf ex:Animal subclassof 1.0".to_string(),
        ];

        bridge.parse_axioms(&axioms).unwrap();
        let graph = bridge.serialize_to_gpu().unwrap();

        let manager = GpuTransferManager::new(1024 * 1024 * 1024); // 1GB
        let handle = manager.upload_to_gpu(&graph).unwrap();

        assert_eq!(handle.constraint_data_size, 8); // 2 constraints × 4 elements
        assert_eq!(handle.weights_size, 2);
        assert!(handle.memory_bytes > 0);
    }

    #[test]
    fn test_deserialize_violations() {
        let mut bridge = GpuBridge::new();
        let axioms = vec!["ex:Dog subClassOf ex:Animal subclassof 1.0".to_string()];

        bridge.parse_axioms(&axioms).unwrap();
        let graph = bridge.serialize_to_gpu().unwrap();

        let mut violations = GpuViolationResult::new();
        violations.add_violation(0, 0.8);

        let reports = bridge.deserialize_violations(&violations, &graph);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].subject, "ex:Dog");
        assert_eq!(reports[0].object, "ex:Animal");
    }

    #[test]
    fn test_violation_report_format() {
        let report = ViolationReport {
            constraint_index: 42,
            subject: "ex:Dog".to_string(),
            predicate: "subClassOf".to_string(),
            object: "ex:Animal".to_string(),
            constraint_type: ConstraintType::SubClassOf,
            severity: 0.85,
        };

        let formatted = report.format();
        assert!(formatted.contains("42"));
        assert!(formatted.contains("0.85"));
        assert!(formatted.contains("ex:Dog"));
        assert!(formatted.contains("SubClassOf"));
    }
}
