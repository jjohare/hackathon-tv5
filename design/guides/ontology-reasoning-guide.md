# Ontology Reasoning and OWL Integration Guide

**Target Audience**: Ontology Engineers, Backend Developers
**Prerequisites**: Understanding of RDF/OWL, Rust basics
**Estimated Implementation Time**: 2-3 weeks

---

## Table of Contents

1. [OWL Ontology Design](#1-owl-ontology-design)
2. [Rust OWL Reasoner Setup](#2-rust-owl-reasoner-setup)
3. [Reasoning Pipeline](#3-reasoning-pipeline)
4. [GPU Integration](#4-gpu-integration)
5. [Performance Optimization](#5-performance-optimization)
6. [Production Deployment](#6-production-deployment)

---

## 1. OWL Ontology Design

### 1.1 GMC-O Ontology Structure

**Global Media Content Ontology (GMC-O)** for TV5 Monde

```turtle
# gmc-o-ontology.ttl

@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix media: <http://tv5monde.com/ontology/media#> .
@prefix sem: <http://tv5monde.com/ontology/semantic#> .
@prefix ctx: <http://tv5monde.com/ontology/context#> .

# Ontology declaration
<http://tv5monde.com/ontology/gmc-o> rdf:type owl:Ontology ;
    rdfs:label "Global Media Content Ontology"@en ;
    rdfs:comment "Semantic ontology for TV5 Monde recommendation engine"@en .

#############################
# Top-Level Classes
#############################

media:MediaContent rdf:type owl:Class ;
    rdfs:label "Media Content"@en ;
    rdfs:comment "Abstract class for all media types"@en .

media:Film rdf:type owl:Class ;
    rdfs:subClassOf media:MediaContent ;
    rdfs:label "Film"@en .

media:TVSeries rdf:type owl:Class ;
    rdfs:subClassOf media:MediaContent ;
    rdfs:label "TV Series"@en .

media:Documentary rdf:type owl:Class ;
    rdfs:subClassOf media:MediaContent ;
    rdfs:label "Documentary"@en .

#############################
# Genre Hierarchy
#############################

media:Genre rdf:type owl:Class ;
    rdfs:label "Genre"@en .

media:Drama rdf:type owl:Class ;
    rdfs:subClassOf media:Genre ;
    rdfs:label "Drama"@en .

media:PsychologicalDrama rdf:type owl:Class ;
    rdfs:subClassOf media:Drama ;
    rdfs:label "Psychological Drama"@en .

media:SocialDrama rdf:type owl:Class ;
    rdfs:subClassOf media:Drama ;
    rdfs:label "Social Drama"@en .

media:Comedy rdf:type owl:Class ;
    rdfs:subClassOf media:Genre ;
    rdfs:label "Comedy"@en .

media:DarkComedy rdf:type owl:Class ;
    rdfs:subClassOf media:Comedy ;
    rdfs:label "Dark Comedy"@en .

media:RomanticComedy rdf:type owl:Class ;
    rdfs:subClassOf media:Comedy ;
    rdfs:label "Romantic Comedy"@en .

media:SciFi rdf:type owl:Class ;
    rdfs:subClassOf media:Genre ;
    rdfs:label "Science Fiction"@en .

media:SpaceOpera rdf:type owl:Class ;
    rdfs:subClassOf media:SciFi ;
    rdfs:label "Space Opera"@en .

media:Cyberpunk rdf:type owl:Class ;
    rdfs:subClassOf media:SciFi ;
    rdfs:label "Cyberpunk"@en .

#############################
# Visual Aesthetics
#############################

media:VisualTone rdf:type owl:Class ;
    rdfs:label "Visual Tone"@en .

media:Noir rdf:type media:VisualTone ;
    rdfs:label "Noir"@en ;
    rdfs:comment "High contrast, desaturated, shadowy"@en .

media:Neon rdf:type media:VisualTone ;
    rdfs:label "Neon"@en ;
    rdfs:comment "Vibrant, saturated, futuristic"@en .

media:Pastel rdf:type media:VisualTone ;
    rdfs:label "Pastel"@en ;
    rdfs:comment "Soft, muted, dreamlike"@en .

media:Naturalistic rdf:type media:VisualTone ;
    rdfs:label "Naturalistic"@en ;
    rdfs:comment "True-to-life color grading"@en .

#############################
# Narrative Structure
#############################

media:NarrativeStructure rdf:type owl:Class ;
    rdfs:label "Narrative Structure"@en .

media:Linear rdf:type media:NarrativeStructure ;
    rdfs:label "Linear"@en .

media:NonLinear rdf:type media:NarrativeStructure ;
    rdfs:label "Non-Linear"@en .

media:Circular rdf:type media:NarrativeStructure ;
    rdfs:label "Circular"@en .

media:Episodic rdf:type media:NarrativeStructure ;
    rdfs:label "Episodic"@en .

#############################
# Psychographic States
#############################

sem:PsychographicState rdf:type owl:Class ;
    rdfs:label "Psychographic State"@en ;
    rdfs:comment "Emotional/cognitive state induced by content"@en .

sem:Energizing rdf:type sem:PsychographicState ;
    rdfs:label "Energizing"@en .

sem:Contemplative rdf:type sem:PsychographicState ;
    rdfs:label "Contemplative"@en .

sem:Comforting rdf:type sem:PsychographicState ;
    rdfs:label "Comforting"@en .

sem:Thrilling rdf:type sem:PsychographicState ;
    rdfs:label "Thrilling"@en .

#############################
# Cultural Context
#############################

ctx:CulturalContext rdf:type owl:Class ;
    rdfs:label "Cultural Context"@en .

ctx:Francophone rdf:type ctx:CulturalContext ;
    rdfs:label "Francophone"@en .

ctx:European rdf:type ctx:CulturalContext ;
    rdfs:label "European"@en .

ctx:NorthAmerican rdf:type ctx:CulturalContext ;
    rdfs:label "North American"@en .

#############################
# Object Properties
#############################

media:hasGenre rdf:type owl:ObjectProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range media:Genre ;
    rdfs:label "has genre"@en .

media:hasVisualTone rdf:type owl:ObjectProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range media:VisualTone ;
    rdfs:label "has visual tone"@en .

media:hasNarrativeStructure rdf:type owl:ObjectProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range media:NarrativeStructure ;
    rdfs:label "has narrative structure"@en .

sem:inducesPsychographicState rdf:type owl:ObjectProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range sem:PsychographicState ;
    rdfs:label "induces psychographic state"@en .

ctx:hasCulturalContext rdf:type owl:ObjectProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range ctx:CulturalContext ;
    rdfs:label "has cultural context"@en .

#############################
# Datatype Properties
#############################

media:releaseYear rdf:type owl:DatatypeProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range xsd:integer .

media:duration rdf:type owl:DatatypeProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range xsd:integer ;
    rdfs:comment "Duration in minutes"@en .

media:language rdf:type owl:DatatypeProperty ;
    rdfs:domain media:MediaContent ;
    rdfs:range xsd:string .

#############################
# Semantic Rules (SWRL)
#############################

# Rule 1: Dark visual tone + psychological drama → contemplative state
[ rdf:type swrl:Imp ;
  swrl:body [
    rdf:type swrl:AtomList ;
    rdf:first [ rdf:type swrl:ClassAtom ;
                swrl:classPredicate media:PsychologicalDrama ;
                swrl:argument1 ?x ] ;
    rdf:rest [ rdf:type swrl:AtomList ;
               rdf:first [ rdf:type swrl:IndividualPropertyAtom ;
                           swrl:propertyPredicate media:hasVisualTone ;
                           swrl:argument1 ?x ;
                           swrl:argument2 media:Noir ] ;
               rdf:rest rdf:nil ]
  ] ;
  swrl:head [ rdf:type swrl:AtomList ;
              rdf:first [ rdf:type swrl:IndividualPropertyAtom ;
                          swrl:propertyPredicate sem:inducesPsychographicState ;
                          swrl:argument1 ?x ;
                          swrl:argument2 sem:Contemplative ] ;
              rdf:rest rdf:nil ]
] .

# Rule 2: SciFi subgenre inheritance
[ rdf:type swrl:Imp ;
  swrl:body [
    rdf:type swrl:AtomList ;
    rdf:first [ rdf:type swrl:IndividualPropertyAtom ;
                swrl:propertyPredicate media:hasGenre ;
                swrl:argument1 ?x ;
                swrl:argument2 media:SpaceOpera ] ;
    rdf:rest rdf:nil
  ] ;
  swrl:head [ rdf:type swrl:AtomList ;
              rdf:first [ rdf:type swrl:IndividualPropertyAtom ;
                          swrl:propertyPredicate media:hasGenre ;
                          swrl:argument1 ?x ;
                          swrl:argument2 media:SciFi ] ;
              rdf:rest rdf:nil ]
] .
```

### 1.2 Example Instance Data

```turtle
# Example film instance
@prefix : <http://tv5monde.com/data#> .

:Film12345 rdf:type media:Film ;
    rdfs:label "Amélie"@fr ;
    media:releaseYear 2001 ;
    media:duration 122 ;
    media:language "fr" ;
    media:hasGenre media:RomanticComedy ;
    media:hasVisualTone media:Pastel ;
    media:hasNarrativeStructure media:Episodic ;
    sem:inducesPsychographicState sem:Comforting ;
    ctx:hasCulturalContext ctx:Francophone .
```

---

## 2. Rust OWL Reasoner Setup

### 2.1 Project Setup

```bash
cargo new --lib ontology-reasoner
cd ontology-reasoner
```

### 2.2 Dependencies

```toml
# Cargo.toml
[package]
name = "ontology-reasoner"
version = "0.1.0"
edition = "2021"

[dependencies]
# RDF/OWL parsing
oxigraph = "0.3"
sophia = "0.8"

# Data structures
petgraph = "0.6"
dashmap = "5.5"
rayon = "1.7"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

[dev-dependencies]
criterion = "0.5"
```

### 2.3 Core Reasoner Implementation

```rust
// src/reasoner.rs
use oxigraph::model::*;
use oxigraph::store::Store;
use petgraph::graphmap::DiGraphMap;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

pub struct OWLReasoner {
    store: Store,
    class_hierarchy: DiGraphMap<String, ()>,
    property_hierarchy: DiGraphMap<String, ()>,
}

impl OWLReasoner {
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: Store::new()?,
            class_hierarchy: DiGraphMap::new(),
            property_hierarchy: DiGraphMap::new(),
        })
    }

    /// Load ontology from Turtle file
    pub fn load_ontology(&mut self, ttl_data: &str) -> Result<()> {
        self.store.load_from_reader(
            oxigraph::io::GraphFormat::Turtle,
            ttl_data.as_bytes(),
        )?;

        // Build class hierarchy
        self.build_class_hierarchy()?;

        // Build property hierarchy
        self.build_property_hierarchy()?;

        Ok(())
    }

    /// Build class hierarchy graph for transitive closure
    fn build_class_hierarchy(&mut self) -> Result<()> {
        let rdfs_subclass = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf")?;

        for quad in self.store.quads_for_pattern(None, Some(&rdfs_subclass), None, None) {
            let quad = quad?;
            let subject = quad.subject.to_string();
            let object = quad.object.to_string();

            self.class_hierarchy.add_edge(subject, object, ());
        }

        Ok(())
    }

    fn build_property_hierarchy(&mut self) -> Result<()> {
        let rdfs_subproperty = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")?;

        for quad in self.store.quads_for_pattern(None, Some(&rdfs_subproperty), None, None) {
            let quad = quad?;
            let subject = quad.subject.to_string();
            let object = quad.object.to_string();

            self.property_hierarchy.add_edge(subject, object, ());
        }

        Ok(())
    }

    /// Compute transitive closure for class hierarchy
    pub fn get_superclasses(&self, class: &str) -> HashSet<String> {
        use petgraph::visit::Bfs;

        let mut superclasses = HashSet::new();
        let mut bfs = Bfs::new(&self.class_hierarchy, class.to_string());

        while let Some(node) = bfs.next(&self.class_hierarchy) {
            superclasses.insert(node);
        }

        superclasses
    }

    /// Check if instance belongs to class (including inferred via subclass)
    pub fn is_instance_of(&self, instance: &str, class: &str) -> Result<bool> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let instance_node = NamedNode::new(instance)?;

        // Get direct types
        let mut direct_types = HashSet::new();
        for quad in self.store.quads_for_pattern(Some(&instance_node), Some(&rdf_type), None, None) {
            let quad = quad?;
            direct_types.insert(quad.object.to_string());
        }

        // Check if any direct type is subclass of target class
        for direct_type in direct_types {
            let superclasses = self.get_superclasses(&direct_type);
            if superclasses.contains(class) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Apply SWRL-style rule: if body matches, add head triples
    pub fn apply_rule(
        &mut self,
        body_patterns: &[(String, String, String)],
        head_triple: (String, String, String),
    ) -> Result<usize> {
        let mut instances = HashSet::new();

        // Find instances matching body patterns
        for (s, p, o) in body_patterns {
            let subj = NamedNode::new(s)?;
            let pred = NamedNode::new(p)?;
            let obj = NamedNode::new(o)?;

            for quad in self.store.quads_for_pattern(Some(&subj), Some(&pred), Some(&obj), None) {
                let quad = quad?;
                instances.insert(quad.subject.to_string());
            }
        }

        // Add inferred triples
        let mut added = 0;
        for instance in instances {
            let triple = QuadRef::new(
                &NamedNode::new(&head_triple.0)?,
                &NamedNode::new(&head_triple.1)?,
                &NamedNode::new(&head_triple.2)?,
                GraphNameRef::DefaultGraph,
            );

            self.store.insert(triple)?;
            added += 1;
        }

        Ok(added)
    }

    /// Query for instances with semantic constraints
    pub fn query_instances(
        &self,
        class_filter: Option<&str>,
        property_constraints: &[(String, String)],
    ) -> Result<Vec<String>> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;

        let mut candidates: HashSet<String> = HashSet::new();

        // Get instances of class
        if let Some(class) = class_filter {
            let class_node = NamedNode::new(class)?;
            for quad in self.store.quads_for_pattern(None, Some(&rdf_type), Some(&class_node), None) {
                let quad = quad?;
                candidates.insert(quad.subject.to_string());
            }
        } else {
            // Get all instances
            for quad in self.store.quads_for_pattern(None, Some(&rdf_type), None, None) {
                let quad = quad?;
                candidates.insert(quad.subject.to_string());
            }
        }

        // Filter by property constraints
        for (property, value) in property_constraints {
            let prop_node = NamedNode::new(property)?;
            let val_node = NamedNode::new(value)?;

            candidates.retain(|instance| {
                let inst_node = NamedNode::new(instance).unwrap();
                self.store
                    .quads_for_pattern(Some(&inst_node), Some(&prop_node), Some(&val_node), None)
                    .count()
                    > 0
            });
        }

        Ok(candidates.into_iter().collect())
    }
}
```

---

## 3. Reasoning Pipeline

### 3.1 Inference Engine

```rust
// src/inference.rs
use crate::reasoner::OWLReasoner;
use anyhow::Result;
use rayon::prelude::*;

pub struct InferenceEngine {
    reasoner: OWLReasoner,
    rules: Vec<InferenceRule>,
}

#[derive(Clone)]
pub struct InferenceRule {
    pub name: String,
    pub body: Vec<(String, String, String)>,
    pub head: (String, String, String),
}

impl InferenceEngine {
    pub fn new(reasoner: OWLReasoner) -> Self {
        Self {
            reasoner,
            rules: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    /// Apply all rules until fixed point (no new inferences)
    pub fn materialize(&mut self) -> Result<usize> {
        let mut total_inferred = 0;
        let mut iteration = 0;

        loop {
            let mut inferred = 0;

            for rule in &self.rules {
                inferred += self.reasoner.apply_rule(&rule.body, rule.head.clone())?;
            }

            total_inferred += inferred;
            iteration += 1;

            tracing::info!(
                "Materialization iteration {}: {} new triples inferred",
                iteration,
                inferred
            );

            if inferred == 0 || iteration > 100 {
                break;
            }
        }

        Ok(total_inferred)
    }

    /// Query with semantic reasoning
    pub fn semantic_query(
        &self,
        user_preferences: &UserPreferences,
    ) -> Result<Vec<RecommendationCandidate>> {
        let mut candidates = Vec::new();

        // Apply transitive closure for genre preferences
        let mut expanded_genres = user_preferences.preferred_genres.clone();
        for genre in &user_preferences.preferred_genres {
            let superclasses = self.reasoner.get_superclasses(genre);
            expanded_genres.extend(superclasses);
        }

        // Query with expanded genres
        for genre in expanded_genres {
            let instances = self.reasoner.query_instances(
                Some("http://tv5monde.com/ontology/media#MediaContent"),
                &[(
                    "http://tv5monde.com/ontology/media#hasGenre".to_string(),
                    genre,
                )],
            )?;

            for instance in instances {
                candidates.push(RecommendationCandidate {
                    id: instance,
                    score: 1.0,
                    matched_rules: vec![format!("Genre: {}", genre)],
                });
            }
        }

        Ok(candidates)
    }
}

#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub preferred_genres: Vec<String>,
    pub preferred_tones: Vec<String>,
    pub target_psychographic_state: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RecommendationCandidate {
    pub id: String,
    pub score: f32,
    pub matched_rules: Vec<String>,
}
```

### 3.2 GPU Integration Bridge

```rust
// src/gpu_bridge.rs
use anyhow::Result;
use std::collections::HashMap;

/// Bridge between ontology reasoning and GPU kernels
pub struct OntologyGPUBridge {
    constraint_matrix: Vec<f32>,
    constraint_map: HashMap<String, usize>,
}

impl OntologyGPUBridge {
    /// Convert ontology constraints to GPU-compatible matrix
    pub fn constraints_to_matrix(
        &self,
        instances: &[String],
        constraints: &[(String, String, String)],
    ) -> Result<Vec<f32>> {
        let n = instances.len();
        let mut matrix = vec![0.0f32; n * n];

        for (i, inst_a) in instances.iter().enumerate() {
            for (j, inst_b) in instances.iter().enumerate() {
                if i == j {
                    continue;
                }

                // Check ontology constraints
                let weight = self.compute_constraint_weight(inst_a, inst_b, constraints)?;
                matrix[i * n + j] = weight;
            }
        }

        Ok(matrix)
    }

    fn compute_constraint_weight(
        &self,
        inst_a: &str,
        inst_b: &str,
        constraints: &[(String, String, String)],
    ) -> Result<f32> {
        // Weight based on shared ontology classes
        let mut weight = 0.0;

        // Same genre: +0.5
        // Same visual tone: +0.3
        // Same narrative structure: +0.2

        // Placeholder logic (replace with actual ontology queries)
        weight += 0.5;  // Example

        Ok(weight)
    }

    /// Send constraints to GPU kernel
    pub unsafe fn upload_to_gpu(
        &self,
        constraint_matrix: &[f32],
        device_ptr: *mut f32,
    ) -> Result<()> {
        use std::ptr;

        // Copy matrix to GPU (placeholder - use actual CUDA API)
        ptr::copy_nonoverlapping(
            constraint_matrix.as_ptr(),
            device_ptr,
            constraint_matrix.len(),
        );

        Ok(())
    }
}
```

---

## 4. GPU Integration

### 4.1 Ontology Constraints Kernel

```cuda
// src/kernels/ontology_constraints.cu
#include <cuda_runtime.h>

// Apply ontology constraints to embeddings
__global__ void ontology_constraints_kernel(
    float* embeddings,            // [N, D] content embeddings
    const float* constraint_matrix, // [N, N] ontology constraint weights
    float* adjusted_embeddings,   // [N, D] output
    int N,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // Apply constraints from neighbors
    for (int d = 0; d < D; d++) {
        float adjusted = embeddings[idx * D + d];

        // Weighted influence from ontologically related items
        for (int neighbor = 0; neighbor < N; neighbor++) {
            float weight = constraint_matrix[idx * N + neighbor];
            if (weight > 0.1f) {  // Threshold for relevance
                adjusted += weight * embeddings[neighbor * D + d];
            }
        }

        adjusted_embeddings[idx * D + d] = adjusted / (1.0f + N);
    }
}

// Host function
extern "C" void apply_ontology_constraints(
    float* embeddings,
    const float* constraint_matrix,
    float* adjusted_embeddings,
    int N,
    int D,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    ontology_constraints_kernel<<<blocks, threads, 0, stream>>>(
        embeddings, constraint_matrix, adjusted_embeddings, N, D
    );
}
```

### 4.2 Rust-CUDA Integration

```rust
// src/gpu_ontology.rs
use cudarc::driver::*;
use anyhow::Result;

pub struct GPUOntologyProcessor {
    device: Arc<CudaDevice>,
    module: CudaModule,
}

impl GPUOntologyProcessor {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;
        let ptx = compile_ptx(include_str!("kernels/ontology_constraints.cu"))?;
        let module = device.load_ptx(ptx, "ontology_constraints", &["apply_ontology_constraints"])?;

        Ok(Self { device, module })
    }

    pub fn apply_constraints(
        &self,
        embeddings: &[f32],
        constraint_matrix: &[f32],
        n: usize,
        d: usize,
    ) -> Result<Vec<f32>> {
        // Upload to GPU
        let d_embeddings = self.device.htod_copy(embeddings)?;
        let d_constraints = self.device.htod_copy(constraint_matrix)?;
        let d_output = self.device.alloc_zeros::<f32>(n * d)?;

        // Launch kernel
        let func = self.module.get_func("apply_ontology_constraints")?;
        let cfg = LaunchConfig {
            grid_dim: ((n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_embeddings,
                &d_constraints,
                &d_output,
                n as i32,
                d as i32,
                0u64,
            ))?;
        }

        // Download result
        Ok(self.device.dtoh_sync_copy(&d_output)?)
    }
}
```

---

## 5. Performance Optimization

### 5.1 Transitive Closure Caching

```rust
// Cache transitive closure results
use std::sync::Arc;
use dashmap::DashMap;

pub struct CachedReasoner {
    reasoner: Arc<OWLReasoner>,
    superclass_cache: DashMap<String, HashSet<String>>,
}

impl CachedReasoner {
    pub fn get_superclasses_cached(&self, class: &str) -> HashSet<String> {
        if let Some(cached) = self.superclass_cache.get(class) {
            return cached.clone();
        }

        let superclasses = self.reasoner.get_superclasses(class);
        self.superclass_cache.insert(class.to_string(), superclasses.clone());

        superclasses
    }
}
```

### 5.2 Parallel Rule Application

```rust
use rayon::prelude::*;

impl InferenceEngine {
    pub fn materialize_parallel(&mut self) -> Result<usize> {
        let total_inferred: usize = self.rules
            .par_iter()
            .map(|rule| {
                self.reasoner.apply_rule(&rule.body, rule.head.clone()).unwrap_or(0)
            })
            .sum();

        Ok(total_inferred)
    }
}
```

---

## 6. Production Deployment

### 6.1 Neo4j Integration

```rust
// Store ontology in Neo4j for production queries
use neo4rs::*;

pub struct Neo4jOntologyStore {
    graph: Graph,
}

impl Neo4jOntologyStore {
    pub async fn new(uri: &str, user: &str, pass: &str) -> Result<Self> {
        let config = ConfigBuilder::default()
            .uri(uri)
            .user(user)
            .password(pass)
            .build()?;

        let graph = Graph::connect(config).await?;

        Ok(Self { graph })
    }

    pub async fn create_indexes(&self) -> Result<()> {
        self.graph.run(query("CREATE INDEX media_genre IF NOT EXISTS FOR (n:Media) ON (n.genre)")).await?;
        self.graph.run(query("CREATE INDEX media_tone IF NOT EXISTS FOR (n:Media) ON (n.visualTone)")).await?;

        Ok(())
    }

    pub async fn store_ontology_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<()> {
        let query_str = format!(
            "MERGE (s {{id: $subject}}) \
             MERGE (o {{id: $object}}) \
             MERGE (s)-[:{predicate}]->(o)"
        );

        self.graph
            .run(
                query(&query_str)
                    .param("subject", subject)
                    .param("object", object),
            )
            .await?;

        Ok(())
    }

    pub async fn query_with_reasoning(
        &self,
        genre: &str,
        max_depth: usize,
    ) -> Result<Vec<String>> {
        // Cypher query with variable-length path (transitive closure)
        let query_str = format!(
            "MATCH (m:Media)-[:hasGenre*1..{max_depth}]->(g:Genre {{id: $genre}}) \
             RETURN m.id AS media_id"
        );

        let mut result = self.graph.execute(query(&query_str).param("genre", genre)).await?;

        let mut media_ids = Vec::new();
        while let Some(row) = result.next().await? {
            if let Some(id) = row.get::<String>("media_id") {
                media_ids.push(id);
            }
        }

        Ok(media_ids)
    }
}
```

---

## Performance Targets

**Target Metrics (100M triples):**
- **Transitive closure**: <50ms
- **Rule application**: <100ms per iteration
- **Query latency**: <5ms (simple), <20ms (complex)
- **Materialization**: <10 minutes (full ontology)
- **GPU constraint application**: <5ms

**Optimization Checklist:**
- [ ] Transitive closure cached
- [ ] Parallel rule application enabled
- [ ] Neo4j indexes created
- [ ] GPU kernels optimized
- [ ] Rule fixed-point converges in <10 iterations

---

**Next Steps:**
1. Integrate with vector search pipeline
2. Implement SWRL rule parser
3. Deploy Neo4j cluster
4. GPU kernel performance tuning

**Related Guides:**
- [GPU Setup Guide](gpu-setup-guide.md)
- [Vector Search Implementation](vector-search-implementation.md)
- [Learning Pipeline Guide](learning-pipeline-guide.md)
