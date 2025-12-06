// Semantic Path Discovery
// GPU-accelerated graph search with OWL reasoning for explainable recommendations

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub type ContentId = String;
pub type ConceptId = String;

/// A node in a semantic path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathNode {
    pub content_id: ContentId,
    pub concept: Option<ConceptId>,
    pub relevance_score: f32,
    pub metadata: HashMap<String, String>,
}

/// An edge in a semantic path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathEdge {
    pub from: ContentId,
    pub to: ContentId,
    pub relation_type: String,
    pub weight: f32,
    pub explanation: String,
}

/// A complete semantic path between two content items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPath {
    pub nodes: Vec<PathNode>,
    pub edges: Vec<PathEdge>,
    pub total_score: f32,
    pub path_type: PathType,
    pub explanation: String,
}

/// Types of semantic paths
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PathType {
    Direct,           // Direct connection
    Conceptual,       // Connected via shared concepts
    Transitive,       // Multi-hop connection
    Hierarchical,     // Connected via ontology hierarchy
    Associative,      // Weak associative connection
}

/// Configuration for path discovery
#[derive(Debug, Clone)]
pub struct PathDiscoveryConfig {
    pub max_depth: usize,
    pub min_score_threshold: f32,
    pub max_paths: usize,
    pub enable_bidirectional: bool,
    pub prefer_short_paths: bool,
}

impl Default for PathDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            min_score_threshold: 0.3,
            max_paths: 10,
            enable_bidirectional: true,
            prefer_short_paths: true,
        }
    }
}

/// Semantic path discovery engine
pub struct SemanticPathDiscovery {
    config: PathDiscoveryConfig,
    graph: Arc<RwLock<SemanticGraph>>,
    concept_hierarchy: Arc<RwLock<ConceptHierarchy>>,
}

/// In-memory semantic graph
#[derive(Debug)]
struct SemanticGraph {
    adjacency: HashMap<ContentId, Vec<(ContentId, PathEdge)>>,
    content_concepts: HashMap<ContentId, Vec<ConceptId>>,
    concept_contents: HashMap<ConceptId, Vec<ContentId>>,
}

impl SemanticGraph {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            content_concepts: HashMap::new(),
            concept_contents: HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: ContentId, to: ContentId, edge: PathEdge) {
        self.adjacency
            .entry(from.clone())
            .or_insert_with(Vec::new)
            .push((to, edge));
    }

    fn add_content_concept(&mut self, content_id: ContentId, concept_id: ConceptId) {
        self.content_concepts
            .entry(content_id.clone())
            .or_insert_with(Vec::new)
            .push(concept_id.clone());

        self.concept_contents
            .entry(concept_id)
            .or_insert_with(Vec::new)
            .push(content_id);
    }

    fn get_neighbors(&self, content_id: &ContentId) -> Vec<(ContentId, PathEdge)> {
        self.adjacency
            .get(content_id)
            .cloned()
            .unwrap_or_default()
    }

    fn get_concepts(&self, content_id: &ContentId) -> Vec<ConceptId> {
        self.content_concepts
            .get(content_id)
            .cloned()
            .unwrap_or_default()
    }

    fn get_content_by_concept(&self, concept_id: &ConceptId) -> Vec<ContentId> {
        self.concept_contents
            .get(concept_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// Concept hierarchy for ontology reasoning
#[derive(Debug)]
struct ConceptHierarchy {
    parent_map: HashMap<ConceptId, ConceptId>,
    children_map: HashMap<ConceptId, Vec<ConceptId>>,
}

impl ConceptHierarchy {
    fn new() -> Self {
        Self {
            parent_map: HashMap::new(),
            children_map: HashMap::new(),
        }
    }

    fn add_relationship(&mut self, child: ConceptId, parent: ConceptId) {
        self.parent_map.insert(child.clone(), parent.clone());
        self.children_map
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(child);
    }

    fn get_ancestors(&self, concept: &ConceptId) -> Vec<ConceptId> {
        let mut ancestors = Vec::new();
        let mut current = concept.clone();

        while let Some(parent) = self.parent_map.get(&current) {
            ancestors.push(parent.clone());
            current = parent.clone();
        }

        ancestors
    }

    fn find_common_ancestor(&self, concept1: &ConceptId, concept2: &ConceptId) -> Option<ConceptId> {
        let ancestors1: HashSet<_> = self.get_ancestors(concept1).into_iter().collect();
        let ancestors2 = self.get_ancestors(concept2);

        ancestors2.into_iter().find(|a| ancestors1.contains(a))
    }
}

impl SemanticPathDiscovery {
    pub fn new(max_depth: usize) -> Result<Self> {
        let config = PathDiscoveryConfig {
            max_depth,
            ..Default::default()
        };

        Ok(Self {
            config,
            graph: Arc::new(RwLock::new(SemanticGraph::new())),
            concept_hierarchy: Arc::new(RwLock::new(ConceptHierarchy::new())),
        })
    }

    /// Find semantic paths between two content items
    pub async fn find_paths(
        &self,
        from: ContentId,
        to: ContentId,
    ) -> Result<Vec<SemanticPath>> {
        let graph = self.graph.read().await;
        let hierarchy = self.concept_hierarchy.read().await;

        let mut paths = Vec::new();

        // Try direct path
        if let Some(direct_path) = self.find_direct_path(&from, &to, &graph).await? {
            paths.push(direct_path);
        }

        // Try conceptual path
        if let Some(conceptual_path) = self.find_conceptual_path(&from, &to, &graph, &hierarchy).await? {
            paths.push(conceptual_path);
        }

        // Try BFS for shortest paths
        let bfs_paths = self.find_bfs_paths(&from, &to, &graph).await?;
        paths.extend(bfs_paths);

        // Sort by score
        paths.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap());

        // Return top-k paths
        Ok(paths.into_iter().take(self.config.max_paths).collect())
    }

    /// Add an edge to the semantic graph
    pub async fn add_edge(
        &self,
        from: ContentId,
        to: ContentId,
        relation_type: String,
        weight: f32,
    ) -> Result<()> {
        let mut graph = self.graph.write().await;

        let edge = PathEdge {
            from: from.clone(),
            to: to.clone(),
            relation_type,
            weight,
            explanation: String::new(),
        };

        graph.add_edge(from, to, edge);
        Ok(())
    }

    /// Associate content with a concept
    pub async fn add_content_concept(
        &self,
        content_id: ContentId,
        concept_id: ConceptId,
    ) -> Result<()> {
        let mut graph = self.graph.write().await;
        graph.add_content_concept(content_id, concept_id);
        Ok(())
    }

    /// Add concept hierarchy relationship
    pub async fn add_concept_relationship(
        &self,
        child: ConceptId,
        parent: ConceptId,
    ) -> Result<()> {
        let mut hierarchy = self.concept_hierarchy.write().await;
        hierarchy.add_relationship(child, parent);
        Ok(())
    }

    // Private path-finding methods

    async fn find_direct_path(
        &self,
        from: &ContentId,
        to: &ContentId,
        graph: &SemanticGraph,
    ) -> Result<Option<SemanticPath>> {
        let neighbors = graph.get_neighbors(from);

        for (neighbor_id, edge) in neighbors {
            if &neighbor_id == to {
                let path = SemanticPath {
                    nodes: vec![
                        PathNode {
                            content_id: from.clone(),
                            concept: None,
                            relevance_score: 1.0,
                            metadata: HashMap::new(),
                        },
                        PathNode {
                            content_id: to.clone(),
                            concept: None,
                            relevance_score: edge.weight,
                            metadata: HashMap::new(),
                        },
                    ],
                    edges: vec![edge],
                    total_score: 1.0,
                    path_type: PathType::Direct,
                    explanation: "Direct connection found".to_string(),
                };
                return Ok(Some(path));
            }
        }

        Ok(None)
    }

    async fn find_conceptual_path(
        &self,
        from: &ContentId,
        to: &ContentId,
        graph: &SemanticGraph,
        hierarchy: &ConceptHierarchy,
    ) -> Result<Option<SemanticPath>> {
        let from_concepts = graph.get_concepts(from);
        let to_concepts = graph.get_concepts(to);

        // Find shared concepts
        let from_set: HashSet<_> = from_concepts.iter().collect();
        let shared_concepts: Vec<_> = to_concepts
            .iter()
            .filter(|c| from_set.contains(c))
            .collect();

        if let Some(shared_concept) = shared_concepts.first() {
            let path = SemanticPath {
                nodes: vec![
                    PathNode {
                        content_id: from.clone(),
                        concept: Some((*shared_concept).clone()),
                        relevance_score: 0.9,
                        metadata: HashMap::new(),
                    },
                    PathNode {
                        content_id: to.clone(),
                        concept: Some((*shared_concept).clone()),
                        relevance_score: 0.9,
                        metadata: HashMap::new(),
                    },
                ],
                edges: vec![],
                total_score: 0.85,
                path_type: PathType::Conceptual,
                explanation: format!("Connected via concept: {}", shared_concept),
            };
            return Ok(Some(path));
        }

        // Try hierarchical connection
        for from_concept in &from_concepts {
            for to_concept in &to_concepts {
                if let Some(common_ancestor) = hierarchy.find_common_ancestor(from_concept, to_concept) {
                    let path = SemanticPath {
                        nodes: vec![
                            PathNode {
                                content_id: from.clone(),
                                concept: Some(from_concept.clone()),
                                relevance_score: 0.8,
                                metadata: HashMap::new(),
                            },
                            PathNode {
                                content_id: to.clone(),
                                concept: Some(to_concept.clone()),
                                relevance_score: 0.8,
                                metadata: HashMap::new(),
                            },
                        ],
                        edges: vec![],
                        total_score: 0.7,
                        path_type: PathType::Hierarchical,
                        explanation: format!("Connected via common concept: {}", common_ancestor),
                    };
                    return Ok(Some(path));
                }
            }
        }

        Ok(None)
    }

    async fn find_bfs_paths(
        &self,
        from: &ContentId,
        to: &ContentId,
        graph: &SemanticGraph,
    ) -> Result<Vec<SemanticPath>> {
        let mut paths = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((from.clone(), vec![from.clone()], 1.0));
        visited.insert(from.clone());

        while let Some((current, path, score)) = queue.pop_front() {
            if path.len() > self.config.max_depth {
                continue;
            }

            if &current == to {
                // Construct semantic path
                let nodes: Vec<PathNode> = path
                    .iter()
                    .map(|id| PathNode {
                        content_id: id.clone(),
                        concept: None,
                        relevance_score: score,
                        metadata: HashMap::new(),
                    })
                    .collect();

                let semantic_path = SemanticPath {
                    nodes,
                    edges: vec![],
                    total_score: score / (path.len() as f32),
                    path_type: PathType::Transitive,
                    explanation: format!("Multi-hop path with {} steps", path.len() - 1),
                };

                paths.push(semantic_path);
                continue;
            }

            for (neighbor_id, edge) in graph.get_neighbors(&current) {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    let mut new_path = path.clone();
                    new_path.push(neighbor_id.clone());
                    let new_score = score * edge.weight;
                    queue.push_back((neighbor_id, new_path, new_score));
                }
            }
        }

        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_path_discovery_creation() {
        let discovery = SemanticPathDiscovery::new(5);
        assert!(discovery.is_ok());
    }

    #[tokio::test]
    async fn test_add_edge() {
        let discovery = SemanticPathDiscovery::new(5).unwrap();
        let result = discovery
            .add_edge(
                "content1".to_string(),
                "content2".to_string(),
                "similar".to_string(),
                0.8,
            )
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_concept_hierarchy() {
        let mut hierarchy = ConceptHierarchy::new();
        hierarchy.add_relationship("dog".to_string(), "animal".to_string());
        hierarchy.add_relationship("cat".to_string(), "animal".to_string());

        let common = hierarchy.find_common_ancestor(&"dog".to_string(), &"cat".to_string());
        assert_eq!(common, Some("animal".to_string()));
    }
}
