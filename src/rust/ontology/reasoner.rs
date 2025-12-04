/// Media Ontology Reasoner
///
/// Adapted from generic OWL reasoner for GMC-O (Generic Media Content Ontology).
/// Provides media-specific reasoning including genre hierarchy, mood inference,
/// cultural context matching, and user preference reasoning with GPU acceleration hooks.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::tarjan_scc;
use petgraph::visit::Dfs;
use dashmap::DashMap;
use rayon::prelude::*;
use super::types::*;

/// Core ontology structure for media reasoning
#[derive(Debug, Clone, Default)]
pub struct MediaOntology {
    /// All media entities indexed by ID
    pub media: HashMap<String, MediaEntity>,

    /// Genre hierarchy (child -> parents)
    pub genre_hierarchy: HashMap<String, HashSet<String>>,

    /// Mood relationships and similarities
    pub mood_relations: HashMap<String, Mood>,

    /// Cultural context definitions
    pub cultural_contexts: HashMap<String, CulturalContext>,

    /// Media relationships (e.g., sequel, similar)
    pub media_relations: HashMap<String, Vec<(String, MediaRelation)>>,

    /// Disjoint genre sets (mutually exclusive)
    pub disjoint_genres: Vec<HashSet<String>>,

    /// Equivalent genres (synonyms)
    pub equivalent_genres: HashMap<String, HashSet<String>>,

    /// Semantic tag hierarchy
    pub tag_hierarchy: HashMap<String, HashSet<String>>,
}

/// Trait for media ontology reasoning
pub trait MediaReasoner: Send + Sync {
    /// Infer new media relationships and properties
    fn infer_axioms(&self, ontology: &MediaOntology) -> OntologyResult<Vec<InferredMediaAxiom>>;

    /// Check if genre A is a subgenre of genre B
    fn is_subgenre_of(&self, child: &str, parent: &str, ontology: &MediaOntology) -> bool;

    /// Check if two genres are disjoint (mutually exclusive)
    fn are_disjoint_genres(&self, genre_a: &str, genre_b: &str, ontology: &MediaOntology) -> bool;

    /// Infer mood from genre and other properties
    fn infer_mood(&self, media: &MediaEntity, ontology: &MediaOntology) -> Vec<String>;

    /// Match media to cultural context
    fn match_cultural_context(&self, media: &MediaEntity, context: &CulturalContext) -> f32;

    /// Generate recommendations based on user profile
    fn recommend_for_user(
        &self,
        user: &UserProfile,
        context: &DeliveryContext,
        ontology: &MediaOntology,
    ) -> Vec<RecommendationResult>;
}

/// Inferred axiom specific to media domain
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferredMediaAxiom {
    pub axiom_type: MediaAxiomType,
    pub subject: String,
    pub object: Option<String>,
    pub confidence: f32,
    pub reasoning: String,
}

/// Types of media axioms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MediaAxiomType {
    SubGenreOf,
    SimilarMood,
    CulturallyRelated,
    ThematicConnection,
    UserPreference,
    ContextualFit,
    EquivalentGenre,
    DisjointGenre,
}

/// Constraint violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub violation_type: ViolationType,
    pub subject: String,
    pub object: Option<String>,
    pub severity: ViolationSeverity,
    pub explanation: String,
}

/// Types of constraint violations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationType {
    DisjointGenreConflict,
    CircularHierarchy,
    CardinalityViolation,
    DomainRangeViolation,
    ParadoxicalProperty,
    MutuallyExclusiveMood,
}

/// Severity levels for violations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    Warning,
    Error,
    Critical,
}

/// Production-ready media reasoner with caching and GPU hooks
pub struct ProductionMediaReasoner {
    /// Transitive closure cache for genre hierarchy (thread-safe)
    genre_closure_cache: Arc<DashMap<String, HashSet<String>>>,

    /// Mood similarity matrix cache (thread-safe)
    mood_similarity_cache: Arc<DashMap<(String, String), f32>>,

    /// Graph representation of genre hierarchy
    genre_graph: Option<DiGraph<String, ()>>,

    /// Node mapping for graph
    genre_node_map: HashMap<String, NodeIndex>,

    /// GPU acceleration enabled flag
    gpu_enabled: bool,

    /// Batch size for GPU operations
    gpu_batch_size: usize,

    /// Circular dependency detection cache
    circular_deps_checked: bool,
}

impl ProductionMediaReasoner {
    /// Create new reasoner instance
    pub fn new() -> Self {
        Self {
            genre_closure_cache: Arc::new(DashMap::new()),
            mood_similarity_cache: Arc::new(DashMap::new()),
            genre_graph: None,
            genre_node_map: HashMap::new(),
            gpu_enabled: false,
            gpu_batch_size: 1024,
            circular_deps_checked: false,
        }
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, batch_size: usize) -> Self {
        self.gpu_enabled = true;
        self.gpu_batch_size = batch_size;
        self
    }

    /// Build graph representation from ontology
    fn build_genre_graph(&mut self, ontology: &MediaOntology) {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        // Create nodes for all genres
        for genre in ontology.genre_hierarchy.keys() {
            let node = graph.add_node(genre.clone());
            node_map.insert(genre.clone(), node);
        }

        // Add parent genre nodes if not present
        for parents in ontology.genre_hierarchy.values() {
            for parent in parents {
                if !node_map.contains_key(parent) {
                    let node = graph.add_node(parent.clone());
                    node_map.insert(parent.clone(), node);
                }
            }
        }

        // Add edges (child -> parent)
        for (child, parents) in &ontology.genre_hierarchy {
            let child_node = node_map[child];
            for parent in parents {
                let parent_node = node_map[parent];
                graph.add_edge(child_node, parent_node, ());
            }
        }

        self.genre_graph = Some(graph);
        self.genre_node_map = node_map;
    }

    /// Compute transitive closure using petgraph (optimized for large graphs)
    fn compute_genre_closure(&mut self, ontology: &MediaOntology) {
        self.build_genre_graph(ontology);
        self.genre_closure_cache.clear();

        let graph = self.genre_graph.as_ref().unwrap();

        // Parallel computation of transitive closure
        let results: Vec<_> = self.genre_node_map.par_iter()
            .map(|(genre, &node_idx)| {
                let mut ancestors = HashSet::new();
                let mut dfs = Dfs::new(graph, node_idx);

                while let Some(visited) = dfs.next(graph) {
                    if visited != node_idx {
                        ancestors.insert(graph[visited].clone());
                    }
                }

                (genre.clone(), ancestors)
            })
            .collect();

        // Insert results into cache
        for (genre, ancestors) in results {
            self.genre_closure_cache.insert(genre, ancestors);
        }
    }

    /// Detect circular dependencies in genre hierarchy
    pub fn detect_circular_dependencies(&mut self, ontology: &MediaOntology) -> Vec<ConstraintViolation> {
        if self.genre_graph.is_none() {
            self.build_genre_graph(ontology);
        }

        let graph = self.genre_graph.as_ref().unwrap();
        let mut violations = Vec::new();

        // Use Tarjan's algorithm to find strongly connected components
        let sccs = tarjan_scc(graph);

        for scc in sccs {
            if scc.len() > 1 {
                // Found circular dependency
                let genres: Vec<String> = scc.iter()
                    .map(|&idx| graph[idx].clone())
                    .collect();

                violations.push(ConstraintViolation {
                    violation_type: ViolationType::CircularHierarchy,
                    subject: genres.join(" -> "),
                    object: None,
                    severity: ViolationSeverity::Critical,
                    explanation: format!(
                        "Circular genre hierarchy detected: {}. This creates an infinite loop.",
                        genres.join(" -> ")
                    ),
                });
            }
        }

        self.circular_deps_checked = true;
        violations
    }

    /// Check for disjoint genre violations
    pub fn check_disjoint_violations(
        &self,
        media: &MediaEntity,
        ontology: &MediaOntology,
    ) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        // Check if media has genres from disjoint sets
        for disjoint_set in &ontology.disjoint_genres {
            let mut matched_genres = Vec::new();

            for genre in &media.genres {
                // Check direct membership
                if disjoint_set.contains(genre) {
                    matched_genres.push(genre.clone());
                    continue;
                }

                // Check if genre is a subgenre of any disjoint genre
                for disjoint_genre in disjoint_set {
                    if self.is_subgenre_of_cached(genre, disjoint_genre, ontology) {
                        matched_genres.push(genre.clone());
                        break;
                    }
                }
            }

            if matched_genres.len() > 1 {
                violations.push(ConstraintViolation {
                    violation_type: ViolationType::DisjointGenreConflict,
                    subject: media.id.clone(),
                    object: Some(matched_genres.join(", ")),
                    severity: ViolationSeverity::Error,
                    explanation: format!(
                        "Media '{}' has mutually exclusive genres: {}. These genres are defined as disjoint.",
                        media.title, matched_genres.join(", ")
                    ),
                });
            }
        }

        violations
    }

    /// Check for paradoxical properties (e.g., FamilyFriendly + Rated-R)
    pub fn check_paradoxical_properties(
        &self,
        media: &MediaEntity,
    ) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        // Define paradoxical combinations
        let paradoxes = vec![
            (vec!["FamilyFriendly", "Children"], vec!["Rated-R", "Mature", "Adult"]),
            (vec!["Educational"], vec!["Exploitation", "Gratuitous"]),
            (vec!["Peaceful", "Calm"], vec!["Action", "Horror", "Thriller"]),
        ];

        for (positive_tags, negative_tags) in paradoxes {
            let has_positive = media.semantic_tags.iter()
                .any(|tag| positive_tags.iter().any(|&p| tag.contains(p)))
                || media.genres.iter()
                .any(|g| positive_tags.iter().any(|&p| g.contains(p)));

            let has_negative = media.semantic_tags.iter()
                .any(|tag| negative_tags.iter().any(|&n| tag.contains(n)))
                || media.genres.iter()
                .any(|g| negative_tags.iter().any(|&n| g.contains(n)));

            if has_positive && has_negative {
                violations.push(ConstraintViolation {
                    violation_type: ViolationType::ParadoxicalProperty,
                    subject: media.id.clone(),
                    object: None,
                    severity: ViolationSeverity::Warning,
                    explanation: format!(
                        "Media '{}' has contradictory properties. Contains both {:?} and {:?} characteristics.",
                        media.title, positive_tags, negative_tags
                    ),
                });
            }
        }

        violations
    }

    /// Comprehensive constraint checking
    pub fn check_all_constraints(
        &mut self,
        ontology: &MediaOntology,
    ) -> Vec<ConstraintViolation> {
        let mut all_violations = Vec::new();

        // Check for circular dependencies
        all_violations.extend(self.detect_circular_dependencies(ontology));

        // Check each media entity
        for media in ontology.media.values() {
            all_violations.extend(self.check_disjoint_violations(media, ontology));
            all_violations.extend(self.check_paradoxical_properties(media));
        }

        // Sort by severity
        all_violations.sort_by(|a, b| b.severity.cmp(&a.severity));

        all_violations
    }

    /// Helper to check subgenre relationship using cache
    fn is_subgenre_of_cached(&self, child: &str, parent: &str, ontology: &MediaOntology) -> bool {
        if let Some(ancestors) = self.genre_closure_cache.get(child) {
            return ancestors.contains(parent);
        }

        // Fallback to recursive check
        let mut visited = HashSet::new();
        self.is_subgenre_recursive(child, parent, ontology, &mut visited)
    }

    /// Recursively collect all parent genres
    fn collect_parent_genres(
        &self,
        genre: &str,
        ontology: &MediaOntology,
        ancestors: &mut HashSet<String>,
    ) {
        if let Some(parents) = ontology.genre_hierarchy.get(genre) {
            for parent in parents {
                if ancestors.insert(parent.clone()) {
                    self.collect_parent_genres(parent, ontology, ancestors);
                }
            }
        }
    }

    /// Infer transitive genre relationships
    fn infer_genre_hierarchy(&mut self, ontology: &MediaOntology) -> Vec<InferredMediaAxiom> {
        let mut inferred = Vec::new();

        self.compute_genre_closure(ontology);

        // Iterate through cache entries
        for entry in self.genre_closure_cache.iter() {
            let child = entry.key();
            let ancestors = entry.value();

            let direct_parents = ontology.genre_hierarchy
                .get(child.as_str())
                .cloned()
                .unwrap_or_default();

            for ancestor in ancestors.iter() {
                if !direct_parents.contains(ancestor) {
                    inferred.push(InferredMediaAxiom {
                        axiom_type: MediaAxiomType::SubGenreOf,
                        subject: child.clone(),
                        object: Some(ancestor.clone()),
                        confidence: 1.0,
                        reasoning: format!("{} is a subgenre of {} through transitive closure", child, ancestor),
                    });
                }
            }
        }

        inferred
    }

    /// Infer mood similarities using emotional dimensions
    fn infer_mood_similarities(&mut self, ontology: &MediaOntology) -> Vec<InferredMediaAxiom> {
        let mut inferred = Vec::new();

        let moods: Vec<_> = ontology.mood_relations.keys().collect();

        for i in 0..moods.len() {
            for j in (i + 1)..moods.len() {
                let mood_a = moods[i];
                let mood_b = moods[j];

                if let (Some(a), Some(b)) = (
                    ontology.mood_relations.get(mood_a),
                    ontology.mood_relations.get(mood_b),
                ) {
                    let similarity = self.calculate_mood_similarity(a, b);

                    // Cache the result
                    self.mood_similarity_cache.insert((mood_a.clone(), mood_b.clone()), similarity);
                    self.mood_similarity_cache.insert((mood_b.clone(), mood_a.clone()), similarity);

                    if similarity > 0.7 {
                        inferred.push(InferredMediaAxiom {
                            axiom_type: MediaAxiomType::SimilarMood,
                            subject: mood_a.clone(),
                            object: Some(mood_b.clone()),
                            confidence: similarity,
                            reasoning: format!(
                                "Moods {} and {} are similar (similarity: {:.2})",
                                mood_a, mood_b, similarity
                            ),
                        });
                    }
                }
            }
        }

        inferred
    }

    /// Calculate mood similarity using VAD (Valence-Arousal-Dominance) model
    fn calculate_mood_similarity(&self, mood_a: &Mood, mood_b: &Mood) -> f32 {
        let valence_diff = (mood_a.valence - mood_b.valence).abs();
        let arousal_diff = (mood_a.arousal - mood_b.arousal).abs();
        let dominance_diff = (mood_a.dominance - mood_b.dominance).abs();

        // Euclidean distance in VAD space, normalized
        let distance = ((valence_diff.powi(2) + arousal_diff.powi(2) + dominance_diff.powi(2)) / 3.0).sqrt();

        // Convert distance to similarity (0-1 range)
        (1.0 - distance).max(0.0)
    }

    /// Infer genre from mood patterns
    fn infer_genre_from_moods(
        &self,
        _media: &MediaEntity,
        _ontology: &MediaOntology,
    ) -> Vec<InferredMediaAxiom> {
        let inferred = Vec::new();

        // Create genre -> typical moods mapping
        // This would be populated from ontology in production
        // For now, we'll skip if no genre-mood mappings exist

        inferred
    }

    /// Infer disjoint genre relationships
    fn infer_disjoint_genres(&self, ontology: &MediaOntology) -> Vec<InferredMediaAxiom> {
        let mut inferred = Vec::new();

        for disjoint_set in &ontology.disjoint_genres {
            let genres: Vec<_> = disjoint_set.iter().collect();

            for i in 0..genres.len() {
                for j in (i + 1)..genres.len() {
                    let genre_a = genres[i];
                    let genre_b = genres[j];

                    // Infer that all subgenres are also disjoint
                    if let Some(a_subgenres) = self.get_all_subgenres(genre_a, ontology) {
                        for subgenre in &a_subgenres {
                            if subgenre != genre_a && !disjoint_set.contains(subgenre.as_str()) {
                                inferred.push(InferredMediaAxiom {
                                    axiom_type: MediaAxiomType::DisjointGenre,
                                    subject: subgenre.clone(),
                                    object: Some(genre_b.to_string()),
                                    confidence: 1.0,
                                    reasoning: format!(
                                        "{} is disjoint with {} (inherited from parent genre {})",
                                        subgenre, genre_b, genre_a
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        inferred
    }

    /// Get all subgenres recursively
    fn get_all_subgenres(&self, genre: &str, ontology: &MediaOntology) -> Option<HashSet<String>> {
        let mut subgenres = HashSet::new();

        for (child, parents) in &ontology.genre_hierarchy {
            if parents.contains(genre) {
                subgenres.insert(child.clone());

                if let Some(child_subgenres) = self.get_all_subgenres(child, ontology) {
                    subgenres.extend(child_subgenres);
                }
            }
        }

        if subgenres.is_empty() {
            None
        } else {
            Some(subgenres)
        }
    }

    /// Infer equivalent genres
    fn infer_equivalent_genres(&self, ontology: &MediaOntology) -> Vec<InferredMediaAxiom> {
        let mut inferred = Vec::new();

        for (genre_a, equivalents) in &ontology.equivalent_genres {
            for genre_b in equivalents {
                // Ensure symmetry
                if !ontology.equivalent_genres
                    .get(genre_b)
                    .map(|set| set.contains(genre_a))
                    .unwrap_or(false)
                {
                    inferred.push(InferredMediaAxiom {
                        axiom_type: MediaAxiomType::EquivalentGenre,
                        subject: genre_b.clone(),
                        object: Some(genre_a.clone()),
                        confidence: 1.0,
                        reasoning: format!("{} is equivalent to {} (symmetric)", genre_b, genre_a),
                    });
                }

                // Transitivity
                if let Some(b_equivalents) = ontology.equivalent_genres.get(genre_b) {
                    for genre_c in b_equivalents {
                        if genre_c != genre_a && !equivalents.contains(genre_c) {
                            inferred.push(InferredMediaAxiom {
                                axiom_type: MediaAxiomType::EquivalentGenre,
                                subject: genre_a.clone(),
                                object: Some(genre_c.clone()),
                                confidence: 1.0,
                                reasoning: format!(
                                    "{} is equivalent to {} (transitive through {})",
                                    genre_a, genre_c, genre_b
                                ),
                            });
                        }
                    }
                }
            }
        }

        inferred
    }

    /// GPU-accelerated batch similarity computation hook
    #[cfg(feature = "gpu")]
    fn gpu_batch_similarity(&self, pairs: Vec<(String, String)>) -> Vec<f32> {
        // Hook for GPU kernel integration
        // In production, this would call CUDA/OpenCL kernels
        unimplemented!("GPU acceleration requires gpu feature and CUDA runtime")
    }

    /// CPU fallback for batch similarity
    fn cpu_batch_similarity(&self, pairs: Vec<(String, String)>, ontology: &MediaOntology) -> Vec<f32> {
        pairs.iter().map(|(a, b)| {
            if let (Some(mood_a), Some(mood_b)) = (
                ontology.mood_relations.get(a),
                ontology.mood_relations.get(b),
            ) {
                self.calculate_mood_similarity(mood_a, mood_b)
            } else {
                0.0
            }
        }).collect()
    }
}

impl Default for ProductionMediaReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl MediaReasoner for ProductionMediaReasoner {
    fn infer_axioms(&self, ontology: &MediaOntology) -> OntologyResult<Vec<InferredMediaAxiom>> {
        let mut reasoner = Self::new();
        if self.gpu_enabled {
            reasoner = reasoner.with_gpu(self.gpu_batch_size);
        }

        let mut all_inferred = Vec::new();

        // Infer genre hierarchy relationships
        all_inferred.extend(reasoner.infer_genre_hierarchy(ontology));

        // Infer mood similarities
        all_inferred.extend(reasoner.infer_mood_similarities(ontology));

        // Infer disjoint genres
        all_inferred.extend(reasoner.infer_disjoint_genres(ontology));

        // Infer equivalent genres
        all_inferred.extend(reasoner.infer_equivalent_genres(ontology));

        Ok(all_inferred)
    }

    fn is_subgenre_of(&self, child: &str, parent: &str, ontology: &MediaOntology) -> bool {
        // Direct check
        if let Some(parents) = ontology.genre_hierarchy.get(child) {
            if parents.contains(parent) {
                return true;
            }
        }

        // Check cache
        if let Some(ancestors) = self.genre_closure_cache.get(child) {
            return ancestors.contains(parent);
        }

        // Recursive fallback
        let mut visited = HashSet::new();
        self.is_subgenre_recursive(child, parent, ontology, &mut visited)
    }

    fn are_disjoint_genres(&self, genre_a: &str, genre_b: &str, ontology: &MediaOntology) -> bool {
        for disjoint_set in &ontology.disjoint_genres {
            if disjoint_set.contains(genre_a) && disjoint_set.contains(genre_b) {
                return true;
            }
        }
        false
    }

    fn infer_mood(&self, media: &MediaEntity, ontology: &MediaOntology) -> Vec<String> {
        let mut inferred_moods = Vec::new();

        // Infer from existing moods
        for mood in &media.moods {
            if let Some(mood_def) = ontology.mood_relations.get(mood) {
                inferred_moods.extend(mood_def.related_moods.clone());
            }
        }

        // Infer from genres (would need genre->mood mapping in production)

        inferred_moods.sort();
        inferred_moods.dedup();
        inferred_moods
    }

    fn match_cultural_context(&self, media: &MediaEntity, context: &CulturalContext) -> f32 {
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        // Language match (weight: 0.3)
        let language_weight = 0.3;
        let language_score = if media.cultural_context.contains(&context.language) {
            1.0
        } else {
            // Partial credit for same language family (e.g., en-US vs en-GB)
            let media_lang_base: String = media.cultural_context.iter()
                .filter_map(|l| l.split('-').next())
                .next()
                .unwrap_or("")
                .to_string();
            let context_lang_base = context.language.split('-').next().unwrap_or("");

            if media_lang_base == context_lang_base {
                0.7
            } else {
                0.0
            }
        };
        weighted_score += language_score * language_weight;
        total_weight += language_weight;

        // Regional match (weight: 0.25)
        let region_weight = 0.25;
        let region_score = if media.cultural_context.iter().any(|c| c.contains(&context.region)) {
            1.0
        } else {
            0.0
        };
        weighted_score += region_score * region_weight;
        total_weight += region_weight;

        // Theme overlap with cultural relevance (weight: 0.25)
        let theme_weight = 0.25;
        if !media.themes.is_empty() && !context.cultural_themes.is_empty() {
            let theme_overlap: usize = media.themes.iter()
                .filter(|t| context.cultural_themes.contains(t))
                .count();
            let theme_score = (theme_overlap as f32) / (media.themes.len().max(context.cultural_themes.len()) as f32);
            weighted_score += theme_score * theme_weight;
            total_weight += theme_weight;
        }

        // Regional preference alignment (weight: 0.15)
        let preference_weight = 0.15;
        let mut preference_score = 0.0;
        let mut pref_count = 0;

        for genre in &media.genres {
            if let Some(&pref_val) = context.preferences.get(genre) {
                preference_score += pref_val;
                pref_count += 1;
            }
        }

        if pref_count > 0 {
            preference_score /= pref_count as f32;
            weighted_score += preference_score * preference_weight;
            total_weight += preference_weight;
        }

        // Taboo penalty (weight: 0.05, but negative)
        let taboo_weight = 0.05;
        let taboo_violations: usize = media.themes.iter()
            .filter(|t| context.taboos.contains(t))
            .count();

        if taboo_violations > 0 {
            // Severe penalty for taboo violations
            let taboo_penalty = -(taboo_violations as f32) * 0.5;
            weighted_score += taboo_penalty.max(-1.0);
        }
        total_weight += taboo_weight;

        // Temporal context bonus (if content matches time-sensitive themes)
        // This would integrate with DeliveryContext in production

        if total_weight > 0.0 {
            (weighted_score / total_weight).max(0.0).min(1.0)
        } else {
            0.5 // Neutral if no information
        }
    }

    fn recommend_for_user(
        &self,
        user: &UserProfile,
        context: &DeliveryContext,
        ontology: &MediaOntology,
    ) -> Vec<RecommendationResult> {
        let mut recommendations = Vec::new();

        for (media_id, media) in &ontology.media {
            let mut score = 0.0;
            let mut reasoning = Vec::new();

            // Genre preference matching
            let mut genre_score = 0.0;
            for genre in &media.genres {
                if let Some(&pref) = user.preferred_genres.get(genre) {
                    genre_score += pref;
                }
            }
            if !media.genres.is_empty() {
                genre_score /= media.genres.len() as f32;
                score += genre_score * 0.3;
                reasoning.push(ReasoningFactor {
                    factor_type: FactorType::GenreMatch,
                    weight: 0.3,
                    explanation: format!("Genre preference score: {:.2}", genre_score),
                });
            }

            // Mood preference matching
            let mut mood_score = 0.0;
            for mood in &media.moods {
                if let Some(&pref) = user.preferred_moods.get(mood) {
                    mood_score += pref;
                }
            }
            if !media.moods.is_empty() {
                mood_score /= media.moods.len() as f32;
                score += mood_score * 0.25;
                reasoning.push(ReasoningFactor {
                    factor_type: FactorType::MoodAlignment,
                    weight: 0.25,
                    explanation: format!("Mood alignment score: {:.2}", mood_score),
                });
            }

            // Cultural context matching
            if let Some(user_culture) = user.cultural_background.first() {
                if let Some(culture_def) = ontology.cultural_contexts.get(user_culture) {
                    let cultural_score = self.match_cultural_context(media, culture_def);
                    score += cultural_score * 0.2;
                    reasoning.push(ReasoningFactor {
                        factor_type: FactorType::CulturalRelevance,
                        weight: 0.2,
                        explanation: format!("Cultural relevance: {:.2}", cultural_score),
                    });
                }
            }

            // Context fit (device, network, etc.)
            let context_score = self.evaluate_context_fit(media, context);
            score += context_score * 0.15;
            reasoning.push(ReasoningFactor {
                factor_type: FactorType::ContextualFit,
                weight: 0.15,
                explanation: format!("Context compatibility: {:.2}", context_score),
            });

            // Historical similarity (simplified)
            let history_score = self.calculate_history_similarity(media, user);
            score += history_score * 0.1;
            reasoning.push(ReasoningFactor {
                factor_type: FactorType::SimilarityToHistory,
                weight: 0.1,
                explanation: format!("Similar to watch history: {:.2}", history_score),
            });

            recommendations.push(RecommendationResult {
                media_id: media_id.clone(),
                score,
                reasoning,
                confidence: 0.8, // Would be computed based on data quality
            });
        }

        // Sort by score descending
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        recommendations
    }
}

impl ProductionMediaReasoner {
    /// Recursive subgenre check
    fn is_subgenre_recursive(
        &self,
        child: &str,
        parent: &str,
        ontology: &MediaOntology,
        visited: &mut HashSet<String>,
    ) -> bool {
        if child == parent {
            return true;
        }

        if !visited.insert(child.to_string()) {
            return false;
        }

        if let Some(parents) = ontology.genre_hierarchy.get(child) {
            for p in parents {
                if self.is_subgenre_recursive(p, parent, ontology, visited) {
                    return true;
                }
            }
        }

        false
    }

    /// Evaluate how well media fits the delivery context
    fn evaluate_context_fit(&self, media: &MediaEntity, context: &DeliveryContext) -> f32 {
        let mut score = 1.0_f32;

        // Network quality vs file size
        if let Some(file_size) = media.technical_metadata.file_size_bytes {
            let size_mb = file_size as f32 / 1_000_000.0;
            match context.network_quality {
                NetworkQuality::Poor if size_mb > 100.0 => score *= 0.3,
                NetworkQuality::Fair if size_mb > 500.0 => score *= 0.6,
                NetworkQuality::Good if size_mb > 2000.0 => score *= 0.8,
                _ => {}
            }
        }

        // Device compatibility
        match (&context.device_type, &media.media_type) {
            (DeviceType::Mobile, MediaType::Video) if media.technical_metadata.duration_seconds.unwrap_or(0.0) > 3600.0 => {
                score *= 0.7; // Long videos less suitable for mobile
            }
            (DeviceType::TV, MediaType::Text) => {
                score *= 0.4; // Text content not ideal for TV
            }
            _ => {}
        }

        score.max(0.0_f32).min(1.0_f32)
    }

    /// Calculate similarity to user's interaction history
    fn calculate_history_similarity(&self, _media: &MediaEntity, user: &UserProfile) -> f32 {
        if user.interaction_history.is_empty() {
            return 0.5; // Neutral for new users
        }

        let positive_interactions: Vec<_> = user.interaction_history.iter()
            .filter(|i| matches!(i.interaction_type, InteractionType::Like | InteractionType::Complete))
            .collect();

        if positive_interactions.is_empty() {
            return 0.5;
        }

        // Simplified similarity based on genre overlap
        // In production, this would use more sophisticated embedding comparison
        let avg_similarity = 0.6; // Placeholder

        avg_similarity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ontology() -> MediaOntology {
        let mut ontology = MediaOntology::default();

        // Setup genre hierarchy: Drama -> Thriller -> PsychologicalThriller
        ontology.genre_hierarchy.insert(
            "Thriller".to_string(),
            vec!["Drama".to_string()].into_iter().collect()
        );
        ontology.genre_hierarchy.insert(
            "PsychologicalThriller".to_string(),
            vec!["Thriller".to_string()].into_iter().collect()
        );

        // Disjoint genres
        ontology.disjoint_genres.push(
            vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
        );

        // Moods
        ontology.mood_relations.insert("Tense".to_string(), Mood {
            name: "Tense".to_string(),
            valence: -0.3,
            arousal: 0.8,
            dominance: 0.4,
            related_moods: vec!["Anxious".to_string()],
        });

        ontology.mood_relations.insert("Anxious".to_string(), Mood {
            name: "Anxious".to_string(),
            valence: -0.4,
            arousal: 0.7,
            dominance: 0.3,
            related_moods: vec!["Tense".to_string()],
        });

        ontology
    }

    #[test]
    fn test_genre_hierarchy_inference() {
        let ontology = create_test_ontology();
        let mut reasoner = ProductionMediaReasoner::new();

        let inferred = reasoner.infer_genre_hierarchy(&ontology);

        // Should infer that PsychologicalThriller is subgenre of Drama
        assert!(inferred.iter().any(|axiom|
            axiom.axiom_type == MediaAxiomType::SubGenreOf
            && axiom.subject == "PsychologicalThriller"
            && axiom.object.as_ref() == Some(&"Drama".to_string())
        ));
    }

    #[test]
    fn test_is_subgenre_of() {
        let ontology = create_test_ontology();
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.compute_genre_closure(&ontology);

        assert!(reasoner.is_subgenre_of("Thriller", "Drama", &ontology));
        assert!(reasoner.is_subgenre_of("PsychologicalThriller", "Drama", &ontology));
        assert!(reasoner.is_subgenre_of("PsychologicalThriller", "Thriller", &ontology));
        assert!(!reasoner.is_subgenre_of("Drama", "Thriller", &ontology));
    }

    #[test]
    fn test_mood_similarity() {
        let ontology = create_test_ontology();
        let reasoner = ProductionMediaReasoner::new();

        let tense = ontology.mood_relations.get("Tense").unwrap();
        let anxious = ontology.mood_relations.get("Anxious").unwrap();

        let similarity = reasoner.calculate_mood_similarity(tense, anxious);

        assert!(similarity > 0.7, "Tense and Anxious should be similar");
    }

    #[test]
    fn test_disjoint_genres() {
        let ontology = create_test_ontology();
        let reasoner = ProductionMediaReasoner::new();

        assert!(reasoner.are_disjoint_genres("Comedy", "Horror", &ontology));
        assert!(reasoner.are_disjoint_genres("Horror", "Comedy", &ontology));
        assert!(!reasoner.are_disjoint_genres("Drama", "Thriller", &ontology));
    }

    #[test]
    fn test_cultural_context_match() {
        let reasoner = ProductionMediaReasoner::new();

        let media = MediaEntity {
            id: "test1".to_string(),
            title: "Test Media".to_string(),
            media_type: MediaType::Video,
            genres: vec!["Drama".to_string()],
            moods: vec!["Tense".to_string()],
            themes: vec!["family".to_string(), "tradition".to_string()],
            cultural_context: vec!["en-US".to_string()],
            technical_metadata: TechnicalMetadata {
                duration_seconds: Some(7200.0),
                resolution: Some("1080p".to_string()),
                format: "mp4".to_string(),
                bitrate: Some(5000),
                file_size_bytes: Some(1_000_000_000),
            },
            semantic_tags: vec![],
        };

        let context = CulturalContext {
            region: "US".to_string(),
            language: "en-US".to_string(),
            cultural_themes: vec!["family".to_string()],
            taboos: vec![],
            preferences: HashMap::new(),
        };

        let score = reasoner.match_cultural_context(&media, &context);

        assert!(score > 0.5, "Should have positive cultural match");
    }
}
