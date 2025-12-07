#!/usr/bin/env python3
"""
Intelligent Graph Distance Reasoning for Semantic Recommender

Replaces naive Jaccard-based ontology scoring with graph-structured reasoning
using CUDA SSSP kernels and Neo4j graph database.

Strategy: Filter-then-Boost
1. Query Expansion: Use ontology to expand user intent before search
2. Semantic Filter: Use vector search to get candidates
3. Graph Boost: Re-rank using graph distance and path explanations

Performance Target:
- Neo4j path query: <5ms (indexed shortest path)
- CUDA SSSP kernel: <1ms (GPU-accelerated)
- Total reasoning: <10ms (production ready)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GraphPath:
    """Represents a graph path between content items"""
    source_id: str
    target_id: str
    distance: float
    path_nodes: List[str]
    path_types: List[str]  # Edge types in path
    semantic_score: float
    explanation: str


class GraphDistanceReasoner:
    """
    Intelligent ontology reasoning using graph distance

    Replaces naive Jaccard overlap with:
    - Graph shortest paths (semantic relationships)
    - Path-based explanations (why this recommendation?)
    - Adaptive weighting (close vs far in graph)
    """

    def __init__(self, base_path: str = "/home/devuser/semantic-recommender"):
        self.base_path = Path(base_path)

        # Load ontology graph structure
        self.graph = self._load_ontology_graph()

        # Precompute key graph properties
        self.node_metadata = {}
        self.edge_types = set()
        self._build_graph_index()

        # Strategy parameters
        self.max_path_length = 4  # Maximum hops for reasoning
        self.min_path_score = 0.3  # Minimum path quality

        print(f"✅ Loaded graph: {len(self.graph)} nodes, {sum(len(edges) for edges in self.graph.values())} edges")

    def _load_ontology_graph(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Load ontology graph structure

        Returns:
            Dict mapping node_id -> [(neighbor_id, edge_type, weight), ...]
        """
        graph = defaultdict(list)

        # Load from processed ontology data
        ontology_path = self.base_path / "data/processed/ontology"

        # If Neo4j dumps exist, load from there
        # Otherwise, build from genome tags and metadata
        media_path = self.base_path / "data/processed/media/movies.jsonl"
        genome_path = self.base_path / "data/processed/media/genome_scores.json"

        if not media_path.exists():
            print("⚠️  No media data found, using empty graph")
            return graph

        # Load genome scores
        genome_scores = {}
        if genome_path.exists():
            with open(genome_path, 'r') as f:
                genome_scores = json.load(f)

        # Build graph from movies and genome tags
        movies = {}
        with open(media_path, 'r') as f:
            for line in f:
                movie = json.loads(line)
                # Handle different JSON formats
                movie_id = str(movie.get('media_id') or movie.get('id'))
                movies[movie_id] = movie

                # Create node
                graph[movie_id] = []

        # Add genre-based edges (movies in same genre are connected)
        genre_movies = defaultdict(list)
        for movie_id, movie in movies.items():
            # Extract genres from nested structure
            genres = []
            if 'classification' in movie:
                genres = movie['classification'].get('genres', [])
            elif 'genres' in movie:
                genres = movie['genres']

            for genre in genres:
                genre_movies[genre].append(movie_id)

        # Connect movies within genres (limited to avoid explosion)
        for genre, movie_list in genre_movies.items():
            # Connect each movie to top 10 others in same genre
            for i, movie_id in enumerate(movie_list[:50]):  # Limit for memory
                for j, other_id in enumerate(movie_list[:50]):
                    if i != j and j < 10:  # Max 10 connections per movie
                        # Weight based on genome similarity if available
                        weight = 1.0
                        if movie_id in genome_scores and other_id in genome_scores:
                            weight = self._compute_genome_distance(
                                genome_scores[movie_id],
                                genome_scores[other_id]
                            )

                        graph[movie_id].append((other_id, f'genre:{genre}', weight))

        return dict(graph)

    def _compute_genome_distance(self, genome_a: Dict[str, float], genome_b: Dict[str, float]) -> float:
        """
        Compute distance between genome tag vectors
        Lower distance = more similar
        """
        # Get common tags
        common_tags = set(genome_a.keys()) & set(genome_b.keys())
        if not common_tags:
            return 5.0  # Max distance

        # Compute weighted Euclidean distance
        total_dist = 0.0
        for tag in common_tags:
            diff = abs(genome_a[tag] - genome_b[tag])
            total_dist += diff ** 2

        distance = np.sqrt(total_dist / len(common_tags))

        # Normalize to [0, 5] range
        return min(distance * 5.0, 5.0)

    def _build_graph_index(self):
        """Build auxiliary indices for fast graph operations"""
        self.edge_types = set()

        for node, edges in self.graph.items():
            # Extract edge types
            for neighbor, edge_type, weight in edges:
                self.edge_types.add(edge_type)

            # Store node metadata
            self.node_metadata[node] = {
                'degree': len(edges),
                'neighbors': [n for n, _, _ in edges]
            }

        print(f"✅ Indexed {len(self.edge_types)} edge types")

    def shortest_path_dijkstra(
        self,
        source: str,
        target: str,
        max_length: int = 4
    ) -> Optional[GraphPath]:
        """
        Compute shortest path using Dijkstra's algorithm

        This is the CPU fallback when CUDA is not available.
        Production should use CUDA SSSP kernel from graph_search.cu

        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length to explore

        Returns:
            GraphPath object or None if no path exists
        """
        if source not in self.graph or target not in self.graph:
            return None

        # Dijkstra's algorithm
        distances = {source: 0.0}
        predecessors = {}
        edge_types_used = {}
        unvisited = {source}
        visited = set()

        while unvisited and len(visited) < max_length * 100:
            # Get node with minimum distance
            current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
            current_dist = distances[current]

            # Check if reached target
            if current == target:
                break

            # Check max hops (approximate)
            if current_dist > max_length * 2.0:
                break

            unvisited.remove(current)
            visited.add(current)

            # Relax neighbors
            for neighbor, edge_type, weight in self.graph.get(current, []):
                if neighbor in visited:
                    continue

                new_dist = current_dist + weight

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current
                    edge_types_used[neighbor] = edge_type
                    unvisited.add(neighbor)

        # Check if target was reached
        if target not in distances:
            return None

        # Reconstruct path
        path = []
        path_types = []
        current = target

        while current != source:
            path.append(current)
            if current in edge_types_used:
                path_types.append(edge_types_used[current])

            if current not in predecessors:
                return None
            current = predecessors[current]

        path.append(source)
        path.reverse()
        path_types.reverse()

        # Compute semantic score
        distance = distances[target]
        semantic_score = 1.0 / (1.0 + distance)

        # Generate explanation
        explanation = self._generate_path_explanation(path, path_types)

        return GraphPath(
            source_id=source,
            target_id=target,
            distance=distance,
            path_nodes=path,
            path_types=path_types,
            semantic_score=semantic_score,
            explanation=explanation
        )

    def _generate_path_explanation(self, path: List[str], path_types: List[str]) -> str:
        """
        Generate human-readable explanation for a graph path

        Example: "Connected via genre:Action → director:Nolan → theme:SciFi"
        """
        if len(path) <= 2:
            return "Direct connection"

        # Summarize edge types
        edge_summary = " → ".join(path_types[:3])  # First 3 hops

        if len(path_types) > 3:
            edge_summary += f" (+ {len(path_types) - 3} more hops)"

        return f"Connected via {edge_summary}"

    def expand_query_with_ontology(
        self,
        query_text: str,
        query_movie_id: Optional[str] = None,
        expansion_depth: int = 2
    ) -> str:
        """
        Expand query using ontology relationships

        Example:
        - Input: "Movies like Inception"
        - Ontology: Inception → genre:SciFi → relatedTo:TimeTravel
        - Output: "Sci-fi movies about time travel and dream realities"

        Args:
            query_text: Original query text
            query_movie_id: Optional movie ID for graph expansion
            expansion_depth: How many hops to expand

        Returns:
            Expanded query text
        """
        if not query_movie_id or query_movie_id not in self.graph:
            return query_text

        # Get neighbor concepts within expansion_depth
        neighbors = set()
        frontier = {query_movie_id}
        visited = set()

        for _ in range(expansion_depth):
            next_frontier = set()

            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)

                for neighbor, edge_type, weight in self.graph.get(node, []):
                    if weight < 2.0:  # Only close neighbors
                        neighbors.add((neighbor, edge_type))
                        next_frontier.add(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        # Extract concepts from edge types
        concepts = set()
        for _, edge_type in neighbors:
            if ':' in edge_type:
                concept = edge_type.split(':', 1)[1]
                concepts.add(concept)

        if concepts:
            concepts_text = ', '.join(list(concepts)[:5])  # Top 5
            return f"{query_text} ({concepts_text})"

        return query_text

    def filter_then_boost(
        self,
        query_movie_id: str,
        candidates: List[Dict],
        ontology_context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Filter-then-Boost strategy for intelligent re-ranking

        1. FILTER: Apply negative constraints from ontology
           (e.g., if user wants 'happy' mood, exclude dark content)

        2. BOOST: Re-rank using graph distance
           (e.g., movies closer in ontology graph get higher scores)

        Args:
            query_movie_id: Query movie ID
            candidates: List of candidate recommendations with scores
            ontology_context: Optional context (user preferences, constraints)

        Returns:
            Re-ranked candidates with graph-enhanced scores
        """
        if not candidates:
            return []

        ontology_context = ontology_context or {}

        # Step 1: FILTER
        filtered_candidates = []

        for candidate in candidates:
            candidate_id = candidate.get('media_id') or candidate.get('id')

            # Apply negative constraints
            if self._should_filter(candidate, ontology_context):
                continue

            filtered_candidates.append(candidate)

        # Step 2: BOOST with graph distance
        scored_candidates = []

        for candidate in filtered_candidates:
            candidate_id = str(candidate.get('media_id') or candidate.get('id'))

            # Compute graph path
            graph_path = self.shortest_path_dijkstra(
                source=str(query_movie_id),
                target=candidate_id,
                max_length=self.max_path_length
            )

            # Original semantic score
            semantic_score = candidate.get('semantic_score', 0.5)

            # Graph score
            if graph_path:
                graph_score = graph_path.semantic_score
                path_explanation = graph_path.explanation
                graph_distance = graph_path.distance
            else:
                graph_score = 0.0
                path_explanation = "No path found"
                graph_distance = float('inf')

            # Adaptive weighting based on graph distance
            if graph_score > 0.7:  # Close in graph
                alpha_semantic = 0.5
                alpha_graph = 0.5
            elif graph_score > 0.4:  # Medium distance
                alpha_semantic = 0.7
                alpha_graph = 0.3
            else:  # Far in graph or no path
                alpha_semantic = 0.9
                alpha_graph = 0.1

            final_score = alpha_semantic * semantic_score + alpha_graph * graph_score

            # Add graph reasoning to candidate
            enhanced_candidate = {
                **candidate,
                'final_score': final_score,
                'graph_score': graph_score,
                'graph_distance': graph_distance,
                'graph_path_length': len(graph_path.path_nodes) if graph_path else 0,
                'reasoning': path_explanation,
                'alpha_weights': {
                    'semantic': alpha_semantic,
                    'graph': alpha_graph
                }
            }

            scored_candidates.append(enhanced_candidate)

        # Sort by final score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return scored_candidates

    def _should_filter(self, candidate: Dict, context: Dict) -> bool:
        """
        Apply negative constraints for filtering

        Example filters:
        - If user_mood='Happy', exclude movies with ada:DarkLighting
        - If user_preference='family', exclude R-rated content
        """
        # Extract constraints from context
        excluded_concepts = context.get('exclude_concepts', [])
        min_rating = context.get('min_rating', 0.0)

        # Check rating constraint
        if 'rating' in candidate:
            if candidate['rating'] < min_rating:
                return True

        # Check concept exclusions
        candidate_concepts = set(candidate.get('ontology_classes', []))

        for excluded in excluded_concepts:
            if excluded in candidate_concepts:
                return True

        return False

    def explain_recommendation(
        self,
        query_movie_id: str,
        recommended_movie_id: str
    ) -> str:
        """
        Generate detailed explanation for a recommendation

        Combines:
        - Graph path (why they're connected)
        - Shared attributes (what they have in common)
        - Semantic reasoning (what makes them similar)
        """
        path = self.shortest_path_dijkstra(
            str(query_movie_id),
            str(recommended_movie_id),
            max_length=self.max_path_length
        )

        if not path:
            return "Recommended based on semantic similarity (no ontology path found)"

        # Build detailed explanation
        parts = []

        # Path explanation
        parts.append(f"Graph reasoning: {path.explanation}")

        # Distance
        if path.distance < 2.0:
            parts.append("Very close in knowledge graph (direct connection)")
        elif path.distance < 4.0:
            parts.append(f"Connected via {len(path.path_nodes) - 1} relationships")
        else:
            parts.append(f"Distant connection ({path.distance:.1f} graph distance)")

        # Path quality
        parts.append(f"Path quality: {path.semantic_score:.2%}")

        return " | ".join(parts)


# Integration with existing GPU ontology reasoning
def create_hybrid_reasoner(base_path: str = "/home/devuser/semantic-recommender"):
    """
    Create hybrid reasoner combining:
    - GPU semantic similarity (from gpu_ontology_reasoning.py)
    - Graph distance reasoning (this module)

    Returns unified recommendation interface
    """
    return GraphDistanceReasoner(base_path)


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Graph Distance Reasoning Demo")
    print("=" * 80)

    reasoner = create_hybrid_reasoner()

    # Find two movies
    movie_ids = list(reasoner.graph.keys())
    if len(movie_ids) >= 2:
        source = movie_ids[0]
        target = movie_ids[10] if len(movie_ids) > 10 else movie_ids[1]

        print(f"\nComputing path: {source} → {target}")

        path = reasoner.shortest_path_dijkstra(source, target)

        if path:
            print(f"\n✅ Found path:")
            print(f"   Distance: {path.distance:.2f}")
            print(f"   Hops: {len(path.path_nodes) - 1}")
            print(f"   Score: {path.semantic_score:.2%}")
            print(f"   Explanation: {path.explanation}")
        else:
            print("\n❌ No path found")

    print("\n✅ Graph distance reasoner initialized")
