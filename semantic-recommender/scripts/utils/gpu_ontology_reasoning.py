#!/usr/bin/env python3
"""
GPU-Accelerated Ontology Reasoning for Semantic Recommender

Integrates intelligent graph distance reasoning with GPU-accelerated similarity
for hybrid semantic+ontology recommendations.

Architecture:
1. GPU: Fast semantic similarity (PyTorch CUDA)
2. Graph: Intelligent graph distance reasoning (replaces naive Jaccard)
3. Hybrid: Filter-then-Boost strategy with adaptive weighting

Changes from V1 (naive Jaccard):
- Replaced Jaccard overlap with graph shortest paths
- Added path-based explanations
- Adaptive weighting based on graph proximity
- Query expansion using ontology

Performance Target:
- GPU similarity: 0.1-0.5 ms (measured)
- Graph reasoning: <5 ms (Dijkstra or CUDA SSSP)
- Total hybrid: <10 ms (production ready)
"""

import torch
import numpy as np
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Import graph distance reasoner
try:
    from .graph_distance_reasoner import GraphDistanceReasoner
    GRAPH_REASONER_AVAILABLE = True
except ImportError:
    GRAPH_REASONER_AVAILABLE = False
    print("⚠️  Graph distance reasoner not available, falling back to Jaccard")

class GPUOntologyReasoner:
    """
    Hybrid GPU semantic + ontology reasoning system

    Uses:
    - PyTorch GPU for fast embedding similarity
    - Whelk-rs (Rust) for EL++ ontology reasoning
    - Weighted hybrid scoring for final ranking
    """

    def __init__(self, base_path="/home/devuser/semantic-recommender"):
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 80)
        print("GPU-Accelerated Ontology Reasoning System V2")
        print("=" * 80)
        print(f"\nDevice: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Load embeddings (GPU)
        self.load_embeddings()

        # Load ontology mappings (CPU)
        self.load_ontology_mappings()

        # Initialize graph distance reasoner
        if GRAPH_REASONER_AVAILABLE:
            print("\n🧠 Initializing Graph Distance Reasoner...")
            self.graph_reasoner = GraphDistanceReasoner(base_path=str(base_path))
            self.use_graph_reasoning = True
            print("✅ Graph reasoning enabled")
        else:
            self.graph_reasoner = None
            self.use_graph_reasoning = False
            print("⚠️  Graph reasoning disabled (using Jaccard fallback)")

        # Scoring weights (DEPRECATED - now using adaptive weighting)
        # Kept for backward compatibility
        self.weights = {
            'semantic': 0.7,     # Semantic similarity weight
            'ontology': 0.2,     # Ontology concept matching weight (deprecated)
            'genre': 0.1         # Genre overlap weight (deprecated)
        }

    def load_embeddings(self):
        """Load embeddings to GPU"""
        print("\n" + "=" * 80)
        print("Loading Embeddings to GPU")
        print("=" * 80)

        media_path = self.base_path / "data/embeddings/media"

        # Load media vectors
        media_vectors = np.load(media_path / "content_vectors.npy")
        self.media_embeddings_gpu = torch.from_numpy(media_vectors).to(self.device)
        print(f"✅ Media embeddings: {media_vectors.shape} on {self.device}")

        # Load metadata
        self.media_metadata = {}
        self.media_ids = []
        with open(media_path / "metadata.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                media_id = item['media_id']
                self.media_ids.append(media_id)
                self.media_metadata[media_id] = item

        print(f"✅ Metadata: {len(self.media_metadata)} movies")

        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"✅ GPU Memory: {allocated:.2f} GB allocated")

    def load_ontology_mappings(self):
        """
        Load ontology concept mappings

        Maps MovieLens genome tags to AdA film ontology concepts
        """
        print("\n" + "=" * 80)
        print("Loading Ontology Mappings")
        print("=" * 80)

        # Genome tag → AdA ontology concept mapping
        # Expanded mapping based on actual MovieLens genome tags (1127 total)
        # Covers visual style, narrative, themes, cinematography, editing, sound
        self.genome_to_ada = {
            # Visual Style & Cinematography
            'dark': ['ada:DarkLighting', 'ada:HighContrast'],
            'noir': ['ada:FilmNoirStyle', 'ada:ShadowsAndLight'],
            'colorful': ['ada:SaturatedColor', 'ada:BrightLighting'],
            'visually appealing': ['ada:HighProductionValue', 'ada:AestheticComposition'],
            'cinematography': ['ada:HighProductionValue', 'ada:ArtisticCinematography'],
            'beautiful': ['ada:VisualBeauty', 'ada:AestheticComposition'],

            # Camera Work & Shot Composition
            'tracking shot': ['ada:TrackingShot', 'ada:FluidCameraMovement'],
            'close-up': ['ada:CloseUpShot', 'ada:IntimateFraming'],
            'long take': ['ada:LongTake', 'ada:ContinuousShot'],
            'handheld camera': ['ada:HandheldCamera', 'ada:DynamicCamerawork'],

            # Editing & Pacing
            'fast-paced': ['ada:RapidEditing', 'ada:ShortAverageShotLength'],
            'slow': ['ada:SlowPacing', 'ada:LongTakes'],
            'non-linear': ['ada:NonLinearNarrative', 'ada:ComplexTemporalStructure'],
            'flashback': ['ada:FlashbackNarrative', 'ada:TemporalDisplacement'],

            # Sound & Music
            'atmospheric': ['ada:AtmosphericSound', 'ada:AmbientSoundDesign'],
            'good soundtrack': ['ada:MemorableScore', 'ada:MusicDriven'],
            'dialogue': ['ada:DialogueDriven', 'ada:VerbalNarrative'],
            'talky': ['ada:DialogueHeavy', 'ada:VerbalNarrative'],

            # Narrative Structure & Complexity
            'cerebral': ['movies:IntellectualFilm', 'movies:ComplexNarrative'],
            'philosophical': ['movies:PhilosophicalThemes', 'movies:ExistentialContent'],
            'twist ending': ['movies:PlotTwist', 'movies:SurpriseRevelation'],
            'complex': ['movies:ComplexNarrative', 'movies:LayeredStoryline'],
            'storytelling': ['movies:NarrativeExcellence', 'movies:CompellingStory'],
            'enigmatic': ['movies:Mysterious', 'movies:AmbiguousNarrative'],
            'predictable': ['movies:ConventionalPlot', 'movies:FormulaicNarrative'],

            # Character & Psychology
            'character study': ['movies:CharacterDriven', 'movies:PsychologicalDepth'],
            'intimate': ['movies:PersonalStory', 'movies:EmotionalDepth'],
            'relationships': ['movies:RelationshipDriven', 'movies:InterpersonalDynamics'],
            'mentor': ['movies:MentorRelationship', 'movies:GuidanceFigure'],
            'obsession': ['movies:ObsessiveCharacter', 'movies:PsychologicalIntensity'],

            # Mood & Atmosphere
            'suspenseful': ['ada:Suspense', 'ada:TensionBuilding'],
            'suspense': ['ada:Suspense', 'ada:TensionBuilding'],
            'intense': ['movies:HighEmotionalIntensity', 'ada:Suspense'],
            'melancholic': ['movies:MelancholicMood', 'movies:Sadness'],
            'dramatic': ['movies:DramaticIntensity', 'movies:EmotionalWeight'],
            'weird': ['movies:Surreal', 'movies:Unconventional'],

            # Intellectual & Thematic Depth
            'thought-provoking': ['movies:ThoughtProvoking', 'movies:IntellectuallyStimulating'],
            'social commentary': ['movies:SocialCritique', 'movies:PoliticalThemes'],
            'life philosophy': ['movies:PhilosophicalReflection', 'movies:ExistentialThemes'],

            # Action & Violence
            'action': ['movies:ActionSequences', 'ada:HighActionDensity'],
            'chase': ['movies:ChaseSequence', 'ada:DynamicMovement'],
            'violence': ['movies:GraphicViolence', 'movies:ViolentContent'],
            'brutality': ['movies:IntenseViolence', 'movies:HarshRealism'],

            # Emotional Tone
            'romantic': ['movies:RomanticThemes', 'movies:LoveStory'],
            'comedy': ['movies:ComedyGenre', 'movies:HumorousTone'],
            'fun movie': ['movies:Entertaining', 'movies:LightHearted'],
            'great ending': ['movies:SatisfyingConclusion', 'movies:MemorableEnding'],

            # Crime & Corruption
            'murder': ['movies:MurderPlot', 'movies:CrimeDrama'],
            'corruption': ['movies:CorruptionTheme', 'movies:MoralDecay'],
            'vengeance': ['movies:RevengeTheme', 'movies:Retribution'],
            'betrayal': ['movies:BetrayalTheme', 'movies:Treachery'],
            'greed': ['movies:GreedTheme', 'movies:MaterialismCritique'],

            # Existential & Psychological
            'loneliness': ['movies:IsolationTheme', 'movies:Alienation'],
            'destiny': ['movies:FateTheme', 'movies:Determinism'],
            'runaway': ['movies:EscapeNarrative', 'movies:Journey'],

            # Production Quality Indicators
            'criterion': ['movies:ArtHouseCinema', 'movies:CriticallyAcclaimed'],
            'imdb top 250': ['movies:HighlyRated', 'movies:ClassicCinema'],
            'great acting': ['movies:ExceptionalPerformances', 'ada:StrongActing'],
            'great movie': ['movies:CinematicExcellence', 'movies:HighQuality'],
            'good': ['movies:WellExecuted', 'movies:Quality'],
            'great': ['movies:Outstanding', 'movies:Exceptional'],

            # Specific Styles
            'independent film': ['movies:IndependentCinema', 'movies:ArtistControl'],
            'original': ['movies:OriginalStory', 'movies:Innovative'],
            'culture clash': ['movies:CulturalConflict', 'movies:CrossCultural'],
            'pg-13': ['movies:MainstreamAudience', 'movies:ModerateContent'],
        }

        print(f"✅ Loaded {len(self.genome_to_ada)} genome tag mappings")

        # Load genome scores if available
        genome_path = self.base_path / "data/processed/media/genome_scores.json"
        if genome_path.exists():
            with open(genome_path, 'r') as f:
                self.genome_scores = json.load(f)
            print(f"✅ Loaded genome scores for {len(self.genome_scores)} movies")
        else:
            self.genome_scores = {}
            print("⚠️  No genome scores found")

        # Movie → ontology classes mapping
        self.movie_ontology_classes = {}
        self._build_ontology_classes()

    def _build_ontology_classes(self):
        """Build movie → ontology classes mapping"""
        print("\n🔨 Building movie ontology class mappings...")

        for media_id in self.media_ids:
            classes = set()

            # Map genres to ontology
            metadata = self.media_metadata[media_id]
            for genre in metadata.get('genres', []):
                classes.add(f"movies:{genre}Genre")

            # Map genome tags to AdA concepts (if available)
            # genome_scores uses plain IDs like "1", "2", "3"
            # metadata uses "ml_1", "ml_2", "ml_3"
            # Strip "ml_" prefix to match genome_scores keys
            genome_id = media_id.replace('ml_', '') if media_id.startswith('ml_') else media_id

            if genome_id in self.genome_scores:
                genome_tags = self.genome_scores[genome_id]
                for tag, score in genome_tags.items():
                    # Lowered threshold from 0.7 to 0.5 for better coverage
                    # Median score is 0.654, mean is 0.684, so 0.5 captures meaningful tags
                    if score > 0.5 and tag in self.genome_to_ada:
                        classes.update(self.genome_to_ada[tag])

            self.movie_ontology_classes[media_id] = list(classes)

        mapped_count = sum(1 for classes in self.movie_ontology_classes.values() if classes)
        avg_classes = np.mean([len(classes) for classes in self.movie_ontology_classes.values()])

        print(f"✅ Mapped {mapped_count}/{len(self.media_ids)} movies to ontology")
        print(f"✅ Average {avg_classes:.1f} classes per movie")

    def gpu_semantic_similarity(self, query_id: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Fast GPU semantic similarity search

        Returns:
            List of (media_id, similarity_score) tuples
        """
        # Get query vector
        query_idx = self.media_ids.index(query_id)
        query_vector = self.media_embeddings_gpu[query_idx]

        # Normalize
        query_norm = query_vector / torch.norm(query_vector)

        # Compute similarities (GPU)
        start = time.time()
        similarities = torch.matmul(self.media_embeddings_gpu, query_norm)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        gpu_time = (time.time() - start) * 1000

        # Get top-k
        top_k_vals, top_k_indices = torch.topk(similarities, k=top_k)

        # Convert to list
        results = []
        for idx, sim in zip(top_k_indices.cpu().numpy(), top_k_vals.cpu().numpy()):
            media_id = self.media_ids[idx]
            results.append((media_id, float(sim)))

        return results, gpu_time

    def ontology_similarity(self, query_id: str, candidate_id: str) -> float:
        """
        Compute ontology-based similarity

        Uses Jaccard similarity on ontology classes
        """
        query_classes = set(self.movie_ontology_classes.get(query_id, []))
        candidate_classes = set(self.movie_ontology_classes.get(candidate_id, []))

        if not query_classes or not candidate_classes:
            return 0.0

        intersection = len(query_classes & candidate_classes)
        union = len(query_classes | candidate_classes)

        return intersection / union if union > 0 else 0.0

    def genre_similarity(self, query_id: str, candidate_id: str) -> float:
        """Compute genre overlap similarity"""
        query_genres = set(self.media_metadata[query_id].get('genres', []))
        candidate_genres = set(self.media_metadata[candidate_id].get('genres', []))

        if not query_genres or not candidate_genres:
            return 0.0

        intersection = len(query_genres & candidate_genres)
        union = len(query_genres | candidate_genres)

        return intersection / union if union > 0 else 0.0

    def hybrid_recommend(
        self,
        query_id: str,
        top_k: int = 10,
        semantic_candidates: int = 100,
        ontology_context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Hybrid recommendation combining:
        1. GPU semantic similarity (FAST: ~0.5ms)
        2. Graph distance reasoning (INTELLIGENT: ~5ms)
        3. Filter-then-Boost strategy (ADAPTIVE)

        V2 Changes:
        - Replaced naive Jaccard with graph shortest paths
        - Adaptive weighting based on graph proximity
        - Path-based explanations
        - Optional filtering based on ontology context

        Args:
            query_id: Movie ID to find similar movies for
            top_k: Number of recommendations to return
            semantic_candidates: Number of semantic candidates to re-rank
            ontology_context: Optional context for filtering (user preferences, constraints)

        Returns:
            List of recommendation dicts with scores and explanations
        """
        print(f"\n🎬 Query: {self.media_metadata[query_id]['title']}")

        # Step 1: GPU semantic similarity (fast)
        semantic_results, gpu_time = self.gpu_semantic_similarity(query_id, semantic_candidates)
        print(f"⚡ GPU semantic search: {gpu_time:.3f} ms ({semantic_candidates} candidates)")

        # Step 2: Graph reasoning or fallback to Jaccard
        start = time.time()

        if self.use_graph_reasoning:
            # V2: Intelligent graph distance reasoning
            print("🧠 Using graph distance reasoning...")

            # Convert semantic results to candidate format
            candidates = []
            for candidate_id, sem_score in semantic_results:
                if candidate_id == query_id:
                    continue

                candidates.append({
                    'media_id': candidate_id,
                    'title': self.media_metadata[candidate_id]['title'],
                    'semantic_score': sem_score,
                    'genres': self.media_metadata[candidate_id].get('genres', [])
                })

            # Apply filter-then-boost strategy
            hybrid_scores = self.graph_reasoner.filter_then_boost(
                query_movie_id=query_id,
                candidates=candidates,
                ontology_context=ontology_context or {}
            )

        else:
            # V1: Fallback to naive Jaccard overlap (DEPRECATED)
            print("⚠️  Using Jaccard fallback (graph reasoning unavailable)...")

            hybrid_scores = []

            for candidate_id, sem_score in semantic_results:
                # Skip self
                if candidate_id == query_id:
                    continue

                # Compute all similarity components
                onto_score = self.ontology_similarity(query_id, candidate_id)
                genre_score = self.genre_similarity(query_id, candidate_id)

                # Hybrid score (naive linear combination)
                final_score = (
                    self.weights['semantic'] * sem_score +
                    self.weights['ontology'] * onto_score +
                    self.weights['genre'] * genre_score
                )

                # Explanation
                query_classes = self.movie_ontology_classes.get(query_id, [])
                candidate_classes = self.movie_ontology_classes.get(candidate_id, [])
                shared_classes = list(set(query_classes) & set(candidate_classes))

                hybrid_scores.append({
                    'media_id': candidate_id,
                    'title': self.media_metadata[candidate_id]['title'],
                    'final_score': final_score,
                    'semantic_score': sem_score,
                    'ontology_score': onto_score,
                    'genre_score': genre_score,
                    'shared_ontology_classes': shared_classes[:5],  # Top 5
                    'genres': self.media_metadata[candidate_id].get('genres', []),
                    'reasoning': 'Jaccard overlap (deprecated)'
                })

            hybrid_scores.sort(key=lambda x: x['final_score'], reverse=True)

        reasoning_time = (time.time() - start) * 1000

        if self.use_graph_reasoning:
            print(f"🧠 Graph reasoning: {reasoning_time:.3f} ms")
        else:
            print(f"⚙️  Jaccard reasoning: {reasoning_time:.3f} ms")

        # Return top-k
        results = hybrid_scores[:top_k]

        total_time = gpu_time + reasoning_time
        print(f"✅ Total time: {total_time:.3f} ms (GPU: {gpu_time:.1f}ms, Reasoning: {reasoning_time:.1f}ms)")

        return results, {
            'gpu_time_ms': gpu_time,
            'reasoning_time_ms': reasoning_time,
            'total_time_ms': total_time,
            'reasoning_method': 'graph_distance' if self.use_graph_reasoning else 'jaccard'
        }

    def explain_recommendation(self, query_id: str, candidate_id: str) -> str:
        """
        Generate human-readable explanation

        V2: Uses graph reasoning when available for path-based explanations
        """
        if self.use_graph_reasoning and self.graph_reasoner:
            # Use graph distance reasoning for explanation
            return self.graph_reasoner.explain_recommendation(query_id, candidate_id)

        # V1 Fallback: Jaccard-based explanation (deprecated)
        query_meta = self.media_metadata[query_id]
        candidate_meta = self.media_metadata[candidate_id]

        # Shared genres
        query_genres = set(query_meta.get('genres', []))
        candidate_genres = set(candidate_meta.get('genres', []))
        shared_genres = query_genres & candidate_genres

        # Shared ontology classes
        query_classes = set(self.movie_ontology_classes.get(query_id, []))
        candidate_classes = set(self.movie_ontology_classes.get(candidate_id, []))
        shared_classes = query_classes & candidate_classes

        # Build explanation
        parts = []

        if shared_genres:
            parts.append(f"Shared genres: {', '.join(shared_genres)}")

        if shared_classes:
            # Group by prefix
            ada_classes = [c for c in shared_classes if c.startswith('ada:')]
            movie_classes = [c for c in shared_classes if c.startswith('movies:')]

            if ada_classes:
                parts.append(f"Film techniques: {', '.join(ada_classes[:3])}")
            if movie_classes:
                parts.append(f"Themes: {', '.join(movie_classes[:3])}")

        return " | ".join(parts) if parts else "Semantic similarity"


def demo_hybrid_reasoning():
    """
    Demonstration of GPU-accelerated hybrid reasoning
    """
    print("\n" + "=" * 80)
    print("DEMO: GPU-Accelerated Ontology Reasoning")
    print("=" * 80)

    reasoner = GPUOntologyReasoner()

    # Find a movie with rich genome tags
    test_movie = None
    for media_id in reasoner.media_ids[:1000]:
        if media_id in reasoner.genome_scores and reasoner.movie_ontology_classes[media_id]:
            test_movie = media_id
            break

    if not test_movie:
        test_movie = reasoner.media_ids[0]

    print(f"\n📽️  Test Movie: {reasoner.media_metadata[test_movie]['title']}")
    print(f"   Genres: {reasoner.media_metadata[test_movie].get('genres', [])}")
    print(f"   Ontology Classes: {len(reasoner.movie_ontology_classes[test_movie])}")

    # Get hybrid recommendations
    results, timing = reasoner.hybrid_recommend(test_movie, top_k=10)

    print(f"\n📋 Top 10 Hybrid Recommendations:")
    print(f"{'Rank':<6} {'Title':<50} {'Score':<8} {'Sem':<6} {'Onto':<6} {'Genre':<6}")
    print("-" * 90)

    for i, rec in enumerate(results, 1):
        print(f"{i:<6} {rec['title'][:48]:<50} {rec['final_score']:.3f}   "
              f"{rec['semantic_score']:.3f}  {rec['ontology_score']:.3f}  "
              f"{rec['genre_score']:.3f}")

        if rec['shared_ontology_classes']:
            print(f"       → {', '.join(rec['shared_ontology_classes'][:3])}")

    print(f"\n⏱️  Performance:")
    print(f"   GPU Semantic: {timing['gpu_time_ms']:.3f} ms")
    print(f"   CPU Ontology: {timing['cpu_time_ms']:.3f} ms")
    print(f"   Total: {timing['total_time_ms']:.3f} ms")

    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")


if __name__ == "__main__":
    demo_hybrid_reasoning()
