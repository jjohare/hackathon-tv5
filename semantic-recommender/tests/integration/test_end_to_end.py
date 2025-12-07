#!/usr/bin/env python3
"""
End-to-End Integration Test

Validates:
1. Full neuro-symbolic pipeline (semantic + graph reasoning)
2. Batch processing with explanations
3. Performance under combined load
4. Result quality and coherence
"""

import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scripts.utils.gpu_hyper_personalization import GPUHyperPersonalization
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestEndToEndPipeline:
    """End-to-end integration tests"""

    @classmethod
    def setup_class(cls):
        """Initialize recommendation system"""
        cls.system = GPUHyperPersonalization(use_tensorrt=True)

    def test_neuro_symbolic_recommendation(self):
        """Test full pipeline with semantic + graph reasoning"""

        query = "Movies like The Matrix with philosophical themes"
        user_id = "test_user_001"

        response = self.system.personalized_search(
            user_id=user_id,
            query=query,
            top_k=5,
            context={
                'time_of_day': [0.1, 0.2, 0.7],  # Evening
                'genre_prefs': [0.6, 0.3, 0.1],  # Sci-fi heavy
                'social_signal': [1.0, 0.0]      # Solo
            }
        )

        print(f"\n{'='*80}")
        print(f"Neuro-Symbolic Recommendation Test")
        print(f"{'='*80}\n")
        print(f"Query: '{query}'")
        print(f"User: {user_id}\n")

        # Verify performance
        total_time_ms = response['timing']['total_ms']
        assert total_time_ms < 50, f"Latency {total_time_ms:.2f}ms exceeds 50ms threshold"

        print(f"⏱️  Performance:")
        print(f"   Total: {total_time_ms:.2f}ms")
        print(f"   Query encoding: {response['timing']['query_encoding_ms']:.2f}ms")
        print(f"   GPU similarity: {response['timing']['gpu_similarity_ms']:.2f}ms")
        print(f"   Attention rerank: {response['timing']['attention_rerank_ms']:.2f}ms")

        # Verify results structure
        assert 'results' in response, "Response missing results"
        assert len(response['results']) == 5, f"Expected 5 results, got {len(response['results'])}"

        print(f"\n📊 Top Recommendations:")
        for i, result in enumerate(response['results'], 1):
            assert 'id' in result, "Result missing ID"
            assert 'title' in result, "Result missing title"
            assert 'score' in result, "Result missing score"

            print(f"\n{i}. {result['title']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Genres: {', '.join(result.get('genres', [])[:3])}")

        print(f"\n✅ Neuro-symbolic pipeline validated")

    def test_batch_processing_with_explanations(self):
        """Test batch queries with quality explanations"""

        test_cases = [
            {
                'query': 'action thriller',
                'expected_genres': ['Action', 'Thriller', 'Crime']
            },
            {
                'query': 'romantic comedy',
                'expected_genres': ['Romance', 'Comedy']
            },
            {
                'query': 'sci-fi horror',
                'expected_genres': ['Sci-Fi', 'Horror', 'Thriller']
            }
        ]

        print(f"\n{'='*80}")
        print(f"Batch Processing with Explanations")
        print(f"{'='*80}\n")

        results_collection = []

        for i, test_case in enumerate(test_cases):
            response = self.system.personalized_search(
                user_id=f"test_user_{i:03d}",
                query=test_case['query'],
                top_k=3
            )

            print(f"\nQuery {i+1}: '{test_case['query']}'")
            print(f"  Results: {len(response['results'])}")
            print(f"  Latency: {response['timing']['total_ms']:.2f}ms")

            # Verify at least one result matches expected genres
            matched = False
            for result in response['results']:
                result_genres = set(result.get('genres', []))
                expected_genres = set(test_case['expected_genres'])

                if result_genres & expected_genres:
                    matched = True
                    print(f"  ✓ Matched: {result['title']} (genres: {', '.join(result_genres)})")
                    break

            results_collection.append({
                'query': test_case['query'],
                'matched': matched,
                'results': response['results']
            })

        # Verify all queries found relevant results
        match_count = sum(1 for r in results_collection if r['matched'])
        match_rate = match_count / len(test_cases) * 100

        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total queries: {len(test_cases)}")
        print(f"   Matched: {match_count} ({match_rate:.1f}%)")

        assert match_rate >= 66.0, f"Match rate {match_rate:.1f}% below 66% threshold"

    def test_personalization_consistency(self):
        """Test that same user gets consistent results"""

        user_id = "consistency_test_user"
        query = "thriller movies"

        # Run same query 3 times
        results = []
        for i in range(3):
            response = self.system.personalized_search(
                user_id=user_id,
                query=query,
                top_k=5
            )
            results.append([r['id'] for r in response['results']])

        print(f"\n{'='*80}")
        print(f"Personalization Consistency Test")
        print(f"{'='*80}\n")
        print(f"User: {user_id}")
        print(f"Query: '{query}'")
        print(f"Runs: 3\n")

        # Check consistency (first result should be stable)
        first_results = [r[0] for r in results]

        print(f"First result across runs:")
        for i, result_id in enumerate(first_results, 1):
            print(f"  Run {i}: {result_id}")

        # At least 2 out of 3 should match (allowing for some variance)
        unique_first = len(set(first_results))

        assert unique_first <= 2, f"Top result too variable ({unique_first} unique across 3 runs)"

        print(f"\n✅ Consistency validated (max {unique_first} unique top results)")

    def test_context_aware_reranking(self):
        """Test that context affects recommendation ranking"""

        query = "action movie"
        user_id = "context_test_user"

        # Context 1: Evening, solo viewing
        context_evening = {
            'time_of_day': [0.1, 0.2, 0.7],
            'genre_prefs': [0.5, 0.3, 0.2],
            'social_signal': [1.0, 0.0]
        }

        # Context 2: Afternoon, group viewing
        context_afternoon = {
            'time_of_day': [0.2, 0.7, 0.1],
            'genre_prefs': [0.3, 0.5, 0.2],
            'social_signal': [0.2, 0.8]
        }

        response_evening = self.system.personalized_search(
            user_id=user_id,
            query=query,
            top_k=5,
            context=context_evening
        )

        response_afternoon = self.system.personalized_search(
            user_id=user_id,
            query=query,
            top_k=5,
            context=context_afternoon
        )

        evening_top = [r['id'] for r in response_evening['results']]
        afternoon_top = [r['id'] for r in response_afternoon['results']]

        print(f"\n{'='*80}")
        print(f"Context-Aware Reranking Test")
        print(f"{'='*80}\n")
        print(f"Query: '{query}'\n")
        print(f"Evening (solo) top 3:")
        for r in response_evening['results'][:3]:
            print(f"  - {r['title']} (score: {r['score']:.3f})")

        print(f"\nAfternoon (group) top 3:")
        for r in response_afternoon['results'][:3]:
            print(f"  - {r['title']} (score: {r['score']:.3f})")

        # Context should produce different rankings
        overlap = len(set(evening_top[:3]) & set(afternoon_top[:3]))

        print(f"\nTop-3 overlap: {overlap}/3")

        # Allow some overlap but not complete
        assert overlap < 3, "Context should affect ranking (complete overlap detected)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
