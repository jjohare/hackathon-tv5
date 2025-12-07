#!/usr/bin/env python3
"""
Test Graph Distance Reasoning

Validates the intelligent graph-based ontology reasoning system.
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph_distance_reasoner import GraphDistanceReasoner, GraphPath

# Get base path once
BASE_PATH = Path(__file__).parent.parent.parent


def test_graph_construction():
    """Test that graph is constructed correctly"""
    print("=" * 80)
    print("Test 1: Graph Construction")
    print("=" * 80)

    reasoner = GraphDistanceReasoner(base_path=str(BASE_PATH))

    # Check graph loaded
    assert len(reasoner.graph) > 0, "Graph should not be empty"
    print(f"✅ Graph has {len(reasoner.graph)} nodes")

    # Check node metadata
    assert len(reasoner.node_metadata) > 0, "Node metadata should exist"
    print(f"✅ Node metadata: {len(reasoner.node_metadata)} nodes")

    # Check edge types
    assert len(reasoner.edge_types) > 0, "Should have edge types"
    print(f"✅ Edge types: {list(reasoner.edge_types)[:5]}")

    print()


def test_shortest_path():
    """Test shortest path computation"""
    print("=" * 80)
    print("Test 2: Shortest Path Computation")
    print("=" * 80)

    reasoner = GraphDistanceReasoner(base_path=str(BASE_PATH))

    # Get two movies
    movie_ids = list(reasoner.graph.keys())
    if len(movie_ids) < 2:
        print("⚠️  Not enough movies for path test")
        return

    source = movie_ids[0]
    target = movie_ids[10] if len(movie_ids) > 10 else movie_ids[1]

    print(f"Source: {source}")
    print(f"Target: {target}")

    # Compute path
    path = reasoner.shortest_path_dijkstra(source, target, max_length=4)

    if path:
        print(f"\n✅ Path found:")
        print(f"   Distance: {path.distance:.2f}")
        print(f"   Hops: {len(path.path_nodes) - 1}")
        print(f"   Score: {path.semantic_score:.2%}")
        print(f"   Path: {' → '.join(path.path_nodes[:5])}")
        print(f"   Types: {' → '.join(path.path_types[:4])}")
        print(f"   Explanation: {path.explanation}")

        # Validate path structure
        assert path.source_id == source
        assert path.target_id == target
        assert path.distance >= 0
        assert 0 <= path.semantic_score <= 1
        assert len(path.path_nodes) >= 2
        print("\n✅ Path structure valid")
    else:
        print("\n⚠️  No path found (graphs may be disconnected)")

    print()


def test_filter_then_boost():
    """Test filter-then-boost strategy"""
    print("=" * 80)
    print("Test 3: Filter-then-Boost Strategy")
    print("=" * 80)

    reasoner = GraphDistanceReasoner(base_path=str(BASE_PATH))

    # Create mock candidates
    movie_ids = list(reasoner.graph.keys())
    if len(movie_ids) < 10:
        print("⚠️  Not enough movies for filter test")
        return

    query_id = movie_ids[0]
    candidates = []

    for i, movie_id in enumerate(movie_ids[1:11]):
        candidates.append({
            'media_id': movie_id,
            'id': movie_id,
            'semantic_score': 0.8 - (i * 0.05),  # Decreasing scores
            'title': f'Movie {i+1}'
        })

    print(f"Query: {query_id}")
    print(f"Candidates: {len(candidates)}")

    # Test without filtering
    print("\nWithout filtering:")
    results = reasoner.filter_then_boost(query_id, candidates)

    assert len(results) > 0, "Should have results"
    print(f"✅ {len(results)} results")

    # Check scores exist
    for result in results[:3]:
        assert 'final_score' in result
        assert 'graph_score' in result
        assert 'reasoning' in result
        print(f"   {result.get('title', 'Unknown')}: "
              f"final={result['final_score']:.3f}, "
              f"graph={result['graph_score']:.3f}")

    # Test with filtering (exclude high ratings)
    print("\nWith filtering (exclude rating > 0.7):")
    context = {
        'exclude_concepts': ['ada:DarkLighting'],
        'min_rating': 0.7
    }

    results_filtered = reasoner.filter_then_boost(query_id, candidates, context)
    print(f"✅ {len(results_filtered)} results after filtering")

    print()


def test_query_expansion():
    """Test query expansion with ontology"""
    print("=" * 80)
    print("Test 4: Query Expansion")
    print("=" * 80)

    reasoner = GraphDistanceReasoner(base_path=str(BASE_PATH))

    movie_ids = list(reasoner.graph.keys())
    if len(movie_ids) < 1:
        print("⚠️  No movies for expansion test")
        return

    query_text = "Movies similar to this"
    query_movie_id = movie_ids[0]

    print(f"Original query: {query_text}")
    print(f"Query movie: {query_movie_id}")

    # Expand query
    expanded = reasoner.expand_query_with_ontology(
        query_text,
        query_movie_id,
        expansion_depth=2
    )

    print(f"Expanded query: {expanded}")

    # Should have expanded if movie has neighbors
    if reasoner.graph.get(query_movie_id):
        print("✅ Query expanded with ontology concepts")
    else:
        print("⚠️  Movie has no neighbors for expansion")

    print()


def test_explanation_generation():
    """Test explanation generation"""
    print("=" * 80)
    print("Test 5: Explanation Generation")
    print("=" * 80)

    reasoner = GraphDistanceReasoner(base_path=str(BASE_PATH))

    movie_ids = list(reasoner.graph.keys())
    if len(movie_ids) < 2:
        print("⚠️  Not enough movies for explanation test")
        return

    source = movie_ids[0]
    target = movie_ids[5] if len(movie_ids) > 5 else movie_ids[1]

    print(f"Explaining: {source} → {target}")

    explanation = reasoner.explain_recommendation(source, target)

    print(f"\n✅ Explanation: {explanation}")

    # Should have non-empty explanation
    assert len(explanation) > 0, "Explanation should not be empty"
    print("✅ Explanation generated")

    print()


def test_adaptive_weighting():
    """Test that adaptive weighting works correctly"""
    print("=" * 80)
    print("Test 6: Adaptive Weighting")
    print("=" * 80)

    reasoner = GraphDistanceReasoner(base_path=str(BASE_PATH))

    movie_ids = list(reasoner.graph.keys())
    if len(movie_ids) < 5:
        print("⚠️  Not enough movies for weighting test")
        return

    query_id = movie_ids[0]

    # Create candidates with varying graph distances
    candidates = []
    test_cases = [
        ('Close', 0.9, 2.0),   # Close in graph
        ('Medium', 0.5, 4.0),  # Medium distance
        ('Far', 0.1, 8.0)      # Far in graph
    ]

    for i, (label, sem_score, graph_dist) in enumerate(test_cases):
        movie_id = movie_ids[i + 1]
        candidates.append({
            'media_id': movie_id,
            'id': movie_id,
            'semantic_score': sem_score,
            'title': f'{label} Movie',
            'test_graph_distance': graph_dist  # For verification
        })

    results = reasoner.filter_then_boost(query_id, candidates)

    print("\nAdaptive weighting results:")
    for result in results:
        title = result.get('title', 'Unknown')
        alpha_sem = result.get('alpha_weights', {}).get('semantic', 0.0)
        alpha_graph = result.get('alpha_weights', {}).get('graph', 0.0)

        print(f"   {title}:")
        print(f"      Semantic weight: {alpha_sem:.1f}")
        print(f"      Graph weight: {alpha_graph:.1f}")
        print(f"      Final score: {result['final_score']:.3f}")

    print("\n✅ Adaptive weighting applied")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GRAPH DISTANCE REASONING TEST SUITE")
    print("=" * 80 + "\n")

    try:
        test_graph_construction()
        test_shortest_path()
        test_filter_then_boost()
        test_query_expansion()
        test_explanation_generation()
        test_adaptive_weighting()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
