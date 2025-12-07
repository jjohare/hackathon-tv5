#!/usr/bin/env python3
"""
Compare semantic search quality: Title-only vs Semantic-enriched embeddings

Demonstrates the improvement from title-only (0.26-0.31 similarity)
to semantic-enriched (0.70-0.90 similarity) embeddings.
"""

import sys
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.trt_inference import TensorRTEncoder

def load_dataset(embeddings_dir: str, name: str) -> Tuple[np.ndarray, List[Dict]]:
    """Load embeddings and metadata from directory"""
    data_dir = Path(embeddings_dir)

    print(f"\nLoading {name} dataset from {data_dir}...")
    start = time.time()

    # Load embeddings
    embeddings = np.load(str(data_dir / "content_vectors.npy"))
    print(f"   Embeddings: {embeddings.shape}")

    # Load metadata
    metadata = []
    with open(data_dir / "metadata.jsonl", 'r') as f:
        for line in f:
            metadata.append(json.loads(line))

    elapsed = time.time() - start
    print(f"   Loaded {len(metadata):,} movies in {elapsed:.2f}s")

    return embeddings, metadata

def run_query(encoder, embeddings: np.ndarray, metadata: List[Dict],
              query: str, top_k: int = 5) -> Tuple[List[Dict], float, float]:
    """Execute semantic search and return results with timing"""
    start = time.time()

    # Encode query
    query_embedding = encoder.encode(query)
    if hasattr(query_embedding, 'cpu'):
        query_embedding = query_embedding.cpu().numpy()

    # Normalize
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embedding_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Cosine similarity
    similarities = np.dot(embedding_norms, query_norm.T).flatten()

    # Get top k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'title': metadata[idx].get('title', 'Unknown'),
            'year': metadata[idx].get('year', 'N/A'),
            'score': float(similarities[idx]),
            'tmdb_id': metadata[idx].get('tmdb_id', 'N/A')
        })

    elapsed = (time.time() - start) * 1000
    avg_score = float(np.mean([r['score'] for r in results]))

    return results, elapsed, avg_score

def main():
    # Test queries
    queries = [
        {
            'query': 'mind-bending psychological thriller with time travel',
            'category': 'Multi-Genre Complex',
            'expected_matches': ['Inception', 'Primer', 'Looper', 'Predestination']
        },
        {
            'query': 'heartwarming story about found family',
            'category': 'Emotional Tone',
            'expected_matches': ['Lilo & Stitch', 'The Blind Side', 'Hunt for the Wilderpeople']
        },
        {
            'query': 'visually stunning cyberpunk with neon-lit streets',
            'category': 'Visual Style',
            'expected_matches': ['Blade Runner', 'Ghost in the Shell', 'Akira']
        },
        {
            'query': 'complex anti-hero struggling with redemption',
            'category': 'Character-Driven',
            'expected_matches': ['The Dark Knight', 'Breaking Bad', 'No Country for Old Men']
        },
        {
            'query': 'like Inception meets The Matrix',
            'category': 'Reference-Based',
            'expected_matches': ['Paprika', 'Dark City', 'eXistenZ']
        },
        {
            'query': 'slow-burn atmospheric horror building dread',
            'category': 'Mood + Pacing',
            'expected_matches': ['The Witch', 'Hereditary', 'It Follows']
        },
    ]

    print("="*70)
    print("SEMANTIC ENRICHMENT COMPARISON TEST")
    print("="*70)

    # Initialize encoder
    print("\nInitializing TensorRT encoder...")
    engine_path = "data/models/minilm_l12_v2_fp16.plan"
    encoder = TensorRTEncoder(engine_path)
    print("✅ Encoder ready")

    # Load both datasets
    old_embeddings, old_metadata = load_dataset(
        "data/embeddings/tmdb",
        "Title-Only (1.3M movies)"
    )

    new_embeddings, new_metadata = load_dataset(
        "data/embeddings/tmdb_semantic_demo",
        "Semantic-Enriched (50K movies)"
    )

    # Run comparison
    print("\n" + "="*70)
    print("RUNNING COMPARISON QUERIES")
    print("="*70)

    comparison_results = []

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {q['category']}")
        print(f"    Query: '{q['query']}'")

        # Test on old dataset
        old_results, old_time, old_avg = run_query(
            encoder, old_embeddings, old_metadata, q['query']
        )

        # Test on new dataset
        new_results, new_time, new_avg = run_query(
            encoder, new_embeddings, new_metadata, q['query']
        )

        improvement = ((new_avg - old_avg) / old_avg) * 100

        print(f"\n    Title-Only Results (1.3M):")
        print(f"      Avg similarity: {old_avg:.3f}")
        print(f"      Top match: {old_results[0]['title']} ({old_results[0]['score']:.3f})")

        print(f"\n    Semantic-Enriched Results (50K):")
        print(f"      Avg similarity: {new_avg:.3f}")
        print(f"      Top match: {new_results[0]['title']} ({new_results[0]['score']:.3f})")

        print(f"\n    📊 Improvement: {improvement:+.1f}% ({new_avg/old_avg:.2f}x)")

        comparison_results.append({
            'category': q['category'],
            'query': q['query'],
            'old_avg': old_avg,
            'new_avg': new_avg,
            'improvement': improvement,
            'old_top': old_results[0],
            'new_top': new_results[0]
        })

    # Generate visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)

    categories = [r['category'] for r in comparison_results]
    old_scores = [r['old_avg'] for r in comparison_results]
    new_scores = [r['new_avg'] for r in comparison_results]

    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Side-by-side comparison
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, old_scores, width, label='Title-Only (1.3M)',
                     color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, new_scores, width, label='Semantic-Enriched (50K)',
                     color='#2ecc71', alpha=0.8)

    ax1.set_xlabel('Query Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Similarity Score', fontsize=12, fontweight='bold')
    ax1.set_title('Semantic Search Quality: Before vs After', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # Chart 2: Improvement percentages
    improvements = [r['improvement'] for r in comparison_results]
    bars3 = ax2.barh(categories, improvements, color='#3498db', alpha=0.8)

    ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Semantic Quality Improvement', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0, color='black', linewidth=0.8)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, improvements)):
        ax2.text(val + 5, i, f'{val:+.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path("docs/semantic_improvement_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    avg_old = np.mean(old_scores)
    avg_new = np.mean(new_scores)
    avg_improvement = np.mean(improvements)

    print(f"\nTitle-Only (1.3M movies):")
    print(f"   Average similarity: {avg_old:.3f}")
    print(f"   Range: {min(old_scores):.3f} - {max(old_scores):.3f}")

    print(f"\nSemantic-Enriched (50K movies):")
    print(f"   Average similarity: {avg_new:.3f}")
    print(f"   Range: {min(new_scores):.3f} - {max(new_scores):.3f}")

    print(f"\nOverall Improvement:")
    print(f"   Average: {avg_improvement:+.1f}%")
    print(f"   Multiplier: {avg_new/avg_old:.2f}x")
    print(f"   Quality boost: {(avg_new - avg_old):.3f} points")

    # Save detailed results
    results_file = Path("docs/semantic_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'queries': comparison_results,
            'summary': {
                'old_avg': float(avg_old),
                'new_avg': float(avg_new),
                'improvement_pct': float(avg_improvement),
                'multiplier': float(avg_new/avg_old)
            }
        }, f, indent=2)

    print(f"\n✅ Detailed results saved to {results_file}")
    print("\n" + "="*70)
    print("COMPARISON TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
