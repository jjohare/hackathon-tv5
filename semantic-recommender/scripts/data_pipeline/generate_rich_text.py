#!/usr/bin/env python3
"""
Generate rich text from TMDB metadata for semantic embeddings.
Combines: title + overview + tagline + keywords + genres + cast.
"""

import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

def generate_rich_text(movie: Dict) -> str:
    """
    Generate rich semantic text from movie metadata.

    Template: "{title}. {tagline}. {overview}.
               Genres: {genres}. Keywords: {keywords}.
               Starring: {cast}. Directed by: {director}."

    Fallback strategy if fields are missing:
    1. Full metadata → complete template
    2. No overview → title + tagline + keywords + genres
    3. No tagline → title + overview + keywords + genres
    4. Minimal → title only
    """

    title = movie.get('title', '').strip()
    overview = movie.get('overview', '').strip()
    tagline = movie.get('tagline', '').strip()
    genres = movie.get('genres', [])
    keywords = movie.get('keywords', [])
    cast = movie.get('cast', [])
    director = movie.get('director', '')

    # Build rich text parts
    parts = []

    # Title (always include)
    if title:
        parts.append(title)

    # Tagline (if available)
    if tagline:
        parts.append(tagline)

    # Overview/plot (primary semantic content)
    if overview:
        parts.append(overview)

    # Genres
    if genres:
        genre_text = ", ".join(genres[:5])  # Top 5 genres
        parts.append(f"Genres: {genre_text}")

    # Keywords (important for semantic matching)
    if keywords:
        keyword_text = ", ".join(keywords[:10])  # Top 10 keywords
        parts.append(f"Keywords: {keyword_text}")

    # Cast (recognition factor)
    if cast:
        cast_text = ", ".join(cast[:5])  # Top 5 cast
        parts.append(f"Starring: {cast_text}")

    # Director
    if director:
        parts.append(f"Directed by: {director}")

    # Join all parts with proper spacing
    rich_text = ". ".join(parts)

    # Ensure proper ending
    if rich_text and not rich_text.endswith('.'):
        rich_text += '.'

    return rich_text

def process_dataset(
    input_file: str,
    output_file: str,
    stats_file: str = None
):
    """Process enriched dataset to generate rich text."""

    print(f"Loading enriched dataset from {input_file}...")
    with open(input_file, 'r') as f:
        movies = [json.loads(line) for line in f]

    print(f"   Loaded {len(movies):,} enriched movies")

    # Statistics
    stats = {
        'total': len(movies),
        'with_overview': 0,
        'with_tagline': 0,
        'with_keywords': 0,
        'with_genres': 0,
        'with_cast': 0,
        'with_director': 0,
        'avg_text_length': 0,
        'min_text_length': float('inf'),
        'max_text_length': 0
    }

    print(f"\nGenerating rich text...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text_lengths = []

    with open(output_file, 'w') as out_f:
        for movie in tqdm(movies, desc="Processing"):
            # Generate rich text
            rich_text = generate_rich_text(movie)

            # Update statistics
            if movie.get('overview'):
                stats['with_overview'] += 1
            if movie.get('tagline'):
                stats['with_tagline'] += 1
            if movie.get('keywords'):
                stats['with_keywords'] += 1
            if movie.get('genres'):
                stats['with_genres'] += 1
            if movie.get('cast'):
                stats['with_cast'] += 1
            if movie.get('director'):
                stats['with_director'] += 1

            text_len = len(rich_text)
            text_lengths.append(text_len)
            stats['min_text_length'] = min(stats['min_text_length'], text_len)
            stats['max_text_length'] = max(stats['max_text_length'], text_len)

            # Create output entry
            output_entry = {
                'tmdb_id': movie.get('tmdb_id'),
                'title': movie.get('title'),
                'text': rich_text,
                'release_date': movie.get('release_date'),
                'vote_average': movie.get('vote_average'),
                'vote_count': movie.get('vote_count')
            }

            out_f.write(json.dumps(output_entry) + '\n')

    # Calculate average text length
    if text_lengths:
        stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)

    # Print summary
    print("\n" + "="*70)
    print("RICH TEXT GENERATION COMPLETE")
    print("="*70)
    print(f"   Total movies: {stats['total']:,}")
    print(f"\n   Coverage:")
    print(f"      With overview: {stats['with_overview']:,} ({stats['with_overview']/stats['total']*100:.1f}%)")
    print(f"      With tagline: {stats['with_tagline']:,} ({stats['with_tagline']/stats['total']*100:.1f}%)")
    print(f"      With keywords: {stats['with_keywords']:,} ({stats['with_keywords']/stats['total']*100:.1f}%)")
    print(f"      With genres: {stats['with_genres']:,} ({stats['with_genres']/stats['total']*100:.1f}%)")
    print(f"      With cast: {stats['with_cast']:,} ({stats['with_cast']/stats['total']*100:.1f}%)")
    print(f"      With director: {stats['with_director']:,} ({stats['with_director']/stats['total']*100:.1f}%)")
    print(f"\n   Text length:")
    print(f"      Average: {stats['avg_text_length']:.0f} characters")
    print(f"      Min: {stats['min_text_length']:,} characters")
    print(f"      Max: {stats['max_text_length']:,} characters")

    print(f"\n✅ Rich text saved to {output_file}")

    # Save stats
    if stats_file:
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Statistics saved to {stats_file}")

    return stats

if __name__ == "__main__":
    process_dataset(
        input_file="../../data/processed/demo_subset_50k_enriched.jsonl",
        output_file="../../data/processed/demo_subset_50k_rich_text.jsonl",
        stats_file="../../data/processed/demo_subset_50k_rich_text_stats.json"
    )
