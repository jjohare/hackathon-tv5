#!/usr/bin/env python3
"""
Parse MovieLens datasets into structured JSONL format
Handles: movies, ratings, tags, genome scores, links
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import sys

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "ml-25m"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure output directories exist
for subdir in ["media", "users", "interactions"]:
    (PROCESSED_DIR / subdir).mkdir(parents=True, exist_ok=True)


def extract_year_from_title(title: str) -> Tuple[str, Optional[int]]:
    """Extract year from movie title like 'Toy Story (1995)'"""
    match = re.match(r'(.+?)\s*\((\d{4})\)\s*$', title)
    if match:
        clean_title = match.group(1).strip()
        year = int(match.group(2))
        return clean_title, year
    return title.strip(), None


def classify_time_of_day(timestamp: int) -> str:
    """Classify timestamp into time of day period"""
    dt = datetime.fromtimestamp(timestamp)
    hour = dt.hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    elif 21 <= hour < 24:
        return "night"
    else:
        return "late_night"


def get_day_of_week(timestamp: int) -> str:
    """Get day of week from timestamp"""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%A")


def parse_movies():
    """Parse movies.csv into structured media assets"""
    print("📽️  Parsing movies.csv...")

    # Read movies
    movies_df = pd.read_csv(RAW_DIR / "movies.csv")

    # Read links for external IDs
    links_df = pd.read_csv(RAW_DIR / "links.csv")
    links_dict = {
        row['movieId']: {
            'imdb_id': f"tt{int(row['imdbId']):07d}" if pd.notna(row['imdbId']) else None,
            'tmdb_id': int(row['tmdbId']) if pd.notna(row['tmdbId']) else None
        }
        for _, row in links_df.iterrows()
    }

    # Process each movie
    media_assets = []
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        title, year = extract_year_from_title(row['title'])
        genres = row['genres'].split('|') if row['genres'] != '(no genres listed)' else []

        # Get external IDs
        external_ids = links_dict.get(movie_id, {})

        asset = {
            'media_id': f"ml_{movie_id}",
            'identifiers': {
                'movielens_id': int(movie_id),
                'imdb_id': external_ids.get('imdb_id'),
                'tmdb_id': external_ids.get('tmdb_id'),
                'internal_id': f"media_{movie_id:08d}"
            },
            'metadata': {
                'title': title,
                'original_title': row['title'],
                'year': year or 0,
                'language': 'en',  # Default
                'country': ['US']   # Default
            },
            'classification': {
                'genres': genres,
                'themes': [],  # Will populate from genome
                'moods': [],
                'tags': []
            }
        }
        media_assets.append(asset)

    # Save as JSONL
    output_path = PROCESSED_DIR / "media" / "movies.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for asset in media_assets:
            f.write(json.dumps(asset, ensure_ascii=False) + '\n')

    print(f"✅ Processed {len(media_assets)} movies → {output_path}")
    return len(media_assets)


def parse_genome_tags():
    """Parse genome tags and scores for semantic enrichment"""
    print("🧬 Parsing genome data...")

    # Load tags
    tags_df = pd.read_csv(RAW_DIR / "genome-tags.csv")
    tag_map = dict(zip(tags_df['tagId'], tags_df['tag']))

    # Process scores in chunks (large file - 416MB)
    print("   Loading genome scores (this may take a minute)...")
    scores_df = pd.read_csv(RAW_DIR / "genome-scores.csv")

    # Group by movie and filter strong associations
    movie_genomes = {}
    for movie_id, group in scores_df.groupby('movieId'):
        genome = {
            tag_map[row['tagId']]: float(row['relevance'])
            for _, row in group.iterrows()
            if row['relevance'] > 0.5  # Only strong associations
        }
        if genome:  # Only save if there are strong tags
            movie_genomes[str(movie_id)] = genome

    # Save as JSON
    output_path = PROCESSED_DIR / "media" / "genome_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(movie_genomes, f, indent=2)

    print(f"✅ Processed genome for {len(movie_genomes)} movies → {output_path}")
    print(f"   Total tags: {len(tag_map)}, Movies with genome: {len(movie_genomes)}")
    return len(movie_genomes)


def parse_interactions():
    """Parse ratings into interaction events"""
    print("⭐ Parsing ratings.csv...")

    output_path = PROCESSED_DIR / "interactions" / "ratings.jsonl"

    # Process in chunks for memory efficiency (647MB file)
    chunksize = 1_000_000
    chunks = pd.read_csv(RAW_DIR / "ratings.csv", chunksize=chunksize)

    total_processed = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk_idx, df in enumerate(chunks):
            for idx, row in df.iterrows():
                interaction = {
                    'interaction_id': f"int_{total_processed:010d}",
                    'user_id': f"user_{int(row['userId']):08d}",
                    'media_id': f"ml_{int(row['movieId'])}",
                    'timestamp': int(row['timestamp']),
                    'iso_timestamp': datetime.fromtimestamp(row['timestamp']).isoformat(),
                    'interaction_type': 'Rate',
                    'rating': float(row['rating']),
                    'context': {
                        'device': 'unknown',
                        'time_of_day': classify_time_of_day(row['timestamp']),
                        'day_of_week': get_day_of_week(row['timestamp'])
                    },
                    'feedback': {
                        'explicit_rating': float(row['rating'])
                    }
                }
                f.write(json.dumps(interaction) + '\n')
                total_processed += 1

            print(f"   Processed chunk {chunk_idx+1} ({total_processed:,} ratings)")

    print(f"✅ Processed {total_processed:,} interactions → {output_path}")
    return total_processed


def parse_tags():
    """Parse user-applied tags"""
    print("🏷️  Parsing tags.csv...")

    df = pd.read_csv(RAW_DIR / "tags.csv")

    tag_interactions = []
    for idx, row in df.iterrows():
        # Normalize tag (lowercase, replace special chars)
        # Handle potential NaN values in tags
        tag_value = row['tag']
        if pd.isna(tag_value):
            continue
        normalized_tag = re.sub(r'[^a-z0-9_]', '_', str(tag_value).lower())

        interaction = {
            'interaction_id': f"tag_{idx:010d}",
            'user_id': f"user_{int(row['userId']):08d}",
            'media_id': f"ml_{int(row['movieId'])}",
            'timestamp': int(row['timestamp']),
            'iso_timestamp': datetime.fromtimestamp(row['timestamp']).isoformat(),
            'interaction_type': 'Tag',
            'tag': row['tag'],
            'tag_normalized': normalized_tag
        }
        tag_interactions.append(interaction)

    # Save as JSONL
    output_path = PROCESSED_DIR / "interactions" / "tags.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for interaction in tag_interactions:
            f.write(json.dumps(interaction, ensure_ascii=False) + '\n')

    print(f"✅ Processed {len(tag_interactions):,} tag applications → {output_path}")
    return len(tag_interactions)


def generate_stats():
    """Generate summary statistics"""
    print("\n📊 Summary Statistics:")

    stats = {
        'movies': 0,
        'ratings': 0,
        'tags': 0,
        'genome_movies': 0,
        'unique_users': set(),
        'date_range': {'min': None, 'max': None}
    }

    # Count movies
    with open(PROCESSED_DIR / "media" / "movies.jsonl") as f:
        stats['movies'] = sum(1 for _ in f)

    # Count ratings and extract user IDs
    with open(PROCESSED_DIR / "interactions" / "ratings.jsonl") as f:
        for line in f:
            data = json.loads(line)
            stats['ratings'] += 1
            stats['unique_users'].add(data['user_id'])

            # Track date range
            ts = data['timestamp']
            if stats['date_range']['min'] is None or ts < stats['date_range']['min']:
                stats['date_range']['min'] = ts
            if stats['date_range']['max'] is None or ts > stats['date_range']['max']:
                stats['date_range']['max'] = ts

    # Count tags
    with open(PROCESSED_DIR / "interactions" / "tags.jsonl") as f:
        stats['tags'] = sum(1 for _ in f)

    # Count genome movies
    with open(PROCESSED_DIR / "media" / "genome_scores.json") as f:
        genome_data = json.load(f)
        stats['genome_movies'] = len(genome_data)

    # Convert dates
    min_date = datetime.fromtimestamp(stats['date_range']['min']).strftime('%Y-%m-%d')
    max_date = datetime.fromtimestamp(stats['date_range']['max']).strftime('%Y-%m-%d')

    print(f"   Movies: {stats['movies']:,}")
    print(f"   Ratings: {stats['ratings']:,}")
    print(f"   Tags: {stats['tags']:,}")
    print(f"   Unique Users: {len(stats['unique_users']):,}")
    print(f"   Movies with Genome: {stats['genome_movies']:,} ({stats['genome_movies']/stats['movies']*100:.1f}%)")
    print(f"   Date Range: {min_date} to {max_date}")

    # Save stats
    stats['unique_users'] = len(stats['unique_users'])
    stats['date_range']['min_iso'] = min_date
    stats['date_range']['max_iso'] = max_date

    stats_path = PROCESSED_DIR / "processing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n💾 Stats saved to {stats_path}")


def main():
    """Main execution"""
    print("=" * 60)
    print("🎬 MovieLens Dataset Parser")
    print("=" * 60)
    print(f"Input:  {RAW_DIR}")
    print(f"Output: {PROCESSED_DIR}")
    print("=" * 60)
    print()

    try:
        # Check if input directory exists
        if not RAW_DIR.exists():
            print(f"❌ Error: Input directory not found: {RAW_DIR}")
            print("   Please ensure MovieLens dataset is downloaded.")
            sys.exit(1)

        # Parse all data
        parse_movies()
        parse_genome_tags()
        parse_interactions()
        parse_tags()

        # Generate statistics
        generate_stats()

        print("\n" + "=" * 60)
        print("✅ PARSING COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
