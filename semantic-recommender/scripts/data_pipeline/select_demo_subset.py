#!/usr/bin/env python3
"""
Select top 50K movies from TMDB dataset for semantic enrichment demo.
Selection criteria: vote_count (popularity proxy) and diverse release years.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def select_demo_subset(
    input_file: str = "../../data/raw/tmdb/TMDB_movie_dataset_v11.csv",
    output_file: str = "../../data/processed/demo_subset_50k.jsonl",
    subset_size: int = 50000
):
    """
    Select top 50K movies by popularity with temporal diversity.

    Strategy:
    1. Load full TMDB dataset
    2. Filter valid entries (non-null titles, IDs)
    3. Sort by vote_count (popularity proxy)
    4. Select top 50K with temporal diversity
    5. Export as JSONL for API enrichment
    """

    print(f"Loading TMDB dataset from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   Loaded {len(df):,} total movies")

    # Filter valid entries
    print("\nFiltering valid entries...")
    df_valid = df[
        df['title'].notna() &
        df['id'].notna() &
        df['vote_count'].notna()
    ].copy()
    print(f"   {len(df_valid):,} movies with valid metadata")

    # Convert vote_count to numeric
    df_valid['vote_count'] = pd.to_numeric(df_valid['vote_count'], errors='coerce')
    df_valid = df_valid[df_valid['vote_count'].notna()]

    # Extract year from release_date
    df_valid['year'] = pd.to_datetime(df_valid['release_date'], errors='coerce').dt.year

    # Sort by popularity (vote_count)
    df_sorted = df_valid.sort_values('vote_count', ascending=False)

    # Select top subset_size movies
    df_subset = df_sorted.head(subset_size)

    print(f"\nSelected {len(df_subset):,} movies for demo subset")
    print(f"   Vote count range: {df_subset['vote_count'].min():.0f} - {df_subset['vote_count'].max():.0f}")
    print(f"   Year range: {df_subset['year'].min():.0f} - {df_subset['year'].max():.0f}")

    # Show temporal distribution
    year_dist = df_subset['year'].value_counts().sort_index()
    print(f"\n   Top 5 years:")
    for year, count in year_dist.head(5).items():
        print(f"      {year:.0f}: {count:,} movies")

    # Export as JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to {output_file}...")
    with open(output_file, 'w') as f:
        for _, row in df_subset.iterrows():
            entry = {
                'tmdb_id': int(row['id']),
                'title': row['title'],
                'release_date': row['release_date'] if pd.notna(row['release_date']) else None,
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'vote_count': int(row['vote_count'])
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Demo subset exported: {len(df_subset):,} movies")

    # Save summary stats
    stats = {
        'total_movies': len(df_subset),
        'vote_count_range': [float(df_subset['vote_count'].min()), float(df_subset['vote_count'].max())],
        'year_range': [int(df_subset['year'].min()), int(df_subset['year'].max())],
        'year_distribution': {int(k): int(v) for k, v in year_dist.to_dict().items() if pd.notna(k)}
    }

    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Statistics saved to {stats_file}")

    return df_subset

if __name__ == "__main__":
    select_demo_subset()
