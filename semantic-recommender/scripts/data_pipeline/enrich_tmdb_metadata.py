#!/usr/bin/env python3
"""
Enrich TMDB movies with full metadata via TMDB API.
Fetches: overview, tagline, keywords, genres, cast (top 5), director.

Rate limit: 50 req/sec (TMDB free tier)
Expected time: ~20-30 minutes for 50K movies
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
import os
from tqdm import tqdm

class TMDBEnricher:
    def __init__(self, api_key: str, use_bearer_token: bool = False):
        self.api_key = api_key
        self.use_bearer_token = use_bearer_token
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()

        # Set authentication header for Bearer token (v4 API)
        if use_bearer_token:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json;charset=utf-8'
            })

        # Rate limiting: 50 req/sec = 20ms per request
        self.min_request_interval = 0.021  # 21ms to be safe
        self.last_request_time = 0

        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': {}
        }

    def _rate_limit(self):
        """Enforce rate limit of 50 req/sec."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def fetch_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch full movie details from TMDB API."""
        self._rate_limit()

        try:
            # Fetch movie details
            url = f"{self.base_url}/movie/{tmdb_id}"

            # Use Bearer token or API key
            if self.use_bearer_token:
                params = {'append_to_response': 'credits,keywords'}
            else:
                params = {
                    'api_key': self.api_key,
                    'append_to_response': 'credits,keywords'
                }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 404:
                self.stats['failed'] += 1
                return None

            response.raise_for_status()
            data = response.json()

            # Extract rich metadata
            enriched = {
                'tmdb_id': tmdb_id,
                'title': data.get('title', ''),
                'overview': data.get('overview', ''),
                'tagline': data.get('tagline', ''),
                'genres': [g['name'] for g in data.get('genres', [])],
                'keywords': [k['name'] for k in data.get('keywords', {}).get('keywords', [])],
                'release_date': data.get('release_date', ''),
                'vote_average': data.get('vote_average', 0),
                'vote_count': data.get('vote_count', 0)
            }

            # Extract top 5 cast members
            credits = data.get('credits', {})
            cast = credits.get('cast', [])
            enriched['cast'] = [c['name'] for c in cast[:5]]

            # Extract director
            crew = credits.get('crew', [])
            directors = [c['name'] for c in crew if c.get('job') == 'Director']
            enriched['director'] = directors[0] if directors else None

            self.stats['successful'] += 1
            return enriched

        except requests.exceptions.RequestException as e:
            error_type = type(e).__name__
            self.stats['errors'][error_type] = self.stats['errors'].get(error_type, 0) + 1
            self.stats['failed'] += 1
            return None

    def enrich_dataset(
        self,
        input_file: str,
        output_file: str,
        checkpoint_interval: int = 1000
    ):
        """Enrich full dataset with metadata."""

        # Load input dataset
        print(f"Loading input dataset from {input_file}...")
        with open(input_file, 'r') as f:
            movies = [json.loads(line) for line in f]

        print(f"   Loaded {len(movies):,} movies to enrich")

        # Check for existing progress
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_file = output_file.replace('.jsonl', '_checkpoint.json')
        processed_ids = set()

        if os.path.exists(checkpoint_file):
            print(f"\nLoading checkpoint from {checkpoint_file}...")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_ids = set(checkpoint_data['processed_ids'])
                self.stats = checkpoint_data.get('stats', self.stats)
            print(f"   Resuming from {len(processed_ids):,} processed movies")

        # Open output file in append mode
        mode = 'a' if processed_ids else 'w'

        print(f"\nEnriching movies via TMDB API...")
        print(f"   Rate limit: 50 req/sec (~{self.min_request_interval*1000:.0f}ms per request)")
        print(f"   Checkpoint interval: every {checkpoint_interval:,} movies\n")

        with open(output_file, mode) as out_f:
            for i, movie in enumerate(tqdm(movies, desc="Enriching")):
                tmdb_id = movie['tmdb_id']

                # Skip if already processed
                if tmdb_id in processed_ids:
                    continue

                self.stats['total_processed'] += 1

                # Fetch metadata
                enriched = self.fetch_movie_details(tmdb_id)

                if enriched:
                    # Preserve original fields
                    enriched['original_title'] = movie.get('title')
                    enriched['year'] = movie.get('year')

                    # Write to output
                    out_f.write(json.dumps(enriched) + '\n')
                    out_f.flush()

                processed_ids.add(tmdb_id)

                # Checkpoint
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_file, processed_ids)

        # Final checkpoint
        self._save_checkpoint(checkpoint_file, processed_ids)

        # Print summary
        print("\n" + "="*70)
        print("ENRICHMENT COMPLETE")
        print("="*70)
        print(f"   Total processed: {self.stats['total_processed']:,}")
        print(f"   Successful: {self.stats['successful']:,}")
        print(f"   Failed: {self.stats['failed']:,}")

        if self.stats['errors']:
            print(f"\n   Errors by type:")
            for error_type, count in self.stats['errors'].items():
                print(f"      {error_type}: {count:,}")

        print(f"\n✅ Enriched dataset saved to {output_file}")

    def _save_checkpoint(self, checkpoint_file: str, processed_ids: set):
        """Save checkpoint for resumability."""
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed_ids': list(processed_ids),
                'stats': self.stats,
                'timestamp': time.time()
            }, f)

def main():
    # Get API key/token from environment
    api_key = os.getenv('TMDB_API_KEY') or os.getenv('TMDB_BEARER_TOKEN')

    if not api_key:
        print("❌ Error: TMDB_API_KEY or TMDB_BEARER_TOKEN environment variable not set")
        print("\nTo set your API key (v3):")
        print("   export TMDB_API_KEY='your_api_key_here'")
        print("\nOr Bearer token (v4):")
        print("   export TMDB_BEARER_TOKEN='your_bearer_token_here'")
        print("\nGet credentials at: https://www.themoviedb.org/settings/api")
        return

    # Detect if using Bearer token (starts with 'eyJ' which is JWT)
    use_bearer = api_key.startswith('eyJ')

    if use_bearer:
        print("Detected Bearer Token (v4 API)")
    else:
        print("Detected API Key (v3 API)")

    enricher = TMDBEnricher(api_key, use_bearer_token=use_bearer)

    enricher.enrich_dataset(
        input_file="../../data/processed/demo_subset_50k.jsonl",
        output_file="../../data/processed/demo_subset_50k_enriched.jsonl",
        checkpoint_interval=1000
    )

if __name__ == "__main__":
    main()
