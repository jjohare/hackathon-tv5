#!/usr/bin/env python3
"""
Stage 1b: TMDB API Enrichment - Fetch Full Semantic Metadata

Fetches complete TMDB metadata for 1.3M movies using TMDB API v3:
- Overview, tagline, keywords, genres, cast (top 5), director, crew
- Rate limiting: 50 req/sec (TMDB API limit)
- Checkpointing: Save every 10K movies (resumable)
- Graceful fallbacks: If API fails, use existing data

Input:  data/processed/tmdb/movies_clean.jsonl (1.3M movies with tmdb_ids)
Output: data/processed/tmdb/movies_enriched.jsonl

Performance Target: ~7-8 hours for 1.3M movies with rate limiting
"""

import os
import sys
import json
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TMDB API Configuration
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_RATE_LIMIT_CALLS = 50  # 50 requests per second
TMDB_RATE_LIMIT_PERIOD = 1  # 1 second


class TMDBEnrichment:
    """TMDB API enrichment pipeline for semantic metadata."""

    def __init__(
        self,
        base_path: str = None,
        api_key: str = None,
        checkpoint_interval: int = 10000,
        max_retries: int = 3
    ):
        """
        Initialize TMDB enrichment pipeline.

        Args:
            base_path: Base directory for semantic-recommender (auto-detected if None)
            api_key: TMDB API key (reads from TMDB_API_KEY env var if None)
            checkpoint_interval: Save checkpoint every N movies
            max_retries: Maximum API retry attempts
        """
        if base_path is None:
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent.parent
        else:
            self.base_path = Path(base_path)

        # API Key
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDB API key required. Set TMDB_API_KEY environment variable or pass api_key parameter.\n"
                "Get your API key from: https://www.themoviedb.org/settings/api"
            )

        # Paths
        self.input_path = self.base_path / "data/processed/tmdb/movies_clean.jsonl"
        self.output_dir = self.base_path / "data/processed/tmdb"
        self.output_path = self.output_dir / "movies_enriched.jsonl"
        self.checkpoint_path = self.output_dir / "enrichment_checkpoint.json"

        # Parameters
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries

        # Statistics
        self.stats = {
            'total_movies': 0,
            'processed_movies': 0,
            'enriched_movies': 0,
            'api_failures': 0,
            'rate_limit_hits': 0,
            'api_calls': 0,
            'checkpoints_saved': 0,
            'processing_time': 0
        }

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'SemanticRecommender/1.0'
        })

    def create_directories(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {self.output_dir}")

    @sleep_and_retry
    @limits(calls=TMDB_RATE_LIMIT_CALLS, period=TMDB_RATE_LIMIT_PERIOD)
    def fetch_movie_details(self, tmdb_id: str) -> Optional[Dict]:
        """
        Fetch full movie details from TMDB API with rate limiting.

        Args:
            tmdb_id: TMDB movie ID

        Returns:
            Movie details dictionary or None if failed
        """
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {
            'api_key': self.api_key,
            'append_to_response': 'keywords,credits'
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                self.stats['api_calls'] += 1

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit hit - should be rare with decorator
                    self.stats['rate_limit_hits'] += 1
                    wait_time = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"⚠️  Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    # Movie not found - skip
                    return None
                else:
                    logger.warning(f"⚠️  API error {response.status_code} for movie {tmdb_id}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️  Request error for movie {tmdb_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        self.stats['api_failures'] += 1
        return None

    def extract_enriched_data(self, api_response: Dict) -> Dict:
        """
        Extract enriched metadata from API response.

        Args:
            api_response: TMDB API response

        Returns:
            Enriched metadata dictionary
        """
        enriched = {}

        # Basic info
        enriched['tagline'] = api_response.get('tagline', '')
        enriched['overview'] = api_response.get('overview', '')

        # Genres
        genres = api_response.get('genres', [])
        enriched['genres'] = [g['name'] for g in genres if 'name' in g]

        # Keywords
        keywords_data = api_response.get('keywords', {}).get('keywords', [])
        enriched['keywords'] = [k['name'] for k in keywords_data if 'name' in k]

        # Cast (top 5)
        credits = api_response.get('credits', {})
        cast = credits.get('cast', [])
        enriched['cast'] = [
            {
                'name': actor['name'],
                'character': actor.get('character', ''),
                'order': actor.get('order', 999)
            }
            for actor in cast[:5]
        ]

        # Crew (director, producers)
        crew = credits.get('crew', [])
        directors = [c['name'] for c in crew if c.get('job') == 'Director']
        enriched['director'] = directors[0] if directors else ''

        producers = [c['name'] for c in crew if c.get('job') in ('Producer', 'Executive Producer')]
        enriched['producers'] = producers[:3]  # Top 3 producers

        # Production companies
        companies = api_response.get('production_companies', [])
        enriched['production_companies'] = [c['name'] for c in companies if 'name' in c]

        # Production countries
        countries = api_response.get('production_countries', [])
        enriched['production_countries'] = [c['name'] for c in countries if 'name' in c]

        # Spoken languages
        languages = api_response.get('spoken_languages', [])
        enriched['spoken_languages'] = [l['english_name'] for l in languages if 'english_name' in l]

        return enriched

    def load_checkpoint(self) -> Tuple[int, List[Dict]]:
        """
        Load checkpoint if exists.

        Returns:
            Tuple of (start_index, processed_movies)
        """
        if not self.checkpoint_path.exists():
            logger.info("ℹ️  No checkpoint found, starting from beginning")
            return 0, []

        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            start_index = checkpoint.get('processed_count', 0)
            logger.info(f"✅ Loaded checkpoint: {start_index:,} movies processed")

            # Load partial output
            processed_movies = []
            if self.output_path.exists():
                with open(self.output_path, 'r') as f:
                    for line in f:
                        processed_movies.append(json.loads(line))

            return start_index, processed_movies

        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return 0, []

    def save_checkpoint(self, processed_count: int) -> None:
        """
        Save checkpoint.

        Args:
            processed_count: Number of movies processed
        """
        try:
            checkpoint = {
                'processed_count': processed_count,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats
            }

            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            self.stats['checkpoints_saved'] += 1
            logger.info(f"💾 Checkpoint saved: {processed_count:,} movies")

        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")

    def load_movies(self) -> List[Dict]:
        """
        Load TMDB movies from cleaned dataset.

        Returns:
            List of movie dictionaries
        """
        logger.info(f"\n📖 Loading TMDB movies: {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        movies = []
        with open(self.input_path, 'r') as f:
            for line in f:
                movies.append(json.loads(line))

        self.stats['total_movies'] = len(movies)
        logger.info(f"✅ Loaded {len(movies):,} movies")

        return movies

    def enrich_movies(
        self,
        movies: List[Dict],
        start_index: int = 0
    ) -> List[Dict]:
        """
        Enrich movies with TMDB API data.

        Args:
            movies: List of base movie dictionaries
            start_index: Index to resume from

        Returns:
            List of enriched movie dictionaries
        """
        logger.info(f"\n⚡ Enriching movies with TMDB API...")
        logger.info(f"   Starting from: {start_index:,}")
        logger.info(f"   Rate limit: {TMDB_RATE_LIMIT_CALLS} req/sec")

        start_time = datetime.now()
        enriched_movies = []

        # Open output file for appending
        mode = 'a' if start_index > 0 else 'w'
        with open(self.output_path, mode) as out_file:
            with tqdm(
                total=len(movies) - start_index,
                desc="Enriching movies",
                unit="movies",
                initial=0
            ) as pbar:
                for i in range(start_index, len(movies)):
                    movie = movies[i]
                    tmdb_id = movie.get('tmdb_id')

                    if not tmdb_id:
                        # No TMDB ID - keep base movie data
                        enriched_movie = movie.copy()
                        enriched_movie['enriched'] = False
                    else:
                        # Fetch API data
                        api_data = self.fetch_movie_details(tmdb_id)

                        if api_data:
                            # Extract enriched data
                            enriched_data = self.extract_enriched_data(api_data)

                            # Merge with base movie
                            enriched_movie = movie.copy()
                            enriched_movie.update(enriched_data)
                            enriched_movie['enriched'] = True
                            self.stats['enriched_movies'] += 1
                        else:
                            # API failed - use base movie data with fallback
                            enriched_movie = movie.copy()
                            enriched_movie['enriched'] = False
                            enriched_movie['tagline'] = ''
                            enriched_movie['keywords'] = enriched_movie.get('keywords', [])
                            enriched_movie['cast'] = []
                            enriched_movie['director'] = ''
                            enriched_movie['producers'] = []

                    # Write to output
                    out_file.write(json.dumps(enriched_movie) + '\n')
                    enriched_movies.append(enriched_movie)

                    self.stats['processed_movies'] = i + 1
                    pbar.update(1)

                    # Checkpoint every N movies
                    if (i + 1) % self.checkpoint_interval == 0:
                        self.save_checkpoint(i + 1)

                    # Periodic stats update
                    if (i + 1) % 1000 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = (i + 1 - start_index) / elapsed if elapsed > 0 else 0
                        eta_seconds = (len(movies) - i - 1) / rate if rate > 0 else 0
                        eta_hours = eta_seconds / 3600

                        pbar.set_postfix({
                            'enriched': f"{self.stats['enriched_movies']:,}",
                            'api_calls': f"{self.stats['api_calls']:,}",
                            'rate': f"{rate:.1f}/s",
                            'eta': f"{eta_hours:.1f}h"
                        })

        # Calculate processing time
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n✅ Enrichment complete: {len(enriched_movies):,} movies")

        return enriched_movies

    def validate_output(self) -> bool:
        """
        Validate enriched output.

        Returns:
            True if valid, False otherwise
        """
        logger.info("\n🔍 Validating output...")

        if not self.output_path.exists():
            logger.error(f"❌ Output file not found: {self.output_path}")
            return False

        # Count lines
        with open(self.output_path, 'r') as f:
            line_count = sum(1 for _ in f)

        if line_count != self.stats['total_movies']:
            logger.error(f"❌ Line count mismatch: {line_count:,} != {self.stats['total_movies']:,}")
            return False

        # Sample validation
        try:
            with open(self.output_path, 'r') as f:
                # Check first enriched movie
                for line in f:
                    movie = json.loads(line)
                    if movie.get('enriched'):
                        required_fields = ['tmdb_id', 'title', 'overview', 'genres', 'keywords', 'cast']
                        for field in required_fields:
                            if field not in movie:
                                logger.error(f"❌ Missing field '{field}' in enriched movie")
                                return False

                        logger.info(f"✅ Sample enriched movie: {movie['title']}")
                        logger.info(f"   Genres: {', '.join(movie['genres'][:3])}")
                        logger.info(f"   Keywords: {len(movie['keywords'])} keywords")
                        logger.info(f"   Cast: {len(movie['cast'])} actors")
                        break

            logger.info("✅ Validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False

    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1B: TMDB API ENRICHMENT STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total movies:         {self.stats['total_movies']:,}")
        logger.info(f"Processed movies:     {self.stats['processed_movies']:,}")
        logger.info(f"Enriched movies:      {self.stats['enriched_movies']:,} ({self.stats['enriched_movies']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"API calls:            {self.stats['api_calls']:,}")
        logger.info(f"API failures:         {self.stats['api_failures']:,} ({self.stats['api_failures']/self.stats['api_calls']*100:.1f}%)")
        logger.info(f"Rate limit hits:      {self.stats['rate_limit_hits']:,}")
        logger.info(f"Checkpoints saved:    {self.stats['checkpoints_saved']}")
        logger.info(f"Processing time:      {self.stats['processing_time']:.2f} seconds ({self.stats['processing_time']/3600:.2f} hours)")
        logger.info(f"Throughput:           {self.stats['processed_movies']/self.stats['processing_time']:.1f} movies/second")

        # File size
        if self.output_path.exists():
            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Output file size:     {file_size_mb:.2f} MB")

        logger.info("=" * 70)

    def run(self) -> bool:
        """
        Execute complete Stage 1b pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("🌐 TMDB API Enrichment - Stage 1b")
        logger.info("=" * 70)

        try:
            # Create directories
            self.create_directories()

            # Load movies
            movies = self.load_movies()

            # Check for checkpoint
            start_index, _ = self.load_checkpoint()

            # Enrich movies
            self.enrich_movies(movies, start_index=start_index)

            # Validate output
            if not self.validate_output():
                logger.warning("⚠️  Validation failed but continuing...")

            # Print statistics
            self.print_statistics()

            # Remove checkpoint on success
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.info("🗑️  Removed checkpoint (processing complete)")

            logger.info("\n✅ Stage 1b complete!")
            return True

        except Exception as e:
            logger.error(f"\n❌ Stage 1b failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Close session
            self.session.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TMDB API Enrichment - Stage 1b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run enrichment with API key from environment
  export TMDB_API_KEY="your_api_key_here"
  python stage1b_enrich_tmdb.py

  # Run with API key parameter
  python stage1b_enrich_tmdb.py --api-key "your_api_key_here"

  # Resume from checkpoint
  python stage1b_enrich_tmdb.py  # Automatically detects checkpoint

  # Custom checkpoint interval
  python stage1b_enrich_tmdb.py --checkpoint-interval 5000

Get API Key:
  1. Visit: https://www.themoviedb.org/settings/api
  2. Register for free account
  3. Request API key (instant approval)
  4. Copy API Key (v3 auth)

Output:
  data/processed/tmdb/movies_enriched.jsonl
        """
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to semantic-recommender directory'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='TMDB API key (or set TMDB_API_KEY env var)'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Save checkpoint every N movies (default: 10000)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum API retry attempts (default: 3)'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = TMDBEnrichment(
        base_path=args.base_path,
        api_key=args.api_key,
        checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries
    )
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
