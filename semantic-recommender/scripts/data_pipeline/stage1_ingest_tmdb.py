#!/usr/bin/env python3
"""
Stage 1: TMDB Dataset Ingestion and Cleaning

Parses TMDB CSV dataset (930k movies), handles JSON fields, cleans missing values,
and maps IMDB IDs to MovieLens IDs for integration.

Input:  data/raw/tmdb/TMDB_movie_dataset_v11.csv (930k rows × 24 columns)
Output: data/processed/tmdb/movies_clean.jsonl

Performance Target: ~60 seconds for 930k movies on CPU
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from tqdm import tqdm
import ast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBDataIngestion:
    """TMDB dataset ingestion and cleaning pipeline."""

    def __init__(self, base_path: str = None):
        """
        Initialize TMDB ingestion pipeline.

        Args:
            base_path: Base directory for semantic-recommender (auto-detected if None)
        """
        if base_path is None:
            # Auto-detect: go up from scripts/data_pipeline to project root
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent.parent
        else:
            self.base_path = Path(base_path)

        # Paths
        self.tmdb_raw_path = self.base_path / "data/raw/tmdb/TMDB_movie_dataset_v11.csv"
        self.ml_links_path = self.base_path / "data/raw/ml-25m/links.csv"
        self.output_dir = self.base_path / "data/processed/tmdb"
        self.output_path = self.output_dir / "movies_clean.jsonl"

        # Statistics
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'skipped_rows': 0,
            'ml_matches': 0,
            'missing_overview': 0,
            'missing_genres': 0,
            'missing_release_date': 0,
            'processing_time': 0
        }

    def create_directories(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {self.output_dir}")

    def load_movielens_mapping(self) -> Dict[str, str]:
        """
        Load MovieLens IMDB ID → MovieLens ID mapping.

        Returns:
            Dictionary mapping IMDB ID (tt1234567) to MovieLens ID (ml_1)
        """
        logger.info("\n📖 Loading MovieLens IMDB mapping...")

        if not self.ml_links_path.exists():
            logger.warning(f"⚠️  MovieLens links.csv not found: {self.ml_links_path}")
            return {}

        try:
            # Load links.csv
            links_df = pd.read_csv(self.ml_links_path)

            # Create IMDB ID → MovieLens ID mapping
            # Format IMDB IDs as "tt1234567" to match TMDB format
            imdb_to_ml = {}
            for _, row in links_df.iterrows():
                if pd.notna(row['imdbId']):
                    imdb_id = f"tt{int(row['imdbId']):07d}"
                    ml_id = f"ml_{int(row['movieId'])}"
                    imdb_to_ml[imdb_id] = ml_id

            logger.info(f"✅ Loaded {len(imdb_to_ml):,} MovieLens IMDB mappings")
            return imdb_to_ml

        except Exception as e:
            logger.error(f"❌ Error loading MovieLens mapping: {e}")
            return {}

    def parse_json_field(self, value: str, field_name: str = 'field') -> List[str]:
        """
        Parse JSON string field (genres, keywords, production_companies).

        Args:
            value: JSON string or list
            field_name: Name of field being parsed

        Returns:
            List of extracted values (names from objects)
        """
        if pd.isna(value) or value == '' or value == '[]':
            return []

        try:
            # Try parsing as JSON
            if isinstance(value, str):
                parsed = ast.literal_eval(value)
            else:
                parsed = value

            # Extract 'name' field from list of dicts
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    return [item.get('name', str(item)) for item in parsed if 'name' in item]
                else:
                    return [str(item) for item in parsed]

            return []

        except Exception as e:
            # Fallback: treat as empty
            return []

    def extract_year(self, release_date: str) -> Optional[int]:
        """
        Extract year from release_date field.

        Args:
            release_date: Date string (various formats)

        Returns:
            Year as integer or None
        """
        if pd.isna(release_date) or release_date == '':
            return None

        try:
            # Try parsing as date
            date_obj = pd.to_datetime(release_date, errors='coerce')
            if pd.notna(date_obj):
                return int(date_obj.year)
        except:
            pass

        # Try extracting 4-digit year
        import re
        match = re.search(r'(\d{4})', str(release_date))
        if match:
            year = int(match.group(1))
            if 1800 <= year <= 2030:
                return year

        return None

    def clean_movie_row(
        self,
        row: pd.Series,
        imdb_to_ml: Dict[str, str]
    ) -> Optional[Dict]:
        """
        Clean and transform a single TMDB movie row.

        Args:
            row: Pandas Series with TMDB movie data
            imdb_to_ml: IMDB ID to MovieLens ID mapping

        Returns:
            Cleaned movie dictionary or None if invalid
        """
        # Required fields
        if pd.isna(row.get('id')) or pd.isna(row.get('title')):
            return None

        # Extract IMDB ID
        imdb_id = row.get('imdb_id', '')
        if pd.isna(imdb_id):
            imdb_id = ''

        # Map to MovieLens ID if available
        ml_id = imdb_to_ml.get(imdb_id, None)
        if ml_id:
            self.stats['ml_matches'] += 1

        # Parse JSON fields
        genres = self.parse_json_field(row.get('genres', '[]'), 'genres')
        keywords = self.parse_json_field(row.get('keywords', '[]'), 'keywords')
        production_companies = self.parse_json_field(
            row.get('production_companies', '[]'),
            'production_companies'
        )

        # Extract year
        year = self.extract_year(row.get('release_date', ''))

        # Track missing data
        overview = row.get('overview', '')
        if pd.isna(overview) or overview == '':
            self.stats['missing_overview'] += 1
            overview = ''

        if len(genres) == 0:
            self.stats['missing_genres'] += 1

        if year is None:
            self.stats['missing_release_date'] += 1

        # Build cleaned movie object
        movie = {
            'tmdb_id': str(int(row['id'])),
            'imdb_id': imdb_id,
            'ml_id': ml_id,  # None if no match
            'title': row.get('title', ''),
            'original_title': row.get('original_title', row.get('title', '')),
            'overview': overview,
            'year': year,
            'release_date': row.get('release_date', ''),
            'genres': genres,
            'keywords': keywords,
            'production_companies': production_companies,
            'original_language': row.get('original_language', ''),
            'popularity': float(row.get('popularity', 0.0)) if pd.notna(row.get('popularity')) else 0.0,
            'vote_average': float(row.get('vote_average', 0.0)) if pd.notna(row.get('vote_average')) else 0.0,
            'vote_count': int(row.get('vote_count', 0)) if pd.notna(row.get('vote_count')) else 0,
            'runtime': int(row.get('runtime', 0)) if pd.notna(row.get('runtime')) else 0,
            'budget': int(row.get('budget', 0)) if pd.notna(row.get('budget')) else 0,
            'revenue': int(row.get('revenue', 0)) if pd.notna(row.get('revenue')) else 0,
            'adult': bool(row.get('adult', False)) if pd.notna(row.get('adult')) else False,
        }

        return movie

    def process_dataset(self, chunk_size: int = 10000) -> None:
        """
        Process TMDB dataset in chunks for memory efficiency.

        Args:
            chunk_size: Number of rows to process at a time
        """
        logger.info(f"\n📊 Processing TMDB dataset: {self.tmdb_raw_path}")

        if not self.tmdb_raw_path.exists():
            raise FileNotFoundError(f"TMDB dataset not found: {self.tmdb_raw_path}")

        # Load MovieLens mapping
        imdb_to_ml = self.load_movielens_mapping()

        # Get total rows for progress bar
        logger.info("📏 Counting total rows...")
        total_rows = sum(1 for _ in open(self.tmdb_raw_path)) - 1  # Exclude header
        self.stats['total_rows'] = total_rows
        logger.info(f"✅ Total rows: {total_rows:,}")

        # Process in chunks
        start_time = datetime.now()

        with open(self.output_path, 'w') as out_file:
            # Read CSV in chunks
            chunks = pd.read_csv(
                self.tmdb_raw_path,
                chunksize=chunk_size,
                low_memory=False
            )

            with tqdm(total=total_rows, desc="Processing movies", unit="rows") as pbar:
                for chunk in chunks:
                    # Process each row in chunk
                    for _, row in chunk.iterrows():
                        movie = self.clean_movie_row(row, imdb_to_ml)

                        if movie:
                            # Write as JSONL
                            out_file.write(json.dumps(movie) + '\n')
                            self.stats['valid_rows'] += 1
                        else:
                            self.stats['skipped_rows'] += 1

                        pbar.update(1)

        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()

        logger.info(f"\n✅ Processing complete: {self.output_path}")

    def validate_output(self) -> bool:
        """
        Validate output file.

        Returns:
            True if valid, False otherwise
        """
        logger.info("\n🔍 Validating output...")

        if not self.output_path.exists():
            logger.error(f"❌ Output file not found: {self.output_path}")
            return False

        # Check file size
        file_size = self.output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"📦 File size: {file_size_mb:.2f} MB")

        # Sample validation: read first and last records
        try:
            with open(self.output_path, 'r') as f:
                lines = f.readlines()

            # Validate first record
            first_movie = json.loads(lines[0])
            required_fields = ['tmdb_id', 'title', 'overview', 'genres']
            for field in required_fields:
                if field not in first_movie:
                    logger.error(f"❌ Missing field '{field}' in first record")
                    return False

            # Validate last record
            last_movie = json.loads(lines[-1])
            for field in required_fields:
                if field not in last_movie:
                    logger.error(f"❌ Missing field '{field}' in last record")
                    return False

            logger.info(f"✅ Sample validation passed")
            logger.info(f"   First movie: {first_movie['title']} ({first_movie.get('year', 'N/A')})")
            logger.info(f"   Last movie: {last_movie['title']} ({last_movie.get('year', 'N/A')})")

            return True

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False

    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: TMDB INGESTION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total rows processed:    {self.stats['total_rows']:,}")
        logger.info(f"Valid movies:            {self.stats['valid_rows']:,}")
        logger.info(f"Skipped rows:            {self.stats['skipped_rows']:,}")
        logger.info(f"MovieLens matches:       {self.stats['ml_matches']:,} ({self.stats['ml_matches']/self.stats['valid_rows']*100:.1f}%)")
        logger.info(f"Missing overview:        {self.stats['missing_overview']:,} ({self.stats['missing_overview']/self.stats['valid_rows']*100:.1f}%)")
        logger.info(f"Missing genres:          {self.stats['missing_genres']:,} ({self.stats['missing_genres']/self.stats['valid_rows']*100:.1f}%)")
        logger.info(f"Missing release_date:    {self.stats['missing_release_date']:,} ({self.stats['missing_release_date']/self.stats['valid_rows']*100:.1f}%)")
        logger.info(f"Processing time:         {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Throughput:              {self.stats['valid_rows']/self.stats['processing_time']:.0f} movies/second")
        logger.info("=" * 70)

    def run(self) -> bool:
        """
        Execute complete Stage 1 pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("🎬 TMDB Dataset Ingestion - Stage 1")
        logger.info("=" * 70)

        try:
            # Create directories
            self.create_directories()

            # Process dataset
            self.process_dataset()

            # Validate output
            if not self.validate_output():
                return False

            # Print statistics
            self.print_statistics()

            logger.info("\n✅ Stage 1 complete!")
            return True

        except Exception as e:
            logger.error(f"\n❌ Stage 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TMDB Dataset Ingestion - Stage 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ingestion
  python stage1_ingest_tmdb.py

  # Custom base path
  python stage1_ingest_tmdb.py --base-path /path/to/semantic-recommender

Output:
  data/processed/tmdb/movies_clean.jsonl
        """
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to semantic-recommender directory'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = TMDBDataIngestion(base_path=args.base_path)
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
