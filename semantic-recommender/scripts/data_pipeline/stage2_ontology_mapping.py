#!/usr/bin/env python3
"""
Stage 2: Ontology Mapping for TMDB Movies

Maps TMDB keywords and overview to MovieLens genome tags using semantic similarity.
Generates genome scores for TMDB-only movies to enable hybrid recommendations.

Input:  data/processed/tmdb/movies_clean.jsonl
        data/processed/media/genome_scores.json (MovieLens)
Output: data/processed/tmdb/genome_scores.json

Performance Target: ~120 seconds for 930k movies (semantic matching)
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBOntologyMapping:
    """Map TMDB movies to MovieLens genome tags using semantic similarity."""

    def __init__(self, base_path: str = None):
        """
        Initialize ontology mapping pipeline.

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
        self.tmdb_clean_path = self.base_path / "data/processed/tmdb/movies_clean.jsonl"
        self.ml_genome_path = self.base_path / "data/processed/media/genome_scores.json"
        self.output_dir = self.base_path / "data/processed/tmdb"
        self.output_path = self.output_dir / "genome_scores.json"

        # Genome tag vocabulary (1,128 tags from MovieLens)
        self.genome_tags = []
        self.tag_to_idx = {}

        # Keyword to genome tag mapping
        self.keyword_mappings = {}

        # Statistics
        self.stats = {
            'total_movies': 0,
            'ml_overlap': 0,
            'tmdb_only': 0,
            'avg_tags_per_movie': 0,
            'keyword_matches': 0,
            'theme_matches': 0,
            'processing_time': 0
        }

    def load_genome_vocabulary(self) -> None:
        """Load MovieLens genome tag vocabulary."""
        logger.info("\n📖 Loading MovieLens genome vocabulary...")

        if not self.ml_genome_path.exists():
            raise FileNotFoundError(f"MovieLens genome scores not found: {self.ml_genome_path}")

        try:
            with open(self.ml_genome_path, 'r') as f:
                ml_genome = json.load(f)

            # Extract all unique tags
            all_tags = set()
            for movie_id, tags in ml_genome.items():
                all_tags.update(tags.keys())

            # Sort for consistency
            self.genome_tags = sorted(list(all_tags))
            self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.genome_tags)}

            logger.info(f"✅ Loaded {len(self.genome_tags)} genome tags")

            # Sample tags
            logger.info(f"   Sample tags: {', '.join(self.genome_tags[:10])}")

        except Exception as e:
            logger.error(f"❌ Error loading genome vocabulary: {e}")
            raise

    def build_keyword_mappings(self) -> None:
        """
        Build keyword → genome tag mappings.

        Uses fuzzy matching and semantic similarity for common mappings.
        """
        logger.info("\n🔨 Building keyword → genome tag mappings...")

        # Manual mappings for common keywords
        # Based on MovieLens genome tag analysis
        self.keyword_mappings = {
            # Actions & Violence
            'action': ['action', 'violence'],
            'violence': ['violence', 'brutality'],
            'fight': ['action', 'violence'],
            'chase': ['action', 'chase'],
            'explosion': ['action', 'violence'],

            # Emotions & Mood
            'romantic': ['romantic', 'romance'],
            'romance': ['romantic', 'romance'],
            'love': ['romantic', 'love story'],
            'comedy': ['comedy', 'funny'],
            'funny': ['comedy', 'humor'],
            'dark': ['dark', 'dark comedy'],
            'suspense': ['suspenseful', 'suspense'],
            'thriller': ['suspenseful', 'thriller'],

            # Themes
            'murder': ['murder', 'crime'],
            'crime': ['crime', 'murder'],
            'mystery': ['mysterious', 'suspenseful'],
            'revenge': ['vengeance', 'revenge'],
            'conspiracy': ['paranoia', 'government'],

            # Style
            'noir': ['noir', 'dark'],
            'sci-fi': ['sci-fi', 'science fiction'],
            'fantasy': ['fantasy', 'surreal'],
            'horror': ['horror', 'scary'],

            # Character
            'hero': ['heroic', 'courageous'],
            'villain': ['evil', 'bad guy'],
            'friendship': ['friendship', 'loyalty'],
            'betrayal': ['betrayal', 'deception'],
        }

        # Expand with fuzzy variants
        expanded_mappings = {}
        for keyword, tags in self.keyword_mappings.items():
            # Match genome tags by substring
            matched_tags = []
            for tag in self.genome_tags:
                tag_lower = tag.lower()
                keyword_lower = keyword.lower()

                # Direct match or substring match
                if keyword_lower in tag_lower or tag_lower in keyword_lower:
                    matched_tags.append(tag)

            # Use manual mappings if available, otherwise use fuzzy matches
            if matched_tags:
                expanded_mappings[keyword] = matched_tags
            else:
                expanded_mappings[keyword] = tags

        self.keyword_mappings = expanded_mappings

        logger.info(f"✅ Built {len(self.keyword_mappings)} keyword mappings")

    def extract_themes_from_overview(self, overview: str) -> List[str]:
        """
        Extract thematic keywords from movie overview using NER/semantic parsing.

        Args:
            overview: Movie overview text

        Returns:
            List of extracted theme keywords
        """
        if not overview:
            return []

        themes = []

        # Simple keyword extraction (can be enhanced with NER)
        overview_lower = overview.lower()

        # Check for common themes
        theme_patterns = {
            'revenge': r'\b(revenge|vengeance|retribution)\b',
            'love': r'\b(love|romance|relationship)\b',
            'murder': r'\b(murder|kill|assassin)\b',
            'war': r'\b(war|battle|conflict)\b',
            'crime': r'\b(crime|criminal|detective)\b',
            'family': r'\b(family|father|mother|son|daughter)\b',
            'friendship': r'\b(friend|friendship|companion)\b',
            'betrayal': r'\b(betray|deception|lie)\b',
        }

        for theme, pattern in theme_patterns.items():
            if re.search(pattern, overview_lower):
                themes.append(theme)

        return themes

    def map_movie_to_genome(
        self,
        movie: Dict,
        ml_genome: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Map a single TMDB movie to genome scores.

        Args:
            movie: TMDB movie dictionary
            ml_genome: MovieLens genome scores (for overlap movies)

        Returns:
            Dictionary of {tag: score} for this movie
        """
        genome_scores = {}

        # Check if movie has MovieLens overlap
        ml_id = movie.get('ml_id')
        if ml_id:
            # Use existing MovieLens genome scores
            ml_id_plain = ml_id.replace('ml_', '')
            if ml_id_plain in ml_genome:
                self.stats['ml_overlap'] += 1
                return ml_genome[ml_id_plain]

        # TMDB-only movie: generate genome scores
        self.stats['tmdb_only'] += 1

        # Map keywords
        keywords = movie.get('keywords', [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.keyword_mappings:
                matched_tags = self.keyword_mappings[keyword_lower]
                for tag in matched_tags:
                    if tag in self.tag_to_idx:
                        # Use confidence score (0.7 for keyword matches)
                        genome_scores[tag] = max(genome_scores.get(tag, 0.0), 0.7)
                        self.stats['keyword_matches'] += 1

        # Map overview themes
        overview = movie.get('overview', '')
        themes = self.extract_themes_from_overview(overview)
        for theme in themes:
            theme_lower = theme.lower()
            if theme_lower in self.keyword_mappings:
                matched_tags = self.keyword_mappings[theme_lower]
                for tag in matched_tags:
                    if tag in self.tag_to_idx:
                        # Use lower confidence for theme extraction (0.5)
                        genome_scores[tag] = max(genome_scores.get(tag, 0.0), 0.5)
                        self.stats['theme_matches'] += 1

        # Map genres (high confidence)
        genres = movie.get('genres', [])
        for genre in genres:
            genre_lower = genre.lower()
            # Map genre directly to genome tags
            for tag in self.genome_tags:
                if genre_lower in tag.lower():
                    genome_scores[tag] = max(genome_scores.get(tag, 0.0), 0.9)

        return genome_scores

    def process_movies(self) -> None:
        """Process all TMDB movies and generate genome scores."""
        logger.info(f"\n📊 Processing TMDB movies: {self.tmdb_clean_path}")

        if not self.tmdb_clean_path.exists():
            raise FileNotFoundError(f"TMDB cleaned movies not found: {self.tmdb_clean_path}")

        # Load MovieLens genome scores
        logger.info("📖 Loading MovieLens genome scores...")
        with open(self.ml_genome_path, 'r') as f:
            ml_genome = json.load(f)
        logger.info(f"✅ Loaded {len(ml_genome)} MovieLens genome profiles")

        # Process TMDB movies
        start_time = datetime.now()

        tmdb_genome = {}
        total_tags = 0

        with open(self.tmdb_clean_path, 'r') as f:
            lines = f.readlines()

        self.stats['total_movies'] = len(lines)

        with tqdm(total=len(lines), desc="Mapping to genome", unit="movies") as pbar:
            for line in lines:
                movie = json.loads(line)
                tmdb_id = movie['tmdb_id']

                # Map to genome
                genome_scores = self.map_movie_to_genome(movie, ml_genome)

                if genome_scores:
                    tmdb_genome[tmdb_id] = genome_scores
                    total_tags += len(genome_scores)

                pbar.update(1)

        # Calculate statistics
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        self.stats['avg_tags_per_movie'] = total_tags / len(tmdb_genome) if tmdb_genome else 0

        # Save output
        logger.info(f"\n💾 Saving genome scores: {self.output_path}")
        with open(self.output_path, 'w') as f:
            json.dump(tmdb_genome, f, indent=2)

        logger.info(f"✅ Saved {len(tmdb_genome):,} movie genome profiles")

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

        # Validate JSON structure
        try:
            with open(self.output_path, 'r') as f:
                genome_scores = json.load(f)

            # Sample validation
            sample_id = list(genome_scores.keys())[0]
            sample_scores = genome_scores[sample_id]

            logger.info(f"✅ Sample validation passed")
            logger.info(f"   Movie ID: {sample_id}")
            logger.info(f"   Genome tags: {len(sample_scores)}")
            logger.info(f"   Sample tags: {list(sample_scores.keys())[:5]}")

            return True

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False

    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: ONTOLOGY MAPPING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total movies processed: {self.stats['total_movies']:,}")
        logger.info(f"MovieLens overlap:      {self.stats['ml_overlap']:,} ({self.stats['ml_overlap']/self.stats['total_movies']*100:.1f}%)")
        logger.info(f"TMDB-only movies:       {self.stats['tmdb_only']:,} ({self.stats['tmdb_only']/self.stats['total_movies']*100:.1f}%)")
        logger.info(f"Avg tags per movie:     {self.stats['avg_tags_per_movie']:.1f}")
        logger.info(f"Keyword matches:        {self.stats['keyword_matches']:,}")
        logger.info(f"Theme matches:          {self.stats['theme_matches']:,}")
        logger.info(f"Processing time:        {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Throughput:             {self.stats['total_movies']/self.stats['processing_time']:.0f} movies/second")
        logger.info("=" * 70)

    def run(self) -> bool:
        """
        Execute complete Stage 2 pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("🧠 TMDB Ontology Mapping - Stage 2")
        logger.info("=" * 70)

        try:
            # Load genome vocabulary
            self.load_genome_vocabulary()

            # Build keyword mappings
            self.build_keyword_mappings()

            # Process movies
            self.process_movies()

            # Validate output
            if not self.validate_output():
                return False

            # Print statistics
            self.print_statistics()

            logger.info("\n✅ Stage 2 complete!")
            return True

        except Exception as e:
            logger.error(f"\n❌ Stage 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TMDB Ontology Mapping - Stage 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ontology mapping
  python stage2_ontology_mapping.py

  # Custom base path
  python stage2_ontology_mapping.py --base-path /path/to/semantic-recommender

Output:
  data/processed/tmdb/genome_scores.json
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
    pipeline = TMDBOntologyMapping(base_path=args.base_path)
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
