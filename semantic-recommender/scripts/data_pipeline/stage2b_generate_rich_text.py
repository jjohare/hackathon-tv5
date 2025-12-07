#!/usr/bin/env python3
"""
Stage 2b: Generate Rich Semantic Text for Embeddings

Generates comprehensive semantic text from enriched TMDB metadata:
- Combines: title, tagline, overview, genres, keywords, cast, director
- Validates: Every movie has non-empty text
- Fallback: Uses title + keywords if overview missing

Input:  data/processed/tmdb/movies_enriched.jsonl
Output: data/processed/tmdb/movies_rich_text.jsonl

Performance Target: ~2 minutes for 1.3M movies (pure text processing)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RichTextGenerator:
    """Generate rich semantic text for movie embeddings."""

    def __init__(self, base_path: str = None):
        """
        Initialize rich text generator.

        Args:
            base_path: Base directory for semantic-recommender (auto-detected if None)
        """
        if base_path is None:
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent.parent
        else:
            self.base_path = Path(base_path)

        # Paths
        self.input_path = self.base_path / "data/processed/tmdb/movies_enriched.jsonl"
        self.output_dir = self.base_path / "data/processed/tmdb"
        self.output_path = self.output_dir / "movies_rich_text.jsonl"

        # Statistics
        self.stats = {
            'total_movies': 0,
            'processed_movies': 0,
            'has_overview': 0,
            'has_tagline': 0,
            'has_keywords': 0,
            'has_cast': 0,
            'has_director': 0,
            'fallback_used': 0,
            'empty_text': 0,
            'avg_text_length': 0,
            'processing_time': 0
        }

        self.text_lengths = []

    def create_directories(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {self.output_dir}")

    def generate_rich_text(self, movie: Dict) -> str:
        """
        Generate rich semantic text from movie metadata.

        Text format:
        "{title}. {tagline}. {overview}. Genres: {genres}. Keywords: {keywords}.
         Starring: {cast}. Directed by {director}."

        Args:
            movie: Movie dictionary with enriched metadata

        Returns:
            Rich semantic text string
        """
        components = []

        # 1. Title (required)
        title = movie.get('title', '')
        if title:
            components.append(title)

        # 2. Tagline (optional)
        tagline = movie.get('tagline', '').strip()
        if tagline:
            components.append(tagline)
            self.stats['has_tagline'] += 1

        # 3. Overview (primary semantic content)
        overview = movie.get('overview', '').strip()
        if overview:
            components.append(overview)
            self.stats['has_overview'] += 1

        # 4. Genres
        genres = movie.get('genres', [])
        if genres:
            genres_str = ', '.join(genres)
            components.append(f"Genres: {genres_str}")

        # 5. Keywords (semantic tags)
        keywords = movie.get('keywords', [])
        if keywords:
            # Limit to top 10 keywords for relevance
            keywords_str = ', '.join(keywords[:10])
            components.append(f"Keywords: {keywords_str}")
            self.stats['has_keywords'] += 1

        # 6. Cast (top actors)
        cast = movie.get('cast', [])
        if cast:
            # Extract actor names
            actor_names = [actor.get('name', '') for actor in cast if actor.get('name')]
            if actor_names:
                cast_str = ', '.join(actor_names[:5])
                components.append(f"Starring: {cast_str}")
                self.stats['has_cast'] += 1

        # 7. Director
        director = movie.get('director', '').strip()
        if director:
            components.append(f"Directed by {director}")
            self.stats['has_director'] += 1

        # 8. Production companies (additional context)
        companies = movie.get('production_companies', [])
        if companies:
            companies_str = ', '.join(companies[:3])
            components.append(f"Production: {companies_str}")

        # 9. Year (temporal context)
        year = movie.get('year')
        if year:
            components.append(f"Released in {year}")

        # Join all components
        rich_text = '. '.join(components)

        # Ensure proper ending
        if rich_text and not rich_text.endswith('.'):
            rich_text += '.'

        # Fallback if text is empty or too short
        if not rich_text or len(rich_text) < 10:
            # Use title + genres + keywords as fallback
            fallback_components = []

            if title:
                fallback_components.append(title)

            if genres:
                fallback_components.append(f"Genres: {', '.join(genres)}")

            if keywords:
                fallback_components.append(f"Keywords: {', '.join(keywords[:5])}")

            rich_text = '. '.join(fallback_components) + '.'
            self.stats['fallback_used'] += 1

        # Track if still empty (should be rare)
        if not rich_text or len(rich_text) < 5:
            self.stats['empty_text'] += 1

        return rich_text

    def process_movies(self) -> None:
        """Process all movies and generate rich text."""
        logger.info(f"\n📖 Loading enriched movies: {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        # Count total movies
        with open(self.input_path, 'r') as f:
            self.stats['total_movies'] = sum(1 for _ in f)

        logger.info(f"✅ Found {self.stats['total_movies']:,} movies")

        logger.info(f"\n⚡ Generating rich semantic text...")

        start_time = datetime.now()

        with open(self.input_path, 'r') as in_file, \
             open(self.output_path, 'w') as out_file:

            with tqdm(
                total=self.stats['total_movies'],
                desc="Generating text",
                unit="movies"
            ) as pbar:
                for line in in_file:
                    movie = json.loads(line)

                    # Generate rich text
                    rich_text = self.generate_rich_text(movie)

                    # Track text length
                    self.text_lengths.append(len(rich_text))

                    # Create output record
                    output = {
                        'tmdb_id': movie.get('tmdb_id'),
                        'imdb_id': movie.get('imdb_id'),
                        'ml_id': movie.get('ml_id'),
                        'title': movie.get('title'),
                        'year': movie.get('year'),
                        'rich_text': rich_text,
                        'enriched': movie.get('enriched', False)
                    }

                    # Write to output
                    out_file.write(json.dumps(output) + '\n')

                    self.stats['processed_movies'] += 1
                    pbar.update(1)

        # Calculate stats
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()

        if self.text_lengths:
            self.stats['avg_text_length'] = sum(self.text_lengths) / len(self.text_lengths)

        logger.info(f"\n✅ Generated rich text for {self.stats['processed_movies']:,} movies")

    def validate_output(self) -> bool:
        """
        Validate generated rich text.

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
                # Check first few records
                for i, line in enumerate(f):
                    if i >= 3:
                        break

                    movie = json.loads(line)

                    # Check required fields
                    if 'rich_text' not in movie:
                        logger.error(f"❌ Missing 'rich_text' field in record {i}")
                        return False

                    # Check text is non-empty
                    if not movie['rich_text'] or len(movie['rich_text']) < 5:
                        logger.error(f"❌ Empty or too short text in record {i}: {movie.get('title')}")
                        return False

                    if i == 0:
                        logger.info(f"✅ Sample movie: {movie['title']} ({movie.get('year', 'N/A')})")
                        logger.info(f"   Rich text length: {len(movie['rich_text'])} chars")
                        logger.info(f"   Sample text: {movie['rich_text'][:200]}...")

            logger.info("✅ Validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False

    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2B: RICH TEXT GENERATION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total movies:         {self.stats['total_movies']:,}")
        logger.info(f"Processed movies:     {self.stats['processed_movies']:,}")

        logger.info(f"\nContent Coverage:")
        logger.info(f"  Has overview:       {self.stats['has_overview']:,} ({self.stats['has_overview']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"  Has tagline:        {self.stats['has_tagline']:,} ({self.stats['has_tagline']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"  Has keywords:       {self.stats['has_keywords']:,} ({self.stats['has_keywords']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"  Has cast:           {self.stats['has_cast']:,} ({self.stats['has_cast']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"  Has director:       {self.stats['has_director']:,} ({self.stats['has_director']/self.stats['processed_movies']*100:.1f}%)")

        logger.info(f"\nText Quality:")
        logger.info(f"  Fallback used:      {self.stats['fallback_used']:,} ({self.stats['fallback_used']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"  Empty text:         {self.stats['empty_text']:,} ({self.stats['empty_text']/self.stats['processed_movies']*100:.1f}%)")
        logger.info(f"  Avg text length:    {self.stats['avg_text_length']:.0f} chars")

        if self.text_lengths:
            logger.info(f"  Min text length:    {min(self.text_lengths)} chars")
            logger.info(f"  Max text length:    {max(self.text_lengths)} chars")

        logger.info(f"\nPerformance:")
        logger.info(f"  Processing time:    {self.stats['processing_time']:.2f} seconds")
        logger.info(f"  Throughput:         {self.stats['processed_movies']/self.stats['processing_time']:.0f} movies/second")

        # File size
        if self.output_path.exists():
            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Output file size:   {file_size_mb:.2f} MB")

        logger.info("=" * 70)

    def run(self) -> bool:
        """
        Execute complete Stage 2b pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("📝 Rich Semantic Text Generation - Stage 2b")
        logger.info("=" * 70)

        try:
            # Create directories
            self.create_directories()

            # Process movies
            self.process_movies()

            # Validate output
            if not self.validate_output():
                return False

            # Print statistics
            self.print_statistics()

            logger.info("\n✅ Stage 2b complete!")
            return True

        except Exception as e:
            logger.error(f"\n❌ Stage 2b failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rich Semantic Text Generation - Stage 2b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate rich text
  python stage2b_generate_rich_text.py

  # Custom base path
  python stage2b_generate_rich_text.py --base-path /path/to/semantic-recommender

Output:
  data/processed/tmdb/movies_rich_text.jsonl

Format:
  Each line contains:
  - tmdb_id, imdb_id, ml_id, title, year
  - rich_text: Comprehensive semantic description
  - enriched: Whether movie was enriched with TMDB API

Sample Rich Text:
  "Inception. Your mind is the scene of the crime. Cobb, a skilled thief who
   commits corporate espionage by infiltrating the subconscious of his targets
   is offered a chance to regain his old life as payment for a task considered
   to be impossible: "inception", the implantation of another person's idea into
   a target's subconscious. Genres: Action, Science Fiction, Adventure.
   Keywords: dreams, subconscious, heist, mind control.
   Starring: Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page, Tom Hardy,
   Ken Watanabe. Directed by Christopher Nolan. Released in 2010."
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
    generator = RichTextGenerator(base_path=args.base_path)
    success = generator.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
