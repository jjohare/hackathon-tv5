#!/usr/bin/env python3
"""
Stage 3: GPU-Accelerated Embedding Generation for TMDB Movies

Uses TensorRT engine to generate embeddings for 930k TMDB movies with:
- Batch processing (batch_size=32)
- GPU acceleration via TensorRT
- Checkpointing every 10k movies (resumable)
- Progress tracking and ETA

Input:  data/processed/tmdb/movies_clean.jsonl
        data/models/minilm_l12_v2_fp16.plan (TensorRT engine)
Output: data/embeddings/tmdb/content_vectors.npy (1.4GB)
        data/embeddings/tmdb/metadata.jsonl

Performance Target: ~15 minutes for 930k movies on A100 GPU (1000 movies/second)
"""

import os
import sys
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from trt_inference import TensorRTEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBEmbeddingGenerator:
    """GPU-accelerated embedding generation for TMDB movies."""

    def __init__(
        self,
        base_path: str = None,
        batch_size: int = 32,
        checkpoint_interval: int = 10000
    ):
        """
        Initialize embedding generator.

        Args:
            base_path: Base directory for semantic-recommender (auto-detected if None)
            batch_size: Batch size for GPU processing
            checkpoint_interval: Save checkpoint every N movies
        """
        if base_path is None:
            # Auto-detect: go up from scripts/data_pipeline to project root
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent.parent
        else:
            self.base_path = Path(base_path)

        # Paths
        self.tmdb_clean_path = self.base_path / "data/processed/tmdb/movies_clean.jsonl"
        self.trt_engine_path = self.base_path / "data/models/minilm_l12_v2_fp16.plan"
        self.output_dir = self.base_path / "data/embeddings/tmdb"
        self.embeddings_path = self.output_dir / "content_vectors.npy"
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoint.npz"

        # Parameters
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        # Model
        self.encoder = None

        # Statistics
        self.stats = {
            'total_movies': 0,
            'processed_movies': 0,
            'embedding_dim': 0,
            'gpu_time': 0,
            'total_time': 0,
            'checkpoints_saved': 0
        }

    def create_directories(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {self.output_dir}")

    def initialize_encoder(self) -> None:
        """Initialize TensorRT encoder."""
        logger.info("\n🚀 Initializing TensorRT encoder...")

        if not self.trt_engine_path.exists():
            logger.warning(f"⚠️  TensorRT engine not found: {self.trt_engine_path}")
            logger.info("   Falling back to PyTorch model (slower)")
            self.encoder = TensorRTEncoder(
                engine_path=str(self.trt_engine_path),
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        else:
            self.encoder = TensorRTEncoder(
                engine_path=str(self.trt_engine_path),
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

        # Test encoding
        test_embedding = self.encoder.encode(["Test sentence"])
        self.stats['embedding_dim'] = test_embedding.shape[1]

        logger.info(f"✅ Encoder initialized")
        logger.info(f"   Using TensorRT: {self.encoder.use_tensorrt}")
        logger.info(f"   Embedding dim: {self.stats['embedding_dim']}")

    def load_checkpoint(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Load checkpoint if exists.

        Returns:
            Tuple of (embeddings_array, start_index)
        """
        if not self.checkpoint_path.exists():
            logger.info("ℹ️  No checkpoint found, starting from beginning")
            return None, 0

        try:
            checkpoint = np.load(self.checkpoint_path, allow_pickle=True)
            embeddings = checkpoint['embeddings']
            start_index = int(checkpoint['processed_count'])

            logger.info(f"✅ Loaded checkpoint: {start_index:,} movies processed")
            logger.info(f"   Resuming from index {start_index}")

            return embeddings, start_index

        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            logger.info("   Starting from beginning")
            return None, 0

    def save_checkpoint(
        self,
        embeddings: np.ndarray,
        processed_count: int
    ) -> None:
        """
        Save checkpoint.

        Args:
            embeddings: Embeddings array
            processed_count: Number of movies processed
        """
        try:
            np.savez_compressed(
                self.checkpoint_path,
                embeddings=embeddings,
                processed_count=processed_count
            )
            self.stats['checkpoints_saved'] += 1
            logger.info(f"💾 Checkpoint saved: {processed_count:,} movies")

        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")

    def load_movies(self) -> Tuple[List[Dict], List[str]]:
        """
        Load TMDB movies.

        Returns:
            Tuple of (movies_list, texts_list)
        """
        logger.info(f"\n📖 Loading TMDB movies: {self.tmdb_clean_path}")

        if not self.tmdb_clean_path.exists():
            raise FileNotFoundError(f"TMDB cleaned movies not found: {self.tmdb_clean_path}")

        movies = []
        texts = []

        with open(self.tmdb_clean_path, 'r') as f:
            for line in f:
                movie = json.loads(line)
                movies.append(movie)

                # Use overview as primary text for embedding
                # Fallback to title if no overview
                text = movie.get('overview', '')
                if not text:
                    text = movie.get('title', '')

                texts.append(text)

        self.stats['total_movies'] = len(movies)
        logger.info(f"✅ Loaded {len(movies):,} movies")

        return movies, texts

    def generate_embeddings(
        self,
        texts: List[str],
        start_index: int = 0,
        checkpoint_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate embeddings for all texts with GPU acceleration.

        Args:
            texts: List of text strings
            start_index: Index to resume from (for checkpointing)
            checkpoint_embeddings: Existing embeddings from checkpoint

        Returns:
            Embeddings array (num_movies, embedding_dim)
        """
        logger.info(f"\n⚡ Generating embeddings with GPU acceleration...")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Starting from: {start_index:,}")

        total_movies = len(texts)

        # Initialize or use checkpoint embeddings
        if checkpoint_embeddings is not None:
            embeddings = checkpoint_embeddings
        else:
            embeddings = np.zeros((total_movies, self.stats['embedding_dim']), dtype=np.float32)

        # Process in batches
        start_time = datetime.now()
        gpu_start = datetime.now()

        with tqdm(
            total=total_movies - start_index,
            desc="Generating embeddings",
            unit="movies",
            initial=0
        ) as pbar:
            for i in range(start_index, total_movies, self.batch_size):
                batch_end = min(i + self.batch_size, total_movies)
                batch_texts = texts[i:batch_end]

                # Encode batch (GPU)
                batch_embeddings = self.encoder.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    convert_to_tensor=True
                )

                # Convert to numpy and store
                embeddings[i:batch_end] = batch_embeddings.cpu().numpy()

                pbar.update(batch_end - i)

                # Checkpoint every N movies
                if (batch_end - start_index) % self.checkpoint_interval == 0:
                    self.save_checkpoint(embeddings, batch_end)

        # Calculate timing
        end_time = datetime.now()
        self.stats['gpu_time'] = (end_time - gpu_start).total_seconds()
        self.stats['total_time'] = (end_time - start_time).total_seconds()
        self.stats['processed_movies'] = total_movies

        logger.info(f"\n✅ Generated {total_movies:,} embeddings")

        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        movies: List[Dict]
    ) -> None:
        """
        Save embeddings and metadata.

        Args:
            embeddings: Embeddings array
            movies: List of movie dictionaries
        """
        logger.info(f"\n💾 Saving embeddings...")

        # Save embeddings as numpy array
        np.save(self.embeddings_path, embeddings)
        logger.info(f"✅ Saved embeddings: {self.embeddings_path}")

        file_size = self.embeddings_path.stat().st_size / (1024 * 1024 * 1024)
        logger.info(f"   File size: {file_size:.2f} GB")

        # Save metadata (JSONL)
        with open(self.metadata_path, 'w') as f:
            for movie in movies:
                metadata = {
                    'tmdb_id': movie['tmdb_id'],
                    'imdb_id': movie.get('imdb_id', ''),
                    'ml_id': movie.get('ml_id'),
                    'title': movie['title'],
                    'year': movie.get('year'),
                    'genres': movie.get('genres', [])
                }
                f.write(json.dumps(metadata) + '\n')

        logger.info(f"✅ Saved metadata: {self.metadata_path}")

        # Remove checkpoint
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("🗑️  Removed checkpoint (processing complete)")

    def validate_output(self) -> bool:
        """
        Validate output files.

        Returns:
            True if valid, False otherwise
        """
        logger.info("\n🔍 Validating output...")

        # Check embeddings file
        if not self.embeddings_path.exists():
            logger.error(f"❌ Embeddings file not found: {self.embeddings_path}")
            return False

        # Load and validate
        try:
            embeddings = np.load(self.embeddings_path)
            logger.info(f"✅ Embeddings shape: {embeddings.shape}")

            if embeddings.shape[0] != self.stats['total_movies']:
                logger.error(f"❌ Row count mismatch: {embeddings.shape[0]} != {self.stats['total_movies']}")
                return False

            if embeddings.shape[1] != self.stats['embedding_dim']:
                logger.error(f"❌ Dimension mismatch: {embeddings.shape[1]} != {self.stats['embedding_dim']}")
                return False

            # Check for NaN/Inf
            if np.isnan(embeddings).any():
                logger.error("❌ Found NaN values in embeddings")
                return False

            if np.isinf(embeddings).any():
                logger.error("❌ Found Inf values in embeddings")
                return False

            logger.info("✅ Embeddings validation passed")

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False

        # Check metadata file
        if not self.metadata_path.exists():
            logger.error(f"❌ Metadata file not found: {self.metadata_path}")
            return False

        logger.info("✅ Metadata validation passed")

        return True

    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 3: GPU EMBEDDING GENERATION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total movies:        {self.stats['total_movies']:,}")
        logger.info(f"Processed movies:    {self.stats['processed_movies']:,}")
        logger.info(f"Embedding dimension: {self.stats['embedding_dim']}")
        logger.info(f"GPU time:            {self.stats['gpu_time']:.2f} seconds")
        logger.info(f"Total time:          {self.stats['total_time']:.2f} seconds")
        logger.info(f"Throughput:          {self.stats['processed_movies']/self.stats['gpu_time']:.0f} movies/second")
        logger.info(f"Checkpoints saved:   {self.stats['checkpoints_saved']}")

        # File size
        if self.embeddings_path.exists():
            file_size = self.embeddings_path.stat().st_size / (1024 * 1024 * 1024)
            logger.info(f"Output file size:    {file_size:.2f} GB")

        logger.info("=" * 70)

    def run(self) -> bool:
        """
        Execute complete Stage 3 pipeline.

        Returns:
            True if successful, False otherwise
        """
        logger.info("⚡ TMDB GPU Embedding Generation - Stage 3")
        logger.info("=" * 70)

        try:
            # Create directories
            self.create_directories()

            # Initialize encoder
            self.initialize_encoder()

            # Load movies
            movies, texts = self.load_movies()

            # Check for checkpoint
            checkpoint_embeddings, start_index = self.load_checkpoint()

            # Generate embeddings
            embeddings = self.generate_embeddings(
                texts,
                start_index=start_index,
                checkpoint_embeddings=checkpoint_embeddings
            )

            # Save embeddings and metadata
            self.save_embeddings(embeddings, movies)

            # Validate output
            if not self.validate_output():
                return False

            # Print statistics
            self.print_statistics()

            logger.info("\n✅ Stage 3 complete!")
            return True

        except Exception as e:
            logger.error(f"\n❌ Stage 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TMDB GPU Embedding Generation - Stage 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run embedding generation
  python stage3_gpu_embeddings.py

  # Custom batch size
  python stage3_gpu_embeddings.py --batch-size 64

  # Custom checkpoint interval
  python stage3_gpu_embeddings.py --checkpoint-interval 5000

  # Resume from checkpoint
  python stage3_gpu_embeddings.py  # Automatically detects checkpoint

Output:
  data/embeddings/tmdb/content_vectors.npy (1.4GB)
  data/embeddings/tmdb/metadata.jsonl
        """
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to semantic-recommender directory'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for GPU processing (default: 32)'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Save checkpoint every N movies (default: 10000)'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = TMDBEmbeddingGenerator(
        base_path=args.base_path,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval
    )
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
