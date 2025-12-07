#!/usr/bin/env python3
"""
TMDB Pipeline Testing Script

Quick validation tests for each stage without processing full dataset.
Useful for development and CI/CD.

Usage:
  # Test all stages
  python test_pipeline.py

  # Test specific stage
  python test_pipeline.py --stage 1
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from stage1_ingest_tmdb import TMDBDataIngestion
from stage2_ontology_mapping import TMDBOntologyMapping
from stage3_gpu_embeddings import TMDBEmbeddingGenerator


class PipelineTest:
    """Test suite for TMDB pipeline."""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.temp_dir = None
        self.results = {
            'stage1': None,
            'stage2': None,
            'stage3': None
        }

    def setup_temp_dir(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp(prefix='tmdb_pipeline_test_')
        print(f"📁 Test directory: {self.temp_dir}")

    def cleanup_temp_dir(self):
        """Remove temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🗑️  Cleaned up: {self.temp_dir}")

    def test_stage1_functions(self) -> bool:
        """Test Stage 1 helper functions."""
        print("\n" + "=" * 70)
        print("Testing Stage 1: Ingestion Functions")
        print("=" * 70)

        try:
            pipeline = TMDBDataIngestion(base_path=self.base_path)

            # Test JSON parsing
            genres_str = '[{"id": 28, "name": "Action"}, {"id": 18, "name": "Drama"}]'
            genres = pipeline.parse_json_field(genres_str, 'genres')
            assert genres == ['Action', 'Drama'], f"Expected ['Action', 'Drama'], got {genres}"
            print("✅ JSON parsing: PASS")

            # Test year extraction
            year1 = pipeline.extract_year("2020-05-15")
            assert year1 == 2020, f"Expected 2020, got {year1}"

            year2 = pipeline.extract_year("May 15, 2020")
            assert year2 == 2020, f"Expected 2020, got {year2}"

            year3 = pipeline.extract_year("")
            assert year3 is None, f"Expected None, got {year3}"
            print("✅ Year extraction: PASS")

            self.results['stage1'] = True
            return True

        except Exception as e:
            print(f"❌ Stage 1 test failed: {e}")
            self.results['stage1'] = False
            return False

    def test_stage2_functions(self) -> bool:
        """Test Stage 2 ontology mapping functions."""
        print("\n" + "=" * 70)
        print("Testing Stage 2: Ontology Mapping Functions")
        print("=" * 70)

        try:
            pipeline = TMDBOntologyMapping(base_path=self.base_path)

            # Check if genome vocabulary can be loaded
            if pipeline.ml_genome_path.exists():
                pipeline.load_genome_vocabulary()
                print(f"✅ Genome vocabulary loaded: {len(pipeline.genome_tags)} tags")

                # Test keyword mappings
                pipeline.build_keyword_mappings()
                print(f"✅ Keyword mappings built: {len(pipeline.keyword_mappings)} mappings")

                # Test theme extraction
                overview = "A story of revenge and betrayal in a dark world of crime."
                themes = pipeline.extract_themes_from_overview(overview)
                print(f"✅ Theme extraction: {themes}")

                self.results['stage2'] = True
                return True
            else:
                print("⚠️  MovieLens genome scores not found, skipping Stage 2 test")
                self.results['stage2'] = None
                return True

        except Exception as e:
            print(f"❌ Stage 2 test failed: {e}")
            self.results['stage2'] = False
            return False

    def test_stage3_initialization(self) -> bool:
        """Test Stage 3 encoder initialization."""
        print("\n" + "=" * 70)
        print("Testing Stage 3: GPU Embedding Generation")
        print("=" * 70)

        try:
            pipeline = TMDBEmbeddingGenerator(
                base_path=self.base_path,
                batch_size=4
            )

            # Test encoder initialization
            pipeline.initialize_encoder()
            print(f"✅ Encoder initialized")
            print(f"   Using TensorRT: {pipeline.encoder.use_tensorrt}")
            print(f"   Embedding dim: {pipeline.stats['embedding_dim']}")

            # Test small batch encoding
            test_texts = [
                "A thrilling action movie",
                "A romantic comedy",
                "A dark mystery thriller"
            ]
            embeddings = pipeline.encoder.encode(test_texts)
            print(f"✅ Test encoding: shape={embeddings.shape}")

            assert embeddings.shape[0] == 3, f"Expected 3 embeddings, got {embeddings.shape[0]}"
            assert embeddings.shape[1] == pipeline.stats['embedding_dim'], \
                f"Expected dim {pipeline.stats['embedding_dim']}, got {embeddings.shape[1]}"

            # Check for NaN/Inf
            assert not np.isnan(embeddings.cpu().numpy()).any(), "Found NaN values"
            assert not np.isinf(embeddings.cpu().numpy()).any(), "Found Inf values"
            print("✅ Embedding validation: PASS")

            self.results['stage3'] = True
            return True

        except Exception as e:
            print(f"❌ Stage 3 test failed: {e}")
            self.results['stage3'] = False
            return False

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        for stage, result in self.results.items():
            if result is True:
                status = "✅ PASS"
            elif result is False:
                status = "❌ FAIL"
            else:
                status = "⏭️  SKIP"

            print(f"{stage.upper()}: {status}")

        print("=" * 70)

        # Overall result
        failures = [k for k, v in self.results.items() if v is False]
        if failures:
            print(f"\n❌ {len(failures)} test(s) failed")
            return False
        else:
            print("\n✅ All tests passed")
            return True

    def run(self, stage: int = None) -> bool:
        """
        Run tests.

        Args:
            stage: Specific stage to test (None = all stages)

        Returns:
            True if all tests passed, False otherwise
        """
        print("🧪 TMDB Pipeline Testing")
        print("=" * 70)

        try:
            self.setup_temp_dir()

            # Run tests
            if stage is None or stage == 1:
                self.test_stage1_functions()

            if stage is None or stage == 2:
                self.test_stage2_functions()

            if stage is None or stage == 3:
                self.test_stage3_initialization()

            # Print summary
            success = self.print_summary()

            return success

        finally:
            self.cleanup_temp_dir()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TMDB Pipeline Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='Test specific stage (default: all)'
    )

    args = parser.parse_args()

    # Run tests
    tester = PipelineTest()
    success = tester.run(stage=args.stage)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
