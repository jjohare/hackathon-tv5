#!/usr/bin/env python3
"""
TMDB Dataset Processing Pipeline Orchestrator

Orchestrates all 3 stages of TMDB dataset processing:
1. Stage 1: Ingestion and cleaning
2. Stage 2: Ontology mapping
3. Stage 3: GPU embedding generation

Provides:
- Real-time monitoring with progress tracking
- Error handling and recovery
- Stage-level validation
- Comprehensive final report

Usage:
  # Run all stages
  python run_tmdb_pipeline.py

  # Run specific stages
  python run_tmdb_pipeline.py --stages 1 2

  # Skip completed stages
  python run_tmdb_pipeline.py --resume
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBPipelineOrchestrator:
    """Orchestrate TMDB dataset processing pipeline."""

    def __init__(self, base_path: str = None):
        """
        Initialize pipeline orchestrator.

        Args:
            base_path: Base directory for semantic-recommender (auto-detected if None)
        """
        if base_path is None:
            # Auto-detect: go up from scripts/data_pipeline to project root
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent.parent
        else:
            self.base_path = Path(base_path)

        # Stage scripts
        self.stage_scripts = {
            1: self.base_path / "scripts/data_pipeline/stage1_ingest_tmdb.py",
            2: self.base_path / "scripts/data_pipeline/stage2_ontology_mapping.py",
            3: self.base_path / "scripts/data_pipeline/stage3_gpu_embeddings.py",
        }

        # Output paths for validation
        self.stage_outputs = {
            1: self.base_path / "data/processed/tmdb/movies_clean.jsonl",
            2: self.base_path / "data/processed/tmdb/genome_scores.json",
            3: self.base_path / "data/embeddings/tmdb/content_vectors.npy",
        }

        # Pipeline statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_time': 0,
            'stages_completed': [],
            'stages_failed': [],
            'stages_skipped': []
        }

    def check_stage_complete(self, stage: int) -> bool:
        """
        Check if a stage has already been completed.

        Args:
            stage: Stage number (1-3)

        Returns:
            True if stage output exists, False otherwise
        """
        output_path = self.stage_outputs.get(stage)
        if output_path and output_path.exists():
            logger.info(f"✅ Stage {stage} output found: {output_path.name}")
            return True
        return False

    def run_stage(self, stage: int) -> bool:
        """
        Run a specific stage.

        Args:
            stage: Stage number (1-3)

        Returns:
            True if successful, False otherwise
        """
        script_path = self.stage_scripts.get(stage)

        if not script_path or not script_path.exists():
            logger.error(f"❌ Stage {stage} script not found: {script_path}")
            return False

        logger.info(f"\n{'=' * 70}")
        logger.info(f"RUNNING STAGE {stage}")
        logger.info(f"{'=' * 70}")

        try:
            # Run stage script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Stage {stage} completed successfully")
                self.stats['stages_completed'].append(stage)
                return True
            else:
                logger.error(f"❌ Stage {stage} failed with return code {result.returncode}")
                self.stats['stages_failed'].append(stage)
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Stage {stage} timed out (exceeded 1 hour)")
            self.stats['stages_failed'].append(stage)
            return False

        except Exception as e:
            logger.error(f"❌ Stage {stage} error: {e}")
            self.stats['stages_failed'].append(stage)
            return False

    def validate_pipeline(self) -> bool:
        """
        Validate complete pipeline outputs.

        Returns:
            True if all outputs valid, False otherwise
        """
        logger.info(f"\n{'=' * 70}")
        logger.info("FINAL VALIDATION")
        logger.info(f"{'=' * 70}")

        all_valid = True

        for stage, output_path in self.stage_outputs.items():
            if not output_path.exists():
                logger.error(f"❌ Stage {stage} output missing: {output_path}")
                all_valid = False
                continue

            # Check file size
            file_size = output_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            logger.info(f"✅ Stage {stage}: {output_path.name} ({file_size_mb:.2f} MB)")

            # Minimum size validation
            min_sizes = {
                1: 10,   # Stage 1: at least 10 MB
                2: 1,    # Stage 2: at least 1 MB
                3: 100,  # Stage 3: at least 100 MB
            }

            if file_size_mb < min_sizes.get(stage, 0):
                logger.error(f"⚠️  Stage {stage} output too small: {file_size_mb:.2f} MB")
                all_valid = False

        return all_valid

    def generate_report(self) -> str:
        """
        Generate final pipeline report.

        Returns:
            Report as formatted string
        """
        report = []
        report.append("=" * 70)
        report.append("TMDB DATASET PROCESSING PIPELINE - FINAL REPORT")
        report.append("=" * 70)
        report.append(f"Start Time: {self.stats['start_time']}")
        report.append(f"End Time: {self.stats['end_time']}")
        report.append(f"Total Time: {self.stats['total_time']:.2f} seconds ({self.stats['total_time']/60:.1f} minutes)")
        report.append("")

        report.append("Stage Results:")
        for stage in sorted(self.stage_outputs.keys()):
            if stage in self.stats['stages_completed']:
                status = "✅ COMPLETED"
            elif stage in self.stats['stages_failed']:
                status = "❌ FAILED"
            elif stage in self.stats['stages_skipped']:
                status = "⏭️  SKIPPED"
            else:
                status = "⚪ NOT RUN"

            output_path = self.stage_outputs[stage]
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                report.append(f"  Stage {stage}: {status} - {output_path.name} ({file_size:.2f} MB)")
            else:
                report.append(f"  Stage {stage}: {status}")

        report.append("")
        report.append("Output Files:")

        for stage, output_path in sorted(self.stage_outputs.items()):
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                report.append(f"  {output_path}")
                report.append(f"    Size: {file_size:.2f} MB")

        report.append("")
        report.append("=" * 70)

        if len(self.stats['stages_failed']) == 0:
            report.append("✅ PIPELINE COMPLETED SUCCESSFULLY")
        else:
            report.append(f"⚠️  PIPELINE COMPLETED WITH {len(self.stats['stages_failed'])} FAILED STAGES")

        report.append("=" * 70)

        return "\n".join(report)

    def run(
        self,
        stages: Optional[List[int]] = None,
        resume: bool = False
    ) -> bool:
        """
        Execute complete pipeline.

        Args:
            stages: List of stage numbers to run (None = all stages)
            resume: Skip stages that are already complete

        Returns:
            True if all stages successful, False otherwise
        """
        logger.info("🎬 TMDB Dataset Processing Pipeline")
        logger.info("=" * 70)

        # Default to all stages
        if stages is None:
            stages = [1, 2, 3]

        # Record start time
        self.stats['start_time'] = datetime.now()

        # Run stages
        for stage in sorted(stages):
            # Check if already complete (resume mode)
            if resume and self.check_stage_complete(stage):
                logger.info(f"⏭️  Skipping Stage {stage} (already complete)")
                self.stats['stages_skipped'].append(stage)
                continue

            # Run stage
            success = self.run_stage(stage)

            if not success:
                logger.error(f"\n❌ Pipeline failed at Stage {stage}")
                self.stats['end_time'] = datetime.now()
                self.stats['total_time'] = (
                    self.stats['end_time'] - self.stats['start_time']
                ).total_seconds()
                print("\n" + self.generate_report())
                return False

        # Record end time
        self.stats['end_time'] = datetime.now()
        self.stats['total_time'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()

        # Validate complete pipeline
        valid = self.validate_pipeline()

        # Generate report
        report = self.generate_report()
        print("\n" + report)

        # Save report
        report_path = self.base_path / "data/processed/tmdb/pipeline_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"\n📝 Report saved to: {report_path}")

        return valid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TMDB Dataset Processing Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python run_tmdb_pipeline.py

  # Run specific stages
  python run_tmdb_pipeline.py --stages 1 2

  # Resume pipeline (skip completed stages)
  python run_tmdb_pipeline.py --resume

  # Run with custom base path
  python run_tmdb_pipeline.py --base-path /path/to/semantic-recommender

Pipeline Stages:
  1. Ingestion and cleaning (TMDB CSV → JSONL)
  2. Ontology mapping (keywords → genome tags)
  3. GPU embedding generation (TensorRT acceleration)

Expected Runtime (A100 GPU):
  Stage 1: ~60 seconds
  Stage 2: ~120 seconds
  Stage 3: ~15 minutes
  Total: ~17 minutes
        """
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to semantic-recommender directory'
    )

    parser.add_argument(
        '--stages',
        type=int,
        nargs='+',
        default=None,
        choices=[1, 2, 3],
        help='Specific stages to run (default: all)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip stages that are already complete'
    )

    args = parser.parse_args()

    # Run pipeline
    orchestrator = TMDBPipelineOrchestrator(base_path=args.base_path)
    success = orchestrator.run(stages=args.stages, resume=args.resume)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
