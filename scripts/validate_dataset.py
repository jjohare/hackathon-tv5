#!/usr/bin/env python3
"""
Validate synthetic dataset quality and distribution properties.
Generates comprehensive quality report with visualizations.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
from collections import Counter
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate synthetic dataset quality."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.results = {}

    def validate_zipf_distribution(self, data, field='popularity'):
        """Check if popularity follows Zipf's law."""
        logger.info(f"Validating Zipf distribution for {field}")

        sorted_values = np.sort(data[field])[::-1]
        ranks = np.arange(1, len(sorted_values) + 1)

        # Fit Zipf: log(y) = -alpha * log(x) + c
        log_ranks = np.log(ranks)
        log_values = np.log(sorted_values + 1e-10)  # Avoid log(0)

        # Linear regression
        coeffs = np.polyfit(log_ranks, log_values, 1)
        alpha = -coeffs[0]

        self.results['zipf_alpha'] = alpha
        logger.info(f"  Zipf alpha: {alpha:.3f} (expected ~1.5)")

        return 1.0 <= alpha <= 2.0

    def validate_temporal_patterns(self, data):
        """Validate temporal patterns in timestamps."""
        logger.info("Validating temporal patterns")

        if 'timestamp' not in data.columns:
            logger.warning("  No timestamp column found")
            return True

        # Convert to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Check daily patterns
        data['hour'] = data['timestamp'].dt.hour
        hourly_counts = data['hour'].value_counts().sort_index()

        # Peak hours should be morning (8-10), lunch (12-14), evening (19-22)
        peak_hours = [8, 9, 12, 13, 20, 21]
        peak_activity = hourly_counts[peak_hours].sum()
        total_activity = hourly_counts.sum()

        peak_ratio = peak_activity / total_activity
        self.results['peak_hour_ratio'] = peak_ratio
        logger.info(f"  Peak hour activity: {peak_ratio:.2%} (expected >30%)")

        return peak_ratio > 0.3

    def validate_cultural_diversity(self, data):
        """Validate cultural diversity in content."""
        logger.info("Validating cultural diversity")

        if 'cultural_context' not in data.columns:
            logger.warning("  No cultural_context column found")
            return True

        # Parse cultural contexts
        regions = []
        languages = []

        for ctx in data['cultural_context']:
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
            regions.append(ctx.get('region', 'Unknown'))
            languages.append(ctx.get('language', 'Unknown'))

        # Calculate diversity (Shannon entropy)
        region_counts = Counter(regions)
        region_probs = np.array(list(region_counts.values())) / len(regions)
        region_entropy = -np.sum(region_probs * np.log(region_probs))

        language_counts = Counter(languages)
        language_probs = np.array(list(language_counts.values())) / len(languages)
        language_entropy = -np.sum(language_probs * np.log(language_probs))

        self.results['region_diversity'] = region_entropy
        self.results['language_diversity'] = language_entropy

        logger.info(f"  Region diversity (entropy): {region_entropy:.3f}")
        logger.info(f"  Language diversity (entropy): {language_entropy:.3f}")

        # High entropy means good diversity
        return region_entropy > 1.0 and language_entropy > 1.0

    def validate_embeddings(self, data):
        """Validate embedding properties."""
        logger.info("Validating embeddings")

        if 'embedding' not in data.columns:
            logger.warning("  No embedding column found")
            return True

        embeddings = np.stack(data['embedding'].values)

        # Check dimensions
        assert embeddings.shape[1] == 1024, f"Expected 1024 dims, got {embeddings.shape[1]}"

        # Check for NaN/Inf
        has_nan = np.isnan(embeddings).any()
        has_inf = np.isinf(embeddings).any()

        if has_nan or has_inf:
            logger.error(f"  Found NaN: {has_nan}, Inf: {has_inf}")
            return False

        # Check value range
        emb_min = embeddings.min()
        emb_max = embeddings.max()
        emb_mean = embeddings.mean()
        emb_std = embeddings.std()

        self.results['embedding_min'] = float(emb_min)
        self.results['embedding_max'] = float(emb_max)
        self.results['embedding_mean'] = float(emb_mean)
        self.results['embedding_std'] = float(emb_std)

        logger.info(f"  Range: [{emb_min:.3f}, {emb_max:.3f}]")
        logger.info(f"  Mean: {emb_mean:.3f}, Std: {emb_std:.3f}")

        return True

    def validate_consistency(self, data):
        """Validate data consistency."""
        logger.info("Validating data consistency")

        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.warning(f"  Found {total_missing} missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.warning(f"    {col}: {count}")

        # Check for duplicates
        if 'content_id' in data.columns:
            duplicates = data['content_id'].duplicated().sum()
            logger.info(f"  Duplicate content_ids: {duplicates}")
            self.results['duplicate_ids'] = int(duplicates)

        return total_missing == 0

    def generate_report(self):
        """Generate validation report."""
        logger.info("\n" + "="*50)
        logger.info("VALIDATION REPORT")
        logger.info("="*50)

        for key, value in self.results.items():
            logger.info(f"{key}: {value}")

        # Save to JSON
        report_path = self.data_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nReport saved to: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate synthetic dataset")
    parser.add_argument("--data-dir", type=str,
                        default="/home/devuser/workspace/hackathon-tv5/data/embedded",
                        help="Directory with parquet files")

    args = parser.parse_args()

    validator = DatasetValidator(args.data_dir)

    # Find parquet files
    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        logger.error(f"No parquet files found in {data_dir}")
        return

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Validate first file as sample
    sample_file = parquet_files[0]
    logger.info(f"Validating sample file: {sample_file}")

    table = pq.read_table(sample_file)
    df = table.to_pandas()

    logger.info(f"Sample size: {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")

    # Run validations
    all_valid = True

    all_valid &= validator.validate_consistency(df)

    if 'popularity' in df.columns:
        all_valid &= validator.validate_zipf_distribution(df)

    if 'cultural_context' in df.columns:
        all_valid &= validator.validate_cultural_diversity(df)

    if 'embedding' in df.columns:
        all_valid &= validator.validate_embeddings(df)

    # Generate report
    validator.generate_report()

    if all_valid:
        logger.info("\n✓ All validations passed!")
    else:
        logger.warning("\n⚠ Some validations failed. Review report.")


if __name__ == "__main__":
    main()
