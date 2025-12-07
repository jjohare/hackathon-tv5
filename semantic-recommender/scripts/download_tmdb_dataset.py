#!/usr/bin/env python3
"""
TMDB Movies Dataset Downloader
Downloads the TMDB Movies Dataset from Kaggle with verification and progress tracking.

Dataset: asaniczka/tmdb-movies-dataset-2023-930k-movies
Source: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

Requirements:
    - Kaggle API credentials at ~/.kaggle/kaggle.json
    - kaggle package installed (pip install kaggle)
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import shutil


class TMDBDatasetDownloader:
    """Download and verify TMDB Movies Dataset from Kaggle."""

    DATASET_ID = "asaniczka/tmdb-movies-dataset-2023-930k-movies"

    def __init__(self, base_path: str = None):
        """
        Initialize downloader.

        Args:
            base_path: Base directory for semantic-recommender (auto-detected if None)
        """
        if base_path is None:
            # Auto-detect: go up from scripts/ to project root
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent
        else:
            self.base_path = Path(base_path)

        self.data_dir = self.base_path / "data" / "raw" / "tmdb"
        self.kaggle_config = Path.home() / ".kaggle" / "kaggle.json"

    def check_kaggle_credentials(self) -> bool:
        """
        Check if Kaggle API credentials exist and are valid.

        Returns:
            True if credentials exist, False otherwise
        """
        if not self.kaggle_config.exists():
            print(f"❌ Kaggle credentials not found at {self.kaggle_config}")
            print("\nTo set up Kaggle API credentials:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New Token'")
            print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
            print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False

        # Verify file permissions (should be 600)
        stat = os.stat(self.kaggle_config)
        permissions = oct(stat.st_mode)[-3:]
        if permissions != "600":
            print(f"⚠️  Warning: Kaggle credentials have permissions {permissions}")
            print(f"   Recommended: chmod 600 {self.kaggle_config}")

        # Verify JSON format
        try:
            with open(self.kaggle_config, 'r') as f:
                creds = json.load(f)
                if 'username' not in creds or 'key' not in creds:
                    print("❌ Invalid kaggle.json format (missing username or key)")
                    return False
        except json.JSONDecodeError:
            print("❌ Invalid JSON in kaggle.json")
            return False

        print(f"✅ Kaggle credentials found: {creds['username']}")
        return True

    def check_kaggle_cli(self) -> bool:
        """
        Check if kaggle CLI is installed.

        Returns:
            True if kaggle CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✅ Kaggle CLI installed: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        print("❌ Kaggle CLI not installed")
        print("   Install with: pip install kaggle")
        return False

    def is_already_downloaded(self) -> bool:
        """
        Check if dataset is already downloaded.

        Returns:
            True if dataset exists, False otherwise
        """
        if not self.data_dir.exists():
            return False

        # Check for expected files
        expected_files = [
            "TMDB_movie_dataset_v11.csv",
        ]

        existing_files = list(self.data_dir.glob("*.csv"))
        if len(existing_files) > 0:
            print(f"✅ Found {len(existing_files)} CSV files in {self.data_dir}")
            for f in existing_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.1f} MB)")
            return True

        return False

    def create_directories(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {self.data_dir}")

    def download_dataset(self) -> bool:
        """
        Download dataset using Kaggle CLI.

        Returns:
            True if download successful, False otherwise
        """
        print(f"\n📥 Downloading dataset: {self.DATASET_ID}")
        print(f"   Destination: {self.data_dir}")

        try:
            # Use kaggle datasets download command
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', self.DATASET_ID,
                '-p', str(self.data_dir),
                '--unzip'
            ]

            print(f"   Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ Download failed:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return False

            print("✅ Download completed")
            if result.stdout:
                print(f"   {result.stdout.strip()}")

            return True

        except subprocess.TimeoutExpired:
            print("❌ Download timed out (exceeded 10 minutes)")
            return False
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

    def verify_files(self) -> Tuple[bool, List[Dict]]:
        """
        Verify downloaded files and gather metadata.

        Returns:
            Tuple of (success, file_info_list)
        """
        print("\n🔍 Verifying downloaded files...")

        if not self.data_dir.exists():
            print(f"❌ Directory not found: {self.data_dir}")
            return False, []

        # Get all files
        all_files = list(self.data_dir.glob("*"))
        if not all_files:
            print(f"❌ No files found in {self.data_dir}")
            return False, []

        file_info = []
        total_size = 0

        for filepath in sorted(all_files):
            if filepath.is_file():
                size = filepath.stat().st_size
                total_size += size

                info = {
                    'name': filepath.name,
                    'path': str(filepath),
                    'size_bytes': size,
                    'size_mb': size / (1024 * 1024),
                    'extension': filepath.suffix,
                }

                # Calculate MD5 for smaller files
                if size < 100 * 1024 * 1024:  # < 100MB
                    info['md5'] = self._calculate_md5(filepath)

                file_info.append(info)

        print(f"✅ Found {len(file_info)} files:")
        for info in file_info:
            print(f"   - {info['name']}")
            print(f"     Size: {info['size_mb']:.2f} MB")
            if 'md5' in info:
                print(f"     MD5: {info['md5']}")

        print(f"\n📊 Total size: {total_size / (1024 * 1024):.2f} MB")

        return True, file_info

    def _calculate_md5(self, filepath: Path) -> str:
        """
        Calculate MD5 hash of file.

        Args:
            filepath: Path to file

        Returns:
            MD5 hash as hex string
        """
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def generate_summary(self, file_info: List[Dict]) -> str:
        """
        Generate download summary.

        Args:
            file_info: List of file information dictionaries

        Returns:
            Summary as formatted string
        """
        summary = []
        summary.append("=" * 70)
        summary.append("TMDB MOVIES DATASET DOWNLOAD SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Dataset ID: {self.DATASET_ID}")
        summary.append(f"Download Location: {self.data_dir}")
        summary.append(f"Total Files: {len(file_info)}")
        summary.append("")

        total_size = sum(info['size_bytes'] for info in file_info)
        summary.append(f"Total Size: {total_size / (1024 * 1024):.2f} MB")
        summary.append("")

        summary.append("Files:")
        for info in file_info:
            summary.append(f"  - {info['name']}")
            summary.append(f"    Size: {info['size_mb']:.2f} MB")
            summary.append(f"    Type: {info['extension']}")
            if 'md5' in info:
                summary.append(f"    MD5: {info['md5']}")
            summary.append("")

        summary.append("=" * 70)
        summary.append("Next Steps:")
        summary.append("1. Verify data quality with validate script")
        summary.append("2. Process data with generate_platform_data.py")
        summary.append("3. Generate embeddings with generate_embeddings.py")
        summary.append("=" * 70)

        return "\n".join(summary)

    def run(self, force: bool = False) -> bool:
        """
        Execute complete download workflow.

        Args:
            force: If True, re-download even if already exists

        Returns:
            True if successful, False otherwise
        """
        print("🎬 TMDB Movies Dataset Downloader")
        print("=" * 70)

        # Check credentials
        if not self.check_kaggle_credentials():
            return False

        # Check CLI
        if not self.check_kaggle_cli():
            return False

        # Check if already downloaded
        if not force and self.is_already_downloaded():
            print("\n✅ Dataset already downloaded (use --force to re-download)")
            success, file_info = self.verify_files()
            if success:
                print("\n" + self.generate_summary(file_info))
            return True

        # Create directories
        self.create_directories()

        # Download
        if not self.download_dataset():
            return False

        # Verify
        success, file_info = self.verify_files()
        if not success:
            return False

        # Print summary
        print("\n" + self.generate_summary(file_info))

        # Save summary to file
        summary_file = self.data_dir / "download_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self.generate_summary(file_info))
        print(f"\n📝 Summary saved to: {summary_file}")

        return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download TMDB Movies Dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset (skip if already exists)
  python download_tmdb_dataset.py

  # Force re-download
  python download_tmdb_dataset.py --force

  # Custom base path
  python download_tmdb_dataset.py --base-path /path/to/semantic-recommender

Requirements:
  1. Kaggle API credentials at ~/.kaggle/kaggle.json
  2. pip install kaggle
        """
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if already exists'
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to semantic-recommender directory'
    )

    args = parser.parse_args()

    downloader = TMDBDatasetDownloader(base_path=args.base_path)
    success = downloader.run(force=args.force)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
