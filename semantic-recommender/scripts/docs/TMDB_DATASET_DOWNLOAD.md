# TMDB Movies Dataset Download Guide

## Overview

Script to download the TMDB Movies Dataset (930K+ movies) from Kaggle for the semantic recommender system.

**Dataset**: `asaniczka/tmdb-movies-dataset-2023-930k-movies`
**Source**: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

## Prerequisites

### 1. Kaggle API Credentials

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. Save `kaggle.json` to `~/.kaggle/kaggle.json`
5. Set proper permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

Your `kaggle.json` should look like:
```json
{
  "username": "your-kaggle-username",
  "key": "your-api-key-here"
}
```

### 2. Install Kaggle CLI

```bash
pip install kaggle
```

Or install all dependencies:
```bash
pip install -r scripts/requirements.txt
```

## Usage

### Basic Download

```bash
cd semantic-recommender/scripts
python download_tmdb_dataset.py
```

This will:
- Check for Kaggle credentials
- Verify kaggle CLI is installed
- Skip download if files already exist
- Download to `data/raw/tmdb/`
- Unzip automatically
- Verify file integrity
- Generate summary report

### Force Re-download

```bash
python download_tmdb_dataset.py --force
```

### Custom Base Path

```bash
python download_tmdb_dataset.py --base-path /path/to/semantic-recommender
```

## Output

### Directory Structure

```
semantic-recommender/
└── data/
    └── raw/
        └── tmdb/
            ├── TMDB_movie_dataset_v11.csv
            └── download_summary.txt
```

### Expected Files

- **TMDB_movie_dataset_v11.csv**: Main dataset file with 930K+ movies
  - Size: ~500MB (uncompressed)
  - Columns: id, title, vote_average, vote_count, status, release_date, revenue, runtime, adult, backdrop_path, budget, homepage, imdb_id, original_language, original_title, overview, popularity, poster_path, tagline, genres, production_companies, production_countries, spoken_languages, keywords

### Download Summary

After successful download, a summary file is created at `data/raw/tmdb/download_summary.txt`:

```
======================================================================
TMDB MOVIES DATASET DOWNLOAD SUMMARY
======================================================================
Dataset ID: asaniczka/tmdb-movies-dataset-2023-930k-movies
Download Location: /path/to/data/raw/tmdb
Total Files: 1

Total Size: 500.00 MB

Files:
  - TMDB_movie_dataset_v11.csv
    Size: 500.00 MB
    Type: .csv
    MD5: <hash>

======================================================================
Next Steps:
1. Verify data quality with validate script
2. Process data with generate_platform_data.py
3. Generate embeddings with generate_embeddings.py
======================================================================
```

## Features

### ✅ Credential Verification
- Checks for `~/.kaggle/kaggle.json`
- Validates file permissions (should be 600)
- Verifies JSON format and required fields

### ✅ Resumable Downloads
- Skips download if files already exist
- Use `--force` flag to re-download

### ✅ Progress Tracking
- Shows download progress via Kaggle CLI
- Reports file sizes and verification status

### ✅ Integrity Verification
- MD5 hash calculation for smaller files
- File size reporting
- Completeness checks

### ✅ Error Handling
- Timeout protection (10 minute max)
- Clear error messages
- Automatic cleanup on failure

## Troubleshooting

### Error: "Kaggle credentials not found"

**Solution**: Follow prerequisite steps to create `~/.kaggle/kaggle.json`

### Error: "Kaggle CLI not installed"

**Solution**: Install with `pip install kaggle`

### Error: "403 Forbidden"

**Possible Causes**:
1. Invalid API credentials
2. Dataset requires acceptance of terms
3. Rate limiting

**Solution**:
1. Verify credentials at https://www.kaggle.com/settings/account
2. Visit dataset page and accept terms: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
3. Wait a few minutes and retry

### Error: "Download timed out"

**Solution**:
- Check internet connection
- Retry download (script will resume where it left off)
- Increase timeout in script if needed

### Warning: "permissions {permissions}"

**Solution**: Set correct permissions with `chmod 600 ~/.kaggle/kaggle.json`

## Dataset Information

### Content
- **930,000+ movies** from TMDB (The Movie Database)
- Updated regularly with latest releases
- Comprehensive metadata including genres, cast, crew, keywords

### Use Cases
1. Semantic search and recommendations
2. Content-based filtering
3. Collaborative filtering
4. Hybrid recommendation systems
5. Movie analytics and trends

### Data Fields

Key fields in TMDB dataset:
- **id**: Unique movie identifier
- **title**: Movie title
- **overview**: Plot summary (for embeddings)
- **genres**: Movie genres (JSON)
- **keywords**: Associated keywords (JSON)
- **vote_average**: Rating (1-10)
- **popularity**: Popularity score
- **release_date**: Release date
- **runtime**: Duration in minutes
- **budget/revenue**: Financial data

## Integration

### Next Steps After Download

1. **Validate Data Quality**
   ```bash
   python scripts/validate_data.py --source tmdb
   ```

2. **Generate Platform Data**
   ```bash
   python scripts/generate_platform_data.py --source tmdb
   ```

3. **Generate Embeddings**
   ```bash
   python scripts/generate_embeddings.py --source tmdb
   ```

4. **Populate Databases**
   ```bash
   python scripts/populate_milvus.py --source tmdb
   python scripts/populate_neo4j.py --source tmdb
   ```

## Performance

- **Download Time**: 2-5 minutes (depends on connection)
- **Disk Space**: ~1GB (compressed + uncompressed)
- **Memory**: <100MB during download
- **Network**: ~500MB download

## License

The TMDB dataset is provided under Kaggle's terms of service. Please review:
- Kaggle Terms: https://www.kaggle.com/terms
- TMDB API Terms: https://www.themoviedb.org/documentation/api/terms-of-use

## Support

For issues with:
- **Script**: Open issue in semantic-recommender repository
- **Dataset**: Contact dataset author on Kaggle
- **Kaggle API**: Visit https://github.com/Kaggle/kaggle-api

## Script Reference

**Location**: `scripts/download_tmdb_dataset.py`
**Class**: `TMDBDatasetDownloader`
**Dependencies**: `kaggle`, `pathlib`, `hashlib`, `subprocess`

### Public Methods

```python
downloader = TMDBDatasetDownloader(base_path=None)
downloader.check_kaggle_credentials()  # Returns: bool
downloader.check_kaggle_cli()          # Returns: bool
downloader.is_already_downloaded()     # Returns: bool
downloader.download_dataset()          # Returns: bool
downloader.verify_files()              # Returns: (bool, List[Dict])
downloader.run(force=False)            # Returns: bool
```

### Programmatic Usage

```python
from download_tmdb_dataset import TMDBDatasetDownloader

# Initialize
downloader = TMDBDatasetDownloader(
    base_path="/path/to/semantic-recommender"
)

# Run download workflow
success = downloader.run(force=False)

if success:
    # Verify and get file info
    verified, file_info = downloader.verify_files()
    for info in file_info:
        print(f"Downloaded: {info['name']} ({info['size_mb']:.2f} MB)")
else:
    print("Download failed")
```

## Related Files

- `.gitignore`: Excludes `data/raw/tmdb/` from version control
- `requirements.txt`: Includes `kaggle>=1.5.16` dependency
- `docs/DATA_SOURCES.md`: Dataset attribution and licensing

---

**Last Updated**: 2025-12-07
**Script Version**: 1.0.0
