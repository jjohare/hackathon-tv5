# Semantic Recommender CLI

GPU-accelerated semantic movie recommendation system with command-line interface.

## Installation

### From Source

```bash
# Build with CPU support only
cargo build --release

# Build with CUDA GPU support
cargo build --release --features gpu

# Build optimized for A100
cargo build --profile a100 --features gpu
```

Binary will be available at `target/release/semantic-rec`

## Quick Start

```bash
# Run demo test
semantic-rec test

# Search for movies
semantic-rec query "action thriller with car chases"

# Interactive mode
semantic-rec interactive

# Load full dataset
semantic-rec load --dataset data/movies.csv

# Run benchmarks
semantic-rec bench --iterations 1000

# System information
semantic-rec info
```

## Commands

### `test`
Run demonstration query with sample data.

```bash
semantic-rec test --query "romantic comedy" --limit 10
```

**Options:**
- `--query, -q <TEXT>` - Query text (default: "action thriller with car chases")
- `--limit, -k <NUM>` - Number of results (default: 10)

### `query`
Execute single search query.

```bash
semantic-rec query "sci-fi with AI themes" --limit 20 --explain
```

**Options:**
- `TEXT` - Search query (required)
- `--limit, -k <NUM>` - Number of results (default: 10)
- `--explain` - Show detailed explanations

### `load`
Load and index full movie dataset.

```bash
semantic-rec load \
    --dataset data/movies.csv \
    --embeddings data/embeddings.npy \
    --force
```

**Options:**
- `--dataset <PATH>` - Movie dataset CSV (default: data/movies.csv)
- `--embeddings <PATH>` - Pre-computed embeddings file (optional)
- `--force` - Force recompute embeddings

### `bench`
Run comprehensive benchmarks.

```bash
semantic-rec bench \
    --iterations 1000 \
    --compare-python \
    --output results.json
```

**Options:**
- `--iterations, -i <NUM>` - Number of iterations (default: 100)
- `--compare-python` - Compare with Python baseline
- `--output, -o <FILE>` - Save results to file

### `compare`
Compare Rust vs Python implementations.

```bash
semantic-rec compare --queries 50 --threshold 0.01
```

**Options:**
- `--queries, -q <NUM>` - Number of test queries (default: 50)
- `--threshold <FLOAT>` - Acceptable difference (default: 0.01)

### `interactive`
Start interactive query mode.

```bash
semantic-rec interactive
```

Type queries and get instant results. Type `quit` to exit.

### `info`
Display system information and GPU status.

```bash
semantic-rec info
```

Shows CPU, memory, GPU info, and build configuration.

## Global Options

Available for all commands:

- `--device <TYPE>` - Device selection: `cpu`, `cuda`, `auto` (default: auto)
- `--verbose, -v` - Enable verbose logging
- `--output <FORMAT>` - Output format: `table`, `json`, `csv` (default: table)

## Examples

### Basic Usage

```bash
# Quick test
semantic-rec test

# Search with GPU
semantic-rec --device cuda query "action movie"

# JSON output
semantic-rec --output json query "comedy"
```

### Advanced Workflows

```bash
# Load dataset and run benchmarks
semantic-rec load --dataset data/movies.csv
semantic-rec bench --iterations 1000 --output bench.json

# Compare implementations
semantic-rec compare --queries 100 --threshold 0.005

# Interactive exploration
semantic-rec --device cuda interactive
```

### Performance Testing

```bash
# Benchmark single queries
semantic-rec bench --iterations 10000

# Benchmark with Python comparison
semantic-rec bench --compare-python --output comparison.json

# Memory stress test
cargo test --release --test integration_test test_gpu_memory_stress
```

## Dataset Format

The CLI expects CSV format with these columns:

```csv
id,title,description,genres,year,rating
1,"The Matrix","A hacker discovers reality is simulated","Action,Sci-Fi",1999,8.7
```

**Required fields:**
- `id` - Unique identifier
- `title` - Movie title

**Optional fields:**
- `description` - Plot summary
- `genres` - Comma-separated genres
- `year` - Release year
- `rating` - User rating

## Output Formats

### Table (Default)

```
┌──────┬─────────────────┬─────────┐
│ Rank │ Title           │ Score   │
├──────┼─────────────────┼─────────┤
│ 1    │ The Matrix      │ 0.9542  │
│ 2    │ Inception       │ 0.9234  │
└──────┴─────────────────┴─────────┘
```

### JSON

```json
{
  "query": "action thriller",
  "results": [
    {"title": "The Matrix", "score": 0.9542},
    {"title": "Inception", "score": 0.9234}
  ],
  "elapsed_ms": 15.3,
  "count": 2
}
```

### CSV

```csv
rank,title,score
1,The Matrix,0.9542
2,Inception,0.9234
```

## Performance

### Expected Performance (A100 GPU)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Single query | < 5ms | 200+ QPS |
| Batch (10 queries) | < 20ms | 500+ QPS |
| Index 62K movies | < 2s | - |

### Memory Requirements

- **CPU**: ~2GB RAM
- **GPU**: ~4GB VRAM (for 62K movies with 768-dim embeddings)

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation
nvidia-smi

# Rebuild with CPU fallback
cargo build --release --features cpu-only
```

### Out of Memory

```bash
# Reduce batch size or use CPU
semantic-rec --device cpu query "test"

# Or increase GPU memory pool size (coming soon)
```

### Slow Performance

```bash
# Verify GPU is being used
semantic-rec info

# Check if debug build (use --release)
cargo build --release --features gpu
```

## Development

### Build from Source

```bash
git clone <repo-url>
cd semantic-recommender/crates/cli
cargo build --release
```

### Run Tests

```bash
# Unit tests
cargo test

# Integration tests (requires dataset)
cargo test --test integration_test -- --ignored

# Benchmarks
cargo bench
```

### Debug Logging

```bash
# Enable verbose output
RUST_LOG=debug semantic-rec --verbose test
```

## License

MIT License - see LICENSE file for details.

## Support

- Issues: [GitHub Issues]
- Documentation: [Full Docs]
- API Reference: `cargo doc --open`
