#!/bin/bash
# Comprehensive Benchmark Execution Script
# Runs all Criterion benchmarks and generates comparison reports

set -e

echo "=========================================="
echo "Rust Benchmark Suite"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: cargo not found. Please install Rust.${NC}"
    exit 1
fi

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

# Function to run a benchmark
run_benchmark() {
    local name=$1
    local description=$2

    echo -e "${YELLOW}Running: $description${NC}"
    cargo bench --bench "$name" 2>&1 | tee "$RESULTS_DIR/${name}_results.txt"
    echo -e "${GREEN}✓ $description complete${NC}"
    echo ""
}

# Run all benchmarks
echo "Starting benchmark suite..."
echo ""

run_benchmark "latency_benchmark" "Latency Benchmark (P50/P95/P99)"
run_benchmark "throughput_benchmark" "Throughput Benchmark (QPS)"
run_benchmark "memory_benchmark" "Memory Tracking Benchmark"
run_benchmark "cache_benchmark" "Cache Performance Benchmark"

echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""

# Generate summary
echo "Generating summary..."

SUMMARY_FILE="$RESULTS_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# Benchmark Results Summary

**Generated**: $(date)
**Platform**: $(uname -s) $(uname -m)
**Rust Version**: $(rustc --version)

## Benchmark Execution

EOF

for bench in latency throughput memory cache; do
    echo "- ✓ ${bench}_benchmark" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

## Results Location

- **HTML Reports**: \`target/criterion/report/index.html\`
- **Raw Results**: \`$RESULTS_DIR/\`

## Benchmark Categories

### 1. Latency Benchmark
- End-to-end query latency
- Component breakdown (encode, fusion, similarity, attention)
- Cold start vs warm cache
- Percentile statistics (P50, P95, P99)

**Target**: <5ms average latency

### 2. Throughput Benchmark
- Sequential QPS (10, 100, 1000 queries)
- Parallel QPS with rayon
- Batch size impact analysis
- vs Python baseline comparison
- Sustained throughput testing

**Target**: >200 QPS (2× Python's 94 QPS)

### 3. Memory Benchmark
- Allocation patterns
- Peak memory usage
- Memory leak detection
- Memory pressure testing
- Fragmentation analysis
- Bandwidth utilization

**Target**: Efficient memory usage with no leaks

### 4. Cache Benchmark
- Hit vs miss latency
- Hit rate measurement
- Cache rebuild time
- LRU eviction performance
- Working set size impact
- Concurrent access patterns

**Target**: >80% hit rate, <100ms rebuild time

## Viewing Results

### HTML Reports

\`\`\`bash
# Open in browser
firefox target/criterion/report/index.html

# Or start local server
cd target/criterion
python3 -m http.server 8000
# Visit http://localhost:8000/report/index.html
\`\`\`

### Compare with Baseline

\`\`\`bash
# Save current results as baseline
cargo bench -- --save-baseline rust-v1

# Compare with baseline
cargo bench -- --baseline rust-v1
\`\`\`

## Python vs Rust Comparison

See \`docs/BENCHMARK_COMPARISON.md\` for detailed comparison.

**Key Metrics**:
- Latency: Python 11.42ms → Rust target <5ms (2.3× improvement)
- Throughput: Python 94 QPS → Rust target >200 QPS (2× improvement)
- Memory: Python 609 MB → Rust target ~479 MB (21% savings)
- Cache: Python 420ms rebuild → Rust target <100ms (4× improvement)

## Next Steps

1. Review HTML reports for detailed performance analysis
2. Compare results with Python baseline
3. Identify performance bottlenecks
4. Optimize critical paths
5. Re-run benchmarks to validate improvements

---

**Benchmark Suite Version**: 1.0
**Dataset**: 62,423 movies (MovieLens)
**Embedding Dimension**: 768
EOF

echo -e "${GREEN}Summary generated: $SUMMARY_FILE${NC}"
echo ""

# Check if HTML reports were generated
if [ -d "target/criterion/report" ]; then
    echo -e "${GREEN}HTML reports available at:${NC}"
    echo "  target/criterion/report/index.html"
    echo ""
    echo "To view:"
    echo "  firefox target/criterion/report/index.html"
else
    echo -e "${YELLOW}Warning: HTML reports not found${NC}"
fi

echo ""
echo -e "${GREEN}✓ All benchmarks complete!${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "Summary: $SUMMARY_FILE"
echo "HTML reports: target/criterion/report/index.html"
echo ""
