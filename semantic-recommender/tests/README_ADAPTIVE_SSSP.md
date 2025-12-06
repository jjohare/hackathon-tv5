# Adaptive SSSP Test Suite

## Quick Start

```bash
# Run all tests
cargo test --test adaptive_sssp_tests

# Run with output
cargo test --test adaptive_sssp_tests -- --nocapture

# Run benchmark
cargo test --test adaptive_sssp_tests benchmark_adaptive_sssp -- --nocapture
```

## What's Tested

✅ **13 comprehensive tests** covering:

1. **Algorithm Selection** (3 tests)
   - Small graph (1K nodes) → GPU Dijkstra
   - Medium graph (100K nodes) → GPU Dijkstra  
   - Large graph (10M nodes) → Hybrid Duan

2. **Correctness** (3 tests)
   - GPU Dijkstra vs ground truth
   - Hybrid Duan vs ground truth
   - Both algorithms match each other

3. **Performance** (3 tests)
   - Benchmark comparison
   - Crossover point detection
   - Memory efficiency

4. **Adaptive Behavior** (2 tests)
   - Adaptive algorithm switching
   - Graceful fallback on errors

5. **Edge Cases** (1 test)
   - Single node, disconnected, self-loops

6. **Full Benchmark Suite** (1 test)
   - Multiple graph types and sizes

## Test Data

### Graph Generators
- **Erdős-Rényi**: Random graphs with edge probability
- **Scale-Free**: Preferential attachment (real-world networks)
- **Grid**: 2D lattice (spatial networks)

### Sizes Tested
- Small: 1K nodes
- Medium: 10K-100K nodes
- Large: 1M-10M nodes (simulated for selection logic)

## Documentation

- **Full Docs**: `tests/docs/ADAPTIVE_SSSP_TESTS.md`
- **Summary**: `tests/docs/TEST_SUMMARY.md`
- **Test File**: `tests/adaptive_sssp_tests.rs` (763 lines)

## Current Status

✅ Test structure complete
✅ Mock implementations functional
✅ All 13 tests implemented
⏳ Awaiting CUDA kernel integration

## Key Features

- **Adaptive Selection**: Automatically chooses best algorithm
- **Correctness Validation**: < 1e-5 error vs ground truth
- **Performance Benchmarks**: Timing and speedup measurements
- **Graceful Fallback**: Automatic recovery from failures
- **Edge Case Coverage**: Robust handling of corner cases

## Integration

When CUDA kernels are ready:
1. Replace `gpu_dijkstra()` mock with FFI call
2. Implement real `hybrid_duan_sssp()` with GPU/CPU split
3. Re-run benchmarks with actual CUDA performance
4. Tune crossover threshold (currently 1M nodes)

---

Created: 2025-12-04 | Status: ✅ Complete
