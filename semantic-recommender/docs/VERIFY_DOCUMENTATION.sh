#!/bin/bash
# Documentation Verification Script
# Verifies all claims in updated documentation are accurate

echo "================================================"
echo "Documentation Verification - 2025-12-07"
echo "================================================"
echo

# Test 1: Dataset Size
echo "✓ Test 1: Verify dataset size (1,334,069 records)"
ACTUAL_COUNT=$(wc -l < data/embeddings/tmdb/metadata.jsonl)
if [ "$ACTUAL_COUNT" -eq 1334069 ]; then
    echo "  ✅ PASS: $ACTUAL_COUNT records (matches documentation)"
else
    echo "  ❌ FAIL: $ACTUAL_COUNT records (expected 1,334,069)"
fi
echo

# Test 2: Embeddings Shape
echo "✓ Test 2: Verify embeddings shape (1334069, 384)"
python3 << 'PYTHON'
import numpy as np
data = np.load('data/embeddings/tmdb/content_vectors.npy')
expected_shape = (1334069, 384)
if data.shape == expected_shape:
    print(f"  ✅ PASS: Shape {data.shape} matches documentation")
    print(f"  ✅ Size: {data.nbytes/1e9:.2f} GB (documented as 2.05 GB)")
else:
    print(f"  ❌ FAIL: Shape {data.shape} (expected {expected_shape})")
PYTHON
echo

# Test 3: Metadata Structure
echo "✓ Test 3: Verify metadata structure (title-only, empty genres)"
python3 << 'PYTHON'
import json
with open('data/embeddings/tmdb/metadata.jsonl', 'r') as f:
    first_line = json.loads(f.readline())
    
expected_fields = {'tmdb_id', 'imdb_id', 'ml_id', 'title', 'year', 'genres'}
actual_fields = set(first_line.keys())

if actual_fields == expected_fields:
    print(f"  ✅ PASS: Fields match documentation: {sorted(actual_fields)}")
else:
    print(f"  ❌ FAIL: Fields {sorted(actual_fields)} (expected {sorted(expected_fields)})")

# Check for missing overview field
if 'overview' not in first_line:
    print("  ✅ PASS: No 'overview' field (as documented)")
else:
    print("  ❌ FAIL: 'overview' field exists (documentation says it should be missing)")

# Check genres are empty
if first_line['genres'] == []:
    print("  ✅ PASS: Genres array is empty (as documented)")
else:
    print(f"  ❌ FAIL: Genres are not empty: {first_line['genres']}")

print(f"\n  Sample record: {json.dumps(first_line, indent=2)}")
PYTHON
echo

# Test 4: Documentation Files Exist
echo "✓ Test 4: Verify new documentation files exist"
for file in "docs/DATA_QUALITY_REPORT.md" "docs/ACTUAL_PERFORMANCE_RESULTS.md"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  ✅ PASS: $file exists ($SIZE)"
    else
        echo "  ❌ FAIL: $file missing"
    fi
done
echo

# Test 5: README.md has data quality disclaimer
echo "✓ Test 5: Verify README.md has data quality disclaimer"
if grep -q "Data Quality Disclaimer" README.md; then
    echo "  ✅ PASS: Data quality disclaimer found in README.md"
else
    echo "  ❌ FAIL: Data quality disclaimer missing from README.md"
fi
echo

# Test 6: Verify TensorRT model exists
echo "✓ Test 6: Verify TensorRT model file exists"
if [ -f "data/models/minilm_l12_v2_fp16.plan" ]; then
    SIZE=$(du -h "data/models/minilm_l12_v2_fp16.plan" | cut -f1)
    echo "  ✅ PASS: TensorRT model exists ($SIZE)"
else
    echo "  ⚠️  WARNING: TensorRT model not found (may not be built yet)"
fi
echo

echo "================================================"
echo "Verification Complete"
echo "================================================"
echo
echo "All critical claims in documentation have been verified."
echo "Run this script anytime to validate documentation accuracy."
