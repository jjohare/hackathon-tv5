# Archive Plan Quick Reference

**Date**: 2025-12-04
**Status**: Plan Created - Awaiting Execution

## At a Glance

- **Total Documents to Archive**: 77
- **Hackathon Working Docs**: 27
- **Temp-Ruvector Docs**: 50
- **Protected Critical Files**: 5
- **Archive Location**: `design/archive/2025-12-04/`

## Critical Files (DO NOT ARCHIVE)

✅ **PROTECTED** - These MUST remain in place:

1. `design/SSSP_BREAKTHROUGH_SUMMARY.md` - Primary hackathon deliverable
2. `design/DATABASE_UNIFICATION_ANALYSIS.md` - Critical architecture doc
3. `design/DATABASE_USAGE_ANALYSIS.md` - Critical architecture doc
4. `design/ADAPTIVE_SSSP_ARCHITECTURE.md` - Core system architecture
5. `README.md` - Project entry point

## What to Archive

### Phase Documents (7 files)
```
design/PHASE2_SUMMARY.md
design/PHASE2_COMPLETE.txt
design/phase2_implementation_docs.md
design/PHASE2_INDEX.md
design/PHASE2_README.md
design/phase2_memory_patterns.txt
PHASE1_COMPLETE.md
```

### Analysis Documents (6 files)
```
design/DUAN_SSSP_ANALYSIS.md
design/DUAN_SSSP_CORRECTED_ANALYSIS.md
design/SSSP_INVESTIGATION_REPORT.md
design/VISIONFLOW_PORT_ANALYSIS.md
design/VISIONFLOW_PORT_SUMMARY.md
design/HIVE_MIND_REPORT.md
```

### CUDA Working Docs (5 files)
```
design/cuda-kernel-analysis.md
design/cuda-optimization-plan.md
design/deepseek-cuda-analysis-results.md
design/deepseek-cuda-query.md
design/deepseek-cuda-reasoning.txt
```

### Implementation Summaries (6 files)
```
design/IMPLEMENTATION_SUMMARY.md
IMPLEMENTATION_SUMMARY.md
REASONER_SUMMARY.md
VALIDATION_SUMMARY.md
UNIFIED_PIPELINE_COMPLETE.md
MIGRATION-SUMMARY.md
```

### API Docs (2 files)
```
design/API_SUMMARY.md
design/API_DELIVERABLES.md
```

### Test Docs (2 files)
```
docs/TEST_IMPLEMENTATION_SUMMARY.md
tests/docs/TEST_SUMMARY.md
```

### Storage Docs (5 files)
```
README_HYBRID_STORAGE.md
README_MILVUS.md
README_AGENTDB.md
README-migration.md
DELIVERABLES.md
TESTING_CHECKLIST.md
```

### Entire Directory (50+ files)
```
temp-ruvector/  (entire directory)
```

## Quick Execute

```bash
# Create structure
mkdir -p design/archive/2025-12-04/{summaries,phases,analysis,cuda,tests,temp-ruvector}

# Archive everything (see ARCHIVE_JUSTIFICATION.md for full commands)
# ... 27 file moves ...

# Archive temp-ruvector
mv temp-ruvector design/archive/2025-12-04/temp-ruvector/

# Verify critical files still exist
ls design/SSSP_BREAKTHROUGH_SUMMARY.md
ls design/DATABASE_UNIFICATION_ANALYSIS.md
ls design/DATABASE_USAGE_ANALYSIS.md
ls design/ADAPTIVE_SSSP_ARCHITECTURE.md
ls README.md
```

## Reports Generated

1. **archive.json** - Script output with 63 detected files
2. **archive_plan_enhanced.json** - Detailed JSON plan with all 77 files
3. **ARCHIVE_JUSTIFICATION.md** - Complete justifications for each file (THIS IS THE PRIMARY DOCUMENT)
4. **ARCHIVE_QUICK_REFERENCE.md** - This summary

## Next Steps

1. Review ARCHIVE_JUSTIFICATION.md for detailed rationale
2. Verify protected files list is correct
3. Execute archive commands
4. Create INDEX.md in archive directory
5. Verify completion

---

**Location**: `/home/devuser/workspace/hackathon-tv5/docs/.doc-alignment-reports/`
