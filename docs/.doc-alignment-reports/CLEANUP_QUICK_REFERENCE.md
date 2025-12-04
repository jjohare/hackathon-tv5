# Document Cleanup Quick Reference

**Generated**: 2025-12-04
**Total Files**: 935 markdown files analyzed

---

## Summary Statistics

| Category | Files | % | Size (KB) | Action |
|----------|-------|---|-----------|--------|
| **KEEP** | 120 | 12.8% | 2,224 | Keep in active docs |
| **EXTRACT** | 45 | 4.8% | 343 | Extract content, then archive |
| **ARCHIVE** | 501 | 53.6% | 7,935 | Move to archive/ |
| **DELETE** | 269 | 28.8% | 2,341 | Permanently remove |
| **TOTAL** | 935 | 100% | 12,844 | - |

**Impact**: 87.2% reduction (935 → 120 active files)

---

## Quick Actions

### 1. Extract Novel Content (Priority: HIGH, Time: 1-2 hrs)

Review 45 EXTRACT files, copy unique sections to main docs, then archive originals.

**Top 10 Extractions**:
1. `design/research/vector-database-architecture.md` → `ARCHITECTURE.md`
2. `design/integration/ADAPTIVE_SSSP_API_REFERENCE.md` → `docs/API_GUIDE.md`
3. `CUDA_BUILD_QUICKREF.md` → `docs/DEVELOPMENT.md`
4. `docs/adaptive_sssp_quick_reference.md` → `design/ADAPTIVE_SSSP_ARCHITECTURE.md`
5. `docs/npm-readme.md` → `docs/API_GUIDE.md`

**Full list**: See `docs/DOCUMENT_CLEANUP_ANALYSIS.md` Appendix A

### 2. Archive Historical Docs (Priority: MEDIUM, Time: 30 min)

```bash
# Automated archival
bash design/scripts/archive_documents.sh

# Result: 501 files → design/archive/2025-12-04-final/
```

### 3. Delete Chaff (Priority: LOW, Time: 15 min)

```bash
# Dry run first (safe)
DRY_RUN=1 bash design/scripts/delete_chaff.sh

# Review output, then actually delete
DRY_RUN=0 bash design/scripts/delete_chaff.sh
```

---

## Top 20 KEEP Files (Active Documentation)

| Score | File |
|-------|------|
| 9 | `README.md` |
| 9 | `ARCHITECTURE.md` |
| 9 | `design/ADAPTIVE_SSSP_ARCHITECTURE.md` |
| 8 | `PERFORMANCE.md` |
| 8 | `CONTRIBUTING.md` |
| 8 | `README_PHASE1.md` |
| 8 | `T4_OPTIMIZATION_COMPLETE.md` |
| 8 | `docs/DEPLOYMENT.md` |
| 8 | `docs/hybrid_sssp_ffi_implementation.md` |
| 8 | `docs/INTEGRATION_GUIDE.md` |
| 7 | `docs/QUICK_START.md` |
| 7 | `docs/API_GUIDE.md` |
| 7 | `docs/TROUBLESHOOTING.md` |
| 7 | `docs/DEVELOPMENT.md` |
| 7 | `design/architecture/system-architecture.md` |
| 7 | `design/guides/deployment-guide.md` |
| 7 | `src/cuda/README.md` |

---

## Archival Breakdown

**501 files to archive**:
- **Implementation summaries**: 127 files (`*_IMPLEMENTATION_SUMMARY.md`)
- **Phase reports**: 89 files (`PHASE*_COMPLETE.md`, `PHASE*_SUMMARY.md`)
- **Status reports**: 76 files (`*_STATUS.md`)
- **Working documents**: 143 files (DRAFT/WIP/TEMP markers)
- **temp-ruvector**: 66 files (temporary project copy)

**Archive location**: `design/archive/2025-12-04-final/`

---

## Deletion Breakdown

**269 files to delete**:
- **Empty files**: 43 files (<100 bytes)
- **Generic templates**: 87 files (copy-paste from crate templates)
- **temp-datadesigner**: 106 files (unrelated project)
- **Duplicate reports**: 12 files (LINK_FIX_SUMMARY.md copies)
- **Empty drafts**: 21 files (marked DRAFT with no content)

**Safety**: All run in dry-run mode first for review

---

## Novel Content to Extract

**Top 15 high-value extractions**:

| Source | Target | Unique Content |
|--------|--------|----------------|
| `CUDA_BUILD_QUICKREF.md` | `docs/DEVELOPMENT.md` | CUDA compilation commands |
| `docs/adaptive_sssp_quick_reference.md` | `design/ADAPTIVE_SSSP_ARCHITECTURE.md` | Algorithm selection logic |
| `docs/npm-readme.md` | `docs/API_GUIDE.md` | Node.js FFI examples |
| `docs/QUICK_START_HNSW_LSH.md` | `docs/QUICK_START.md` | Vector index config |
| `docs/validation/ffi_audit_checklist.md` | `docs/DEVELOPMENT.md` | FFI safety checklist |
| `design/research/vector-database-architecture.md` | `ARCHITECTURE.md` | DB architecture analysis |
| `design/integration/ADAPTIVE_SSSP_API_REFERENCE.md` | `docs/API_GUIDE.md` | API signatures |
| `design/archive/.../PHASE2_SUMMARY.md` | `PERFORMANCE.md` | Phase 2 insights |
| `design/archive/.../VISIONFLOW_PORT_SUMMARY.md` | `docs/MIGRATION_GUIDE.md` | Port learnings |

**Extraction method**:
1. Read source file
2. Identify unique paragraphs/code blocks
3. Copy to target with attribution: `_(Source: original-file.md)_`
4. Move original to archive

---

## Scoring Explanation

Each document scored 0-10 on four dimensions:

- **Novelty (30%)**: Unique content not in main docs
- **Relevance (30%)**: Applicable to current project
- **Quality (25%)**: Professional vs rough notes
- **Integration (15%)**: Standalone vs already integrated

**Total score** → Category:
- **7-10**: KEEP (core documentation)
- **4-6**: EXTRACT (novel content)
- **2-4**: ARCHIVE (historical value)
- **0-2**: DELETE (no value)

---

## Implementation Checklist

- [ ] **Phase 1: Extract** (1-2 hours)
  - [ ] Review 45 EXTRACT files
  - [ ] Copy novel sections to targets
  - [ ] Add source attribution
  - [ ] Move originals to archive

- [ ] **Phase 2: Archive** (30 minutes)
  - [ ] Run `design/scripts/archive_documents.sh`
  - [ ] Verify 501 files moved
  - [ ] Update links in KEEP files

- [ ] **Phase 3: Delete** (15 minutes)
  - [ ] Dry run: `DRY_RUN=1 bash design/scripts/delete_chaff.sh`
  - [ ] Review output
  - [ ] Delete: `DRY_RUN=0 bash design/scripts/delete_chaff.sh`

- [ ] **Phase 4: Validate** (30 minutes)
  - [ ] Verify all links work
  - [ ] Generate final doc index
  - [ ] Update README.md

---

## Files & Reports

- **Full Analysis**: `docs/DOCUMENT_CLEANUP_ANALYSIS.md`
- **Raw Scores**: `docs/.doc-alignment-reports/document-scores.json`
- **Archive Script**: `design/scripts/archive_documents.sh`
- **Delete Script**: `design/scripts/delete_chaff.sh`
- **Scorer**: `design/scripts/document_scorer.py`

---

## Success Criteria

- [ ] Active docs reduced to ~120 files (87.2% reduction)
- [ ] All novel content extracted and integrated
- [ ] 501 files archived to `design/archive/2025-12-04-final/`
- [ ] 269 chaff files deleted
- [ ] Zero broken links in active docs
- [ ] Archive index created

**Estimated Time**: 2-3 hours total
**Impact**: 935 → 120 active files (cleaner, more discoverable)

---

**Next Steps**: Review this summary, then execute Phase 1 (Extract)
