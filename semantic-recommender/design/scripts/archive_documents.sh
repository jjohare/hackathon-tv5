#!/bin/bash
# Archive Historical Documents Script
# Generated: 2025-12-04
# Purpose: Move 501 archival candidates to design/archive/2025-12-04-final/

set -e  # Exit on error

ARCHIVE_BASE="design/archive/2025-12-04-final"
ARCHIVE_DATE=$(date +%Y-%m-%d-%H%M%S)

# Create archive structure
echo "📁 Creating archive directories..."
mkdir -p "$ARCHIVE_BASE"/{summaries,phases,status,working,temp-projects,extracted,duplicates}

# Archive implementation summaries
echo "📦 Archiving implementation summaries..."
find . -type f -name "*_IMPLEMENTATION_SUMMARY.md" \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*" \
    ! -path "*/design/archive/*" \
    -exec mv -v {} "$ARCHIVE_BASE/summaries/" \;

# Archive phase reports
echo "📦 Archiving phase reports..."
find . -type f \( -name "PHASE*_COMPLETE.md" -o -name "PHASE*_SUMMARY.md" \) \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*" \
    ! -path "*/design/archive/*" \
    -exec mv -v {} "$ARCHIVE_BASE/phases/" \;

# Archive status reports
echo "📦 Archiving status reports..."
find . -type f -name "*_STATUS.md" \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*" \
    ! -path "*/design/archive/*" \
    -exec mv -v {} "$ARCHIVE_BASE/status/" \;

# Archive working documents (DRAFT, WIP, WORKING)
echo "📦 Archiving working documents..."
find . -type f \( -name "*DRAFT*.md" -o -name "*WIP*.md" -o -name "*WORKING*.md" \) \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*" \
    ! -path "*/design/archive/*" \
    -exec mv -v {} "$ARCHIVE_BASE/working/" \;

# Archive temp-ruvector (if not already archived)
echo "📦 Archiving temp-ruvector project..."
if [ -d "temp-ruvector" ]; then
    mv -v temp-ruvector "$ARCHIVE_BASE/temp-projects/" 2>/dev/null || true
fi

# Archive temp-datadesigner (unrelated project)
echo "📦 Archiving temp-datadesigner project..."
if [ -d "archive/temp-directories/temp-datadesigner" ]; then
    mv -v archive/temp-directories/temp-datadesigner "$ARCHIVE_BASE/temp-projects/" 2>/dev/null || true
fi

# Archive duplicate summaries
echo "📦 Archiving duplicate summaries..."
[ -f "MIGRATION-SUMMARY.md" ] && mv -v MIGRATION-SUMMARY.md "$ARCHIVE_BASE/duplicates/"
[ -f "docs/INTEGRATION_SUMMARY.md" ] && mv -v docs/INTEGRATION_SUMMARY.md "$ARCHIVE_BASE/duplicates/"
[ -f "docs/BUILD_SUMMARY.md" ] && mv -v docs/BUILD_SUMMARY.md "$ARCHIVE_BASE/duplicates/"
[ -f "T4_QUICK_REFERENCE.md" ] && mv -v T4_QUICK_REFERENCE.md "$ARCHIVE_BASE/duplicates/"
[ -f "README_AGENTDB.md" ] && mv -v README_AGENTDB.md "$ARCHIVE_BASE/duplicates/"

# Create archive index
echo "📝 Creating archive index..."
cat > "$ARCHIVE_BASE/ARCHIVE_INDEX.md" << 'EOF'
# Archive Index: 2025-12-04 Final Documentation Cleanup

**Archive Date**: $(date +%Y-%m-%d)
**Purpose**: Historical documentation from hackathon-tv5 development

## Archive Structure

- **summaries/** - Implementation summaries (127 files)
- **phases/** - Phase completion reports (89 files)
- **status/** - Status update documents (76 files)
- **working/** - Working documents, drafts (143 files)
- **temp-projects/** - Temporary project copies (66 files)
- **extracted/** - Documents with content extracted to main docs
- **duplicates/** - Duplicate documents archived

## Why Archived?

These documents were moved to archive because:
1. Content integrated into main documentation
2. Historical development records
3. Superseded by newer documentation
4. Working documents no longer active

## Accessing Archived Content

All archived files remain accessible via git history:
```bash
git log --all --full-history -- <filename>
```

## Related Reports

- Main analysis: `docs/DOCUMENT_CLEANUP_ANALYSIS.md`
- Scoring data: `docs/.doc-alignment-reports/document-scores.json`
EOF

# Count archived files
SUMMARIES_COUNT=$(find "$ARCHIVE_BASE/summaries" -name "*.md" 2>/dev/null | wc -l)
PHASES_COUNT=$(find "$ARCHIVE_BASE/phases" -name "*.md" 2>/dev/null | wc -l)
STATUS_COUNT=$(find "$ARCHIVE_BASE/status" -name "*.md" 2>/dev/null | wc -l)
WORKING_COUNT=$(find "$ARCHIVE_BASE/working" -name "*.md" 2>/dev/null | wc -l)
TEMP_COUNT=$(find "$ARCHIVE_BASE/temp-projects" -name "*.md" 2>/dev/null | wc -l)
TOTAL_COUNT=$((SUMMARIES_COUNT + PHASES_COUNT + STATUS_COUNT + WORKING_COUNT + TEMP_COUNT))

echo ""
echo "✅ Archive complete!"
echo "   Summaries: $SUMMARIES_COUNT files"
echo "   Phases: $PHASES_COUNT files"
echo "   Status: $STATUS_COUNT files"
echo "   Working: $WORKING_COUNT files"
echo "   Temp projects: $TEMP_COUNT files"
echo "   Total archived: $TOTAL_COUNT files"
echo ""
echo "📁 Archive location: $ARCHIVE_BASE"
echo "📝 Archive index: $ARCHIVE_BASE/ARCHIVE_INDEX.md"
