#!/bin/bash
# Delete Chaff Documents Script
# Generated: 2025-12-04
# Purpose: Remove 269 files with no archival value
# ⚠️ WARNING: Review carefully before running!

set -e  # Exit on error

# Safety: Run in dry-run mode first
DRY_RUN=${DRY_RUN:-1}  # Set to 0 to actually delete

if [ "$DRY_RUN" = "1" ]; then
    echo "🔍 DRY RUN MODE - No files will be deleted"
    echo "   Set DRY_RUN=0 to actually delete files"
    echo ""
    ACTION="echo [DRY-RUN] Would delete:"
else
    echo "⚠️  DELETION MODE - Files will be permanently removed!"
    echo "   Press Ctrl-C within 5 seconds to cancel..."
    sleep 5
    ACTION="rm -v"
fi

DELETE_COUNT=0

# 1. Delete empty markdown files (<100 bytes)
echo "🗑️  Deleting empty files (<100 bytes)..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        $ACTION "$file"
        ((DELETE_COUNT++))
    fi
done < <(find . -type f -name "*.md" -size -100c \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*" \
    ! -path "*/venv/*")

# 2. Delete temp-datadesigner (unrelated project)
echo "🗑️  Deleting temp-datadesigner project..."
if [ -d "archive/temp-directories/temp-datadesigner" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY-RUN] Would delete: archive/temp-directories/temp-datadesigner/"
        TEMP_FILES=$(find archive/temp-directories/temp-datadesigner -name "*.md" | wc -l)
        echo "   (Contains $TEMP_FILES markdown files)"
        ((DELETE_COUNT+=TEMP_FILES))
    else
        rm -rv archive/temp-directories/temp-datadesigner
        echo "   Deleted temp-datadesigner/"
    fi
fi

# 3. Delete duplicate LINK_FIX_SUMMARY.md files (keep only docs/LINK_FIX_SUMMARY.md)
echo "🗑️  Deleting duplicate link fix reports..."
while IFS= read -r file; do
    if [ -f "$file" ] && [ "$file" != "./docs/LINK_FIX_SUMMARY.md" ]; then
        $ACTION "$file"
        ((DELETE_COUNT++))
    fi
done < <(find . -type f -name "LINK_FIX_SUMMARY.md" \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*")

# 4. Delete empty draft files in archive
echo "🗑️  Deleting empty drafts..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        $ACTION "$file"
        ((DELETE_COUNT++))
    fi
done < <(find design/archive -type f \( -name "*DRAFT*.md" -o -name "*WIP*.md" \) -size -500c 2>/dev/null)

# 5. Delete generic README templates from temp-ruvector crates
echo "🗑️  Deleting generic crate READMEs..."
if [ -d "archive/temp-directories/temp-ruvector/crates" ]; then
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Check if it's a generic template (contains "# crate-name" and <100 lines)
            LINES=$(wc -l < "$file")
            if [ "$LINES" -lt 100 ] && grep -q "^# ruvector-" "$file" 2>/dev/null; then
                $ACTION "$file"
                ((DELETE_COUNT++))
            fi
        fi
    done < <(find archive/temp-directories/temp-ruvector/crates -type f -name "README.md" 2>/dev/null)
fi

# 6. Delete obsolete working documents
echo "🗑️  Deleting obsolete working docs..."
OBSOLETE_FILES=(
    "design/archive/2025-12-04/working/ontology.md"
    "design/archive/2025-12-04/working/ENHANCEMENTS.md"
)

for file in "${OBSOLETE_FILES[@]}"; do
    if [ -f "$file" ]; then
        $ACTION "$file"
        ((DELETE_COUNT++))
    fi
done

# 7. Clean up empty directories
echo "🗑️  Removing empty directories..."
if [ "$DRY_RUN" = "1" ]; then
    EMPTY_DIRS=$(find . -type d -empty ! -path "*/node_modules/*" ! -path "*/.git/*" | wc -l)
    echo "[DRY-RUN] Would remove $EMPTY_DIRS empty directories"
else
    find . -type d -empty ! -path "*/node_modules/*" ! -path "*/.git/*" -delete
    echo "   Removed empty directories"
fi

# Summary
echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "✅ Dry run complete!"
    echo "   Would delete: $DELETE_COUNT files"
    echo ""
    echo "To actually delete files, run:"
    echo "   DRY_RUN=0 bash design/scripts/delete_chaff.sh"
else
    echo "✅ Deletion complete!"
    echo "   Deleted: $DELETE_COUNT files"
    echo ""
    echo "⚠️  Remember to commit changes:"
    echo "   git add -A"
    echo "   git commit -m 'docs: remove chaff files after analysis'"
fi
