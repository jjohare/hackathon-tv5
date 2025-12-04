# Quick Fix Guide - Documentation Link Issues

## Priority 1: Critical Broken Links

### Fix Typos in File References

**Issue**: `temp-ruvector/npm/packages/agentic-synth/examples/docs/DSPY_INTEGRATION_SUMMARY.md:456`
```markdown
# Wrong
[Contributing](../../CONTRIBUTIN.md)

# Fix
[Contributing](../../CONTRIBUTING.md)
```

### Fix Absolute Paths in Skills

**Issue**: `.claude/skills/performance-analysis/SKILL.md:553-555`
```markdown
# Wrong - hardcoded workspace paths
[Guide](/workspaces/claude-code-flow/.claude/docs/guide.md)

# Fix - use relative paths
[Guide](../../docs/guide.md)
```

### Remove Placeholder Links

**Issue**: `.claude/skills/skill-builder/SKILL.md:338-339`
```markdown
# Wrong - placeholder links
[Related Skill 1](#)
[Related Skill 2](#)

# Fix - either remove or add proper targets
[Related Skill 1](../related-skill/SKILL.md)
# or remove if not applicable
```

## Priority 2: Orphan Document Integration

### Main Documentation Orphans

Add links to these important orphan documents:

1. **DELIVERABLES.md** - Link from main README
2. **IMPLEMENTATION_SUMMARY.md** - Link from docs/index.md
3. **MIGRATION-SUMMARY.md** - Link from migration guide
4. **README_AGENTDB.md** - Link from AgentDB documentation

### Example Integration

In main `README.md`:
```markdown
## Documentation

- [Project Deliverables](DELIVERABLES.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Migration Guide](MIGRATION-SUMMARY.md)
```

## Priority 3: Anchor Fixes

### Common Anchor Issues

```markdown
# Wrong - anchor doesn't exist
[See API Reference](#api-reference)

# Fix - check actual section header
[See API Reference](#api-documentation)

# Or add the missing section
## API Reference
```

## Batch Fix Commands

### Find and Replace Common Issues

```bash
# Fix CONTRIBUTIN typo across project
find . -name "*.md" -type f -exec sed -i 's/CONTRIBUTIN\.md/CONTRIBUTING.md/g' {} \;

# Fix hardcoded workspace paths
find . -name "*.md" -type f -exec sed -i 's|/workspaces/claude-code-flow/||g' {} \;

# List all placeholder links
grep -r "\](#)" --include="*.md" .
```

### Clean Up Orphan Documents

```bash
# List orphan documents by category
jq -r '.orphan_docs[]' docs/.doc-alignment-reports/links.json | grep "README" | sort

# Archive old orphans
mkdir -p archive/old-docs
mv MIGRATION-SUMMARY.md archive/old-docs/
```

## Validation After Fixes

```bash
# Re-run validation
python /home/devuser/workspace/project/multi-agent-docker/skills/docs-alignment/scripts/validate_links.py \
  --root /home/devuser/workspace/hackathon-tv5 \
  --docs-dir docs \
  --output /home/devuser/workspace/hackathon-tv5/docs/.doc-alignment-reports/links.json

# Check improvement
python3 << 'EOF'
import json
with open('docs/.doc-alignment-reports/links.json') as f:
    data = json.load(f)
    rate = (data['valid_links'] / data['total_links'] * 100)
    print(f"Link validation: {rate:.1f}%")
    print(f"Broken links: {len(data['broken_links'])}")
    print(f"Orphans: {len(data['orphan_docs'])}")
EOF
```

## Quick Wins

### Top 5 Quick Fixes (< 5 minutes each)

1. **Fix CONTRIBUTIN typo** - 1 file
2. **Remove placeholder # links** - ~15 files in skill-builder
3. **Fix absolute paths in performance-analysis skill** - 3 links
4. **Link DELIVERABLES.md from README** - 1 line
5. **Archive obviously outdated files** - ~10 files

### Expected Impact

After these quick fixes:
- Broken links: 331 → ~310 (6% reduction)
- Orphans: 444 → ~434 (2% reduction)
- Validation rate: 90.4% → ~91.0%

## Pre-Commit Hook

Prevent future issues:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Validate links before commit
python /home/devuser/workspace/project/multi-agent-docker/skills/docs-alignment/scripts/validate_links.py \
  --root . \
  --docs-dir docs \
  --output .link-check.json

# Check if validation rate dropped
RATE=$(jq -r '(.valid_links / .total_links * 100)' .link-check.json)
if (( $(echo "$RATE < 90" | bc -l) )); then
  echo "Link validation below 90%: $RATE%"
  exit 1
fi

rm .link-check.json
```

## Resources

- **Full Report**: `docs/.doc-alignment-reports/links.json`
- **Summary**: `docs/.doc-alignment-reports/LINK_VALIDATION_SUMMARY.md`
- **This Guide**: `docs/.doc-alignment-reports/QUICK_FIX_GUIDE.md`
