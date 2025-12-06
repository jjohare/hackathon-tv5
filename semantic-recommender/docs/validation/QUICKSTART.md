# Ontology Validation - Quick Start

## 5-Minute Setup

### 1. Install Python Dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Validate Main Ontology

```bash
python scripts/validate_ontology.py design/ontology/expanded-media-ontology.ttl
```

### 3. View Results

```bash
# Export JSON report
python scripts/validate_ontology.py \
  --json report.json \
  design/ontology/expanded-media-ontology.ttl

# Generate HTML report
python scripts/generate_validation_report.py report.json -o report.html

# Open in browser
xdg-open report.html  # Linux
open report.html      # macOS
```

## What Gets Validated

✅ All classes have labels
✅ All properties have domain/range
✅ No circular inheritance
✅ Disjoint classes respected
✅ All references valid
✅ Functional properties correct
✅ No orphaned classes

## Example Output

```
======================================================================
ONTOLOGY VALIDATION REPORT
======================================================================

Ontology: design/ontology/expanded-media-ontology.ttl
Status: ✓ PASSED

Statistics:
  triples: 536
  classes: 67
  object_properties: 12
  datatype_properties: 8
  functional_properties: 2

✓ No issues found!

======================================================================
```

## Testing Broken Ontologies

```bash
# These should fail validation
python scripts/validate_ontology.py tests/fixtures/broken_ontologies/circular_inheritance.ttl
python scripts/validate_ontology.py tests/fixtures/broken_ontologies/missing_labels.ttl
```

## CI/CD Integration

Push changes to trigger automatic validation:

```bash
git add design/ontology/expanded-media-ontology.ttl
git commit -m "Update ontology"
git push
# GitHub Actions runs validation automatically
```

## Pre-commit Hooks

Prevent invalid ontologies from being committed:

```bash
pip install pre-commit
pre-commit install

# Now validation runs automatically on commit
git commit -m "Update ontology"  # Validation runs here
```

## Rust Validator (Optional)

For faster validation:

```bash
cd src/rust/ontology
cargo build --release
cargo run --release --example validate_ontology -- \
  ../../../design/ontology/expanded-media-ontology.ttl
```

## Next Steps

- Read full guide: `/docs/validation/VALIDATION_GUIDE.md`
- View test fixtures: `/tests/fixtures/broken_ontologies/`
- Check CI workflow: `/.github/workflows/ontology-validation.yml`

## Troubleshooting

**Import errors?**
```bash
pip install -r scripts/requirements.txt
```

**Validation fails?**
Check errors in output. Most common issues:
- Missing rdfs:label on classes
- ObjectProperty without domain/range
- Circular rdfs:subClassOf chains

**Need help?**
Open an issue or see `/docs/validation/VALIDATION_GUIDE.md`
