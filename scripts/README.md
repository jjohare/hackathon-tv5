# Ontology Validation Scripts

Comprehensive validation tooling for GMC-O ontology files.

## Scripts

### validate_ontology.py

Primary validation script using rdflib.

**Usage:**
```bash
# Basic validation
python validate_ontology.py design/ontology/expanded-media-ontology.ttl

# With JSON export
python validate_ontology.py --json report.json design/ontology/expanded-media-ontology.ttl

# Strict mode (warnings as errors)
python validate_ontology.py --strict design/ontology/expanded-media-ontology.ttl
```

**Validation Checks:**
- Class labels and comments
- Property domains and ranges
- Circular inheritance detection
- Disjoint class constraints
- Referenced classes existence
- Cardinality constraints
- Orphaned classes detection
- Transitive property validation

### generate_validation_report.py

Generate HTML reports from JSON validation results.

**Usage:**
```bash
python generate_validation_report.py report.json -o report.html
```

## Installation

```bash
pip install -r requirements.txt
```

## Integration

These scripts are integrated with:
- GitHub Actions (`.github/workflows/ontology-validation.yml`)
- Pre-commit hooks (`.pre-commit-config.yaml`)
- Rust validator (`src/rust/ontology/validator.rs`)

See `/home/devuser/workspace/hackathon-tv5/docs/validation/VALIDATION_GUIDE.md` for comprehensive documentation.
