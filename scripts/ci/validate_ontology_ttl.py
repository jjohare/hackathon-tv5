#!/usr/bin/env python3
"""Turtle/RDF validation for the GMC-O media ontology.

Parses each supplied Turtle file with rdflib and reports structural counts
(triples, owl:Class, owl:ObjectProperty, owl:DatatypeProperty). A file that
fails to parse, is empty, or is missing causes a non-zero exit so the CI lane
turns red only on a genuine ontology regression.

This is the hosted-runner blocking lane: it needs no GPU, no CUDA toolchain and
no service containers, unlike the integration suite in test-hybrid.yml.

Usage:
    python scripts/ci/validate_ontology_ttl.py FILE [FILE ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from rdflib import Graph
    from rdflib.namespace import OWL, RDF
except ImportError:  # pragma: no cover - surfaced clearly in CI
    print("ERROR: rdflib is not installed (pip install rdflib)", file=sys.stderr)
    sys.exit(2)


def validate(path: Path) -> bool:
    if not path.exists():
        print(f"FAIL {path}: file not found")
        return False

    graph = Graph()
    try:
        graph.parse(path.as_posix(), format="turtle")
    except Exception as exc:  # rdflib raises many parser-specific types
        print(f"FAIL {path}: {type(exc).__name__}: {exc}")
        return False

    triples = len(graph)
    if triples == 0:
        print(f"FAIL {path}: parsed but contains zero triples")
        return False

    classes = len(set(graph.subjects(RDF.type, OWL.Class)))
    obj_props = len(set(graph.subjects(RDF.type, OWL.ObjectProperty)))
    data_props = len(set(graph.subjects(RDF.type, OWL.DatatypeProperty)))
    print(
        f"OK   {path}: {triples} triples, "
        f"{classes} owl:Class, {obj_props} owl:ObjectProperty, "
        f"{data_props} owl:DatatypeProperty"
    )
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    results = [validate(Path(arg)) for arg in argv]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} ontology file(s) valid")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
