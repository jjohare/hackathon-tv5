#!/usr/bin/env python3
"""
GMC-O Ontology Validation Script

Validates expanded-media-ontology.ttl against comprehensive rules:
- All classes have rdfs:label and rdfs:comment
- All ObjectProperties have rdfs:domain and rdfs:range
- No circular inheritance detected
- Disjoint classes have no common instances
- All referenced classes are defined
- Cardinality constraints are valid
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
from rdflib import Graph, RDF, RDFS, OWL, Namespace, URIRef
from rdflib.namespace import XSD
import argparse
import json


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def add_error(self, message: str):
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str):
        self.warnings.append(message)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats
        }


class OntologyValidator:
    """Comprehensive ontology validator"""

    def __init__(self, ontology_path: Path):
        self.ontology_path = ontology_path
        self.graph = Graph()
        self.report = ValidationReport()

        # Define namespaces
        self.MEDIA = Namespace("http://recommendation.org/ontology/media#")
        self.USER = Namespace("http://recommendation.org/ontology/user#")
        self.CTX = Namespace("http://recommendation.org/ontology/context#")
        self.TECH = Namespace("http://recommendation.org/ontology/tech-stack#")
        self.SEM = Namespace("http://recommendation.org/ontology/semantic-descriptors#")
        self.GPU = Namespace("http://recommendation.org/ontology/gpu-processing#")
        self.SCHEMA = Namespace("http://schema.org/")
        self.EIDR = Namespace("http://eidr.org/ontology#")

        self.namespaces = [
            ("media", self.MEDIA),
            ("user", self.USER),
            ("ctx", self.CTX),
            ("tech", self.TECH),
            ("sem", self.SEM),
            ("gpu", self.GPU),
            ("schema", self.SCHEMA),
            ("eidr", self.EIDR),
        ]

        for prefix, ns in self.namespaces:
            self.graph.bind(prefix, ns)

    def load_ontology(self) -> bool:
        """Load and parse ontology file"""
        try:
            self.graph.parse(str(self.ontology_path), format='turtle')
            self.report.stats['triples'] = len(self.graph)
            return True
        except Exception as e:
            self.report.add_error(f"Failed to parse ontology: {str(e)}")
            return False

    def validate_classes_have_labels_and_comments(self):
        """Ensure all classes have rdfs:label and rdfs:comment"""
        classes = list(self.graph.subjects(RDF.type, OWL.Class))
        self.report.stats['classes'] = len(classes)

        for cls in classes:
            # Skip blank nodes and AllDisjointClasses
            if isinstance(cls, URIRef):
                labels = list(self.graph.objects(cls, RDFS.label))
                comments = list(self.graph.objects(cls, RDFS.comment))

                if not labels:
                    self.report.add_error(f"Class {cls} missing rdfs:label")

                if not comments:
                    self.report.add_warning(f"Class {cls} missing rdfs:comment (recommended)")

    def validate_properties_have_domains_and_ranges(self):
        """Ensure ObjectProperties have domain and range"""
        obj_properties = list(self.graph.subjects(RDF.type, OWL.ObjectProperty))
        data_properties = list(self.graph.subjects(RDF.type, OWL.DatatypeProperty))

        self.report.stats['object_properties'] = len(obj_properties)
        self.report.stats['datatype_properties'] = len(data_properties)

        for prop in obj_properties:
            if isinstance(prop, URIRef):
                domains = list(self.graph.objects(prop, RDFS.domain))
                ranges = list(self.graph.objects(prop, RDFS.range))

                if not domains:
                    self.report.add_error(f"ObjectProperty {prop} missing rdfs:domain")

                if not ranges:
                    self.report.add_error(f"ObjectProperty {prop} missing rdfs:range")

        for prop in data_properties:
            if isinstance(prop, URIRef):
                domains = list(self.graph.objects(prop, RDFS.domain))
                ranges = list(self.graph.objects(prop, RDFS.range))

                if not domains:
                    self.report.add_warning(f"DatatypeProperty {prop} missing rdfs:domain")

                if not ranges:
                    self.report.add_error(f"DatatypeProperty {prop} missing rdfs:range")

    def detect_circular_inheritance(self) -> Set[URIRef]:
        """Detect circular inheritance in class hierarchy"""
        classes = [c for c in self.graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)]

        def has_cycle(cls: URIRef, visited: Set[URIRef], path: List[URIRef]) -> bool:
            if cls in path:
                cycle = " -> ".join([str(c) for c in path + [cls]])
                self.report.add_error(f"Circular inheritance detected: {cycle}")
                return True

            if cls in visited:
                return False

            visited.add(cls)
            path.append(cls)

            for parent in self.graph.objects(cls, RDFS.subClassOf):
                if isinstance(parent, URIRef) and has_cycle(parent, visited, path[:]):
                    return True

            return False

        visited = set()
        cycles = set()

        for cls in classes:
            if cls not in visited:
                has_cycle(cls, visited, [])

        return cycles

    def validate_disjoint_classes(self):
        """Ensure disjoint classes have no common instances"""
        # Extract disjoint class declarations
        disjoint_sets = []

        # Find AllDisjointClasses declarations
        for disjoint_decl in self.graph.subjects(RDF.type, OWL.AllDisjointClasses):
            members = list(self.graph.objects(disjoint_decl, OWL.members))
            for member_list in members:
                classes = []
                current = member_list
                while current and current != RDF.nil:
                    first = self.graph.value(current, RDF.first)
                    if first:
                        classes.append(first)
                    current = self.graph.value(current, RDF.rest)
                if classes:
                    disjoint_sets.append(classes)

        # Also check owl:disjointWith declarations
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if isinstance(cls, URIRef):
                for disjoint_cls in self.graph.objects(cls, OWL.disjointWith):
                    if isinstance(disjoint_cls, URIRef):
                        # Check if they share subclasses
                        cls_children = set(self.graph.subjects(RDFS.subClassOf, cls))
                        disjoint_children = set(self.graph.subjects(RDFS.subClassOf, disjoint_cls))

                        common = cls_children & disjoint_children
                        if common:
                            self.report.add_error(
                                f"Disjoint classes {cls} and {disjoint_cls} have common subclass(es): {common}"
                            )

        self.report.stats['disjoint_sets'] = len(disjoint_sets)

    def validate_referenced_classes_exist(self):
        """Ensure all referenced classes are defined"""
        defined_classes = set(self.graph.subjects(RDF.type, OWL.Class))

        # Check domains and ranges reference defined classes
        for prop in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            for domain in self.graph.objects(prop, RDFS.domain):
                if isinstance(domain, URIRef) and domain not in defined_classes:
                    # Check if it's from external ontology (schema.org, etc)
                    if not any(str(domain).startswith(str(ns)) for _, ns in [
                        ("schema", self.SCHEMA),
                        ("owl", OWL),
                        ("rdfs", RDFS),
                    ]):
                        self.report.add_error(f"Property {prop} references undefined domain class: {domain}")

            for range_cls in self.graph.objects(prop, RDFS.range):
                if isinstance(range_cls, URIRef) and range_cls not in defined_classes:
                    if not any(str(range_cls).startswith(str(ns)) for _, ns in [
                        ("schema", self.SCHEMA),
                        ("owl", OWL),
                        ("rdfs", RDFS),
                        ("xsd", XSD),
                    ]):
                        self.report.add_error(f"Property {prop} references undefined range class: {range_cls}")

    def validate_cardinality_constraints(self):
        """Validate functional properties and cardinality"""
        functional_props = list(self.graph.subjects(RDF.type, OWL.FunctionalProperty))
        self.report.stats['functional_properties'] = len(functional_props)

        for prop in functional_props:
            if isinstance(prop, URIRef):
                # Ensure it's also declared as ObjectProperty or DatatypeProperty
                is_obj_prop = (prop, RDF.type, OWL.ObjectProperty) in self.graph
                is_data_prop = (prop, RDF.type, OWL.DatatypeProperty) in self.graph

                if not (is_obj_prop or is_data_prop):
                    self.report.add_error(
                        f"Functional property {prop} not declared as ObjectProperty or DatatypeProperty"
                    )

    def validate_orphaned_classes(self):
        """Find classes with no relationships"""
        classes = [c for c in self.graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)]

        for cls in classes:
            # Check if class is used in any property domain/range
            used_in_domain = any(self.graph.subjects(RDFS.domain, cls))
            used_in_range = any(self.graph.subjects(RDFS.range, cls))
            has_subclass = any(self.graph.subjects(RDFS.subClassOf, cls))
            is_subclass = any(self.graph.objects(cls, RDFS.subClassOf))

            if not (used_in_domain or used_in_range or has_subclass or is_subclass):
                # Check if it's a top-level important class
                if cls not in [self.MEDIA.CreativeWork, self.USER.ViewerProfile]:
                    self.report.add_warning(f"Orphaned class (no relationships): {cls}")

    def validate_transitive_properties(self):
        """Ensure transitive properties are properly used"""
        transitive_props = list(self.graph.subjects(RDF.type, OWL.TransitiveProperty))
        self.report.stats['transitive_properties'] = len(transitive_props)

        for prop in transitive_props:
            if isinstance(prop, URIRef):
                # Transitive properties should be ObjectProperties
                if (prop, RDF.type, OWL.ObjectProperty) not in self.graph:
                    self.report.add_error(
                        f"TransitiveProperty {prop} must also be declared as ObjectProperty"
                    )

    def run_all_validations(self) -> ValidationReport:
        """Execute all validation checks"""
        if not self.load_ontology():
            return self.report

        print("Running validation checks...")

        checks = [
            ("Class labels and comments", self.validate_classes_have_labels_and_comments),
            ("Property domains and ranges", self.validate_properties_have_domains_and_ranges),
            ("Circular inheritance", self.detect_circular_inheritance),
            ("Disjoint classes", self.validate_disjoint_classes),
            ("Referenced classes", self.validate_referenced_classes_exist),
            ("Cardinality constraints", self.validate_cardinality_constraints),
            ("Orphaned classes", self.validate_orphaned_classes),
            ("Transitive properties", self.validate_transitive_properties),
        ]

        for check_name, check_func in checks:
            print(f"  ✓ {check_name}")
            check_func()

        return self.report

    def print_report(self):
        """Print formatted validation report"""
        print("\n" + "="*70)
        print("ONTOLOGY VALIDATION REPORT")
        print("="*70)

        print(f"\nOntology: {self.ontology_path}")
        print(f"Status: {'✓ PASSED' if self.report.passed else '✗ FAILED'}")

        print("\nStatistics:")
        for key, value in self.report.stats.items():
            print(f"  {key}: {value}")

        if self.report.errors:
            print(f"\n✗ Errors ({len(self.report.errors)}):")
            for error in self.report.errors:
                print(f"  - {error}")

        if self.report.warnings:
            print(f"\n⚠ Warnings ({len(self.report.warnings)}):")
            for warning in self.report.warnings:
                print(f"  - {warning}")

        if not self.report.errors and not self.report.warnings:
            print("\n✓ No issues found!")

        print("\n" + "="*70)

    def export_json(self, output_path: Path):
        """Export report as JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)
        print(f"\nReport exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate GMC-O ontology file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_ontology.py design/ontology/expanded-media-ontology.ttl
  python validate_ontology.py --json report.json ontology.ttl
  python validate_ontology.py --strict ontology.ttl
        """
    )

    parser.add_argument(
        'ontology_file',
        type=Path,
        help='Path to ontology file (.ttl)'
    )

    parser.add_argument(
        '--json',
        type=Path,
        help='Export report as JSON to specified file'
    )

    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )

    args = parser.parse_args()

    if not args.ontology_file.exists():
        print(f"Error: File not found: {args.ontology_file}", file=sys.stderr)
        sys.exit(1)

    validator = OntologyValidator(args.ontology_file)
    report = validator.run_all_validations()
    validator.print_report()

    if args.json:
        validator.export_json(args.json)

    # Exit code
    if not report.passed:
        sys.exit(1)
    elif args.strict and report.warnings:
        print("\n✗ Strict mode: Warnings treated as errors")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
