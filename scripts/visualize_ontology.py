#!/usr/bin/env python3
"""
GMC-O Ontology Visualization Pipeline
Generates multiple visual representations of the Global Media & Context Ontology
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class OntologyClass:
    """Represents an OWL class with its properties"""
    uri: str
    label: str
    comment: str = ""
    parent: str = ""
    namespace: str = ""
    disjoint_with: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)

@dataclass
class OntologyProperty:
    """Represents an OWL property"""
    uri: str
    label: str
    domain: str = ""
    range: str = ""
    property_type: str = ""  # ObjectProperty, DatatypeProperty, etc.
    comment: str = ""

class OntologyParser:
    """Parse Turtle ontology files"""

    def __init__(self, ttl_file: Path):
        self.ttl_file = ttl_file
        self.prefixes: Dict[str, str] = {}
        self.classes: Dict[str, OntologyClass] = {}
        self.properties: Dict[str, OntologyProperty] = {}
        self.parse()

    def parse(self):
        """Parse the Turtle file"""
        content = self.ttl_file.read_text()

        # Extract prefixes
        prefix_pattern = r'@prefix\s+(\w+):\s+<([^>]+)>'
        for match in re.finditer(prefix_pattern, content):
            prefix, uri = match.groups()
            self.prefixes[prefix] = uri

        # Parse classes
        self._parse_classes(content)

        # Parse properties
        self._parse_properties(content)

    def _parse_classes(self, content: str):
        """Extract OWL classes"""
        # Pattern to match class definitions
        class_pattern = r'(\w+:\w+)\s+a\s+owl:Class\s*;(.*?)(?=\n\n|\n\w+:\w+\s+a\s+owl:|\Z)'

        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_uri, body = match.groups()

            # Parse class attributes
            label = self._extract_value(body, r'rdfs:label\s+"([^"]+)"')
            comment = self._extract_value(body, r'rdfs:comment\s+"([^"]+)"')
            parent = self._extract_value(body, r'rdfs:subClassOf\s+(\w+:\w+)')
            disjoint = re.findall(r'owl:disjointWith\s+(\w+:\w+)', body)

            namespace = class_uri.split(':')[0] if ':' in class_uri else ''

            self.classes[class_uri] = OntologyClass(
                uri=class_uri,
                label=label or class_uri,
                comment=comment,
                parent=parent,
                namespace=namespace,
                disjoint_with=disjoint
            )

    def _parse_properties(self, content: str):
        """Extract OWL properties"""
        # Object and Datatype properties
        prop_types = ['ObjectProperty', 'DatatypeProperty', 'FunctionalProperty',
                      'TransitiveProperty', 'AnnotationProperty']

        for prop_type in prop_types:
            pattern = rf'(\w+:\w+)\s+a\s+owl:{prop_type}\s*;(.*?)(?=\n\n|\n\w+:\w+\s+a\s+owl:|\Z)'

            for match in re.finditer(pattern, content, re.DOTALL):
                prop_uri, body = match.groups()

                label = self._extract_value(body, r'rdfs:label\s+"([^"]+)"')
                domain = self._extract_value(body, r'rdfs:domain\s+(\w+:\w+)')
                range_val = self._extract_value(body, r'rdfs:range\s+(\w+:\w+)')
                comment = self._extract_value(body, r'rdfs:comment\s+"([^"]+)"')

                self.properties[prop_uri] = OntologyProperty(
                    uri=prop_uri,
                    label=label or prop_uri,
                    domain=domain,
                    range=range_val,
                    property_type=prop_type,
                    comment=comment
                )

    @staticmethod
    def _extract_value(text: str, pattern: str) -> str:
        """Extract single value using regex"""
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def get_namespace_color(self, namespace: str) -> str:
        """Get color for namespace"""
        colors = {
            'media': '#FF6B6B',    # Red
            'user': '#4ECDC4',     # Teal
            'ctx': '#45B7D1',      # Blue
            'gpu': '#FFA07A',      # Light Salmon
            'tech': '#98D8C8',     # Mint
            'sem': '#FFE66D',      # Yellow
            'schema': '#D4A5A5',   # Mauve
            'owl': '#95E1D3',      # Light Green
        }
        return colors.get(namespace, '#CCCCCC')

class GraphVizGenerator:
    """Generate GraphViz DOT files"""

    def __init__(self, parser: OntologyParser):
        self.parser = parser

    def generate_full_hierarchy(self, output_file: Path):
        """Generate complete class hierarchy"""
        dot_lines = [
            'digraph GMC_O_Ontology {',
            '    rankdir=TB;',
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial", fontsize=10];',
            '    ',
            '    // Legend',
            '    subgraph cluster_legend {',
            '        label="Namespaces";',
            '        style=filled;',
            '        color=lightgrey;',
        ]

        # Add legend nodes
        for ns, color in [('media', '#FF6B6B'), ('user', '#4ECDC4'),
                          ('ctx', '#45B7D1'), ('gpu', '#FFA07A')]:
            dot_lines.append(f'        legend_{ns} [label="{ns}:", fillcolor="{color}"];')

        dot_lines.append('    }')
        dot_lines.append('    ')

        # Add class nodes grouped by namespace
        for namespace in ['media', 'user', 'ctx', 'gpu', 'tech', 'sem']:
            dot_lines.append(f'    // {namespace.upper()} namespace')

            for class_uri, cls in self.parser.classes.items():
                if cls.namespace == namespace:
                    color = self.parser.get_namespace_color(namespace)
                    tooltip = cls.comment.replace('"', '\\"') if cls.comment else ''
                    label = cls.label or cls.uri.split(':')[1]

                    dot_lines.append(
                        f'    "{class_uri}" [label="{label}", '
                        f'fillcolor="{color}", tooltip="{tooltip}"];'
                    )

            dot_lines.append('    ')

        # Add subclass relationships
        dot_lines.append('    // Subclass relationships')
        for class_uri, cls in self.parser.classes.items():
            if cls.parent:
                dot_lines.append(f'    "{cls.parent}" -> "{class_uri}" [label="subClassOf"];')

        # Add disjoint relationships
        dot_lines.append('    ')
        dot_lines.append('    // Disjoint classes')
        for class_uri, cls in self.parser.classes.items():
            for disjoint in cls.disjoint_with:
                dot_lines.append(
                    f'    "{class_uri}" -> "{disjoint}" '
                    f'[style=dashed, color=red, label="disjoint"];'
                )

        dot_lines.append('}')

        output_file.write_text('\n'.join(dot_lines))
        print(f"✓ Generated full hierarchy: {output_file}")

    def generate_namespace_view(self, namespace: str, output_file: Path):
        """Generate focused view of single namespace"""
        dot_lines = [
            f'digraph {namespace.upper()}_Namespace {{',
            '    rankdir=TB;',
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial"];',
            '    ',
        ]

        color = self.parser.get_namespace_color(namespace)

        # Add classes from this namespace
        for class_uri, cls in self.parser.classes.items():
            if cls.namespace == namespace:
                label = cls.label or cls.uri.split(':')[1]
                tooltip = cls.comment.replace('"', '\\"') if cls.comment else ''

                dot_lines.append(
                    f'    "{class_uri}" [label="{label}", '
                    f'fillcolor="{color}", tooltip="{tooltip}"];'
                )

        # Add relationships within namespace
        dot_lines.append('    ')
        for class_uri, cls in self.parser.classes.items():
            if cls.namespace == namespace and cls.parent:
                if cls.parent.startswith(f'{namespace}:'):
                    dot_lines.append(f'    "{cls.parent}" -> "{class_uri}";')

        dot_lines.append('}')

        output_file.write_text('\n'.join(dot_lines))
        print(f"✓ Generated {namespace} namespace view: {output_file}")

class MermaidGenerator:
    """Generate Mermaid diagrams for documentation"""

    def __init__(self, parser: OntologyParser):
        self.parser = parser

    def generate_class_hierarchy(self, namespace: str, output_file: Path):
        """Generate Mermaid class diagram for namespace"""
        lines = [
            '```mermaid',
            'classDiagram',
            '    %% GMC-O ' + namespace.upper() + ' Namespace',
            '    ',
        ]

        # Get all classes in namespace
        ns_classes = {uri: cls for uri, cls in self.parser.classes.items()
                      if cls.namespace == namespace}

        # Define classes
        for class_uri, cls in ns_classes.items():
            class_name = cls.label.replace(' ', '')
            lines.append(f'    class {class_name} {{')

            # Add properties if this class is a domain
            props = [p for p in self.parser.properties.values() if p.domain == class_uri]
            for prop in props[:3]:  # Limit to 3 for readability
                prop_name = prop.label.replace(' ', '_')
                lines.append(f'        +{prop_name}')

            lines.append('    }')

        lines.append('    ')

        # Add inheritance relationships
        for class_uri, cls in ns_classes.items():
            if cls.parent and cls.parent in ns_classes:
                child_name = cls.label.replace(' ', '')
                parent_name = self.parser.classes[cls.parent].label.replace(' ', '')
                lines.append(f'    {parent_name} <|-- {child_name}')

        lines.append('```')

        output_file.write_text('\n'.join(lines))
        print(f"✓ Generated Mermaid diagram for {namespace}: {output_file}")

    def generate_genre_tree(self, output_file: Path):
        """Generate focused Genre hierarchy"""
        lines = [
            '```mermaid',
            'graph TD',
            '    Genre[Genre]',
            '    ',
        ]

        # Find Genre class and its subclasses
        genre_classes = [
            (uri, cls) for uri, cls in self.parser.classes.items()
            if 'genre' in cls.label.lower() or cls.parent == 'media:Genre'
        ]

        for uri, cls in genre_classes:
            if cls.parent == 'media:Genre':
                label = cls.label.replace(' ', '<br/>')
                lines.append(f'    Genre --> {uri.split(":")[1]}["{label}"]')

        lines.append('```')

        output_file.write_text('\n'.join(lines))
        print(f"✓ Generated Genre tree: {output_file}")

    def generate_context_overview(self, output_file: Path):
        """Generate Context domain overview"""
        lines = [
            '```mermaid',
            'graph LR',
            '    subgraph Cultural',
            '        Holiday',
            '        Halloween["Halloween"]',
            '        Christmas["Christmas"]',
            '        Valentines["Valentine Day"]',
            '    end',
            '    ',
            '    subgraph Social',
            '        SocialSetting[Social Setting]',
            '        DateNight[Date Night]',
            '        Family[Family Gathering]',
            '        Solo[Solo Viewing]',
            '    end',
            '    ',
            '    subgraph Environmental',
            '        TimeOfDay[Time of Day]',
            '        Device[Device Type]',
            '    end',
            '    ',
            '    Holiday --> Halloween',
            '    Holiday --> Christmas',
            '    Holiday --> Valentines',
            '    SocialSetting --> DateNight',
            '    SocialSetting --> Family',
            '    SocialSetting --> Solo',
            '```',
        ]

        output_file.write_text('\n'.join(lines))
        print(f"✓ Generated Context overview: {output_file}")

class WebVOWLGenerator:
    """Generate WebVOWL-compatible JSON"""

    def __init__(self, parser: OntologyParser):
        self.parser = parser

    def generate(self, output_file: Path):
        """Generate WebVOWL JSON format"""
        data = {
            "header": {
                "title": "GMC-O: Global Media & Context Ontology",
                "description": "GPU-accelerated recommendation ontology for TV5 Monde",
                "version": "2.0",
                "author": ["TV5 Monde Team"],
                "iri": "http://recommendation.org/ontology/media"
            },
            "namespace": [],
            "class": [],
            "property": []
        }

        # Add namespaces
        for prefix, uri in self.parser.prefixes.items():
            data["namespace"].append({"prefix": prefix, "iri": uri})

        # Add classes
        for idx, (class_uri, cls) in enumerate(self.parser.classes.items()):
            node = {
                "id": str(idx),
                "type": "owl:Class",
                "iri": class_uri,
                "label": cls.label,
                "comment": cls.comment,
                "attributes": ["external"] if cls.namespace in ['schema', 'owl'] else []
            }

            if cls.parent:
                node["subClassOf"] = cls.parent

            data["class"].append(node)

        # Add properties
        for idx, (prop_uri, prop) in enumerate(self.parser.properties.items()):
            node = {
                "id": f"p{idx}",
                "type": f"owl:{prop.property_type}",
                "iri": prop_uri,
                "label": prop.label,
                "domain": prop.domain,
                "range": prop.range,
                "comment": prop.comment
            }
            data["property"].append(node)

        output_file.write_text(json.dumps(data, indent=2))
        print(f"✓ Generated WebVOWL JSON: {output_file}")

class HTMLViewerGenerator:
    """Generate standalone HTML viewer"""

    def generate(self, output_file: Path, webvowl_json: Path):
        """Create HTML viewer with embedded WebVOWL data"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GMC-O Ontology Explorer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .nav {{
            background: #f8f9fa;
            padding: 15px 30px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .nav button {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .nav button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .nav button.active {{
            background: #667eea;
            color: white;
        }}
        .content {{
            padding: 30px;
            min-height: 600px;
        }}
        .namespace-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .namespace-card {{
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s;
            cursor: pointer;
        }}
        .namespace-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }}
        .namespace-card.media {{ border-color: #FF6B6B; }}
        .namespace-card.user {{ border-color: #4ECDC4; }}
        .namespace-card.ctx {{ border-color: #45B7D1; }}
        .namespace-card.gpu {{ border-color: #FFA07A; }}
        .namespace-card h3 {{
            font-size: 1.5em;
            margin-bottom: 10px;
        }}
        .class-list {{
            margin-top: 20px;
        }}
        .class-item {{
            padding: 15px;
            background: #f8f9fa;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .class-item h4 {{
            color: #2d3748;
            margin-bottom: 5px;
        }}
        .class-item p {{
            color: #718096;
            font-size: 0.9em;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
            padding: 20px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 8px;
            color: white;
        }}
        .stat {{
            text-align: center;
        }}
        .stat h3 {{
            font-size: 2.5em;
            margin-bottom: 5px;
        }}
        .stat p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 GMC-O Ontology Explorer</h1>
            <p>Global Media & Context Ontology for GPU-Accelerated Recommendations</p>
        </div>

        <div class="nav">
            <button class="active" onclick="showView('overview')">Overview</button>
            <button onclick="showView('media')">Media Domain</button>
            <button onclick="showView('user')">User Domain</button>
            <button onclick="showView('ctx')">Context Domain</button>
            <button onclick="showView('gpu')">GPU Processing</button>
            <button onclick="showView('properties')">Properties</button>
        </div>

        <div class="content" id="content">
            <!-- Content loaded by JavaScript -->
        </div>
    </div>

    <script>
        const ontologyData = {json.dumps(json.loads(webvowl_json.read_text()), indent=2)};

        function showView(view) {{
            const buttons = document.querySelectorAll('.nav button');
            buttons.forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            const content = document.getElementById('content');

            if (view === 'overview') {{
                content.innerHTML = generateOverview();
            }} else if (view === 'properties') {{
                content.innerHTML = generatePropertiesView();
            }} else {{
                content.innerHTML = generateNamespaceView(view);
            }}
        }}

        function generateOverview() {{
            const classes = ontologyData.class;
            const properties = ontologyData.property;

            const namespaces = {{}};
            classes.forEach(cls => {{
                const ns = cls.iri.split(':')[0];
                namespaces[ns] = (namespaces[ns] || 0) + 1;
            }});

            let html = '<h2>Ontology Statistics</h2>';
            html += '<div class="stats">';
            html += '<div class="stat"><h3>' + classes.length + '</h3><p>Classes</p></div>';
            html += '<div class="stat"><h3>' + properties.length + '</h3><p>Properties</p></div>';
            html += '<div class="stat"><h3>' + Object.keys(namespaces).length + '</h3><p>Namespaces</p></div>';
            html += '</div>';
            html += '<h2>Namespaces</h2>';
            html += '<div class="namespace-grid">' + generateNamespaceCards() + '</div>';
            return html;
        }}

        function generateNamespaceCards() {{
            const namespaces = {{
                'media': {{ name: 'Media & Content', desc: 'Films, series, genres, visual aesthetics', color: '#FF6B6B' }},
                'user': {{ name: 'User & Psychographics', desc: 'Profiles, preferences, tolerance levels', color: '#4ECDC4' }},
                'ctx': {{ name: 'Context', desc: 'Cultural, social, environmental factors', color: '#45B7D1' }},
                'gpu': {{ name: 'GPU Processing', desc: 'Semantic analysis, vector search, graph traversal', color: '#FFA07A' }}
            }};

            let html = '';
            for (const [key, val] of Object.entries(namespaces)) {{
                const count = ontologyData.class.filter(c => c.iri.startsWith(key)).length;
                html += '<div class="namespace-card ' + key + '" onclick="showView(\'' + key + '\')">';
                html += '<h3>' + val.name + '</h3>';
                html += '<p>' + val.desc + '</p>';
                html += '<p style="margin-top:10px; font-weight:600; color:' + val.color + ';">';
                html += count + ' classes</p></div>';
            }}
            return html;
        }}

        function generateNamespaceView(namespace) {{
            const classes = ontologyData.class.filter(c => c.iri.startsWith(namespace));

            let html = '<h2>' + namespace.toUpperCase() + ' Namespace</h2>';
            html += '<p style="margin-bottom:20px; color:#718096;">';
            html += classes.length + ' classes in this namespace</p>';
            html += '<div class="class-list">';

            classes.forEach(cls => {{
                html += '<div class="class-item">';
                html += '<h4>' + cls.label + '</h4>';
                html += '<p>' + (cls.comment || 'No description') + '</p>';
                if (cls.subClassOf) {{
                    html += '<p style="margin-top:5px;"><strong>Parent:</strong> ' + cls.subClassOf + '</p>';
                }}
                html += '</div>';
            }});

            html += '</div>';
            return html;
        }}

        function generatePropertiesView() {{
            const properties = ontologyData.property;

            let html = '<h2>Properties</h2>';
            html += '<p style="margin-bottom:20px; color:#718096;">';
            html += properties.length + ' properties defining relationships and attributes</p>';
            html += '<div class="class-list">';

            properties.forEach(prop => {{
                html += '<div class="class-item">';
                html += '<h4>' + prop.label + '</h4>';
                html += '<p>' + (prop.comment || 'No description') + '</p>';
                html += '<p style="margin-top:5px;">';
                html += '<strong>Type:</strong> ' + prop.type + '<br>';
                if (prop.domain) {{
                    html += '<strong>Domain:</strong> ' + prop.domain + '<br>';
                }}
                if (prop.range) {{
                    html += '<strong>Range:</strong> ' + prop.range;
                }}
                html += '</p></div>';
            }});

            html += '</div>';
            return html;
        }}

        // Initialize with overview
        showView('overview');
    </script>
</body>
</html>"""

        output_file.write_text(html)
        print(f"✓ Generated HTML viewer: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate GMC-O ontology visualizations')
    parser.add_argument('--ttl', type=Path,
                       default=Path('/home/devuser/workspace/hackathon-tv5/design/ontology/expanded-media-ontology.ttl'),
                       help='Path to Turtle ontology file')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('/home/devuser/workspace/hackathon-tv5/design/ontology/visualizations'),
                       help='Output directory for visualizations')
    parser.add_argument('--format', choices=['all', 'graphviz', 'mermaid', 'webvowl', 'html'],
                       default='all', help='Output format')

    args = parser.parse_args()

    print("🎨 GMC-O Ontology Visualization Pipeline")
    print("=" * 50)

    # Parse ontology
    print(f"\n📖 Parsing ontology: {args.ttl}")
    onto_parser = OntologyParser(args.ttl)
    print(f"   Found {len(onto_parser.classes)} classes")
    print(f"   Found {len(onto_parser.properties)} properties")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\n🎨 Generating visualizations...")

    if args.format in ['all', 'graphviz']:
        print("\n📊 GraphViz diagrams:")
        gv = GraphVizGenerator(onto_parser)

        # Full hierarchy
        gv.generate_full_hierarchy(args.output_dir / 'full-hierarchy.dot')

        # Namespace views
        for ns in ['media', 'user', 'ctx', 'gpu']:
            gv.generate_namespace_view(ns, args.output_dir / f'{ns}-namespace.dot')

    if args.format in ['all', 'mermaid']:
        print("\n📐 Mermaid diagrams:")
        mm = MermaidGenerator(onto_parser)

        # Namespace diagrams
        for ns in ['media', 'user', 'ctx', 'gpu']:
            mm.generate_class_hierarchy(ns, args.output_dir / f'{ns}-classes.mmd')

        # Specialized views
        mm.generate_genre_tree(args.output_dir / 'genre-tree.mmd')
        mm.generate_context_overview(args.output_dir / 'context-overview.mmd')

    if args.format in ['all', 'webvowl', 'html']:
        print("\n🌐 WebVOWL JSON:")
        wv = WebVOWLGenerator(onto_parser)
        webvowl_json = args.output_dir / 'ontology.json'
        wv.generate(webvowl_json)

    if args.format in ['all', 'html']:
        print("\n🖥️  Interactive HTML viewer:")
        hv = HTMLViewerGenerator()
        hv.generate(args.output_dir / 'index.html', webvowl_json)

    print("\n✅ Visualization pipeline complete!")
    print(f"\n📁 Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Generate PNG/SVG from DOT files:")
    print("     dot -Tpng full-hierarchy.dot -o full-hierarchy.png")
    print("     dot -Tsvg full-hierarchy.dot -o full-hierarchy.svg")
    print("  2. Open HTML viewer:")
    print(f"     open {args.output_dir / 'index.html'}")
    print("  3. Embed Mermaid diagrams in documentation")

if __name__ == '__main__':
    main()
