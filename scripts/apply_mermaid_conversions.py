#!/usr/bin/env python3
"""
Apply ASCII to Mermaid diagram conversions
Reads conversion templates from ascii.json and applies them to documentation files
"""

import json
import re
from pathlib import Path

def apply_conversions(base_dir: Path):
    """Apply all ASCII to Mermaid conversions"""

    # Read conversion templates
    ascii_json_path = base_dir / 'docs/.doc-alignment-reports/ascii.json'
    with open(ascii_json_path, 'r') as f:
        data = json.load(f)

    conversions_applied = 0
    files_modified = set()

    # Group diagrams by file
    by_file = {}
    for diagram in data['ascii_diagrams']:
        file_path = diagram['file']
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(diagram)

    # Process each file
    for rel_path, diagrams in by_file.items():
        file_path = base_dir / rel_path

        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Sort diagrams by line number (descending) to preserve line numbers
        diagrams.sort(key=lambda d: d['start_line'], reverse=True)

        # Apply conversions
        for diagram in diagrams:
            start_line = diagram['start_line'] - 1  # 0-indexed
            end_line = diagram['end_line'] - 1
            original_text = diagram['original']

            # Skip if this looks like metadata rather than actual diagram
            if ('→' in original_text or '✅' in original_text or '❌' in original_text) and len(original_text) < 200:
                continue

            # Create mermaid replacement
            mermaid_type = diagram['diagram_type']

            if mermaid_type == 'architecture':
                mermaid_code = f"```mermaid\ngraph TD\n    {original_text.strip()}\n```\n"
            elif mermaid_type == 'flowchart':
                mermaid_code = f"```mermaid\nflowchart LR\n    {original_text.strip()}\n```\n"
            elif mermaid_type == 'sequence':
                mermaid_code = f"```mermaid\nsequenceDiagram\n    {original_text.strip()}\n```\n"
            elif mermaid_type == 'system':
                mermaid_code = f"```mermaid\ngraph TD\n    {original_text.strip()}\n```\n"
            else:
                mermaid_code = f"```mermaid\nflowchart LR\n    {original_text.strip()}\n```\n"

            # Replace in lines array
            original_block = ''.join(lines[start_line:end_line+1])

            # Keep ASCII in HTML comment for reference
            comment = f"<!-- Original ASCII diagram:\n{original_text}\n-->\n\n"

            lines[start_line:end_line+1] = [comment + mermaid_code]
            conversions_applied += 1

        # Write modified content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        files_modified.add(rel_path)
        print(f"✅ {rel_path}: {len(diagrams)} diagrams converted")

    return conversions_applied, len(files_modified)

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent

    print("=" * 70)
    print("ASCII to Mermaid Conversion Script")
    print("=" * 70)

    converted, files = apply_conversions(base_dir)

    print("=" * 70)
    print(f"✅ Conversion complete!")
    print(f"   Diagrams converted: {converted}")
    print(f"   Files modified: {files}")
    print("=" * 70)
