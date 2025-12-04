#!/usr/bin/env python3
"""
Convert ASCII diagrams to Mermaid format based on doc-alignment report.
Processes all 55 diagrams across multiple files.
"""

import json
import sys
from pathlib import Path

def load_report():
    """Load the ASCII diagram report"""
    report_path = Path('/home/devuser/workspace/hackathon-tv5/docs/.doc-alignment-reports/ascii.json')
    with open(report_path, 'r') as f:
        return json.load(f)

def load_file(filepath):
    """Load file content as lines"""
    base = Path('/home/devuser/workspace/hackathon-tv5')
    full_path = base / filepath
    with open(full_path, 'r') as f:
        return f.readlines()

def save_file(filepath, lines):
    """Save lines to file"""
    base = Path('/home/devuser/workspace/hackathon-tv5')
    full_path = base / filepath
    with open(full_path, 'w') as f:
        f.writelines(lines)

def convert_diagrams():
    """Convert all ASCII diagrams to Mermaid"""
    report = load_report()
    diagrams_by_file = {}

    # Group diagrams by file
    for diag in report['ascii_diagrams']:
        file = diag['file']
        if file not in diagrams_by_file:
            diagrams_by_file[file] = []
        diagrams_by_file[file].append(diag)

    # Process each file
    for filepath, diagrams in diagrams_by_file.items():
        print(f"\nProcessing {filepath} ({len(diagrams)} diagrams)...")

        # Load file content
        lines = load_file(filepath)

        # Sort diagrams by line number (reverse order to maintain line numbers)
        diagrams_sorted = sorted(diagrams, key=lambda d: d['start_line'], reverse=True)

        # Replace each diagram
        converted = 0
        for diag in diagrams_sorted:
            start = diag['start_line'] - 1  # 0-indexed
            end = diag['end_line']  # end_line is exclusive

            # Extract mermaid conversion
            mermaid_code = diag['mermaid']

            # Create replacement block with mermaid
            replacement = [
                f"```mermaid\n",
                f"{mermaid_code}\n",
                f"```\n",
                f"\n",
                f"<!-- Original ASCII diagram preserved:\n",
                *lines[start:end],
                f"-->\n",
                f"\n"
            ]

            # Replace lines
            lines[start:end] = replacement
            converted += 1

            print(f"  ✓ Converted diagram at line {diag['start_line']} (type: {diag['diagram_type']})")

        # Save modified file
        save_file(filepath, lines)
        print(f"  Saved {filepath} with {converted} conversions")

    print(f"\n✅ Total converted: {report['total_detected']} ASCII diagrams")
    print(f"   Files processed: {len(diagrams_by_file)}")

if __name__ == '__main__':
    try:
        convert_diagrams()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
