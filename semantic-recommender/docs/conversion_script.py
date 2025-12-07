#!/usr/bin/env python3
"""
UK English Documentation Conversion Script
Converts American English to British English in markdown files
while preserving code blocks, URLs, and technical identifiers.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# American → British conversions from DOCUMENTATION_ARCHITECTURE.md
CONVERSIONS: Dict[str, str] = {
    # Primary conversions
    r'\boptimize\b': 'optimise',
    r'\boptimized\b': 'optimised',
    r'\boptimizing\b': 'optimising',
    r'\boptimization\b': 'optimisation',
    r'\banalyze\b': 'analyse',
    r'\banalyzed\b': 'analysed',
    r'\banalyzing\b': 'analysing',
    r'\banalysis\b': 'analysis',  # Already correct
    r'\bcolor\b': 'colour',
    r'\bcolored\b': 'coloured',
    r'\bcoloring\b': 'colouring',
    r'\bcolors\b': 'colours',
    r'\bbehavior\b': 'behaviour',
    r'\bbehaviors\b': 'behaviours',
    r'\bbehavioral\b': 'behavioural',
    r'\bcenter\b': 'centre',
    r'\bcentered\b': 'centred',
    r'\bcentering\b': 'centring',
    r'\bcenters\b': 'centres',
    r'\binitialize\b': 'initialise',
    r'\binitialized\b': 'initialised',
    r'\binitializing\b': 'initialising',
    r'\binitialization\b': 'initialisation',
    r'\borganization\b': 'organisation',
    r'\borganizations\b': 'organisations',
    r'\borganizational\b': 'organisational',
    r'\bvisualize\b': 'visualise',
    r'\bvisualized\b': 'visualised',
    r'\bvisualizing\b': 'visualising',
    r'\bvisualization\b': 'visualisation',
    r'\bsynchronize\b': 'synchronise',
    r'\bsynchronized\b': 'synchronised',
    r'\bsynchronizing\b': 'synchronising',
    r'\bsynchronization\b': 'synchronisation',
    r'\bcustomize\b': 'customise',
    r'\bcustomized\b': 'customised',
    r'\bcustomizing\b': 'customising',
    r'\bcustomization\b': 'customisation',
    # Additional common conversions
    r'\brecognize\b': 'recognise',
    r'\brecognized\b': 'recognised',
    r'\brecognizing\b': 'recognising',
    r'\brecognition\b': 'recognition',  # Already correct
    r'\brealize\b': 'realise',
    r'\brealized\b': 'realised',
    r'\brealizing\b': 'realising',
    r'\brealization\b': 'realisation',
    r'\bspecialize\b': 'specialise',
    r'\bspecialized\b': 'specialised',
    r'\bspecializing\b': 'specialising',
    r'\bspecialization\b': 'specialisation',
    r'\bparameterize\b': 'parameterise',
    r'\bparameterized\b': 'parameterised',
    r'\bparameterizing\b': 'parameterising',
    r'\bparameterization\b': 'parameterisation',
    r'\bmodularize\b': 'modularise',
    r'\bmodularized\b': 'modularised',
    r'\bmodularizing\b': 'modularising',
    r'\bmodularization\b': 'modularisation',
}

# Patterns to exclude from conversion
EXCLUDE_PATTERNS = [
    r'```[\s\S]*?```',  # Code blocks
    r'`[^`]+`',  # Inline code
    r'https?://[^\s\)]+',  # URLs
    r'\[.*?\]\(.*?\)',  # Markdown links (check URL part only)
    r'^\s*[#]+\s+.*$',  # Headers with potential code
]


def extract_protected_regions(text: str) -> Tuple[str, List[str]]:
    """Extract code blocks, inline code, and URLs to protect from conversion."""
    placeholders = []
    protected_text = text

    # Extract code blocks first (multiline)
    code_block_pattern = r'```[\s\S]*?```'
    matches = list(re.finditer(code_block_pattern, protected_text))
    for i, match in enumerate(reversed(matches)):
        placeholder = f'<<<CODEBLOCK_{len(matches)-1-i}>>>'
        placeholders.insert(0, match.group(0))
        protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]

    # Extract inline code
    inline_code_pattern = r'`[^`]+`'
    matches = list(re.finditer(inline_code_pattern, protected_text))
    offset = len(placeholders)
    for i, match in enumerate(reversed(matches)):
        placeholder = f'<<<INLINE_{len(matches)-1-i+offset}>>>'
        placeholders.insert(offset, match.group(0))
        protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]

    # Extract URLs
    url_pattern = r'https?://[^\s\)\]>]+'
    matches = list(re.finditer(url_pattern, protected_text))
    offset = len(placeholders)
    for i, match in enumerate(reversed(matches)):
        placeholder = f'<<<URL_{len(matches)-1-i+offset}>>>'
        placeholders.insert(offset, match.group(0))
        protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]

    return protected_text, placeholders


def restore_protected_regions(text: str, placeholders: List[str]) -> str:
    """Restore protected regions after conversion."""
    result = text
    for i, placeholder_text in enumerate(placeholders):
        # Try different placeholder formats
        for pattern in [f'<<<CODEBLOCK_{i}>>>', f'<<<INLINE_{i}>>>', f'<<<URL_{i}>>>']:
            if pattern in result:
                result = result.replace(pattern, placeholder_text, 1)
                break
    return result


def convert_to_uk_english(text: str) -> Tuple[str, int]:
    """Convert American English to British English in markdown text."""
    # Protect code blocks, inline code, and URLs
    protected_text, placeholders = extract_protected_regions(text)

    changes_made = 0
    converted = protected_text

    # Apply conversions
    for american_pattern, british in CONVERSIONS.items():
        # Use word boundaries to avoid partial matches
        new_text = re.sub(american_pattern, british, converted, flags=re.IGNORECASE)
        if new_text != converted:
            # Count changes (case-insensitive)
            changes_made += len(re.findall(american_pattern, converted, re.IGNORECASE))
            converted = new_text

    # Restore protected regions
    final_text = restore_protected_regions(converted, placeholders)

    return final_text, changes_made


def process_file(file_path: Path) -> Tuple[bool, int]:
    """Process a single markdown file."""
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Convert to UK English
        converted_content, changes = convert_to_uk_english(original_content)

        # Only write if changes were made
        if changes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(converted_content)
            return True, changes

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False, 0


def main():
    """Main conversion process."""
    if len(sys.argv) < 2:
        print("Usage: python conversion_script.py <file1.md> [file2.md ...]")
        sys.exit(1)

    total_files_modified = 0
    total_changes = 0
    results = []

    for file_path_str in sys.argv[1:]:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist", file=sys.stderr)
            continue

        modified, changes = process_file(file_path)
        try:
            display_path = file_path.relative_to(Path.cwd())
        except ValueError:
            display_path = file_path

        if modified:
            total_files_modified += 1
            total_changes += changes
            results.append((file_path, changes))
            print(f"✓ {display_path}: {changes} changes")
        else:
            print(f"- {display_path}: no changes")

    # Summary
    print("\n" + "="*60)
    print(f"Total files processed: {len(sys.argv)-1}")
    print(f"Total files modified: {total_files_modified}")
    print(f"Total changes made: {total_changes}")
    print("="*60)

    # Detailed results
    if results:
        print("\nModified files:")
        for file_path, changes in sorted(results, key=lambda x: -x[1]):
            try:
                display_path = file_path.relative_to(Path.cwd())
            except ValueError:
                display_path = file_path
            print(f"  {display_path}: {changes} changes")


if __name__ == '__main__':
    main()
