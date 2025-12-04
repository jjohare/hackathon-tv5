#!/usr/bin/env python3
"""
Generate HTML validation report from JSON validation results

Usage:
    python generate_validation_report.py validation-report.json -o report.html
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ontology Validation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        header {{
            border-bottom: 3px solid #007bff;
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}

        h1 {{
            color: #007bff;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .status {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: bold;
            margin-top: 1rem;
        }}

        .status.passed {{
            background: #28a745;
            color: white;
        }}

        .status.failed {{
            background: #dc3545;
            color: white;
        }}

        .section {{
            margin: 2rem 0;
        }}

        .section h2 {{
            color: #555;
            margin-bottom: 1rem;
            border-bottom: 2px solid #eee;
            padding-bottom: 0.5rem;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}

        .stat-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }}

        .stat-card .label {{
            color: #666;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }}

        .errors, .warnings {{
            list-style: none;
        }}

        .errors li, .warnings li {{
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            border-left: 4px solid;
        }}

        .errors li {{
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }}

        .warnings li {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}

        .icon {{
            margin-right: 0.5rem;
        }}

        .success-message {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 4px;
            text-align: center;
            font-size: 1.1rem;
        }}

        footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
            font-size: 0.875rem;
        }}

        .timestamp {{
            color: #999;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Ontology Validation Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
            <div class="status {status_class}">
                {status_icon} {status_text}
            </div>
        </header>

        <section class="section">
            <h2>📊 Statistics</h2>
            <div class="stats">
                {stats_html}
            </div>
        </section>

        {errors_section}

        {warnings_section}

        {success_section}

        <footer>
            <p>GMC-O Ontology Validator | TV5 Monde Media Gateway</p>
        </footer>
    </div>
</body>
</html>
"""


def generate_html_report(json_data: dict, ontology_path: str = "") -> str:
    """Generate HTML report from JSON validation data"""

    passed = json_data.get('passed', False)
    errors = json_data.get('errors', [])
    warnings = json_data.get('warnings', [])
    stats = json_data.get('stats', {})

    # Status
    status_class = 'passed' if passed else 'failed'
    status_text = 'PASSED' if passed else 'FAILED'
    status_icon = '✓' if passed else '✗'

    # Stats HTML
    stats_html = ""
    for key, value in stats.items():
        label = key.replace('_', ' ').title()
        stats_html += f"""
        <div class="stat-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """

    # Errors section
    errors_section = ""
    if errors:
        errors_list = "".join([f'<li><span class="icon">✗</span>{err}</li>' for err in errors])
        errors_section = f"""
        <section class="section">
            <h2>❌ Errors ({len(errors)})</h2>
            <ul class="errors">
                {errors_list}
            </ul>
        </section>
        """

    # Warnings section
    warnings_section = ""
    if warnings:
        warnings_list = "".join([f'<li><span class="icon">⚠</span>{warn}</li>' for warn in warnings])
        warnings_section = f"""
        <section class="section">
            <h2>⚠️ Warnings ({len(warnings)})</h2>
            <ul class="warnings">
                {warnings_list}
            </ul>
        </section>
        """

    # Success section
    success_section = ""
    if not errors and not warnings:
        success_section = """
        <section class="section">
            <div class="success-message">
                ✓ No issues found! Ontology is valid.
            </div>
        </section>
        """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return HTML_TEMPLATE.format(
        timestamp=timestamp,
        status_class=status_class,
        status_icon=status_icon,
        status_text=status_text,
        stats_html=stats_html,
        errors_section=errors_section,
        warnings_section=warnings_section,
        success_section=success_section,
        ontology_path=ontology_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML validation report from JSON"
    )

    parser.add_argument(
        'json_file',
        type=Path,
        help='Path to JSON validation report'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('validation-report.html'),
        help='Output HTML file path (default: validation-report.html)'
    )

    parser.add_argument(
        '--ontology',
        type=str,
        default='',
        help='Path to ontology file (for display)'
    )

    args = parser.parse_args()

    # Load JSON
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)

    # Generate HTML
    html = generate_html_report(json_data, args.ontology)

    # Write output
    with open(args.output, 'w') as f:
        f.write(html)

    print(f"✓ HTML report generated: {args.output}")


if __name__ == '__main__':
    main()
