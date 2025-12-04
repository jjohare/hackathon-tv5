#!/usr/bin/env python3
"""
Performance Test Results Analysis
Analyzes load test JSON output and generates visualizations
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_file):
    """Load JSON results file"""
    with open(results_file) as f:
        return json.load(f)


def analyze_load_test(data):
    """Analyze load test results and print summary"""
    print("\n" + "="*50)
    print("         LOAD TEST RESULTS")
    print("="*50)
    print(f"Total Requests:     {data['total_requests']:,}")
    print(f"Successful:         {data['successful_requests']:,}")
    print(f"Failed:             {data['failed_requests']:,}")
    print(f"Duration:           {data['duration_secs']:.1f}s")
    print(f"Actual QPS:         {data['actual_qps']:.1f}")
    print()
    print("Latency Distribution:")
    print(f"  p50:   {data['p50_latency_ms']:>8.2f}ms")
    print(f"  p95:   {data['p95_latency_ms']:>8.2f}ms")
    print(f"  p99:   {data['p99_latency_ms']:>8.2f}ms")
    print(f"  p99.9: {data['p99_9_latency_ms']:>8.2f}ms")
    print(f"  max:   {data['max_latency_ms']:>8.2f}ms")
    print()
    print(f"Error Rate:         {data['error_rate']*100:.3f}%")
    print()

    # Target validation
    print("="*50)
    print("       TARGET VALIDATION")
    print("="*50)

    targets = []

    if data['p99_latency_ms'] < 10.0:
        targets.append(("✓", f"p99 latency: {data['p99_latency_ms']:.2f}ms", "< 10ms", True))
    else:
        targets.append(("✗", f"p99 latency: {data['p99_latency_ms']:.2f}ms", "< 10ms", False))

    if data['actual_qps'] >= 7000:
        targets.append(("✓", f"Sustained QPS: {data['actual_qps']:.0f}", "≥ 7,000", True))
    else:
        targets.append(("✗", f"Sustained QPS: {data['actual_qps']:.0f}", "≥ 7,000", False))

    if data['error_rate'] < 0.01:
        targets.append(("✓", f"Error rate: {data['error_rate']*100:.2f}%", "< 1%", True))
    else:
        targets.append(("✗", f"Error rate: {data['error_rate']*100:.2f}%", "< 1%", False))

    for icon, metric, target, passed in targets:
        print(f"{icon} {metric:40s} (target: {target})")

    all_passed = all(t[3] for t in targets)
    print()
    if all_passed:
        print("🎉 ALL TARGETS MET!")
    else:
        print("⚠️  Some targets not met - review performance")
    print("="*50)
    print()

    return targets


def plot_latency_distribution(data, output_dir):
    """Generate latency distribution plots"""
    measurements = data['measurements']
    latencies = [m['latency_us'] / 1000.0 for m in measurements if m['success']]

    if not latencies:
        print("No successful measurements to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(latencies, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=10, color='r', linestyle='--', linewidth=2, label='p99 target (10ms)')
    ax.axvline(x=data['p99_latency_ms'], color='g', linestyle='--', linewidth=2,
               label=f'Actual p99 ({data["p99_latency_ms"]:.2f}ms)')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Time series
    ax = axes[0, 1]
    timestamps = [m['timestamp'] / 1_000_000.0 for m in measurements if m['success']]  # Convert to seconds
    ax.plot(timestamps, latencies, alpha=0.5, linewidth=0.5)
    ax.axhline(y=10, color='r', linestyle='--', linewidth=2, label='p99 target')
    ax.axhline(y=data['p99_latency_ms'], color='g', linestyle='--', linewidth=2,
               label=f'Actual p99')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CDF
    ax = axes[1, 0]
    sorted_latencies = np.sort(latencies)
    cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
    ax.plot(sorted_latencies, cdf * 100)
    ax.axvline(x=10, color='r', linestyle='--', linewidth=2, label='p99 target')
    ax.axhline(y=99, color='orange', linestyle='--', linewidth=1, label='p99 line')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative Percentage')
    ax.set_title('Latency CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Box plot by query type
    ax = axes[1, 1]
    query_types = {}
    for m in measurements:
        if m['success']:
            qtype = m.get('query_type', 'unknown')
            if qtype not in query_types:
                query_types[qtype] = []
            query_types[qtype].append(m['latency_us'] / 1000.0)

    if query_types:
        ax.boxplot(query_types.values(), labels=query_types.keys())
        ax.axhline(y=10, color='r', linestyle='--', linewidth=2, label='p99 target')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency by Query Type')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'latency-analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


def plot_throughput_analysis(data, output_dir):
    """Generate throughput analysis plots"""
    measurements = data['measurements']

    # Calculate QPS over time (1-second windows)
    max_timestamp = max(m['timestamp'] for m in measurements)
    window_size = 1_000_000  # 1 second in microseconds
    num_windows = int(max_timestamp / window_size) + 1

    qps_by_window = [0] * num_windows
    errors_by_window = [0] * num_windows

    for m in measurements:
        window = int(m['timestamp'] / window_size)
        qps_by_window[window] += 1
        if not m['success']:
            errors_by_window[window] += 1

    time_windows = list(range(num_windows))
    error_rate_by_window = [
        (errors / qps * 100) if qps > 0 else 0
        for errors, qps in zip(errors_by_window, qps_by_window)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # QPS over time
    ax = axes[0]
    ax.plot(time_windows, qps_by_window, linewidth=2)
    ax.axhline(y=7000, color='r', linestyle='--', linewidth=2, label='Target (7,000 QPS)')
    ax.fill_between(time_windows, 0, qps_by_window, alpha=0.3)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Queries per Second')
    ax.set_title('Throughput Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error rate over time
    ax = axes[1]
    ax.plot(time_windows, error_rate_by_window, color='red', linewidth=2)
    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Target (<1%)')
    ax.fill_between(time_windows, 0, error_rate_by_window, color='red', alpha=0.3)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'throughput-analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


def generate_markdown_report(data, targets, output_dir):
    """Generate Markdown report"""
    report_path = Path(output_dir) / 'performance-report.md'

    with open(report_path, 'w') as f:
        f.write("# Performance Test Report\n\n")
        f.write(f"**Test Date:** {data.get('timestamp', 'N/A')}\n\n")

        f.write("## Executive Summary\n\n")
        all_passed = all(t[3] for t in targets)
        if all_passed:
            f.write("✅ **All performance targets met**\n\n")
        else:
            f.write("⚠️ **Some performance targets not met**\n\n")

        f.write("## Test Configuration\n\n")
        f.write(f"- **Duration:** {data['duration_secs']:.1f}s\n")
        f.write(f"- **Total Requests:** {data['total_requests']:,}\n")
        f.write(f"- **Actual QPS:** {data['actual_qps']:.1f}\n\n")

        f.write("## Results\n\n")
        f.write("### Latency\n\n")
        f.write("| Percentile | Latency (ms) |\n")
        f.write("|------------|-------------:|\n")
        f.write(f"| p50        | {data['p50_latency_ms']:>8.2f} |\n")
        f.write(f"| p95        | {data['p95_latency_ms']:>8.2f} |\n")
        f.write(f"| p99        | {data['p99_latency_ms']:>8.2f} |\n")
        f.write(f"| p99.9      | {data['p99_9_latency_ms']:>8.2f} |\n")
        f.write(f"| max        | {data['max_latency_ms']:>8.2f} |\n\n")

        f.write("### Target Validation\n\n")
        f.write("| Target | Actual | Status |\n")
        f.write("|--------|--------|--------|\n")
        for icon, metric, target, passed in targets:
            status = "✅ Pass" if passed else "❌ Fail"
            f.write(f"| {target} | {metric.replace(icon, '').strip()} | {status} |\n")
        f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("![Latency Analysis](latency-analysis.png)\n\n")
        f.write("![Throughput Analysis](throughput-analysis.png)\n\n")

    print(f"Report saved to: {report_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze-results.py <results-file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    output_dir = Path(results_file).parent

    print(f"Analyzing: {results_file}")

    data = load_results(results_file)
    targets = analyze_load_test(data)

    print("\nGenerating visualizations...")
    plot_latency_distribution(data, output_dir)
    plot_throughput_analysis(data, output_dir)

    print("\nGenerating report...")
    generate_markdown_report(data, targets, output_dir)

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
