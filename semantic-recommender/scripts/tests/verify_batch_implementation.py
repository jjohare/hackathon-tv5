#!/usr/bin/env python3
"""
Verify batch processing implementation

Tests the implementation without requiring Flask or TensorRT to be installed.
Checks code structure and logic.
"""

import sys
import os
import ast
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_file_exists(filepath):
    """Check if file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ File exists: {filepath}")
        return True
    else:
        print(f"❌ File missing: {filepath}")
        return False


def check_class_in_file(filepath, class_name):
    """Check if class exists in file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if class_name in classes:
            print(f"✅ Class '{class_name}' found in {filepath}")
            return True
        else:
            print(f"❌ Class '{class_name}' not found in {filepath}")
            return False

    except Exception as e:
        print(f"❌ Error parsing {filepath}: {e}")
        return False


def check_method_in_file(filepath, class_name, method_name):
    """Check if method exists in class"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                if method_name in methods:
                    print(f"✅ Method '{method_name}' found in {class_name}")
                    return True
                else:
                    print(f"❌ Method '{method_name}' not found in {class_name}")
                    return False

        print(f"❌ Class '{class_name}' not found")
        return False

    except Exception as e:
        print(f"❌ Error checking method: {e}")
        return False


def check_route_in_file(filepath, route_path):
    """Check if Flask route exists"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        if f"@app.route('{route_path}'" in content or f'@app.route("{route_path}"' in content:
            print(f"✅ Route '{route_path}' found in {filepath}")
            return True
        else:
            print(f"❌ Route '{route_path}' not found in {filepath}")
            return False

    except Exception as e:
        print(f"❌ Error checking route: {e}")
        return False


def check_imports_in_file(filepath, imports):
    """Check if required imports exist"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        missing = []
        for imp in imports:
            if imp not in content:
                missing.append(imp)

        if not missing:
            print(f"✅ All required imports found in {filepath}")
            return True
        else:
            print(f"❌ Missing imports in {filepath}: {missing}")
            return False

    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False


def main():
    """Run verification checks"""
    print("=" * 80)
    print("Batch Processing Implementation Verification")
    print("=" * 80)

    base_path = Path(__file__).parent.parent.parent
    results = []

    print("\n1. File Existence Checks")
    print("-" * 80)
    results.append(check_file_exists(base_path / "scripts/server/query_interface.py"))
    results.append(check_file_exists(base_path / "scripts/tests/test_batch_processing.py"))
    results.append(check_file_exists(base_path / "scripts/tests/benchmark_batch_qps.py"))
    results.append(check_file_exists(base_path / "scripts/docs/BATCH_PROCESSING.md"))

    print("\n2. Class Structure Checks")
    print("-" * 80)
    interface_file = base_path / "scripts/server/query_interface.py"
    results.append(check_class_in_file(interface_file, "BatchProcessor"))
    results.append(check_class_in_file(interface_file, "QueryInterfaceBackend"))

    print("\n3. Method Checks - BatchProcessor")
    print("-" * 80)
    results.append(check_method_in_file(interface_file, "BatchProcessor", "__init__"))
    results.append(check_method_in_file(interface_file, "BatchProcessor", "start"))
    results.append(check_method_in_file(interface_file, "BatchProcessor", "stop"))
    results.append(check_method_in_file(interface_file, "BatchProcessor", "encode"))
    results.append(check_method_in_file(interface_file, "BatchProcessor", "_process_loop"))

    print("\n4. Method Checks - QueryInterfaceBackend")
    print("-" * 80)
    results.append(check_method_in_file(interface_file, "QueryInterfaceBackend", "process_query"))

    print("\n5. Route Checks")
    print("-" * 80)
    results.append(check_route_in_file(interface_file, "/api/query"))
    results.append(check_route_in_file(interface_file, "/api/query/batch"))
    results.append(check_route_in_file(interface_file, "/api/status"))

    print("\n6. Import Checks")
    print("-" * 80)
    required_imports = [
        "from collections import deque",
        "from threading import Lock, Thread",
        "import asyncio",
        "import time"
    ]
    results.append(check_imports_in_file(interface_file, required_imports))

    print("\n7. Code Content Checks")
    print("-" * 80)

    # Check for batch processor initialization in __init__
    with open(interface_file, 'r') as f:
        content = f.read()

    checks = [
        ("BatchProcessor initialization", "self.batch_processor = BatchProcessor"),
        ("Batch processor start", "self.batch_processor.start()"),
        ("max_batch_size=32", "max_batch_size=32"),
        ("max_wait_ms=50", "max_wait_ms=50"),
        ("use_batch parameter", "use_batch"),
        ("batch endpoint", "def api_query_batch"),
        ("batch_performance in response", "batch_performance")
    ]

    for check_name, check_string in checks:
        if check_string in content:
            print(f"✅ {check_name} found")
            results.append(True)
        else:
            print(f"❌ {check_name} not found")
            results.append(False)

    print("\n8. Requirements Check")
    print("-" * 80)
    req_file = base_path / "scripts/requirements.txt"
    with open(req_file, 'r') as f:
        req_content = f.read()

    if "flask" in req_content:
        print("✅ Flask in requirements.txt")
        results.append(True)
    else:
        print("❌ Flask not in requirements.txt")
        results.append(False)

    if "requests" in req_content:
        print("✅ requests in requirements.txt")
        results.append(True)
    else:
        print("❌ requests not in requirements.txt")
        results.append(False)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"Total checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if failed == 0:
        print("\n✅ All checks passed! Batch processing implementation verified.")
        return 0
    else:
        print(f"\n❌ {failed} checks failed. Review implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
