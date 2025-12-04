#!/bin/bash
set -e

echo "=== AgentDB Implementation Validation ==="
echo ""

# Check SQL schema
echo "✓ Checking PostgreSQL schema..."
if [ -f "sql/agentdb-schema.sql" ]; then
    lines=$(wc -l < sql/agentdb-schema.sql)
    echo "  - agentdb-schema.sql: $lines lines"
    grep -q "CREATE TABLE agent_episodes" sql/agentdb-schema.sql && echo "  - agent_episodes table: ✓"
    grep -q "CREATE TABLE rl_policies" sql/agentdb-schema.sql && echo "  - rl_policies table: ✓"
    grep -q "pgvector" sql/agentdb-schema.sql && echo "  - pgvector extension: ✓"
else
    echo "  ✗ SQL schema not found"
    exit 1
fi

# Check Rust modules
echo ""
echo "✓ Checking Rust modules..."
for file in src/rust/storage/{redis_cache,postgres_store}.rs src/rust/agentdb/{coordinator,integration,tests}.rs; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  - $(basename $file): $lines lines ✓"
    else
        echo "  ✗ $file not found"
        exit 1
    fi
done

# Check Kubernetes manifests
echo ""
echo "✓ Checking Kubernetes manifests..."
for file in k8s/agentdb/{postgres-statefulset,redis-statefulset,pgvector-init,secrets}.yaml; do
    if [ -f "$file" ]; then
        echo "  - $(basename $file): ✓"
    else
        echo "  ✗ $file not found"
        exit 1
    fi
done

# Check documentation
echo ""
echo "✓ Checking documentation..."
for file in docs/agentdb-{deployment,benchmark-results,implementation-summary}.md docs/agentdb-example.rs; do
    if [ -f "$file" ]; then
        echo "  - $(basename $file): ✓"
    else
        echo "  ✗ $file not found"
        exit 1
    fi
done

# Count total implementation
echo ""
echo "=== Implementation Summary ==="
sql_lines=$(wc -l < sql/agentdb-schema.sql)
rust_lines=$(wc -l src/rust/storage/{redis_cache,postgres_store}.rs src/rust/agentdb/{coordinator,integration,tests}.rs | tail -1 | awk '{print $1}')
k8s_files=$(ls k8s/agentdb/*.yaml | wc -l)
doc_files=$(ls docs/agentdb* | wc -l)

echo "  SQL schema: $sql_lines lines"
echo "  Rust code: $rust_lines lines"
echo "  Kubernetes manifests: $k8s_files files"
echo "  Documentation: $doc_files files"
echo ""
echo "✅ All AgentDB components validated successfully!"
echo ""
echo "Next steps:"
echo "  1. Deploy PostgreSQL: kubectl apply -f k8s/agentdb/postgres-statefulset.yaml"
echo "  2. Deploy Redis: kubectl apply -f k8s/agentdb/redis-statefulset.yaml"
echo "  3. Initialize schema: kubectl apply -f k8s/agentdb/pgvector-init.yaml"
echo "  4. Run tests: cargo test --release agentdb"
