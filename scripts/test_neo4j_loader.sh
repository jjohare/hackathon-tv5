#!/bin/bash
# Test script for Neo4j integration

set -e

echo "=== Neo4j Integration Test Script ==="

# Check if Neo4j is running
echo "1. Checking Neo4j connectivity..."
if ! nc -z localhost 7687 2>/dev/null; then
    echo "ERROR: Neo4j not running on localhost:7687"
    echo "Start with: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5.14"
    exit 1
fi
echo "✓ Neo4j is accessible"

# Setup schema
echo -e "\n2. Setting up Neo4j schema..."
cypher-shell -u neo4j -p password -f scripts/setup_neo4j_schema.cypher || {
    echo "ERROR: Schema setup failed"
    exit 1
}
echo "✓ Schema setup complete"

# Run integration tests
echo -e "\n3. Running integration tests..."
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"

cd /home/devuser/workspace/hackathon-tv5
cargo test --test neo4j_integration_tests -- --ignored --nocapture

echo -e "\n=== All tests passed ==="
