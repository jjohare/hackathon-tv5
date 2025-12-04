#!/bin/bash

# Quick rollback script
# Usage: ./scripts/rollback.sh [--confirm] [--delete-milvus]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONFIRM=false
DELETE_MILVUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --confirm)
            CONFIRM=true
            shift
            ;;
        --delete-milvus)
            DELETE_MILVUS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--confirm] [--delete-milvus]"
            exit 1
            ;;
    esac
done

if [ "$CONFIRM" = false ]; then
    echo -e "${YELLOW}⚠️  ROLLBACK PREVIEW${NC}"
    echo ""
    echo "This will:"
    echo "  1. Switch back to Neo4j-only mode"
    echo "  2. Update Kubernetes deployment"
    echo "  3. Route all queries to Neo4j"
    if [ "$DELETE_MILVUS" = true ]; then
        echo -e "  4. ${RED}DELETE Milvus data${NC}"
    else
        echo "  4. Preserve Milvus data for retry"
    fi
    echo ""
    echo "Re-run with --confirm to execute"
    exit 0
fi

echo -e "${RED}🔄 Starting rollback...${NC}\n"

# Run migration tool rollback
if [ "$DELETE_MILVUS" = true ]; then
    cargo run --bin migrate -- rollback --confirm
else
    cargo run --bin migrate -- rollback --confirm --preserve-milvus
fi

echo ""
echo -e "${GREEN}✅ Rollback completed${NC}"
echo ""
echo "Verify deployment:"
echo "  kubectl rollout status deployment/recommendation-engine"
echo "  kubectl logs -f deployment/recommendation-engine"
