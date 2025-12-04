#!/bin/bash
set -e

# Migration orchestration script
# Usage: ./scripts/run-migration.sh [--dry-run] [--skip-validation]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DRY_RUN=""
SKIP_VALIDATION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--skip-validation]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      NEO4J → HYBRID ARCHITECTURE MIGRATION                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No data will be modified${NC}"
    echo ""
fi

# Phase 1: Preflight Checks
echo -e "${BLUE}═══ Phase 1: Preflight Checks ═══${NC}"
cargo run --bin migrate -- preflight --check-connectivity || {
    echo -e "${RED}❌ Preflight checks failed. Please fix errors and retry.${NC}"
    exit 1
}
echo -e "${GREEN}✅ Preflight checks passed${NC}\n"

# Confirmation prompt
if [[ -z "$DRY_RUN" ]]; then
    read -p "Continue with migration? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Migration cancelled."
        exit 0
    fi
fi

# Phase 2: Deploy Milvus Cluster
echo -e "${BLUE}═══ Phase 2: Deploy Milvus Cluster ═══${NC}"
./scripts/deploy-milvus.sh || {
    echo -e "${RED}❌ Milvus deployment failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Milvus cluster deployed${NC}\n"

# Phase 3: Enable Dual-Write Mode
echo -e "${BLUE}═══ Phase 3: Enable Dual-Write Mode ═══${NC}"
kubectl set env deployment/recommendation-engine STORAGE_MODE=dual_write || {
    echo -e "${RED}❌ Failed to update deployment${NC}"
    exit 1
}
kubectl rollout status deployment/recommendation-engine --timeout=5m || {
    echo -e "${RED}❌ Deployment rollout failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Dual-write mode enabled${NC}\n"

# Wait for pods to stabilize
echo "Waiting 30 seconds for pods to stabilize..."
sleep 30

# Phase 4: Migrate Historical Embeddings
echo -e "${BLUE}═══ Phase 4: Migrate Historical Embeddings ═══${NC}"
cargo run --bin migrate -- migrate-embeddings --batch-size 1000 $DRY_RUN || {
    echo -e "${RED}❌ Embedding migration failed${NC}"
    echo "Rolling back to Neo4j-only mode..."
    cargo run --bin migrate -- rollback --confirm
    exit 1
}
echo -e "${GREEN}✅ Historical embeddings migrated${NC}\n"

# Phase 5: Validate Migration
if [[ "$SKIP_VALIDATION" != true ]]; then
    echo -e "${BLUE}═══ Phase 5: Validate Migration ═══${NC}"
    cargo run --bin migrate -- validate --sample-size 1000 --fix-inconsistencies || {
        echo -e "${RED}❌ Validation failed${NC}"
        read -p "Rollback to Neo4j-only? (yes/no): " rollback
        if [[ "$rollback" == "yes" ]]; then
            cargo run --bin migrate -- rollback --confirm
            exit 1
        fi
    }
    echo -e "${GREEN}✅ Validation passed${NC}\n"
else
    echo -e "${YELLOW}⚠️  Skipping validation${NC}\n"
fi

# Phase 6: Switch to Shadow Milvus Mode
echo -e "${BLUE}═══ Phase 6: Shadow Milvus Mode (Testing) ═══${NC}"
kubectl set env deployment/recommendation-engine STORAGE_MODE=shadow_milvus
kubectl rollout status deployment/recommendation-engine --timeout=5m
echo -e "${GREEN}✅ Shadow mode enabled - monitoring performance...${NC}\n"

# Monitor for 5 minutes
echo "Monitoring performance for 5 minutes..."
timeout 300 cargo run --bin migrate -- monitor --interval-secs 10 || true

# Check metrics
echo -e "\n${BLUE}Checking performance metrics...${NC}"
MILVUS_LATENCY=$(kubectl exec -n default deployment/recommendation-engine -- \
    curl -s localhost:9090/metrics | grep 'milvus_search_latency_ms' | awk '{print $2}' || echo "0")
NEO4J_LATENCY=$(kubectl exec -n default deployment/recommendation-engine -- \
    curl -s localhost:9090/metrics | grep 'neo4j_search_latency_ms' | awk '{print $2}' || echo "0")

echo "  • Milvus latency: ${MILVUS_LATENCY} ms"
echo "  • Neo4j latency: ${NEO4J_LATENCY} ms"

# Phase 7: Switch to Full Hybrid Mode
if [[ -z "$DRY_RUN" ]]; then
    read -p "Performance acceptable. Switch to full hybrid mode? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Staying in shadow mode. Manual intervention required."
        exit 0
    fi
fi

echo -e "${BLUE}═══ Phase 7: Full Hybrid Mode ═══${NC}"
kubectl set env deployment/recommendation-engine STORAGE_MODE=hybrid
kubectl rollout status deployment/recommendation-engine --timeout=5m
echo -e "${GREEN}✅ Hybrid mode activated${NC}\n"

# Phase 8: Generate Report
echo -e "${BLUE}═══ Phase 8: Generate Migration Report ═══${NC}"
REPORT_FILE="migration-report-$(date +%Y%m%d-%H%M%S).md"
cargo run --bin migrate -- report --output-file "$REPORT_FILE" --format markdown
echo -e "${GREEN}✅ Report generated: $REPORT_FILE${NC}\n"

# Final Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         MIGRATION COMPLETED SUCCESSFULLY                  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Next Steps:${NC}"
echo "  1. Monitor application metrics: kubectl port-forward svc/grafana 3000:3000"
echo "  2. Check error rates: kubectl logs -f deployment/recommendation-engine"
echo "  3. Review report: cat $REPORT_FILE"
echo ""
echo -e "${BLUE}🔄 Rollback Command (if needed):${NC}"
echo "  cargo run --bin migrate -- rollback --confirm --preserve-milvus"
echo ""
