#!/bin/bash

# Monitor latency during migration
# Usage: ./scripts/monitor-latency.sh [duration_minutes]

DURATION=${1:-60}  # Default 60 minutes
INTERVAL=10        # Check every 10 seconds

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Monitoring latency for ${DURATION} minutes (Ctrl+C to stop)${NC}\n"

END_TIME=$(($(date +%s) + DURATION * 60))

while [ $(date +%s) -lt $END_TIME ]; do
    # Query Prometheus metrics
    NEO4J_P50=$(curl -s http://localhost:9090/api/v1/query \
        --data-urlencode 'query=histogram_quantile(0.50, rate(neo4j_search_latency_ms_bucket[1m]))' | \
        jq -r '.data.result[0].value[1] // "N/A"')

    NEO4J_P95=$(curl -s http://localhost:9090/api/v1/query \
        --data-urlencode 'query=histogram_quantile(0.95, rate(neo4j_search_latency_ms_bucket[1m]))' | \
        jq -r '.data.result[0].value[1] // "N/A"')

    MILVUS_P50=$(curl -s http://localhost:9090/api/v1/query \
        --data-urlencode 'query=histogram_quantile(0.50, rate(milvus_search_latency_ms_bucket[1m]))' | \
        jq -r '.data.result[0].value[1] // "N/A"')

    MILVUS_P95=$(curl -s http://localhost:9090/api/v1/query \
        --data-urlencode 'query=histogram_quantile(0.95, rate(milvus_search_latency_ms_bucket[1m]))' | \
        jq -r '.data.result[0].value[1] // "N/A"')

    ERROR_RATE=$(curl -s http://localhost:9090/api/v1/query \
        --data-urlencode 'query=rate(migration_errors_total[1m])' | \
        jq -r '.data.result[0].value[1] // "0"')

    DUAL_WRITE_SUCCESS=$(curl -s http://localhost:9090/api/v1/query \
        --data-urlencode 'query=(dual_write_success_total / dual_write_attempts_total) * 100' | \
        jq -r '.data.result[0].value[1] // "N/A"')

    # Clear screen and display
    clear
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           MIGRATION LATENCY MONITOR                        ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Neo4j latency
    echo -e "${YELLOW}Neo4j Latency:${NC}"
    if [ "$NEO4J_P50" != "N/A" ]; then
        printf "  • p50: %8.2f ms\n" "$NEO4J_P50"
        printf "  • p95: %8.2f ms\n" "$NEO4J_P95"
    else
        echo "  • No data"
    fi
    echo ""

    # Milvus latency
    echo -e "${GREEN}Milvus Latency:${NC}"
    if [ "$MILVUS_P50" != "N/A" ]; then
        printf "  • p50: %8.2f ms\n" "$MILVUS_P50"
        printf "  • p95: %8.2f ms\n" "$MILVUS_P95"

        # Calculate speedup
        if [ "$NEO4J_P95" != "N/A" ] && [ "$(echo "$MILVUS_P95 > 0" | bc)" -eq 1 ]; then
            SPEEDUP=$(echo "scale=2; $NEO4J_P95 / $MILVUS_P95" | bc)
            echo -e "  • ${GREEN}Speedup: ${SPEEDUP}x${NC}"
        fi
    else
        echo "  • No data"
    fi
    echo ""

    # Error rate
    echo -e "${BLUE}System Health:${NC}"
    ERROR_RATE_FLOAT=$(printf "%.4f" "$ERROR_RATE")
    if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
        echo -e "  • Error rate: ${RED}${ERROR_RATE_FLOAT}/sec${NC}"
    else
        echo -e "  • Error rate: ${GREEN}${ERROR_RATE_FLOAT}/sec${NC}"
    fi

    if [ "$DUAL_WRITE_SUCCESS" != "N/A" ]; then
        DUAL_WRITE_FLOAT=$(printf "%.2f" "$DUAL_WRITE_SUCCESS")
        if (( $(echo "$DUAL_WRITE_SUCCESS < 99" | bc -l) )); then
            echo -e "  • Dual-write success: ${YELLOW}${DUAL_WRITE_FLOAT}%${NC}"
        else
            echo -e "  • Dual-write success: ${GREEN}${DUAL_WRITE_FLOAT}%${NC}"
        fi
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"

    sleep $INTERVAL
done

echo -e "\n${GREEN}Monitoring completed${NC}"
