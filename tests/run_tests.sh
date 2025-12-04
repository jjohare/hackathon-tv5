#!/bin/bash
set -e

# Comprehensive Test Runner for Hybrid Architecture
# Usage: ./tests/run_tests.sh [quick|full|chaos|load|bench]

COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_NC='\033[0m'

log_info() {
    echo -e "${COLOR_GREEN}[INFO]${COLOR_NC} $1"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[WARN]${COLOR_NC} $1"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_NC} $1"
}

check_docker() {
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is required but not installed"
        exit 1
    fi
}

start_services() {
    log_info "Starting test infrastructure..."
    docker-compose -f tests/docker-compose.test.yml up -d

    log_info "Waiting for services to be ready..."
    sleep 20

    # Health checks
    log_info "Checking Milvus..."
    curl -f http://localhost:9091/healthz || log_warn "Milvus may not be ready"

    log_info "Checking Neo4j..."
    docker exec $(docker ps -q -f name=neo4j) cypher-shell -u neo4j -p testpassword "RETURN 1" || log_warn "Neo4j may not be ready"

    log_info "Checking PostgreSQL..."
    docker exec $(docker ps -q -f name=postgres) pg_isready -U postgres || log_warn "PostgreSQL may not be ready"

    log_info "Checking Redis..."
    docker exec $(docker ps -q -f name=redis) redis-cli ping || log_warn "Redis may not be ready"

    log_info "All services started"
}

stop_services() {
    log_info "Stopping test infrastructure..."
    docker-compose -f tests/docker-compose.test.yml down -v
}

run_quick_tests() {
    log_info "Running quick integration tests..."
    cargo test --test hybrid_integration_tests --release -- \
        test_hybrid_search_latency_p99_under_10ms \
        test_vector_search_accuracy \
        test_cache_effectiveness

    if [ $? -eq 0 ]; then
        log_info "✅ Quick tests PASSED"
    else
        log_error "❌ Quick tests FAILED"
        exit 1
    fi
}

run_full_integration_tests() {
    log_info "Running full integration test suite..."
    cargo test --test hybrid_integration_tests --release -- --test-threads=1

    if [ $? -eq 0 ]; then
        log_info "✅ Integration tests PASSED"
    else
        log_error "❌ Integration tests FAILED"
        exit 1
    fi
}

run_chaos_tests() {
    log_warn "Running DESTRUCTIVE chaos tests..."
    log_warn "This will kill containers and simulate failures"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Chaos tests skipped"
        return
    fi

    cargo test --test chaos_tests --release -- --ignored --test-threads=1

    if [ $? -eq 0 ]; then
        log_info "✅ Chaos tests PASSED"
    else
        log_error "❌ Chaos tests FAILED"
    fi

    # Restart services after chaos
    stop_services
    start_services
}

run_load_tests() {
    log_warn "Running load tests (7000 QPS)..."
    log_warn "This requires significant CPU/memory resources"
    log_warn "Recommended: 16+ CPU cores, 32GB+ RAM"

    # Check system resources
    CPU_CORES=$(nproc)
    if [ $CPU_CORES -lt 8 ]; then
        log_error "Insufficient CPU cores: $CPU_CORES (recommended: 16+)"
        exit 1
    fi

    log_info "CPU cores: $CPU_CORES"

    cargo test --test load_tests --release -- \
        test_sustained_load_7000_qps \
        --ignored \
        --nocapture

    if [ $? -eq 0 ]; then
        log_info "✅ Load tests PASSED"
    else
        log_error "❌ Load tests FAILED"
    fi
}

run_benchmarks() {
    log_info "Running performance benchmarks..."

    cargo bench --bench hybrid_benchmarks -- \
        --output-format bencher | tee benchmark_results.txt

    log_info "Benchmark results saved to benchmark_results.txt"

    # Parse and validate key metrics
    log_info "Key metrics:"
    grep "vector_search" benchmark_results.txt | head -5
    grep "hybrid_vs" benchmark_results.txt
}

run_all_tests() {
    log_info "Running COMPLETE test suite..."

    run_full_integration_tests
    run_benchmarks

    log_info "All non-destructive tests complete"
    log_warn "Run './tests/run_tests.sh chaos' for chaos tests"
    log_warn "Run './tests/run_tests.sh load' for load tests"
}

cleanup() {
    log_info "Cleaning up..."
    stop_services
    log_info "Cleanup complete"
}

# Main script
check_docker

MODE=${1:-full}

case $MODE in
    quick)
        start_services
        run_quick_tests
        stop_services
        ;;
    full)
        start_services
        run_all_tests
        stop_services
        ;;
    chaos)
        start_services
        run_chaos_tests
        stop_services
        ;;
    load)
        start_services
        run_load_tests
        stop_services
        ;;
    bench)
        start_services
        run_benchmarks
        stop_services
        ;;
    *)
        log_error "Unknown mode: $MODE"
        echo "Usage: $0 [quick|full|chaos|load|bench]"
        exit 1
        ;;
esac

log_info "Test run complete!"
