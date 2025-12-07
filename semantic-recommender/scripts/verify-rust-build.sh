#!/bin/bash
# Rust Build Verification Script
# Validates all Rust components before A100 deployment

set -e

echo "=================================="
echo "Rust Build Verification"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Rust installation
echo "1. Checking Rust installation..."
if command -v cargo &> /dev/null; then
    RUST_VERSION=$(cargo --version)
    success "Rust installed: $RUST_VERSION"
else
    error "Rust not found. Install from https://rustup.rs"
    exit 1
fi

# Check workspace structure
echo ""
echo "2. Checking workspace structure..."
if [ -f "Cargo.toml" ]; then
    success "Workspace Cargo.toml found"
else
    error "Workspace Cargo.toml not found"
    exit 1
fi

EXPECTED_MEMBERS=("src/rust" "src/api" "crates/cli" "crates/gpu-embeddings" "crates/temporal-cache")
for member in "${EXPECTED_MEMBERS[@]}"; do
    if [ -d "$member" ]; then
        success "Member found: $member"
    else
        warning "Member not found: $member (may be optional)"
    fi
done

# Check CLI crate
echo ""
echo "3. Checking CLI crate..."
if [ -f "crates/cli/Cargo.toml" ]; then
    success "CLI Cargo.toml found"
else
    error "CLI Cargo.toml not found"
    exit 1
fi

CLI_FILES=(
    "crates/cli/src/main.rs"
    "crates/cli/src/commands/mod.rs"
    "crates/cli/src/commands/test.rs"
    "crates/cli/src/commands/bench.rs"
    "crates/cli/tests/integration_test.rs"
)

for file in "${CLI_FILES[@]}"; do
    if [ -f "$file" ]; then
        success "CLI file: $file"
    else
        error "Missing: $file"
        exit 1
    fi
done

# Check documentation
echo ""
echo "4. Checking documentation..."
DOCS=(
    "docs/rust-code-review.md"
    "docs/migration-guide.md"
    "docs/rust-deployment.md"
    "docs/REVIEW-SUMMARY.md"
    "crates/cli/README.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        success "Documentation: $doc"
    else
        warning "Missing: $doc"
    fi
done

# Run cargo check
echo ""
echo "5. Running cargo check..."
if cargo check --workspace --all-features 2>&1 | grep -q "error"; then
    error "Cargo check failed"
    cargo check --workspace --all-features
    exit 1
else
    success "Cargo check passed"
fi

# Check for unsafe code
echo ""
echo "6. Analyzing unsafe code..."
UNSAFE_COUNT=$(grep -r "unsafe" src/rust --include="*.rs" | wc -l)
echo "   Found $UNSAFE_COUNT unsafe occurrences"
if [ "$UNSAFE_COUNT" -gt 0 ]; then
    warning "Review unsafe code in docs/rust-code-review.md"
else
    success "No unsafe code found"
fi

# Check profiles
echo ""
echo "7. Checking build profiles..."
if grep -q "\[profile.a100\]" Cargo.toml; then
    success "A100 profile configured"
else
    warning "A100 profile not found"
fi

if grep -q "\[profile.release\]" Cargo.toml; then
    success "Release profile configured"
else
    error "Release profile not found"
    exit 1
fi

# Test compilation (without running)
echo ""
echo "8. Testing compilation..."
echo "   Building dev profile..."
if cargo build --quiet 2>&1 | grep -q "error"; then
    error "Dev build failed"
    exit 1
else
    success "Dev build succeeded"
fi

echo "   Building release profile (this may take a while)..."
if cargo build --release --quiet 2>&1 | grep -q "error"; then
    error "Release build failed"
    exit 1
else
    success "Release build succeeded"
fi

# Check binary
echo ""
echo "9. Checking binary output..."
if [ -f "target/release/semantic-rec" ]; then
    BINARY_SIZE=$(du -h target/release/semantic-rec | cut -f1)
    success "Binary created: $BINARY_SIZE"
else
    warning "Binary not found at target/release/semantic-rec"
    warning "Build may not have CLI crate as binary"
fi

# Run tests
echo ""
echo "10. Running tests..."
if cargo test --workspace --lib --quiet 2>&1 | grep -q "test result: FAILED"; then
    error "Tests failed"
    cargo test --workspace --lib
    exit 1
else
    success "Unit tests passed"
fi

# Summary
echo ""
echo "=================================="
echo "Verification Summary"
echo "=================================="
echo ""

success "All checks passed!"
echo ""
echo "Next steps:"
echo "  1. Review code: docs/rust-code-review.md"
echo "  2. Run integration tests: cargo test --test integration_test -- --ignored"
echo "  3. Build for A100: cargo build --profile a100 --features gpu"
echo "  4. Deploy to A100 VM: Follow docs/rust-deployment.md"
echo ""
echo "Build artifacts:"
echo "  - Dev binary:     target/debug/semantic-rec"
echo "  - Release binary: target/release/semantic-rec"
echo "  - Documentation:  target/doc/ (run: cargo doc --open)"
echo ""

exit 0
