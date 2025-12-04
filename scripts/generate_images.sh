#!/bin/bash
# Generate PNG/SVG images from GraphViz DOT files
# Requires: graphviz (sudo apt-get install graphviz)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_DIR="$SCRIPT_DIR/../design/ontology/visualizations"

echo "🎨 Generating PNG/SVG images from DOT files..."
echo "================================================"

# Check if graphviz is installed
if ! command -v dot &> /dev/null; then
    echo "❌ Error: GraphViz not installed"
    echo ""
    echo "Install GraphViz:"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  macOS:         brew install graphviz"
    echo "  Alpine:        apk add graphviz"
    exit 1
fi

cd "$VIZ_DIR"

# Generate full hierarchy in both formats
echo "📊 Generating full hierarchy..."
dot -Tpng full-hierarchy.dot -o full-hierarchy.png
dot -Tsvg full-hierarchy.dot -o full-hierarchy.svg
echo "   ✓ full-hierarchy.png"
echo "   ✓ full-hierarchy.svg"

# Generate namespace-specific views
echo ""
echo "📦 Generating namespace views..."
for ns in media user ctx gpu; do
    dot -Tpng ${ns}-namespace.dot -o ${ns}-namespace.png
    dot -Tsvg ${ns}-namespace.dot -o ${ns}-namespace.svg
    echo "   ✓ ${ns}-namespace.png"
    echo "   ✓ ${ns}-namespace.svg"
done

echo ""
echo "✅ Image generation complete!"
echo ""
echo "📁 Output directory: $VIZ_DIR"
ls -lh *.png *.svg 2>/dev/null || true
