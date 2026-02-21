#!/usr/bin/env bash
#
# Build PDF documentation from Markdown with Mermaid diagram support.
#
# Usage:
#   ./scripts/build_docs.sh [--check-only]
#
# Options:
#   --check-only    Only check if required tools are installed
#
# Prerequisites:
#   - pandoc
#   - mermaid-cli (mmdc): npm i -g @mermaid-js/mermaid-cli
#   - xelatex or tectonic (PDF engine)
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories (relative to repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
FILTER_PATH="$DOCS_DIR/filters/mermaid.lua"
INPUT_FILE="$DOCS_DIR/REPORT.md"
OUTPUT_FILE="$DOCS_DIR/REPORT.pdf"

# Configuration
PDF_ENGINE=""  # Auto-detected

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

check_command() {
    local cmd="$1"
    local install_hint="${2:-}"
    
    if command -v "$cmd" &> /dev/null; then
        success "$cmd found: $(command -v "$cmd")"
        return 0
    else
        error "$cmd not found"
        if [[ -n "$install_hint" ]]; then
            echo "       Install: $install_hint"
        fi
        return 1
    fi
}

detect_pdf_engine() {
    if command -v xelatex &> /dev/null; then
        PDF_ENGINE="xelatex"
        return 0
    elif command -v tectonic &> /dev/null; then
        PDF_ENGINE="tectonic"
        return 0
    elif command -v pdflatex &> /dev/null; then
        PDF_ENGINE="pdflatex"
        return 0
    else
        return 1
    fi
}

check_prerequisites() {
    local has_errors=0
    
    echo ""
    info "Checking prerequisites..."
    echo ""
    
    # Check pandoc
    check_command "pandoc" "https://pandoc.org/installing.html" || has_errors=1
    
    # Check mermaid-cli
    check_command "mmdc" "npm install -g @mermaid-js/mermaid-cli" || has_errors=1
    
    # Check PDF engine
    if detect_pdf_engine; then
        success "PDF engine found: $PDF_ENGINE"
    else
        error "No PDF engine found (xelatex, tectonic, or pdflatex)"
        echo "       Install one of:"
        echo "         - TeX Live: https://tug.org/texlive/"
        echo "         - MacTeX (macOS): brew install --cask mactex"
        echo "         - Tectonic: brew install tectonic / cargo install tectonic"
        has_errors=1
    fi
    
    echo ""
    
    if [[ $has_errors -eq 1 ]]; then
        error "Missing prerequisites. Please install the required tools."
        return 1
    fi
    
    success "All prerequisites satisfied"
    return 0
}

build_pdf() {
    info "Building PDF documentation..."
    echo ""
    
    # Ensure output directories exist
    mkdir -p "$DOCS_DIR/diagrams"
    mkdir -p "$DOCS_DIR/_cache"
    
    # Check input file exists
    if [[ ! -f "$INPUT_FILE" ]]; then
        error "Input file not found: $INPUT_FILE"
        exit 1
    fi
    
    # Check filter exists
    if [[ ! -f "$FILTER_PATH" ]]; then
        error "Lua filter not found: $FILTER_PATH"
        exit 1
    fi
    
    # Detect PDF engine if not set
    if [[ -z "$PDF_ENGINE" ]]; then
        if ! detect_pdf_engine; then
            error "No PDF engine available"
            exit 1
        fi
    fi
    
    info "Using PDF engine: $PDF_ENGINE"
    info "Input: $INPUT_FILE"
    info "Output: $OUTPUT_FILE"
    info "Filter: $FILTER_PATH"
    echo ""
    
    # Change to docs directory for relative paths to work
    cd "$DOCS_DIR"
    
    # Build the PDF
    pandoc \
        "REPORT.md" \
        --lua-filter="filters/mermaid.lua" \
        --pdf-engine="$PDF_ENGINE" \
        -V geometry:margin=1in \
        -V colorlinks=true \
        -V linkcolor=blue \
        -V urlcolor=blue \
        --standalone \
        --toc \
        --toc-depth=3 \
        -o "REPORT.pdf"
    
    echo ""
    
    if [[ -f "$OUTPUT_FILE" ]]; then
        success "PDF generated successfully: $OUTPUT_FILE"
        ls -lh "$OUTPUT_FILE"
    else
        error "PDF generation failed"
        exit 1
    fi
}

main() {
    echo ""
    echo "==========================================="
    echo "   Speech Emotion Recognition - Doc Build "
    echo "==========================================="
    
    # Handle --check-only flag
    if [[ "${1:-}" == "--check-only" ]]; then
        check_prerequisites
        exit $?
    fi
    
    # Check prerequisites first
    if ! check_prerequisites; then
        exit 1
    fi
    
    echo ""
    
    # Build the PDF
    build_pdf
}

main "$@"
