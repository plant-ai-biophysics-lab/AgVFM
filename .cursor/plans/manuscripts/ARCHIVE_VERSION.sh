#!/bin/bash
# Script to archive current manuscript version before making changes
# Usage: ./ARCHIVE_VERSION.sh v1.0 "Description of changes"

set -e

VERSION=${1:-"v1.0"}
DESCRIPTION=${2:-"Archived version"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS_DIR="${SCRIPT_DIR}/versions/${VERSION}"

echo "📦 Archiving manuscript version ${VERSION}..."

# Create version directory
mkdir -p "${VERSIONS_DIR}"

# Copy current files
cp "${SCRIPT_DIR}/manuscript.md" "${VERSIONS_DIR}/"
cp "${SCRIPT_DIR}/manuscript.pdf" "${VERSIONS_DIR}/"

echo "✅ Archived to: ${VERSIONS_DIR}"
echo ""
echo "📝 Next steps:"
echo "   1. Update manuscript_index.md with version ${VERSION} entry"
echo "   2. Add description: ${DESCRIPTION}"
echo "   3. Make your changes to manuscript.md"
echo "   4. Regenerate PDF: python scripts/generate_plan_pdf.py .cursor/plans/manuscripts/manuscript.md"
