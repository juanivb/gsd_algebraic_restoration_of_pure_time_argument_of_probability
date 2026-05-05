#!/usr/bin/env bash
# sync_paper.sh — copy the working paper from paper_latex/paper_30.{tex,pdf}
# into repo/paper/ with the public-facing name.
#
# Usage (from repo root):
#     bash scripts/sync_paper.sh /path/to/paper_latex
#
# Or with default sibling layout:
#     bash scripts/sync_paper.sh
#
# Public-facing name follows the convention used for the rest of the
# GSD programme: gsd_<descriptive_slug>.{tex,pdf}.

set -euo pipefail

# Default: assume the paper_latex/ folder is a sibling of the repo.
DEFAULT_LATEX_DIR="$(cd "$(dirname "$0")/.." && pwd)/../paper_latex"
LATEX_DIR="${1:-$DEFAULT_LATEX_DIR}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/paper"

PUBLIC_NAME="gsd_On_The_Algebraic_Restoration_in_Pure-Time_Of_The_Argument_of_Probability"

if [ ! -f "${LATEX_DIR}/paper_30.tex" ]; then
    echo "Error: ${LATEX_DIR}/paper_30.tex not found." >&2
    echo "Pass the paper_latex/ directory as the first argument." >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

cp "${LATEX_DIR}/paper_30.tex" "${TARGET_DIR}/${PUBLIC_NAME}.tex"

if [ -f "${LATEX_DIR}/paper_30.pdf" ]; then
    cp "${LATEX_DIR}/paper_30.pdf" "${TARGET_DIR}/${PUBLIC_NAME}.pdf"
    echo "✓ Synced PDF: ${TARGET_DIR}/${PUBLIC_NAME}.pdf"
else
    echo "⚠ PDF not found in ${LATEX_DIR}; only .tex was synced."
fi

echo "✓ Synced TeX: ${TARGET_DIR}/${PUBLIC_NAME}.tex"
echo
echo "If this is the first sync after a recompile, remember to:"
echo "    cd ${REPO_ROOT}"
echo "    git add paper/${PUBLIC_NAME}.{tex,pdf}"
echo "    git commit -m 'Sync paper from paper_latex/paper_30.tex'"
