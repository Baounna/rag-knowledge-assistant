#!/usr/bin/env bash
# Render a few corpus documents to PDF so the PDF connector runs on real
# content. Companies genuinely export policies to PDF, and this puts the
# heading-inference heuristic in parsers/pdf.py under real load -- that parser
# is the weakest link in ingestion and deserves harder input than a synthetic
# file.
#
# Uses cupsfilter (macOS built-in). On Linux use `pandoc` or `wkhtmltopdf`.
set -euo pipefail
SRC="${1:-corpus/markdown}"; DEST="${2:-corpus/pdf}"; COUNT="${3:-12}"
mkdir -p "$DEST"
n=0
for md in $(ls -S "$SRC"/*.md 2>/dev/null | head -"$COUNT"); do
  base=$(basename "$md" .md)
  # Strip frontmatter and markdown syntax; cupsfilter renders plain text.
  awk 'BEGIN{fm=0} /^---$/{fm++; next} fm>=2 || fm==0' "$md" \
    | sed -E 's/\[([^]]+)\]\([^)]+\)/\1/g; s/[*_`]//g; s/^#+ //' \
    > "/tmp/${base}.txt"
  cupsfilter "/tmp/${base}.txt" > "$DEST/${base}.pdf" 2>/dev/null
  rm -f "$md" "/tmp/${base}.txt"    # exists only as a PDF now, not indexed twice
  n=$((n+1))
done
echo "  $n PDFs written to $DEST"
