#!/usr/bin/env bash
# Build every paper figure from its standalone .tex into ../../figures/ as pdf + png.
# The figures/ folder is meant to hold ONLY the rendered pdf/png; the generators live here.
#
#   Usage:  bash scripts/figures/build_figures.sh      (from the paper root, or anywhere)
#
# Requires: pdflatex (with tikz/pgfplots/standalone) and pdftocairo (poppler).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FIGDIR="$(cd "$HERE/../.." && pwd)/figures"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$FIGDIR"

FIGURES=(architecture quadtree mechanism baseline_iqr perf_distribution)

for name in "${FIGURES[@]}"; do
    echo "building $name ..."
    # Two passes so pgfplots fillbetween / axis sizing settle.
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMP" "$HERE/$name.tex" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMP" "$HERE/$name.tex" >/dev/null
    cp "$TMP/$name.pdf" "$FIGDIR/$name.pdf"
    pdftocairo -png -r 300 -singlefile "$FIGDIR/$name.pdf" "$FIGDIR/$name"
done

echo "done -> $FIGDIR"
ls -1 "$FIGDIR"
