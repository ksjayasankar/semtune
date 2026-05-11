#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TEXTS_DIR="$ROOT_DIR/resources/texts"
SEED="${SEED:-42}"

cd "$ROOT_DIR"

for text_file in "$TEXTS_DIR"/*.txt; do
    stem="$(basename "$text_file" .txt)"
    out_dir="output/$stem/$SEED"
    echo
    echo "== $stem =="
    python -m semtune --text "$text_file" --seed "$SEED" --out "$out_dir" 2>&1 \
        | grep -v -E "(sentence_transformers|^\\s*$)" || true
done

echo
echo "Done. Artefacts under output/<name>/$SEED/"
