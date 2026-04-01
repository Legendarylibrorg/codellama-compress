#!/usr/bin/env bash
set -euo pipefail

# Thin convenience wrapper around the Python CLI.
#
# Usage:
#   ./run.sh
#
# Optional env:
#   OUT_ROOT=output/runs
#   RUN_ID=...
#   CONFIG=path/to/config.json

OUT_ROOT="${OUT_ROOT:-output/runs}"
RUN_ID="${RUN_ID:-}"
CONFIG="${CONFIG:-}"

args=(--out-root "$OUT_ROOT")
if [ -n "$RUN_ID" ]; then
  args+=(--run-id "$RUN_ID")
fi
if [ -n "$CONFIG" ]; then
  args+=(--config "$CONFIG")
fi

codellama-compress distill run "${args[@]}"

RUN_DIR="$(ls -1dt "$OUT_ROOT"/* | head -n 1)"

codellama-compress prune mask-mlp --model-dir "$RUN_DIR/distilled" "${args[@]}"
codellama-compress finetune run --model-dir "$RUN_DIR/pruned" "${args[@]}"

echo "Done. Latest run: $RUN_DIR"
