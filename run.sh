#!/usr/bin/env bash
set -euo pipefail

# Thin convenience wrapper around the Python CLI.
#
# Usage:
#   ./run.sh
#
# Optional env:
#   OUT_ROOT=output/runs
#   RUN_ID=...          # explicit run id (overrides hash-derived id)
#   CONFIG=path/to/config.json
#   REPLAY_FROM=...     # prior run dir for prune/finetune hash checks

OUT_ROOT="${OUT_ROOT:-output/runs}"
RUN_ID="${RUN_ID:-}"
CONFIG="${CONFIG:-}"
REPLAY_FROM="${REPLAY_FROM:-}"

args=(--out-root "$OUT_ROOT")
if [ -n "$RUN_ID" ]; then
  args+=(--run-id "$RUN_ID")
fi
if [ -n "$CONFIG" ]; then
  args+=(--config "$CONFIG")
fi

codellama-compress distill run "${args[@]}"

if [ -n "$RUN_ID" ]; then
  RUN_DIR="$OUT_ROOT/$RUN_ID"
elif [ -f "$OUT_ROOT/.last_run" ]; then
  RUN_DIR="$OUT_ROOT/$(tr -d '\n' < "$OUT_ROOT/.last_run")"
else
  echo "error: could not resolve run directory (set RUN_ID or check $OUT_ROOT/.last_run)" >&2
  exit 1
fi

replay_args=()
if [ -n "$REPLAY_FROM" ]; then
  replay_args=(--replay-from "$REPLAY_FROM")
fi

codellama-compress prune mask-mlp \
  --model-dir "$RUN_DIR/distilled" \
  "${replay_args[@]}" \
  "${args[@]}"

codellama-compress finetune run \
  --model-dir "$RUN_DIR/pruned" \
  --replay-from "$RUN_DIR" \
  "${args[@]}"

echo "Done. Run directory: $RUN_DIR"
