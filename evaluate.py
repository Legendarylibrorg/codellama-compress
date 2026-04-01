#!/usr/bin/env python3
"""
Legacy entrypoint (kept for backward compatibility).

New code lives in the package. Prefer:

  codellama-compress evaluate run --model-dir <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from codellama_compress.evaluate import evaluate_model_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a model directory (legacy shim).")
    p.add_argument("--model", required=True, help="Path to HF model directory.")
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else None
    res = evaluate_model_dir(Path(args.model), out_path=out_path)
    print(res)


if __name__ == "__main__":
    main()
