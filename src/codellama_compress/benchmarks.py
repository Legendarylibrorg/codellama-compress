from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reporting import jsonl_writer


def _parse_tasks(tasks: list[str] | str) -> list[str]:
    if isinstance(tasks, str):
        tasks = [tasks]
    out: list[str] = []
    for t in tasks:
        for part in str(t).split(","):
            p = part.strip()
            if p:
                out.append(p)
    # preserve order, de-dupe
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def run_benchmarks(
    *,
    model_dir: Path,
    tasks: list[str] | str,
    out_dir: Path,
    seed: int = 42,
    limit: int | None = None,
    save_per_sample: bool = True,
) -> dict[str, Any]:
    """
    Run a benchmark harness (EleutherAI lm-eval-harness) and write research-grade artifacts.

    Outputs:
    - bench_summary.json: aggregate results
    - bench_config.json: run configuration
    - bench_details.jsonl: per-sample (if available / enabled)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_list = _parse_tasks(tasks)
    if not tasks_list:
        raise ValueError("No tasks specified")

    try:
        from lm_eval import evaluator  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            'Benchmarking requires the eval extra. Install with: pip install ".[eval]"'
        ) from e

    # lm-eval-harness HF backend
    from .security import hf_pretrained_model_args

    model_args = hf_pretrained_model_args(model_dir)
    res = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks_list,
        num_fewshot=0,
        batch_size="auto",
        device="cuda" if _cuda_available() else "cpu",
        limit=limit,
        seed=seed,
        log_samples=bool(save_per_sample),
    )

    # Save config + summary (keep full harness output for reproducibility)
    (out_dir / "bench_config.json").write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "tasks": tasks_list,
                "seed": seed,
                "limit": limit,
                "save_per_sample": save_per_sample,
                "backend": "lm-eval[hf]",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "bench_summary.json").write_text(
        json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if save_per_sample:
        samples = res.get("samples") or {}
        with jsonl_writer(out_dir / "bench_details.jsonl") as write:
            for task_name, rows in samples.items():
                for row in rows or []:
                    write({"task": task_name, "row": row})

    return res


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
