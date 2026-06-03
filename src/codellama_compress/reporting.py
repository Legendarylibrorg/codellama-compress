from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import save_json, to_jsonable


def now_unix() -> float:
    return time.time()


@contextmanager
def jsonl_writer(path: Path) -> Iterator[callable]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a", encoding="utf-8")
    try:

        def write(obj: Any) -> None:
            f.write(json.dumps(obj, default=to_jsonable) + "\n")
            f.flush()

        yield write
    finally:
        f.close()


def write_provenance(run_dir: Path, *, extra: dict[str, Any] | None = None) -> None:
    try:
        import torch
    except Exception:  # pragma: no cover - depends on optional runtime deps
        torch = None  # type: ignore[assignment]

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    prov: dict[str, Any] = {
        "time_unix": now_unix(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": cuda_available,
        "cuda_device_name": (torch.cuda.get_device_name(0) if cuda_available else None),
        "pid": os.getpid(),
    }
    if extra:
        prov.update(extra)
    save_json(run_dir / "provenance.json", prov)


def write_metrics(run_dir: Path, *, stage: str, metrics: dict[str, Any]) -> None:
    """
    Write stage metrics into a single metrics.json file (merged by stage).
    """
    path = run_dir / "metrics.json"
    cur: dict[str, Any] = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text())
        except Exception:
            cur = {}
    cur[stage] = metrics
    save_json(path, cur)


def dataclass_dict(dc: Any) -> dict[str, Any]:
    return asdict(dc) if hasattr(dc, "__dataclass_fields__") else dict(dc)


def dataset_provenance(dataset_cfg: Any) -> dict[str, Any]:
    """
    Best-effort dataset identity without forcing downloads.
    """
    if not hasattr(dataset_cfg, "name"):
        return {}

    out: dict[str, Any] = {"dataset_cfg": dataclass_dict(dataset_cfg)}
    revision = getattr(dataset_cfg, "revision", None)
    if revision:
        out["dataset_revision"] = revision
    try:
        from datasets import load_dataset_builder  # type: ignore

        builder_kw: dict[str, Any] = {}
        if revision:
            builder_kw["revision"] = revision
        b = load_dataset_builder(dataset_cfg.name, dataset_cfg.config, **builder_kw)
        info = getattr(b, "info", None)
        out["builder"] = {
            "dataset_name": getattr(info, "dataset_name", None) if info else None,
            "config_name": getattr(b, "config_name", None),
            "version": str(getattr(info, "version", None)) if info else None,
        }
    except Exception:
        pass
    return out


def write_samples_jsonl(
    *,
    run_dir: Path,
    stage: str,
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
) -> None:
    import torch

    path = run_dir / "logs" / "samples.jsonl"
    model.eval()
    with jsonl_writer(path) as write:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            write(
                {
                    "stage": stage,
                    "prompt": prompt,
                    "text": text,
                    "max_new_tokens": max_new_tokens,
                }
            )
