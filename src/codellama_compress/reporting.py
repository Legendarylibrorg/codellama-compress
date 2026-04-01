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

import torch

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
    prov: dict[str, Any] = {
        "time_unix": now_unix(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
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
