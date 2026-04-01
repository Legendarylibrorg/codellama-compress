from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import save_json


def new_run_dir(root: Path = Path("output/runs"), run_id: str | None = None) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = ts
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_cmd(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
        return out.strip() + "\n"
    except Exception as e:  # pragma: no cover
        return f"ERROR running {cmd!r}: {e}\n"


def write_env_report(run_dir: Path) -> None:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    save_json(run_dir / "env.json", env)

    write_text(run_dir / "pip_freeze.txt", _run_cmd([sys.executable, "-m", "pip", "freeze"]))

    if sys.platform.startswith("linux"):
        write_text(run_dir / "nvidia-smi.txt", _run_cmd(["nvidia-smi"]))

    save_json(run_dir / "git_state.json", _git_state())


def _git_state() -> dict[str, Any]:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return {"sha": sha, "dirty": None}
    sha_out = _run_cmd(["git", "rev-parse", "HEAD"]).strip()
    dirty_out = _run_cmd(["git", "status", "--porcelain"]).strip()
    return {"sha": sha_out or None, "dirty": bool(dirty_out)}


def save_effective_config(run_dir: Path, cfg: Any) -> None:
    save_json(run_dir / "config.json", cfg)


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except Exception:
        # Fall back silently; symlinks can be restricted on some filesystems.
        pass
