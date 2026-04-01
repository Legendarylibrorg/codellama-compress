from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    reason: Literal["ok", "timeout", "runtime_error", "internal_error", "unsupported_platform"]


def _limit_resources() -> None:
    # Linux/Unix only; best-effort safety. This is NOT a perfect sandbox.
    try:
        import resource  # type: ignore

        # CPU seconds
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        # Address space (bytes) ~1GB
        mem = 1_000_000_000
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        # File size (bytes) ~10MB
        resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
        # Number of processes
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except Exception:
        pass


def run_python_sandboxed(*, code: str, timeout_s: float = 3.0) -> ExecResult:
    if os.name != "posix":
        return ExecResult(
            ok=False,
            exit_code=1,
            stdout="",
            stderr="Sandboxed execution is only supported on POSIX.",
            reason="unsupported_platform",
        )

    try:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            script = td_p / "main.py"
            script.write_text(code, encoding="utf-8")

            env = {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PATH": os.environ.get("PATH", ""),
            }
            cmd = [sys.executable, "-I", str(script)]
            try:
                r = subprocess.run(
                    cmd,
                    cwd=str(td_p),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=_limit_resources,
                    check=False,
                )
                ok = r.returncode == 0
                return ExecResult(
                    ok=ok,
                    exit_code=int(r.returncode),
                    stdout=r.stdout or "",
                    stderr=r.stderr or "",
                    reason="ok" if ok else "runtime_error",
                )
            except subprocess.TimeoutExpired as e:
                return ExecResult(
                    ok=False,
                    exit_code=124,
                    stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                    stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                    reason="timeout",
                )
    except Exception as e:  # pragma: no cover
        return ExecResult(
            ok=False,
            exit_code=1,
            stdout="",
            stderr=str(e),
            reason="internal_error",
        )
