from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_CODE_BYTES = 256_000
MAX_CAPTURE_CHARS = 64_000

# Prepended to model-generated programs. Not a substitute for OS-level isolation.
_SANDBOX_GUARD = """\
from __future__ import annotations
import builtins as _builtins

_ALLOWED_IMPORT_ROOTS = frozenset({
    "builtins", "math", "itertools", "functools", "collections", "typing",
    "re", "random", "string", "heapq", "bisect", "array", "statistics",
    "decimal", "fractions", "copy", "operator", "enum", "dataclasses",
})

_BLOCKED_BUILTINS = frozenset({
    "open", "eval", "exec", "compile", "input", "breakpoint",
    "__import__", "help", "exit", "quit", "license", "copyright", "credits",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
    "memoryview", "bytearray", "open_code", "__loader__", "__spec__",
    "__package__", "__builtins__", "__name__", "__doc__", "__file__",
})

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is blocked in the code-eval sandbox")
    return _builtins.__import__(name, globals, locals, fromlist, level)

def _blocked_builtin(name, *args, **kwargs):
    raise RuntimeError(f"builtin {name!r} is blocked in the code-eval sandbox")

_builtins.__import__ = _restricted_import
_bdict = _builtins.__dict__
for _blocked_name in _BLOCKED_BUILTINS:
    if _blocked_name in _bdict:
        _bdict[_blocked_name] = lambda *a, _n=_blocked_name, **k: _blocked_builtin(_n)
"""


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

        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        mem = 1_000_000_000
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
        if hasattr(resource, "RLIMIT_CORE"):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        if hasattr(resource, "RLIMIT_NOFILE"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass


def _truncate(text: str, *, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _minimal_env() -> dict[str, str]:
    py = Path(sys.executable).resolve()
    bindir = str(py.parent)
    return {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONSAFEPATH": "1",
        "PATH": bindir,
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def run_python_sandboxed(*, code: str, timeout_s: float = 3.0) -> ExecResult:
    """
    Execute Python in an isolated subprocess with resource limits.

    Limitations: blocks common imports and dangerous builtins but is not a full
    sandbox. Do not run untrusted code on sensitive hosts; use containers/VMs.
    """
    if os.name != "posix":
        return ExecResult(
            ok=False,
            exit_code=1,
            stdout="",
            stderr="Sandboxed execution is only supported on POSIX.",
            reason="unsupported_platform",
        )

    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return ExecResult(
            ok=False,
            exit_code=1,
            stdout="",
            stderr=f"code exceeds {MAX_CODE_BYTES} bytes",
            reason="internal_error",
        )

    try:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            script = td_p / "main.py"
            script.write_text(_SANDBOX_GUARD + "\n" + code, encoding="utf-8")

            cmd = [sys.executable, "-I", "-S", str(script)]
            try:
                r = subprocess.run(
                    cmd,
                    cwd=str(td_p),
                    env=_minimal_env(),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=_limit_resources,
                    close_fds=True,
                    check=False,
                )
                ok = r.returncode == 0
                return ExecResult(
                    ok=ok,
                    exit_code=int(r.returncode),
                    stdout=_truncate(r.stdout or ""),
                    stderr=_truncate(r.stderr or ""),
                    reason="ok" if ok else "runtime_error",
                )
            except subprocess.TimeoutExpired as e:
                return ExecResult(
                    ok=False,
                    exit_code=124,
                    stdout=_truncate((e.stdout or "") if isinstance(e.stdout, str) else ""),
                    stderr=_truncate((e.stderr or "") if isinstance(e.stderr, str) else ""),
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
