from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# Run directories are created under output/runs/<run_id>.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Reject shell/metacharacters in values embedded into generated scripts.
_UNSAFE_SHELL_CHARS = frozenset("\n\r\x00$`\"'\\;|&<>(){}[]*?!#~")

_MAX_CONFIG_BYTES = 1_000_000


def assert_safe_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "Unsafe run_id: use 1-128 chars of letters, digits, '.', '_', or '-' "
            "(must start with alphanumeric)."
        )
    return run_id


def assert_safe_single_line(value: str, *, field: str) -> str:
    if any(c in value for c in ("\n", "\r", "\x00")):
        raise ValueError(f"Unsafe {field}: contains control characters")
    return value


def assert_safe_shell_token(value: str, *, field: str) -> str:
    assert_safe_single_line(value, field=field)
    if any(c in value for c in _UNSAFE_SHELL_CHARS):
        raise ValueError(f"Unsafe {field}: contains shell metacharacters")
    return value


def assert_safe_modelfile_name(value: str) -> str:
    assert_safe_single_line(value, field="model_name")
    if '"' in value or "\\" in value:
        raise ValueError("Unsafe model_name: quotes or backslashes are not allowed")
    return value


def clamp_tcp_port(port: int) -> int:
    if not (1 <= int(port) <= 65535):
        raise ValueError(f"Unsafe port: {port} (expected 1-65535)")
    return int(port)


def resolve_user_path(path: Path, *, must_exist: bool = False) -> Path:
    """
    Resolve a user-supplied filesystem path and reject obvious traversal patterns.
    """
    if ".." in path.parts:
        raise ValueError(f"Unsafe path (contains '..'): {path}")
    resolved = path.expanduser().resolve(strict=False)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def resolve_path_under_base(path: Path, *, base: Path, must_exist: bool = False) -> Path:
    resolved = resolve_user_path(path, must_exist=must_exist)
    base_r = base.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(base_r)
    except ValueError as e:
        raise ValueError(f"Path must be under {base_r}: {resolved}") from e
    return resolved


def load_bounded_json_config(path: Path, *, max_bytes: int = _MAX_CONFIG_BYTES) -> dict[str, Any]:
    p = resolve_user_path(path, must_exist=True)
    if not p.is_file():
        raise ValueError(f"Config path is not a file: {p}")
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Config file too large ({size} bytes > {max_bytes})")
    if p.suffix.lower() != ".json":
        raise ValueError(f"Unsupported config file type: {p.suffix}. Use JSON.")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def hf_pretrained_model_args(model_dir: Path) -> str:
    """
    Build lm-eval-harness model_args for a local HF directory.
    """
    s = str(resolve_user_path(model_dir, must_exist=True))
    if "," in s or "=" in s:
        raise ValueError("model_dir must not contain ',' or '=' for lm-eval model_args")
    return f"pretrained={s},trust_remote_code=False"


def merge_dataclass_fields(dc: Any, updates: dict[str, Any]) -> Any:
    """Shallow merge allowing only known dataclass fields."""
    if not hasattr(dc, "__dataclass_fields__"):
        raise TypeError("merge_dataclass_fields expects a dataclass instance")
    allowed = set(dc.__dataclass_fields__)
    base = dict(dc.__dict__)
    for key, val in (updates or {}).items():
        if key in allowed:
            base[key] = val
    return dc.__class__(**base)
