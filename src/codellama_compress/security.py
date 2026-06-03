from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import fields
from pathlib import Path
from typing import Any

# Run directories are created under output/runs/<run_id>.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HF_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# Reject shell/metacharacters in values embedded into generated scripts.
_UNSAFE_SHELL_CHARS = frozenset("\n\r\x00$`\"'\\;|&<>(){}[]*?!#~")

_MAX_CONFIG_BYTES = 1_000_000

TRUST_REMOTE_CODE_ENV = "CODELLAMA_COMPRESS_TRUST_REMOTE_CODE"
ALLOW_CODE_EXEC_ENV = "CODELLAMA_COMPRESS_ALLOW_CODE_EXEC"
IN_CONTAINER_ENV = "CODELLAMA_COMPRESS_IN_CONTAINER"
DATASET_ALLOWLIST_EXTRA_ENV = "CODELLAMA_COMPRESS_DATASET_ALLOWLIST_EXTRA"

# Default Hugging Face dataset ids permitted for training/calibration.
_DEFAULT_ALLOWED_DATASETS = frozenset(
    {
        "bigcode/starcoderdata",
    }
)


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


def _walk_path_no_symlinks(path: Path) -> Path:
    """Expand a path without following symlinks in intermediate components."""
    expanded = path.expanduser()
    if expanded.is_absolute():
        current = Path(expanded.anchor)
        parts = expanded.parts[1:]
    else:
        current = Path.cwd()
        parts = expanded.parts
    for part in parts:
        if part in (".", ""):
            continue
        if part == "..":
            raise ValueError(f"Unsafe path (contains '..'): {path}")
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Unsafe path (symlinks not allowed): {current}")
    return current


def resolve_user_path(path: Path, *, must_exist: bool = False) -> Path:
    """
    Resolve a user-supplied filesystem path and reject traversal and symlinks.
    """
    if ".." in path.parts:
        raise ValueError(f"Unsafe path (contains '..'): {path}")
    resolved = _walk_path_no_symlinks(path)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def resolve_path_under_base(path: Path, *, base: Path, must_exist: bool = False) -> Path:
    resolved = resolve_user_path(path, must_exist=must_exist)
    base_r = resolve_user_path(base, must_exist=False)
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
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {p}") from e
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
    if not isinstance(updates, dict):
        raise ValueError("Dataclass updates must be a JSON object")
    allowed = {f.name for f in fields(dc)}
    base = {f.name: getattr(dc, f.name) for f in fields(dc)}
    for key, val in (updates or {}).items():
        if key in allowed:
            base[key] = val
    return dc.__class__(**base)


def _assert_safe_hf_dataset_id(dataset_id: str) -> str:
    assert_safe_single_line(dataset_id, field="dataset_id")
    if not _HF_DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError(
            f"Unsafe dataset id: {dataset_id!r}. Expected an explicit namespace/name Hub id."
        )
    return dataset_id


def allowed_dataset_ids() -> frozenset[str]:
    extra = os.environ.get(DATASET_ALLOWLIST_EXTRA_ENV, "")
    extras = {_assert_safe_hf_dataset_id(x.strip()) for x in extra.split(",") if x.strip()}
    return _DEFAULT_ALLOWED_DATASETS | extras


def assert_allowed_dataset(dataset_name: str) -> None:
    if dataset_name not in allowed_dataset_ids():
        raise ValueError(
            f"Dataset {dataset_name!r} is not in the allowlist. "
            f"Allowed: {sorted(allowed_dataset_ids())}. "
            f"To add datasets for this process, set {DATASET_ALLOWLIST_EXTRA_ENV} "
            "(comma-separated Hub ids)."
        )


def normalize_training_text(text: str) -> str:
    """Unicode-normalize training samples to reduce hidden-character poisoning."""
    return unicodedata.normalize("NFKC", text)


def resolve_trust_remote_code(config_flag: bool) -> bool:
    """
    trust_remote_code is disabled unless explicitly enabled via environment variable.

    Config JSON may request it, but that alone is insufficient (break-glass).
    """
    env_on = os.environ.get(TRUST_REMOTE_CODE_ENV, "").strip() in ("1", "true", "yes")
    if config_flag and not env_on:
        return False
    return bool(config_flag and env_on)


def trust_remote_code_audit_record(*, config_flag: bool, effective: bool) -> dict[str, Any]:
    return {
        "trust_remote_code_config": config_flag,
        "trust_remote_code_effective": effective,
        "trust_remote_code_env": TRUST_REMOTE_CODE_ENV,
    }


def is_isolated_execution_environment() -> bool:
    if Path("/.dockerenv").exists():
        return True
    if os.environ.get(IN_CONTAINER_ENV, "").strip() in ("1", "true", "yes"):
        return True
    if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
        return True
    return False


def assert_code_exec_permitted(*, allow_insecure: bool = False) -> None:
    """
    Gate execution of model-generated Python on the host.

    Permitted when:
    - Running in a container/CI-isolated environment, or
    - Host ack: --allow-insecure-code-exec AND CODELLAMA_COMPRESS_ALLOW_CODE_EXEC=1
    """
    if is_isolated_execution_environment():
        return
    env_ack = os.environ.get(ALLOW_CODE_EXEC_ENV, "").strip() in ("1", "true", "yes")
    if allow_insecure and env_ack:
        return
    raise RuntimeError(
        "Refusing to execute model-generated code on this host. "
        "Run inside a container/CI, or pass --allow-insecure-code-exec and set "
        f"{ALLOW_CODE_EXEC_ENV}=1 after reviewing SECURITY.md."
    )


def dataset_load_extra_kwargs(dataset_cfg: Any) -> dict[str, Any]:
    """Extra kwargs for datasets.load_dataset after allowlist validation."""
    assert_allowed_dataset(dataset_cfg.name)
    assert_safe_single_line(str(dataset_cfg.config), field="dataset.config")
    assert_safe_single_line(str(dataset_cfg.split), field="dataset.split")
    kw: dict[str, Any] = {
        "split": dataset_cfg.split,
        "streaming": dataset_cfg.streaming,
    }
    revision = getattr(dataset_cfg, "revision", None)
    if revision:
        kw["revision"] = revision
    return kw
