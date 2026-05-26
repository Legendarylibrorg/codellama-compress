from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import save_json, to_jsonable

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
ARTIFACTS_NAME = "artifacts.jsonl"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=to_jsonable,
    )


def content_fingerprint(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_directory(
    path: Path,
    *,
    include_globs: tuple[str, ...] = ("**/*",),
    exclude_names: frozenset[str] = frozenset({".git", "__pycache__"}),
) -> str:
    """
    Content-address a directory: sorted relative path + per-file sha256 lines.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    lines: list[str] = []
    files: list[Path] = []
    for pattern in include_globs:
        files.extend(path.glob(pattern))
    seen: set[Path] = set()
    for f in sorted({p.resolve() for p in files if p.is_file() and not p.is_symlink()}):
        if f in seen:
            continue
        seen.add(f)
        rel_parts = f.relative_to(path.resolve()).parts
        if any(part in exclude_names for part in rel_parts):
            continue
        rel = f.relative_to(path.resolve()).as_posix()
        lines.append(f"{rel}\0{sha256_file(f)}")
    return sha256_text("\n".join(lines))


def derive_run_id(config_fingerprint: str, *, prefix: str = "r") -> str:
    """Short, filesystem-safe run id from a config fingerprint."""
    short = config_fingerprint[:16]
    return f"{prefix}{short}"


def apply_global_seeds(seed: int, *, strict: bool = False) -> None:
    """Set Python, PyTorch, and optional NumPy RNG seeds for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_NAME


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = _manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    save_json(_manifest_path(run_dir), manifest)


def init_manifest(
    run_dir: Path,
    *,
    config_fingerprint: str,
    determinism: Any,
    effective_config: dict[str, Any],
    pipeline_fingerprint: dict[str, Any] | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "package_version": __version__,
        "config_fingerprint": config_fingerprint,
        "pipeline_fingerprint": to_jsonable(pipeline_fingerprint or {}),
        "determinism": (
            asdict(determinism)
            if hasattr(determinism, "__dataclass_fields__")
            else dict(determinism)
        ),
        "effective_config": to_jsonable(effective_config),
        "stages": [],
    }
    if stage:
        manifest["stages"].append({"name": stage, "status": "started"})
    write_manifest(run_dir, manifest)
    return manifest


def append_artifact_record(
    run_dir: Path,
    *,
    stage: str,
    artifact_path: Path,
    role: str = "output",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a stage artifact line to artifacts.jsonl and update manifest stages."""
    try:
        rel_path = str(artifact_path.relative_to(run_dir))
    except ValueError:
        rel_path = str(artifact_path)

    if artifact_path.is_dir():
        digest = hash_directory(artifact_path)
    elif artifact_path.is_file():
        digest = sha256_file(artifact_path)
    else:
        raise FileNotFoundError(artifact_path)

    record: dict[str, Any] = {
        "stage": stage,
        "role": role,
        "path": rel_path,
        "sha256": digest,
    }
    if extra:
        record.update(extra)

    artifacts_path = run_dir / ARTIFACTS_NAME
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    with artifacts_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=to_jsonable) + "\n")

    manifest = load_manifest(run_dir) if _manifest_path(run_dir).exists() else None
    if manifest is not None:
        stages: list[dict[str, Any]] = list(manifest.get("stages") or [])
        stages = [s for s in stages if s.get("name") != stage or s.get("status") != "started"]
        stages.append(
            {
                "name": stage,
                "role": role,
                "path": rel_path,
                "sha256": digest,
                **(extra or {}),
            }
        )
        manifest["stages"] = stages
        write_manifest(run_dir, manifest)

    return record


def artifact_sha256_for_stage(run_dir: Path, stage: str, *, role: str = "output") -> str | None:
    path = run_dir / ARTIFACTS_NAME
    if not path.exists():
        manifest = load_manifest(run_dir)
        for s in manifest.get("stages") or []:
            if s.get("name") == stage and s.get("role", "output") == role:
                return s.get("sha256")
        return None
    last: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("stage") == stage and rec.get("role", "output") == role:
            last = rec.get("sha256")
    return last


def verify_artifact_sha256(path: Path, expected_sha256: str) -> bool:
    if path.is_dir():
        actual = hash_directory(path)
    else:
        actual = sha256_file(path)
    return actual == expected_sha256


def assert_replay_inputs(
    replay_run_dir: Path,
    *,
    stage: str,
    input_path: Path,
    expected_role: str = "output",
    expected_from_stage: str | None = None,
) -> None:
    """
    Verify an input artifact matches a prior stage recorded in replay_run_dir.
    """
    src_stage = expected_from_stage or _infer_prior_stage(stage)
    expected = artifact_sha256_for_stage(replay_run_dir, src_stage, role=expected_role)
    if expected is None:
        raise RuntimeError(
            f"Replay manifest at {replay_run_dir} has no sha256 for stage {src_stage!r}"
        )
    if not verify_artifact_sha256(input_path, expected):
        raise RuntimeError(
            f"Replay hash mismatch for {input_path}: expected sha256 {expected} "
            f"from stage {src_stage!r} in {replay_run_dir}"
        )


def _infer_prior_stage(stage: str) -> str:
    chain = {
        "prune": "distill",
        "finetune": "prune",
        "quantize_gptq": "finetune",
        "quantize_awq": "finetune",
    }
    if stage not in chain:
        raise ValueError(f"No default prior stage for replay of {stage!r}")
    return chain[stage]


def verify_manifest(run_dir: Path, *, strict_config: bool = True) -> dict[str, Any]:
    """
    Recompute fingerprints and artifact hashes; return a verification report.
    """
    manifest = load_manifest(run_dir)
    report: dict[str, Any] = {"ok": True, "errors": [], "artifacts": []}

    stored_fp = manifest.get("config_fingerprint")
    pipeline = manifest.get("pipeline_fingerprint")
    effective = manifest.get("effective_config")
    if strict_config:
        if pipeline:
            recomputed = content_fingerprint(pipeline)
        elif effective is not None:
            recomputed = content_fingerprint(effective)
        else:
            recomputed = None
        if recomputed is not None and stored_fp != recomputed:
            report["ok"] = False
            report["errors"].append(
                f"config_fingerprint mismatch: stored={stored_fp!r} recomputed={recomputed!r}"
            )

    artifacts_path = run_dir / ARTIFACTS_NAME
    records: list[dict[str, Any]] = []
    if artifacts_path.exists():
        for line in artifacts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    else:
        records = list(manifest.get("stages") or [])

    for rec in records:
        rel = rec.get("path")
        expected = rec.get("sha256")
        if not rel or not expected:
            continue
        artifact = run_dir / rel
        entry = {"path": rel, "expected": expected, "ok": False}
        if not artifact.exists():
            entry["error"] = "missing"
            report["ok"] = False
            report["errors"].append(f"Missing artifact: {artifact}")
        elif verify_artifact_sha256(artifact, expected):
            entry["ok"] = True
        else:
            entry["error"] = "hash_mismatch"
            report["ok"] = False
            report["errors"].append(f"Hash mismatch: {artifact}")
        report["artifacts"].append(entry)

    return report


def hash_calibration_texts(texts: list[str]) -> str:
    """Fingerprint ordered calibration snippets for GPTQ/AWQ replay checks."""
    lines = [sha256_text(t) for t in texts]
    return sha256_text("\n".join(lines))


def backfill_manifest_from_run_dir(run_dir: Path) -> dict[str, Any]:
    """
    Create or refresh manifest.json from config.json and known artifact dirs.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {run_dir}")
    effective_config = json.loads(config_path.read_text(encoding="utf-8"))
    fp = content_fingerprint(effective_config)
    det = effective_config.get("determinism") or {
        "seed": 42,
        "deterministic": True,
        "hash_run_id": True,
    }
    manifest = init_manifest(
        run_dir,
        config_fingerprint=fp,
        determinism=det,
        effective_config=effective_config,
    )

    stage_dirs = [
        ("distill", "distilled"),
        ("prune", "pruned"),
        ("finetune", "finetuned"),
        ("quantize_gptq", "quantized-gptq"),
        ("quantize_awq", "quantized-awq"),
    ]
    for stage, dirname in stage_dirs:
        p = run_dir / dirname
        if p.is_dir():
            append_artifact_record(run_dir, stage=stage, artifact_path=p, role="output")
    return manifest
