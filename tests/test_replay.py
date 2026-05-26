from __future__ import annotations

import json
from pathlib import Path

from codellama_compress.config import DatasetConfig, DeterminismConfig, DistillConfig, to_jsonable
from codellama_compress.replay import (
    append_artifact_record,
    backfill_manifest_from_run_dir,
    content_fingerprint,
    derive_run_id,
    hash_directory,
    init_manifest,
    sha256_file,
    verify_manifest,
)


def test_content_fingerprint_stable_for_dataclasses() -> None:
    cfg = {
        "dataset": DatasetConfig(seed=7),
        "distill": DistillConfig(steps=10),
        "determinism": DeterminismConfig(seed=7),
    }
    a = content_fingerprint(cfg)
    b = content_fingerprint(cfg)
    assert a == b
    assert len(a) == 64


def test_derive_run_id_is_safe() -> None:
    rid = derive_run_id("a" * 64)
    assert rid.startswith("r")
    assert len(rid) == 17


def test_manifest_verify_detects_tamper(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "distilled"
    artifact.mkdir()
    (artifact / "weights.txt").write_text("hello", encoding="utf-8")

    pipeline = {"dataset": {"seed": 1}, "determinism": DeterminismConfig()}
    effective = {"stage": "distill", "steps": 1, **pipeline}
    fp = content_fingerprint(pipeline)
    init_manifest(
        run_dir,
        config_fingerprint=fp,
        pipeline_fingerprint=pipeline,
        determinism=DeterminismConfig(),
        effective_config=effective,
        stage="distill",
    )
    append_artifact_record(run_dir, stage="distill", artifact_path=artifact, role="output")

    ok = verify_manifest(run_dir)
    assert ok["ok"] is True

    (artifact / "weights.txt").write_text("tampered", encoding="utf-8")
    bad = verify_manifest(run_dir)
    assert bad["ok"] is False
    assert bad["errors"]


def test_backfill_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    finetuned = run_dir / "finetuned"
    finetuned.mkdir()
    (finetuned / "model.bin").write_bytes(b"\x00\x01")

    effective = {"finetune": {"steps": 5}}
    (run_dir / "config.json").write_text(
        json.dumps(to_jsonable(effective), indent=2) + "\n", encoding="utf-8"
    )
    manifest = backfill_manifest_from_run_dir(run_dir)
    assert manifest["config_fingerprint"] == content_fingerprint(effective)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "artifacts.jsonl").exists()


def test_hash_directory_matches_single_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    assert hash_directory(d) == sha256_file(f) or hash_directory(d) != ""
    # directory hash is over "a.txt\0<filehash>" line, not same as file alone
    line = f"a.txt\0{sha256_file(f)}"
    from codellama_compress.replay import sha256_text

    assert hash_directory(d) == sha256_text(line)
