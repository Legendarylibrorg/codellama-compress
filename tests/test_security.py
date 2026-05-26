from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from codellama_compress.code_exec import run_python_sandboxed
from codellama_compress.config import DistillConfig, load_config_file, merge_dataclass
from codellama_compress.export import write_export_bundle
from codellama_compress.io import new_run_dir
from codellama_compress.security import (
    assert_safe_run_id,
    load_bounded_json_config,
    merge_dataclass_fields,
)


def test_assert_safe_run_id_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        assert_safe_run_id("../etc")


def test_merge_dataclass_ignores_unknown_fields() -> None:
    dc = DistillConfig()
    merged = merge_dataclass(dc, {"steps": 5, "trust_remote_code": True, "__class__": "evil"})
    assert merged.steps == 5
    assert merged.trust_remote_code is True
    assert merged.teacher_model == dc.teacher_model


def test_merge_dataclass_fields_same_behavior() -> None:
    dc = DistillConfig()
    assert merge_dataclass_fields(dc, {"__init__": "nope"}).steps == dc.steps


def test_load_bounded_json_config_rejects_large_file(tmp_path: Path) -> None:
    p = tmp_path / "big.json"
    p.write_text('{"distill": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_bounded_json_config(p, max_bytes=4)


def test_load_config_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"distill": {"steps": 3}}), encoding="utf-8")
    assert load_config_file(p)["distill"]["steps"] == 3


def test_new_run_dir_rejects_unsafe_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        new_run_dir(tmp_path, run_id="../../etc")


def test_export_rejects_shell_metacharacters(tmp_path: Path) -> None:
    out_dir = tmp_path / "export"
    bad = tmp_path / "$(echo pwned)"
    with pytest.raises(ValueError):
        write_export_bundle(model_dir=bad, out_dir=out_dir, model_name="x", port=8000)


def test_run_python_sandboxed_blocks_os_import() -> None:
    r = run_python_sandboxed(code="import os\n")
    if os.name != "posix":
        assert r.reason == "unsupported_platform"
    else:
        assert r.ok is False
        assert "blocked" in (r.stderr or "").lower() or "ImportError" in (r.stderr or "")
