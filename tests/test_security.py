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
    ALLOW_CODE_EXEC_ENV,
    TRUST_REMOTE_CODE_ENV,
    assert_allowed_dataset,
    assert_code_exec_permitted,
    assert_safe_run_id,
    load_bounded_json_config,
    merge_dataclass_fields,
    normalize_training_text,
    resolve_trust_remote_code,
    resolve_user_path,
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


def test_resolve_trust_remote_code_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUST_REMOTE_CODE_ENV, raising=False)
    assert resolve_trust_remote_code(True) is False
    assert resolve_trust_remote_code(False) is False
    monkeypatch.setenv(TRUST_REMOTE_CODE_ENV, "1")
    assert resolve_trust_remote_code(True) is True
    assert resolve_trust_remote_code(False) is False


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


def test_resolve_user_path_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "secret"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        resolve_user_path(link, must_exist=True)


def test_assert_allowed_dataset_blocks_unknown() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        assert_allowed_dataset("evil/unknown-dataset")


def test_normalize_training_text_nfkc() -> None:
    assert normalize_training_text("café") == normalize_training_text("café")


def test_assert_code_exec_permitted_requires_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("CODELLAMA_COMPRESS_IN_CONTAINER", raising=False)
    monkeypatch.delenv(ALLOW_CODE_EXEC_ENV, raising=False)
    with pytest.raises(RuntimeError, match="Refusing to execute"):
        assert_code_exec_permitted(allow_insecure=False)
    with pytest.raises(RuntimeError, match="Refusing to execute"):
        assert_code_exec_permitted(allow_insecure=True)
    monkeypatch.setenv(ALLOW_CODE_EXEC_ENV, "1")
    assert_code_exec_permitted(allow_insecure=True)


def test_run_python_sandboxed_blocks_os_import() -> None:
    r = run_python_sandboxed(code="import os\n")
    if os.name != "posix":
        assert r.reason == "unsupported_platform"
    else:
        assert r.ok is False
        assert "blocked" in (r.stderr or "").lower() or "ImportError" in (r.stderr or "")


def test_run_python_sandboxed_blocks_open() -> None:
    r = run_python_sandboxed(code="open('/etc/passwd')\n")
    if os.name != "posix":
        assert r.reason == "unsupported_platform"
    else:
        assert r.ok is False
        err = r.stderr or ""
        assert "blocked" in err.lower() or "RuntimeError" in err
