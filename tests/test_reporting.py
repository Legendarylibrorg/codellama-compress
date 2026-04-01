from __future__ import annotations

import json
from pathlib import Path

from codellama_compress.reporting import jsonl_writer, write_metrics, write_provenance


def test_jsonl_writer_appends(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    with jsonl_writer(p) as write:
        write({"a": 1})
        write({"b": 2})
    lines = p.read_text().splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


def test_write_metrics_merges_by_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir, stage="a", metrics={"x": 1})
    write_metrics(run_dir, stage="b", metrics={"y": 2})
    write_metrics(run_dir, stage="a", metrics={"x": 3})
    d = json.loads((run_dir / "metrics.json").read_text())
    assert d["a"]["x"] == 3
    assert d["b"]["y"] == 2


def test_write_provenance_writes_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_provenance(run_dir, extra={"k": "v"})
    d = json.loads((run_dir / "provenance.json").read_text())
    assert d["k"] == "v"
