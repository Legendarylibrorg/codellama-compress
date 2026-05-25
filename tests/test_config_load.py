import json
from pathlib import Path

from codellama_compress.config import load_config_file


def test_load_config_file_json(tmp_path: Path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"dataset": {"name": "x"}}))
    d = load_config_file(p)
    assert d["dataset"]["name"] == "x"


def test_merge_dataclass_shallow():
    from codellama_compress.config import DatasetConfig, merge_dataclass

    base = DatasetConfig()
    merged = merge_dataclass(base, {"name": "y", "streaming": False})
    assert merged.name == "y"
    assert merged.streaming is False


def test_merge_dataclass_ignores_unknown_keys():
    from codellama_compress.config import DatasetConfig, merge_dataclass

    base = DatasetConfig()
    merged = merge_dataclass(base, {"name": "z", "not_a_field": 1})
    assert merged.name == "z"
    assert merged.config == base.config
