from pathlib import Path

from codellama_compress.export import write_export_bundle


def test_export_bundle_writes_files(tmp_path: Path):
    out_dir = tmp_path / "export"
    write_export_bundle(model_dir=tmp_path / "model", out_dir=out_dir, model_name="x", port=8000)
    assert (out_dir / "vllm_server.sh").exists()
    assert (out_dir / "Dockerfile").exists()
    assert (out_dir / "convert_gguf.sh").exists()
    assert (out_dir / "Modelfile").exists()
    assert (out_dir / "README.md").exists()
    assert (out_dir / "speculative_decoding.py").exists()
    assert "target_model='" in (out_dir / "speculative_decoding.py").read_text()


def test_export_bundle_rejects_control_chars(tmp_path: Path):
    out_dir = tmp_path / "export"
    bad = Path("bad\npath")
    try:
        write_export_bundle(model_dir=bad, out_dir=out_dir, model_name="x", port=8000)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unsafe model_dir")
