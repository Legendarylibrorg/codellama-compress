from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import DatasetConfig, GPTQConfig, save_json
from .replay import hash_calibration_texts
from .reporting import write_metrics, write_provenance
from .security import dataset_load_extra_kwargs, normalize_training_text


def _sample_texts(dataset_cfg: DatasetConfig, n: int) -> list[str]:
    ds = load_dataset(
        dataset_cfg.name,
        dataset_cfg.config,
        **{**dataset_load_extra_kwargs(dataset_cfg), "streaming": True},
    ).shuffle(buffer_size=dataset_cfg.shuffle_buffer, seed=dataset_cfg.seed)
    out: list[str] = []
    for row in ds:
        txt = row.get("content") or row.get("text") or ""
        if isinstance(txt, str) and txt.strip():
            out.append(normalize_training_text(txt))
            if len(out) >= n:
                break
    return out


def run_awq_quantization(
    *,
    run_dir: Path,
    in_model_dir: Path,
    out_dir: Path,
    dataset_cfg: DatasetConfig,
    cfg: GPTQConfig,
) -> None:
    """
    AWQ quantization using autoawq.

    We reuse `GPTQConfig` for calibration knobs (samples/seq_len).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(run_dir, extra={"stage": "quantize_awq"})
    try:
        from awq import AutoAWQForCausalLM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError('autoawq is not installed. Install with: pip install ".[quant]"') from e

    tok = AutoTokenizer.from_pretrained(in_model_dir, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoAWQForCausalLM.from_pretrained(in_model_dir)
    quant_config = {
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
        "version": "GEMM",
    }
    calib = _sample_texts(dataset_cfg, cfg.calibration_samples)
    calib_fingerprint = hash_calibration_texts(calib)
    model.quantize(tok, quant_config=quant_config, calib_data=calib)
    model.save_quantized(out_dir)
    tok.save_pretrained(out_dir)

    save_json(
        out_dir / "awq_info.json",
        {
            "input_model": str(in_model_dir),
            "output_model": str(out_dir),
            "awq": {"calibration_samples": cfg.calibration_samples, "seed": cfg.seed},
            "dataset": dataset_cfg,
            "calibration_fingerprint": calib_fingerprint,
        },
    )
    write_metrics(
        run_dir,
        stage="quantize_awq",
        metrics={"output_dir": str(out_dir), "calibration_samples": cfg.calibration_samples},
    )
