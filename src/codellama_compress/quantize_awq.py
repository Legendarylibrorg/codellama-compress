from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from .config import DatasetConfig, GPTQConfig, save_json
from .data import sample_calibration_texts
from .replay import hash_calibration_texts
from .reporting import write_metrics, write_provenance
from .training_utils import ensure_pad_token


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
    ensure_pad_token(tok)

    model = AutoAWQForCausalLM.from_pretrained(in_model_dir)
    quant_config = {
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
        "version": "GEMM",
    }
    calib = sample_calibration_texts(dataset_cfg, samples=cfg.calibration_samples, seed=cfg.seed)
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
