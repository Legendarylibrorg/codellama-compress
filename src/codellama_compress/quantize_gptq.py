from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import DatasetConfig, GPTQConfig, save_json
from .data import sample_calibration_texts
from .replay import hash_calibration_texts
from .reporting import write_metrics, write_provenance
from .training_utils import ensure_pad_token


def _build_calibration_data(tokenizer, dataset_cfg: DatasetConfig, cfg: GPTQConfig):
    data = []
    texts_used = sample_calibration_texts(
        dataset_cfg,
        samples=cfg.calibration_samples,
        seq_len=cfg.calibration_seq_len,
        seed=cfg.seed,
    )
    for text in tqdm(texts_used, desc="Calibration samples"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.calibration_seq_len,
            padding="max_length",
        )
        data.append(enc["input_ids"])
    return torch.cat(data, dim=0), texts_used


def run_gptq_quantization(
    *,
    run_dir: Path,
    in_model_dir: Path,
    out_dir: Path,
    dataset_cfg: DatasetConfig,
    cfg: GPTQConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(run_dir, extra={"stage": "quantize_gptq"})

    # Imported lazily because auto-gptq is optional/heavy.
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            'auto-gptq is not installed. Install with: pip install ".[quant]"'
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(in_model_dir, use_fast=True, trust_remote_code=False)
    ensure_pad_token(tokenizer)

    quantize_config = BaseQuantizeConfig(
        bits=cfg.bits,
        group_size=cfg.group_size,
        desc_act=cfg.desc_act,
        damp_percent=cfg.damp_percent,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        in_model_dir,
        quantize_config=quantize_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    calib_ids, calib_texts = _build_calibration_data(tokenizer, dataset_cfg, cfg)
    calib_fingerprint = hash_calibration_texts(calib_texts)
    examples = [{"input_ids": calib_ids[i : i + 1]} for i in range(calib_ids.shape[0])]

    model.quantize(examples, use_triton=False)
    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)

    save_json(
        out_dir / "gptq_info.json",
        {
            "input_model": str(in_model_dir),
            "output_model": str(out_dir),
            "gptq": cfg,
            "dataset": dataset_cfg,
            "calibration_fingerprint": calib_fingerprint,
            "calibration_samples": len(calib_texts),
        },
    )
    write_metrics(
        run_dir,
        stage="quantize_gptq",
        metrics={"output_dir": str(out_dir), "bits": cfg.bits, "group_size": cfg.group_size},
    )
