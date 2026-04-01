from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import DatasetConfig, GPTQConfig, save_json


def _iter_texts(dataset_cfg: DatasetConfig) -> Iterable[str]:
    ds = load_dataset(
        dataset_cfg.name,
        dataset_cfg.config,
        split=dataset_cfg.split,
        streaming=True,
    ).shuffle(buffer_size=dataset_cfg.shuffle_buffer, seed=dataset_cfg.seed)
    for row in ds:
        txt = row.get("content") or row.get("text") or ""
        if isinstance(txt, str) and txt.strip():
            yield txt


def _build_calibration_data(tokenizer, dataset_cfg: DatasetConfig, cfg: GPTQConfig):
    gen = _iter_texts(dataset_cfg)
    rng = np.random.default_rng(cfg.seed)
    data = []
    for _ in tqdm(range(cfg.calibration_samples), desc="Calibration samples"):
        text = next(gen)
        # truncate a random window for variety
        if len(text) > cfg.calibration_seq_len * 4:
            start = int(rng.integers(0, max(1, len(text) - cfg.calibration_seq_len * 4)))
            text = text[start : start + cfg.calibration_seq_len * 4]
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.calibration_seq_len,
            padding="max_length",
        )
        data.append(enc["input_ids"])
    return torch.cat(data, dim=0)


def run_gptq_quantization(
    *,
    run_dir: Path,
    in_model_dir: Path,
    out_dir: Path,
    dataset_cfg: DatasetConfig,
    cfg: GPTQConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imported lazily because auto-gptq is optional/heavy.
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "auto-gptq is not installed. Install with: pip install -e '.[quant]'"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(in_model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    calib_ids = _build_calibration_data(tokenizer, dataset_cfg, cfg)
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
        },
    )
