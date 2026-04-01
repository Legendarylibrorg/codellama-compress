from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import DatasetConfig, DistillConfig, save_json


def _precision_kwargs(precision: str) -> dict:
    return {"mixed_precision": "bf16" if precision == "bf16" else "fp16"}


def _model_dtype(precision: str):
    return torch.bfloat16 if precision == "bf16" else torch.float16


def _iter_texts(dataset_cfg: DatasetConfig) -> Iterable[str]:
    ds = load_dataset(
        dataset_cfg.name,
        dataset_cfg.config,
        split=dataset_cfg.split,
        streaming=dataset_cfg.streaming,
    )
    if dataset_cfg.streaming:
        ds = ds.shuffle(buffer_size=dataset_cfg.shuffle_buffer, seed=dataset_cfg.seed)
    n = 0
    for row in ds:
        txt = row.get("content") or row.get("text") or ""
        if not isinstance(txt, str) or not txt.strip():
            continue
        yield txt
        n += 1
        if dataset_cfg.max_train_samples is not None and n >= dataset_cfg.max_train_samples:
            break


def _tokenize(tokenizer, text: str, seq_len: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )
    return {k: v for k, v in enc.items()}


def run_finetune(
    *,
    run_dir: Path,
    in_model_dir: Path,
    out_dir: Path,
    dataset_cfg: DatasetConfig,
    cfg: DistillConfig,
) -> None:
    """
    Post-pruning recovery fine-tuning (LM loss only).

    We reuse `DistillConfig` for training knobs; teacher/distill-related fields are ignored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(**_precision_kwargs(cfg.precision))
    device = accelerator.device

    if cfg.trust_remote_code and accelerator.is_local_main_process:
        accelerator.print(
            "WARNING: trust_remote_code=True. Only use this with models you trust; "
            "it can execute arbitrary code from the model repository."
        )

    tokenizer = AutoTokenizer.from_pretrained(in_model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        in_model_dir,
        torch_dtype=_model_dtype(cfg.precision),
        device_map=None,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.steps
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    texts = iter(_iter_texts(dataset_cfg))
    losses: list[float] = []
    recent = deque(maxlen=20)
    pbar = tqdm(range(cfg.steps), disable=not accelerator.is_local_main_process)
    model.train()

    for step in pbar:
        try:
            text = next(texts)
        except StopIteration:
            texts = iter(_iter_texts(dataset_cfg))
            text = next(texts)

        batch = _tokenize(tokenizer, text[: cfg.seq_len * 4], cfg.seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch, labels=batch["input_ids"])
        loss = out.loss / cfg.grad_accum_steps
        accelerator.backward(loss)

        if (step + 1) % cfg.grad_accum_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().cpu()) * cfg.grad_accum_steps
        losses.append(loss_value)
        recent.append(loss_value)
        if recent:
            pbar.set_postfix(loss=float(sum(recent) / len(recent)))

    if accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        save_json(
            out_dir / "finetune_log.json",
            {"losses": losses, "steps": cfg.steps, "config": asdict(cfg)},
        )

    accelerator.wait_for_everyone()
