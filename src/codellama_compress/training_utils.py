from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .security import TRUST_REMOTE_CODE_ENV


def precision_kwargs(precision: str) -> dict[str, str]:
    return {"mixed_precision": "bf16" if precision == "bf16" else "fp16"}


def model_dtype(precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == "bf16" else torch.float16


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def tokenize_text(tokenizer: Any, text: str, seq_len: int) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
    )
    return {k: v for k, v in enc.items()}


def print_trust_remote_code_notice(
    accelerator: Any, *, requested: bool, effective: bool
) -> None:
    if requested and not effective and accelerator.is_local_main_process:
        accelerator.print(
            f"NOTE: config requests trust_remote_code but it is disabled. "
            f"Set {TRUST_REMOTE_CODE_ENV}=1 only for models you fully trust."
        )
    if effective and accelerator.is_local_main_process:
        accelerator.print(
            "WARNING: trust_remote_code is enabled. Only use with models you trust; "
            "it can execute arbitrary code from the model repository."
        )


def latest_checkpoint(checkpoint_root: Path) -> Path | None:
    candidates = sorted(checkpoint_root.glob("step_*"), key=lambda p: p.name)
    return candidates[-1] if candidates else None


def rotate_checkpoints(checkpoint_root: Path, *, keep: int | None) -> None:
    if keep is None or keep <= 0:
        return
    all_ckpts = sorted(checkpoint_root.glob("step_*"), key=lambda p: p.name)
    for old in all_ckpts[:-keep]:
        for child in old.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        for child in sorted(old.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            old.rmdir()
        except OSError:
            pass
