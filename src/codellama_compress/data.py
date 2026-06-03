from __future__ import annotations

import random
from collections.abc import Iterable

from datasets import load_dataset

from .config import DatasetConfig
from .security import dataset_load_extra_kwargs, normalize_training_text


def iter_dataset_texts(dataset_cfg: DatasetConfig, *, force_streaming: bool = False) -> Iterable[str]:
    """Yield normalized non-empty code/text samples from an allowlisted dataset."""
    kwargs = dataset_load_extra_kwargs(dataset_cfg)
    if force_streaming:
        kwargs = {**kwargs, "streaming": True}

    ds = load_dataset(dataset_cfg.name, dataset_cfg.config, **kwargs)
    if dataset_cfg.streaming or force_streaming:
        ds = ds.shuffle(buffer_size=dataset_cfg.shuffle_buffer, seed=dataset_cfg.seed)

    n = 0
    for row in ds:
        txt = row.get("content") or row.get("text") or ""
        if not isinstance(txt, str) or not txt.strip():
            continue
        yield normalize_training_text(txt)
        n += 1
        if dataset_cfg.max_train_samples is not None and n >= dataset_cfg.max_train_samples:
            break


def sample_calibration_texts(
    dataset_cfg: DatasetConfig,
    *,
    samples: int,
    seq_len: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Collect deterministic calibration text samples from a streaming dataset."""
    rng = random.Random(dataset_cfg.seed if seed is None else seed)
    texts: list[str] = []
    for text in iter_dataset_texts(dataset_cfg, force_streaming=True):
        if seq_len is not None and len(text) > seq_len * 4:
            start = rng.randrange(0, max(1, len(text) - seq_len * 4))
            text = text[start : start + seq_len * 4]
        texts.append(text)
        if len(texts) >= samples:
            break
    return texts
