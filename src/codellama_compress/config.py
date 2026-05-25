from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "bigcode/starcoderdata"
    config: str = "python"
    split: str = "train"
    streaming: bool = True
    shuffle_buffer: int = 10_000
    max_train_samples: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class DistillConfig:
    teacher_model: str = "codellama/CodeLlama-13b-hf"
    student_model: str = "codellama/CodeLlama-7b-hf"
    seq_len: int = 512
    steps: int = 1_000
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    precision: Literal["bf16", "fp16"] = "bf16"
    gradient_checkpointing: bool = True
    temperature: float = 2.0
    alpha: float = 0.5  # distill vs hard-loss
    save_every_steps: int = 200
    keep_last_n_checkpoints: int = 3
    resume: Literal["auto", "none"] | str = "auto"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    damp_percent: float = 0.01
    calibration_samples: int = 128
    calibration_seq_len: int = 512
    seed: int = 42


def to_jsonable(d: Any) -> Any:
    if hasattr(d, "__dataclass_fields__"):
        return asdict(d)
    if isinstance(d, Path):
        return str(d)
    return d


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=to_jsonable) + "\n")


def load_config_file(path: Path) -> dict[str, Any]:
    from .security import load_bounded_json_config

    return load_bounded_json_config(path)


def merge_dataclass(dc: Any, updates: dict[str, Any]) -> Any:
    """
    Shallow merge a dict into a dataclass instance (returns a new instance).

    Intended for CLI config overrides where nested structures are handled at the top level.
    """
    from .security import merge_dataclass_fields

    return merge_dataclass_fields(dc, updates)
