from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from .config import DatasetConfig, DistillConfig, save_json
from .replay import apply_global_seeds
from .reporting import (
    dataset_provenance,
    jsonl_writer,
    write_metrics,
    write_provenance,
    write_samples_jsonl,
)
from .security import resolve_path_under_base


def _precision_kwargs(precision: str) -> dict:
    # accelerate uses mixed_precision; model dtype is still set explicitly
    if precision == "bf16":
        return {"mixed_precision": "bf16"}
    return {"mixed_precision": "fp16"}


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


def run_distillation(
    *,
    run_dir: Path,
    out_dir: Path,
    dataset_cfg: DatasetConfig,
    cfg: DistillConfig,
    seed: int = 42,
) -> None:
    apply_global_seeds(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(
        run_dir, extra={"stage": "distill", "seed": seed, **dataset_provenance(dataset_cfg)}
    )
    steps_log_path = run_dir / "logs" / "distill_train_steps.jsonl"

    accelerator = Accelerator(**_precision_kwargs(cfg.precision))
    device = accelerator.device

    if cfg.trust_remote_code and accelerator.is_local_main_process:
        accelerator.print(
            "WARNING: trust_remote_code=True. Only use this with models you trust; "
            "it can execute arbitrary code from the model repository."
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        torch_dtype=_model_dtype(cfg.precision),
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
        torch_dtype=_model_dtype(cfg.precision),
        device_map=None,  # let accelerate place
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    optimizer = AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.steps
    )

    student, optimizer, scheduler = accelerator.prepare(student, optimizer, scheduler)

    # Resume support (basic): user can pass a checkpoint path; "auto" finds latest.
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(exist_ok=True)
    if cfg.resume != "none":
        resume_path: Path | None = None
        if cfg.resume == "auto":
            candidates = sorted(ckpt_root.glob("step_*"), key=lambda p: p.name)
            resume_path = candidates[-1] if candidates else None
        else:
            resume_path = resolve_path_under_base(
                Path(str(cfg.resume)), base=ckpt_root, must_exist=True
            )
        if resume_path and (resume_path / "accelerate_state").exists():
            accelerator.print(f"Resuming from {resume_path}")
            accelerator.load_state(resume_path / "accelerate_state")

    texts = _iter_texts(dataset_cfg)
    data_iter = iter(texts)

    losses: list[float] = []
    recent = deque(maxlen=20)
    pbar = tqdm(range(cfg.steps), disable=not accelerator.is_local_main_process)

    student.train()
    with jsonl_writer(steps_log_path) as write_step:
        for step in pbar:
            step_t0 = None
            if accelerator.is_local_main_process:
                import time

                step_t0 = time.time()
            # Sample next text (loop if needed when streaming)
            try:
                text = next(data_iter)
            except StopIteration:
                data_iter = iter(_iter_texts(dataset_cfg))
                text = next(data_iter)

            batch = _tokenize(tokenizer, text[: cfg.seq_len * 4], cfg.seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                t_out = teacher(**batch)
                t_logits = t_out.logits

            s_out = student(**batch, labels=batch["input_ids"])
            s_logits = s_out.logits
            hard_loss = s_out.loss

            # KL on last dimension; shift handled implicitly by labels loss, but for distill we align logits.
            T = cfg.temperature
            soft_teacher = F.softmax(t_logits / T, dim=-1)
            soft_student = F.log_softmax(s_logits / T, dim=-1)
            distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

            loss = cfg.alpha * distill_loss + (1.0 - cfg.alpha) * hard_loss
            loss = loss / cfg.grad_accum_steps
            accelerator.backward(loss)

            if (step + 1) % cfg.grad_accum_steps == 0:
                accelerator.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.detach().cpu()) * cfg.grad_accum_steps
            losses.append(loss_value)
            recent.append(loss_value)
            if recent:
                pbar.set_postfix(loss=float(sum(recent) / len(recent)))

            if accelerator.is_local_main_process:
                lr = (
                    float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else None
                )
                dt = None
                if step_t0 is not None:
                    import time

                    dt = float(time.time() - step_t0)
                write_step(
                    {
                        "stage": "distill",
                        "step": step + 1,
                        "loss": loss_value,
                        "lr": lr,
                        "dt_seconds": dt,
                    }
                )

        if (
            accelerator.is_local_main_process
            and cfg.save_every_steps > 0
            and (step + 1) % cfg.save_every_steps == 0
        ):
            ckpt_dir = ckpt_root / f"step_{step + 1:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(ckpt_dir / "accelerate_state")
            # rotate checkpoints
            keep = cfg.keep_last_n_checkpoints
            if keep and keep > 0:
                all_ckpts = sorted(ckpt_root.glob("step_*"), key=lambda p: p.name)
                for old in all_ckpts[:-keep]:
                    # best-effort cleanup
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

    # Save final model (main process only)
    if accelerator.is_local_main_process:
        accelerator.print(f"Saving distilled model to {out_dir}")
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        write_samples_jsonl(
            run_dir=run_dir,
            stage="distill",
            model=unwrapped,
            tokenizer=tokenizer,
            prompts=[
                "def fibonacci(n):",
                "def binary_search(arr, target):",
                "def quicksort(arr):",
            ],
        )
        write_metrics(
            run_dir,
            stage="distill",
            metrics={
                "steps": cfg.steps,
                "final_loss": (losses[-1] if losses else None),
                "loss_mean_recent": (sum(recent) / len(recent) if recent else None),
            },
        )
        save_json(
            out_dir / "training_log.json",
            {"losses": losses, "steps": cfg.steps, "config": asdict(cfg)},
        )

    accelerator.wait_for_everyone()
