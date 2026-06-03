from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path

from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import DatasetConfig, DistillConfig, save_json
from .data import iter_dataset_texts
from .replay import apply_global_seeds
from .reporting import (
    dataset_provenance,
    jsonl_writer,
    write_metrics,
    write_provenance,
    write_samples_jsonl,
)
from .security import (
    resolve_trust_remote_code,
    trust_remote_code_audit_record,
)
from .training_utils import (
    ensure_pad_token,
    model_dtype,
    precision_kwargs,
    print_trust_remote_code_notice,
    tokenize_text,
)


def run_finetune(
    *,
    run_dir: Path,
    in_model_dir: Path,
    out_dir: Path,
    dataset_cfg: DatasetConfig,
    cfg: DistillConfig,
    seed: int = 42,
) -> None:
    """
    Post-pruning recovery fine-tuning (LM loss only).

    We reuse `DistillConfig` for training knobs; teacher/distill-related fields are ignored.
    """
    apply_global_seeds(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    trust_rc = resolve_trust_remote_code(cfg.trust_remote_code)
    write_provenance(
        run_dir,
        extra={
            "stage": "finetune",
            "seed": seed,
            **dataset_provenance(dataset_cfg),
            **trust_remote_code_audit_record(
                config_flag=cfg.trust_remote_code, effective=trust_rc
            ),
        },
    )
    steps_log_path = run_dir / "logs" / "finetune_train_steps.jsonl"

    accelerator = Accelerator(**precision_kwargs(cfg.precision))
    device = accelerator.device

    print_trust_remote_code_notice(
        accelerator, requested=cfg.trust_remote_code, effective=trust_rc
    )

    tokenizer = AutoTokenizer.from_pretrained(
        in_model_dir, use_fast=True, trust_remote_code=trust_rc
    )
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        in_model_dir,
        torch_dtype=model_dtype(cfg.precision),
        device_map=None,
        trust_remote_code=trust_rc,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.steps
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    texts = iter(iter_dataset_texts(dataset_cfg))
    losses: list[float] = []
    recent = deque(maxlen=20)
    pbar = tqdm(range(cfg.steps), disable=not accelerator.is_local_main_process)
    model.train()
    with jsonl_writer(steps_log_path) as write_step:
        for step in pbar:
            step_t0 = None
            if accelerator.is_local_main_process:
                import time

                step_t0 = time.time()

            try:
                text = next(texts)
            except StopIteration:
                texts = iter(iter_dataset_texts(dataset_cfg))
                text = next(texts)

            batch = tokenize_text(tokenizer, text[: cfg.seq_len * 4], cfg.seq_len)
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
                lr = (
                    float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else None
                )
                dt = None
                if step_t0 is not None:
                    import time

                    dt = float(time.time() - step_t0)
                write_step(
                    {
                        "stage": "finetune",
                        "step": step + 1,
                        "loss": loss_value,
                        "lr": lr,
                        "dt_seconds": dt,
                    }
                )

    if accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        write_samples_jsonl(
            run_dir=run_dir,
            stage="finetune",
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
            stage="finetune",
            metrics={
                "steps": cfg.steps,
                "final_loss": (losses[-1] if losses else None),
                "loss_mean_recent": (sum(recent) / len(recent) if recent else None),
            },
        )
        save_json(
            out_dir / "finetune_log.json",
            {"losses": losses, "steps": cfg.steps, "config": asdict(cfg)},
        )

    accelerator.wait_for_everyone()
