from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import save_json
from .reporting import write_metrics, write_provenance, write_samples_jsonl
from .security import resolve_user_path


@dataclass(frozen=True)
class EvalResult:
    model: str
    perplexity: float
    tokens_per_second: float
    avg_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Model: {self.model} | "
            f"PPL: {self.perplexity:.2f} | "
            f"Tokens/s: {self.tokens_per_second:.1f} | "
            f"Avg time: {self.avg_time_ms:.0f}ms"
        )


def _default_texts() -> list[str]:
    return [
        "def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    return n * calculate_factorial(n - 1)\n",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop() if self.items else None\n",
    ]


def compute_perplexity(model, tokenizer, texts: Iterable[str], device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, labels=enc["input_ids"])
        total_loss += float(out.loss) * int(enc["input_ids"].shape[1])
        total_tokens += int(enc["input_ids"].shape[1])
    avg_loss = total_loss / max(1, total_tokens)
    return float(torch.exp(torch.tensor(avg_loss)))


def measure_speed(
    model, tokenizer, prompt: str, device: torch.device, runs: int = 5
) -> tuple[float, float]:
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model.generate(**enc, max_new_tokens=8, do_sample=False)
    times = []
    toks = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        toks.append(int(out.shape[1] - enc.input_ids.shape[1]))
    avg_time = float(fmean(times)) if times else 0.0
    avg_tokens = float(fmean(toks)) if toks else 0.0
    return (avg_tokens / max(1e-9, avg_time), avg_time * 1000.0)


def evaluate_model_dir(model_dir: Path, out_path: Path | None = None) -> EvalResult:
    # This is intentionally a lightweight smoke evaluation.
    model_dir = resolve_user_path(model_dir, must_exist=True)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    device = model.device
    ppl = compute_perplexity(model, tok, _default_texts(), device)
    tps, ms = measure_speed(model, tok, "def fibonacci(n):", device)
    res = EvalResult(model=str(model_dir), perplexity=ppl, tokens_per_second=tps, avg_time_ms=ms)
    if out_path is not None:
        save_json(out_path, res.to_dict())
    return res


def evaluate_into_run_dir(*, run_dir: Path, model_dir: Path) -> EvalResult:
    model_dir = resolve_user_path(model_dir, must_exist=True)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    write_provenance(run_dir, extra={"stage": "evaluate"})
    device = model.device
    ppl = compute_perplexity(model, tok, _default_texts(), device)
    tps, ms = measure_speed(model, tok, "def fibonacci(n):", device)
    res = EvalResult(model=str(model_dir), perplexity=ppl, tokens_per_second=tps, avg_time_ms=ms)
    write_samples_jsonl(
        run_dir=run_dir,
        stage="evaluate",
        model=model,
        tokenizer=tok,
        prompts=[
            "def fibonacci(n):",
            "def binary_search(arr, target):",
            "def quicksort(arr):",
        ],
    )
    write_metrics(run_dir, stage="evaluate", metrics=res.to_dict())
    return res
