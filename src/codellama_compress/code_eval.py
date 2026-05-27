from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .code_exec import run_python_sandboxed
from .reporting import jsonl_writer, write_metrics, write_provenance
from .security import assert_code_exec_permitted, is_isolated_execution_environment, resolve_user_path


@dataclass(frozen=True)
class CodeEvalResult:
    task: str
    pass_at_k: dict[str, float]
    n_problems: int


def _pass_at_k(n: int, c: int, k: int) -> float:
    # Standard pass@k estimator (OpenAI HumanEval).
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - (k / i)
    return 1.0 - prod


def _generate_completions(
    *,
    model,
    tok,
    prompt: str,
    k: int,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    g = torch.Generator(device=model.device)
    g.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    outs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=k,
        pad_token_id=tok.eos_token_id,
        generator=g,
    )
    texts: list[str] = []
    for seq in outs:
        full = tok.decode(seq, skip_special_tokens=True)
        # Strip prompt prefix when possible
        if full.startswith(prompt):
            full = full[len(prompt) :]
        texts.append(full)
    return texts


def _humaneval_program(prompt: str, completion: str, test: str) -> str:
    return (
        "# Generated solution\n"
        + prompt
        + completion
        + "\n\n# Tests\n"
        + test
        + "\n"
    )


def _mbpp_program(prompt: str, completion: str, tests: list[str]) -> str:
    # MBPP prompts are natural language; we expect completion to define a function.
    joined = "\n".join(tests)
    return "# Generated solution\n" + completion + "\n\n# Tests\n" + joined + "\n"


def run_code_eval(
    *,
    run_dir: Path,
    model_dir: Path,
    suite: Literal["humaneval", "mbpp"],
    k: int = 10,
    max_new_tokens: int = 256,
    seed: int = 42,
    temperature: float = 0.2,
    top_p: float = 0.95,
    limit: int | None = None,
    allow_insecure_code_exec: bool = False,
) -> CodeEvalResult:
    assert_code_exec_permitted(allow_insecure=allow_insecure_code_exec)
    out_dir = run_dir / "code_eval" / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(
        run_dir,
        extra={
            "stage": f"code_eval_{suite}",
            "suite": suite,
            "code_exec_isolated": is_isolated_execution_environment(),
            "allow_insecure_code_exec": allow_insecure_code_exec,
        },
    )

    model_dir = resolve_user_path(model_dir, must_exist=True)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )

    details_path = out_dir / "details.jsonl"
    total = 0
    solved = 0
    pass_counts: list[int] = []

    if suite == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        rows = list(ds)
        if limit is not None:
            rows = rows[:limit]
        with jsonl_writer(details_path) as write:
            for idx, row in enumerate(rows):
                total += 1
                prompt = row["prompt"]
                test = row["test"]
                completions = _generate_completions(
                    model=model,
                    tok=tok,
                    prompt=prompt,
                    k=k,
                    max_new_tokens=max_new_tokens,
                    seed=seed + idx,
                    temperature=temperature,
                    top_p=top_p,
                )
                ok_count = 0
                for j, comp in enumerate(completions):
                    code = _humaneval_program(prompt, comp, test)
                    r = run_python_sandboxed(code=code)
                    ok_count += int(r.ok)
                    write(
                        {
                            "suite": suite,
                            "index": idx,
                            "completion_idx": j,
                            "ok": r.ok,
                            "reason": r.reason,
                            "exit_code": r.exit_code,
                        }
                    )
                pass_counts.append(ok_count)
                if ok_count > 0:
                    solved += 1
    else:
        ds = load_dataset("google-research-datasets/mbpp", split="test")
        rows = list(ds)
        if limit is not None:
            rows = rows[:limit]
        with jsonl_writer(details_path) as write:
            for idx, row in enumerate(rows):
                total += 1
                prompt = row.get("text", "")
                tests = row.get("test_list") or []
                completions = _generate_completions(
                    model=model,
                    tok=tok,
                    prompt=prompt,
                    k=k,
                    max_new_tokens=max_new_tokens,
                    seed=seed + idx,
                    temperature=temperature,
                    top_p=top_p,
                )
                ok_count = 0
                for j, comp in enumerate(completions):
                    code = _mbpp_program(prompt, comp, tests)
                    r = run_python_sandboxed(code=code)
                    ok_count += int(r.ok)
                    write(
                        {
                            "suite": suite,
                            "index": idx,
                            "completion_idx": j,
                            "ok": r.ok,
                            "reason": r.reason,
                            "exit_code": r.exit_code,
                        }
                    )
                pass_counts.append(ok_count)
                if ok_count > 0:
                    solved += 1

    ks = [1, 5, 10]
    pass_at: dict[str, float] = {}
    for kk in ks:
        kk = min(kk, k)
        vals = [_pass_at_k(k, c, kk) for c in pass_counts]
        pass_at[f"pass@{kk}"] = float(sum(vals) / max(1, len(vals)))

    res = CodeEvalResult(task=suite, pass_at_k=pass_at, n_problems=total)
    summary = {**res.__dict__, "solved_any": solved}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_metrics(run_dir, stage=f"code_eval_{suite}", metrics=summary)
    return res
