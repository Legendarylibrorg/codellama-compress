from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DatasetConfig, DistillConfig, GPTQConfig, load_config_file, merge_dataclass
from .io import assert_disk_budget, new_run_dir, save_effective_config, write_env_report


def _load_blob(config: str | None) -> dict:
    if not config:
        return {}
    return load_config_file(Path(config))


def _dc_from_blob(blob: dict, key: str, dc):
    return merge_dataclass(dc, blob.get(key, {}))


def _load_dataset_cfg(blob: dict) -> DatasetConfig:
    return _dc_from_blob(blob, "dataset", DatasetConfig())


def _load_distill_cfg(blob: dict) -> DistillConfig:
    return _dc_from_blob(blob, "distill", DistillConfig())


def _load_finetune_cfg(blob: dict) -> DistillConfig:
    default = DistillConfig(teacher_model="", alpha=0.0, temperature=1.0)
    return _dc_from_blob(blob, "finetune", default)


def _load_gptq_cfg(blob: dict) -> GPTQConfig:
    return _dc_from_blob(blob, "gptq", GPTQConfig())


def _load_awq_cfg(blob: dict) -> GPTQConfig:
    # AWQ shares calibration knobs with GPTQ; allow either awq or gptq section.
    base = _load_gptq_cfg(blob)
    return _dc_from_blob(blob, "awq", base)


def _p(s: str | Path) -> Path:
    return s if isinstance(s, Path) else Path(s)


def _start_run(
    *,
    out_root: str,
    run_id: str | None,
    effective_config: dict,
    min_free_gb: float | None = None,
    env_report: bool = True,
) -> Path:
    out_root_p = Path(out_root)
    run_dir = new_run_dir(out_root_p, run_id=run_id)
    if min_free_gb is not None:
        assert_disk_budget(root=out_root_p, min_free_gb=min_free_gb)
    if env_report:
        write_env_report(run_dir)
    save_effective_config(run_dir, effective_config)
    return run_dir


def _finish_run(*, out_root: str, run_dir: Path, max_run_dir_gb: float | None = None) -> None:
    if max_run_dir_gb is None:
        return
    assert_disk_budget(root=Path(out_root), max_dir_gb=max_run_dir_gb, dir_path=run_dir)


def _cmd_distill_run(args: argparse.Namespace) -> int:
    blob = _load_blob(args.config)
    ds_cfg = _load_dataset_cfg(blob)
    d_cfg = _load_distill_cfg(blob)

    run_dir = _start_run(
        out_root=args.out_root,
        run_id=args.run_id,
        effective_config={"dataset": ds_cfg, "distill": d_cfg},
        min_free_gb=args.min_free_gb,
        env_report=args.env_report,
    )

    from .distill import run_distillation

    out_dir = run_dir / "distilled"
    run_distillation(run_dir=run_dir, out_dir=out_dir, dataset_cfg=ds_cfg, cfg=d_cfg)
    _finish_run(out_root=args.out_root, run_dir=run_dir, max_run_dir_gb=args.max_run_dir_gb)
    print(f"Done. Distilled model at: {out_dir}")
    return 0


def _cmd_prune_mask_mlp(args: argparse.Namespace) -> int:
    ratio = args.ratio
    method = args.method
    if args.config:
        blob = _load_blob(args.config)
        pr = blob.get("prune", {})
        ratio = float(pr.get("ratio", ratio))
        method = str(pr.get("method", method))

    run_dir = _start_run(
        out_root=args.out_root,
        run_id=args.run_id,
        effective_config={
            "prune": {"ratio": ratio, "method": method},
            "model_dir": str(args.model_dir),
        },
        env_report=args.env_report,
    )

    from .prune import run_mlp_mask_prune

    model_dir = _p(args.model_dir)
    out_dir = run_dir / "pruned"
    run_mlp_mask_prune(in_model_dir=model_dir, out_dir=out_dir, ratio=ratio, method=method)
    print(f"Done. Pruned model at: {out_dir}")
    return 0


def _cmd_finetune_run(args: argparse.Namespace) -> int:
    blob = _load_blob(args.config)
    ds_cfg = _load_dataset_cfg(blob)
    ft_cfg = _load_finetune_cfg(blob)

    run_dir = _start_run(
        out_root=args.out_root,
        run_id=args.run_id,
        effective_config={"dataset": ds_cfg, "finetune": ft_cfg, "model_dir": str(args.model_dir)},
        min_free_gb=args.min_free_gb,
        env_report=args.env_report,
    )

    from .finetune import run_finetune

    out_dir = run_dir / "finetuned"
    model_dir = _p(args.model_dir)
    run_finetune(
        run_dir=run_dir,
        in_model_dir=model_dir,
        out_dir=out_dir,
        dataset_cfg=ds_cfg,
        cfg=ft_cfg,
    )
    _finish_run(out_root=args.out_root, run_dir=run_dir, max_run_dir_gb=args.max_run_dir_gb)
    print(f"Done. Fine-tuned model at: {out_dir}")
    return 0


def _cmd_quant_gptq(args: argparse.Namespace) -> int:
    blob = _load_blob(args.config)
    ds_cfg = _load_dataset_cfg(blob)
    q_cfg = _load_gptq_cfg(blob)

    run_dir = _start_run(
        out_root=args.out_root,
        run_id=args.run_id,
        effective_config={"dataset": ds_cfg, "gptq": q_cfg, "model_dir": str(args.model_dir)},
        min_free_gb=args.min_free_gb,
        env_report=args.env_report,
    )

    from .quantize_gptq import run_gptq_quantization

    out_dir = run_dir / "quantized-gptq"
    model_dir = _p(args.model_dir)
    run_gptq_quantization(
        run_dir=run_dir,
        in_model_dir=model_dir,
        out_dir=out_dir,
        dataset_cfg=ds_cfg,
        cfg=q_cfg,
    )
    _finish_run(out_root=args.out_root, run_dir=run_dir, max_run_dir_gb=args.max_run_dir_gb)
    print(f"Done. Quantized model at: {out_dir}")
    return 0


def _cmd_quant_awq(args: argparse.Namespace) -> int:
    blob = _load_blob(args.config)
    ds_cfg = _load_dataset_cfg(blob)
    q_cfg = _load_awq_cfg(blob)

    run_dir = _start_run(
        out_root=args.out_root,
        run_id=args.run_id,
        effective_config={"dataset": ds_cfg, "awq": q_cfg, "model_dir": str(args.model_dir)},
        min_free_gb=args.min_free_gb,
        env_report=args.env_report,
    )

    from .quantize_awq import run_awq_quantization

    out_dir = run_dir / "quantized-awq"
    model_dir = _p(args.model_dir)
    run_awq_quantization(
        run_dir=run_dir,
        in_model_dir=model_dir,
        out_dir=out_dir,
        dataset_cfg=ds_cfg,
        cfg=q_cfg,
    )
    _finish_run(out_root=args.out_root, run_dir=run_dir, max_run_dir_gb=args.max_run_dir_gb)
    print(f"Done. Quantized model at: {out_dir}")
    return 0


def _cmd_quant_bnb(args: argparse.Namespace) -> int:
    from .quantize_bnb import write_bnb_bundle

    write_bnb_bundle(model_id_or_dir=args.model, out_dir=Path(args.out_dir))
    print(f"Wrote bnb load bundle to {args.out_dir}")
    return 0


def _cmd_evaluate_run(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_into_run_dir, evaluate_model_dir

    out_path = Path(args.out_path) if args.out_path else None
    if args.run_dir:
        res = evaluate_into_run_dir(run_dir=Path(args.run_dir), model_dir=Path(args.model_dir))
    else:
        res = evaluate_model_dir(Path(args.model_dir), out_path=out_path)
    print(res)
    return 0


def _cmd_evaluate_benchmark(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from .benchmarks import run_benchmarks

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.out_root or args.run_id:
        run_dir = new_run_dir(Path(args.out_root or "output/runs"), run_id=args.run_id)
        out_dir = run_dir / "benchmarks"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("output/benchmarks") / ts

    res = run_benchmarks(
        model_dir=Path(args.model_dir),
        tasks=args.tasks,
        out_dir=out_dir,
        seed=args.seed,
        limit=args.max_samples,
        save_per_sample=args.save_per_sample,
    )
    print(f"Wrote benchmark results to {out_dir}")
    # Also print high-level result keys for quick view
    print({"tasks": args.tasks, "results_keys": sorted((res.get("results") or {}).keys())})
    return 0


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    model_name = args.model_name
    port = args.port
    if args.config:
        blob = _load_blob(args.config)
        exp = blob.get("export", {})
        model_name = str(exp.get("model_name", model_name))
        port = int(exp.get("port", port))

    from .export import write_export_bundle

    write_export_bundle(
        model_dir=_p(args.model_dir),
        out_dir=_p(args.out_dir),
        model_name=model_name,
        port=port,
    )
    print(f"Wrote export bundle to {args.out_dir}")
    return 0


def _cmd_util_env_report(args: argparse.Namespace) -> int:
    write_env_report(_p(args.run_dir))
    print(f"Wrote env report to {args.run_dir}")
    return 0


def _cmd_util_verify_artifact(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", torch_dtype=torch.float16
    )
    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True))
    return 0


def _cmd_util_speculative(args: argparse.Namespace) -> int:
    from .speculative import speculative_generate

    if args.allow_mismatched_tokenizers:
        print(
            "WARNING: --allow-mismatched-tokenizers is enabled. "
            "This can produce incorrect output if draft/target tokenizers differ."
        )

    text, stats = speculative_generate(
        prompt=args.prompt,
        target_model=args.target_model,
        draft_model=args.draft_model,
        max_new_tokens=args.max_new_tokens,
        num_speculative_tokens=args.num_speculative_tokens,
        allow_mismatched_tokenizers=args.allow_mismatched_tokenizers,
    )
    print(text)
    print(stats.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codellama-compress")
    sp = p.add_subparsers(dest="cmd", required=True)

    # distill run
    distill = sp.add_parser("distill", help="Knowledge distillation commands.")
    distill_sp = distill.add_subparsers(dest="sub", required=True)
    distill_run = distill_sp.add_parser("run", help="Run teacher->student logit distillation.")
    distill_run.add_argument("--run-id", default=None)
    distill_run.add_argument("--out-root", default="output/runs")
    distill_run.add_argument("--config", default=None, help="JSON config file.")
    distill_run.add_argument(
        "--env-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write env report files (pip freeze, nvidia-smi on Linux, etc.) into the run dir.",
    )
    distill_run.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if free space under this many GB.",
    )
    distill_run.add_argument(
        "--max-run-dir-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if the run directory exceeds this many GB.",
    )
    distill_run.set_defaults(func=_cmd_distill_run)

    # prune mask-mlp
    prune = sp.add_parser("prune", help="Pruning/masking commands.")
    prune_sp = prune.add_subparsers(dest="sub", required=True)
    prune_mask = prune_sp.add_parser(
        "mask-mlp", help="Mask a fraction of MLP neurons (shape-preserving)."
    )
    prune_mask.add_argument("--model-dir", required=True)
    prune_mask.add_argument("--run-id", default=None)
    prune_mask.add_argument("--out-root", default="output/runs")
    prune_mask.add_argument("--ratio", type=float, default=0.25)
    prune_mask.add_argument("--method", default="magnitude")
    prune_mask.add_argument("--config", default=None, help="JSON config file.")
    prune_mask.add_argument(
        "--env-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write env report files (pip freeze, nvidia-smi on Linux, etc.) into the run dir.",
    )
    prune_mask.set_defaults(func=_cmd_prune_mask_mlp)

    # finetune run
    finetune = sp.add_parser("finetune", help="Fine-tuning commands.")
    finetune_sp = finetune.add_subparsers(dest="sub", required=True)
    finetune_run = finetune_sp.add_parser(
        "run", help="Run post-prune recovery fine-tuning (LM loss)."
    )
    finetune_run.add_argument("--model-dir", required=True)
    finetune_run.add_argument("--run-id", default=None)
    finetune_run.add_argument("--out-root", default="output/runs")
    finetune_run.add_argument("--config", default=None, help="JSON config file.")
    finetune_run.add_argument(
        "--env-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write env report files (pip freeze, nvidia-smi on Linux, etc.) into the run dir.",
    )
    finetune_run.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if free space under this many GB.",
    )
    finetune_run.add_argument(
        "--max-run-dir-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if the run directory exceeds this many GB.",
    )
    finetune_run.set_defaults(func=_cmd_finetune_run)

    # quantize
    quant = sp.add_parser("quantize", help="Quantization commands (some require extras).")
    quant_sp = quant.add_subparsers(dest="sub", required=True)
    qg = quant_sp.add_parser("gptq", help="Quantize a model with GPTQ (requires '.[quant]').")
    qg.add_argument("--model-dir", required=True)
    qg.add_argument("--run-id", default=None)
    qg.add_argument("--out-root", default="output/runs")
    qg.add_argument("--config", default=None, help="JSON config file.")
    qg.add_argument(
        "--env-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write env report files (pip freeze, nvidia-smi on Linux, etc.) into the run dir.",
    )
    qg.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if free space under this many GB.",
    )
    qg.add_argument(
        "--max-run-dir-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if the run directory exceeds this many GB.",
    )
    qg.set_defaults(func=_cmd_quant_gptq)

    qa = quant_sp.add_parser("awq", help="Quantize a model with AWQ (requires '.[quant]').")
    qa.add_argument("--model-dir", required=True)
    qa.add_argument("--run-id", default=None)
    qa.add_argument("--out-root", default="output/runs")
    qa.add_argument("--config", default=None, help="JSON config file.")
    qa.add_argument(
        "--env-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write env report files (pip freeze, nvidia-smi on Linux, etc.) into the run dir.",
    )
    qa.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if free space under this many GB.",
    )
    qa.add_argument(
        "--max-run-dir-gb",
        type=float,
        default=None,
        help="Optional safety guard: error if the run directory exceeds this many GB.",
    )
    qa.set_defaults(func=_cmd_quant_awq)

    qb = quant_sp.add_parser("bnb")
    qb.add_argument("--model", required=True, help="HF model id or local dir.")
    qb.add_argument("--out-dir", default="output/bnb")
    qb.set_defaults(func=_cmd_quant_bnb)

    # evaluate run
    ev = sp.add_parser("evaluate", help="Evaluation commands (lightweight smoke checks).")
    ev_sp = ev.add_subparsers(dest="sub", required=True)
    ev_run = ev_sp.add_parser("run", help="Smoke-evaluate a model directory (small prompts).")
    ev_run.add_argument("--model-dir", required=True)
    ev_run.add_argument("--out-path", default=None)
    ev_run.add_argument(
        "--run-dir",
        default=None,
        help="If set, also write research-grade metrics/provenance into this run directory.",
    )
    ev_run.add_argument("--config", default=None, help="JSON config file (reserved).")
    ev_run.set_defaults(func=_cmd_evaluate_run)

    ev_bench = ev_sp.add_parser("benchmark", help="Run research benchmarks (requires '.[eval]').")
    ev_bench.add_argument("--model-dir", required=True)
    ev_bench.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated task list (e.g. humaneval,mbpp).",
    )
    ev_bench.add_argument(
        "--out-dir", default=None, help="Output directory for benchmark artifacts."
    )
    ev_bench.add_argument(
        "--out-root", default=None, help="If set, write under output/runs/<run_id>/benchmarks."
    )
    ev_bench.add_argument(
        "--run-id", default=None, help="Used with --out-root for run directory naming."
    )
    ev_bench.add_argument("--seed", type=int, default=42)
    ev_bench.add_argument(
        "--max-samples", type=int, default=None, help="Optional limit for quick runs."
    )
    ev_bench.add_argument(
        "--save-per-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-sample details JSONL (large).",
    )
    ev_bench.set_defaults(func=_cmd_evaluate_benchmark)

    # export bundle
    ex = sp.add_parser("export", help="Generate helper artifacts for serving/export.")
    ex_sp = ex.add_subparsers(dest="sub", required=True)
    ex_bundle = ex_sp.add_parser(
        "bundle", help="Write a serving/export helper bundle (scripts + Dockerfile)."
    )
    ex_bundle.add_argument("--model-dir", required=True)
    ex_bundle.add_argument("--out-dir", default="output/export")
    ex_bundle.add_argument("--model-name", default="codellama-compressed")
    ex_bundle.add_argument("--port", type=int, default=8000)
    ex_bundle.add_argument("--config", default=None, help="JSON config file (optional).")
    ex_bundle.set_defaults(func=_cmd_export_bundle)

    # util
    util = sp.add_parser("util", help="Utilities.")
    util_sp = util.add_subparsers(dest="sub", required=True)
    env = util_sp.add_parser("env-report")
    env.add_argument("--run-dir", default="output/runs/_env_report")
    env.set_defaults(func=_cmd_util_env_report)

    va = util_sp.add_parser("verify-artifact")
    va.add_argument("--model-dir", required=True)
    va.add_argument("--prompt", default="def fibonacci(n):")
    va.add_argument("--max-new-tokens", type=int, default=64)
    va.set_defaults(func=_cmd_util_verify_artifact)

    sd = util_sp.add_parser(
        "speculative", help="Run speculative decoding with draft+target models."
    )
    sd.add_argument("--target-model", required=True)
    sd.add_argument("--draft-model", required=True)
    sd.add_argument("--prompt", default="def fibonacci(n):")
    sd.add_argument("--max-new-tokens", type=int, default=256)
    sd.add_argument("--num-speculative-tokens", type=int, default=5)
    sd.add_argument("--allow-mismatched-tokenizers", action="store_true")
    sd.set_defaults(func=_cmd_util_speculative)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
