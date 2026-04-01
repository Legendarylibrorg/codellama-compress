from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from .config import DatasetConfig, DistillConfig, GPTQConfig, load_config_file, merge_dataclass
from .io import new_run_dir, save_effective_config, write_env_report

app = typer.Typer(add_completion=False, help="Code LLM compression pipeline (distill + quantize).")
distill_app = typer.Typer(add_completion=False, help="Knowledge distillation commands.")
quant_app = typer.Typer(add_completion=False, help="Quantization commands.")
util_app = typer.Typer(add_completion=False, help="Utility commands.")
prune_app = typer.Typer(add_completion=False, help="Pruning commands.")
finetune_app = typer.Typer(add_completion=False, help="Fine-tuning commands.")
eval_app = typer.Typer(add_completion=False, help="Evaluation commands.")
export_app = typer.Typer(add_completion=False, help="Export/deploy helper commands.")

app.add_typer(distill_app, name="distill")
app.add_typer(prune_app, name="prune")
app.add_typer(finetune_app, name="finetune")
app.add_typer(quant_app, name="quantize")
app.add_typer(eval_app, name="evaluate")
app.add_typer(export_app, name="export")
app.add_typer(util_app, name="util")


def _merge_dict_into_dataclass(dc, updates: dict) -> object:
    return merge_dataclass(dc, updates)


@distill_app.command("run")
def distill_run(
    run_id: Optional[str] = typer.Option(None, help="Run ID (default: timestamp UTC)."),
    out_root: Path = typer.Option(Path("output/runs"), help="Root output directory."),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file."),
    print_effective_config: bool = typer.Option(False, help="Print resolved config and exit."),
):
    """
    Run teacher->student logit distillation.
    """
    ds_cfg = DatasetConfig()
    d_cfg = DistillConfig()

    if config is not None:
        blob = load_config_file(config)
        ds_cfg = _merge_dict_into_dataclass(ds_cfg, blob.get("dataset", {}))  # type: ignore[assignment]
        d_cfg = _merge_dict_into_dataclass(d_cfg, blob.get("distill", {}))  # type: ignore[assignment]

    effective = {"dataset": ds_cfg, "distill": d_cfg}
    if print_effective_config:
        print(effective)
        raise typer.Exit(code=0)

    run_dir = new_run_dir(out_root, run_id=run_id)
    write_env_report(run_dir)
    save_effective_config(run_dir, effective)

    from .distill import run_distillation

    stage_dir = run_dir / "distilled"
    run_distillation(run_dir=run_dir, out_dir=stage_dir, dataset_cfg=ds_cfg, cfg=d_cfg)
    print(f"[green]Done.[/green] Distilled model at: {stage_dir}")


@quant_app.command("gptq")
def quantize_gptq(
    model_dir: Path = typer.Option(..., help="Input model directory (HF format)."),
    run_id: Optional[str] = typer.Option(None, help="Run ID (default: timestamp UTC)."),
    out_root: Path = typer.Option(Path("output/runs"), help="Root output directory."),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file."),
    print_effective_config: bool = typer.Option(False, help="Print resolved config and exit."),
):
    """
    Quantize a model using GPTQ (auto-gptq).
    """
    ds_cfg = DatasetConfig()
    q_cfg = GPTQConfig()

    if config is not None:
        blob = load_config_file(config)
        ds_cfg = _merge_dict_into_dataclass(ds_cfg, blob.get("dataset", {}))  # type: ignore[assignment]
        q_cfg = _merge_dict_into_dataclass(q_cfg, blob.get("gptq", {}))  # type: ignore[assignment]

    effective = {"dataset": ds_cfg, "gptq": q_cfg, "model_dir": str(model_dir)}
    if print_effective_config:
        print(effective)
        raise typer.Exit(code=0)

    run_dir = new_run_dir(out_root, run_id=run_id)
    write_env_report(run_dir)
    save_effective_config(run_dir, effective)

    from .quantize_gptq import run_gptq_quantization

    out_dir = run_dir / "quantized-gptq"
    run_gptq_quantization(
        run_dir=run_dir, in_model_dir=model_dir, out_dir=out_dir, dataset_cfg=ds_cfg, cfg=q_cfg
    )
    print(f"[green]Done.[/green] Quantized model at: {out_dir}")


@quant_app.command("awq")
def quantize_awq(
    model_dir: Path = typer.Option(..., help="Input model directory (HF format)."),
    run_id: Optional[str] = typer.Option(None, help="Run ID (default: timestamp UTC)."),
    out_root: Path = typer.Option(Path("output/runs"), help="Root output directory."),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file."),
):
    """
    Quantize a model using AWQ (autoawq).
    """
    ds_cfg = DatasetConfig()
    q_cfg = GPTQConfig()  # reuse shared calibration knobs (bits ignored by AWQ for now)
    if config is not None:
        blob = load_config_file(config)
        ds_cfg = _merge_dict_into_dataclass(ds_cfg, blob.get("dataset", {}))  # type: ignore[assignment]
        q_cfg = _merge_dict_into_dataclass(q_cfg, blob.get("awq", blob.get("gptq", {})))  # type: ignore[assignment]

    run_dir = new_run_dir(out_root, run_id=run_id)
    write_env_report(run_dir)
    save_effective_config(run_dir, {"dataset": ds_cfg, "awq": q_cfg, "model_dir": str(model_dir)})

    from .quantize_awq import run_awq_quantization

    out_dir = run_dir / "quantized-awq"
    run_awq_quantization(
        run_dir=run_dir, in_model_dir=model_dir, out_dir=out_dir, dataset_cfg=ds_cfg, cfg=q_cfg
    )
    print(f"[green]Done.[/green] Quantized model at: {out_dir}")


@quant_app.command("bnb")
def quantize_bnb(
    model_id_or_dir: str = typer.Option(..., help="HF model id or local dir."),
    out_dir: Path = typer.Option(Path("output/bnb"), help="Output metadata directory."),
):
    """
    Bitsandbytes quantization is primarily a *load-time* optimization.

    This command writes a small metadata bundle describing how to load the model with 4-bit/8-bit bnb.
    """
    from .quantize_bnb import write_bnb_bundle

    write_bnb_bundle(model_id_or_dir=model_id_or_dir, out_dir=out_dir)
    print(f"[green]Wrote[/green] bnb load bundle to {out_dir}")


@prune_app.command("mask-mlp")
def prune_mask_mlp(
    model_dir: Path = typer.Option(..., help="Input model dir (HF)."),
    run_id: Optional[str] = typer.Option(None, help="Run ID."),
    out_root: Path = typer.Option(Path("output/runs"), help="Root output directory."),
    ratio: float = typer.Option(0.25, help="Fraction of MLP neurons to mask per layer."),
    method: str = typer.Option("magnitude", help="magnitude|wanda"),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file."),
):
    """
    Structured MLP neuron *masking* for Llama-like models.

    Notes: true shape-changing pruning is not generally compatible with standard HF configs
    (intermediate size is usually fixed across layers). Masking keeps shapes but removes capacity.
    """
    if config is not None:
        blob = load_config_file(config)
        prune_blob = blob.get("prune", {})
        ratio = float(prune_blob.get("ratio", ratio))
        method = str(prune_blob.get("method", method))

    run_dir = new_run_dir(out_root, run_id=run_id)
    write_env_report(run_dir)
    save_effective_config(
        run_dir, {"prune": {"ratio": ratio, "method": method}, "model_dir": str(model_dir)}
    )

    from .prune import run_mlp_mask_prune

    out_dir = run_dir / "pruned"
    run_mlp_mask_prune(in_model_dir=model_dir, out_dir=out_dir, ratio=ratio, method=method)
    print(f"[green]Done.[/green] Pruned model at: {out_dir}")


@finetune_app.command("run")
def finetune_run(
    model_dir: Path = typer.Option(..., help="Input model dir (HF)."),
    run_id: Optional[str] = typer.Option(None, help="Run ID."),
    out_root: Path = typer.Option(Path("output/runs"), help="Root output directory."),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file."),
):
    """
    Post-prune recovery fine-tuning (SFT-style LM training) on a code dataset.
    """
    ds_cfg = DatasetConfig()
    # reuse distill config for basic training knobs (teacher ignored)
    ft_cfg = DistillConfig(teacher_model="", alpha=0.0, temperature=1.0)
    if config is not None:
        blob = load_config_file(config)
        ds_cfg = _merge_dict_into_dataclass(ds_cfg, blob.get("dataset", {}))  # type: ignore[assignment]
        ft_cfg = _merge_dict_into_dataclass(ft_cfg, blob.get("finetune", blob.get("distill", {})))  # type: ignore[assignment]

    run_dir = new_run_dir(out_root, run_id=run_id)
    write_env_report(run_dir)
    save_effective_config(
        run_dir, {"dataset": ds_cfg, "finetune": ft_cfg, "model_dir": str(model_dir)}
    )

    from .finetune import run_finetune

    out_dir = run_dir / "finetuned"
    run_finetune(
        run_dir=run_dir, in_model_dir=model_dir, out_dir=out_dir, dataset_cfg=ds_cfg, cfg=ft_cfg
    )
    print(f"[green]Done.[/green] Fine-tuned model at: {out_dir}")


@eval_app.command("run")
def evaluate_run(
    model_dir: Path = typer.Option(..., help="Model directory (HF)."),
    out_path: Optional[Path] = typer.Option(None, help="Write JSON metrics to this path."),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file (optional)."),
):
    from .evaluate import evaluate_model_dir

    _ = config  # reserved for future evaluation config
    res = evaluate_model_dir(model_dir, out_path=out_path)
    print(res)


@export_app.command("bundle")
def export_bundle(
    model_dir: Path = typer.Option(..., help="Model directory to serve/export."),
    out_dir: Path = typer.Option(Path("output/export"), help="Where to write export bundle."),
    model_name: str = typer.Option("codellama-compressed", help="Name used in scripts/Modelfile."),
    port: int = typer.Option(8000, help="Default server port."),
    config: Optional[Path] = typer.Option(None, help="YAML/JSON config file (optional)."),
):
    """
    Generate deployment helper artifacts (vLLM script, Dockerfile, GGUF conversion script, Ollama Modelfile).
    """
    from .export import write_export_bundle

    if config is not None:
        blob = load_config_file(config)
        exp = blob.get("export", {})
        model_name = str(exp.get("model_name", model_name))
        port = int(exp.get("port", port))

    write_export_bundle(model_dir=model_dir, out_dir=out_dir, model_name=model_name, port=port)
    print(f"[green]Wrote[/green] export bundle to {out_dir}")


@util_app.command("env-report")
def util_env_report(
    run_dir: Path = typer.Option(Path("output/runs/_env_report"), help="Output directory."),
):
    write_env_report(run_dir)
    print(f"[green]Wrote[/green] env report to {run_dir}")


@util_app.command("verify-artifact")
def util_verify_artifact(
    model_dir: Path = typer.Option(..., help="Model directory to load."),
    prompt: str = typer.Option("def fibonacci(n):", help="Prompt for sanity generation."),
    max_new_tokens: int = typer.Option(64, help="Max new tokens."),
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.float16
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)


@util_app.command("speculative")
def util_speculative(
    target_model: str = typer.Option(..., help="Target model (HF id or local dir)."),
    draft_model: str = typer.Option(..., help="Draft model (HF id or local dir)."),
    prompt: str = typer.Option("def fibonacci(n):", help="Prompt."),
    max_new_tokens: int = typer.Option(256, help="Max new tokens."),
    num_speculative_tokens: int = typer.Option(5, help="Draft tokens per step."),
    allow_mismatched_tokenizers: bool = typer.Option(
        False, help="Allow draft/target tokenizer mismatch (unsafe)."
    ),
):
    """
    Run speculative decoding locally (draft proposes, target verifies).
    """
    from .speculative import speculative_generate

    if allow_mismatched_tokenizers:
        print(
            "[yellow]WARNING[/yellow]: --allow-mismatched-tokenizers is enabled. "
            "This can produce incorrect output if draft/target tokenizers differ."
        )

    text, stats = speculative_generate(
        prompt=prompt,
        target_model=target_model,
        draft_model=draft_model,
        max_new_tokens=max_new_tokens,
        num_speculative_tokens=num_speculative_tokens,
        allow_mismatched_tokenizers=allow_mismatched_tokenizers,
    )
    print(text)
    print(stats.to_dict())
