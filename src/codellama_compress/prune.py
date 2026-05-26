from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .replay import apply_global_seeds
from .security import resolve_user_path


def _iter_llama_mlp_modules(model):
    # Llama-like: model.model.layers[i].mlp.{gate_proj,up_proj,down_proj}
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        return
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if gate is None or up is None or down is None:
            continue
        yield i, mlp, gate, up, down


@torch.no_grad()
def run_mlp_mask_prune(
    *,
    in_model_dir: Path,
    out_dir: Path,
    ratio: float = 0.25,
    method: Literal["magnitude", "wanda"] = "magnitude",
    seed: int = 42,
) -> None:
    """
    Mask (zero) a fraction of intermediate neurons in Llama MLPs.

    This keeps model shapes compatible with HF configs, but reduces effective capacity.
    """
    apply_global_seeds(seed)
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must be in (0, 1)")

    out_dir.mkdir(parents=True, exist_ok=True)
    in_model_dir = resolve_user_path(in_model_dir, must_exist=True)

    tok = AutoTokenizer.from_pretrained(in_model_dir, use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        in_model_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=False,
    )
    model.eval()

    pruned_layers = 0
    total_masked = 0

    for _layer_idx, _mlp, gate, up, down in _iter_llama_mlp_modules(model):
        # gate/up weights: [intermediate, hidden]
        gate_w = gate.weight.data
        up_w = up.weight.data
        inter = gate_w.shape[0]
        k = int(inter * (1.0 - ratio))
        k = max(1, min(inter, k))

        if method == "wanda":
            # Approximated WANDA-like score using weight magnitude * row-norm.
            score = (gate_w.abs().sum(dim=1) * gate_w.norm(dim=1)) + (
                up_w.abs().sum(dim=1) * up_w.norm(dim=1)
            )
        else:
            score = gate_w.abs().sum(dim=1) + up_w.abs().sum(dim=1)

        keep_idx = torch.topk(score, k=k, largest=True).indices
        keep_mask = torch.zeros(inter, dtype=gate_w.dtype)
        keep_mask[keep_idx] = 1.0

        # Apply row mask to gate/up and column mask to down.
        gate.weight.data.mul_(keep_mask[:, None])
        up.weight.data.mul_(keep_mask[:, None])
        down.weight.data.mul_(keep_mask[None, :])

        pruned_layers += 1
        total_masked += int(inter - k)

    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    (out_dir / "prune_info.json").write_text(
        "{\n"
        f'  "ratio": {ratio},\n'
        f'  "method": "{method}",\n'
        f'  "layers_touched": {pruned_layers},\n'
        f'  "total_neurons_masked": {total_masked}\n'
        "}\n"
    )
