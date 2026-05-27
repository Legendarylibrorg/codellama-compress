from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class SpeculativeStats:
    tokens_generated: int
    tokens_accepted: int
    acceptance_rate: float
    seconds: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_generated": self.tokens_generated,
            "tokens_accepted": self.tokens_accepted,
            "acceptance_rate": self.acceptance_rate,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
        }


def _device_map(device: Literal["auto", "cuda", "cpu"]) -> Any:
    if device == "auto":
        return "auto"
    if device == "cpu":
        return {"": "cpu"}
    return "auto"


def _model_device(model) -> torch.device:
    # Works for non-sharded models; for device_map="auto" models, parameters should still
    # have concrete devices (or at least the first parameter does).
    return next(model.parameters()).device


@torch.inference_mode()
def speculative_generate(
    *,
    prompt: str,
    target_model: str | Path,
    draft_model: str | Path,
    max_new_tokens: int = 256,
    num_speculative_tokens: int = 5,
    temperature: float = 0.0,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    allow_mismatched_tokenizers: bool = False,
) -> tuple[str, SpeculativeStats]:
    """
    Speculative decoding (Leviathan et al. 2022 style) with greedy/argmax verification.

    Implementation notes:
    - Draft proposes up to K tokens greedily.
    - Target runs once on the whole proposed span to verify how many match its greedy tokens.
    - If mismatch occurs, we take the target's greedy token at the mismatch position.
    - Repeats until max_new_tokens reached.
    """
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")

    tok = AutoTokenizer.from_pretrained(target_model, use_fast=True, trust_remote_code=False)
    draft_tok = AutoTokenizer.from_pretrained(draft_model, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if draft_tok.pad_token is None:
        draft_tok.pad_token = draft_tok.eos_token

    if not allow_mismatched_tokenizers:
        mismatch = []
        if tok.vocab_size != draft_tok.vocab_size:
            mismatch.append(f"vocab_size target={tok.vocab_size} draft={draft_tok.vocab_size}")
        if tok.eos_token_id != draft_tok.eos_token_id:
            mismatch.append(
                f"eos_token_id target={tok.eos_token_id} draft={draft_tok.eos_token_id}"
            )
        if tok.bos_token_id != draft_tok.bos_token_id:
            mismatch.append(
                f"bos_token_id target={tok.bos_token_id} draft={draft_tok.bos_token_id}"
            )
        if mismatch:
            raise ValueError(
                "Draft/target tokenizers appear incompatible: "
                + "; ".join(mismatch)
                + ". Pass allow_mismatched_tokenizers=True to override."
            )

    target = AutoModelForCausalLM.from_pretrained(
        target_model,
        device_map=_device_map(device),
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        draft_model,
        device_map=_device_map(device),
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )

    # Keep separate tensors per-model to avoid device_map mismatches.
    target_device = _model_device(target)
    draft_device = _model_device(draft)

    input_ids_t = tok(prompt, return_tensors="pt").input_ids.to(target_device)
    input_ids_d = input_ids_t.to(draft_device)

    tokens_generated = 0
    tokens_accepted = 0
    t0 = time.time()

    while tokens_generated < max_new_tokens:
        k = min(num_speculative_tokens, max_new_tokens - tokens_generated)

        # 1) Draft proposes k tokens greedily.
        draft_ids = input_ids_d
        proposed = []
        for _ in range(k):
            d_out = draft(draft_ids)
            next_logits = d_out.logits[:, -1, :]
            if temperature and temperature > 0:
                next_logits = next_logits / temperature
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            proposed.append(next_id)
            draft_ids = torch.cat([draft_ids, next_id.to(draft_ids.device)], dim=1)

        proposed_ids_d = torch.cat(proposed, dim=1) if proposed else input_ids_d[:, :0]
        proposed_ids_t = proposed_ids_d.to(target_device)
        candidate = torch.cat([input_ids_t, proposed_ids_t], dim=1)

        # 2) Target verifies in one forward pass; compare target greedy tokens on each proposed position.
        t_out = target(candidate)
        logits = t_out.logits  # [B, L, V]

        # logits at positions that predict the proposed tokens:
        # proposed token j is predicted at position (prefix_len + j - 1)
        prefix_len = input_ids_t.shape[1]
        accept = 0
        for j in range(k):
            pos = prefix_len + j - 1
            greedy = torch.argmax(logits[:, pos, :], dim=-1)
            if int(greedy.item()) == int(proposed_ids_t[0, j].item()):
                accept += 1
            else:
                break

        # 3) Accept matched tokens; then append either (a) one extra target token (if mismatch) or (b) one more target token after full accept.
        if accept > 0:
            input_ids_t = torch.cat([input_ids_t, proposed_ids_t[:, :accept]], dim=1)
            input_ids_d = input_ids_t.to(draft_device)
            tokens_generated += accept
            tokens_accepted += accept

        if tokens_generated >= max_new_tokens:
            break

        # Next token from target:
        # - if mismatch: use target greedy token at mismatch position
        # - if full accept: use target greedy token after the accepted span
        if accept < k:
            pos = prefix_len + accept - 1
        else:
            pos = prefix_len + k - 1
        next_id = torch.argmax(logits[:, pos, :], dim=-1, keepdim=True).to(target_device)
        input_ids_t = torch.cat([input_ids_t, next_id], dim=1)
        input_ids_d = input_ids_t.to(draft_device)
        tokens_generated += 1

    dt = time.time() - t0
    text = tok.decode(input_ids_t[0], skip_special_tokens=True)
    stats = SpeculativeStats(
        tokens_generated=tokens_generated,
        tokens_accepted=tokens_accepted,
        acceptance_rate=(tokens_accepted / max(1, tokens_generated)),
        seconds=dt,
        tokens_per_second=(tokens_generated / max(1e-9, dt)),
    )
    return text, stats
