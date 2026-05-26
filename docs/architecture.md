# Architecture

This repo is a **Linux-first** pipeline for compressing a code LLM via:

- Distillation (teacher → student, KL on logits)
- Shape-preserving MLP masking (“pruning”)
- Post-prune recovery fine-tuning
- Quantization (optional backends)
- Evaluation + export helpers

## Core entrypoint

The supported interface is the CLI:

```bash
codellama-compress --help
```

## Stages and artifacts

Runs are stored under:

```
output/runs/<run_id>/
```

`<run_id>` defaults to a **hash of the pipeline config** (`--hash-run-id`, on by default) so the same config reuses the same directory. Pass `--no-hash-run-id` for UTC timestamp ids, or set `--run-id` explicitly.

Each run writes `manifest.json` and `artifacts.jsonl` with SHA-256 fingerprints of stage outputs for replay verification (`codellama-compress util manifest-verify`).

Typical stage directories:

- `distilled/`: HF model directory saved after distillation
- `pruned/`: HF model directory after MLP masking
- `finetuned/`: HF model directory after recovery fine-tuning
- `quantized-gptq/`: optional GPTQ artifact directory
- `quantized-awq/`: optional AWQ artifact directory

## Optional dependencies

Some commands require extras:

- Dev tooling: `pip install -r requirements-dev.txt`
- `pip install ".[quant]"`: GPTQ/AWQ/bitsandbytes helpers

## Notes

- “Pruning” is implemented as **masking** to preserve HF model shapes.
- Export helpers (vLLM/Docker/GGUF) are generated scripts; GGUF conversion requires `llama.cpp`.

