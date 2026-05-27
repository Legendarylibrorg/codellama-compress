# Code Llama 7B Compression & Optimization Pipeline

Comprehensive pipeline for compressing and optimizing Meta's Code Llama 7B model while maintaining code generation quality. Includes state-of-the-art inference optimizations.

## Linux-first quickstart

```bash
git clone https://github.com/Legendarylibrorg/codellama-compress.git
cd codellama-compress

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

Notes:
- `pyproject.toml` is the dependency source of truth.
- `requirements.txt` is auto-generated (see `scripts/export_requirements.py`).
- Only run models/repos you trust. `trust_remote_code` requires config **and** `CODELLAMA_COMPRESS_TRUST_REMOTE_CODE=1`.
- Training datasets are allowlisted by default; see `SECURITY.md` for `CODELLAMA_COMPRESS_DATASET_ALLOWLIST_EXTRA`.
- `evaluate code` requires a container/CI environment **or** `--allow-insecure-code-exec` plus `CODELLAMA_COMPRESS_ALLOW_CODE_EXEC=1`.
- Env reports are off by default; pass `--env-report` when you need `pip freeze` / `nvidia-smi` in a run directory.
- Architecture overview: `docs/architecture.md`.
- This CLI now uses Python stdlib `argparse` (no Typer/Rich/PyYAML/Numpy/Scipy).

Run the pipeline (basic):

```bash
# Distill (teacher -> student)
codellama-compress distill run

# Get latest run directory (used below)
RUN_DIR="$(ls -1dt output/runs/* | head -n 1)"

# Prune (MLP masking)
codellama-compress prune mask-mlp --model-dir "$RUN_DIR/distilled" --ratio 0.25 --method magnitude

# Finetune (post-prune recovery)
codellama-compress finetune run --model-dir "$RUN_DIR/pruned"

# Evaluate (smoke check)
codellama-compress evaluate run --model-dir "$RUN_DIR/finetuned"

# Export bundle (vLLM/Docker/GGUF helper scripts)
codellama-compress export bundle --model-dir "$RUN_DIR/finetuned" --out-dir output/export
```

Tip: the long-running commands include disk safety guards (see `--help` for `--min-free-gb` and `--max-run-dir-gb`).

Optional: quantization (requires extra dependencies):

```bash
pip install ".[quant]"
codellama-compress quantize gptq --model-dir "$RUN_DIR/finetuned"
codellama-compress quantize awq --model-dir "$RUN_DIR/finetuned"
```

Optional: research benchmarks (full suite; largest dependency footprint):

```bash
pip install ".[eval]"
codellama-compress evaluate benchmark --model-dir "$RUN_DIR/finetuned" --tasks humaneval,mbpp
```

Optional: code benchmarks with execution (Linux-first; runs generated code):

```bash
# In Docker/CI, or on host with explicit ack:
export CODELLAMA_COMPRESS_ALLOW_CODE_EXEC=1
codellama-compress evaluate code --model-dir "$RUN_DIR/finetuned" --suite humaneval --k 10 \
  --allow-insecure-code-exec
codellama-compress evaluate code --model-dir "$RUN_DIR/finetuned" --suite mbpp --k 10 \
  --allow-insecure-code-exec
```

Start a vLLM server (after exporting) and query it:

```bash
bash ./output/export/vllm_server.sh

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"codellama","prompt":"def fibonacci(n):","max_tokens":100}'
```

## Development

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt

pre-commit install
ruff check .
black --check .
pytest -q
```

## Features

### Implemented

- **Distillation**: teacher → student KL distillation (`codellama-compress distill run`)
- **Pruning (masking)**: shape-preserving MLP neuron masking (`codellama-compress prune mask-mlp`)
- **Recovery fine-tune**: post-prune fine-tuning (`codellama-compress finetune run`)
- **Evaluation (smoke check)**: small perplexity + speed sanity check (`codellama-compress evaluate run`)
- **Speculative decoding**: draft proposes, target verifies (`codellama-compress util speculative`)
- **Export bundle**: generates vLLM/Docker/GGUF helper scripts (`codellama-compress export bundle`)

### Optional (extras)

- **GPTQ quantization**: install `pip install ".[quant]"`, then `codellama-compress quantize gptq ...`
- **AWQ quantization**: install `pip install ".[quant]"`, then `codellama-compress quantize awq ...`
- **bitsandbytes bundle**: `codellama-compress quantize bnb ...` (load-time quantization)

### Planned

- KV cache quantization
- Flash-Attn integration toggles
- HumanEval/MBPP pass@k evaluation
- ONNX export

## Quick Start

```bash
See **Linux-first quickstart** above.
```

For Linux + RTX 4090 guidance, see `docs/linux-4090.md`.

## Configuration

Prefer using a config file (JSON) and overriding with CLI flags. A full config system is supported by the CLI commands.

`run.sh` is kept as a thin wrapper for convenience; the source of truth is the Python CLI.

Key knobs live in the CLI dataclasses:
- `DatasetConfig` in `src/codellama_compress/config.py`
- `DistillConfig` in `src/codellama_compress/config.py` (also reused for fine-tune)
- `GPTQConfig` in `src/codellama_compress/config.py` (also reused for AWQ calibration knobs)
- `DeterminismConfig` in `src/codellama_compress/config.py` (seeds, hash-based run ids, replay)

### Determinism and hash-based replay

By default, pipeline commands use `--deterministic` and `--hash-run-id`:

- **Same config → same run directory** under `output/runs/` (id like `r<16-char-hex>` from a pipeline fingerprint).
- Each stage records **SHA-256 content hashes** in `manifest.json` and `artifacts.jsonl`.
- **Replay a later stage** against a prior run: pass `--replay-from output/runs/<prior_run>` so input model dirs are hash-checked.
- **Verify** a run: `codellama-compress util manifest-verify --run-dir output/runs/<run_id>`
- **Backfill** manifests for older runs: `codellama-compress util manifest-create --run-dir ...`

Use `--no-hash-run-id` for timestamp-based run ids, or `--no-deterministic` only when you accept non-reproducible RNG.

## Pipeline Stages (conceptual)

This is a high-level roadmap/checklist. Not every stage below is implemented in this repo today—see **Features → Implemented** for what’s actually available.

```
1. Baseline Evaluation
   └── Measure original model quality
   └── 📊 Perplexity, speed, completions

2. Knowledge Distillation (teacher → student)
   └── Temperature-scaled soft labels (KL)
   └── 📊 EVALUATE: Compare to baseline

3. Structured Pruning (MLP masking)
   └── Target MLP layers (`gate_proj`, `up_proj`)
   └── Mask ~25% of intermediate neurons (shape-preserving)
   └── 📊 EVALUATE: Measure quality loss

4. Fine-tuning
   └── Recover quality on StarCoder data
   └── 📊 EVALUATE: Verify recovery

5. Quantization (AWQ/GPTQ)
   └── 4-bit quantization
   └── Calibrated on code samples
   └── 📊 EVALUATE: Final quality

6. LoRA Compatibility Test
   └── Verify adapter support

7. Speculative Decoding Setup
   └── Draft model configuration
   └── 2-3x inference speedup

8. KV Cache Quantization
   └── INT8 KV cache
   └── ~50% VRAM savings

9. HumanEval Benchmark
   └── Real code execution tests
   └── Pass@1 measurement

10. GGUF Multi-Quantization
    └── q2_k, q3_k_m, q4_k_m, q5_k_m, q8_0
    └── Ollama Modelfile

11. Flash Attention
    └── Memory-efficient attention
    └── 1.5-2x speedup

12. vLLM / Continuous Batching
    └── High-throughput serving
    └── Docker deployment

13. Advanced Optimizations
    └── Layer-wise quantization configs
    └── Attention pattern optimizations
```

## Output Structure

```
output/
└── runs/
    └── <run_id>/  # defaults to UTC timestamp (or pass --run-id)
        ├── config.json
        ├── env.json
        ├── pip_freeze.txt
        ├── nvidia-smi.txt              # Linux only
        ├── git_state.json
        ├── distilled/                  # HF model dir
        ├── pruned/                     # HF model dir
        ├── finetuned/                  # HF model dir
        ├── quantized-gptq/             # HF model dir (if run)
        └── quantized-awq/              # HF model dir (if run)
```

## Usage Examples

### Load Compressed Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./output/quantized",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./output/quantized")

prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2)
print(tokenizer.decode(outputs[0]))
```

### Speculative Decoding (2-3x faster)
```python
from codellama_compress.speculative import speculative_generate

output, stats = speculative_generate(
    prompt="def fibonacci(n):",
    target_model="./output/quantized",
    draft_model="codellama/CodeLlama-7b-hf",
    num_speculative_tokens=5,
    max_new_tokens=256,
)
print(f"Speed: {stats.tokens_per_second:.1f} tok/s")
print(f"Acceptance rate: {stats.acceptance_rate:.1%}")
```

### vLLM Serving (High Throughput)
```bash
# Using the generated script
bash ./output/export/vllm_server.sh

# Or manually
python -m vllm.entrypoints.openai.api_server \
    --model ./output/quantized \
    --port 8000 \
    --dtype float16

# Query the API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "codellama", "prompt": "def fibonacci(n):", "max_tokens": 100}'
```

### Docker Deployment
```bash
# Build image
docker build -t codellama-compressed -f ./output/export/Dockerfile .

# Run
docker run --gpus all -p 8000:8000 codellama-compressed

# Or use docker-compose
docker-compose -f ./output/export/docker-compose.yml up
```

### llama.cpp / Ollama (GGUF)
```bash
# Convert to multiple GGUF quantizations
bash ./output/export/convert_gguf.sh ./output/finetuned ./output/gguf "q4_k_m,q5_k_m,q8_0"

# Use with Ollama
cp ./output/export/Modelfile ./output/gguf/
cd ./output/gguf
ollama create codellama-compressed -f Modelfile
ollama run codellama-compressed

# Or use with llama.cpp directly
./llama.cpp/llama-cli -m ./output/gguf/model-q4_k_m.gguf -p "def fibonacci(n):" -n 100
```

### Optimized Inference (Flash Attention + torch.compile)
```python
from output.export.optimized_inference import OptimizedCodeLlama

model = OptimizedCodeLlama(
    model_path="./output/quantized",
    use_flash_attention=True,  # Requires Ampere+ GPU
)

# Single generation
output = model.generate("def quicksort(arr):", max_new_tokens=256)

# Batch generation
outputs = model.generate_batch(
    ["def fibonacci(n):", "class Stack:", "def binary_search(arr, target):"],
    max_new_tokens=128,
)

# Benchmark
stats = model.benchmark()
print(f"Speed: {stats['tokens_per_second']:.1f} tok/s")
```

### Add LoRA Adapter
```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("./output/quantized", ...)
model = PeftModel.from_pretrained(model, "path/to/lora")
```

## Expected Results

| Stage | Perplexity | Size | Tokens/s | Quality |
|-------|------------|------|----------|---------|
| Baseline (7B) | ~3.5 | 14GB | ~30 | 100% |
| Distilled | ~3.6 | 14GB | ~30 | ~98% |
| Pruned (25%) | ~4.2 | 11GB | ~35 | ~85% |
| Fine-tuned | ~3.8 | 11GB | ~35 | ~92% |
| Quantized (4-bit) | ~3.9 | 4GB | ~50 | ~90% |

### Inference Optimizations

| Optimization | Speedup | Memory Savings |
|--------------|---------|----------------|
| Flash Attention 2 | 1.5-2x | ~50% attention |
| Speculative Decoding | 2-3x | - |
| KV Cache INT8 | - | ~50% KV cache |
| vLLM Batching | 3-10x* | Optimized |
| torch.compile | 1.2-1.5x | - |

*Throughput increase for concurrent requests

### GGUF Sizes

| Format | Size | Quality |
|--------|------|---------|
| q8_0 | ~7GB | ~99% |
| q5_k_m | ~5GB | ~97% |
| q4_k_m | ~4GB | ~95% |
| q3_k_m | ~3GB | ~90% |
| q2_k | ~2.5GB | ~80% |

*Results vary based on hardware and settings*

## Quality Metrics

### Perplexity
Lower is better. Measures how well the model predicts code.

### HumanEval Pass@k
Percentage of coding problems solved correctly. The gold standard for code LLMs.
- **Pass@1**: First attempt success rate
- **Pass@10**: Success with 10 attempts

### Tokens/second
Inference throughput. Higher is better.

### KV Cache Memory
Memory used for storing attention keys/values. Important for long contexts.

## Requirements

- Python 3.10+
- CUDA GPU with 16GB+ VRAM (for distillation with 13B teacher)
- 8GB+ VRAM for inference/quantization
- ~50GB disk space

### Dependencies
```
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
datasets
peft
bitsandbytes
auto-gptq>=0.6.0
autoawq>=0.1.8
flash-attn (optional, requires Ampere+ GPU)
vllm (optional, for high-throughput serving)
llama-cpp-python (optional, for GGUF)
human-eval (optional, for benchmarks)
```

## Troubleshooting

### Out of Memory
- Reduce `DISTILL_STEPS` or use smaller batch size
- Use `QUANT_METHOD=bnb` for lower memory quantization
- Skip distillation if 13B teacher doesn't fit

### Slow Training
- Reduce `CALIBRATION_SAMPLES`
- Use streaming dataset loading (already enabled)

### Quality Too Low
- Reduce `PRUNE_RATIO` (try 0.15 instead of 0.25)
- Increase `FINETUNE_STEPS` (try 1000)
- Use 8-bit instead of 4-bit quantization

## References

- [Code Llama Paper](https://arxiv.org/abs/2308.12950)
- [WANDA Pruning](https://arxiv.org/abs/2306.11695)
- [AWQ Quantization](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [HumanEval Benchmark](https://arxiv.org/abs/2107.03374)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

This pipeline is MIT licensed. The Code Llama model has its own [license](https://github.com/facebookresearch/codellama/blob/main/LICENSE).
