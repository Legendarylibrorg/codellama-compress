# Linux + RTX 4090 Setup

This project is Linux-first and tested with an RTX 4090 (24GB). Distillation and quantization are GPU-heavy; start with conservative settings and scale up.

## 1) Prereqs

- Ubuntu (or similar Linux)
- NVIDIA driver installed (a recent production driver is recommended)
- A working GPU runtime:
  - `nvidia-smi` should show the 4090
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

## 2) Install

```bash
git clone https://github.com/Legendarylibrorg/codellama-compress.git
cd codellama-compress

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Known-good starting point (recommended)

This repo tracks dependencies via `pyproject.toml` (ranges). If you want fewer surprises on a 4090 box, start with:

- Python 3.10–3.11
- Recent NVIDIA driver
- PyTorch installed from the official PyTorch CUDA wheels (Linux)

Then install optional quantization deps only when you need them:

```bash
pip install -e ".[quant]"
```

Note: the CLI is stdlib-based (`argparse`) to reduce supply-chain risk.

## 3) Run (recommended)

Distill (teacher -> student):

```bash
codellama-compress distill run
```

Quantize (GPTQ):

```bash
RUN_DIR="$(ls -1dt output/runs/* | head -n 1)"
codellama-compress quantize gptq --model-dir "$RUN_DIR/distilled"
```

## 4) Practical 4090 notes

- **13B teacher + 7B student may not fit** in 24GB for full logits distillation depending on sequence length and settings. If you hit OOM:
  - reduce `seq_len` (e.g. 256–512)
  - increase gradient accumulation
  - switch to a smaller teacher
- **Precision**:
  - Prefer **bf16** on Ada (4090), otherwise fp16.

- **Guards (disk safety)**:
  - The CLI has disk-space guards to prevent accidental disk fill.
  - Override if needed:
    - `--min-free-gb`
    - `--max-run-dir-gb`

