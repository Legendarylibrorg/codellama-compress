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
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3) Run (recommended)

Distill (teacher -> student):

```bash
codellama-compress distill run
```

Quantize (GPTQ):

```bash
codellama-compress quantize gptq --model-dir output/runs/<run_id>/distilled
```

## 4) Practical 4090 notes

- **13B teacher + 7B student may not fit** in 24GB for full logits distillation depending on sequence length and settings. If you hit OOM:
  - reduce `seq_len` (e.g. 256–512)
  - increase gradient accumulation
  - switch to a smaller teacher
- **Precision**:
  - Prefer **bf16** on Ada (4090), otherwise fp16.

