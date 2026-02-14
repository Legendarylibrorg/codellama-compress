# Code Llama 7B Compression & Optimization Pipeline

Comprehensive pipeline for compressing and optimizing Meta's Code Llama 7B model while maintaining code generation quality.

## Features

### Knowledge Distillation
- **13B → 7B Transfer** - Learn from larger Code Llama 13B teacher
- **Temperature-scaled KL divergence** - Soft label distillation
- **EMA Weights** - Exponential moving average for stability
- **Cosine LR Schedule** - With warmup for smooth training

### Structured Pruning
- **WANDA Pruning** - Weight AND Activation based importance
- **Magnitude Pruning** - Simple L1-based pruning
- **MLP Layer Targeting** - Focus on gate_proj/up_proj layers
- **Configurable Ratio** - Default 25% reduction

### Quantization
- **AWQ** - Activation-aware Weight Quantization (4-bit)
- **GPTQ** - Post-training quantization with calibration
- **bitsandbytes** - NF4/INT8 quantization fallback
- **Calibration Dataset** - StarCoder data for calibration

### Quality Evaluation
- **Perplexity** - Language modeling quality
- **Pass@k Ready** - HumanEval/MBPP integration
- **Inference Speed** - Tokens per second measurement
- **Per-stage Evaluation** - Track quality through pipeline

### Export & Deployment
- **GGUF** - For llama.cpp deployment
- **ONNX** - For ONNX Runtime
- **vLLM** - High-throughput serving
- **LoRA Compatibility** - Verified adapter support

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

## Configuration

Edit `run.sh` to adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| **Model** | | |
| `BASE_MODEL` | `codellama/CodeLlama-7b-hf` | Base model to compress |
| `TEACHER_MODEL` | `codellama/CodeLlama-13b-hf` | Teacher for distillation |
| `USE_INSTRUCT` | `true` | Use instruct variant |
| **Distillation** | | |
| `DISTILL_STEPS` | `1000` | Distillation training steps |
| `DISTILL_LR` | `2e-5` | Learning rate |
| `EMA_DECAY` | `0.9999` | EMA decay rate |
| **Pruning** | | |
| `PRUNE_RATIO` | `0.25` | Fraction of neurons to remove |
| `PRUNE_METHOD` | `wanda` | wanda, magnitude, or sparsegpt |
| **Quantization** | | |
| `QUANT_BITS` | `4` | 4 or 8 bit quantization |
| `QUANT_METHOD` | `awq` | awq, gptq, or bnb |
| `CALIBRATION_SAMPLES` | `128` | Samples for calibration |
| **Fine-tuning** | | |
| `FINETUNE_STEPS` | `500` | Post-pruning recovery steps |
| `FINETUNE_LR` | `1e-5` | Fine-tuning learning rate |
| **Quality** | | |
| `MIN_PASS_AT_1_RETENTION` | `0.85` | Minimum Pass@1 retention |
| `MAX_PERPLEXITY_INCREASE` | `1.5` | Maximum perplexity ratio |

## Pipeline Stages

```
1. Baseline Evaluation
   └── Measure original model quality
   └── 📊 Perplexity, speed, completions

2. Knowledge Distillation (13B → 7B)
   └── Temperature-scaled soft labels
   └── EMA weight averaging
   └── 📊 EVALUATE: Compare to baseline

3. Structured Pruning (WANDA)
   └── Target MLP layers (gate_proj, up_proj)
   └── Remove 25% of neurons
   └── 📊 EVALUATE: Measure quality loss

4. Fine-tuning
   └── Recover quality on StarCoder data
   └── 📊 EVALUATE: Verify recovery

5. Quantization (AWQ/GPTQ)
   └── 4-bit quantization
   └── Calibrated on code samples
   └── 📊 EVALUATE: Final quality

6. Export
   └── GGUF, ONNX, vLLM ready

7. LoRA Compatibility Test
   └── Verify adapter support
```

## Output Structure

```
output/
├── eval/
│   ├── baseline/           # Original model metrics
│   ├── distilled/          # Post-distillation metrics
│   ├── pruned/             # Post-pruning metrics
│   ├── finetuned/          # Post-finetuning metrics
│   ├── quantized/          # Final metrics
│   ├── full_report.json    # Comprehensive report
│   └── lora_compatibility.json
├── distilled/              # Distilled model
├── pruned/                 # Pruned model
├── finetuned/              # Fine-tuned model
├── quantized/              # Final quantized model
└── export/
    └── README.md           # Export instructions
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

### vLLM Serving
```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./output/quantized \
    --port 8000
```

### llama.cpp (GGUF)
```bash
# Convert to GGUF
python -m llama_cpp.convert ./output/quantized --outfile model.gguf

# Quantize further
./quantize model.gguf model-q4_k_m.gguf q4_k_m

# Run
./main -m model-q4_k_m.gguf -p "def fibonacci(n):" -n 100
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

*Results vary based on hardware and settings*

## Quality Metrics

### Perplexity
Lower is better. Measures how well the model predicts code.

### Pass@k (HumanEval)
Percentage of problems solved correctly. Standard benchmark for code LLMs.

### Tokens/second
Inference throughput. Higher is better.

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
human-eval (optional)
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

## License

This pipeline is MIT licensed. The Code Llama model has its own [license](https://github.com/facebookresearch/codellama/blob/main/LICENSE).
