# Code Llama 7B Compression & Optimization Pipeline

Comprehensive pipeline for compressing and optimizing Meta's Code Llama 7B model while maintaining code generation quality. Includes state-of-the-art inference optimizations.

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
- **Layer-wise Quantization** - Different bits for different layers
- **Calibration Dataset** - StarCoder data for calibration

### Inference Optimizations
- **Speculative Decoding** - 2-3x speedup with draft model
- **KV Cache Quantization** - ~50% VRAM reduction for KV cache
- **Flash Attention 2** - Memory-efficient attention kernels
- **Continuous Batching** - High throughput with vLLM
- **Attention Optimizations** - Sliding window, GQA, token merging

### Quality Evaluation
- **Perplexity** - Language modeling quality
- **HumanEval Benchmark** - Actual code execution tests
- **Pass@k** - HumanEval/MBPP integration
- **Inference Speed** - Tokens per second measurement
- **Per-stage Evaluation** - Track quality through pipeline

### Export & Deployment
- **Multiple GGUF Formats** - q2_k, q3_k_m, q4_k_m, q5_k_m, q8_0
- **ONNX** - For ONNX Runtime
- **vLLM Server** - High-throughput serving with OpenAI API
- **Docker Deployment** - Ready-to-use containers
- **Ollama Integration** - Modelfile for easy deployment
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
| **Advanced Optimizations** | | |
| `USE_FLASH_ATTENTION` | `true` | Enable Flash Attention 2 |
| `USE_KV_CACHE_QUANT` | `true` | Enable KV cache quantization |
| `SPECULATIVE_DRAFT_MODEL` | `codellama/CodeLlama-7b-hf` | Draft model for speculative decoding |
| `SPECULATIVE_NUM_TOKENS` | `5` | Tokens to speculate |
| `GGUF_QUANTS` | `q2_k,q3_k_m,q4_k_m,q5_k_m,q8_0` | GGUF quantization levels |
| `USE_CONTINUOUS_BATCHING` | `true` | Enable vLLM batching |
| `MAX_BATCH_SIZE` | `8` | Maximum batch size |

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
├── eval/
│   ├── baseline_metrics.json        # Original model metrics
│   ├── distillation_metrics.json    # Post-distillation metrics
│   ├── pruning_metrics.json         # Post-pruning metrics
│   ├── finetuning_metrics.json      # Post-finetuning metrics
│   ├── quantization_metrics.json    # Quantization metrics
│   ├── lora_compatibility.json      # LoRA support verification
│   ├── speculative_decoding.json    # Speculative decoding stats
│   ├── kv_cache_savings.json        # KV cache memory savings
│   ├── humaneval_results.json       # HumanEval benchmark results
│   ├── gguf_estimates.json          # GGUF size estimates
│   ├── flash_attention_benchmark.json
│   ├── deployment_info.json         # Deployment configuration
│   ├── layerwise_quant_configs.json # Layer-wise quant options
│   ├── attention_savings.json       # Attention optimization stats
│   └── pipeline_summary.json        # Comprehensive summary
├── distilled/              # Distilled model
├── pruned/                 # Pruned model
├── finetuned/              # Fine-tuned model
├── quantized/              # Final quantized model
└── export/
    ├── speculative_decoding.py      # Speculative decoding module
    ├── kv_cache_quant.py            # KV cache quantization
    ├── optimized_inference.py       # Optimized inference wrapper
    ├── vllm_server.sh               # vLLM server script
    ├── vllm_inference.py            # vLLM Python module
    ├── convert_gguf.sh              # GGUF conversion script
    ├── Modelfile                    # Ollama Modelfile
    ├── Dockerfile                   # Docker deployment
    ├── docker-compose.yml           # Docker Compose config
    ├── layerwise_quant.py           # Layer-wise quantization
    ├── attention_optimization.py    # Attention optimizations
    └── README.md                    # Export instructions
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
from output.export.speculative_decoding import SpeculativeDecoder

decoder = SpeculativeDecoder(
    target_model_path="./output/quantized",
    draft_model_path="codellama/CodeLlama-7b-hf",
    num_speculative_tokens=5,
)

output, stats = decoder.generate("def fibonacci(n):", max_new_tokens=256)
print(f"Speed: {stats['tokens_per_second']:.1f} tok/s")
print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
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
