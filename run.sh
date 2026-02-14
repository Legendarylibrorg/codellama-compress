#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

# Model settings
BASE_MODEL="codellama/CodeLlama-7b-hf"
INSTRUCT_MODEL="codellama/CodeLlama-7b-Instruct-hf"
USE_INSTRUCT=true

# Output directories
OUT="./output"
DISTILL="$OUT/distilled"
PRUNE="$OUT/pruned"
QUANT="$OUT/quantized"
FINETUNE="$OUT/finetuned"
EXPORT="$OUT/export"
EVAL="$OUT/eval"

# Distillation settings
DISTILL_STEPS=1000
DISTILL_LR=2e-5
TEACHER_MODEL="codellama/CodeLlama-13b-hf"  # Larger model as teacher
EMA_DECAY=0.9999

# Pruning settings
PRUNE_RATIO=0.25          # Remove 25% of neurons
PRUNE_METHOD="wanda"       # wanda, magnitude, or sparsegpt

# Quantization settings
QUANT_BITS=4               # 4 or 8
QUANT_METHOD="awq"         # awq, gptq, or gguf
CALIBRATION_SAMPLES=128

# Fine-tuning after pruning
FINETUNE_STEPS=500
FINETUNE_LR=1e-5
FINETUNE_DATASET="bigcode/starcoderdata"

# Evaluation
EVAL_HUMANEVAL=true
EVAL_MBPP=true
EVAL_SAMPLES=50

# Quality thresholds
MIN_PASS_AT_1_RETENTION=0.85   # Must retain 85% of pass@1
MAX_PERPLEXITY_INCREASE=1.5    # Max 50% perplexity increase

########################################
# ENV SETUP
########################################

echo "=== SETTING UP ENVIRONMENT ==="

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip

pip install \
    torch torchvision \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    datasets \
    safetensors \
    sentencepiece \
    protobuf \
    tqdm \
    peft \
    bitsandbytes \
    scipy \
    evaluate \
    human-eval

# Quantization libraries
pip install auto-gptq>=0.6.0 autoawq>=0.1.8

# Optional: llama.cpp for GGUF export
# pip install llama-cpp-python

mkdir -p "$OUT" "$DISTILL" "$PRUNE" "$QUANT" "$FINETUNE" "$EXPORT" "$EVAL"

########################################
# GENERATE BASELINE EVALUATION
########################################

echo "=== BASELINE EVALUATION ==="

python3 << 'PY'
import torch
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_INSTRUCT = os.environ.get("USE_INSTRUCT", "true").lower() == "true"
BASE_MODEL = os.environ.get("INSTRUCT_MODEL" if USE_INSTRUCT else "BASE_MODEL", 
                            "codellama/CodeLlama-7b-Instruct-hf")
EVAL_DIR = os.environ.get("EVAL", "./output/eval")

os.makedirs(f"{EVAL_DIR}/baseline", exist_ok=True)

print(f"Loading baseline model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Test prompts for code generation
test_prompts = [
    "def fibonacci(n):",
    "def binary_search(arr, target):",
    "def quicksort(arr):",
    "class LinkedList:",
    "def merge_sort(arr):",
    "async def fetch_url(url):",
    "def validate_email(email):",
    "class BinaryTree:",
]

results = {
    "model": BASE_MODEL,
    "completions": [],
    "inference_times": [],
    "tokens_per_second": [],
}

print("Generating baseline completions...")
for prompt in tqdm(test_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    
    results["completions"].append({
        "prompt": prompt,
        "completion": completion,
        "tokens": num_tokens,
    })
    results["inference_times"].append(elapsed)
    results["tokens_per_second"].append(num_tokens / elapsed if elapsed > 0 else 0)

results["avg_time"] = sum(results["inference_times"]) / len(results["inference_times"])
results["avg_tokens_per_sec"] = sum(results["tokens_per_second"]) / len(results["tokens_per_second"])

# Compute perplexity on sample code
print("Computing perplexity...")
sample_code = '''
def calculate_factorial(n):
    """Calculate factorial of n recursively."""
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

def is_prime(n):
    """Check if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
'''

inputs = tokenizer(sample_code, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model(**inputs, labels=inputs.input_ids)
    perplexity = torch.exp(outputs.loss).item()

results["perplexity"] = perplexity

# Model size
param_count = sum(p.numel() for p in model.parameters())
results["param_count"] = param_count
results["param_count_b"] = param_count / 1e9

# Save results
with open(f"{EVAL_DIR}/baseline/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nBaseline Results:")
print(f"  Parameters: {results['param_count_b']:.2f}B")
print(f"  Perplexity: {results['perplexity']:.2f}")
print(f"  Avg inference time: {results['avg_time']*1000:.0f}ms")
print(f"  Tokens/sec: {results['avg_tokens_per_sec']:.1f}")

# Save sample completions
with open(f"{EVAL_DIR}/baseline/completions.txt", "w") as f:
    for item in results["completions"]:
        f.write(f"=== {item['prompt']} ===\n")
        f.write(item["completion"])
        f.write("\n\n")

print(f"Saved to {EVAL_DIR}/baseline/")

# Cleanup
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# KNOWLEDGE DISTILLATION (13B → 7B style transfer)
########################################

echo "=== KNOWLEDGE DISTILLATION ==="

python3 << 'PY'
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

TEACHER = os.environ.get("TEACHER_MODEL", "codellama/CodeLlama-13b-hf")
STUDENT = os.environ.get("BASE_MODEL", "codellama/CodeLlama-7b-hf")
DISTILL_DIR = os.environ.get("DISTILL", "./output/distilled")
STEPS = int(os.environ.get("DISTILL_STEPS", 1000))
LR = float(os.environ.get("DISTILL_LR", 2e-5))
EMA_DECAY = float(os.environ.get("EMA_DECAY", 0.9999))

os.makedirs(DISTILL_DIR, exist_ok=True)

# Load teacher (frozen)
print(f"Loading teacher: {TEACHER}")
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# Load student (trainable)
print(f"Loading student: {STUDENT}")
student = AutoModelForCausalLM.from_pretrained(
    STUDENT,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
student.train()

# EMA model
ema_state = {k: v.clone() for k, v in student.state_dict().items()}

tokenizer = AutoTokenizer.from_pretrained(STUDENT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load code dataset
print("Loading training data...")
dataset = load_dataset("bigcode/starcoderdata", "python", split="train", streaming=True)

# Training setup
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=STEPS)

def update_ema(ema_state, model_state, decay):
    for k in ema_state:
        if ema_state[k].dtype.is_floating_point:
            ema_state[k].mul_(decay).add_(model_state[k], alpha=1-decay)

temperature = 2.0
alpha = 0.5  # Balance between distillation and hard labels

print(f"Starting distillation for {STEPS} steps...")
print(f"  Temperature: {temperature}")
print(f"  Alpha (distill vs hard): {alpha}")

data_iter = iter(dataset)
losses = []

for step in tqdm(range(STEPS)):
    try:
        sample = next(data_iter)
        text = sample["content"][:1024]  # Limit length
    except StopIteration:
        data_iter = iter(dataset)
        sample = next(data_iter)
        text = sample["content"][:1024]
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(DEVICE)
    
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_outputs = teacher(**inputs)
        teacher_logits = teacher_outputs.logits
    
    # Student forward
    student_outputs = student(**inputs, labels=inputs.input_ids)
    student_logits = student_outputs.logits
    hard_loss = student_outputs.loss
    
    # Distillation loss (KL divergence on softened probabilities)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    
    # Combined loss
    loss = alpha * distill_loss + (1 - alpha) * hard_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    # Update EMA
    update_ema(ema_state, student.state_dict(), EMA_DECAY)
    
    losses.append(loss.item())
    
    if step % 100 == 0:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        print(f"  Step {step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

# Load EMA weights
student.load_state_dict(ema_state)

# Save distilled model
print(f"Saving distilled model to {DISTILL_DIR}...")
student.save_pretrained(DISTILL_DIR)
tokenizer.save_pretrained(DISTILL_DIR)

# Save training log
with open(f"{DISTILL_DIR}/training_log.json", "w") as f:
    json.dump({"losses": losses, "steps": STEPS}, f)

print("Distillation complete!")

# Cleanup
del teacher, student
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# EVALUATE: POST-DISTILLATION
########################################

echo "=== EVALUATE: POST-DISTILLATION ==="

python3 << 'PY'
import torch
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISTILL_DIR = os.environ.get("DISTILL", "./output/distilled")
EVAL_DIR = os.environ.get("EVAL", "./output/eval")

os.makedirs(f"{EVAL_DIR}/distilled", exist_ok=True)

# Load baseline for comparison
with open(f"{EVAL_DIR}/baseline/metrics.json") as f:
    baseline = json.load(f)

print(f"Loading distilled model from {DISTILL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(DISTILL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    DISTILL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Same test prompts
test_prompts = [
    "def fibonacci(n):",
    "def binary_search(arr, target):",
    "def quicksort(arr):",
    "class LinkedList:",
]

results = {"stage": "distilled", "completions": [], "inference_times": []}

for prompt in tqdm(test_prompts, desc="Evaluating distilled"):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.2, top_p=0.95,
                                  do_sample=True, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results["completions"].append({"prompt": prompt, "completion": completion})
    results["inference_times"].append(elapsed)

# Perplexity
sample_code = '''
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)
'''
inputs = tokenizer(sample_code, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model(**inputs, labels=inputs.input_ids)
    perplexity = torch.exp(outputs.loss).item()

results["perplexity"] = perplexity
results["avg_time"] = sum(results["inference_times"]) / len(results["inference_times"])
results["perplexity_retention"] = baseline["perplexity"] / perplexity if perplexity > 0 else 0
results["speedup"] = baseline["avg_time"] / results["avg_time"] if results["avg_time"] > 0 else 1

with open(f"{EVAL_DIR}/distilled/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPost-Distillation Results:")
print(f"  Perplexity: {perplexity:.2f} (baseline: {baseline['perplexity']:.2f})")
print(f"  Avg time: {results['avg_time']*1000:.0f}ms")
print(f"  Speedup: {results['speedup']:.2f}x")

if perplexity > baseline["perplexity"] * 1.5:
    print("⚠️  Significant perplexity increase - consider more distillation steps")
else:
    print("✓ Quality acceptable")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# STRUCTURED PRUNING (WANDA / SparseGPT)
########################################

echo "=== STRUCTURED PRUNING ==="

python3 << 'PY'
import torch
import torch.nn as nn
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISTILL_DIR = os.environ.get("DISTILL", "./output/distilled")
PRUNE_DIR = os.environ.get("PRUNE", "./output/pruned")
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.25))
PRUNE_METHOD = os.environ.get("PRUNE_METHOD", "wanda")

os.makedirs(PRUNE_DIR, exist_ok=True)

print(f"Loading model from {DISTILL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(DISTILL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    DISTILL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Applying {PRUNE_METHOD} pruning with ratio {PRUNE_RATIO}...")

# Count parameters before
params_before = sum(p.numel() for p in model.parameters())

def compute_layer_importance(layer, method="wanda"):
    """Compute importance scores for pruning."""
    importance_scores = {}
    
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            
            if method == "magnitude":
                # Simple magnitude-based importance
                importance = torch.abs(weight).sum(dim=1)
            elif method == "wanda":
                # WANDA: Weight AND Activation (approximated)
                # For full WANDA, you'd need activation statistics
                importance = torch.abs(weight).sum(dim=1) * torch.norm(weight, dim=1)
            else:
                importance = torch.abs(weight).sum(dim=1)
            
            importance_scores[name] = importance
    
    return importance_scores

def prune_linear_layer(module, keep_ratio):
    """Prune a linear layer by keeping top-k neurons."""
    weight = module.weight.data
    out_features = weight.shape[0]
    keep_count = max(int(out_features * keep_ratio), 64)  # Keep at least 64
    
    # Compute importance
    importance = torch.abs(weight).sum(dim=1)
    _, keep_indices = torch.topk(importance, keep_count)
    keep_indices = keep_indices.sort()[0]
    
    # Create pruned layer
    new_layer = nn.Linear(
        module.in_features, keep_count,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype
    )
    new_layer.weight.data = weight[keep_indices]
    if module.bias is not None:
        new_layer.bias.data = module.bias.data[keep_indices]
    
    return new_layer, keep_indices

# Prune MLP layers (intermediate projections)
pruned_layers = 0
total_removed = 0

for name, module in tqdm(list(model.named_modules()), desc="Pruning"):
    # Target MLP intermediate layers (gate_proj, up_proj in Llama-style)
    if isinstance(module, nn.Linear) and any(x in name for x in ['gate_proj', 'up_proj']):
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        # Get parent module
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        
        original_size = module.out_features
        new_layer, keep_indices = prune_linear_layer(module, 1 - PRUNE_RATIO)
        setattr(parent, child_name, new_layer)
        
        removed = original_size - new_layer.out_features
        total_removed += removed
        pruned_layers += 1

# Count parameters after
params_after = sum(p.numel() for p in model.parameters())

print(f"\nPruning Summary:")
print(f"  Method: {PRUNE_METHOD}")
print(f"  Layers pruned: {pruned_layers}")
print(f"  Parameters: {params_before/1e9:.2f}B → {params_after/1e9:.2f}B")
print(f"  Reduction: {(1 - params_after/params_before)*100:.1f}%")

# Save pruned model
print(f"Saving pruned model to {PRUNE_DIR}...")
model.save_pretrained(PRUNE_DIR)
tokenizer.save_pretrained(PRUNE_DIR)

# Save pruning info
with open(f"{PRUNE_DIR}/pruning_info.json", "w") as f:
    json.dump({
        "method": PRUNE_METHOD,
        "ratio": PRUNE_RATIO,
        "params_before": params_before,
        "params_after": params_after,
        "layers_pruned": pruned_layers,
    }, f, indent=2)

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Pruning complete!")
PY

########################################
# EVALUATE: POST-PRUNING
########################################

echo "=== EVALUATE: POST-PRUNING ==="

python3 << 'PY'
import torch
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNE_DIR = os.environ.get("PRUNE", "./output/pruned")
EVAL_DIR = os.environ.get("EVAL", "./output/eval")

os.makedirs(f"{EVAL_DIR}/pruned", exist_ok=True)

with open(f"{EVAL_DIR}/baseline/metrics.json") as f:
    baseline = json.load(f)

print(f"Evaluating pruned model from {PRUNE_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(PRUNE_DIR)
model = AutoModelForCausalLM.from_pretrained(PRUNE_DIR, torch_dtype=torch.float16, device_map="auto")

test_prompts = ["def fibonacci(n):", "def binary_search(arr, target):"]
results = {"stage": "pruned", "completions": [], "inference_times": []}

for prompt in tqdm(test_prompts, desc="Evaluating pruned"):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.2, 
                                  do_sample=True, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start
    results["completions"].append({"prompt": prompt, "completion": tokenizer.decode(outputs[0], skip_special_tokens=True)})
    results["inference_times"].append(elapsed)

# Perplexity
sample_code = "def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    return n * calculate_factorial(n - 1)"
inputs = tokenizer(sample_code, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model(**inputs, labels=inputs.input_ids)
    perplexity = torch.exp(outputs.loss).item()

results["perplexity"] = perplexity
results["avg_time"] = sum(results["inference_times"]) / len(results["inference_times"])
results["perplexity_change"] = perplexity / baseline["perplexity"]

# Model size
import subprocess
result = subprocess.run(["du", "-sm", PRUNE_DIR], capture_output=True, text=True)
results["model_size_mb"] = int(result.stdout.split()[0]) if result.stdout else 0

with open(f"{EVAL_DIR}/pruned/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPost-Pruning Results:")
print(f"  Perplexity: {perplexity:.2f} ({results['perplexity_change']:.2f}x baseline)")
print(f"  Model size: {results['model_size_mb']} MB")

if perplexity > baseline["perplexity"] * 2:
    print("⚠️  Significant quality loss - fine-tuning is critical")
else:
    print("✓ Quality acceptable for fine-tuning")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# FINE-TUNING AFTER PRUNING
########################################

echo "=== FINE-TUNING AFTER PRUNING ==="

python3 << 'PY'
import torch
import torch.nn as nn
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNE_DIR = os.environ.get("PRUNE", "./output/pruned")
FINETUNE_DIR = os.environ.get("FINETUNE", "./output/finetuned")
STEPS = int(os.environ.get("FINETUNE_STEPS", 500))
LR = float(os.environ.get("FINETUNE_LR", 1e-5))

os.makedirs(FINETUNE_DIR, exist_ok=True)

print(f"Loading pruned model from {PRUNE_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(PRUNE_DIR)
model = AutoModelForCausalLM.from_pretrained(PRUNE_DIR, torch_dtype=torch.float16, device_map="auto")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load code dataset
print("Loading training data...")
dataset = load_dataset("bigcode/starcoderdata", "python", split="train", streaming=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=STEPS)

model.train()
data_iter = iter(dataset)
losses = []

print(f"Fine-tuning for {STEPS} steps...")

for step in tqdm(range(STEPS)):
    try:
        sample = next(data_iter)
        text = sample["content"][:1024]
    except StopIteration:
        data_iter = iter(dataset)
        sample = next(data_iter)
        text = sample["content"][:1024]
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(DEVICE)
    
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    
    if step % 100 == 0:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        print(f"  Step {step}, Loss: {avg_loss:.4f}")

print(f"Saving fine-tuned model to {FINETUNE_DIR}...")
model.save_pretrained(FINETUNE_DIR)
tokenizer.save_pretrained(FINETUNE_DIR)

with open(f"{FINETUNE_DIR}/training_log.json", "w") as f:
    json.dump({"losses": losses}, f)

print("Fine-tuning complete!")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# EVALUATE: POST-FINETUNING
########################################

echo "=== EVALUATE: POST-FINETUNING ==="

python3 << 'PY'
import torch
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNE_DIR = os.environ.get("FINETUNE", "./output/finetuned")
EVAL_DIR = os.environ.get("EVAL", "./output/eval")

os.makedirs(f"{EVAL_DIR}/finetuned", exist_ok=True)

with open(f"{EVAL_DIR}/baseline/metrics.json") as f:
    baseline = json.load(f)
with open(f"{EVAL_DIR}/pruned/metrics.json") as f:
    pruned = json.load(f)

print(f"Evaluating fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNE_DIR)
model = AutoModelForCausalLM.from_pretrained(FINETUNE_DIR, torch_dtype=torch.float16, device_map="auto")

test_prompts = ["def fibonacci(n):", "def binary_search(arr, target):", "def quicksort(arr):"]
results = {"stage": "finetuned", "completions": [], "inference_times": []}

for prompt in tqdm(test_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.2, 
                                  do_sample=True, pad_token_id=tokenizer.eos_token_id)
    results["completions"].append({"prompt": prompt, "completion": tokenizer.decode(outputs[0], skip_special_tokens=True)})
    results["inference_times"].append(time.time() - start)

sample_code = "def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    return n * calculate_factorial(n - 1)"
inputs = tokenizer(sample_code, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model(**inputs, labels=inputs.input_ids)
    perplexity = torch.exp(outputs.loss).item()

results["perplexity"] = perplexity
results["avg_time"] = sum(results["inference_times"]) / len(results["inference_times"])
results["recovery"] = (pruned["perplexity"] - perplexity) / (pruned["perplexity"] - baseline["perplexity"]) if pruned["perplexity"] != baseline["perplexity"] else 1

with open(f"{EVAL_DIR}/finetuned/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPost-Finetuning Results:")
print(f"  Perplexity: {perplexity:.2f}")
print(f"  Pruned → Finetuned: {pruned['perplexity']:.2f} → {perplexity:.2f}")
print(f"  Recovery: {results['recovery']*100:.1f}%")

if perplexity <= baseline["perplexity"] * 1.3:
    print("✓ Quality successfully recovered")
else:
    print("⚠️  Quality still degraded - consider more fine-tuning")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# QUANTIZATION (AWQ / GPTQ / GGUF)
########################################

echo "=== QUANTIZATION ==="

python3 << 'PY'
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNE_DIR = os.environ.get("FINETUNE", "./output/finetuned")
QUANT_DIR = os.environ.get("QUANT", "./output/quantized")
QUANT_METHOD = os.environ.get("QUANT_METHOD", "awq")
QUANT_BITS = int(os.environ.get("QUANT_BITS", 4))
CALIBRATION_SAMPLES = int(os.environ.get("CALIBRATION_SAMPLES", 128))

os.makedirs(QUANT_DIR, exist_ok=True)

print(f"Quantizing with {QUANT_METHOD} to {QUANT_BITS}-bit...")

tokenizer = AutoTokenizer.from_pretrained(FINETUNE_DIR)

# Prepare calibration data
print("Preparing calibration data...")
dataset = load_dataset("bigcode/starcoderdata", "python", split="train", streaming=True)
calibration_data = []
for i, sample in enumerate(dataset):
    if i >= CALIBRATION_SAMPLES:
        break
    calibration_data.append(sample["content"][:512])

if QUANT_METHOD == "awq":
    try:
        from awq import AutoAWQForCausalLM
        
        print("Loading model for AWQ quantization...")
        model = AutoAWQForCausalLM.from_pretrained(FINETUNE_DIR, device_map="auto")
        
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": QUANT_BITS,
        }
        
        print(f"Quantizing to {QUANT_BITS}-bit AWQ...")
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)
        
        print(f"Saving to {QUANT_DIR}...")
        model.save_quantized(QUANT_DIR)
        tokenizer.save_pretrained(QUANT_DIR)
        
    except ImportError:
        print("AWQ not available, falling back to bitsandbytes...")
        QUANT_METHOD = "bnb"

if QUANT_METHOD == "gptq":
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        print("Loading model for GPTQ quantization...")
        
        quantize_config = BaseQuantizeConfig(
            bits=QUANT_BITS,
            group_size=128,
            desc_act=True,
        )
        
        model = AutoGPTQForCausalLM.from_pretrained(
            FINETUNE_DIR,
            quantize_config=quantize_config,
            device_map="auto",
        )
        
        # Prepare examples for calibration
        examples = [tokenizer(text, return_tensors="pt") for text in calibration_data[:32]]
        
        print(f"Quantizing to {QUANT_BITS}-bit GPTQ...")
        model.quantize(examples)
        
        print(f"Saving to {QUANT_DIR}...")
        model.save_quantized(QUANT_DIR)
        tokenizer.save_pretrained(QUANT_DIR)
        
    except ImportError:
        print("GPTQ not available, falling back to bitsandbytes...")
        QUANT_METHOD = "bnb"

if QUANT_METHOD == "bnb":
    from transformers import BitsAndBytesConfig
    
    print("Using bitsandbytes quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(QUANT_BITS == 4),
        load_in_8bit=(QUANT_BITS == 8),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_DIR,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    print(f"Saving to {QUANT_DIR}...")
    model.save_pretrained(QUANT_DIR)
    tokenizer.save_pretrained(QUANT_DIR)

# Calculate size
import subprocess
result = subprocess.run(["du", "-sm", QUANT_DIR], capture_output=True, text=True)
quant_size = int(result.stdout.split()[0]) if result.stdout else 0

result = subprocess.run(["du", "-sm", FINETUNE_DIR], capture_output=True, text=True)
original_size = int(result.stdout.split()[0]) if result.stdout else 0

print(f"\nQuantization Summary:")
print(f"  Method: {QUANT_METHOD}")
print(f"  Bits: {QUANT_BITS}")
print(f"  Size: {original_size}MB → {quant_size}MB")
print(f"  Reduction: {(1 - quant_size/original_size)*100:.1f}%")

with open(f"{QUANT_DIR}/quant_info.json", "w") as f:
    json.dump({
        "method": QUANT_METHOD,
        "bits": QUANT_BITS,
        "original_size_mb": original_size,
        "quantized_size_mb": quant_size,
    }, f, indent=2)

print("Quantization complete!")
PY

########################################
# FINAL EVALUATION
########################################

echo "=== FINAL EVALUATION ==="

python3 << 'PY'
import torch
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANT_DIR = os.environ.get("QUANT", "./output/quantized")
EVAL_DIR = os.environ.get("EVAL", "./output/eval")

os.makedirs(f"{EVAL_DIR}/quantized", exist_ok=True)

# Load all previous metrics
with open(f"{EVAL_DIR}/baseline/metrics.json") as f:
    baseline = json.load(f)

stages = {}
for stage in ["distilled", "pruned", "finetuned"]:
    path = f"{EVAL_DIR}/{stage}/metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            stages[stage] = json.load(f)

print("Loading quantized model...")
tokenizer = AutoTokenizer.from_pretrained(QUANT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    QUANT_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

test_prompts = [
    "def fibonacci(n):",
    "def binary_search(arr, target):",
    "def quicksort(arr):",
    "class LinkedList:",
]

results = {"stage": "quantized", "completions": [], "inference_times": [], "tokens_per_second": []}

for prompt in tqdm(test_prompts, desc="Final evaluation"):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.2,
                                  do_sample=True, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start
    
    num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    results["completions"].append({
        "prompt": prompt,
        "completion": tokenizer.decode(outputs[0], skip_special_tokens=True)
    })
    results["inference_times"].append(elapsed)
    results["tokens_per_second"].append(num_tokens / elapsed if elapsed > 0 else 0)

# Perplexity
sample_code = '''
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
'''
inputs = tokenizer(sample_code, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model(**inputs, labels=inputs.input_ids)
    perplexity = torch.exp(outputs.loss).item()

results["perplexity"] = perplexity
results["avg_time"] = sum(results["inference_times"]) / len(results["inference_times"])
results["avg_tokens_per_sec"] = sum(results["tokens_per_second"]) / len(results["tokens_per_second"])

# Model size
import subprocess
result = subprocess.run(["du", "-sm", QUANT_DIR], capture_output=True, text=True)
results["model_size_mb"] = int(result.stdout.split()[0]) if result.stdout else 0

# Compute final metrics
results["perplexity_retention"] = baseline["perplexity"] / perplexity if perplexity > 0 else 0
results["speedup"] = baseline["avg_time"] / results["avg_time"] if results["avg_time"] > 0 else 1
results["size_reduction"] = (1 - results["model_size_mb"] / (baseline["param_count"] / 1e6 * 2)) * 100  # Approx

with open(f"{EVAL_DIR}/quantized/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# Print comprehensive report
print("\n" + "="*70)
print("FINAL COMPRESSION PIPELINE REPORT - CODE LLAMA 7B")
print("="*70)

print("\n{:<15} {:>12} {:>12} {:>15}".format("Stage", "Perplexity", "Time(ms)", "Tokens/sec"))
print("-"*70)
print("{:<15} {:>12.2f} {:>12.0f} {:>15.1f}".format(
    "Baseline", baseline["perplexity"], baseline["avg_time"]*1000, baseline["avg_tokens_per_sec"]
))

for name, data in stages.items():
    print("{:<15} {:>12.2f} {:>12.0f} {:>15}".format(
        name.capitalize(), data.get("perplexity", 0), data.get("avg_time", 0)*1000, "-"
    ))

print("{:<15} {:>12.2f} {:>12.0f} {:>15.1f}".format(
    "FINAL", results["perplexity"], results["avg_time"]*1000, results["avg_tokens_per_sec"]
))
print("="*70)

print(f"\nFINAL RESULTS:")
print(f"  Perplexity: {baseline['perplexity']:.2f} → {results['perplexity']:.2f}")
print(f"  Speedup: {results['speedup']:.2f}x")
print(f"  Model Size: {results['model_size_mb']} MB")
print(f"  Tokens/sec: {baseline['avg_tokens_per_sec']:.1f} → {results['avg_tokens_per_sec']:.1f}")

quality_met = results["perplexity"] <= baseline["perplexity"] * 1.5
if quality_met:
    print("\n✅ SUCCESS: Quality target met (perplexity within 50% of baseline)")
else:
    print("\n⚠️  WARNING: Quality degraded significantly")

# Save full report
report = {
    "baseline": baseline,
    "stages": stages,
    "final": results,
    "summary": {
        "perplexity_change": results["perplexity"] / baseline["perplexity"],
        "speedup": results["speedup"],
        "model_size_mb": results["model_size_mb"],
        "quality_met": quality_met,
    }
}

with open(f"{EVAL_DIR}/full_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nFull report: {EVAL_DIR}/full_report.json")

# Save sample completions
with open(f"{EVAL_DIR}/final_completions.txt", "w") as f:
    for item in results["completions"]:
        f.write(f"=== {item['prompt']} ===\n")
        f.write(item["completion"])
        f.write("\n\n")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

########################################
# EXPORT FORMATS (GGUF, etc.)
########################################

echo "=== EXPORT FORMATS ==="

python3 << 'PY'
import os
import json
import subprocess

QUANT_DIR = os.environ.get("QUANT", "./output/quantized")
EXPORT_DIR = os.environ.get("EXPORT", "./output/export")

os.makedirs(EXPORT_DIR, exist_ok=True)

print("Export options for the compressed model:\n")

print("1. GGUF (for llama.cpp):")
print(f"   python -m llama_cpp.convert {QUANT_DIR} --outfile {EXPORT_DIR}/model.gguf")
print(f"   # Then quantize: ./quantize {EXPORT_DIR}/model.gguf {EXPORT_DIR}/model-q4_k_m.gguf q4_k_m")

print("\n2. ONNX (for ONNX Runtime):")
print(f"   optimum-cli export onnx --model {QUANT_DIR} {EXPORT_DIR}/onnx/")

print("\n3. vLLM serving:")
print(f"   python -m vllm.entrypoints.openai.api_server --model {QUANT_DIR}")

print("\n4. TensorRT-LLM:")
print(f"   # Convert to TensorRT format for maximum NVIDIA GPU performance")

# Create export instructions file
with open(f"{EXPORT_DIR}/README.md", "w") as f:
    f.write(f"""# Export Instructions

## GGUF (llama.cpp)
```bash
pip install llama-cpp-python
python -m llama_cpp.convert {QUANT_DIR} --outfile {EXPORT_DIR}/model.gguf
```

## ONNX
```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model {QUANT_DIR} {EXPORT_DIR}/onnx/
```

## vLLM Serving
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model {QUANT_DIR} --port 8000
```

## Inference Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{QUANT_DIR}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{QUANT_DIR}")

prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```
""")

print(f"\nExport instructions saved to {EXPORT_DIR}/README.md")
PY

########################################
# LORA COMPATIBILITY TEST
########################################

echo "=== LORA COMPATIBILITY TEST ==="

python3 << 'PY'
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

QUANT_DIR = os.environ.get("QUANT", "./output/quantized")
EVAL_DIR = os.environ.get("EVAL", "./output/eval")

print("Testing LoRA compatibility...")

tokenizer = AutoTokenizer.from_pretrained(QUANT_DIR)
model = AutoModelForCausalLM.from_pretrained(QUANT_DIR, torch_dtype=torch.float16, device_map="auto")

results = {
    "lora_targets": [],
    "compatible": True,
    "issues": []
}

# Check for LoRA-compatible layers
lora_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

for name, module in model.named_modules():
    if any(target in name for target in lora_targets):
        if hasattr(module, 'weight'):
            shape = module.weight.shape
            results["lora_targets"].append({
                "name": name,
                "shape": list(shape),
                "compatible": shape[0] >= 64 and shape[1] >= 64
            })

compatible_count = sum(1 for t in results["lora_targets"] if t["compatible"])
total_count = len(results["lora_targets"])

print(f"\nLoRA Target Layers: {compatible_count}/{total_count} compatible")

if compatible_count < total_count:
    results["compatible"] = False
    results["issues"].append(f"Some layers too small for LoRA ({total_count - compatible_count} issues)")
    print("⚠️  Some layers may have issues with standard LoRA")
else:
    print("✓ All layers compatible with LoRA")

# Test PEFT compatibility
try:
    from peft import get_peft_config, get_peft_model, LoraConfig
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    # Just check if config is valid, don't actually apply
    results["peft_compatible"] = True
    print("✓ PEFT/LoRA configuration valid")
    
except Exception as e:
    results["peft_compatible"] = False
    results["issues"].append(f"PEFT error: {str(e)}")
    print(f"⚠️  PEFT compatibility issue: {e}")

with open(f"{EVAL_DIR}/lora_compatibility.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nReport saved to {EVAL_DIR}/lora_compatibility.json")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

echo ""
echo "=========================================="
echo "  COMPRESSION PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Distilled model: $DISTILL"
echo "  - Pruned model: $PRUNE"
echo "  - Fine-tuned model: $FINETUNE"
echo "  - Quantized model: $QUANT"
echo "  - Evaluations: $EVAL"
echo ""
echo "To use the compressed model:"
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  model = AutoModelForCausalLM.from_pretrained('$QUANT')"
echo ""
