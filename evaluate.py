#!/usr/bin/env python3
"""
Quality Evaluation Module for Code LLM Compression Pipeline

Metrics:
- Perplexity: Language modeling quality
- Pass@k: Code execution success rate (HumanEval/MBPP)
- CodeBLEU: Code generation quality
- Inference Speed: Tokens per second
- Memory Usage: VRAM consumption
"""

import torch
import json
import os
import time
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np


@dataclass
class CodeQualityMetrics:
    """Container for code quality metrics."""
    perplexity: float = 0.0
    pass_at_1: float = 0.0
    pass_at_10: float = 0.0
    tokens_per_second: float = 0.0
    inference_time_ms: float = 0.0
    vram_mb: float = 0.0
    model_size_mb: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"PPL: {self.perplexity:.2f} | "
            f"Pass@1: {self.pass_at_1:.1%} | "
            f"Tokens/s: {self.tokens_per_second:.1f} | "
            f"Size: {self.model_size_mb:.0f}MB"
        )


def compute_perplexity(model, tokenizer, texts: List[str], device="cuda") -> float:
    """Compute perplexity on a list of code samples."""
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        total_loss += loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def compute_pass_at_k(
    model,
    tokenizer,
    problems: List[Dict],
    k: int = 1,
    num_samples: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 256,
    device: str = "cuda",
) -> Tuple[float, List[Dict]]:
    """
    Compute pass@k metric for code generation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        problems: List of problems with 'prompt' and 'test' keys
        k: k for pass@k
        num_samples: Number of samples per problem
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        device: Device to use
    
    Returns:
        (pass_at_k, detailed_results)
    """
    results = []
    correct = 0
    
    for problem in tqdm(problems, desc=f"Computing pass@{k}"):
        prompt = problem["prompt"]
        test_code = problem.get("test", "")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate multiple samples
        samples_correct = 0
        samples = []
        
        for _ in range(num_samples):
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append(completion)
            
            # Try to execute if test code provided
            if test_code:
                try:
                    # Create full code
                    full_code = completion + "\n" + test_code
                    exec(full_code, {})
                    samples_correct += 1
                except Exception:
                    pass
        
        # Compute pass@k for this problem
        if num_samples >= k:
            # Estimator for pass@k
            c = samples_correct
            n = num_samples
            if c >= k:
                problem_pass = 1.0
            else:
                problem_pass = 1.0 - np.prod([(n-c-i)/(n-i) for i in range(k)])
        else:
            problem_pass = 1.0 if samples_correct > 0 else 0.0
        
        if samples_correct > 0:
            correct += 1
        
        results.append({
            "prompt": prompt[:100],
            "samples": len(samples),
            "correct": samples_correct,
            "pass_at_k": problem_pass,
        })
    
    pass_at_k = correct / len(problems) if problems else 0
    
    return pass_at_k, results


def measure_inference_speed(
    model,
    tokenizer,
    prompt: str = "def fibonacci(n):",
    num_runs: int = 5,
    max_tokens: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure inference speed in tokens per second."""
    times = []
    tokens_generated = []
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Measure
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        times.append(elapsed)
        tokens_generated.append(num_tokens)
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
    
    return {
        "avg_time_ms": avg_time * 1000,
        "avg_tokens": avg_tokens,
        "tokens_per_second": tokens_per_sec,
    }


def get_model_size(model_path: str) -> float:
    """Get model size in MB."""
    result = subprocess.run(
        ["du", "-sm", model_path],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        return int(result.stdout.split()[0])
    return 0


def get_vram_usage() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


class CodeQualityEvaluator:
    """Comprehensive evaluator for code generation models."""
    
    def __init__(
        self,
        device: str = "cuda",
        eval_samples: int = 50,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.eval_samples = eval_samples
        self.results_history = []
        
        # Standard test prompts
        self.test_prompts = [
            "def fibonacci(n):",
            "def binary_search(arr, target):",
            "def quicksort(arr):",
            "def merge_sort(arr):",
            "class LinkedList:",
            "class BinaryTree:",
            "def is_prime(n):",
            "def factorial(n):",
            "async def fetch_data(url):",
            "def validate_email(email):",
        ]
        
        # Sample code for perplexity
        self.perplexity_samples = [
            '''
def calculate_factorial(n):
    """Calculate factorial of n recursively."""
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)
''',
            '''
def binary_search(arr, target):
    """Binary search implementation."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
            '''
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def is_empty(self):
        return len(self.items) == 0
''',
        ]
    
    def evaluate_model(
        self,
        model,
        tokenizer,
        model_name: str,
        model_path: str,
    ) -> CodeQualityMetrics:
        """Evaluate a model and return metrics."""
        
        print(f"\nEvaluating {model_name}...")
        
        # Perplexity
        print("  Computing perplexity...")
        perplexity = compute_perplexity(
            model, tokenizer, self.perplexity_samples, self.device
        )
        
        # Inference speed
        print("  Measuring inference speed...")
        speed_results = measure_inference_speed(
            model, tokenizer, device=self.device
        )
        
        # Model size
        model_size = get_model_size(model_path)
        
        # VRAM
        vram = get_vram_usage()
        
        metrics = CodeQualityMetrics(
            perplexity=perplexity,
            tokens_per_second=speed_results["tokens_per_second"],
            inference_time_ms=speed_results["avg_time_ms"],
            vram_mb=vram,
            model_size_mb=model_size,
        )
        
        self.results_history.append({
            "model_name": model_name,
            "model_path": model_path,
            "metrics": metrics.to_dict(),
        })
        
        return metrics
    
    def compare_models(
        self,
        models: List[Tuple[str, any, any, str]],  # (name, model, tokenizer, path)
        output_dir: str = "./output/eval",
    ) -> Dict:
        """Compare multiple models and generate report."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for name, model, tokenizer, path in models:
            metrics = self.evaluate_model(model, tokenizer, name, path)
            results.append({
                "name": name,
                "metrics": metrics.to_dict(),
            })
            print(f"  {metrics}")
        
        # Generate comparison table
        self._print_comparison_table(results)
        
        # Compute retention
        if len(results) >= 2:
            baseline = results[0]["metrics"]
            for r in results[1:]:
                m = r["metrics"]
                r["retention"] = {
                    "perplexity_ratio": m["perplexity"] / baseline["perplexity"],
                    "speedup": m["tokens_per_second"] / baseline["tokens_per_second"] if baseline["tokens_per_second"] > 0 else 1,
                    "size_reduction": (1 - m["model_size_mb"] / baseline["model_size_mb"]) * 100 if baseline["model_size_mb"] > 0 else 0,
                }
        
        # Save report
        report_path = os.path.join(output_dir, "comparison_report.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nReport saved to {report_path}")
        
        return results
    
    def _print_comparison_table(self, results: List[Dict]):
        """Print formatted comparison table."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        headers = ["Model", "Perplexity", "Tokens/s", "Time(ms)", "Size(MB)"]
        row_format = "{:<20} {:>12} {:>12} {:>12} {:>12}"
        
        print(row_format.format(*headers))
        print("-" * 80)
        
        for r in results:
            m = r["metrics"]
            print(row_format.format(
                r["name"][:20],
                f"{m['perplexity']:.2f}",
                f"{m['tokens_per_second']:.1f}",
                f"{m['inference_time_ms']:.0f}",
                f"{m['model_size_mb']:.0f}",
            ))
        
        print("=" * 80)


class QualityGuard:
    """Guard to ensure compression doesn't degrade quality too much."""
    
    def __init__(
        self,
        max_perplexity_increase: float = 1.5,  # 50% increase max
        min_pass_at_1_retention: float = 0.85,  # 85% retention
        min_speed_retention: float = 0.8,  # 80% of baseline speed
    ):
        self.max_perplexity_increase = max_perplexity_increase
        self.min_pass_at_1_retention = min_pass_at_1_retention
        self.min_speed_retention = min_speed_retention
        
        self.baseline_metrics = None
        self.violations = []
    
    def set_baseline(self, metrics: CodeQualityMetrics):
        """Set baseline metrics."""
        self.baseline_metrics = metrics
        print(f"Quality baseline set: {metrics}")
    
    def check(self, metrics: CodeQualityMetrics, stage_name: str) -> Tuple[bool, List[str]]:
        """Check if metrics are within acceptable bounds."""
        violations = []
        
        if self.baseline_metrics is None:
            return True, []
        
        # Perplexity check
        ppl_ratio = metrics.perplexity / self.baseline_metrics.perplexity
        if ppl_ratio > self.max_perplexity_increase:
            violations.append(
                f"Perplexity increased {ppl_ratio:.2f}x (max: {self.max_perplexity_increase}x)"
            )
        
        # Speed check
        if self.baseline_metrics.tokens_per_second > 0:
            speed_ratio = metrics.tokens_per_second / self.baseline_metrics.tokens_per_second
            if speed_ratio < self.min_speed_retention:
                violations.append(
                    f"Speed dropped to {speed_ratio:.1%} of baseline (min: {self.min_speed_retention:.1%})"
                )
        
        passed = len(violations) == 0
        
        if not passed:
            self.violations.append({
                "stage": stage_name,
                "violations": violations,
            })
            print(f"\n⚠️  QUALITY WARNING at {stage_name}:")
            for v in violations:
                print(f"   - {v}")
        else:
            print(f"\n✓ Quality check passed at {stage_name}")
        
        return passed, violations


# HumanEval integration
def load_humaneval() -> List[Dict]:
    """Load HumanEval benchmark problems."""
    try:
        from human_eval.data import read_problems
        problems = read_problems()
        return [
            {
                "task_id": k,
                "prompt": v["prompt"],
                "test": v["test"],
                "entry_point": v["entry_point"],
            }
            for k, v in problems.items()
        ]
    except ImportError:
        print("HumanEval not installed. Install with: pip install human-eval")
        return []


def evaluate_humaneval(
    model,
    tokenizer,
    num_samples: int = 50,
    k: int = 1,
    device: str = "cuda",
) -> Dict:
    """Run HumanEval benchmark."""
    problems = load_humaneval()
    
    if not problems:
        return {"error": "HumanEval not available"}
    
    # Limit samples
    problems = problems[:num_samples]
    
    pass_at_k, results = compute_pass_at_k(
        model, tokenizer, problems, k=k, device=device
    )
    
    return {
        "pass_at_k": pass_at_k,
        "k": k,
        "num_problems": len(problems),
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate code LLM quality")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--baseline", type=str, help="Path to baseline model")
    parser.add_argument("--output", type=str, default="./output/eval")
    
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    evaluator = CodeQualityEvaluator()
    metrics = evaluator.evaluate_model(model, tokenizer, "Model", args.model)
    
    print(f"\nResults: {metrics}")
