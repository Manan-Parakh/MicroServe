import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

from engine.engine import Engine

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 64

# Fixed, not sampled — real Poisson/ShareGPT-derived workload generation is Milestone 5's job.
FIXED_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain the difference between a process and a thread.",
    "Once upon a time,",
    "The three laws of thermodynamics are",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[idx]


def run_naive(model_id: str) -> dict:
    engine = Engine(model_id, max_new_tokens=MAX_NEW_TOKENS)

    latencies_s = []
    total_tokens = 0
    wall_start = time.perf_counter()
    for prompt in FIXED_PROMPTS:
        req_start = time.perf_counter()
        engine.add_request(prompt)
        result = engine.step()
        latencies_s.append(time.perf_counter() - req_start)
        total_tokens += len(result.output_token_ids)
    wall_time_s = time.perf_counter() - wall_start

    sorted_latencies = sorted(latencies_s)
    return {
        "engine": "naive",
        "model_id": model_id,
        "commit": _git_commit(),
        "gpu": _gpu_name(),
        "python": platform.python_version(),
        "num_prompts": len(FIXED_PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "workload": "fixed prompt list, sequential, one request at a time (no concurrency at M1)",
        "throughput_tokens_per_sec": total_tokens / wall_time_s,
        "latency_p50_s": _percentile(sorted_latencies, 50),
        "latency_p99_s": _percentile(sorted_latencies, 99),
        "latency_per_request_s": latencies_s,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["naive"], default="naive")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    results = run_naive(args.model_id)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("docs/benchmarks") / f"{timestamp}-m1-naive-baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
