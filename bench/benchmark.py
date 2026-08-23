import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

from engine.continuous_engine import ContinuousEngine
from engine.engine import Engine
from engine.static_engine import StaticEngine

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_BATCH_SIZE = 3

# Fixed, not sampled -- real Poisson/ShareGPT-derived workload generation is
# Milestone 5's job (ADR-0004). Deliberately MIXED max_new_tokens per prompt,
# not uniform: this is the one thing a uniform workload can't demonstrate --
# static batching only collapses when some requests finish long before
# others. All three engines run this exact same workload, so total tokens
# generated is identical regardless of scheduling strategy.
WORKLOAD: list[tuple[str, int]] = [
    ("The capital of France is", 8),
    ("def fibonacci(n):", 60),
    ("Explain the difference between a process and a thread.", 12),
    ("Once upon a time,", 50),
    ("The three laws of thermodynamics are", 10),
    ("Write a haiku about debugging.", 40),
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _common_header(engine_name: str, model_id: str) -> dict:
    return {
        "engine": engine_name,
        "model_id": model_id,
        "commit": _git_commit(),
        "gpu": _gpu_name(),
        "python": platform.python_version(),
        "num_prompts": len(WORKLOAD),
        "workload": "fixed prompt list, MIXED max_new_tokens per prompt (not Poisson -- M5's job)",
    }


def run_naive(model_id: str) -> dict:
    """No concurrency exists in the naive engine, so only throughput and
    request latency are measurable here -- same scope call as ADR-0004."""
    engine = Engine(model_id)

    latencies_s = []
    total_tokens = 0
    wall_start = time.perf_counter()
    for prompt, max_new_tokens in WORKLOAD:
        req_start = time.perf_counter()
        engine.add_request(prompt, max_new_tokens=max_new_tokens)
        result = engine.step()
        latencies_s.append(time.perf_counter() - req_start)
        total_tokens += len(result.output_token_ids)
    wall_time_s = time.perf_counter() - wall_start

    return {
        **_common_header("naive", model_id),
        "max_batch_size": 1,
        "throughput_tokens_per_sec": total_tokens / wall_time_s,
        "latency_p50_s": _percentile(latencies_s, 50),
        "latency_p99_s": _percentile(latencies_s, 99),
        "peak_concurrent_sequences": 1,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time_s,
    }


def run_scheduled(engine_cls, engine_name: str, model_id: str, max_batch_size: int) -> dict:
    """Shared driver for ContinuousEngine and StaticEngine -- both expose the
    same add_request/step()/running shape, so one loop instruments both.

    Per-sequence TTFT/TPOT are measured from the benchmark side by polling
    each Sequence's own output_token_ids after every step() call, since both
    engines mutate the same Sequence objects returned by add_request() in
    place (established in Phase 3's correctness tests) rather than replacing
    them.

    "Wasted slot-steps" is this project's concrete stand-in for the roadmap's
    "GPU idle time" metric: a step where an already-FINISHED sequence still
    occupies a running-batch slot (only possible for StaticEngine, which
    can't shrink a batch until every member is done) counts as one wasted
    slot-step. A wall-clock-based idle-time estimate would be noisier and
    less reproducible than this exact, deterministic count derived straight
    from each engine's own bookkeeping.
    """
    engine = engine_cls(model_id, max_batch_size=max_batch_size)

    seqs = [engine.add_request(prompt, max_new_tokens=n) for prompt, n in WORKLOAD]
    first_token_time: dict[int, float] = {}
    completion_time: dict[int, float] = {}
    peak_concurrent = 0
    wasted_slot_steps = 0
    total_slot_steps = 0

    wall_start = time.perf_counter()
    while not all(seq.is_finished for seq in seqs):
        running = engine.running
        peak_concurrent = max(peak_concurrent, len(running))
        wasted_slot_steps += sum(1 for seq in running if seq.is_finished)
        total_slot_steps += len(running)

        engine.step()

        now = time.perf_counter()
        for seq in seqs:
            if seq.output_token_ids and seq.seq_id not in first_token_time:
                first_token_time[seq.seq_id] = now
            if seq.is_finished and seq.seq_id not in completion_time:
                completion_time[seq.seq_id] = now
    wall_time_s = time.perf_counter() - wall_start

    ttft_s = [first_token_time[seq.seq_id] - wall_start for seq in seqs]
    tpot_s = [
        (completion_time[seq.seq_id] - first_token_time[seq.seq_id])
        / (len(seq.output_token_ids) - 1)
        for seq in seqs
        if len(seq.output_token_ids) > 1
    ]
    total_tokens = sum(len(seq.output_token_ids) for seq in seqs)

    return {
        **_common_header(engine_name, model_id),
        "max_batch_size": max_batch_size,
        "throughput_tokens_per_sec": total_tokens / wall_time_s,
        "ttft_p50_s": _percentile(ttft_s, 50),
        "ttft_p99_s": _percentile(ttft_s, 99),
        "tpot_p50_s": _percentile(tpot_s, 50),
        "tpot_p99_s": _percentile(tpot_s, 99),
        "peak_concurrent_sequences": peak_concurrent,
        "wasted_slot_steps": wasted_slot_steps,
        "total_slot_steps": total_slot_steps,
        "wasted_compute_fraction": wasted_slot_steps / total_slot_steps
        if total_slot_steps
        else 0.0,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time_s,
    }


def run_continuous(model_id: str, max_batch_size: int) -> dict:
    return run_scheduled(ContinuousEngine, "continuous", model_id, max_batch_size)


def run_static(model_id: str, max_batch_size: int) -> dict:
    return run_scheduled(StaticEngine, "static", model_id, max_batch_size)


RUNNERS = {
    "naive": lambda model_id, max_batch_size: run_naive(model_id),
    "continuous": run_continuous,
    "static": run_static,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=[*RUNNERS, "all"], default="all")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    args = parser.parse_args()

    engines_to_run = list(RUNNERS) if args.engine == "all" else [args.engine]
    results = [RUNNERS[name](args.model_id, args.max_batch_size) for name in engines_to_run]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("docs/benchmarks") / f"{timestamp}-m2-continuous-vs-static-vs-naive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
