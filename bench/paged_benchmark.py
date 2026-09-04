"""M3 Phase 6: paged vs. contiguous KV cache at equal VRAM.

The headline number here is **not** throughput. It is *how many sequences fit*
in a fixed KV budget. Per-sequence throughput may even dip, because the paged
path pays a gather per layer per step that the contiguous path does not;
system throughput should still rise because more sequences run at once.
Reporting only throughput would hide both halves of the story.

Holding VRAM equal
------------------
`ContinuousEngine` has no VRAM knob -- it takes `max_batch_size` and sizes its
`StaticCache` to `max(prompt + max_new_tokens)`. So the budget is converted
into each engine's own currency:

    per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * itemsize
    contiguous: max_batch_size = budget // (max_cache_len * per_token_bytes)
    paged:      num_blocks     = budget // (block_size    * per_token_bytes)

The contiguous line *is* the reserve-max rule, written down. Both derivations
are checked against the tensors the engines actually allocate
(`verify_contiguous_budget` / the pool's own `nbytes`), so the arithmetic is
validated rather than trusted.

Separate from `bench/benchmark.py` on purpose, for the same reason
`PagedScheduler` is separate from `Scheduler` (ADR-0013): M2's driver produced
numbers that are already recorded, and an already-measured instrument is closer
to a recorded artifact than to live code.

Utilization, defined precisely
------------------------------
`kv_utilization` = token slots holding a real token / token slots the engine
holds against the budget. For contiguous that denominator is the whole
`StaticCache` (`max_batch_size * max_cache_len`), because reserve-max claims it
all up front whether or not a token ever lands there. For paged it is the pool
(`num_blocks * block_size`). This is the paper's ~40%-vs->90% number computed
for our own engines, and it is the thing that explains the concurrency result
rather than merely restating it.
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from bench.benchmark import WORKLOAD, _percentile
from engine.continuous_engine import ContinuousEngine
from engine.kv_cache import BLOCK_SIZE, BlockExhausted, KVSpec, internal_fragmentation
from engine.paged_engine import PagedEngine

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Budgets chosen so the contiguous engine's reserve-max rule bites at the low
# end and relaxes at the high end. On the dev model one contiguous sequence
# costs max_cache_len * 12 KiB ~= 840 KiB, so 1 MiB fits exactly one.
DEFAULT_SWEEP_MIB = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


# --- provenance ---------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _hardware() -> dict:
    if not torch.cuda.is_available():
        return {"gpu": "CPU", "vram_total_gib": None, "vram_free_gib": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gib": round(total / 1024**3, 2),
        "vram_free_gib": round(free / 1024**3, 2),
    }


# --- turning a byte budget into each engine's own configuration ----------------


def kv_bytes_per_token(spec: KVSpec) -> int:
    """Bytes of KV one token position costs, across every layer.

    The 2 is K and V, not a batch dimension. On Qwen2.5-0.5B in bf16 this is
    2 * 24 * 2 * 64 * 2 = 12 KiB, which makes a 16-token block 192 KiB.
    """
    return 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * spec.dtype.itemsize


@dataclass
class BudgetPlan:
    """One budget, expressed in both engines' currencies."""

    budget_bytes: int
    per_token_bytes: int
    max_cache_len: int
    contiguous_max_batch: int
    paged_num_blocks: int
    contiguous_bytes: int
    paged_bytes: int
    contiguous_fits: bool
    paged_fits: bool


def plan_for_budget(budget_bytes: int, spec: KVSpec, max_cache_len: int) -> BudgetPlan:
    per_token = kv_bytes_per_token(spec)
    contiguous_max_batch = budget_bytes // (max_cache_len * per_token)
    paged_num_blocks = budget_bytes // (spec.block_size * per_token)
    return BudgetPlan(
        budget_bytes=budget_bytes,
        per_token_bytes=per_token,
        max_cache_len=max_cache_len,
        contiguous_max_batch=max(1, contiguous_max_batch),
        paged_num_blocks=max(1, paged_num_blocks),
        contiguous_bytes=max(1, contiguous_max_batch) * max_cache_len * per_token,
        paged_bytes=max(1, paged_num_blocks) * spec.block_size * per_token,
        # A budget that cannot hold even one reserve-max sequence, or one
        # sequence's worth of blocks, is over-subscribed. Recorded rather than
        # clamped silently -- an engine forced above its budget is not a
        # measurement of that budget.
        contiguous_fits=contiguous_max_batch >= 1,
        paged_fits=paged_num_blocks * spec.block_size >= max_cache_len,
    )


def measure_cache_bytes(engine: ContinuousEngine) -> int:
    """Bytes the engine's `StaticCache` currently holds, read off the tensors.

    The whole equal-VRAM claim rests on `contiguous_max_batch` being derived
    correctly, and it is arithmetic rather than a knob the engine exposes -- so
    it gets checked against the tensors that actually exist, sampled at the
    batch's peak, instead of trusted.
    """
    cache = getattr(engine.scheduler, "cache", None)
    layers = getattr(cache, "layers", None) if cache is not None else None
    if not layers:
        return 0
    return sum(
        t.numel() * t.element_size()
        for layer in layers
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None))
        if t is not None
    )


# --- the drivers ---------------------------------------------------------------


def _drive(engine, seqs, allocated_token_slots: int, paged: bool, probe=None) -> dict:
    """Run one engine to completion, sampling memory occupancy every step.

    Sampled per step rather than computed at the end because the interesting
    quantity is *occupancy over time*: a mean utilization of 33% and a peak of
    33% mean very different things, and reserve-max produces the second.
    """
    first_token_time: dict[int, float] = {}
    completion_time: dict[int, float] = {}
    peak_concurrent = 0
    used_slots_samples: list[int] = []
    reserved_slots_samples: list[int] = []
    frag_samples: list[int] = []
    probe_peak = 0

    wall_start = time.perf_counter()
    while not all(s.is_finished for s in seqs):
        running = engine.running
        peak_concurrent = max(peak_concurrent, len(running))
        if paged:
            used_slots_samples.append(sum(s.num_cached_tokens for s in running))
            # A paged sequence reserves only the blocks it owns, and the slack
            # is bounded by BLOCK_SIZE - 1 per sequence -- internal
            # fragmentation, and nothing else.
            reserved_slots_samples.append(sum(len(s.block_table) * BLOCK_SIZE for s in running))
            frag_samples.append(internal_fragmentation(running))
        else:
            used_slots_samples.append(
                sum(len(s.prompt_token_ids) + len(s.output_token_ids) for s in running)
            )
            # Reserve-max: every row of the rectangle is sized for the longest
            # sequence's full prompt + max_new_tokens, whether or not a token
            # ever lands there. This denominator is the entire M3 argument.
            if running:
                reserved = max(len(s.prompt_token_ids) + s.max_new_tokens for s in running) * len(
                    running
                )
            else:
                reserved = 0
            reserved_slots_samples.append(reserved)

        engine.step()
        if probe is not None:
            probe_peak = max(probe_peak, probe(engine))

        now = time.perf_counter()
        for s in seqs:
            if s.output_token_ids and s.seq_id not in first_token_time:
                first_token_time[s.seq_id] = now
            if s.is_finished and s.seq_id not in completion_time:
                completion_time[s.seq_id] = now
    wall_time_s = time.perf_counter() - wall_start

    ttft_s = [first_token_time[s.seq_id] - wall_start for s in seqs]
    tpot_s = [
        (completion_time[s.seq_id] - first_token_time[s.seq_id]) / (len(s.output_token_ids) - 1)
        for s in seqs
        if len(s.output_token_ids) > 1
    ]
    total_tokens = sum(len(s.output_token_ids) for s in seqs)
    used_peak = max(used_slots_samples, default=0)
    used_mean = sum(used_slots_samples) / len(used_slots_samples) if used_slots_samples else 0.0
    # Occupancy of what the engine reserved *for the sequences it was actually
    # running* -- averaged over the run, since a single moment is not the claim.
    paired = [(u, r) for u, r in zip(used_slots_samples, reserved_slots_samples) if r > 0]
    occupancy = sum(u / r for u, r in paired) / len(paired) if paired else 0.0

    return {
        "throughput_tokens_per_sec": total_tokens / wall_time_s,
        "ttft_p50_s": _percentile(ttft_s, 50),
        "ttft_p99_s": _percentile(ttft_s, 99),
        "tpot_p50_s": _percentile(tpot_s, 50),
        "tpot_p99_s": _percentile(tpot_s, 99),
        "peak_concurrent_sequences": peak_concurrent,
        "allocated_token_slots": allocated_token_slots,
        "used_token_slots_peak": used_peak,
        "used_token_slots_mean": round(used_mean, 1),
        "kv_utilization_peak": used_peak / allocated_token_slots if allocated_token_slots else 0.0,
        "kv_utilization_mean": used_mean / allocated_token_slots if allocated_token_slots else 0.0,
        # The paper's number, computed for our own engines: of the KV memory an
        # engine holds for its live sequences, what fraction holds a real token?
        "reserved_slot_occupancy_mean": occupancy,
        "internal_fragmentation_peak": max(frag_samples, default=0) if paged else None,
        "measured_kv_bytes_peak": probe_peak if probe is not None else None,
        "total_tokens": total_tokens,
        "wall_time_s": wall_time_s,
    }


def run_contiguous(model_id: str, plan: BudgetPlan, workload, device=None) -> dict:
    engine = ContinuousEngine(model_id, max_batch_size=plan.contiguous_max_batch, device=device)
    seqs = [engine.add_request(p, max_new_tokens=n) for p, n in workload]
    result = _drive(
        engine,
        seqs,
        allocated_token_slots=plan.contiguous_max_batch * plan.max_cache_len,
        paged=False,
        probe=measure_cache_bytes,
    )
    # The derivation, checked against reality. `measured` is what the engine
    # allocated at its widest moment; it must not exceed the budget we sized
    # `max_batch_size` from, or the two engines were not compared at equal VRAM.
    measured = result["measured_kv_bytes_peak"]
    return {
        "engine": "continuous",
        "max_batch_size": plan.contiguous_max_batch,
        "kv_bytes_reserved": plan.contiguous_bytes,
        "budget_respected": measured <= plan.budget_bytes,
        **result,
    }


def run_paged(model_id: str, plan: BudgetPlan, workload, device=None) -> dict:
    engine = PagedEngine(
        model_id,
        max_batch_size=len(workload),  # never the binding constraint (ADR-0013)
        num_blocks=plan.paged_num_blocks,
        device=device,
    )
    seqs = [engine.add_request(p, max_new_tokens=n) for p, n in workload]
    result = _drive(
        engine,
        seqs,
        allocated_token_slots=plan.paged_num_blocks * BLOCK_SIZE,
        paged=True,
    )
    return {
        "engine": "paged",
        "max_batch_size": len(workload),
        "num_blocks": plan.paged_num_blocks,
        "kv_bytes_reserved": engine.pool.nbytes,
        "num_preemptions": engine.num_preemptions,
        **result,
    }


# --- the sweep -----------------------------------------------------------------


def _spec_and_max_cache_len(model_id: str, workload, device=None):
    """Load the model once to read its KV geometry and the workload's reserve-max."""
    probe = PagedEngine(model_id, num_blocks=1, device=device)
    spec = probe.pool.spec
    max_cache_len = max(len(probe.model_runner.encode(p)) + n for p, n in workload)
    return spec, max_cache_len, probe.model_runner.device


def sweep(model_id: str, budgets_mib: list[float], workload) -> dict:
    spec, max_cache_len, device = _spec_and_max_cache_len(model_id, workload)
    rows = []
    for mib in budgets_mib:
        plan = plan_for_budget(int(mib * 1024**2), spec, max_cache_len)
        row: dict = {"budget_mib": mib, "plan": asdict(plan), "results": {}}
        for name, runner in (("continuous", run_contiguous), ("paged", run_paged)):
            try:
                row["results"][name] = runner(model_id, plan, workload, device=device)
            except BlockExhausted as exc:
                # A pool too small for the workload's longest sequence is a
                # legitimate data point ("did not fit"), not a crashed run.
                row["results"][name] = {"engine": name, "did_not_fit": str(exc)}
        rows.append(row)
    return {
        "model_id": model_id,
        "commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        **_hardware(),
        "block_size": spec.block_size,
        "kv_bytes_per_token": kv_bytes_per_token(spec),
        "max_cache_len": max_cache_len,
        "num_requests": len(workload),
        "workload": "M2 Phase 4 fixed prompt list, mixed max_new_tokens (composes with the M2 run)",
        "rows": rows,
    }


def render_table(report: dict) -> str:
    """The Definition of Done, as a table: more sequences at equal VRAM."""
    lines = [
        "# M3 Phase 6 — paged vs. contiguous at equal VRAM",
        "",
        f"- Model: `{report['model_id']}`  ",
        f"- Commit: `{report['commit']}`  ",
        f"- GPU: {report['gpu']} ({report['vram_total_gib']} GiB total)  ",
        f"- torch {report['torch']}, Python {report['python']}  ",
        f"- KV per token: {report['kv_bytes_per_token'] / 1024:.0f} KiB "
        f"({report['kv_bytes_per_token'] * report['block_size'] / 1024:.0f} KiB per "
        f"{report['block_size']}-token block)  ",
        f"- Reserve-max length for this workload: {report['max_cache_len']} tokens  ",
        f"- Workload: {report['workload']} ({report['num_requests']} requests)",
        "",
        "| budget | contiguous batch | paged blocks | peak concurrent (cont / paged) "
        "| KV util peak (cont / paged) | throughput tok/s (cont / paged) | preemptions |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        plan = row["plan"]
        cont = row["results"].get("continuous", {})
        paged = row["results"].get("paged", {})
        if "did_not_fit" in cont or "did_not_fit" in paged:
            lines.append(
                f"| {row['budget_mib']:.0f} MiB | {plan['contiguous_max_batch']} | "
                f"{plan['paged_num_blocks']} | did not fit | — | — | — |"
            )
            continue
        lines.append(
            f"| {row['budget_mib']:.0f} MiB "
            f"| {plan['contiguous_max_batch']} "
            f"| {plan['paged_num_blocks']} "
            f"| {cont['peak_concurrent_sequences']} / {paged['peak_concurrent_sequences']} "
            f"| {cont['kv_utilization_peak']:.0%} / {paged['kv_utilization_peak']:.0%} "
            f"| {cont['throughput_tokens_per_sec']:.1f} / "
            f"{paged['throughput_tokens_per_sec']:.1f} "
            f"| {paged['num_preemptions']} |"
        )

    lines += [
        "",
        "## Budget check",
        "",
        "The contiguous engine has no VRAM knob, so its `max_batch_size` is derived",
        "from the budget. Measured against the tensors `StaticCache` actually",
        "allocated at its widest moment:",
        "",
    ]
    for row in report["rows"]:
        cont = row["results"].get("continuous", {})
        if "measured_kv_bytes_peak" in cont:
            ok = "within budget" if cont.get("budget_respected") else "OVER BUDGET"
            lines.append(
                f"- {row['budget_mib']:.0f} MiB: measured "
                f"{cont['measured_kv_bytes_peak'] / 1024**2:.2f} MiB peak - **{ok}**"
            )

    lines += ["", "## Cross-check", ""]
    for row in report["rows"]:
        cont = row["results"].get("continuous", {})
        paged = row["results"].get("paged", {})
        if "total_tokens" in cont and "total_tokens" in paged:
            ok = "OK" if cont["total_tokens"] == paged["total_tokens"] else "MISMATCH"
            lines.append(
                f"- {row['budget_mib']:.0f} MiB: total_tokens "
                f"{cont['total_tokens']} vs {paged['total_tokens']} — **{ok}**"
            )
    lines += [
        "",
        "Different memory strategies, identical output. A mismatch here would ",
        "invalidate every number above it.",
        "",
    ]
    return "\n".join(lines)


# --- the validation point ------------------------------------------------------
#
# The sweep above keeps M2's six-request workload so the numbers compose with
# the recorded M2 run -- but six requests caps peak concurrency at six, which
# is below what a generous budget can hold. So the concurrency claim can only
# be read off the sweep's tight budgets.
#
# This second workload exists to answer the obvious objection: does the ratio
# survive at a realistic budget with enough requests to saturate it? Larger
# budget, many more requests, longer generations. It does not compose with M2
# and is not meant to -- it validates, it does not extend.

VALIDATION_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain the difference between a process and a thread.",
    "Once upon a time,",
    "The three laws of thermodynamics are",
    "Write a haiku about debugging.",
    "Summarise how a page table translates a virtual address:",
    "List three reasons a database index can slow down writes.",
]

# Deterministic, long-tailed, and not uniform: a uniform length distribution
# is the one shape under which reserve-max is not wasteful, so it would hide
# the effect being measured. Cycled from a fixed list rather than sampled, so
# the workload is reproducible without carrying a seed.
VALIDATION_LENGTHS = [16, 24, 32, 48, 64, 96, 128, 192, 24, 40, 56, 80]


def validation_workload(num_requests: int) -> list[tuple[str, int]]:
    return [
        (
            VALIDATION_PROMPTS[i % len(VALIDATION_PROMPTS)],
            VALIDATION_LENGTHS[i % len(VALIDATION_LENGTHS)],
        )
        for i in range(num_requests)
    ]


def validate(model_id: str, budget_mib: float, num_requests: int) -> dict:
    workload = validation_workload(num_requests)
    spec, max_cache_len, device = _spec_and_max_cache_len(model_id, workload)
    plan = plan_for_budget(int(budget_mib * 1024**2), spec, max_cache_len)
    results = {}
    for name, runner in (("continuous", run_contiguous), ("paged", run_paged)):
        try:
            results[name] = runner(model_id, plan, workload, device=device)
        except BlockExhausted as exc:
            results[name] = {"engine": name, "did_not_fit": str(exc)}
    return {
        "model_id": model_id,
        "commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        **_hardware(),
        "block_size": spec.block_size,
        "kv_bytes_per_token": kv_bytes_per_token(spec),
        "max_cache_len": max_cache_len,
        "num_requests": num_requests,
        "workload": (
            f"validation: {num_requests} requests, long-tailed max_new_tokens "
            f"{min(VALIDATION_LENGTHS)}-{max(VALIDATION_LENGTHS)} (does NOT compose with M2)"
        ),
        "budget_mib": budget_mib,
        "plan": asdict(plan),
        "results": results,
    }


def render_validation(report: dict) -> str:
    cont = report["results"].get("continuous", {})
    paged = report["results"].get("paged", {})
    plan = report["plan"]
    if "did_not_fit" in cont or "did_not_fit" in paged:
        why = cont.get("did_not_fit") or paged.get("did_not_fit")
        header = "## Validation point"
        return header + "\n\nDid not fit: " + str(why) + "\n"
    ratio = paged["peak_concurrent_sequences"] / max(1, cont["peak_concurrent_sequences"])
    return "\n".join(
        [
            "## Validation point — a budget large enough to saturate",
            "",
            f"- Budget: {report['budget_mib']:.0f} MiB  ",
            f"- Requests: {report['num_requests']}, {report['workload']}  ",
            f"- Reserve-max length: {report['max_cache_len']} tokens "
            f"({plan['max_cache_len'] * plan['per_token_bytes'] / 1024**2:.2f} MiB per "
            "contiguous sequence)",
            "",
            "| | contiguous | paged |",
            "|---|---|---|",
            f"| derived config | max_batch_size={plan['contiguous_max_batch']} "
            f"| num_blocks={plan['paged_num_blocks']} |",
            f"| **peak concurrent sequences** | **{cont['peak_concurrent_sequences']}** "
            f"| **{paged['peak_concurrent_sequences']}** |",
            f"| reserved-slot occupancy | {cont['reserved_slot_occupancy_mean']:.0%} "
            f"| {paged['reserved_slot_occupancy_mean']:.0%} |",
            f"| KV utilization of budget (peak) | {cont['kv_utilization_peak']:.0%} "
            f"| {paged['kv_utilization_peak']:.0%} |",
            f"| throughput (tok/s) | {cont['throughput_tokens_per_sec']:.1f} "
            f"| {paged['throughput_tokens_per_sec']:.1f} |",
            f"| TTFT p50 / p99 (s) | {cont['ttft_p50_s']:.2f} / {cont['ttft_p99_s']:.2f} "
            f"| {paged['ttft_p50_s']:.2f} / {paged['ttft_p99_s']:.2f} |",
            f"| TPOT p50 / p99 (s) | {cont['tpot_p50_s']:.3f} / {cont['tpot_p99_s']:.3f} "
            f"| {paged['tpot_p50_s']:.3f} / {paged['tpot_p99_s']:.3f} |",
            f"| preemptions | — | {paged['num_preemptions']} |",
            f"| internal fragmentation (peak slots) | — | {paged['internal_fragmentation_peak']} |",
            f"| total tokens | {cont['total_tokens']} | {paged['total_tokens']} |",
            "",
            f"**Concurrency ratio: {ratio:.2f}x** at equal VRAM.",
            "",
            f"Cross-check: total_tokens {cont['total_tokens']} vs {paged['total_tokens']} — "
            f"**{'OK' if cont['total_tokens'] == paged['total_tokens'] else 'MISMATCH'}**.",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--budgets-mib",
        type=float,
        nargs="+",
        default=DEFAULT_SWEEP_MIB,
        help="KV budgets to sweep, in MiB",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="also run the large-workload validation point",
    )
    parser.add_argument("--validate-budget-mib", type=float, default=64.0)
    parser.add_argument("--validate-requests", type=int, default=48)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    report = (
        {"rows": [], "skipped": True}
        if args.skip_sweep
        else sweep(args.model_id, args.budgets_mib, WORKLOAD)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("docs/benchmarks") / f"{timestamp}-m3-paged-vs-contiguous"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = "" if args.skip_sweep else render_table(report)
    if args.validate:
        validation = validate(args.model_id, args.validate_budget_mib, args.validate_requests)
        (out_dir / "validation.json").write_text(json.dumps(validation, indent=2))
        table = table + chr(10) + render_validation(validation)
    if not args.skip_sweep:
        (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "README.md").write_text(table, encoding="utf-8")

    print(table)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
