# ADR-0014: How "equal VRAM" is defined for the M3 benchmark

- **Status**: Accepted
- **Date**: 2026-09-03
- **Milestone**: M3 (Phase 6 — the memory-utilization benchmark)
- **Concepts**: KV-cache memory utilization, reserve-max, benchmark
  provenance, capacity vs. bandwidth

## Context

M3's Definition of Done is a comparison: **more concurrent sequences at the
same VRAM budget**. Making that measurable ran into a problem the plan did not
anticipate — `ContinuousEngine` has no VRAM knob. It takes `max_batch_size` and
sizes its `StaticCache` to `max(prompt + max_new_tokens)` over the running
batch. `PagedEngine` takes `num_blocks`. The two engines are configured in
different currencies, so "equal VRAM" is not a setting either of them has.

A second problem appeared once the harness ran. The plan specified reusing M2's
six-request workload so the numbers would compose with the recorded M2 run.
They do — but six requests caps peak concurrency at six, so above a 3 MiB
budget both engines run everything and the ratio collapses to 1.0x. The
workload cannot demonstrate the claim it was chosen to support.

Third, the machine. An RTX 3050 Ti Laptop GPU (4 GiB) running a 0.5B model
cannot be driven into genuine whole-GPU VRAM pressure at any feasible runtime:
it would take thousands of concurrent long sequences, and `ContinuousEngine`'s
resync (ADR-0007) re-prefills the entire batch on every membership change,
which makes that take hours rather than minutes.

## Options considered

### Option A — Derive each engine's configuration from a byte budget
Fix a budget in bytes. Compute
`per_token_bytes = 2 * layers * kv_heads * head_dim * itemsize`, then
`contiguous_max_batch = budget // (max_cache_len * per_token_bytes)` and
`paged_num_blocks = budget // (block_size * per_token_bytes)`. Verify the
contiguous derivation against the tensors `StaticCache` actually allocates.
- **Buys**: one number in, two configurations out, both checkable. Fully
  reproducible — nothing depends on ambient GPU state.
- **Costs**: the contiguous side is derived arithmetic, not a knob the engine
  exposes, so it has to be validated rather than trusted.

### Option B — Fully empirical: raise concurrency until each engine breaks
Sweep offered concurrency upward and record where each stops fitting (OOM for
contiguous, `BlockExhausted` for paged).
- **Buys**: nothing is modelled.
- **Costs**: an OOM on a 4 GiB laptop GPU shared with the desktop compositor is
  not a clean signal. Torch's caching allocator, fragmentation from the
  previous run, and whatever else is resident all move the break point. Same
  commit, same machine, different answer.

### Option C — Don't equalize; report allocated bytes at equal batch size
Run both at the same `max_batch_size` and report how much KV each allocated.
- **Buys**: simplest possible harness.
- **Costs**: answers "paged uses less memory for the same work" rather than
  "paged fits more sequences at the same memory", which is not the milestone's
  Definition of Done.

## Decision

**Option A**, with two workloads rather than one:

1. A **budget sweep** (1–8 MiB) on M2's six-request workload, so the numbers
   compose with the recorded M2 run as the plan required.
2. A **validation point** (64 MiB, 128 requests, long-tailed generation
   lengths) large enough that both engines saturate, which is where the claim
   is actually demonstrated.

And **two utilization metrics**, because one denominator cannot carry both
questions:

- `kv_utilization_of_budget` — real tokens ÷ the fixed budget. Answers "how
  much of the VRAM I paid for is doing work."
- `reserved_slot_occupancy` — real tokens ÷ what the engine reserved *for the
  sequences it is currently running*. This is the paper's ~40%-vs->90% number,
  computed for our own engines, and it is independent of budget size.

## Why the others lose

**Option B loses on reproducibility, which is the property `claude.md`'s
provenance rule exists to protect.** A benchmark number is only meaningful
alongside the commit and hardware that produced it; a number that also depends
on how much of the GPU the window manager happened to be using is not
quotable. A crashed process inside a benchmark loop is also a bad primitive —
recovering allocator state between points is more fragile than the arithmetic
it was meant to avoid.

**Option C loses because it answers a different question.** It is a true and
useful statement, and it is not the Definition of Done. The roadmap is explicit
that the headline is concurrency at fixed VRAM, and reframing the metric to fit
the easier harness would be quietly moving the goalposts.

**On the single-workload version of the plan**: keeping only the M2 workload
would have produced a table whose ratio is 1.0x in four of six rows, which
reads as a failed milestone rather than as a workload too small to measure it.
Keeping only the validation workload would have broken continuity with the M2
run. Both, clearly labelled, is the honest resolution — and the sweep's
low-budget rows carry real information the validation point does not (1 MiB is
the only configuration in which preemption fires at all).

## Consequences

**The result.** 4.92x concurrent sequences at equal VRAM (26 → 128), with
reserved-slot occupancy 42% → 90%. The second number *explains* the first
rather than corroborating it independently, and the writeup says so.

**4.92x is a floor.** The paged engine peaked at 128 — every request in the
workload — with zero preemptions, so its real ceiling at 64 MiB was never
reached. The contiguous engine was genuinely saturated at 26. A larger workload
widens the gap.

**One result must not be credited to paging.** TTFT p99 fell from 38.6 s to
0.46 s. That is the absence of ADR-0007's resync, not paging: the contiguous
engine re-prefills its whole batch on every membership change, and with 26 rows
churning continuously a new arrival queues behind repeated full-batch prefills.
A paged cache has a block table per sequence, so membership changes are free.
Separable win, separately reported — conflating them would overstate paging.

**The gather cost did not show, and that is a bounded claim.** TPOT was
essentially flat (0.112 → 0.098 s p50). At 128 concurrent sequences the
per-layer gather is small next to the weight traffic of a forward pass. It
would show at long context, where the gathered rectangle grows with history.
Saying "paging is free" from this data would be wrong.

**Harness placement.** The M3 driver lives in `bench/paged_benchmark.py`, not
in `bench/benchmark.py` as the plan's Files line said. Same reasoning as
ADR-0013: `bench/benchmark.py` produced numbers that are already recorded in
`docs/benchmarks/`, and an already-measured instrument is closer to a recorded
artifact than to live code. It imports `WORKLOAD` and `_percentile` from the M2
driver so the two cannot silently diverge on the shared definitions.

**Seam left open.** The sweep takes `--budgets-mib` and the validation point
takes `--validate-budget-mib` / `--validate-requests`, so M4's quantized
headline model can re-run both at a realistic budget without touching the
harness. When that happens, the caveat about not reaching whole-GPU pressure
should be revisited rather than copied forward.

## What I learned

The hard part of this phase was not measuring anything — it was noticing that
the workload the plan specified could not exhibit the effect the plan wanted to
measure, and that "the ratio is 1.0x" was a fact about six requests rather than
about paging. A benchmark that cannot fail informatively is not a measurement.
The other thing: two metrics that look like the same quantity ("utilization")
answer completely different questions depending on the denominator, and picking
one silently would have made the whole table unreadable.
