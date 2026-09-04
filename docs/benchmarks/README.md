# Benchmarks

Every benchmark run that gets quoted anywhere (README, an ADR, a conversation)
lands here as raw numbers plus the exact commit and hardware it ran on.
**Numbers without provenance are not quotable** — see `claude.md`.

## What goes in a run

One subfolder or file per run, named
`YYYY-MM-DD_HH-MM-SS-<engine-version>-<short-desc>` (e.g.
`2026-09-02_14-30-05-m1-naive-baseline/`), containing:

- The raw output (CSV/JSON) from `python -m bench.benchmark`.
- A one-line header recording: git commit hash, GPU model, driver/CUDA
  version, model used, and the workload config (Poisson rate, prompt/output
  length distribution).
- The plot, if one was generated for this run.

## The four metrics

Every run records exactly these, per `claude.md`:

- **Throughput** — system-wide tokens/sec
- **TTFT** — time to first token (reflects prefill)
- **TPOT** — time per output token (reflects decode)
- **Peak concurrent sequences** — the memory-pressure number

Latency is reported as p50 and p99, never a bare mean.

## Milestone 5 comparison chart

The four-way comparison (naive → continuous batching → paged → quantized) at
increasing request rate is the headline artifact of Milestone 5. It belongs
here, alongside the exact numbers behind it.


## Runs

| date | milestone | what it measures |
|---|---|---|
| 2026-08-19 / 08-21 | M1 | naive baseline, one request at a time |
| 2026-08-23 | M2 | continuous vs. static vs. naive batching |
| 2026-09-03 | M3 | **paged vs. contiguous KV cache at equal VRAM** |

The M3 run (`2026-09-03_15-57-52-m3-paged-vs-contiguous/`) is produced by
`python -m bench.paged_benchmark --validate`, not by `bench/benchmark.py`.
It carries two workloads:

- a **budget sweep** on M2's six-request workload, so the numbers compose with
  the recorded M2 run;
- a **validation point** (64 MiB, 128 requests, long-tailed lengths) large
  enough for both engines to saturate, which is where the milestone's claim is
  actually demonstrated: **4.92x concurrent sequences at equal VRAM**, with
  reserved-slot occupancy of 42% (contiguous) vs. 90% (paged).

Read the run's own `README.md` before quoting any of it — the TTFT result
belongs to the absence of ADR-0007's resync, not to paging, and the two must
not be conflated.
