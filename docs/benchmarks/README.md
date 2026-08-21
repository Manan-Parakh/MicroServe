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
