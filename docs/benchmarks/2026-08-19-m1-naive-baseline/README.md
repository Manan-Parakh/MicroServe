# 2026-08-19 — M1 naive baseline

- **Commit:** uncommitted (no git history exists yet — re-run once the first
  commit lands so this number carries real provenance; see M0/M1 progress
  entries)
- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (driver-reported CUDA 13.1)
- **Model:** Qwen/Qwen2.5-0.5B-Instruct
- **Workload:** 5 fixed prompts, sequential, one request at a time — no
  concurrency exists yet at M1 (see ADR-0004 for why this isn't the real
  Poisson/ShareGPT workload harness)

## Numbers

| Metric | Value |
|---|---|
| Throughput | 16.15 tokens/sec |
| Latency p50 | 3.93 s |
| Latency p99 | 4.27 s |
| Total tokens generated | 320 |
| Wall time | 19.81 s |

This is the "before" number. Every later milestone (M2 continuous batching,
M3 paged KV cache, M4 quantization) gets compared back to this — see
`docs/benchmarks/README.md` for the four-metric convention and
`docs/plans/m01-naive-baseline-plan.md` for what this run deliberately does
and doesn't do.

Raw data: `results.json` in this folder.
