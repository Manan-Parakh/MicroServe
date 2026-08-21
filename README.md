# MicroServe

A from-scratch LLM inference engine, built to learn how inference engines
actually work — continuous batching, paged KV caching, quantization — not to
ship a production server.

**This is a learning project, not a delivery project.** A working engine is
the byproduct; understanding the systems mechanics underneath it is the
point. See [`claude.md`](claude.md) for the full operating rules (decision
protocol, locked/open decisions, milestone guardrails) — that file governs
how work happens here.

## The plan

Six milestones, each gated by a correctness test and capped by an explicit
guardrail. Full detail, theory reading, and definitions of done live in
[`docs/roadmap.md`](docs/roadmap.md) — this table is just the shape of it.

| # | Milestone | Proves |
|---|---|---|
| 0 | Setup | Repo, tiny model, correctness harness |
| 1 | Naive baseline | The "before" number — one request at a time |
| 2 | Continuous batching | Iteration-level scheduling |
| 3 | Paged KV cache | Block allocator, block table, gather-then-attend — **the big one** |
| 4 | Quantization | 4-bit loading, re-benchmarked |
| 5 | Benchmark suite | Throughput / TTFT / TPOT / peak concurrency, four engine versions, one chart |
| 6 | Prefix sharing (stretch) | Reference-counted blocks, copy-on-write |

## Status

**Milestones 0 and 1 done.** Repo, `uv`-managed environment, and the
correctness harness (`tests/test_correctness.py`, fixed-seed greedy decode
vs. a golden HF reference) are in place; the naive one-request-at-a-time
engine passes it and has a recorded baseline
(`docs/benchmarks/2026-08-19-m1-naive-baseline/`: ~16 tokens/sec, p50 latency
~3.9s on an RTX 3050 Ti). Milestone 2 (continuous batching) is next. See
[`docs/progress.md`](docs/progress.md) for the dated log.

## Running it

```bash
uv sync                                    # one-time: installs the pinned environment
uv run pytest -v                           # the gate every milestone must pass
uv run python -m bench.benchmark --engine naive   # naive baseline benchmark run
```

See `docs/plans/m00-setup-plan.md` and `docs/plans/m01-naive-baseline-plan.md`
(each ends with a "Run Guide") for more detail, including how to drive the
engine directly.

## Docs

- [`docs/roadmap.md`](docs/roadmap.md) — the master plan: what gets built and
  read, in what order.
- [`docs/progress.md`](docs/progress.md) — dated log of decisions and changes.
- [`docs/concepts.md`](docs/concepts.md) — glossary of systems/inference
  concepts as they enter the code.
- [`docs/adr/`](docs/adr) — architecture decision records.
- [`docs/README.md`](docs/README.md) — full index of the docs folder.
