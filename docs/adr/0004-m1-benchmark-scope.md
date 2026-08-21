# ADR-0004: M1 benchmark script scope — bare sequential timing, not the real harness

- **Status**: Accepted
- **Date**: 2026-08-19
- **Milestone**: M1
- **Concepts**: throughput vs. latency, TTFT/TPOT (named but deliberately not
  measured yet)

## Context

`docs/plans/m01-naive-baseline-plan.md` Q1 asks how much of `bench/benchmark.py`
to build now vs. leave for Milestone 5, which owns the real Poisson-arrival,
ShareGPT-derived workload harness per `claude.md`.

## Options considered

### Option A — Bare sequential timing script
`bench/benchmark.py --engine naive` runs a small fixed prompt list one at a
time, times each request, computes throughput and p50/p99 latency, saves JSON
to `docs/benchmarks/`.
- **Buys**: minimal, matches what M1 actually needs (a number to beat), and
  doesn't require the concurrency M2's scheduler will introduce — which the
  real Poisson workload generator exists to stress.
- **Costs**: gets rewritten at M5 rather than incrementally grown.

### Option B — Start building the real harness now
Begin Poisson-process arrival simulation and ShareGPT-derived length sampling
in M1, ahead of schedule.
- **Buys**: less total work at M5, one harness built once.
- **Costs**: a workload generator modeling concurrent arrivals has nothing
  meaningful to stress against an engine that can only run one request at a
  time — M1's engine has no concept of "requests arriving while another is
  running." Building this now means designing it against untested
  assumptions about what M2's scheduler interface will look like.

## Decision

**Option A — bare sequential timing script.**

Scope note beyond Q1 itself: this run also does **not** measure TTFT/TPOT or
peak-concurrent-sequences, even though `docs/benchmarks/README.md` names all
four metrics as the standard. `claude.md`'s benchmark-plumbing/output-shape
items are explicitly "decide freely" — but the reasoning is worth recording:
M1's `Engine.step()` runs a request to full completion in one call, so there's
no per-token timestamp to measure TTFT/TPOT from without adding a streaming
callback, and peak-concurrency is trivially 1 by construction (no
concurrency exists yet). Only throughput and p50/p99 latency are measured,
matching M1's own Definition of Done in `docs/roadmap.md` ("Throughput and
latency recorded"). TTFT/TPOT become measurable and meaningful once decode is
iteration-level (M2); peak-concurrency becomes meaningful once memory
pressure is a real constraint (M3).

## Why the others lose

Option B loses for the same reason ADR-0003 rejected building sampling
coverage early: designing a harness for a capability (concurrent, staggered
request arrival) that doesn't exist yet in the engine means guessing at an
interface M2 hasn't defined. The roadmap already assigns this work to M5 for
exactly this reason.

## Consequences

`bench/benchmark.py`'s current fixed prompt list and single `run_naive()`
function are expected to be replaced, not extended, at M5 — this is stated
up front so nobody mistakes the M1 script for a foundation to build on
incrementally. `docs/benchmarks/<date>-m1-naive-baseline/results.json` is the
first entry in `docs/benchmarks/`; every later milestone's numbers are meant
to be compared against it.

## What I learned

*(owner's to fill in, optional)*
