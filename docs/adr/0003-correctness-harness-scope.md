# ADR-0003: Correctness harness scope — greedy-only, no sampling path

- **Status**: Accepted
- **Date**: 2026-08-19
- **Milestone**: M0
- **Concepts**: greedy decoding, deterministic generation

## Context

`claude.md` Open Decision #1 leaves open whether `tests/test_correctness.py`
covers greedy decoding only, or also a fixed-seed sampling path. This is
explicitly called out as blocking M0, and cheaper to decide now than to
retrofit once M1–M3 engine variants already exist and would all need
updating to support a second decoding mode.

## Options considered

### Option A — Greedy-only
Every correctness test fixes a seed and decodes with `do_sample=False`
(argmax every step).
- **Buys**: removes sampling as a variable entirely — a mismatch between
  golden and candidate can only mean the code computed something different,
  not that it sampled a different (still valid) token from an equivalent
  distribution. Exactly the signal needed to catch batching and block-table
  indexing bugs, which is what this project is most exposed to from M2
  onward.
- **Costs**: doesn't exercise the sampling code path at all — if a future
  milestone's engine needs to support `do_sample=True`, this harness gives no
  coverage of that until extended.

### Option B — Greedy + fixed-seed sampling path
Adds a second harness path: fixed seed, temperature/top-p sampling, asserting
the candidate reproduces the same *sampled* sequence as the golden run.
- **Buys**: closer to how the eventual benchmark workload will actually
  decode (real serving traffic isn't always greedy), and exercises the
  sampling code path for bugs greedy decoding can't reach (e.g. an RNG stream
  consumed in a different order across batched vs. single-sequence code
  paths).
- **Costs**: real added complexity now, for a benefit M0–M1 don't need yet —
  proving two runs with the same seed reproduce the same *sampled* sequence
  requires the RNG to be consumed in the same order regardless of batch
  composition, which is itself a nontrivial property to guarantee once
  continuous batching (M2) enters the picture. Building this before there's
  anything but a single-sequence naive engine to test it against is premature.

## Decision

**Greedy-only for now.**

## Why the others lose

Option B loses on timing, not merit — the property it would need to prove
(RNG-order stability across batch compositions) doesn't even become
observable until M2's scheduler exists, so building the harness for it now
means guessing at a design under untested assumptions. Nothing about Option A
forecloses adding Option B later; the golden-reference machinery
(`_golden_tokens` in `tests/test_correctness.py`) is written so a sampling
variant could be added as a parallel test function without restructuring
what's there.

## Consequences

Every engine variant only ever needs to prove greedy-decode equivalence to
ship. If a later milestone (most likely M2, once batching makes RNG-order
sensitivity real, or M5's benchmark workload wants realistic sampling) needs
sampling coverage, this decision gets revisited explicitly rather than
silently — see `docs/roadmap.md` Open Decision #1 status.

## What I learned

*(owner's to fill in, optional)*
