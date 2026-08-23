# ADR-0006: M2 static-batching baseline built for real, not reused from M1

- **Status**: Accepted
- **Date**: 2026-08-21
- **Milestone**: M2
- **Concepts**: static batching, continuous/iteration-level batching

## Context

`docs/roadmap.md`'s Milestone 2 Definition of Done asks for "a benchmark
demonstrating continuous batching beating static batching on throughput and
GPU idle time." M1's naive engine (one request at a time, no batching at
all) already exists and is the cheapest possible thing to point at as "the
comparison" — but whether it's the *right* comparison for what M2 is
actually claiming needed a real decision.

## Options considered

### Option A — Build a real static-batch engine
A small, separate engine variant: admits a fixed-size batch of sequences
together, runs them via the same forward-step primitive as the continuous
engine, and does not admit any new request until every member of that batch
has finished (the slowest sequence gates the whole batch).
- **Buys**: this is the actual "static batching" the Orca paper contrasts
  against — batched, but rigid about when new work is admitted. It isolates
  exactly one variable (admission timing) between it and the continuous
  engine, since both share Phase 1's forward-step primitive; the comparison
  demonstrates *iteration-level scheduling's* benefit specifically, not just
  "batching beats no batching."
- **Costs**: a small extra build item this milestone wasn't originally going
  to need a second engine for.

### Option B — Reuse M1's naive engine as the "static" baseline
Compare the continuous engine directly against `engine/engine.py` from M1.
- **Buys**: zero new code — the baseline already exists and is already
  benchmarked.
- **Costs**: M1 isn't batching at all (batch size 1), which is a different
  failure mode than static batching's "waits for its slowest member." Using
  it as the M2 comparison would still show continuous batching winning, but
  for the wrong reason — any batching beats no batching — which overstates
  the result and isn't the claim Orca's paper (or this milestone's own DoD
  wording) is actually making.

## Decision

**Option A — build a real, small static-batch engine.**

## Why the others lose

Option B is cheap but dishonest by omission: it would produce an
impressive-looking chart that doesn't actually demonstrate the thing M2
exists to prove (iteration-level scheduling beats fixed-batch scheduling
under variable output lengths). `claude.md`'s own precedent — ADR-0004
keeping M1's baseline "bad on purpose, measured honestly" — is the same
principle here in reverse: a benchmark's credibility depends on the
comparison being apples-to-apples, and M1 vs. continuous-M2 isn't that.

## Consequences

Three engines exist after this milestone (naive, static-batch, continuous-
batch), which conveniently previews the shape of M5's four-way comparison
chart (naive → static → paged → quantized becomes naive → static →
continuous → paged → quantized, one column added per milestone). The
static-batch engine is intentionally minimal — it exists only to be a fair
comparison point, not a serving mode worth optimizing further.

## What I learned

*(owner's to fill in, optional)*
