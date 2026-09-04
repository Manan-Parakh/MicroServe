# ADR-0013: A separate `PagedScheduler`, and prefill-priority steps

- **Status**: Accepted
- **Date**: 2026-09-03
- **Milestone**: M3 (Phase 4 — `PagedEngine`)
- **Concepts**: iteration-level scheduling, head-of-line blocking, chunked
  prefill, benchmark provenance

## Context

Two structural questions had to be answered before Phase 4 could be written,
and both were on the plan's open list.

**First: where does the paged scheduling policy live?** M2's `Scheduler`
already contains the waiting/running queue walk, the eviction check, and the
step loop. The paged version differs in its admission predicate and in having
a preemption path. That is either a parameterization of one class or a second
class.

**Second: what does a single `step()` actually do** when there are both
sequences to admit (which need a prefill) and sequences already running (which
need a decode)? Fusing prefill and decode into one forward call is a locked
decision against, so a step must choose.

## Options considered

### Option A — A separate `PagedScheduler` class
`engine/paged_scheduler.py` stands alone, duplicating the shape of M2's queue
walk (~30 lines) but free to diverge.
- **Buys**: M2's scheduler stays byte-identical to the version its benchmark
  numbers were produced by.
- **Costs**: duplicated queue-walk logic; a bug fixed in one is not fixed in
  the other.

### Option B — Generalize `Scheduler` with an admission policy object
One class taking a policy; M2 passes a slot-only policy, M3 a
slot-plus-blocks policy.
- **Buys**: names the real axis of variation; no duplication.
- **Costs**: edits code four recorded benchmark runs depend on, and the
  abstraction does not actually fit (see below).

### On the step question

**Option 1 — Prefill-priority.** A step that admits anything is a prefill step
for the newly admitted sequences only; the running set skips that iteration.
**Option 2 — Two forward calls per step.** Prefill the new admissions, then
decode everyone. Each step is one iteration for every sequence.
**Option 3 — Chunked prefill.** Split a long prefill into block-sized chunks
and co-schedule them with decodes, so neither starves the other.

## Decision

**A separate `PagedScheduler` class**, and **prefill-priority steps** — a step
either admits-and-prefills, or decodes, never both.

## Why the others lose

**Option B loses on benchmark provenance, and then on fit.** M5's headline
deliverable is a four-way comparison chart across all four engine versions, and
every number in `docs/benchmarks/` is annotated with the commit that produced
it. A shared base class means every Phase 4 change is a change to an
already-measured engine: either the M2 numbers get silently invalidated, or
they get re-run, or — worst — nobody notices which happened. Some duplication
is the price of keeping the "before" number honest, and it is a small price
here because M2's scheduler is *finished*: it will not accumulate fixes that
the paged one misses.

The fit argument is the stronger one though. The abstraction on offer was
"admission is a predicate", and preemption is not an admission predicate. It
removes a sequence from `running` *during* a step, retries the growth walk, and
requeues the victim — a control-flow change, not a swap of a boolean test. To
express it as a policy hook, `Scheduler.step()` would have to grow a retry loop
and a mutable-running-set contract that M2 has no use for. The generalization
would make M2's code more complicated in order to avoid duplicating it.

**Option 2 loses because it makes a "step" mean two different amounts of work.**
Two forward calls in one step breaks the thing iteration-level scheduling is
for: a step is the unit the benchmark measures, and TPOT is derived from it.
A newly admitted sequence would also produce two tokens in its first step,
which makes the first-token timestamp ambiguous — and TTFT is one of the four
metrics.

**Option 3 is correct and out of scope.** Chunked prefill is exactly the fix
for what Option 1 costs, and it is what vLLM does. It is also a scheduling
project of its own: prefill chunks have to be interleaved with decodes under a
token budget, and the block accounting for a partially-prefilled sequence is
new. M3's guardrail is explicit that depth is capped deliberately. Noted here
so a future reader knows the gap is known rather than missed.

## Consequences

**The cost of prefill-priority, stated rather than smoothed over.** A step that
admits spends itself entirely on that prefill, so every already-running
sequence waits an iteration. That is head-of-line blocking, and it trades TPOT
for TTFT. Under a Poisson arrival process with a long-tailed prompt
distribution — which is exactly M5's workload — a burst of arrivals will show
up as a TPOT spike for the sequences already in flight. This must be visible in
the M5 chart and explained there, not filed as noise.

**What paging removed, and what it did not.** `PagedScheduler` issues **no
resync**. M2 rebuilds its entire `StaticCache` on every membership change
(ADR-0007), because a single shared write-position cannot track per-row
positions; a paged cache has a block table per sequence, so one sequence
finishing touches nobody else's KV. This was the plan's "probably not, and
that's a genuine secondary win" — it holds, and
`test_one_sequence_finishing_does_not_disturb_another` pins it by asserting
that no prefill call is issued on the survivor's behalf. Dropping M2's most
expensive recurring operation is a second, separable win of M3 and should be
reported as such rather than folded into the paging number.

**`max_batch_size` becomes an upper bound, not the binding constraint.**
`PagedEngine`'s default is 64, far above M2's 8, deliberately. In M3 the pool
is meant to bind. Setting `max_batch_size` low enough to bind again would
silently un-measure the entire milestone —
`peak_concurrent_sequences` would go back to reporting its own configuration.

**Duplication to watch.** The queue walk, the EOS/length eviction check, and
the step skeleton now exist twice. If a *third* scheduler ever appears, that is
the signal to extract — three instances is where the shared shape is actually
known, two is where it is guessed.

## What I learned

The instinct to unify the two schedulers was about the code, and the reason not
to was about the *numbers*. A benchmark result is only meaningful next to the
exact code that produced it, so an already-measured engine is closer to a
recorded artifact than to live code. The second thing: an abstraction that
would have to complicate the simpler caller in order to accommodate the more
complex one is not an abstraction, it is a coupling.
