# ADR-0012: Preemption by recomputation, not by swapping to host memory

- **Status**: Accepted
- **Date**: 2026-09-03
- **Milestone**: M3 (Phase 4 — `PagedEngine`: admission gated by memory)
- **Concepts**: preemption, recomputation vs. swapping, memory-pressure-bound
  admission, PCIe bandwidth, arithmetic intensity

## Context

Phase 4 introduces a failure mode that has never existed in this codebase.
M2's scheduler admits on one condition — is there a slot under
`max_batch_size` — and that condition, once satisfied, stays satisfied for the
life of the sequence. A paged scheduler admits on two: a slot, **and** enough
free blocks for the prompt. The second one can stop being true *after*
admission, because every running sequence is still growing. A decode step can
need a block that does not exist.

Concretely: with `BLOCK_SIZE = 16`, a sequence consumes one new block every
sixteenth decode step. N sequences all admitted legally can, sixteen steps
later, collectively want more blocks than the pool holds. Nobody did anything
wrong; the pool is simply full. Something has to give, and *what* gives is a
policy decision with three real answers.

The hardware constraint that decided it: this runs on a laptop (ROG Strix G17,
RTX 3060/4060-class; a Legion Pro 5 with an RTX 5060 incoming). Host↔device
transfer is over PCIe on a consumer mobile platform, and it is shared with
everything else the laptop is doing. There is no NVLink, no fat interconnect,
and no second GPU to hide a transfer behind.

## Options considered

### Option A — Preempt by recomputation
On `BlockExhausted` mid-step, evict the most recently admitted running
sequence: free every block it owns, set it back to `WAITING`, discard its KV.
Its *tokens* are kept. When it is readmitted it prefills its prompt plus
everything it had already generated, and continues.
- **Buys**: no new machinery whatsoever. `BlockManager.free` already resets
  `num_cached_tokens`; `Sequence` already separates prompt from output; the
  prefill path already exists and already handles a batch. The entire policy
  is ~15 lines in the scheduler. Nothing new to be wrong.
- **Costs**: the victim's generated tokens are computed twice. Wasted prefill
  compute, proportional to how far it had got.

### Option B — Preempt by swapping to host memory
Copy the victim's blocks over PCIe into a pinned host buffer, free the GPU
blocks, and copy back on readmission. No compute is wasted; the KV is
preserved exactly.
- **Buys**: zero recomputation. Under a workload of very long sequences, where
  re-prefilling 2,000 tokens is genuinely expensive, this is the better trade.
- **Costs**: a second pool, a second allocator, a second block table per
  swapped sequence, and a synchronous transfer sitting in the `step()`
  critical path. Every one of those is new state that can be wrong, and the
  transfer competes for bandwidth with nothing else in this engine — meaning
  the benchmark would partly be measuring PCIe.

### Option C — Refuse admission only; never preempt
Gate hard at admission and guarantee a running sequence can always grow, by
reserving `ceil((prompt + max_new_tokens) / BLOCK_SIZE)` blocks up front.
- **Buys**: the simplest possible scheduler. No preemption path at all.
- **Costs**: it is reserve-max wearing a block table.

## Decision

**Preemption by recomputation.** The most recently admitted running sequence is
evicted, its blocks are freed, and it is requeued at the *front* of the waiting
queue. Readmission re-prefills prompt + generated tokens.

## Why the others lose

**Option B loses on the machinery-versus-compute argument, and the hardware is
what settles it.** The two options trade the same currency in opposite
directions: recomputation spends *compute* to avoid moving bytes; swapping
spends *bandwidth and complexity* to avoid recomputing. On this machine that is
not a close call. A prefill is the compute-bound phase of inference — it is the
one part of this workload with high arithmetic intensity, the part a GPU is
actually good at, and on the dev model it is milliseconds. A host↔device round
trip on a laptop's PCIe link is the opposite: pure data movement, no arithmetic
to amortise it against, on the slowest link in the system, and it must complete
before the step can proceed. We would be paying the machine's weakest resource
to save its strongest one.

And the complexity is not incidental to that trade — it *is* most of the cost.
Swapping needs a host pool sized by a policy nobody has chosen, a host-side
allocator, a mapping from GPU block to host block, and a decision about whether
transfers are synchronous (simple, slow) or overlapped with compute
(fast, and a whole new class of correctness bug where a sequence attends to a
block still in flight). That is a meaningful fraction of the milestone's total
complexity spent on a path that fires only under memory pressure — which is
precisely the path that gets the least testing. Recomputation adds *no new
state at all*: it reuses `free()`, reuses the prefill path, and its
"restore" operation is a function this engine already runs on every admission.

This is not a claim that swapping is wrong in general. It wins when
recomputation is expensive relative to transfer — long sequences, a fast
interconnect, or a server GPU with NVLink and PCIe 5. The PagedAttention paper
reaches the same conclusion for small block sizes and short-to-medium
sequences, which is exactly this workload. **Revisit if** the headline model
moves to long-context workloads (multi-thousand-token sequences) or the
hardware gains a fast host link.

**Option C loses because it un-measures the milestone.** Reserving
`max_new_tokens` of headroom at admission reintroduces exactly the reservation
waste M3 exists to remove — memory held against tokens that may never exist —
while still paying paging's gather cost. `peak_concurrent_sequences`, the
metric this milestone is really moving, would go back to reporting a function
of the configuration rather than of real memory usage, which is the caveat M2's
benchmark README already had to carry. It is also worth being precise about
*why* the naive version of Option C (gate at admission, no reservation) is not
merely simple but actually unsound: it does not prevent mid-generation
exhaustion at all, it just has no answer when it happens.

## Consequences

**Makes easier.** The preemption path is short enough to hold in the head, and
it is exercised on every test run — `tests/test_paged_scheduler.py` uses a fake
runner and a 4-block pool so exhaustion is the normal case rather than a rare
one. Recomputation is also *transparent*: a preempted-and-recomputed sequence
emits the same tokens as one that ran straight through, which
`test_preemption_does_not_change_the_tokens` pins against the real model. That
transparency is what lets the M5 chart treat preemption as a cost rather than a
behaviour change.

**Makes harder.** Throughput now has a component that is not paging: wasted
prefill. `PagedScheduler.num_preemptions` is counted and surfaced on the engine
precisely so M5 can separate the two. A chart that attributes recomputation
cost to paging would understate paging.

**Termination, and the one thing that guarantees it.** Newest-first eviction is
not a fairness choice, it is what makes the scheme terminate: the oldest
sequences keep their blocks and keep making progress, so the pool drains
steadily. Evicting the oldest would repeatedly destroy the most accumulated
work and could livelock a saturated pool. The victim is requeued at the *front*
for the complementary reason — it has the most sunk cost and must not be
overtaken by every request that arrived while it ran.

**The residual failure, deliberately loud.** If only one sequence is running
and it still cannot grow, there is nothing to evict but itself, and preempting
it would readmit it into the identical shortage forever. That is a pool-sizing
error, not a scheduling one, so `_grow_running` raises `BlockExhausted` with a
message naming the pool size and the length that broke it. A silent
truncation there would be the worst possible outcome: correct-looking short
output.

**Seam left open.** Nothing here forecloses swapping. `BlockManager.free` and
the readmission path are the only two touchpoints; a `swap_out(seq)` /
`swap_in(seq)` pair would slot in beside them, and `num_preemptions` would
split into two counters. If M4's quantized headline model makes long-context
workloads feasible, this ADR should get a successor rather than an edit.

## What I learned

The interesting question was never "which is faster" — it was *which resource
is scarce on the machine I actually have*. Recomputation and swapping trade
compute against bandwidth, and on a laptop the compute is the abundant side and
the PCIe link is the scarce one. The second realisation is that "no new state"
is a real engineering benefit and not just tidiness: the swap path would be the
least-tested code in the engine and it would hold the most invariants.
