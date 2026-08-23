# ADR-0007: M2 continuous-batch decode uses StaticCache with a resync-prefill on batch membership change

- **Status**: Accepted
- **Date**: 2026-08-22
- **Milestone**: M2
- **Concepts**: KV cache, batched decode, prefill vs. decode (refines ADR-0005)

## Context

ADR-0005 decided the KV cache should be an owned, pre-allocated, contiguous
buffer rather than `transformers`'s auto-growing `DynamicCache`. Implementing
Phase 1 surfaced a detail ADR-0005 didn't anticipate: `transformers` 5.15.1
requires `past_key_values` to be a `Cache` object (no raw tensors), and its
`StaticCache` — the obvious pre-allocated, in-place-written class matching
ADR-0005's intent — tracks a single write-position shared across the whole
batch (`cumulative_length` is one scalar, not one per row). That's correct
for a batch where every row started together (the static-batch engine), but
wrong for continuous batching: a sequence admitted after other rows have
already advanced would be written into the wrong cache position, silently
corrupting those rows' attention — exactly the kind of block-indexing bug
`claude.md` singles out as the dangerous kind.

## Options considered

### Option A — Resync-prefill on every batch membership change
Whenever the set of running sequences changes (an admission or an eviction),
build a fresh `StaticCache` and run one batched forward pass over every
currently-running sequence's *full* token history so far (prompt + generated
tokens), left-padded to a common length, plus any newly-admitted sequences'
prompts. This realigns every row to the same shared position, after which
plain `StaticCache` decode steps are correct as-is until the next membership
change.
- **Buys**: no custom cache code — `StaticCache` is used exactly as
  designed. The repeated computation on membership change is bounded and
  explicit, and gives a concrete, honest number for "this is the
  recomputation cost paging removes" once M3 lands.
- **Costs**: sequences that have been running a while get their whole
  history re-processed (as a prefill-shaped forward pass) every time the
  batch's membership changes, not just their one new token.

### Option B — Custom per-row-position cache layer
A small custom `Cache` subclass tracking each batch row's own write
position independently (a tensor of positions instead of one scalar), so a
newly-admitted row writes at its own offset without touching other rows'
already-written positions or requiring their recomputation.
- **Buys**: no repeated computation — admission/eviction become O(1) per
  affected row instead of O(current batch size).
- **Costs**: real new indexing code, in the same spirit as a block table —
  genuinely close to what M3's `BlockAllocator` will build, meaning it
  either gets built solo (against `claude.md`'s rule on handing over the
  milestone's core mechanism) or pulled into the same "design and build
  together" treatment as the `Scheduler`, adding real scope to a phase this
  plan scoped as plumbing.

## Decision

**Option A — resync-prefill on membership change, using `StaticCache`
unmodified.**

## Why the others lose

Option B is more efficient but reimplements a simplified block table a
milestone early — directly against the M2 guardrail's own reasoning
("isolates the scheduling problem so Milestone 3 can isolate the memory
problem"; "accepting fragmentation here is correct, not lazy"). Repeated
recomputation on membership change *is* this milestone's version of
fragmentation: an accepted inefficiency that keeps the memory-management
problem out of scope until M3, where paging is what actually removes it.

## Consequences

- `Sequence` (Phase 0) does not need a KV-buffer handle field after all —
  the buffer belongs to the current running *batch*, not to an individual
  sequence, and gets rebuilt on every membership change. What each sequence
  needs instead is its current row/slot index within that shared batch, a
  detail Phase 2's `Scheduler` owns.
- Phase 1's `ModelRunner` gains two batched primitives instead of one
  generic "forward step": a resync/prefill call (left-padded batch of full
  histories → next tokens + a fresh `StaticCache`) and a decode-step call
  (one token per row → next tokens, cache updated in place). Deciding *when*
  to call which is Phase 2's job, not Phase 1's.
- The M2 benchmark (Phase 4) will show this recomputation cost directly
  (throughput dips slightly around admission-heavy periods) — worth noting
  in the benchmark writeup as a legitimate, expected M2 characteristic, not
  a bug.

## What I learned

*(owner's to fill in, optional)*
