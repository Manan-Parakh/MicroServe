# ADR-0005: M2 KV cache ownership — own contiguous per-sequence buffer, not HF's `DynamicCache`

- **Status**: Accepted
- **Date**: 2026-08-21
- **Milestone**: M2
- **Concepts**: KV cache, `past_key_values`, block table (forward pointer to M3)

## Context

`docs/plans/m02-continuous-batching-plan.md` Phase 1 needs a way to run one
forward pass at a time (prefill or a single decode step) across multiple
sequences, since `model.generate()` can no longer be used once the scheduler
needs to interleave sequences. That forward pass needs *something* to hand
the model as `past_key_values`, and where that something comes from — and
who owns writing to it — is a real fork with consequences for how much M3
gets to reuse.

## Options considered

### Option A — Own contiguous per-sequence buffer
MicroServe pre-allocates a fixed-size KV tensor per `Sequence` at admission
time (shaped from the model's own config: `num_layers`, `num_heads`,
`head_dim`, times a max-length bound) and writes each step's new key/value
entries into it directly, then hands it to the model's `forward()`.
- **Buys**: this is a direct dry run for M3 — the `BlockAllocator` M3 builds
  swaps "one buffer per sequence" for "many small blocks shared from one
  pool," but the shape of what gets handed to `forward()` stays conceptually
  the same. It also means the project actually owns and understands its KV
  memory layout, which is the whole point of a project about learning how
  inference engines work.
- **Costs**: more code now — a buffer type, allocation logic, and a
  `ModelRunner` method built around raw tensors instead of `transformers`'s
  own cache abstraction.

### Option B — Reuse `transformers.DynamicCache` per sequence
Each `Sequence` holds its own `DynamicCache` instance, grown automatically
by the model's own `forward()` call. MicroServe never touches KV tensors
directly.
- **Buys**: less code right now, and no risk of a subtly-wrong manual
  tensor write breaking correctness.
- **Costs**: defers the "does this interface look like a block table"
  question entirely to M3. `DynamicCache` isn't block-shaped and doesn't
  expose a stable per-block handle — M3 would likely have to replace this
  layer outright rather than extend it, meaning the KV-management code gets
  written twice instead of once, incrementally.

## Decision

**Option A — own contiguous per-sequence buffer, written to directly.**

## Why the others lose

Option B optimizes for less code in M2 at the cost of the exact continuity
this project's milestone sequencing is designed around: M2 is supposed to
isolate the *scheduling* problem while M3 isolates the *memory* problem, but
that only works cleanly if M2's memory layer is a simplified version of
M3's, not a different abstraction M3 has to tear out. Since the owner is
explicitly building this to learn the mechanism (not just get a working
engine), leaning on `transformers`'s own cache management for M2 also hides
the exact bookkeeping M3's block table replaces — there'd be nothing to
compare M3's allocator against.

## Consequences

`ModelRunner` gains a method built around raw `past_key_values`-shaped
tensors rather than `transformers`'s `Cache` object, which means M2 has to
get the tensor shapes and `forward()` calling convention right by hand
(compute-bound risk area: this is exactly the kind of block-indexing-shaped
bug `claude.md` warns hides easily — correctness tests matter more here, not
less). In exchange, M3's `BlockAllocator` extends this milestone's buffer
concept instead of replacing it, and the owner has already worked through
the KV tensor shape and write pattern once before M3 makes it dynamic.

## What I learned

*(owner's to fill in, optional)*
