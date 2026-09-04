# ADR-0011: Paged KV module split, and explicit pool sizing

- **Status**: Accepted
- **Date**: 2026-08-30
- **Milestone**: M3
- **Concepts**: block/page, free-frame list, arena allocation, benchmark provenance

## Context

Two questions gated M3 Phase 1 (`docs/plans/m03-paged-kv-cache-plan.md`, Q4 and
Q2). Both are interface-shaped: they decide what the owner writes versus what
Claude writes, and they decide whether a benchmark number recorded in
`docs/benchmarks/` can be reproduced six weeks later.

Phase 0 changed the weighting on the first one. It established that the HF
framework seam is not a nuisance layer but a small contract with its own sharp
edges — `get_mask_sizes` is a promise made *before* `update()` runs, layers must
be pre-created or the promise is ignored on exactly the prefill call, and the
pool must be `torch.zeros` because an `inf` in a masked slot corrupts output
without raising. None of that has anything to do with allocating block ids.

## Options considered

### Option A — one module, `engine/kv_cache.py`
Pool, allocator, `Cache` subclass, scatter and gather all together, as the
repo layout in `claude.md` literally states.
- **Buys**: one file to hold in your head; the gather sits next to the pool it
  indexes into.
- **Costs**: the file imports `transformers`, so the pure-bookkeeping tests are
  model-free only by discipline, not by construction. Mixes owner-written and
  Claude-written code in one place.

### Option B — two modules
`engine/kv_cache.py` (pool, `BlockAllocator`, ref counts, index helper — no
`transformers` import) and `engine/paged_cache.py` (the `CacheLayerMixin` /
`Cache` subclasses and the scatter/gather).
- **Buys**: the split falls on the ownership line *and* the risk line at once.
  `tests/test_kv_cache.py` cannot accidentally acquire a model dependency.
- **Costs**: one more file; the index arithmetic is defined in one module and
  consumed in another.

### Option C — pool size from a VRAM fraction (vLLM-style)
`num_blocks` computed at startup from `gpu_memory_utilization=0.9` minus the
measured weight and activation footprint.
- **Buys**: the engine adapts to whatever GPU it lands on — relevant with a
  Legion Pro 5 incoming; it is also what production engines actually do.
- **Costs**: pool size becomes an emergent property of the machine's state at
  launch. Two runs of the same commit differ if something else was resident.

### Option D — explicit `num_blocks`, plus a sizing helper
`PagedKVPool(num_blocks=1024, spec=...)`, with `blocks_for_gib(gib, spec)`
available separately to pick the number.
- **Buys**: the pool size is a recorded input, not an observed output.
- **Costs**: you must size it yourself per machine, and an under-sized pool
  looks like a scheduler problem until you check.

## Decision

**Option B and Option D.** Two modules, split on the ownership line; and an
explicit `num_blocks` with `blocks_for_bytes` / `blocks_for_gib` as sizing aids
that are never called implicitly by the pool.

## Why the others lose

**A loses on test hygiene, not on aesthetics.** The allocator's failure mode is
a block id appearing twice in a list, and the symptom surfaces thousands of
forward passes later as a plausible wrong token. The only defence is a test file
that can run a 2000-step fuzz in under a second — which requires that importing
the module under test cannot pull in `torch.nn` and a model. Under A that
property holds by convention; under B the import graph enforces it.

**C loses on provenance, and only on provenance.** It is the better design for a
serving system, and it is the right call if this project ever grows one — M3
Phase 6's sweep is the thing that kills it here. That sweep holds VRAM fixed and
varies concurrency to find where paging stops helping; if the pool silently
resizes between runs, the independent variable is not actually held fixed and
the resulting chart is not a measurement. `claude.md` already forbids quoting a
number without commit and hardware provenance; a self-sizing pool makes those
two insufficient to reproduce a run.

Revisit C the moment an "Act Two" serving layer exists, or the moment sizing by
hand across two machines becomes a real friction rather than a hypothetical one.

## Consequences

- `engine/kv_cache.py` is guaranteed framework-free; `tests/test_kv_cache.py`
  runs with no model and no GPU.
- The seam between allocation and framework glue is explicit, which is where
  M6's copy-on-write will have to cut in.
- Benchmark configs must now record `--num-blocks` alongside commit and
  hardware. `blocks_for_gib` exists so that number is chosen, not guessed.
- Under-provisioning the pool is now possible and will present as constant
  preemption in Phase 4. The Phase 4 logging should therefore report
  `num_free` alongside its scheduling decisions, or the cause will be invisible.
