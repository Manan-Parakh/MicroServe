# Indexing Structures: Arrays, Hash Maps, and Page Tables

> When your key is already a small dense integer, hashing it is doing work to
> recover the index you started with.

---

## The Problem

Address translation is the hottest lookup in a paged system. MicroServe does
`block_table[pos // BLOCK_SIZE]` for every token position it scatters or gathers —
that is, on every layer, of every forward pass, for every sequence in the batch.
At 24 layers × 64 sequences it happens thousands of times per decode step, and
it's pure bookkeeping sitting between the GPU and its next kernel launch.

A hash map does this correctly. It also computes a hash, probes a bucket array,
compares a key, and touches memory that has nothing to do with the answer — to
retrieve a value that was sitting at a known offset the whole time. Meanwhile the
OS does the same translation in *hardware*, via a multi-level page table walk that
the MMU caches in a TLB, because the cost of getting this wrong compounds over
every memory access in the machine.

The decision rule is about the shape of the key space, and it's one of the few
places where the answer is genuinely absolute rather than a trade-off.

> 🔹 **TLB** — a small hardware cache of recent virtual→physical translations, so
> the CPU doesn't walk the page table on every access. OSTEP Ch 19.

---

## Mental Model

```
key space is 0..n-1, dense, small       ->  array.  index IS the lookup.
key space is sparse / opaque / huge     ->  hash map.
key space is sparse but you need ORDER  ->  sorted array or tree.
key space is huge and sparse but        ->  multi-level table / radix tree.
  CLUSTERED
```

"Clustered" is the one people miss, and it's the entire reason page tables are
multi-level rather than flat: a 64-bit address space is astronomically sparse,
but the pages a process actually uses come in a few dense clumps.

---

## How It Works — the four representations

### Direct-indexed array

The key *is* the offset. One dereference, no comparison, perfectly cache-friendly
when iterated in order.

- **Buys**: fastest possible lookup; zero per-entry overhead; deterministic
  iteration order for free (which is load-bearing whenever row order must match a
  tensor's row order).
- **Costs**: memory is `O(max_key)` whether or not the keys are used, so it
  degrades to mostly-holes the moment the key space is sparse. Keys must be
  non-negative integers.
- **In MicroServe**: `ref_counter[block_id]` and `Sequence.block_table[logical]`.
  Both key spaces are `range(n)` by construction.

### Hash map

Hash the key, probe, compare.

- **Buys**: O(1) average on any hashable key; memory proportional to entries, not
  key range.
- **Costs**: a hash computation and a likely cache miss per lookup; ~3x memory per
  entry versus an array; worst-case O(n) on adversarial or degenerate hashes;
  no ordering.
- **In MicroServe**: `seq_id → Sequence` once ids stop being small and dense
  (they're monotonic and never reused, so after an hour of serving the max id is
  huge and the live set is small). Also the natural home for M6's prefix-hash →
  block-id table, where the key is a content hash and sparse by definition.

### Sorted array + binary search

- **Buys**: contiguous like an array, and supports *range* queries an array and a
  hash map both can't — "the largest key ≤ k", "everything in [a, b)".
- **Costs**: O(log n) lookup, O(n) insert. Good for read-mostly data built once.
- **Where it shows up here**: latency percentiles in `bench/`. You sort the sample
  once and index p50/p99 — a heap would be the wrong tool because you want
  arbitrary quantiles from a static set, not a streaming minimum.

### Multi-level table / radix tree

An array of pointers to arrays. Only allocate the second level for regions
actually in use.

- **Buys**: the memory of a hash map with the *ordering and prefix structure* of
  an array; handles enormous, sparse, clustered key spaces. This is what a real
  page table is (x86-64 uses four levels).
- **Costs**: one pointer chase per level, so translation is a *walk* — which is
  precisely why the TLB exists. More complex.
- **In MicroServe**: not needed, and it's worth understanding why. A sequence's
  logical block space is dense and starts at 0 (positions 0..len), and the largest
  it gets is a few hundred entries. A flat list is not just adequate, it's optimal.
  The OS needs four levels because a process's virtual address space is 2^48 bytes
  with a few megabytes used; a sequence's "address space" is its own length.

---

## Usage

The address translation, which is the whole mechanism:

```python
BLOCK_SIZE = 16

logical_block = pos // BLOCK_SIZE          # which entry in the table
slot          = pos %  BLOCK_SIZE          # offset within the block
physical      = seq.block_table[logical_block]

# the pool read for one layer's keys at that token
pool[physical, 0, layer, :, slot, :]       # [num_kv_heads, head_dim]
```

Write it **once**, as a named helper. Inlined at three call sites, two of them
will eventually disagree — and a block-table off-by-one produces plausible wrong
tokens rather than a crash.

With `BLOCK_SIZE` a power of two the compiler (and CUDA) turn `//` and `%` into a
shift and a mask. That's a small reason to keep block sizes powers of two, and a
much bigger reason in the OS case where the MMU does it in hardware every cycle.

---

## Where This Fits in MicroServe

- `engine/sequence.py` — `block_table: list[int]`, Milestone 3 Phase 2. The list
  index *is* the logical block number, which is exactly why no dict is needed.
- `engine/kv_cache.py` — `ref_counter: list[int]`, direct-indexed by block id.
- `engine/engine.py` / `engine/scheduler.py` — `seq_id → Sequence` lookup, the one
  place a dict is the right answer.
- `bench/` — sorted arrays for percentiles.

The OS equivalent is OSTEP Ch 18–20: linear page tables (the flat array, rejected
for size), multi-level page tables (the radix tree, what x86-64 actually uses),
inverted page tables (a hash map, used when physical memory is the smaller space),
and the TLB as the cache that makes any of it viable. MicroServe's block table is a
*single-level* page table, and the reason it can be is that the logical address
space is tiny and dense — that's the exact property multi-level tables exist to
work around.

---

## Gotchas & Trade-offs

- **The block table's index must be the logical block number, not the position.**
  `block_table[pos]` and `block_table[pos // BLOCK_SIZE]` differ by a factor of 16
  and both are valid Python. This is the canonical M3 bug.
- **The growth boundary is `pos % BLOCK_SIZE == 0`, and off-by-one here is
  invisible.** Allocating the new block one token late means the last write of a
  block goes to `block_table[i]` at slot 16 — an `IndexError` if you're lucky, a
  silent write into the next tensor row if you're indexing a flat view. Test
  exactly at the boundary, one before, one after.
- **A dict for the block table also "works",** and costs you: a hash per token
  position, no ability to slice the table, and no natural way to hand the whole
  table to `torch.tensor()` for a batched gather. That last one is the practical
  killer — the batched gather needs the block ids as a tensor.
- **Sparse doesn't always mean hash map.** If the sparse keys are clustered *and*
  you need ordering, a multi-level table beats both. If they're sparse and you need
  ranges, a sorted structure beats both.
- **Iteration order.** Arrays iterate in index order for free. Python dicts iterate
  in insertion order (guaranteed since 3.7) — usable, but relying on it makes the
  invariant implicit rather than structural.
- **Don't index by `seq_id` into an array.** It's tempting because ids start at 0,
  but they're monotonic and never reused, so the array grows forever while the live
  set stays small. That's the textbook signature of "use a hash map."

---

## Comparison Table

| Representation | Lookup | Memory | Ordering | Pick when |
|---|---|---|---|---|
| Direct-indexed array | O(1), 1 deref | O(max_key) | free, index order | key is dense `range(n)` |
| Hash map | O(1) avg | O(entries) | none | key sparse or non-integer |
| Sorted array | O(log n) | O(entries) | full + ranges | read-mostly, need quantiles/ranges |
| Multi-level / radix | O(levels) | O(used regions) | prefix order | huge, sparse, **clustered** key space |
| Inverted (hash on frame) | O(1) avg | O(physical) | none | physical space ≪ virtual space |

---

## Interview & Career Angle

Page tables are one of the highest-yield OS topics for ML-systems interviews right
now, precisely because PagedAttention made the analogy load-bearing. Being able to
say "our block table is a single-level page table, and it can be because the
logical address space is dense and bounded by sequence length, whereas the OS
needs four levels for a 2^48 sparse space" demonstrates that you understand the
borrowed idea rather than just the vocabulary. The follow-up is always the TLB:
what's our equivalent of a translation cache, and do we need one? (Answer: not
really — we translate in Python once per step and then copy, so the translation
cost is amortised across a whole block, not paid per access.)

Adjacent concepts worth learning next:

- **TLB and translation caching** — why the page-table walk is affordable at all,
  and what "TLB reach" means.
- **[[free-lists-and-block-allocation]]** — where the physical ids in the block
  table come from.
- **Copy-on-write** — the M6 mechanism; two block tables pointing at one block.
- **Inverted page tables** — the hash-map representation, and the conditions that
  make it the right one.

---

## Hard Questions I Should Be Able to Answer

1. Your batched gather produces correct output for sequences shorter than 16
   tokens and garbage beyond. Name the three most likely bugs, in order, and how
   you'd distinguish them in one experiment each.
2. Why is a sequence's block table a flat list when an OS page table needs four
   levels? Give the property that differs, not the size.
3. M6: two sequences share a 4-block prefix. Draw what the two block tables look
   like, and say what has to happen the instant one of them writes to the last
   shared block.
4. A colleague proposes replacing `block_table: list[int]` with a dict so sparse
   sequences don't waste entries. What's wrong with the premise?
5. You need `seq_id → Sequence` lookups on the hot path and the ids are small
   integers starting at 0. Array or dict?

---

## Answers

**1.** In order of likelihood: **(a)** indexing the table by position instead of
logical block — `block_table[pos]` rather than `block_table[pos // 16]`. Under 16
tokens the table has one entry and `pos // 16` is 0, so both expressions agree at
pos 0 and only diverge once you have a second block; that's exactly the observed
symptom. Distinguish it by running a 20-token sequence with `BLOCK_SIZE = 4` — the
failure moves to token 4, proving it's block-boundary-relative, not a magic 16.
**(b)** The permute-before-reshape error in the gather: reshaping
`[n_blocks, heads, block, dim]` directly interleaves heads into the time axis, and
with a single block there's nothing to interleave, so it only appears at two
blocks. Distinguish by filling the pool with marker values (`block_id*100 + slot`)
and reading the gathered tensor by eye. **(c)** The growth rule firing one token
late, so the 17th token writes at slot 16 of block 0. Distinguish by asserting
`len(block_table) == ceil(seq_len / BLOCK_SIZE)` after every append.

**2.** Density, not size. A process's virtual address space is 2^48 bytes with a
few megabytes actually mapped, and those mappings are clustered (code low, stack
high, heap growing between) — a flat table would need 2^36 entries, almost all
empty, so the levels exist to skip unallocated regions wholesale. A sequence's
logical block space is `0 .. ceil(len/16)`, contiguous, fully populated by
construction, and bounded by a few hundred entries. There are no holes to skip, so
a level of indirection would buy nothing and cost a pointer chase. Put another way:
multi-level page tables are a *sparsity* optimisation, and our address space isn't
sparse.

**3.** Both tables hold the same four physical ids in entries 0–3, then diverge:
`A = [7, 2, 9, 4, 11, ...]`, `B = [7, 2, 9, 4, 18, ...]`. `ref_count` for blocks
7, 2, 9, 4 is 2. The moment one sequence needs to *write* into a shared block —
which happens on the first token that lands in a partially-filled shared block, not
on a read — you must copy-on-write: allocate a fresh block, copy the shared block's
contents into it, decrement the original's ref count, and patch that one entry in
the writer's table. Only the last shared block can ever be written, because the
earlier ones are full; that's what makes CoW cheap here. If you skip the copy, the
writer corrupts the other sequence's history, which is the same failure mode as a
double-freed block and just as silent.

**4.** The premise is wrong: a sequence's block table is never sparse. Entry *i*
covers positions `[16i, 16i+16)`, and a sequence occupies every position from 0 to
its current length — there are no gaps to be sparse about. The list is exactly as
long as `ceil(len/16)` and every entry is populated. The dict would add a hash per
lookup and lose the ability to pass the table straight to `torch.tensor()` for the
batched gather, in exchange for saving memory that isn't being wasted. (Sparsity
*could* arise under a swap-to-host preemption scheme where some blocks are evicted
— which is worth naming, because it's the one design where the premise would hold.)

**5.** Dict — because "small integers starting at 0" describes them *today*. Ids
are monotonic and never reused, so after an hour of serving the max id is in the
millions while the live set is 64. An array indexed by `seq_id` grows without
bound and is effectively a memory leak with extra steps. The dense-key rule
requires the key space to be dense *and bounded*, and `seq_id` is neither over the
process's lifetime. If you genuinely wanted array-speed lookup you'd introduce a
separate small dense "slot index" (0..max_batch_size) and map `seq_id → slot` once
at admission — which is exactly what the batch row index already is.

---

## Rules of Thumb

> Dense bounded integer key → array. Sparse or unbounded → hash map. Sparse but
> clustered and ordered → multi-level table.

> Write address translation exactly once, in a named helper. Three inlined copies
> means two future bugs.

> Test block-boundary arithmetic at the boundary, one before, and one after. That
> triple catches every off-by-one that matters.

> "It's currently dense" isn't the test. Ask whether it stays dense and bounded
> for the process's lifetime.

---

## TL;DR

- The block table is a single-level page table, and it's single-level because a
  sequence's logical address space is dense and bounded — the property that forces
  the OS into four levels is exactly the one we don't have.
- Direct-indexed arrays win whenever the key is `range(n)`: one dereference, no
  hashing, free iteration order, and the table converts straight to a tensor for
  the batched gather.
- `pos // BLOCK_SIZE` and `pos % BLOCK_SIZE`, written once. Nearly every M3
  correctness bug is a variation on getting that pair wrong.
