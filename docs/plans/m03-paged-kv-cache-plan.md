# M03 — Paged KV cache

**Status: COMPLETE — 2026-09-03.** All six phases built, the correctness gate
extended and passing (130 tests), the benchmark recorded with provenance, four
ADRs written. Nothing here is pending approval.

**This document is a build log, not a proposal.** It was written as a plan,
then annotated in place as each phase landed, and finally reconciled against
the measured results. The pre-build reasoning is deliberately left standing —
per `claude.md`, the rejected alternatives are the part worth keeping — but
every section now records what *actually* happened alongside what was intended.
Where reality diverged from the plan, the divergence is called out rather than
edited away. There are four such divergences and they are the most useful
paragraphs in the file:

1. **Phase 3's authorship changed mid-milestone** — the owner handed over the
  scatter/gather index math (see *Ownership*, below).
2. **ADR-0013 was reassigned.** Q2 folded into ADR-0011, so 0013 went to the
  scheduler decision instead.
3. **ADR-0014 did not exist in the plan at all.** "Equal VRAM" turned out not
  to be a setting either engine has.
4. **The benchmark workload the plan specified could not measure the claim the
  plan made.** Six requests cap concurrency at six. See *Phase 6 — results*.

**Where to start reading.** If you want to *run* it: **How to run this
milestone**. If you want to know whether it worked: **Phase 6 — results, read
honestly**, which also carries the one suspected implementation defect that is
flagged and not yet fixed. Everything before those two sections is the design
and its record.

**Milestone:** M3 (see `docs/roadmap.md` for the build spec and guardrail).
Per `claude.md`: *"Milestone 3 is the one that matters. If the timeline slips,
it slips on Milestone 6 — never on 3."*

**Scope, as built:** one pre-allocated KV tensor pool; a `BlockAllocator`
handing out fixed-size blocks from a free list; a per-`Sequence` block table
mapping logical token position → physical block id; a gather-then-attend path
that assembles a sequence's scattered blocks into a contiguous scratch tensor
before calling PyTorch's `scaled_dot_product_attention`; a fourth engine built
on all of it; and the memory-utilization benchmark that measures the point.

**Guardrail — held.** No custom CUDA or Triton kernel reads scattered blocks
directly from memory; gather-then-attend with stock
`scaled_dot_product_attention` proves the same idea without kernel authorship.
No Tensor Core instruction-level behavior, PTX/SASS, warp-level programming, or
occupancy tuning was entered. The one place the guardrail pulled hard was
Phase 6's performance finding, where the obvious next move is a fused gather
kernel — noted as a candidate, not built.

**The four decisions, and where they are recorded:**


|     | question                                         | resolution                           | ADR  |
| --- | ------------------------------------------------ | ------------------------------------ | ---- |
| Q1  | separate `PagedScheduler`, or generalize M2's?   | separate class                       | 0013 |
| Q2  | how is `num_blocks` chosen?                      | explicit, with a sizing helper       | 0011 |
| Q3  | what happens when blocks run out mid-generation? | preempt by recomputation             | 0012 |
| Q4  | how many modules?                                | two: pool/allocator, and the HF glue | 0011 |


---



## The reading gate (kept as written — it is still the right list)

Per `claude.md`'s just-in-time rule this was to be read before Phase 0 and not
earlier. It is preserved unedited because it doubles as the reference list for
anyone reviewing the code now: items 3 and 4 are the two things this milestone
implements, and a reviewer who has not read them will not be able to tell a
correct block table from a plausible one.

**The roadmap's theory boxes for M3 are deliberately still unticked.** A theory
box records having read the thing; that is the owner's to tick, not mine, and
ticking it on their behalf would make the roadmap lie about what happened.

1 and 2 are context; 3 and 4 are the two that were actually built against.

1. **Modal GPU Glossary** (`modal.com/gpu-glossary`) — **only** the *Device
  Hardware* pages (SM, CUDA core, warp scheduler, register file) and *Device
   Software* pages (thread, warp, block, grid, the memory hierarchy). **Skip
   every host-side API page.** Purpose here is narrow: you need to be able to
   say *where* the KV cache physically lives (GPU global memory / HBM) and why
   moving data between HBM and the SMs is the expensive part. You are not
   learning to program the GPU — the guardrail forbids that — you're learning
   the vocabulary that makes the next two readings land.
2. **Horace He, "Making Deep Learning Go Brrrr From First Principles"**
  (`horace.io/brrr_intro.html`) — the roofline model. Every op is either
   compute-bound or memory-bandwidth-bound, and most deep-learning ops are
   memory-bound. **Do the exercise before reading his answers:** classify a
   matmul, a ReLU, a layernorm, an elementwise add, and a softmax yourself.
   Why it matters here specifically: this milestone *adds* memory traffic (the
   gather copies blocks into a scratch buffer every layer, every step) in
   exchange for fitting more sequences. Phase 6's benchmark will show
   throughput per sequence get slightly *worse* while total system throughput
   gets much better. Without the roofline framing that result reads like a
   bug; with it, it reads like the expected trade.
3. **OSTEP Ch 13, 15, 18–19, 21–22** (`pages.cs.wisc.edu/~remzi/OSTEP/`) —
  this is roadmap item A1, and its trigger is *now*. Skim the mechanics
   (TLBs, multi-level page tables, swapping) — keep the concept: physical
   memory is chopped into fixed-size pages, and a page table maps a logical
   address to whichever physical frame currently holds it. **This is not an
   analogy for what you're about to build. It is the same data structure.**
   `Sequence.block_table` is a page table; `BlockAllocator`'s free list is a
   free-frame list; block size is page size; the "gather" is what a real MMU
   does in hardware and we have to do in Python because SDPA needs contiguity.
   Read this immediately before Phase 1 — it's the phase it maps onto.
4. **PagedAttention paper** (Kwon et al., 2023, `arxiv.org/abs/2309.06180`) —
  read **fully**, it's short. Every design choice in this milestone traces
   back to it. The number to anchor on: KV-cache memory utilization goes from
   ~20–40% (contiguous, reserve-for-worst-case — which is exactly what M2's
   `StaticCache` does today) to >90%, and that utilization converts directly
   into concurrent requests per GPU. Pay particular attention to §4.3–4.5
   (block table mechanics) and to the preemption discussion (recompute vs.
   swap) — that section is Q3 below, and you'll want to have read it before
   answering.
5. **vLLM's scheduler and block-manager source** — a design reference *during*
  this milestone, best opened after your own Phase 1 sketch exists, not
   before. `vllm/core/block_manager.py`, `vllm/core/block/*`,
   `vllm/core/scheduler.py`. **Read it, don't run it, don't copy it** (M0's
   guardrail against installing vLLM still stands). The specific thing worth
   stealing is *shape*, not code: notice how much of their allocator is
   ref-counting and copy-on-write machinery that we're deliberately deferring
   to M6, and how small the actual "hand out a free block" core is once you
   subtract that.

**What is deliberately NOT on this list:** FlashAttention's tiling algorithm,
CUDA kernel authorship, Tensor Core internals, and anything about
quantization (that's M4's gate). If reading item 1 or 2 makes any of those
look tempting, that's the guardrail's exact failure mode — note it as a "bite"
in `docs/bites/` and move on.

---



## The core mental model this plan is built around

> M2's KV cache reserves a contiguous rectangle sized for the *worst case*:
> `max(prompt_len) + max(max_new_tokens)` columns for every row in the batch, allocated up front, most of it never written. A sequence that stops after 8 tokens still holds a 60-token reservation. Paging replaces that rectangle with fixed-size blocks handed out **on demand, one block at a time, as a sequence actually grows** — and since blocks are indexed through a per-
> sequence table, they don't have to be adjacent in memory. The waste drops
> from "worst case minus actual" to "at most one partially-filled block per
> sequence," and the memory freed is spent on more concurrent sequences.

Every phase below either builds that mechanism or exists to measure it
honestly. The measurement that matters is **peak concurrent sequences at a
fixed VRAM budget**, contiguous vs. paged — not throughput.

---



## Locked decisions this plan honors

1. **No custom CUDA or Triton kernels** (`claude.md` locked decisions, and
  this milestone's guardrail). At attention time we gather a sequence's
   scattered blocks into a contiguous scratch tensor and call PyTorch's
   `scaled_dot_product_attention`. This proves the paging idea without kernel
   authorship.
2. **One pre-allocated KV tensor pool**, shape
  `[num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]`. A
   `block_id` is an index into dim 0. Never per-block `torch.empty`/`del` on
   the GPU — allocator overhead would swamp the very numbers this milestone
   exists to produce. **Correction to the roadmap's stated shape, worth
   flagging:** the roadmap writes `num_heads`; it must be
   `num_key_value_heads`. Qwen2.5-0.5B is a GQA model with 14 attention heads
   but only **2** KV heads (verified against the checkpoint config: 24 layers,
   14 attn heads, 2 KV heads, head_dim 64). Following the roadmap literally
   would allocate a 7× oversized pool and quietly invalidate every memory
   number this milestone produces. Not a decision — a fix.
3. **Block size starts at 16 tokens** (vLLM's default), as a named tunable
  constant in one place, not a magic number.
4. **Prefill and decode stay separate forward calls**, never fused (locked in
  `claude.md`, carried from M2).
5. `engine.add_request(prompt)` / `engine.step()` stays the public API surface.
  `engine/` stays synchronous and framework-free; nothing in `engine/` imports
   from `bench/`.
6. **Ref counts exist in the allocator from day one, but copy-on-write and
  prefix sharing do not.** M6 is the stretch milestone that uses them; a
   `ref_count` array that is always 0 or 1 in M3 costs one line and makes M6
   an extension rather than a rewrite. Anything beyond the counter itself —
   hash-based prefix matching, CoW block splitting — is M6 and out of scope
   here.
7. **The paged path is a fourth engine, not a replacement for the third.**
  `ContinuousEngine` stays exactly as built in M2. M5's Definition of Done
   asks for a four-way comparison chart (naive / static / continuous / paged);
   that requires all four to still run. Deleting M2's engine would delete the
   comparison this milestone exists to win.

---



## What existed before this milestone (the seams it cut into)

- `engine/model_runner.py` — `prefill_batch(token_id_lists, max_cache_len)`
and `decode_step(cache, last_token_ids, attention_mask)`. Both construct or
accept a `transformers` `StaticCache` and pass it as `past_key_values`.
**This is the seam this milestone cuts into**: the paged path needs these
same two primitives but with our own cache object in that slot.
- `engine/scheduler.py` — iteration-level admit/evict, capped by
`max_batch_size` alone. **The paged scheduler's admission decision is
different in kind**: it must also ask "are there enough free blocks?", and
handle the answer being no mid-generation, which cannot happen today.
- `engine/sequence.py` — `Sequence` has no memory-facing fields at all
(ADR-0007 concluded M2's buffer belongs to the *batch*, not the sequence).
M3 reverses that: with paging, memory genuinely is per-sequence again, and
`block_table` is the field that says so.
- `engine/kv_cache.py` — **does not exist yet.** M2's Q2 explicitly reserved
this filename for M3's real allocator. This is where Phase 1 lands.
- `tests/test_correctness.py` — the gate, 9 real-model tests, pinned to CPU +
eager attention (ADR-0010). Every paged code path must pass it before any
M3 number is quoted.
- `tests/test_scheduler.py`, `tests/test_static_engine.py` — the
`FakeModelRunner` pattern (no model, no GPU, sub-second). The allocator and
block table are pure bookkeeping and should be tested this way too, hard,
before any real model touches them.
- `bench/benchmark.py` — M2's mixed-`max_new_tokens` workload and its metric
set. M3 adds a *different* benchmark axis (memory), not a replacement.

---



## Ownership: what was planned, and what actually happened

The table below is the plan as written, under `claude.md`'s "never write a
milestone's core mechanism for them." **It did not survive Phase 3, and the
change was deliberate and is standing.**

Partway into Phase 3 the owner elected to hand the remaining core mechanisms
over — the scatter/gather index math, then the scheduler policy — trading
authorship for review: *"My coding skills have been brushed up, no need for me
to write again."* That is a permanent override of the prime directive's
authorship rule, not a one-off concession, and the rest of the milestone was
built on it. What the rule was protecting is now protected a different way: by
**Phase 3's and Phase 4's "review notes" sections written into the source
files themselves**, one per phase, naming the non-obvious invariants and the
bugs each function is positioned against.

The three that most repay reading before reviewing the code:

- `permute` **before** `reshape`**.** `reshape` walks row-major, so folding the
block axis into time while heads still sit between them interleaves heads
into time. Every shape checks out; the output is silently wrong.
- **Advanced indices separated by a slice broadcast to the front.**
`pool[block_table, :, layer_idx]` yields `[B, nblk, 2, H, block, dim]` — not
`[2, B, nblk, ...]`, which is what you would guess.
- **Capacity is not content.** `BlockManager` grants room; the *caller*
advances `num_cached_tokens` after a successful forward. The invariant is
`0 <= num_cached_tokens <= len(block_table) * BLOCK_SIZE` — capacity runs one
step ahead, never behind. Preemption-by-recomputation drops KV while keeping
tokens, and that state has to be expressible.

`docs/handouts/m03-phase3-paged-cache.md` is superseded but kept: it is the
written argument for *why* those two functions are the milestone, which is
still the right thing to read before reviewing them.

**The plan as written:**


| Component                                                    | Who                                   |
| ------------------------------------------------------------ | ------------------------------------- |
| `BlockAllocator` — free list, alloc/free, ref counts         | **You.** This is the milestone.       |
| Block table indexing — logical position → (block_id, offset) | **You.** The bug-farm, and the point. |
| Gather/scatter index math in the paged cache                 | **You.**                              |
| Paged admission + preemption policy in the scheduler         | **You**, after we design it together. |
| HF `Cache`-API boilerplate around your gather/scatter        | Me. Framework plumbing, not the idea. |
| Pool allocation, sizing helpers, constants                   | Me.                                   |
| Test scaffolding, fakes, fixtures                            | Me.                                   |
| Benchmark plumbing, metrics, output shape                    | Me.                                   |
| Docs, ADRs, concepts entries                                 | Me, from your decisions.              |


*Outcome: Phase 0 ran as planned (a throwaway spike, and it paid for itself — see its findings). Phases 1, 2 and 3 were the owner's. Phase 4 onward were mine, under the waiver above, with review in place of authorship.*

---



## Phase 0 — Spike: prove the HF cache seam holds  ✅

**Goal:** before building an allocator for a cache we can't actually plug in,
confirm that a custom cache object *can* be handed to this model as
`past_key_values` and can return a gathered contiguous K/V from `update()`
without fighting `transformers`' mask machinery.

**Why this phase exists — the honest risk:** M2 already got bitten once here
(ADR-0007: `StaticCache`'s single shared write-position was discovered
mid-build, not at design time). In `transformers` **5.15.1** a `Cache` is a
container of per-layer `CacheLayerMixin` objects, and `Cache.update(k, v, layer_idx, ...)` dispatches into the layer. Separately, the model's masking
utilities inspect the cache to decide the attention mask's shape. A paged
cache that returns a gathered buffer of a *different* length than the mask
expects will either crash or — much worse — silently attend to garbage. Two
days spent finding that in Phase 4 is two days; twenty minutes finding it now
is twenty minutes. **This phase writes no allocator and no block table.** It
writes the dumbest possible custom cache — one that stores K/V contiguously
exactly like `DynamicCache`, adds nothing, and is subclassed purely to prove
the seam — and checks it produces byte-identical tokens to the current path.

- ✅ A throwaway `PassthroughCache` in a scratch file (not in `engine/`):
correct `update()` signature, contiguous storage, no paging.
- ✅ Run it through `ModelRunner`'s existing prefill/decode primitives on the
tiny model, CPU + eager (the gate's config), single sequence.
- ✅ Same again batched, two sequences of different prompt lengths.
- ✅ Write down exactly which cache methods `transformers` actually calls, in
what order, and what the masking path reads off the cache. That note is the
spec Phase 3 implements against.

**Files:** scratch only (`scratchpad/`), plus a findings note that becomes
Phase 3's spec.
**Concepts:** the `Cache` API as a framework seam; why an integration spike
precedes a design when the framework's contract is unverified.
**Done when:** a hand-written cache class produces token-identical output to
the current `StaticCache` path, batched, and we can name every method
`transformers` requires of it.
**Test:** none committed — this is a spike. Its output is knowledge and a
note, and the code is deliberately thrown away.

### Phase 0 findings — this is Phase 3's spec  ✅

Ran three throwaway probes on CPU + eager, tiny model, against the
`StaticCache` path as reference. All code lives in `scratchpad/` and is not
committed.

**1. The seam is small, and it is the *layer*, not the** `Cache`**.** In
`transformers` 5.15.1, `Cache` is a nearly-pure container: `Cache.update(k, v, layer_idx, ...)` forwards straight to `self.layers[layer_idx].update(k, v)`.
Everything we need to override lives on a `CacheLayerMixin` subclass, which
has exactly five abstract methods plus one class attribute:


| **x e**member                                            | what it must do for us                                                   |
| -------------------------------------------------------- | ------------------------------------------------------------------------ |
| `lazy_initialization(k, v)`                              | learn dtype/device from the first real tensor                            |
| `update(k, v, *a, **kw) -> (keys, values)`               | scatter-write new K/V into blocks, return the gathered K/V for attention |
| `get_mask_sizes(query_length) -> (kv_length, kv_offset)` | **declare how wide the returned K will be**                              |
| `get_seq_length() -> int`                                | tokens currently cached                                                  |
| `get_max_length() -> int`                                | `-1` for "no fixed maximum"                                              |
| `is_sliding = False`                                     | class attribute; Qwen2.5 is full-attention                               |


A `PassthroughLayer` storing K/V contiguously (a re-implementation of
`DynamicLayer`, no paging) produced **token-identical output to** `StaticCache`
both single and batched-uneven, 12 steps. The seam holds.

**2.** `get_mask_sizes` **is a contract, and violating it crashes loudly rather
than corrupting silently.** The 4-D causal mask is built once per forward from
`get_mask_sizes` + the 2-D `attention_mask`, *before* `update` runs. Returning
K/V of a different length than declared raises
`RuntimeError: The size of tensor a (16) must match the size of tensor b (14)`.
This is the good outcome — ADR-0007's failure mode was a *silent* one. Phase 3
can iterate fast here.

**3.** `Cache.get_mask_sizes` **short-circuits on an empty layer list — so
pre-create the layers.** With `layer_class_to_replicate` (the `DynamicCache`
style), layers are appended lazily on first `update`, so during **prefill**
`layer_idx >= len(self.layers)` and `Cache.get_mask_sizes` returns
`(query_length, 0)` *without ever consulting our layer*. Our declared length is
ignored exactly on the call where the cache is empty. **Decision for Phase 3:
construct** `PagedCache` **with** `layers=[PagedLayer() for _ in range(num_layers)]`**,
never** `layer_class_to_replicate`**.**

**4. A block-granular gather is legal — we never have to trim.** With layers
pre-created, returning K/V padded up to a multiple of `BLOCK_SIZE` — with
*deliberately loud* finite garbage (`randn * 100`) in the tail — and declaring
that padded length from `get_mask_sizes` gives **token-identical output**.
Causality masks the tail, because every padded position is beyond every
query's position. This is a real simplification: the gather can be
`pool[block_ids]` reshaped, with no ragged per-sequence slicing.
Observed `get_mask_sizes` sequence: `(q=14, true=14, declared=16)`,
`(1, 15, 16)`, `(1, 16, 16)`, `(1, 17, 32)`.

**5. The pool must be zero-initialised, never** `torch.empty`**.** Same probe with
`inf` in the padded tail instead of finite garbage: output collapsed to all
token id `0`. Masked positions get weight zero, and `0 * inf = NaN`, which
propagates through the weighted sum. Finite junk in a masked slot is harmless;
`inf`/`NaN` is fatal and *does* fail quietly-ish (no exception, just wrong
tokens). Since a reused block always holds a dead sequence's finite KV, the
only source of `inf`/`NaN` is uninitialised memory — so `torch.zeros` at pool
creation closes the hole permanently, and `free()` does **not** need to zero.

**Caveat to re-check later:** findings 4 and 5 were verified with
`attn_implementation="eager"` on CPU (the correctness gate's config, ADR-0010).
Re-verify the block-padded gather under SDPA before quoting any GPU benchmark
number from the paged path.

---



## Phase 1 — `BlockAllocator` and the pool  ✅

**Goal:** one pre-allocated tensor pool, and an allocator that hands out and
reclaims block ids from it in O(1), with reference counts present but always
0 or 1.

**The milestone's first core mechanism, and the owner's to write — it was.**
OSTEP (reading item 3) was the gate for this phase: the free list here *is* a
free-frame list and the correspondence is exact, not analogical.

**Q4 and Q2 resolved before starting — see ADR-0011.** Two modules, split on the
ownership line: `engine/kv_cache.py` (pool + allocator, no `transformers`
import) and `engine/paged_cache.py` (Phase 3's framework glue). Pool size is an
explicit `num_blocks`, with `blocks_for_gib()` as a sizing aid — a VRAM-fraction
pool would make Phase 6's fixed-VRAM sweep unreproducible.

- ✅ `BLOCK_SIZE = 16` as a named module constant (locked decision 3).
- ✅ Pool allocated once: `[num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]`, dtype and device matching the model. Dim 1 is the
K/V pair. **Note the ordering choice** — `num_blocks` first means a block id
is a plain index into dim 0 and `pool[block_id]` is a contiguous slice,
which is what makes gather cheap.
- ✅ `num_blocks` passed explicitly; `blocks_for_bytes` / `blocks_for_gib`
provided as sizing aids the pool never calls implicitly (Q2 → ADR-0011).
- ✅ Free list as a stack (`list[int]`, push/pop) — O(1) both ways, and LIFO
reuse keeps recently-touched blocks warm. A `deque` used FIFO would be
equally correct and slightly worse for locality; a set would be O(1) but
loses that ordering property for nothing.
- ✅ `allocate(n) -> list[int]`, `free(block_ids)`, `num_free` — and an
explicit, *catchable* failure when `n > num_free`. Do not let this raise a
bare `IndexError` from `pop()` on an empty list: the scheduler in Phase 4
needs to handle exhaustion as a normal, expected control-flow event, not an
accident. That is the whole reason preemption exists.
- ✅ `ref_count`: a plain list/array indexed by block id. `allocate` sets 1,
`free` decrements and returns the block to the free list only at 0. Always
0-or-1 in M3 (locked decision 6) — the structure is here so M6 is an
extension, not a rewrite.

**Files:** `engine/kv_cache.py` (new), `tests/test_kv_cache.py` (new).
**Concepts:** block/page; free list; the KV pool as an arena allocator; why
allocator overhead on the hot path is itself a benchmark artifact.
**Done when:** blocks can be allocated to exhaustion, freed, and reallocated,
with `num_free` correct at every step and exhaustion raising something the
scheduler can catch.
**Test:** `tests/test_kv_cache.py` — pure bookkeeping, no model, no GPU, in
the `FakeModelRunner` spirit. Allocate-to-exhaustion, free-and-reuse,
double-free is an error, ref counts land where expected, `num_free` is exact
after every operation. **This file should be the most paranoid test file in
the repo** — `claude.md` calls block-indexing bugs "exactly the kind that
hide," and this is the layer where hiding is cheapest.

---



## Phase 2 — `Sequence.block_table`  ✅

**Goal:** each `Sequence` carries its own logical→physical mapping, and grows
it by one block only when it actually needs one.

**Decisions taken before starting:** a thin `BlockManager` in
`engine/kv_cache.py` owns the allocator and the growth rule; `Sequence` stays a
passive dataclass carrying `block_table` and an explicit `num_cached_tokens`
(not derived from the token lists — preemption-by-recomputation drops KV while
keeping tokens, and that state must be expressible). `BlockManager`'s methods
are about **capacity, not content**: the caller advances `num_cached_tokens`
after a successful forward pass, so the invariant is
`0 <= num_cached_tokens <= len(block_table) * BLOCK_SIZE` — capacity runs one
step ahead, never behind.

**The second core mechanism, also the owner's, also written by them.** It is
small — a list and one piece of index arithmetic — and it is where the subtle
bugs live. `tests/test_kv_cache.py` grew to 55 tests around it for that reason:
`claude.md` calls block-indexing bugs "exactly the kind that hide", and this is
the layer where hiding is cheapest.

- ✅ `Sequence.block_table: list[int]` — index *i* holds the physical block id
storing logical token positions `[i*BLOCK_SIZE, (i+1)*BLOCK_SIZE)`. The list
index *is* the logical block number; no dict needed.
- ✅ The index math, both directions: given logical position `p`,
`block_table[p // BLOCK_SIZE]` and offset `p % BLOCK_SIZE`. Write it once,
in one place, as a named helper — not inlined at three call sites where two
of them will eventually disagree.
- ✅ Growth rule: a sequence needs a new block only when its next token would
land at offset 0 of a block it doesn't have. Prefill allocates
`ceil(prompt_len / BLOCK_SIZE)` at admission; decode allocates at most one
more per step, and usually zero.
- ✅ Freeing: a `FINISHED` sequence returns all its blocks, via
`BlockManager.free(seq)` called from the scheduler's eviction path. Same call
serves Phase 4's preemption path for a *not*-finished sequence, which is why it
resets `num_cached_tokens` rather than assuming completion. (Q4 follow-up,
resolved: a thin `BlockManager` owns the growth rule so it isn't expressed once
at admission and again in decode.)
- ✅ **Internal fragmentation is expected and must be measured, not hidden**:
the last block of every sequence is partially filled, wasting on average
`BLOCK_SIZE/2` token-slots per sequence. That's the entire remaining waste
after paging, and Phase 6 should report it as a number rather than let it be
assumed to be zero.

**Files:** `engine/sequence.py`, `engine/kv_cache.py` (the helper may live
with the allocator instead — Q4).
**Concepts:** block table as a page table; logical vs. physical address space;
internal fragmentation as the residual cost of paging.
**Done when:** a sequence can be grown token by token past several block
boundaries and its block table stays consistent with what the allocator thinks
it handed out. **Met** — 26 Phase 2 tests, 55 in the file, 77 in the suite.
**Test:** extend `tests/test_kv_cache.py` — grow across a block boundary
exactly, one before, one after (off-by-one at `p % BLOCK_SIZE == 0` is the
canonical bug); free and confirm every block returns; two sequences
interleaved never receive the same block id.

---



## Phase 3 — `PagedCache`: scatter-write, gather-then-attend  ✅

**Goal:** the object handed to the model as `past_key_values`. On `update()`
for a layer it (a) writes this step's new K/V into the right blocks at the
right offsets, and (b) returns the sequence's full history gathered into a
contiguous scratch tensor for `scaled_dot_product_attention`.

**Split of work (revised mid-phase).** Originally the HF `Cache` boilerplate
was mine and the scatter/gather index math was the owner's. The owner elected
to hand the index math over as well — time pressure, and the coding skills the
exercise was protecting are already back — with review in place of authorship.
`docs/handouts/m03-phase3-paged-cache.md` is superseded; it survives as the
written argument for *why* these two functions are the milestone, which is
still the thing to read before reviewing them.

**Decisions taken before starting.** (a) The gather runs **per layer**, 24
small copies rather than one big one: same bytes moved, but peak scratch is
1/24th, and scratch competes with the pool for the VRAM this milestone exists
to reclaim. (b) Prefill stays **batched**, so M3 and M2 differ in memory
strategy alone and the M5 chart doesn't accidentally compare prefill batching;
the price is a per-row-irregular scatter. (c) Rows are **right-padded** in the
paged path, mirroring M2's left-padding — a paged sequence's KV starts at
logical position 0 whatever the query padding, so left-padding would set the
block table against the causal mask. The cost is `logits[:, -1, :]` no longer
reading every row, hence `PagedBatch.last_logit_index()`. (d) The gate is
**two-tier**: a no-model shuffled-block test that names the bug, plus the
real-model token match.

- ✅ **Scatter (write):** new K/V for each row goes to
`pool[block_id, :, layer_idx, :, offset, :]`. Batched, this is an
`index_copy_`/advanced-indexing write over a precomputed slot-index tensor,
not a Python loop over rows — a per-row loop would work and would also make
the allocator's own cost visible in the benchmark, which defeats the point.
- ✅ **Gather (read):** for each row, collect its blocks in block-table order
into `[batch, num_kv_heads, padded_len, head_dim]`, then hand that to SDPA
exactly as today. **Say the cost out loud:** this copy happens per layer,
per step. It is real memory traffic that M2's path does not pay, and it is
the price of the concurrency win. Reading item 2 (Horace He) before this
phase is what makes that trade legible rather than alarming.
- ✅ **Padding does not disappear.** The gathered scratch buffer is still
rectangular, padded to the batch's longest sequence, because SDPA needs it
that way. What paging removes is *reservation* waste in the pool — memory
held for tokens that may never exist. Worth stating explicitly in the
README, because "paging eliminates padding" is the natural wrong summary and
it would make Phase 6's numbers look like an underperformance.
- ✅ Correct `get_seq_length()` / whatever else Phase 0 found the mask path
reads. Get this wrong and the failure is silent-wrong-tokens, not a crash.

**Files:** `engine/paged_cache.py` (Q4, ADR-0011), `engine/model_runner.py`
(the `paged_prefill` / `paged_decode_step` pair), `tests/test_paged_cache.py`.
**Concepts:** scatter/gather; gather-then-attend as the no-kernel realization
of PagedAttention; memory traffic vs. memory capacity as opposing pressures.
**Done when:** a single sequence prefilled and decoded through `PagedCache`
produces tokens identical to the golden reference.
**Met** — 101 tests green, including both Tier 2 gate prompts (one sized to
cross a block boundary mid-decode) and the untouched M1/M2 correctness gate.
**Test:** the real gate — `tests/test_correctness.py`, extended. Also worth a
fake-model unit test that a scattered gather reassembles a known pattern
(write recognizable integers into blocks in a deliberately shuffled order,
gather, assert the original sequence comes back). That test catches
block-order bugs without a model in the loop.

---



## Phase 4 — `PagedEngine`: admission gated by memory  ✅

**Goal:** the fourth engine. Same `add_request`/`step()` shape; different
admission rule and a failure mode that has never existed in this codebase
before.

**The genuinely new idea, and why it's not just a bigger** `if`**:** M2's
scheduler admits on one condition — is there a slot under `max_batch_size`.
A paged scheduler admits on two: is there a slot, *and* are there enough free
blocks for the prompt. And unlike the slot cap, block availability can run out
**mid-generation** — every running sequence is still growing, and a step can
need a block that doesn't exist. That's a preemption decision, and it's Q3.

- ✅ Admission: slot available **and** `num_free >= ceil(prompt_len / BLOCK_SIZE)`. Otherwise the sequence stays `WAITING` — no partial admission.
- ✅ Per-step block growth for each running sequence.
- ✅ Exhaustion handling per Q3.
- ✅ `max_batch_size` **becomes an upper bound, not the binding constraint.**
In M2, `peak_concurrent_sequences` just reported back `max_batch_size` — the
benchmark README already flags that caveat and defers the real answer to
this milestone. In M3 the binding constraint should be memory. Setting
`max_batch_size` low enough to bind again would silently un-measure the
entire milestone.
- ✅ Does the resync-prefill from ADR-0007 survive? **Probably not, and that's
a genuine secondary win worth measuring.** M2 rebuilds the whole
`StaticCache` on every membership change because a shared write-position
can't track per-row positions. A paged cache has per-sequence block tables —
each row's position is its own — so an eviction need not invalidate anyone
else's KV. If that holds, the paged engine drops M2's most expensive
recurring operation. It also might not hold once the mask path is involved.
Do not assume it either way: land Phase 4 with the resync intact and
correct, *then* try removing it as a separate, separately-verified change.

**Decisions taken.** Q1 → a **separate** `PagedScheduler` (ADR-0013): M5's
four-way chart needs M2's scheduler byte-identical to the version its numbers
came from, and preemption is a control-flow change rather than an admission
predicate, so the "policy object" abstraction would have complicated M2 to
accommodate M3. Q3 → **preemption by recomputation** (ADR-0012): on a laptop
the scarce resource is the PCIe link, not compute, and swapping would add a
host pool, a host allocator, a block map, and a transfer in the step()
critical path — the least-tested code in the engine holding the most
invariants. Step shape → **prefill-priority**, one batch kind per step
(ADR-0013); the TPOT cost is real and named rather than smoothed over.

**Resync: gone, and confirmed rather than assumed.** The plan said to land
Phase 4 with ADR-0007's resync intact and remove it separately. In the paged
design there was nothing to land — the cache holds no per-sequence state, so
"rebuilding" it is not an operation that exists. Verified instead by test:
`test_one_sequence_finishing_does_not_disturb_another` asserts that a
survivor's block table is untouched and that **no prefill call is issued on
its behalf** when a neighbour retires. A separable win of M3; report it apart
from the paging number.

**Files:** `engine/paged_engine.py` (new), `engine/paged_scheduler.py` (new),
`tests/test_paged_scheduler.py` (new), plus an additive `num_tokens` argument
on `BlockManager.allocate_for_prefill` so a recomputed sequence is sized for
prompt + generated.
**Concepts:** memory-pressure-bound admission; preemption; recomputation vs.
swapping as the two ways to make room.
**Done when:** a workload larger than the pool can hold runs to completion
correctly, with sequences waiting (or being preempted) rather than crashing.
**Met** — 21 Phase 4 tests, 122 in the suite. Eight sequences through a pool
that holds two of them drain completely with the pool returned empty, and
`test_preemption_does_not_change_the_tokens` shows a preempted-and-recomputed
sequence matching HF `generate()` exactly on the real model.
**Test:** `tests/test_paged_scheduler.py` with a fake runner and a
deliberately tiny pool — the exhaustion path must be exercised on *every* run,
not only when someone happens to configure a small pool.

---



## Phase 5 — Correctness gate on the paged path  ✅

**Goal:** `tests/test_correctness.py` passes for `PagedEngine`, under
conditions that actually stress the block machinery.

**Why a phase and not a checkbox:** the existing multi-sequence test uses
short prompts and `max_new_tokens=[8, 20]`. With `BLOCK_SIZE = 16`, that
barely crosses a block boundary — it would pass while the indexing was
substantially wrong. The gate needs new cases *designed against this
mechanism*, or it is not a gate.

- ✅ Add `PagedEngine` to the existing parametrized engine tests.
- ✅ New cases chosen adversarially: a prompt of exactly `BLOCK_SIZE` tokens;
one of `BLOCK_SIZE + 1`; generation that crosses two or more boundaries;
two concurrent sequences whose blocks necessarily interleave in the pool.
- ✅ A case where a sequence finishes, its blocks are freed and **reused by a
later sequence**, and the later one's output is still exactly right — the
test that catches stale-KV-from-a-dead-sequence, which is the single
nastiest bug this milestone can produce and the one most likely to look like
"the model got a bit worse."
- ✅ Stays on CPU + eager attention (ADR-0010). Non-negotiable and not worth
re-testing.

**Files:** `tests/test_correctness.py`.
**Concepts:** adversarial test design against a known failure mode.
**Done when:** all existing tests plus the new paged cases pass.
**Met** — 17 tests in `test_correctness.py`, `PagedEngine` added to the
parametrized multi-sequence case, plus five new adversarial cases. The one
that matters most is
`test_paged_reuses_a_dead_sequences_blocks_without_stale_kv`: a finished
sequence's blocks return to the free list *still holding its KV* (`free()`
deliberately does not zero them), the next sequence is handed those same
physical blocks, and its output must still be exactly golden. Reuse is
asserted rather than assumed, so an allocator change cannot quietly leave the
test passing while testing nothing.
**Test:** this phase *is* the test.

---



## Phase 6 — The memory-utilization benchmark  ✅

**Goal:** the roadmap's Definition of Done — higher achievable concurrent
batch size at the same VRAM budget, contiguous vs. paged.

**The headline number is not throughput.** It's *how many sequences fit*.
Throughput may even dip slightly per sequence (Phase 3's gather cost);
total system throughput should still rise because more sequences run at once.
Reporting only throughput would hide both halves of the real story.

- ✅ Fix a VRAM budget for the KV cache. Sweep offered concurrency upward for
`ContinuousEngine` (contiguous) and `PagedEngine` until each stops fitting.
- ✅ Report: peak concurrent sequences, **KV memory utilization** (token-slots
actually holding a real token ÷ token-slots allocated — the paper's ~40% vs.
  > 90% number, computed for our own engines), internal fragmentation
  > (Phase 2), and the four standard metrics for continuity with M2.
- ✅ Same mixed-length workload as M2's Phase 4, so the numbers compose with
the existing run rather than replacing it.
- ✅ Landed in `docs/benchmarks/<timestamp>-m3-paged-vs-contiguous/` with
commit + hardware, per the provenance rule. GPU-only.
- ✅ Sanity cross-check, same as M2's: `total_tokens` identical across
engines. Different memory strategies, identical output.

**Methodology decision (ADR-0014).** "Equal VRAM" is not a setting either
engine has -- `ContinuousEngine` takes `max_batch_size`, `PagedEngine` takes
`num_blocks`. A byte budget is converted into each engine's own currency, and
the contiguous derivation is *verified* against the tensors `StaticCache`
actually allocates rather than trusted.

**Two workloads, and the reason is a finding.** The plan specified reusing
M2's six-request workload so the numbers compose. They do -- but six requests
cap peak concurrency at six, so above 3 MiB the ratio collapses to 1.0x. That
is a fact about the workload, not about paging. So the sweep is kept for
continuity and a **validation point** (64 MiB, 128 requests, long-tailed
lengths) was added where both engines actually saturate. See ADR-0014.

**Files:** `bench/paged_benchmark.py` (new -- *not* `bench/benchmark.py`,
which stays frozen as the instrument that produced the recorded M2 numbers;
same reasoning as ADR-0013), `docs/benchmarks/2026-09-03_15-57-52-m3-paged-vs-contiguous/`.
**Concepts:** KV-cache memory utilization; capacity vs. bandwidth as the two
distinct memory costs.
**Done when:** a table shows paged fitting materially more concurrent
sequences than contiguous at equal VRAM, with utilization numbers explaining
why.
**Met** — **4.92x concurrent sequences at equal VRAM** (26 -> 128 at 64 MiB),
with reserved-slot occupancy **42% -> 90%**. The occupancy number explains the
concurrency number rather than corroborating it independently. Throughput
163 -> 466 tok/s. `total_tokens` identical across engines at every point.
Three caveats recorded in the run's own README: 4.92x is a *floor* (paged never
ran out of blocks, so its ceiling was not reached); the TTFT p99 result
(38.6 s -> 0.46 s) belongs to the absence of ADR-0007's resync and **not** to
paging; and the flat TPOT does not mean the gather is free, only that it is
small next to weight traffic at this context length.
**Test:** measurement, not correctness — Phase 5 is what licenses quoting it.

---



## How to run this milestone

Everything below is the exact set of commands; `uv run` is the project's
runner (Open decision #4, resolved to `uv` + ruff + pytest).

### The gate — run this before quoting any number

```bash
uv run pytest -q                                    # whole suite, 130 tests, ~2.5 min
uv run pytest tests/test_correctness.py -v          # the gate alone (CPU + eager, ADR-0010)
uv run pytest tests/test_kv_cache.py -q             # Phases 1-2, no model, sub-second
uv run pytest tests/test_paged_cache.py -q          # Phase 3 scatter/gather, mostly no model
uv run pytest tests/test_paged_scheduler.py -q      # Phase 4, fake runner + tiny pool
uv run ruff check .                                 # line-length 100, E/F/I/UP
```

`tests/test_kv_cache.py`, `tests/test_paged_cache.py` (Tier 1) and
`tests/test_paged_scheduler.py` (Tier 1) run with **no model and no GPU** — the
`FakeModelRunner` pattern carried over from M2. They are the fast inner loop;
`test_correctness.py` is the slow outer gate.

### See the mechanism run

```bash
uv run python -m engine.paged_engine
```

The demonstration block at the end of `engine/paged_engine.py`. Five scenes,
one per phase, on CPU + eager so scene 5's golden comparison is a real
comparison and not a coin flip on kernel reduction order:

1. **The pool (Phase 1)** — shape `(48, 2, 24, 2, 16, 64)`, 12 KiB/token,
  9.00 MiB for 48 blocks = 768 token slots.
2. **Address translation (Phase 2)** — `position -> (logical_block, slot)` at
  0, 15, 16, 17, 33, next to `blocks_needed`. The off-by-one at
   `p % BLOCK_SIZE == 0` is visible rather than asserted.
3. **Block tables (Phases 2-3)** — two sequences filling concurrently, printing
  e.g. `[47, 45]` and `[46, 44]`: **physical ids are not in order and not
   adjacent**, which is the entire claim of paging in one line of output. Also
   prints internal fragmentation and pool utilization.
4. **Preemption (Phase 4)** — a 2-block pool holding two sequences either of
  which fits alone. Prints the step number a preemption fires on, and that the
   victim keeps its generated tokens while its KV is discarded.
5. **Correctness (Phase 5)** — the preempted sequence's output compared against
  HF `generate()`. This one **asserts**, so the demo fails loudly if the claim
   ever stops being true.

It is a demonstration, not a test. `tests/` is what gates the milestone.

### Reproduce the benchmark

```bash
# the recorded run, exactly:
uv run python -m bench.paged_benchmark --validate --validate-requests 128

# sweep only (fast) -- the M2-composable six-request workload
uv run python -m bench.paged_benchmark

# validation point only, at a different budget
uv run python -m bench.paged_benchmark --skip-sweep --validate \
    --validate-budget-mib 128 --validate-requests 256

# custom budget sweep
uv run python -m bench.paged_benchmark --budgets-mib 1 2 4 8 16
```

Output lands in `docs/benchmarks/<timestamp>-m3-paged-vs-contiguous/` as
`results.json` (sweep), `validation.json` (validation point) and a generated
`README.md`. **GPU-only** — the benchmark is not meaningful on CPU.

> **Provenance wart, worth fixing before M4.** `--validate-requests` defaults
> to **48**, but the recorded run used **128**. Running the tool with its own
> defaults does not reproduce the recorded numbers, and the run's README does
> not print the command that made it. The fix is one line — record `sys.argv`
> into the JSON — and it should happen before M4 produces another run.

---



## Phase 6 — results, read honestly

Full data: `docs/benchmarks/2026-09-03_15-57-52-m3-paged-vs-contiguous/`.
Methodology: **ADR-0014**. RTX 3050 Ti Laptop (4 GiB), Qwen2.5-0.5B-Instruct,
bf16, commit `cceca99`.

### Did the milestone achieve what it set out to?

**Yes, on the Definition of Done — and the honest version of the result is
narrower than the headline.**

The roadmap asks for *higher achievable concurrent batch size at the same VRAM
budget*. At a 64 MiB KV budget, 128 requests, long-tailed generation lengths:


|                                 | contiguous          | paged            | ratio              |
| ------------------------------- | ------------------- | ---------------- | ------------------ |
| derived config                  | `max_batch_size=26` | `num_blocks=341` | —                  |
| KV actually held                | 61.85 MiB           | 63.94 MiB        | both within 64 MiB |
| **peak concurrent sequences**   | **26**              | **128**          | **4.92x**          |
| reserved-slot occupancy (mean)  | 42.0%               | 89.9%            | 2.14x              |
| KV utilization of budget (peak) | 36.5%               | 74.7%            | 2.05x              |
| system throughput               | 163.0 tok/s         | 466.2 tok/s      | 2.86x              |
| **per-sequence decode rate**    | **6.27 tok/s**      | **3.64 tok/s**   | **0.58x**          |
| TTFT p50 / p99                  | 13.20 / 38.61 s     | 0.46 / 0.46 s    | — (see below)      |
| TPOT p50 / p99                  | 0.112 / 0.222 s     | 0.098 / 0.100 s  | —                  |
| preemptions                     | —                   | 0                | —                  |
| `total_tokens`                  | 8600                | 8600             | identical ✓        |


**The result factorizes, and that is the clearest way to read it:**

```
system throughput ratio  =  concurrency ratio  x  per-sequence rate ratio
        2.86x            =        4.92x        x        0.58x
```

Paging bought 4.92x the sequences and gave back 42% of each sequence's decode
rate. That is exactly the trade this plan predicted in Phase 6's preamble
("throughput may even dip slightly per sequence; total system throughput should
still rise") — **the prediction was confirmed quantitatively, including its
sign.** Reporting only the 2.86x, or only the 4.92x, hides one half of it.

The 42% → 90% occupancy line is the PagedAttention paper's headline reproduced
on our own engines. It **explains** the concurrency number rather than
corroborating it independently: a contiguous cache reserves a rectangle sized
for the longest sequence in the batch, so on a long-tailed workload more than
half of what it holds is reserved against tokens that never arrive.

### The sweep, and why its ratio collapses

M2's six-request workload, budgets 1–8 MiB:


| budget | cont. batch | paged blocks | peak concurrent | ratio     | throughput (c/p) | TPOT p50 (c/p) | preempt |
| ------ | ----------- | ------------ | --------------- | --------- | ---------------- | -------------- | ------- |
| 1 MiB  | 1           | 5            | 1 / 5           | **5.00x** | 16.3 / 20.0      | 0.057 / 0.110  | 4       |
| 2 MiB  | 2           | 10           | 2 / 6           | **3.00x** | 30.1 / 33.7      | 0.053 / 0.086  | 0       |
| 3 MiB  | 4           | 16           | 4 / 6           | 1.50x     | 52.8 / 31.6      | 0.056 / 0.093  | 0       |
| 4 MiB  | 5           | 21           | 5 / 6           | 1.20x     | 54.3 / 32.9      | 0.054 / 0.092  | 0       |
| 6 MiB  | 8           | 32           | 6 / 6           | 1.00x     | 55.9 / 32.4      | 0.053 / 0.090  | 0       |
| 8 MiB  | 10          | 42           | 6 / 6           | 1.00x     | 53.6 / 33.1      | 0.055 / 0.087  | 0       |


**This is not a weak result; it is a workload too small to hold one.** Six
requests cap peak concurrency at six for *both* engines, so the moment the
budget lets contiguous run all six there is nothing left for paging to win.
Confirming evidence in the data: at 6 and 8 MiB the contiguous engine's
*measured* peak KV is 4.50 MiB in both rows — it never grew into its budget,
because there were no more requests to admit. The 1.0x is a fact about six
requests, not about paging. Hence the two-workload design (ADR-0014).

The low-budget rows carry information the validation point does not: **1 MiB is
the only configuration in the whole run where preemption fires** (4 times), and
it is the only place the paged path beats contiguous on both axes at once.

### The cost, stated plainly — the number the run README under-reported

At **equal peak concurrency** (6 vs 6, at 6 and 8 MiB), the sweep isolates the
price of gather-then-attend with the concurrency win removed:

- TPOT p50: **1.58x–1.68x worse** for paged
- system throughput: **0.58x–0.62x** of contiguous

So paging costs roughly **40% of decode throughput at fixed concurrency**. The
run README's line about TPOT being "essentially flat" is true *at the
validation point only*, where contiguous is simultaneously slowed by its own
batch growth. Read across the whole run, the honest statement is: **paging is a
straight loss below roughly 19 concurrent sequences and a large win above it.**

That crossover comes from two different curves:


| concurrency | contiguous TPOT | paged TPOT |
| ----------- | --------------- | ---------- |
| 6           | 0.053 s         | 0.090 s    |
| 26          | 0.112 s         | —          |
| 128         | —               | 0.098 s    |


**Contiguous per-step latency grows steeply with batch (2.1x from 6 to 26
sequences). Paged per-step latency is nearly flat (+9% from 6 to 128
sequences).** Extrapolating contiguous linearly, the curves cross at ~19
concurrent sequences. Caveat on that number: part of contiguous's growth is
ADR-0007's resync, not attention, so this is a crossover between *these two
engines*, not a clean gather-vs-nothing crossover.

### A suspected implementation defect — flagged, not yet fixed

**A paged decode step does 21x the work going from 6 to 128 sequences and takes
9% longer.** Nearly-flat step time means the step is dominated by something
that is not the work. At batch 6 that fixed cost shows up as ~37 ms/step of
pure loss against contiguous, on a workload where the gather itself moves
almost no bytes (6 sequences x ~4 blocks ≈ 98K elements per layer). **That is
not bandwidth. It is overhead**, and it is worth a profile before M4.

Four concrete candidates, in the order I would test them:

1. **`block_table_tensor()` rebuilds a padded Python list and does a pageable
  host→device copy every step.** A pageable H2D transfer synchronizes. This is
   per-step, and the padded list is rebuilt from scratch rather than kept
   incrementally.
2. `scatter_index` **does two more pageable H2D transfers per step** —
  `torch.tensor(batch.q_lens, ...)` and `torch.tensor(batch.past_lens, ...)` —
   plus roughly eight small device ops whose launch cost is fixed.
3. `gather_kv` **copies twice per layer.** `pool[block_table, :, layer_idx]` is
  advanced indexing, which materializes a copy; the following
   `.permute(...).reshape(...)` breaks contiguity and forces a *second* copy.
   That is 48 copies per step instead of 24.
4. **24 separate small gathers per step**, each with its own launch and Python
  overhead. Phase 3 chose per-layer gathering deliberately (peak scratch is
   1/24th, and scratch competes with the pool for the VRAM this milestone
   exists to reclaim) — that tradeoff is still right, but its cost was never
   measured.

**The measurement that settles it:** time `scatter_index` and `gather_kv` in
isolation against the model forward, at batch 6 and batch 128, with
`torch.cuda.synchronize()` around each. If the two index helpers account for
most of the ~90 ms, it is (1)+(2) and the fix is caching the block-table tensor
and keeping `q_lens`/`past_lens` on device. If the forward dominates, it is
(3)+(4) and the fix is a single fused gather or a contiguous-friendly pool
layout.

**Why this matters for M4, not just aesthetically.** Quantization shrinks
weight traffic, which is the term this fixed overhead is currently hiding
behind. A constant ~90 ms/step becomes a *larger* share of the step after M4,
so an unprofiled overhead here will surface there as a disappointing
quantization result. Profile before quoting M4 numbers.

*No fix has been made. This section is the finding; the correction is a
separate decision.*

### What must not be credited to paging

**TTFT p99 fell from 38.6 s to 0.46 s — the largest number in the run, and it
is not paging.** It is the absence of ADR-0007's resync. The contiguous engine
rebuilds its entire `StaticCache` on every membership change, re-prefilling
every running sequence; with 26 rows churning continuously, a newly admitted
request queues behind repeated full-batch prefills. A paged cache has a block
table per sequence, so membership changes cost nothing. Real win of M3, and a
**separable** one — conflating it with the paging result would overstate paging
by an order of magnitude.

### Three limits on the headline

1. **4.92x is a floor, not a ceiling.** Paged peaked at 128 — *every request in
  the workload* — with **zero preemptions**, holding 4077 of 5456 slots. It
   never ran out of blocks, so its real limit at 64 MiB was never found.
   Contiguous, by contrast, was genuinely saturated at 26. A larger workload
   widens the gap. **The measurement that would close this: raise**
   `--validate-requests` **until** `num_preemptions > 0`**.**
2. **The contiguous side is provisioned for the workload's global worst case.**
  `contiguous_max_batch = budget // (max_cache_len * per_token_bytes)` with
   `max_cache_len = max(prompt + max_new_tokens)` over *all 128 requests* (203
   tokens), while at runtime `ContinuousEngine` sizes to the *running batch's*
   max. So contiguous holds headroom it cannot always use. That is not a rigged
   comparison — provisioning for the worst case **is** the reserve-max rule,
   and the whole point — but it should be named rather than discovered later.
3. **We never reached genuine whole-GPU VRAM pressure.** A 0.5B model on a
  4 GiB laptop GPU would need thousands of concurrent long sequences, and
   contiguous's resync makes that take hours. The 64 MiB budget reproduces the
   *regime* at a scale that runs in minutes. **The ratio transfers; the
   absolute concurrency figures do not.**



### Minor harness defects found while reading the data

- `vram_free_gib` **is recorded as** `0.0` **in** `validation.json` (and 2.09 in
`results.json`). The probe is read after the model and caches are resident,
so the field is meaningless exactly where it matters most. Cosmetic, but it
is a *provenance* field, and provenance fields that lie are worse than absent
ones.
- **Budget compliance is verified asymmetrically.** Contiguous's is *measured*
against the tensors `StaticCache` actually allocated
(`measured_kv_bytes_peak`); paged's is `null` and its compliance rests on
arithmetic (`num_blocks * block_bytes`). The arithmetic is exact — the pool
is one tensor allocated once — so this is defensible, but recording
`pool.nbytes` would cost one line and make the table symmetric.
- `internal_fragmentation_peak` **is** `null` **for contiguous.** Correct — the
concept does not apply to a reserved rectangle — but a reader comparing
columns will read `null` as "not measured" rather than "not applicable".



### Key pointers for a future reader

- The number that matters is **peak concurrent sequences**, not throughput.
Throughput is downstream of it (`2.86x = 4.92x * 0.58x`).
- **Occupancy explains concurrency; it does not confirm it.** They are the same
fact viewed from two sides.
- **Two utilization denominators exist and answer different questions.**
`kv_utilization_of_budget` = real tokens ÷ the fixed budget ("how much of the
VRAM I paid for is working"). `reserved_slot_occupancy` = real tokens ÷ what
the engine reserved *for the sequences it is currently running* (the paper's
number, independent of budget size). Reading one as the other makes the whole
table incoherent.
- **Paging did not remove padding.** The buffer handed to
`scaled_dot_product_attention` is still a rectangle padded to the batch's
longest row, because SDPA requires one. What went away is *reservation* waste
in the pool. Residual waste is internal fragmentation — 1,440 token slots at
peak, bounded by `BLOCK_SIZE - 1` per sequence (15 x 128 = 1,920 worst case)
rather than by the workload's length distribution. **That bound is the claim
M3 actually makes.**
- `total_tokens` **is identical across engines at every single point** (180 in
every sweep row, 8600 at the validation point). Different memory strategies,
byte-identical output. A mismatch here would invalidate everything above it,
which is why it is checked per row rather than once.
- Benchmarks run **GPU + SDPA** (the real path); the correctness gate runs
**CPU + eager** (ADR-0010). Two configurations by design. Phase 0's findings
4 and 5 were established under eager and **still need re-verifying under
SDPA** before any GPU number from the paged path is quoted in M4.

---



## The four decisions — all resolved

**Kept verbatim, questions and all.** In every case the recommendation was the
option taken, so these read as retrospective justifications — but they were
written *before* the answer was known, and that is exactly what makes them
worth keeping. `claude.md`: the rejected alternatives are the most valuable
part of a decision record. Each question below is now headed with its outcome
and the ADR that carries the full argument.

### Q1 — Paged scheduler: new class, or generalize `Scheduler`?

**→ Resolved: (a), a separate** `PagedScheduler`**. ADR-0013.** The deciding
argument turned out to be one the question only half-anticipated: preemption is
a *control-flow* change, not an admission predicate, so the injected-policy
shape in (b) would have had to reach into M2's step loop rather than into one
`if`. The benchmark-provenance argument stood too — an already-measured engine
is closer to a recorded artifact than to live code.

*(a) A separate* `PagedScheduler` — `Scheduler` stays frozen as M2's
artifact, and the four-way comparison in M5 compares four genuinely
independent implementations. Costs some duplicated admit/evict bookkeeping.
*(b) Generalize* `Scheduler` with an injected memory manager (M2's path
passing a null manager that never refuses). Less duplication, and it makes
"what actually changed between continuous and paged" a small, readable diff.
Risk: refactoring a finished, tested, benchmarked M2 artifact to serve M3, and
if it regresses, an already-recorded M2 number becomes unreproducible.
**Recommendation: (a), a separate class.** Same reasoning as M2's Q3
(`StaticEngine` got its own class, not a mode flag), and it follows locked
decision 7 — the M2 engine must keep running unchanged for M5's chart. The
duplicated bookkeeping is maybe thirty lines, and the two policies diverge
more than they overlap once preemption exists.

### Q2 — How is `num_blocks` chosen?

**→ Resolved: (b), explicit** `num_blocks` **with** `blocks_for_bytes` **/**
`blocks_for_gib` **as sizing aids the pool never calls implicitly. Folded into
ADR-0011** rather than given its own file. This one earned its keep in Phase 6:
the whole benchmark is "hold VRAM fixed and vary concurrency", and a pool that
resized with ambient GPU state would have made every row unreproducible.

*(a) A VRAM fraction* — measure free VRAM at startup, take some percentage
after weights, derive block count. Realistic, matches what vLLM does
(`gpu_memory_utilization`), and makes the engine adaptive to whichever GPU it's
on — relevant given the incoming machine swap. Costs reproducibility: the same
benchmark on two machines allocates different pools.
*(b) An explicit* `num_blocks` *argument*, defaulted, with a helper that
converts a GB budget to a block count. Fully reproducible, trivially
scriptable for Phase 6's sweep, and honest about being a benchmark rig rather
than a product.
**Recommendation: (b), explicit, with the GB→blocks helper.** Phase 6's entire
design is "hold VRAM fixed and vary concurrency" — a pool that silently
resizes with ambient GPU state makes that measurement unrepeatable, and
`claude.md`'s provenance rule effectively demands the fixed version. (a) is
what you'd pick building a product; this is a measurement instrument.

### Q3 — What happens when blocks run out mid-generation?

**→ Resolved: (b), preemption by recomputation. ADR-0012.** The argument that
decided it is the machine: on a laptop the scarce resource is the PCIe link,
not compute. Recomputation and swapping trade the same currency in opposite
directions — prefill is the high-arithmetic-intensity phase (the abundant
resource), a host round trip is pure data movement on the slowest link (the
scarce one). Swapping would also have added a host pool, a host allocator, a
GPU→host block map and a transfer in the `step()` critical path: the
least-tested code in the engine holding the most invariants.

**Termination is guaranteed by newest-first eviction**, not by (a)'s
per-sequence cap as this question assumed. Victims are requeued at the *front*
of the waiting deque (sunk cost), and the guard raises `BlockExhausted` with a
diagnostic naming the length that failed when a single sequence cannot fit in
the whole pool. That guard fired once during Phase 4 — on a badly-parameterised
*test*, not on engine code. It behaved correctly.

*(a) Refuse admission only; never preempt.* Sequences already running always
finish. Simplest, no new failure modes — but it can still deadlock in
principle: every running sequence needs one more block and none exists. With a
per-sequence block cap it's provably safe, at the cost of not really
demonstrating preemption at all.
*(b) Preempt by recomputation (vLLM's default).* Pick a victim (last admitted,
LIFO), free all its blocks, return it to `WAITING`, re-prefill it from scratch
later. Discards work, but it's simple, correct, and it's the mechanism the
paper describes.
*(c) Preempt by swapping to CPU.* Copy blocks to pinned host memory, restore
later. Preserves the work; adds PCIe transfer plumbing and a second allocator.
**Recommendation: (b), preemption by recomputation** — with (a)'s cap as the
safety net so a pathological case can't deadlock. It's the paper's own
mechanism, it's genuinely interesting (a preemption count is a real metric for
Phase 6), and it costs perhaps twenty lines given a working allocator. (c) is a
meaningfully larger build and its win only shows up for long sequences we
aren't running at this model size. **This is the question I'd most like your
call on** — it's the one that changes what Phase 4 is.

### Q4 — Module boundaries: how many files?

**→ Resolved: (b), two modules. ADR-0011.** `engine/kv_cache.py` (pool,
allocator, `BlockManager`, no `transformers` import at all) and
`engine/paged_cache.py` (the HF `Cache` glue). The sub-question resolved as
proposed: the **scheduler** calls `free()`, next to the `FINISHED` transition,
and the same call serves the preemption path — which is why it resets
`num_cached_tokens` rather than assuming completion.

*(a) Everything memory-related in* `engine/kv_cache.py` — pool, allocator,
block-table helper, `PagedCache`. One file, one concept, easy to read
end-to-end; grows to maybe 250 lines.
*(b)* `engine/kv_cache.py` *(pool + allocator) and* `engine/paged_cache.py` *(the
HF* `Cache` *subclass)* — splits the pure-bookkeeping half (yours, heavily
unit-tested, no framework) from the framework-integration half (mine, tested
through the gate). Also splits neatly along the ownership table above.
**Recommendation: (b).** The allocator is a data structure you should be able
to reason about and test with zero `transformers` in scope; the cache is glue
against a framework contract that has already surprised this project once. Two
files, and the seam sits exactly where the risk changes character. Related sub-
question: **who calls** `free()` **on a finished sequence's blocks?** I'd put it in
the scheduler's eviction path, next to the existing `FINISHED` transition —
`Sequence` shouldn't hold a reference to the allocator, and the scheduler
already owns the lifecycle.

---



## ADRs written

Four, where the plan foresaw three. The numbering diverged from the plan and
the divergence is recorded rather than tidied.

- ✅ **ADR-0011 — Paged KV cache module split and pool sizing.** How our cache
reaches the model — a `transformers` `Cache` subclass, vs. patching the
attention module, vs. writing our own decoder loop. Phase 0's spike settled
this before the ADR was written rather than after: the subclass seam holds,
and the thing to override is the *layer*, not the `Cache`. Also carries Q2
(explicit `num_blocks`) and Q4 (two modules).
- ✅ **ADR-0012 — Preemption by recomputation** (Q3), over swapping and over
admission-refusal-only. Written to the owner's brief: the central argument is
that recomputation and swapping trade the same currency in opposite
directions, and on a laptop the memory-transfer side is the expensive one.
Names the conditions under which it should be revisited — a machine where the
link is fast relative to compute, or sequences long enough that re-prefill
dominates.
- ✅ **ADR-0013 — Separate** `PagedScheduler`**, and prefill-priority steps**
(Q1). *Reassigned.* The plan pencilled 0013 in for pool sizing "if (a) is
chosen over (b)"; (b) was chosen and folded into 0011, so 0013 went to the
decision that actually had live alternatives. Also records the cost of
prefill-priority — head-of-line blocking, TPOT traded for TTFT — and names
chunked prefill as the fix that is out of scope here.
- ✅ **ADR-0014 — Equal-VRAM benchmark methodology.** **Not foreseen by this
plan.** It exists because "equal VRAM" is not a setting either engine has
(`ContinuousEngine` takes `max_batch_size`, `PagedEngine` takes `num_blocks`),
and because the plan's own six-request workload could not exhibit the effect
the plan wanted to measure. Also settles the two utilization denominators.
Its "what I learned" is the milestone's sharpest line: *a benchmark that
cannot fail informatively is not a measurement.*

---



## Concepts added to `docs/concepts.md`

All seven planned, plus two that only appeared once the numbers were in:
**reserve-max** (sizing every cache row for the workload's global worst case)
and **capacity vs. bandwidth** as the two distinct memory costs paging trades
between.

- ✅ **Block (page)** — fixed-size unit of KV allocation, 16 tokens →
`engine/kv_cache.py`.
- ✅ **Block table** — per-sequence logical→physical map; an OS page table,
not an analogy for one → `engine/sequence.py`. Source: PagedAttention §4,
OSTEP Ch 18–19.
- ✅ **Internal fragmentation** — the partially-filled last block; the
residual waste paging *doesn't* remove → measured in Phase 6.
- ✅ **Gather-then-attend** — assembling scattered blocks into a contiguous
scratch tensor so stock SDPA can be used, trading memory traffic for the
absence of a custom kernel → `engine/paged_cache.py`.
- ✅ **KV-cache memory utilization** — real token-slots ÷ allocated
token-slots; the paper's ~40% → >90% claim, reproduced on our own engines.
- ✅ **Preemption (recompute vs. swap)** — how a full pool makes room →
`engine/paged_scheduler.py`, ADR-0012.
- ✅ **Roofline / arithmetic intensity** — compute-bound vs.
memory-bandwidth-bound; why the gather costs bandwidth to buy capacity.
Source: Horace He, *Making Deep Learning Go Brrrr*.

---



## Definition of done (whole milestone) — MET

- ✅ All phases above ticked.
- ✅ `tests/test_correctness.py` passes on the paged code path, including the
block-boundary and block-reuse cases from Phase 5.
- ✅ A memory-utilization benchmark showing higher achievable concurrent batch
size at the same VRAM budget, contiguous vs. paged, recorded in
`docs/benchmarks/` with commit + hardware.
- ✅ `docs/progress.md` updated.
- ✅ ADR-0011 onward written and linked (0011-0014).
- ✅ `docs/roadmap.md` Milestone 3 build boxes and DoD ticked. **Theory boxes
left deliberately unticked** — a theory box records having read the thing,
which is the owner's to tick, not mine.
- ✅ README notes the two standing simplifications: prefill/decode unfused,
and gather-then-attend instead of a paged-attention kernel.

---



## Retrospective

**Did it work?** Yes. **4.92x concurrent sequences at equal VRAM** (26 → 128 at
a 64 MiB KV budget), reserved-slot occupancy **42% → 90%**, byte-identical
output at every measured point. That is the roadmap's Definition of Done, met
and recorded with provenance. The full reading — including the parts that are
less flattering — is in *Phase 6 — results, read honestly*.

**What the plan got right.** Phase 6's preamble predicted the shape of the
result before anything was built: *"throughput may even dip slightly per
sequence; total system throughput should still rise."* The measurement
confirmed it including the sign, and the whole result factorizes to say so:
`2.86x system throughput = 4.92x concurrency x 0.58x per-sequence rate`.
Phase 0's insistence on spiking the framework seam before designing against it
paid for itself twice — it found that the override point is the *layer* and not
the `Cache`, and that an empty layer list short-circuits `get_mask_sizes`,
either of which would have cost days in Phase 3.

**What the plan got wrong, in order of how much it cost.**

1. **The benchmark workload.** Reusing M2's six requests was specified for
  continuity, and it does compose — but six requests cap peak concurrency at
   six, so above 3 MiB the ratio collapses to 1.0x. The plan asked for a
   measurement its own workload could not produce. Fixed by adding a second,
   larger workload rather than by dropping the first.
2. **"Equal VRAM" was assumed to be expressible.** It is not a setting either
  engine has. An entire ADR (0014) exists to define it.
3. **The pool shape in the roadmap.** `num_heads` where it must be
  `num_key_value_heads` — Qwen2.5-0.5B has 14 attention heads and **2** KV
   heads. Caught while writing this plan, not while building. Following the
   roadmap literally would have allocated a 7x oversized pool and quietly
   invalidated every memory number the milestone produced. Not a decision; a
   fix.
4. **The ADR numbering.** Minor, but recorded: 0013 was pencilled in for a
  question that turned out not to need its own file.

**What is still open, carried into M4.**

- **The suspected fixed-overhead defect in the paged step** — flagged in Phase
6 with four ranked candidates and the profile that would settle it. **Not
fixed.** It matters most at M4, because quantization shrinks the weight
traffic this overhead is currently hiding behind.
- **Phase 0's findings 4 and 5 were established on CPU + eager.** Re-verify the
block-padded gather under SDPA before quoting any GPU number from the paged
path.
- **The 4.92x is a floor**, not a ceiling — the paged engine never ran out of
blocks. Raising `--validate-requests` until `num_preemptions > 0` would find
the real number.
- **Three harness warts**: the recorded run's command is not captured in its
own JSON, `vram_free_gib` is recorded as `0.0` in `validation.json`, and
budget compliance is measured for contiguous but only computed for paged.

**The one thing to carry forward as a habit.** The hardest work in this
milestone was not building the allocator or the gather — those are a free list
and two index expressions. It was noticing, twice, that a measurement did not
mean what it appeared to mean: that a 1.0x ratio was a fact about six requests
rather than about paging, and that an 84x TTFT improvement belonged to the
absence of ADR-0007's resync rather than to paging. Both would have read as
results. Neither was one.