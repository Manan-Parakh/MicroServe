# Handout — M3 Phase 1: the `BlockAllocator`

Paste this whole file into a fresh Claude chat as your first message. It carries
enough context that the chat doesn't need the repo.

---

## Instructions to the assistant reading this

I am building an LLM inference engine from scratch, alone, to **learn how
inference engines work**. The working engine is the byproduct; the
understanding is the product.

**Do not write the `BlockAllocator` for me.** Not as a "sketch", not as "here's
roughly what it looks like, adapt it", not as a complete class with a couple of
`# TODO` lines in it. This class is the point of the milestone. If I ask you
for the code directly, push back once and offer the alternatives below.

What I want from you instead, in rough order of usefulness:

- **Socratic pressure.** Ask me what happens in the edge case I haven't
  considered. Especially the ones listed under "Where the bugs actually are".
- **Design discussion with real alternatives.** When I pick a data structure,
  tell me what I gave up and when that would bite.
- **Review of code I paste in.** Be harsh about invariants and off-by-ones. Be
  specific: name the input that breaks it, don't say "consider edge cases".
- **Systems context by proper name.** I have just read OSTEP Ch 13, 15, 18–19,
  21–22. Connect what I'm writing to what I read — free lists, arena
  allocation, external vs internal fragmentation — and name things so I can go
  read further on my own. Do not re-explain attention, KV caching, or GQA; I
  know those.
- **Pseudocode, prose, and invariants are fine.** Working Python for the
  allocator's methods is not.

Be concise. Point rather than lecture.

---

## The project, in a paragraph

MicroServe is a from-scratch LLM inference engine on a single GPU. Milestone 1
was a naive one-request-at-a-time baseline. Milestone 2 added continuous
batching (iteration-level scheduling) but still uses a `StaticCache`: one
rectangular KV tensor of `batch x max(prompt + max_new_tokens)`. Milestone 3
replaces that with a **paged KV cache** — vLLM's PagedAttention idea, minus the
custom kernel.

Why: that rectangle wastes memory on two axes at once. Every row is padded out
to the longest row, and every row reserves a generation budget it usually never
spends. On a realistic long-tailed workload I measured 33% utilization. Paging
gets it to 98%, which is 3x more concurrent sequences in the same VRAM. Peak
concurrent sequences is one of the four metrics this whole project reports.

**The hard constraint that shapes everything:** no custom CUDA or Triton
kernels. So a sequence's KV lives in scattered fixed-size blocks, and at
attention time I *gather* those blocks into a contiguous scratch tensor and call
PyTorch's `scaled_dot_product_attention`. vLLM avoids that copy with a kernel; I
pay the copy on purpose, because kernel authorship is a different project.

Phase 1, this phase, is only the memory layer: one big pre-allocated tensor,
and an allocator that hands out and reclaims indices into it.

---

## What already exists (written for me, not by me)

`engine/kv_cache.py` currently holds:

```python
BLOCK_SIZE = 16                    # tunable constant, vLLM's default

class BlockExhausted(RuntimeError): ...

@dataclass(frozen=True)
class KVSpec:                      # the six numbers + byte arithmetic
    num_layers, num_kv_heads, head_dim, dtype, block_size
    KVSpec.from_config(config, dtype)
    .elements_per_token  .bytes_per_token  .bytes_per_block

def blocks_for_bytes(budget_bytes, spec) -> int
def blocks_for_gib(gib, spec) -> int

class PagedKVPool:                 # one torch.zeros, allocated once
    shape [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
    .tensor  .shape  .nbytes  .capacity_tokens

class BlockAllocator:              # <- MINE. currently raises NotImplementedError
```

Concrete numbers for the dev model (Qwen2.5-0.5B-Instruct, bf16, 24 layers,
**2 KV heads** under GQA, head_dim 64): one token of KV is 12 KiB, one 16-token
block is 192 KiB, and 1 GiB of pool is 5,461 blocks.

Two things about the pool shape, so you don't suggest changing them:

- `num_blocks` is dim 0 so a block id is a plain index and `pool[block_id]` is
  contiguous. A gather is then `pool[block_id_tensor]` — one op, large
  contiguous reads. Layers-first would scatter one sequence across the tensor.
- It is `torch.zeros`, never `torch.empty`. I proved in a spike that finite junk
  in a causally-masked slot is harmless, but an `inf` from uninitialised memory
  gives `0 * inf = NaN`, poisons the weighted sum, and **does not raise**. Since
  a reused block always holds some dead sequence's finite KV, zeroing once at
  construction is sufficient and `free()` never has to zero.

---

## What I have to write

```python
class BlockAllocator:
    def __init__(self, num_blocks: int): ...
    def allocate(self, n: int) -> list[int]: ...   # n fresh ids, ref_count -> 1
    def free(self, block_ids) -> None: ...         # decrement; reclaim at 0
    @property
    def num_free(self) -> int: ...
    def ref_count(self, block_id: int) -> int: ...
```

Constraints already settled, so don't reopen them — but do quiz me on *why*
each one is what it is:

1. **Free list is a stack** (`list[int]`, append/pop). O(1) both ways, and LIFO
   reuse returns the most recently touched blocks, which are the warmest.
2. **`ref_count` is a plain list indexed by block id.** In this milestone it is
   always 0 or 1. The structure exists now so that Milestone 6's prefix sharing
   (copy-on-write over shared blocks) is an extension rather than a rewrite.
3. **Exhaustion raises `BlockExhausted`,** a named catchable type — never a bare
   `IndexError` from `.pop()`. In Phase 4 the scheduler catches this as *normal
   control flow*: it is the signal to stop admitting or to preempt. That is the
   entire reason preemption exists in a paged engine.

---

## Where the bugs actually are

These are the questions I want pressure on. Do not answer them for me up front;
ask me what I think first.

- **Partial allocation.** `allocate(n)` with `n > num_free`. If it pops some
  blocks and then raises, those blocks are gone — no block table records them,
  so nothing will ever free them. What is the general name for that, and what
  is the cheap structural fix?
- **Double free.** A block freed twice ends up in the free list twice and gets
  handed to two live sequences, which then overwrite each other's KV. No crash,
  just plausible wrong tokens thousands of forward passes later. How do I detect
  it without a second data structure? (Hint: I already store something that
  knows.)
- **Freeing something never allocated.** Same class of bug, different entry
  point.
- **Boundary arithmetic.** `allocate(0)`, `free([])`, exhaustion at exactly
  `num_blocks`. Off-by-one at a block boundary is the canonical bug in this
  milestone.
- **The invariant.** There is one sentence that, if true after every single
  operation, makes all of the above impossible. What is it? I want to state it
  before I write the code, then assert it in tests.

Then, the design questions I actually want to argue about:

- Free list as a stack vs `deque` used FIFO vs a `set` vs a bitmap over blocks.
  What does each buy, and at what pool size does the answer change?
- Is `ref_count` as a Python list the right call, or should it be a tensor?
  Where would that decision start to matter?
- Should `free()` take a list, or should there be a `free_sequence(seq)` that
  knows about block tables? Where should the "sequence finished, return its
  blocks" call actually live — the allocator, the sequence, or the scheduler?

---

## How I'll know it's done

`tests/test_kv_cache.py` already exists in the repo — 15 allocator tests
currently failing against the `NotImplementedError`, plus a 2000-step fuzz that
re-checks `num_free` and every ref count after every operation. It runs with no
model and no GPU. Done means green.

If you want to help me sharpen the tests, that's welcome. If you want to write
the class, that's the one thing I need you not to do.

---

## Next, so you know where this is heading

- **Phase 2** — `Sequence.block_table: list[int]`, index `i` holding the
  physical block for logical positions `[i*16, (i+1)*16)`. Growth rule: a new
  block is needed exactly when the next token lands at offset 0 of a block I
  don't have.
- **Phase 3** — the HF `Cache` subclass: scatter new K/V into blocks, gather the
  history back into a contiguous rectangle.
- **Phase 4** — scheduler integration, and the open question of what to do when
  blocks run out mid-generation: refuse admission only, preempt by
  recomputation, or preempt by swapping to host memory.
