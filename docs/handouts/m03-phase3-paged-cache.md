# Handout — M3 Phase 3: scatter, gather, and the `Cache` contract

Paste this whole file into a fresh Claude chat as your first message. It carries
enough context that the chat doesn't need the repo.

> **Superseded, 2026-09-03.** The owner elected to hand `scatter_index` and
> `gather_kv` over rather than write them, trading authorship for review. This
> file is kept because it is still the best statement of *why* those two
> functions are the milestone, and so the right thing to read before reviewing
> `engine/paged_cache.py`. Read the "Where the bugs actually are" and "Design
> questions" sections as a review checklist against the code that now exists.

---

## Instructions to the assistant reading this

I am building an LLM inference engine from scratch, alone, to **learn how
inference engines work**. The working engine is the byproduct; the understanding
is the product.

**Do not write `scatter_index` or `gather_kv` for me.** Not as a "sketch", not
as "here's the shape, adapt it", not as a complete function with the permute
left as a `# TODO`. These two functions *are* PagedAttention-without-a-kernel.
Everything else in this milestone is bookkeeping that leads up to them. If I ask
you for the code directly, push back once and offer the alternatives below.

What I want from you instead:

- **Shape interrogation.** Give me a tensor shape, ask me what the next op
  produces, and tell me when I'm wrong. Especially around `permute` vs
  `reshape` and around advanced indexing.
- **Trace-through review.** When I paste code, walk two rows of different
  lengths through it and tell me where it diverges from what I claimed.
- **Adversarial cases**: a decode step at a block boundary, a batch where one
  row owns three blocks and another owns one, a padded query column.
- **Systems context by proper name.** I've read OSTEP Ch 13, 15, 18–19, 21–22
  and Horace He's *Making Deep Learning Go Brrrr*. Connect this to memory
  traffic vs. memory capacity, and to gather/scatter as a hardware idea. Don't
  re-explain attention, KV caching, GQA, or RoPE; I know those.
- **Pseudocode, prose, invariants, counterexamples, drawn memory layouts**: all
  fine. Working Python for the two functions: not fine.

Be concise. Point rather than lecture.

---

## The project, in a paragraph

MicroServe is a from-scratch LLM inference engine on a single GPU. Milestone 2
gave it continuous batching over a rectangular `StaticCache`, which measured 33%
KV utilization on a long-tailed workload. Milestone 3 replaces it with a **paged
KV cache** — vLLM's PagedAttention minus the custom kernel. Fixed-size blocks
scattered across one pre-allocated pool; at attention time we *gather* a
sequence's blocks into a contiguous scratch tensor and call PyTorch's
`scaled_dot_product_attention`, because writing a kernel that reads scattered
blocks directly is a different and much larger project. Target: 98%
utilization, ~3× the concurrent sequences.

Phases 1 and 2 are done and green: the pool, a `BlockAllocator` (LIFO free list,
ref counts, `BlockExhausted` as normal control flow), and the block table —
`Sequence.block_table: list[int]`, a single-level page table, with the growth
rule in one place.

**Phase 3 is where the pool is finally read and written.**

---

## The model dimensions, so shapes are concrete

Qwen2.5-0.5B-Instruct: 24 layers, **2 KV heads** (14 query heads — GQA), head
dim 64, bf16. `BLOCK_SIZE = 16`.

```
pool.shape = [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
                  ^dim 0   ^K/V
```

`block_id` is a plain index into dim 0, so `pool[block_id]` is one contiguous
run. Slicing vocabulary:

```python
pool[b]                        # [2, layers, kv_heads, block, dim]
pool[b, 0, layer]              # [kv_heads, block, dim]      that layer's keys
pool[b, 0, layer, :, slot, :]  # [kv_heads, dim]             one token
```

Address translation, from Phase 2:

```
logical_block = pos // 16      slot = pos % 16
physical      = block_table[logical_block]
```

---

## What `transformers` 5.15.1 demands (already probed, already built)

`Cache.update(k, v, layer_idx)` forwards straight to
`self.layers[layer_idx].update(k, v)`. All the work is on a `CacheLayerMixin`
subclass: `lazy_initialization`, `update`, `get_mask_sizes`, `get_seq_length`,
`get_max_length`. Tensors crossing that boundary are
`[batch, num_kv_heads, seq_len, head_dim]`.

**I have already written all of that**, plus a `PagedBatch` descriptor (which
rows are in this step, their block tables, their past lengths, their real query
lengths), the attention mask, the position ids, and the model-runner calls.
I verified the whole harness end-to-end against a throwaway reference
implementation of the two functions below — single-sequence *and* batched, both
token-identical to HF `generate()`. So if a test fails now, it is my index math,
not the plumbing.

Three findings from the probe that are already encoded and that you should
quiz me on rather than re-derive:

1. **`get_mask_sizes` is a promise made before `update` runs.** The 4-D causal
   mask is built from it. Return a width that disagrees with the keys `update`
   later returns and attention raises a shape error. Loud, which is good.
2. **A block-padded gather is legal.** Returning `ceil(len/16)*16` positions
   with junk in the tail is token-identical to the reference, because causality
   and the per-row mask cover the tail. So the gather never has to trim.
3. **`torch.zeros`, never `torch.empty`.** Finite junk in a masked slot is
   harmless; an `inf` makes `0 × inf = NaN` and the output silently collapses
   without raising.

---

## What I have to write

```python
def scatter_index(batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """rows, cols, blocks, slots -- four parallel 1-D long tensors of length
    N = sum(q_lens)."""

def gather_kv(pool, block_table, layer_idx) -> tuple[Tensor, Tensor]:
    """pool        [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
       block_table [batch, max_blocks]  physical ids in logical order
       returns     2x [batch, num_kv_heads, max_blocks*block_size, head_dim]"""
```

`scatter_index` feeds exactly this, which I've already written in `update()`:

```python
pool[blocks, 0, layer, :, slots, :] = key_states[rows, :, cols, :]
```

**The settled contract, so don't reopen it — but do quiz me on why:**

1. **Rows are right-padded**, unlike M2's left-padding. A paged sequence's KV
   starts at logical position 0 whatever the query padding is, so left-padded
   queries and 0-based KV would put the block table and the causal mask at odds.
   The cost is that `logits[:, -1, :]` no longer reads every row's prediction.
2. **Row r's j-th new token goes to position `past_lens[r] + j`**, not `j`.
3. **N is `sum(q_lens)`, not `batch × padded_q_len`.** Padded query columns must
   not be written.
4. **The gather is per layer, once per step** — 24 small gathers, not one big
   one. Peak scratch is 1/24th, and scratch memory competes with the pool for
   the VRAM this milestone exists to win back.

---

## Where the bugs actually are

Don't answer these for me up front — ask me what I think first.

- **`permute` before `reshape`.** `pool[block_table, 0, layer]` gives
  `[batch, nblk, heads, block, dim]` and I need
  `[batch, heads, nblk*block, dim]`. Ask me what happens if I reshape straight
  from there. Make me say *why* it's silent — every value present, every shape
  checking out, only the tokens wrong.
- **Where the layer index goes.** `pool[block_table][..., layer, :, :, :]` and
  `pool[block_table, kv, layer]` produce the same numbers. Make me say what
  differs, and by what factor, on the hottest path in the engine.
- **Advanced indexing with a slice in the middle.** In
  `pool[blocks, 0, layer, :, slots, :]` the two advanced indices are separated
  by a `:`. Ask me where the broadcast dimension ends up in the result shape,
  and why that determines the shape of the right-hand side.
- **The decode off-by-one.** If `scatter_index` used `j` instead of
  `past + j`, every new token would overwrite the prompt's first tokens. Ask me
  what the output would look like. (Hint: not a crash, and not obviously
  garbage.)
- **Aliasing.** What single property of the four returned tensors, checked how,
  proves that no two rows in a batch write to the same physical address?

Design questions I actually want to argue about:

- The gather copies the *entire* history every layer, every step. What is that
  in bytes for a 1000-token sequence at batch 32, per step? Compare it to
  reading the model weights once. Which dominates, and at what batch size does
  the answer flip?
- vLLM's kernel avoids this copy. Given that I'm not writing one, is there
  anything in *this* design that gets closer — and what would it cost?
- The block table is rebuilt as a tensor every layer. When does that start to
  matter, and what's the fix?
- Padding does not disappear in M3: the gathered rectangle is still padded to
  the batch's longest row. So what exactly did paging remove? Make me state it
  precisely — "paging eliminates padding" is the natural wrong summary, and it
  would make my Phase 6 numbers look like an underperformance.

---

## How I'll know it's done

`tests/test_paged_cache.py`, in two tiers.

**Tier 1, no model, milliseconds.** Recognizable integers written into
deliberately shuffled blocks, gathered, checked. The ones that matter:

- `test_gather_reassembles_shuffled_blocks_in_logical_order` — physical ids are
  meaningless; only the block table's order is
- `test_gather_does_not_interleave_heads_into_time` — two heads with disjoint
  value ranges, so the permute trap shows up as a leak, not a wrong number
- `test_gather_reads_only_the_requested_layer`
- `test_scatter_index_offsets_by_past_length`
- `test_scatter_then_gather_round_trips` — the two halves must be inverses
- `test_scatter_writes_a_decode_token_without_disturbing_the_history`
- `test_two_sequences_never_scatter_into_the_same_slot`

**Tier 2, the real gate.** One sequence prefilled and decoded through
`PagedCache` must emit exactly the tokens HF `generate()` does, on CPU + eager
(pinned, per ADR-0010: SDPA's fused kernels are not numerically invariant to
KV-buffer shape, which is precisely what the paged path changes — so a mismatch
must mean a paging bug, not a kernel picking a different reduction order).

If you want to help me sharpen the tests, welcome. Writing the two functions is
the one thing I need you not to do.

---

## Next, so you know where this is heading

**Phase 4** — the scheduler runs on top of this, and two questions are still
open: whether the paged scheduler is a new class or a generalization of M2's,
and what happens when blocks run out *mid-generation* (refuse admission only /
preempt by recomputation / preempt by swapping to host memory). The last one is
the interesting one and I want to read the PagedAttention paper's preemption
section before deciding.
