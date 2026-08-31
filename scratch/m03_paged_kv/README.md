# M3 scratchpad — paged KV cache, by the numbers

Runnable notes for Milestone 3. Nothing here is imported by `engine/` or
`bench/`, nothing here is tested, and none of it ships. It exists so the
shapes are concrete before you write the allocator.

Run everything from the repo root:

```bash
uv run python scratch/m03_paged_kv/01_model_dims.py
uv run python scratch/m03_paged_kv/02_kv_shapes.py
uv run python scratch/m03_paged_kv/03_the_pool.py
uv run python scratch/m03_paged_kv/04_block_table_and_gather.py
uv run python scratch/m03_paged_kv/05_contract_probe.py     # loads the model
uv run python scratch/m03_paged_kv/06_fragmentation.py
```

`01`, `02` and `05` load Qwen2.5-0.5B on CPU. `03`, `04` and `06` are pure
arithmetic and run instantly.

---

## The one-paragraph version

The KV cache is the only thing in an inference engine whose size you cannot
predict in advance, because you do not know how long a sequence will be until
it stops. The obvious fix — reserve the worst case per sequence — wastes most
of your VRAM, and VRAM is exactly what limits how many requests you can run at
once. Paging fixes it the way operating systems fixed the identical problem
with physical memory: chop the pool into fixed-size blocks, give each sequence
a table mapping its logical positions to whichever physical blocks it happens
to own, and allocate a new block only when it actually needs one. The cost is
that a sequence's KV is no longer contiguous, and `scaled_dot_product_attention`
needs a contiguous rectangle — so we gather the blocks into a scratch buffer
before attending. vLLM avoids that copy with a custom kernel. We don't, on
purpose: that's a different project, and the paging idea is provable without it.

---

## The six numbers everything is built on

From `01_model_dims.py`, for Qwen2.5-0.5B-Instruct:

| | value | note |
|---|---|---|
| `num_hidden_layers` | 24 | the cache is per layer |
| `num_attention_heads` | 14 | **query** heads — never cached |
| `num_key_value_heads` | **2** | **what the cache stores** |
| `head_dim` | 64 | `hidden_size / num_attention_heads` |
| dtype | bf16 | 2 bytes |
| `BLOCK_SIZE` | 16 | tunable; see `06` |

GQA means 14 query heads share 2 KV heads. **The roadmap's stated pool shape
says `num_heads`; it must be `num_key_value_heads`** — sizing with 14 would
overshoot the pool by 7×.

One token of KV = `2 (K,V) × 2 heads × 64 dim × 24 layers` = 6,144 elements =
**12 KiB**. One 16-token block = **192 KiB**. 1 GiB of cache ≈ 87k tokens ≈
5,461 blocks.

---

## What the framework actually demands of us

From `02_kv_shapes.py` and `05_contract_probe.py`. In `transformers` 5.15.1
a `Cache` is nearly a pure container — `Cache.update(k, v, layer_idx, …)`
forwards straight to `self.layers[layer_idx].update(k, v)`. All our work lives
on a `CacheLayerMixin` subclass with five methods:

| member | our job |
|---|---|
| `lazy_initialization(k, v)` | learn dtype/device from the first real tensor |
| `update(k, v, …) -> (keys, values)` | scatter new K/V into blocks; return the gathered K/V |
| `get_mask_sizes(q_len) -> (kv_length, kv_offset)` | **declare how wide the returned K will be** |
| `get_seq_length() -> int` | tokens currently cached |
| `get_max_length() -> int` | `-1` = no fixed maximum |
| `is_sliding = False` | class attr; Qwen2.5 is full-attention |

Shapes crossing that boundary: `[batch, num_kv_heads, seq_len, head_dim]`.
Prefill hands you `q_len = padded_prompt_len` in one call; every decode step
hands you `q_len = 1`.

### Four rules the probe established

1. **`get_mask_sizes` is a promise, made before `update` runs.** The 4-D
   causal mask is built from it. Return a different width and you get
   `RuntimeError: The size of tensor a (16) must match the size of tensor b
   (14)`. Loud failure, not silent corruption — which is the good outcome.
2. **Pre-create the layers.** `Cache.get_mask_sizes` short-circuits while
   `self.layers` is still short, so with `layer_class_to_replicate` our
   declared length is ignored *exactly on the prefill call*. Build
   `PagedCache(layers=[PagedLayer() for _ in range(num_layers)])`.
3. **A block-padded gather is legal.** Returning `ceil(len/16)*16` positions,
   with junk in the tail, and declaring that length, is token-identical to the
   reference — causality masks the tail, since every padded slot is beyond
   every query's position. So the gather never has to trim to an exact length.
4. **`torch.zeros`, never `torch.empty`.** Finite junk in a masked slot is
   harmless; `inf`/`NaN` is not (`0 × inf = NaN` propagates through the
   weighted sum) and it does **not** raise — output silently collapses. A
   *reused* block always holds a dead sequence's finite KV, so zeroing once at
   pool creation is sufficient and `free()` never has to zero.

Verified on CPU + eager (the gate's config, ADR-0010). **Re-verify 3 and 4
under SDPA before quoting any GPU number from the paged path.**

---

## The pool

```
pool.shape = [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
              ^dim 0      ^K/V
```

A `block_id` is a plain index into dim 0, so `pool[block_id]` is one
contiguous run of memory. That's why `num_blocks` goes first: a gather becomes
`pool[list_of_block_ids]`, a single op over large contiguous reads. Put
`num_layers` first instead and one sequence's blocks become a strided mess
across the whole tensor.

Slicing vocabulary (from `03_the_pool.py`):

```python
pool[b]                        # [2, layers, kv_heads, block, dim]  one block
pool[b, 0]                     # [layers, kv_heads, block, dim]     its keys
pool[b, 0, layer]              # [kv_heads, block, dim]
pool[b, 0, layer, :, slot, :]  # [kv_heads, dim]                    one token
```

Allocated **once**. Never per-block `cudaMalloc`/`free` — allocator overhead
would swamp the benchmark numbers this project exists to produce.

---

## Address translation

```
logical_block = pos // BLOCK_SIZE
slot          = pos %  BLOCK_SIZE
physical      = block_table[logical_block]
```

That's the whole mechanism, and it's the same one OSTEP describes:

| OS | MicroServe |
|---|---|
| virtual address | token position in a sequence |
| page | block (16 tokens) |
| physical frame | `pool[block_id]` |
| page table | `Sequence.block_table: list[int]` |
| free-frame list | `BlockAllocator`'s free list |
| MMU translates in hardware | we translate in Python, then **copy** |

The last row is the only real difference, and it's forced by the guardrail:
SDPA needs a contiguous rectangle, so we pay a gather instead of writing a
kernel that reads scattered blocks directly.

The gather, step by step (`04`, with marker values so you can read the result):

```python
pool[block_ids]        # [n_blocks, 2, layers, kv_heads, block, dim]
  [:, 0, layer]        # [n_blocks, kv_heads, block, dim]
  .permute(1, 0, 2, 3) # [kv_heads, n_blocks, block, dim]
  .reshape(H, N*B, D)  # [kv_heads, kv_len, dim]
  .unsqueeze(0)        # [batch, kv_heads, kv_len, head_dim]
```

**Permute before reshape.** `reshape` walks memory row-major, so reshaping
`[n_blocks, heads, block, dim]` directly interleaves heads into the time axis.
`04` prints both so you can see it go wrong.

---

## The payoff, in one table

`06_fragmentation.py`, 32 sequences with long-tailed prompts and early EOS:

| policy | slots held | utilization |
|---|---|---|
| reserve-max (what M2 does today) | 33,792 | 33.0% |
| per-seq-exact (perfect contiguous) | 17,232 | 64.7% |
| **paged** | **11,376** | **98.0%** |

3× more sequences in the same VRAM — and that ratio is *the* M3 number. Two
independent wastes stack up under reserve-max: every row is padded to the
longest row, and every row reserves `max_new_tokens` it usually never spends.
Paging removes both. What it can't remove is the tail of each sequence's last
block — internal fragmentation — but that's bounded by `BLOCK_SIZE`, not by
the workload: ~7 slots per sequence here against a bound of 15.

Two things worth internalising from that script:

- **On a uniform batch, paging buys exactly nothing** (case A: 100% either
  way). A rectangle *is* optimal when every sequence is the same length and
  runs to its full budget. This is why the M5 workload must be Poisson
  arrivals over ShareGPT-derived lengths, not uniform.
- **Utilization alone would pick `BLOCK_SIZE = 1`** (100%). It's not free:
  16× longer block tables, 16× more and smaller reads in a path that is
  already memory-bandwidth-bound, and constant allocator churn. 16 sits where
  the utilization curve has flattened but a block is still a big contiguous
  read.

---

## What this scratchpad deliberately does not contain

`BlockAllocator`, `Sequence.block_table`'s growth rule, the batched
scatter/gather composition, and the admission/preemption policy. Those are the
milestone — see the exercises at the bottom of `04`, and the ownership table
in `docs/plans/m03-paged-kv-cache-plan.md`.

## Read before Phase 1

OSTEP Ch 13, 15, 18–19, 21–22 — the free list you're about to write is a
free-frame list, and the correspondence is exact. Full reading gate at the top
of `docs/plans/m03-paged-kv-cache-plan.md`.
