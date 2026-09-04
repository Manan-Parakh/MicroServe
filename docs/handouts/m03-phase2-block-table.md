# Handout — M3 Phase 2: the block table and the growth rule

Paste this whole file into a fresh Claude chat as your first message. It carries
enough context that the chat doesn't need the repo.

---

## Instructions to the assistant reading this

I am building an LLM inference engine from scratch, alone, to **learn how
inference engines work**. The working engine is the byproduct; the understanding
is the product.

**Do not write the address translation or the growth rule for me.** Not as a
"sketch", not as "here's roughly what it looks like, adapt it", not as a complete
method with a `# TODO` in the middle. This is six lines of arithmetic and it is
the entire point of the phase — it's small precisely *because* the difficulty is
in getting it exactly right, not in typing it. If I ask you for the code
directly, push back once and offer the alternatives below.

What I want from you instead:

- **Adversarial boundary testing.** Give me a position, ask me which block and
  slot it lands in, and tell me if I'm wrong. Especially at `p % BLOCK_SIZE == 0`.
- **Trace-through review.** When I paste code, walk a concrete sequence through
  it token by token across two block boundaries and tell me where it diverges
  from what I claimed it does.
- **Design discussion with real alternatives**, including the ones I've already
  settled — I want to be able to defend them, not just have made them.
- **Systems context by proper name.** I've read OSTEP Ch 13, 15, 18–19, 21–22.
  Connect this to page tables, address translation, and internal fragmentation.
  Don't re-explain attention, KV caching, or GQA; I know those.
- **Pseudocode, prose, invariants, counterexamples**: all fine. Working Python
  for the three helpers or the three `BlockManager` methods: not fine.

Be concise. Point rather than lecture.

---

## The project, in a paragraph

MicroServe is a from-scratch LLM inference engine on a single GPU. Milestone 2
gave it continuous batching but a rectangular `StaticCache` — one tensor of
`batch × max(prompt + max_new_tokens)`, which measured 33% utilization on a
realistic long-tailed workload. Milestone 3 replaces it with a **paged KV cache**
(vLLM's PagedAttention, minus the custom kernel): fixed-size blocks, scattered
across one pre-allocated pool, gathered into a contiguous scratch tensor at
attention time because we call PyTorch's `scaled_dot_product_attention` rather
than writing a kernel. Target is 98% utilization, ~3× the concurrent sequences.

**Phase 1 is done**: `PagedKVPool` (one `torch.zeros` of shape
`[num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]`) and
`BlockAllocator` (LIFO free list, ref counts, `BlockExhausted` as a catchable
type because exhaustion is normal control flow in a paged engine).

**Phase 2, this phase, is the block table**: the per-sequence map from logical
token position to physical block. It is a single-level page table.

---

## What already exists (written for me, not by me)

`BLOCK_SIZE = 16`. `Sequence` now carries two new fields:

```python
@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    block_table: list[int] = field(default_factory=list)
    # entry i holds physical block for positions [i*16, (i+1)*16)
    # the LIST INDEX is the logical block number -- no dict needed

    num_cached_tokens: int = 0
    # positions whose KV is actually WRITTEN in the pool.
    # deliberately not derived from len(prompt)+len(output), because
    # preemption-by-recomputation drops KV while keeping tokens.
```

And `BlockManager` exists as a skeleton holding the allocator, with a
`num_free` property and three `NotImplementedError` methods.

---

## What I have to write

Three pure functions:

```python
def logical_block(pos: int, block_size: int = BLOCK_SIZE) -> int: ...
def slot(pos: int, block_size: int = BLOCK_SIZE) -> int: ...
def blocks_needed(num_tokens: int, block_size: int = BLOCK_SIZE) -> int: ...
```

and three `BlockManager` methods:

```python
def allocate_for_prefill(self, seq) -> None: ...   # at admission
def append_slot(self, seq) -> None: ...            # before each decode step
def free(self, seq) -> None: ...                   # at eviction OR preemption
```

**The contract I've settled on, so don't reopen it — but do quiz me on why:**

1. **These methods are about CAPACITY, not CONTENT.** None of them touches
   `num_cached_tokens` except `free`, which zeroes it. The KV doesn't exist until
   the forward pass writes it, so the caller advances the counter after a
   *successful* step. If the forward raises, the counter must not have moved.
2. **The invariant that follows:**
   `0 <= num_cached_tokens <= len(block_table) * BLOCK_SIZE`. Capacity runs one
   step ahead of content, never behind.
3. **`append_slot` is idempotent.** It asks for capacity; it doesn't consume it.
   Calling it twice without a token in between must not allocate twice.
4. **`free` is also the preemption path**, not just the completion path — which
   is why it resets `num_cached_tokens` rather than assuming the sequence is done.
   The sequence must remain re-admittable afterwards.
5. **`blocks_needed(0) == 0.`** A sequence with no cached tokens owns no blocks.

---

## Where the bugs actually are

Don't answer these for me up front — ask me what I think first.

- **The growth condition.** "A new block is needed when the next token would land
  at offset 0 of a block the sequence doesn't own." Make me write that as an
  expression in terms of `num_cached_tokens` and `len(block_table)`, then attack
  it. There are at least three plausible-looking wrong versions.
- **`blocks_needed(16)`.** Is it 1 or 2? Make me justify it, then make me check
  it's consistent with `logical_block(15)` and `logical_block(16)`.
- **Ceiling division.** `-(-n // b)`, `(n + b - 1) // b`, `math.ceil(n / b)` —
  one of these has a failure mode the others don't. Which, and when?
- **The failure path in `allocate_for_prefill`.** The prompt needs 5 blocks and 3
  are free. What must be true of the pool *and* of `seq.block_table` after the
  exception? What's the general name for getting that wrong?
- **Off-by-one asymmetry.** Growing one token *early* wastes a block. Growing one
  token *late* writes past the end of a block. Which one is worse, and which one
  would a test catch?

Design questions I actually want to argue about:

- Why is a block table a flat `list[int]` when an OS page table is four levels?
  Name the property of the address space that differs — not the size.
- Why does `block_table` live on `Sequence` rather than in a
  `dict[seq_id, list[int]]` inside `BlockManager` (which is what vLLM does)?
  What does each choice make easy?
- Should `allocate_for_prefill` reserve headroom for generation, or only the
  prompt? Prompt-only is what I've built. What breaks, and when?
- `num_cached_tokens` is a second source of truth that can drift out of sync with
  `block_table`. What single assertion, placed where, makes drift impossible to
  ship?

---

## How I'll know it's done

`tests/test_kv_cache.py` has 26 failing tests for this phase (29 Phase 1 tests
already pass). They run with no model and no GPU. The ones that matter:

- `test_the_block_boundary_itself` — positions 15, 16, 17
- `test_blocks_needed_agrees_with_translation` — the two pieces of arithmetic
  can't be allowed to disagree
- `test_append_slot_grows_exactly_at_the_boundary` and
  `test_append_slot_does_not_grow_one_token_early` — the canonical bug, from both
  sides
- `test_growth_token_by_token_across_several_boundaries` — 64 single-token steps,
  re-checking the invariant after every one
- `test_free_is_how_preemption_works_not_just_completion`

If you want to help me sharpen the tests, welcome. Writing the implementation is
the one thing I need you not to do.

---

## Next, so you know where this is heading

- **Phase 3** — the HF `Cache` subclass: scatter new K/V into
  `pool[block_id, :, layer, :, slot, :]`, gather the history back into a
  contiguous `[batch, kv_heads, kv_len, head_dim]` rectangle. Block-padded gather
  is legal — I verified in a spike that causality masks the tail.
- **Phase 4** — scheduler integration, and the open question of what happens when
  blocks run out mid-generation: refuse admission only, preempt by recomputation,
  or preempt by swapping to host memory.
