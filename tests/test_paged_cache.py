"""Phase 3's tests, in two tiers.

Tier 1 needs no model: write recognizable integers into deliberately shuffled
blocks and assert the gather puts them back in order. This is the tier that
makes a failure *readable* -- "block order wrong at row 1" instead of "token 47
differs". Every trap Phase 0 found has a test here.

Tier 2 is the real gate: one sequence prefilled and decoded through
`PagedCache` must emit exactly the tokens HF's own `generate()` does. It loads
the tiny model on CPU and is marked slow.

The batched multi-sequence gate arrives with Phase 4's scheduler.
"""

import pytest
import torch

from engine.kv_cache import (
    BLOCK_SIZE,
    BlockAllocator,
    BlockManager,
    KVSpec,
    PagedKVPool,
    blocks_needed,
)
from engine.paged_cache import PagedBatch, PagedCache, gather_kv, scatter_index
from engine.sequence import Sequence

B = BLOCK_SIZE

# Deliberately tiny and deliberately not square: every axis has a different
# extent, so an accidental transpose or a wrong permute cannot typecheck its
# way past a shape assertion.
TINY = KVSpec(num_layers=3, num_kv_heads=2, head_dim=4, dtype=torch.float32, block_size=B)


@pytest.fixture
def pool():
    return PagedKVPool(num_blocks=16, spec=TINY)


@pytest.fixture
def mgr():
    return BlockManager(BlockAllocator(num_blocks=16))


def make_seq(seq_id: int, prompt_len: int, max_new_tokens: int = 8) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        max_new_tokens=max_new_tokens,
    )


def marker(pos: int, head: int, dim: int) -> float:
    """A value that encodes where it came from, so a misplaced element names its
    own origin in the assertion message."""
    return pos * 1000.0 + head * 10.0 + dim


def fill_marked(pool: PagedKVPool, block_table: list[int], num_tokens: int, layer: int) -> None:
    """Write `marker(pos, ...)` into the pool at every logical position, going
    through the block table -- so this helper is itself a check that the block
    table is being honoured."""
    for pos in range(num_tokens):
        blk = block_table[pos // B]
        s = pos % B
        for h in range(TINY.num_kv_heads):
            for d in range(TINY.head_dim):
                pool.tensor[blk, 0, layer, h, s, d] = marker(pos, h, d)
                pool.tensor[blk, 1, layer, h, s, d] = -marker(pos, h, d)


# --- Tier 1a: the gather ------------------------------------------------------


def test_gather_returns_the_declared_shape(pool):
    block_table = torch.tensor([[3, 1, 7]], dtype=torch.long)
    keys, values = gather_kv(pool.tensor, block_table, layer_idx=0)
    assert keys.shape == (1, TINY.num_kv_heads, 3 * B, TINY.head_dim)
    assert values.shape == keys.shape


def test_gather_reassembles_shuffled_blocks_in_logical_order(pool):
    """The core claim. Blocks are handed out in whatever order the free list
    happens to be in, so the physical ids are meaningless -- only the block
    table's order is meaningful."""
    block_table = [11, 2, 9]  # deliberately not ascending, not contiguous
    n = 3 * B
    fill_marked(pool, block_table, n, layer=1)

    keys, values = gather_kv(pool.tensor, torch.tensor([block_table], dtype=torch.long), 1)

    for pos in range(n):
        for h in range(TINY.num_kv_heads):
            for d in range(TINY.head_dim):
                assert keys[0, h, pos, d].item() == marker(pos, h, d), (
                    f"key at pos {pos} head {h} dim {d} came from the wrong place"
                )
                assert values[0, h, pos, d].item() == -marker(pos, h, d)


def test_gather_does_not_interleave_heads_into_time(pool):
    """Phase 0's silent trap: `reshape` walks row-major, so folding the block
    axis into the time axis while the head axis sits between them scrambles the
    two. Every value is present and every shape checks out -- only the tokens
    are wrong. Two heads with disjoint value ranges catch it immediately."""
    block_table = [5, 0]
    for pos in range(2 * B):
        blk, s = block_table[pos // B], pos % B
        pool.tensor[blk, 0, 0, 0, s, :] = float(pos)  # head 0: 0..31
        pool.tensor[blk, 0, 0, 1, s, :] = float(pos) + 100.0  # head 1: 100..131

    keys, _ = gather_kv(pool.tensor, torch.tensor([block_table], dtype=torch.long), 0)

    assert (keys[0, 0] < 100).all(), "head 1's values leaked into head 0's timeline"
    assert (keys[0, 1] >= 100).all(), "head 0's values leaked into head 1's timeline"
    assert keys[0, 0, :, 0].tolist() == [float(p) for p in range(2 * B)]


def test_gather_reads_only_the_requested_layer(pool):
    """Layers share a block. Reading layer 1 must not see layer 0's values --
    the bug where the layer index is applied after the gather instead of inside
    it shows up here as well as in the profiler."""
    blk = 4
    pool.tensor[blk, 0, 0] = 1.0
    pool.tensor[blk, 0, 1] = 2.0
    pool.tensor[blk, 0, 2] = 3.0
    for layer, expected in enumerate((1.0, 2.0, 3.0)):
        keys, _ = gather_kv(pool.tensor, torch.tensor([[blk]], dtype=torch.long), layer)
        assert (keys == expected).all(), f"layer {layer} gathered another layer's KV"


def test_gather_keeps_rows_independent(pool):
    """Two rows, disjoint blocks, different lengths. Row 0's history must not
    bleed into row 1's, which is the multi-sequence failure that produces
    plausible wrong tokens rather than a crash."""
    a, b = [6, 3], [1, 12]
    for row, table in enumerate((a, b)):
        for pos in range(2 * B):
            blk, s = table[pos // B], pos % B
            pool.tensor[blk, 0, 0, :, s, :] = float(row * 1000 + pos)

    keys, _ = gather_kv(pool.tensor, torch.tensor([a, b], dtype=torch.long), 0)
    assert keys[0, 0, :, 0].tolist() == [float(p) for p in range(2 * B)]
    assert keys[1, 0, :, 0].tolist() == [float(1000 + p) for p in range(2 * B)]


# --- Tier 1b: the scatter -----------------------------------------------------


def test_scatter_index_covers_every_real_token_and_nothing_else():
    """N is `sum(q_lens)`, not `batch * padded_q_len`. Writing the padded query
    columns would clobber a neighbour's KV or the live tail of a block."""
    batch = PagedBatch(block_tables=[[4, 9], [2]], past_lens=[0, 0], q_lens=[20, 5])
    rows, cols, blocks, slots = scatter_index(batch)
    assert len(rows) == len(cols) == len(blocks) == len(slots) == 25
    assert rows.tolist() == [0] * 20 + [1] * 5
    assert cols.tolist() == list(range(20)) + list(range(5))


def test_scatter_index_translates_through_the_block_table():
    batch = PagedBatch(block_tables=[[7, 3]], past_lens=[0], q_lens=[B + 2])
    _, _, blocks, slots = scatter_index(batch)
    assert blocks.tolist() == [7] * B + [3, 3]
    assert slots.tolist() == list(range(B)) + [0, 1]


def test_scatter_index_offsets_by_past_length():
    """A decode step writes at position `past`, not at position 0. This is the
    off-by-one that overwrites the prompt's first token with every new token
    and still produces fluent-looking output."""
    batch = PagedBatch(block_tables=[[8, 5]], past_lens=[B + 3], q_lens=[1])
    _, cols, blocks, slots = scatter_index(batch)
    assert cols.tolist() == [0], "the single new token is at query column 0"
    assert blocks.tolist() == [5], "position 19 lives in logical block 1"
    assert slots.tolist() == [3]


def test_scatter_then_gather_round_trips(pool):
    """The two halves must be inverses. Written separately they can each look
    right and disagree about block order."""
    block_table = [10, 1, 6]
    n = 2 * B + 5
    batch = PagedBatch(block_tables=[block_table], past_lens=[0], q_lens=[n])
    rows, cols, blocks, slots = scatter_index(batch)

    keys = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1).expand(
        1, TINY.num_kv_heads, n, TINY.head_dim
    ).contiguous()
    pool.tensor[blocks, 0, 0, :, slots, :] = keys[rows, :, cols, :]

    gathered, _ = gather_kv(pool.tensor, torch.tensor([block_table], dtype=torch.long), 0)
    assert gathered[0, 0, :n, 0].tolist() == [float(p) for p in range(n)]


def test_scatter_writes_a_decode_token_without_disturbing_the_history(pool):
    """A decode step touches exactly one slot per row. If it touches more, the
    damage is to tokens that were already correct."""
    block_table = [10, 1]
    n = B + 4
    fill_marked(pool, block_table, n, layer=0)
    before = pool.tensor.clone()

    batch = PagedBatch(block_tables=[block_table], past_lens=[n], q_lens=[1])
    rows, cols, blocks, slots = scatter_index(batch)
    new = torch.full((1, TINY.num_kv_heads, 1, TINY.head_dim), 99.0)
    pool.tensor[blocks, 0, 0, :, slots, :] = new[rows, :, cols, :]

    assert (pool.tensor[10] == before[10]).all(), "the full first block was modified"
    changed = (pool.tensor != before).nonzero()
    assert {tuple(c[:2].tolist()) for c in changed} == {(1, 0)}, "wrote outside the target block"
    assert (pool.tensor[1, 0, 0, :, 4, :] == 99.0).all()


# --- Tier 1c: the batch descriptor and the promise ----------------------------


def test_kv_length_is_block_padded_not_exact():
    """Phase 0 finding 3: returning `ceil(len/B)*B` positions with junk in the
    tail is legal, because causality masks the tail. So the gather never has to
    trim, which is what keeps it a single indexing op."""
    batch = PagedBatch(block_tables=[[1, 2]], past_lens=[B + 1], q_lens=[1])
    assert batch.kv_length == 2 * B


def test_kv_length_follows_the_widest_row():
    batch = PagedBatch(block_tables=[[1], [2, 3, 4]], past_lens=[3, 40], q_lens=[1, 1])
    assert batch.kv_length == 3 * B


def test_mask_sizes_promise_matches_the_gathered_width(pool):
    """`get_mask_sizes` is a promise made *before* `update` runs, and the 4-D
    causal mask is built from it. Disagree and attention raises a shape error."""
    cache = PagedCache(pool)
    batch = PagedBatch(block_tables=[[3, 1], [5]], past_lens=[B, 4], q_lens=[1, 1])
    cache.set_batch(batch)

    promised, offset = cache.get_mask_sizes(query_length=1, layer_idx=0)
    assert offset == 0
    gathered, _ = gather_kv(pool.tensor, batch.block_table_tensor(), 0)
    assert gathered.shape[-2] == promised


def test_cache_builds_every_layer_up_front(pool):
    """Phase 0 finding 2: `Cache.get_mask_sizes` short-circuits to
    `(query_length, 0)` while `self.layers` is short, so a lazily-grown cache
    would have our declared width ignored on the prefill call and only there."""
    cache = PagedCache(pool)
    assert len(cache.layers) == TINY.num_layers


def test_attention_mask_covers_each_row_to_its_own_length():
    batch = PagedBatch(block_tables=[[1, 2], [3, 4]], past_lens=[B, 3], q_lens=[2, 1])
    mask = batch.attention_mask()
    assert mask.shape == (2, 2 * B)
    assert mask[0].sum().item() == B + 2
    assert mask[1].sum().item() == 4
    assert mask[0, : B + 2].all() and not mask[0, B + 2 :].any()


def test_position_ids_are_absolute_per_row():
    """RoPE needs the true position. Row 1 is 40 tokens in; its next token is at
    40, not at 0, whatever the padding says."""
    batch = PagedBatch(block_tables=[[1], [1, 2, 3]], past_lens=[0, 40], q_lens=[1, 1])
    assert batch.position_ids(1).tolist() == [[0], [40]]


def test_last_logit_index_is_the_price_of_right_padding():
    """`logits[:, -1, :]` is correct only for the longest row once rows are
    right-padded, and reading it anyway gives the shorter rows a prediction made
    from padding."""
    batch = PagedBatch(block_tables=[[1], [1]], past_lens=[0, 0], q_lens=[9, 3])
    assert batch.last_logit_index().tolist() == [8, 2]


def test_batch_refuses_rows_that_lack_capacity():
    """Capacity is BlockManager's job; a step built without calling append_slot
    must fail here rather than write past the end of a block."""
    with pytest.raises(ValueError, match="needs 2 blocks"):
        PagedBatch(block_tables=[[4]], past_lens=[B], q_lens=[1])


def test_layer_refuses_to_run_without_a_batch(pool):
    cache = PagedCache(pool)
    with pytest.raises(RuntimeError, match="set_batch"):
        cache.get_mask_sizes(query_length=1, layer_idx=0)


# --- Tier 1d: the pieces composed the way the engine will use them ------------


def test_a_sequence_grows_through_manager_and_batch_together(mgr, pool):
    """The Phase 2 growth rule and the Phase 3 addressing have to agree about
    what `num_cached_tokens` means. Walk a sequence across two block boundaries
    driving both."""
    seq = make_seq(1, prompt_len=B - 2)
    mgr.allocate_for_prefill(seq)

    batch = PagedBatch([seq.block_table], [0], [len(seq.prompt_token_ids)])
    _, _, blocks, slots = scatter_index(batch)
    assert len(blocks) == B - 2
    seq.num_cached_tokens = B - 2

    for _ in range(2 * B):
        mgr.append_slot(seq)
        batch = PagedBatch([seq.block_table], [seq.num_cached_tokens], [1])
        _, _, blocks, slots = scatter_index(batch)
        pos = seq.num_cached_tokens
        assert blocks.tolist() == [seq.block_table[pos // B]]
        assert slots.tolist() == [pos % B]
        seq.num_cached_tokens += 1
        assert len(seq.block_table) == blocks_needed(seq.num_cached_tokens)


def test_two_sequences_never_scatter_into_the_same_slot(mgr):
    """The failure this whole milestone has to rule out: two live rows writing
    to one physical address, no crash, plausible wrong tokens."""
    a, b = make_seq(1, prompt_len=B + 5), make_seq(2, prompt_len=B + 5)
    mgr.allocate_for_prefill(a)
    mgr.allocate_for_prefill(b)
    a.num_cached_tokens = b.num_cached_tokens = B + 5

    for _ in range(B):
        for seq in (a, b):
            mgr.append_slot(seq)
        batch = PagedBatch(
            [a.block_table, b.block_table],
            [a.num_cached_tokens, b.num_cached_tokens],
            [1, 1],
        )
        _, _, blocks, slots = scatter_index(batch)
        addresses = list(zip(blocks.tolist(), slots.tolist()))
        assert len(set(addresses)) == len(addresses), f"aliased write: {addresses}"
        a.num_cached_tokens += 1
        b.num_cached_tokens += 1


# --- Tier 2: the real gate ----------------------------------------------------
#
# Pinned to CPU + eager, same as tests/test_correctness.py, for the reason in
# ADR-0010: SDPA's fused kernels are not numerically invariant to KV-buffer
# shape, and the paged path changes exactly that. A mismatch here must mean a
# paging bug, not a kernel picking a different reduction order.

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GATE_KWARGS = {"device": "cpu", "attn_implementation": "eager"}
MAX_NEW_TOKENS = 20
GATE_PROMPTS = [
    "The capital of France is",
    # Long enough to cross a block boundary during decode, which single-block
    # prompts would never exercise.
    "Write a short explanation of why operating systems use virtual memory, "
    "covering fragmentation and address translation:",
]


@pytest.fixture(scope="module")
def runner():
    from engine.model_runner import ModelRunner

    return ModelRunner(MODEL_ID, **GATE_KWARGS)


@pytest.fixture(scope="module")
def golden(runner):
    def _golden(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> list[int]:
        ids = runner.encode(prompt)
        return runner.generate(ids, max_new_tokens)[len(ids) :]

    return _golden


def paged_generate(runner, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
    """Drive one sequence end to end through the paged path.

    This is the shape Phase 4's scheduler will have, minus admission and
    preemption: allocate for the prompt, prefill, then per step ask the manager
    for one more slot, decode, and advance `num_cached_tokens` -- *after* the
    forward pass, never before, because capacity runs ahead of content.
    """
    spec = KVSpec.from_config(runner.model.config, dtype=runner.model.dtype)
    pool = PagedKVPool(num_blocks=64, spec=spec, device=runner.device)
    mgr = BlockManager(BlockAllocator(num_blocks=64))
    cache = PagedCache(pool)

    seq = Sequence(seq_id=0, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens)
    mgr.allocate_for_prefill(seq)

    batch = PagedBatch([seq.block_table], [0], [len(prompt_token_ids)], device=runner.device)
    next_ids = runner.paged_prefill(cache, batch, [prompt_token_ids])
    seq.num_cached_tokens = len(prompt_token_ids)
    generated = [next_ids[0]]

    for _ in range(max_new_tokens - 1):
        mgr.append_slot(seq)
        batch = PagedBatch(
            [seq.block_table], [seq.num_cached_tokens], [1], device=runner.device
        )
        next_ids = runner.paged_decode_step(cache, batch, [generated[-1]])
        seq.num_cached_tokens += 1
        generated.append(next_ids[0])

    return generated


@pytest.mark.parametrize("prompt", GATE_PROMPTS)
def test_paged_single_sequence_matches_golden(runner, golden, prompt):
    """Phase 3's done-when, in one assertion. Every block-order, slot, mask-width
    and permute bug ends here as a token mismatch."""
    assert paged_generate(runner, runner.encode(prompt), MAX_NEW_TOKENS) == golden(prompt)


def test_paged_matches_golden_across_a_block_boundary(runner, golden):
    """A prompt sized so that decode crosses a block boundary a few tokens in --
    the growth path, exercised against real logits rather than bookkeeping."""
    prompt = GATE_PROMPTS[0]
    ids = runner.encode(prompt)
    # Pad the prompt so len(ids) is 3 short of a boundary: decode then grows a
    # block on step 4, mid-generation, with real history already in the pool.
    pad = (-len(ids) - 3) % BLOCK_SIZE
    if pad:
        ids = runner.encode("A" * pad + " " + prompt)
    assert paged_generate(runner, ids, MAX_NEW_TOKENS) == runner.generate(ids, MAX_NEW_TOKENS)[
        len(ids) :
    ]
