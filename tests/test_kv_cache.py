"""The most paranoid test file in the repo, on purpose.

`claude.md` calls block-indexing bugs "exactly the kind that hide", and this is
the layer where hiding is cheapest: a double-freed block id is a plain integer
appearing twice in a list, and the symptom is a plausible-looking wrong token
several thousand forward passes later. So the invariants get checked after
*every* operation, not just at the end.

No model, no GPU, no `transformers`. Pure bookkeeping, in the FakeModelRunner
spirit of tests/test_scheduler.py.
"""

import random

import pytest
import torch

from engine.kv_cache import (
    BLOCK_SIZE,
    BlockAllocator,
    BlockExhausted,
    BlockManager,
    KVSpec,
    PagedKVPool,
    blocks_for_gib,
    blocks_needed,
    internal_fragmentation,
    logical_block,
    slot,
)
from engine.sequence import Sequence


class FakeConfig:
    """Qwen2.5-0.5B-Instruct's real numbers. 14 query heads, 2 KV heads — the
    GQA gap is the whole reason this test exists."""

    num_hidden_layers = 24
    num_attention_heads = 14
    num_key_value_heads = 2
    hidden_size = 896
    head_dim = 64


def qwen_spec(dtype=torch.bfloat16) -> KVSpec:
    return KVSpec.from_config(FakeConfig(), dtype=dtype)


# --- KVSpec: the byte arithmetic ---------------------------------------------


def test_spec_from_config_uses_kv_heads_not_attention_heads():
    spec = qwen_spec()
    assert spec.num_kv_heads == 2
    assert spec.num_layers == 24
    assert spec.head_dim == 64


def test_spec_falls_back_to_hidden_size_over_heads_when_head_dim_absent():
    class NoHeadDim(FakeConfig):
        head_dim = None

    assert KVSpec.from_config(NoHeadDim(), dtype=torch.float32).head_dim == 896 // 14


def test_spec_defaults_to_mha_when_config_has_no_kv_heads():
    class MHA:
        num_hidden_layers = 2
        num_attention_heads = 8
        hidden_size = 64
        head_dim = 8

    assert KVSpec.from_config(MHA(), dtype=torch.float32).num_kv_heads == 8


def test_bytes_per_token_is_12_kib_for_qwen_bf16():
    spec = qwen_spec()
    # 2 (K,V) * 24 layers * 2 kv heads * 64 dim = 6144 elements * 2 bytes
    assert spec.elements_per_token == 6144
    assert spec.bytes_per_token == 12 * 1024
    assert spec.bytes_per_block == 12 * 1024 * BLOCK_SIZE  # 192 KiB


def test_dtype_changes_the_byte_count_but_not_the_element_count():
    assert qwen_spec(torch.float32).bytes_per_token == 2 * qwen_spec(torch.bfloat16).bytes_per_token
    assert (
        qwen_spec(torch.float32).elements_per_token == qwen_spec(torch.bfloat16).elements_per_token
    )


def test_blocks_for_gib_matches_hand_arithmetic():
    # 1 GiB / 192 KiB per block
    assert blocks_for_gib(1.0, qwen_spec()) == 1024**3 // (192 * 1024) == 5461


def test_spec_is_immutable():
    with pytest.raises(Exception):
        qwen_spec().num_layers = 99


# --- PagedKVPool: one tensor, allocated once ---------------------------------


def test_pool_shape_is_the_locked_layout():
    spec = qwen_spec()
    pool = PagedKVPool(num_blocks=8, spec=spec)
    assert tuple(pool.shape) == (8, 2, 24, 2, BLOCK_SIZE, 64)


def test_pool_block_id_indexes_dim_zero_and_is_contiguous():
    pool = PagedKVPool(num_blocks=4, spec=qwen_spec(torch.float32))
    assert pool.tensor[2].is_contiguous()
    assert pool.tensor[2].shape == pool.tensor.shape[1:]


def test_pool_is_zeroed_not_uninitialised():
    """Phase 0 finding 5: an `inf` left in a masked slot makes 0 * inf = NaN and
    silently destroys the output. torch.empty would pass this test only by luck,
    so also assert it is finite everywhere."""
    pool = PagedKVPool(num_blocks=4, spec=qwen_spec(torch.float32))
    assert torch.all(pool.tensor == 0)
    assert torch.isfinite(pool.tensor).all()


def test_pool_size_reporting_matches_the_spec():
    spec = qwen_spec()
    pool = PagedKVPool(num_blocks=16, spec=spec)
    assert pool.nbytes == 16 * spec.bytes_per_block
    assert pool.capacity_tokens == 16 * BLOCK_SIZE


# --- BlockAllocator: the free list -------------------------------------------

N = 8  # small enough that exhaustion is reachable in one line


@pytest.fixture
def alloc():
    return BlockAllocator(num_blocks=N)


def assert_consistent(a: BlockAllocator, live: set[int]):
    """The invariant, checked after every mutation in every test below:
    free + live == every block, exactly once, and ref counts agree."""
    assert a.num_free == N - len(live)
    for b in range(N):
        assert a.ref_count(b) == (1 if b in live else 0), f"block {b} ref count disagrees"


def test_starts_fully_free(alloc):
    assert alloc.num_free == N
    assert_consistent(alloc, set())


def test_allocate_returns_distinct_in_range_ids(alloc):
    ids = alloc.allocate(3)
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert all(0 <= b < N for b in ids)
    assert_consistent(alloc, set(ids))


def test_allocate_zero_is_a_noop(alloc):
    assert alloc.allocate(0) == []
    assert_consistent(alloc, set())


def test_allocate_to_exhaustion_then_refuse(alloc):
    ids = alloc.allocate(N)
    assert sorted(ids) == list(range(N))
    assert alloc.num_free == 0
    with pytest.raises(BlockExhausted):
        alloc.allocate(1)


def test_over_allocation_raises_before_mutating_anything(alloc):
    """A partially-served allocate() that then throws leaks blocks no block table
    records — nothing would ever free them. Check the pool is untouched."""
    with pytest.raises(BlockExhausted):
        alloc.allocate(N + 1)
    assert_consistent(alloc, set())
    assert sorted(alloc.allocate(N)) == list(range(N))


def test_exhaustion_is_catchable_as_more_than_indexerror(alloc):
    """Phase 4's scheduler catches this as normal control flow, so it must not be
    a bare IndexError escaping from list.pop()."""
    alloc.allocate(N)
    with pytest.raises(BlockExhausted):
        alloc.allocate(1)


def test_free_returns_blocks_and_they_are_reusable(alloc):
    ids = alloc.allocate(N)
    alloc.free(ids[:3])
    assert_consistent(alloc, set(ids[3:]))
    again = alloc.allocate(3)
    assert set(again) == set(ids[:3])
    assert_consistent(alloc, set(ids))


def test_full_cycle_returns_to_pristine(alloc):
    ids = alloc.allocate(N)
    alloc.free(ids)
    assert_consistent(alloc, set())
    assert sorted(alloc.allocate(N)) == list(range(N))


def test_free_is_lifo(alloc):
    """Pins the data structure: a stack, so the most recently freed block is the
    next one handed out. Correctness does not depend on this — locality does."""
    ids = alloc.allocate(N)
    alloc.free([ids[0], ids[1]])
    assert alloc.allocate(1) == [ids[1]]


def test_double_free_raises(alloc):
    """The worst bug in the milestone: a block in the free list twice gets handed
    to two live sequences, and they overwrite each other's KV. Wrong tokens, no
    crash."""
    ids = alloc.allocate(2)
    alloc.free(ids)
    with pytest.raises(Exception) as e:
        alloc.free([ids[0]])
    assert not isinstance(e.value, BlockExhausted)
    assert_consistent(alloc, set())


def test_freeing_a_never_allocated_block_raises(alloc):
    with pytest.raises(Exception):
        alloc.free([0])
    assert_consistent(alloc, set())


def test_free_is_all_or_nothing(alloc):
    """The mirror of test_over_allocation_raises_before_mutating_anything, and the
    more dangerous direction. A free that reclaims some ids and then raises leaves
    a block in the free list *and* in a live sequence's block table."""
    live = alloc.allocate(4)
    dead = [b for b in range(N) if b not in live]
    with pytest.raises(Exception):
        alloc.free([live[0], dead[0]])  # one valid, one never allocated
    assert_consistent(alloc, set(live))


def test_freeing_the_same_id_twice_in_one_call_raises(alloc):
    """`free([x, x])` is the same bug as two separate frees, just harder to see —
    a block table that somehow recorded x twice. It must not slip through because
    both entries were validated against the same pre-mutation ref count."""
    ids = alloc.allocate(2)
    with pytest.raises(Exception):
        alloc.free([ids[0], ids[0]])
    assert_consistent(alloc, set(ids))


def test_free_empty_list_is_a_noop(alloc):
    ids = alloc.allocate(2)
    alloc.free([])
    assert_consistent(alloc, set(ids))


def test_two_sequences_never_share_a_block(alloc):
    a = alloc.allocate(3)
    b = alloc.allocate(3)
    assert set(a).isdisjoint(b)
    assert_consistent(alloc, set(a) | set(b))
    alloc.free(a)
    c = alloc.allocate(2)
    assert set(c).isdisjoint(b)


def test_ref_counts_are_only_ever_zero_or_one_in_m3(alloc):
    """Locked decision 6. The mechanism is here so M6's prefix sharing extends it;
    in M3 anything above 1 means a bug."""
    ids = alloc.allocate(4)
    assert all(alloc.ref_count(b) in (0, 1) for b in range(N))
    alloc.free(ids)
    assert all(alloc.ref_count(b) == 0 for b in range(N))


def test_fuzz_invariants_hold_after_every_operation():
    """Random allocate/free traffic. The point is not the final state — it is that
    num_free, the ref counts, and the set of live ids agree at every single step."""
    rng = random.Random(0)
    size = 32
    a = BlockAllocator(num_blocks=size)
    live: set[int] = set()

    for _ in range(2000):
        if rng.random() < 0.5:
            n = rng.randint(0, 5)
            if n > size - len(live):
                with pytest.raises(BlockExhausted):
                    a.allocate(n)
            else:
                got = a.allocate(n)
                assert len(set(got)) == n
                assert set(got).isdisjoint(live), "handed out a block that was already live"
                live |= set(got)
        elif live:
            victims = rng.sample(sorted(live), rng.randint(1, len(live)))
            a.free(victims)
            live -= set(victims)

        assert a.num_free == size - len(live)
        for b in range(size):
            assert a.ref_count(b) == (1 if b in live else 0)


# --- Phase 2: address translation --------------------------------------------
#
# Everything below uses BLOCK_SIZE as imported, so it stays correct if the
# constant is retuned. Hand-computed expectations use B for readability.

B = BLOCK_SIZE


def test_logical_block_and_slot_partition_positions():
    """Every position maps to exactly one (block, slot), and the pair round-trips."""
    for pos in range(0, 5 * B):
        lb, s = logical_block(pos), slot(pos)
        assert 0 <= s < B
        assert lb * B + s == pos, f"pos {pos} does not round-trip"


def test_position_zero_is_block_zero_slot_zero():
    assert logical_block(0) == 0
    assert slot(0) == 0


def test_the_block_boundary_itself():
    """The canonical off-by-one. Last slot of block 0, first slot of block 1."""
    assert (logical_block(B - 1), slot(B - 1)) == (0, B - 1)
    assert (logical_block(B), slot(B)) == (1, 0)
    assert (logical_block(B + 1), slot(B + 1)) == (1, 1)


def test_blocks_needed_at_and_around_every_boundary():
    assert blocks_needed(0) == 0, "a sequence with no KV owns no blocks"
    assert blocks_needed(1) == 1
    assert blocks_needed(B - 1) == 1
    assert blocks_needed(B) == 1, "exactly full is still one block"
    assert blocks_needed(B + 1) == 2
    assert blocks_needed(3 * B) == 3
    assert blocks_needed(3 * B + 1) == 4


def test_blocks_needed_agrees_with_translation():
    """The two pieces of arithmetic must not be able to disagree: the block that
    holds the last position is the last block needed."""
    for n in range(1, 4 * B + 2):
        assert blocks_needed(n) == logical_block(n - 1) + 1


def test_helpers_respect_a_non_default_block_size():
    assert logical_block(9, block_size=4) == 2
    assert slot(9, block_size=4) == 1
    assert blocks_needed(9, block_size=4) == 3


# --- Phase 2: BlockManager ---------------------------------------------------

MGR_BLOCKS = 64


@pytest.fixture
def mgr():
    return BlockManager(BlockAllocator(num_blocks=MGR_BLOCKS))


def make_paged_seq(seq_id: int, prompt_len: int, max_new_tokens: int = 32) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        max_new_tokens=max_new_tokens,
    )


def assert_seq_invariant(seq: Sequence, m: BlockManager):
    """Capacity runs one step ahead of content, never behind. Checked after every
    single mutation in the tests below."""
    capacity = len(seq.block_table) * B
    assert 0 <= seq.num_cached_tokens <= capacity, (
        f"seq {seq.seq_id}: {seq.num_cached_tokens} cached tokens but only "
        f"{capacity} slots owned"
    )
    assert len(set(seq.block_table)) == len(seq.block_table), "duplicate block in table"
    for b in seq.block_table:
        assert m.allocator.ref_count(b) >= 1, f"block {b} in a table but not allocated"


def test_fresh_sequence_owns_nothing(mgr):
    seq = make_paged_seq(1, prompt_len=10)
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert_seq_invariant(seq, mgr)


@pytest.mark.parametrize(
    "prompt_len,expected_blocks",
    [(1, 1), (B - 1, 1), (B, 1), (B + 1, 2), (2 * B, 2), (2 * B + 1, 3)],
)
def test_prefill_allocates_exactly_ceil_prompt_over_block_size(mgr, prompt_len, expected_blocks):
    seq = make_paged_seq(1, prompt_len=prompt_len)
    mgr.allocate_for_prefill(seq)
    assert len(seq.block_table) == expected_blocks
    assert mgr.num_free == MGR_BLOCKS - expected_blocks
    assert_seq_invariant(seq, mgr)


def test_prefill_does_not_claim_tokens_are_cached(mgr):
    """allocate_for_prefill is about CAPACITY. The KV does not exist until the
    forward pass writes it, and the caller advances num_cached_tokens then."""
    seq = make_paged_seq(1, prompt_len=20)
    mgr.allocate_for_prefill(seq)
    assert seq.num_cached_tokens == 0


def test_prefill_refuses_when_the_pool_cannot_cover_the_prompt(mgr):
    seq = make_paged_seq(1, prompt_len=(MGR_BLOCKS + 1) * B)
    with pytest.raises(BlockExhausted):
        mgr.allocate_for_prefill(seq)
    assert mgr.num_free == MGR_BLOCKS, "a refused admission must leak nothing"
    assert seq.block_table == []


def test_append_slot_is_a_noop_inside_a_block(mgr):
    """The common case: 15 of every 16 decode steps allocate nothing."""
    seq = make_paged_seq(1, prompt_len=4)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = 4
    before = mgr.num_free
    for _ in range(B - 4 - 1):
        mgr.append_slot(seq)
        seq.num_cached_tokens += 1
        assert mgr.num_free == before, "grew a block while still inside one"
        assert_seq_invariant(seq, mgr)


def test_append_slot_grows_exactly_at_the_boundary(mgr):
    """A prompt of exactly B fills block 0. The next token needs block 1 -- and
    not one step earlier or later. This is the milestone's canonical off-by-one."""
    seq = make_paged_seq(1, prompt_len=B)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = B
    assert len(seq.block_table) == 1

    mgr.append_slot(seq)  # room for position B, the first of block 1
    assert len(seq.block_table) == 2
    assert_seq_invariant(seq, mgr)


def test_append_slot_does_not_grow_one_token_early(mgr):
    """A prompt of B-1 leaves one free slot. The next token must fit in it."""
    seq = make_paged_seq(1, prompt_len=B - 1)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = B - 1
    before = mgr.num_free

    mgr.append_slot(seq)  # room for position B-1, the last slot of block 0
    assert len(seq.block_table) == 1
    assert mgr.num_free == before
    assert_seq_invariant(seq, mgr)


def test_growth_token_by_token_across_several_boundaries(mgr):
    """The done-when test from the plan: grow one token at a time well past
    several block boundaries, checking the table against the allocator at every
    single step."""
    seq = make_paged_seq(1, prompt_len=5)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = 5

    for _ in range(4 * B):
        mgr.append_slot(seq)
        seq.num_cached_tokens += 1
        assert len(seq.block_table) == blocks_needed(seq.num_cached_tokens)
        assert_seq_invariant(seq, mgr)

    assert seq.num_cached_tokens == 5 + 4 * B
    assert mgr.num_free == MGR_BLOCKS - len(seq.block_table)


def test_append_slot_is_idempotent_if_the_caller_does_not_advance(mgr):
    """Calling append_slot twice without a token in between must not allocate
    twice -- it asks for capacity, it does not consume it."""
    seq = make_paged_seq(1, prompt_len=B)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = B
    mgr.append_slot(seq)
    after_first = mgr.num_free
    mgr.append_slot(seq)
    assert mgr.num_free == after_first
    assert len(seq.block_table) == 2


def test_append_slot_raises_when_the_pool_is_empty(mgr):
    seq = make_paged_seq(1, prompt_len=B)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = B
    hog = mgr.allocator.allocate(mgr.num_free)  # drain the pool
    with pytest.raises(BlockExhausted):
        mgr.append_slot(seq)
    assert len(seq.block_table) == 1, "a failed growth must leave the table intact"
    assert_seq_invariant(seq, mgr)
    mgr.allocator.free(hog)


def test_free_returns_every_block_and_resets_the_sequence(mgr):
    seq = make_paged_seq(1, prompt_len=3 * B + 1)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = 3 * B + 1
    mgr.free(seq)
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0, "freed KV means no cached tokens, even mid-flight"
    assert mgr.num_free == MGR_BLOCKS


def test_free_is_how_preemption_works_not_just_completion(mgr):
    """Phase 4 preempts a RUNNING sequence by dropping its KV and keeping its
    tokens. free() must leave the sequence re-admittable, not destroyed."""
    seq = make_paged_seq(1, prompt_len=B, max_new_tokens=32)
    mgr.allocate_for_prefill(seq)
    seq.num_cached_tokens = B
    seq.output_token_ids = [7, 7, 7]

    mgr.free(seq)
    assert seq.output_token_ids == [7, 7, 7], "tokens survive; only the KV is dropped"

    mgr.allocate_for_prefill(seq)  # re-admitted later
    assert len(seq.block_table) == 1
    assert_seq_invariant(seq, mgr)


def test_two_sequences_interleaved_never_receive_the_same_block(mgr):
    a = make_paged_seq(1, prompt_len=B + 1)
    b = make_paged_seq(2, prompt_len=B + 1)
    mgr.allocate_for_prefill(a)
    mgr.allocate_for_prefill(b)
    a.num_cached_tokens = b.num_cached_tokens = B + 1

    for _ in range(2 * B):
        for seq in (a, b):
            mgr.append_slot(seq)
            seq.num_cached_tokens += 1
        assert set(a.block_table).isdisjoint(b.block_table)
        assert_seq_invariant(a, mgr)
        assert_seq_invariant(b, mgr)


def test_freeing_one_sequence_does_not_disturb_the_other(mgr):
    a = make_paged_seq(1, prompt_len=2 * B)
    b = make_paged_seq(2, prompt_len=2 * B)
    mgr.allocate_for_prefill(a)
    mgr.allocate_for_prefill(b)
    b_blocks = list(b.block_table)

    mgr.free(a)
    assert b.block_table == b_blocks
    for blk in b_blocks:
        assert mgr.allocator.ref_count(blk) == 1
    assert mgr.num_free == MGR_BLOCKS - 2


# --- Phase 2: internal fragmentation is measured, not assumed away -----------


def test_internal_fragmentation_is_zero_on_exact_multiples(mgr):
    seqs = []
    for i in range(3):
        s = make_paged_seq(i, prompt_len=2 * B)
        mgr.allocate_for_prefill(s)
        s.num_cached_tokens = 2 * B
        seqs.append(s)
    assert internal_fragmentation(seqs) == 0


def test_internal_fragmentation_is_bounded_by_block_size_per_sequence(mgr):
    """The M3 claim: waste is bounded by BLOCK_SIZE-1 per sequence, independent
    of the workload. That is what distinguishes it from reserve-max."""
    rng = random.Random(1)
    seqs = []
    for i in range(8):
        n = rng.randint(1, 3 * B)
        s = make_paged_seq(i, prompt_len=n)
        mgr.allocate_for_prefill(s)
        s.num_cached_tokens = n
        seqs.append(s)

    frag = internal_fragmentation(seqs)
    assert 0 <= frag <= (B - 1) * len(seqs)
    assert frag == sum(blocks_needed(s.num_cached_tokens) * B - s.num_cached_tokens for s in seqs)
