"""The paged KV cache's memory layer: the pool, and the allocator that hands out
blocks of it.

Deliberately free of any `transformers` import. Everything here is bookkeeping
over integers and one big tensor, so `tests/test_kv_cache.py` needs no model and
no GPU. The HF `Cache` glue — and the scatter/gather that actually reads and
writes the pool — lives next door in `engine/paged_cache.py` (Q4, ADR-0011).

Vocabulary, which is OSTEP's vocabulary with the nouns renamed:

    page              -> block (BLOCK_SIZE token positions)
    physical frame    -> pool[block_id]
    page table        -> Sequence.block_table (Phase 2)
    free-frame list   -> BlockAllocator's free list
"""

from collections import Counter, deque
from dataclasses import dataclass

import torch

# Locked decision 3: vLLM's default, and a tunable constant rather than a magic
# number scattered through the code. `scratch/m03_paged_kv/06_fragmentation.py`
# is the experiment behind the choice.
BLOCK_SIZE = 16


class BlockExhausted(RuntimeError):
    """Raised when the pool cannot satisfy an allocation.

    A named, catchable type on purpose. In Phase 4 the scheduler treats
    exhaustion as a *normal control-flow event* — it is the signal to stop
    admitting, or to preempt — not as a crash. A bare `IndexError` escaping
    from `.pop()` on an empty list would be indistinguishable from a genuine
    bug at the call site.
    """

class DoubleFreeError(RuntimeError):
    """Raised when freeing a block that is not currently allocated.

    Separate from `BlockExhausted` because the scheduler's response is the
    opposite: exhaustion is expected and handled, a double free is a bug in
    block-table bookkeeping and must reach a human. A block that lands in the
    free list twice gets handed to two live sequences, which then overwrite
    each other's KV -- no crash, just plausible wrong tokens.
    """


@dataclass(frozen=True)
class KVSpec:
    """The six numbers that fix the pool's shape, and the byte arithmetic on them.

    Frozen because every one of these is read-only for the pool's lifetime; a
    mutated `num_kv_heads` after allocation would silently misindex everything.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    block_size: int = BLOCK_SIZE

    @classmethod
    def from_config(cls, config, dtype: torch.dtype, block_size: int = BLOCK_SIZE) -> "KVSpec":
        """Pull the shape out of a HF model config.

        `num_key_value_heads`, never `num_attention_heads`: under GQA the cache
        stores only the KV heads. Qwen2.5-0.5B has 14 query heads and 2 KV
        heads, so confusing the two oversizes the pool by 7x.
        """
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        return cls(
            num_layers=config.num_hidden_layers,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=head_dim,
            dtype=dtype,
            block_size=block_size,
        )

    @property
    def elements_per_token(self) -> int:
        """K and V, every layer, every KV head, every dim — for one token position."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim

    @property
    def bytes_per_token(self) -> int:
        return self.elements_per_token * self.dtype.itemsize

    @property
    def bytes_per_block(self) -> int:
        return self.bytes_per_token * self.block_size


def blocks_for_bytes(budget_bytes: int, spec: KVSpec) -> int:
    """How many blocks fit in a byte budget. Sizing *aid*, not a sizing *policy*.

    Q2/ADR-0011 chose an explicit `num_blocks` over a vLLM-style VRAM fraction:
    a pool whose size depends on whatever else happened to be resident on the
    GPU at launch makes a recorded benchmark number unreproducible, and
    `docs/benchmarks/` demands provenance. Use this to pick a number, then pin
    the number.
    """
    return budget_bytes // spec.bytes_per_block


def blocks_for_gib(gib: float, spec: KVSpec) -> int:
    return blocks_for_bytes(int(gib * 1024**3), spec)


class PagedKVPool:
    """One pre-allocated tensor. Every sequence's KV lives somewhere inside it.

    Shape `[num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]`
    (locked decision 5). Two things about that ordering matter:

    - `num_blocks` is dim 0, so a block id is a plain index and `pool[block_id]`
      is one contiguous run of memory. A gather becomes `pool[block_id_tensor]`:
      a single op over large contiguous reads. Put `num_layers` first instead
      and one sequence's blocks scatter into a strided mess.
    - dim 1 is the K/V pair, so K and V for the same token sit adjacent — the
      attention path always wants both.

    Allocated exactly once, at construction. Never per-block cudaMalloc/free:
    allocator overhead on the hot path would show up inside the very benchmark
    numbers this project exists to produce.
    """

    def __init__(self, num_blocks: int, spec: KVSpec, device: str | torch.device = "cpu"):
        self.num_blocks = num_blocks
        self.spec = spec
        self.device = torch.device(device)
        # zeros, never empty. Phase 0 finding 5: finite junk in a
        # causally-masked slot is harmless, but an `inf` from uninitialised
        # memory makes `0 * inf = NaN` and poisons the whole weighted sum —
        # *without* raising. A reused block always holds some dead sequence's
        # finite KV, so zeroing once here is sufficient and `free()` never has
        # to zero anything.
        self.tensor = torch.zeros(
            (
                num_blocks,
                2,
                spec.num_layers,
                spec.num_kv_heads,
                spec.block_size,
                spec.head_dim,
            ),
            dtype=spec.dtype,
            device=self.device,
        )

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.spec.dtype.itemsize

    @property
    def capacity_tokens(self) -> int:
        """Total token positions the pool can hold, ignoring fragmentation."""
        return self.num_blocks * self.spec.block_size

    def __repr__(self) -> str:
        return (
            f"PagedKVPool(num_blocks={self.num_blocks}, block_size={self.spec.block_size}, "
            f"{self.nbytes / 1024**2:.1f} MiB, device={self.device})"
        )


class BlockAllocator:
    """Hands out and reclaims block ids from a `PagedKVPool`, in O(1) both ways.

    Required surface (the rest of the engine depends on exactly this):

        allocate(n)   -> list[int]   n fresh block ids, ref_count set to 1
        free(ids)     -> None        decrement; return to the free list at 0
        num_free      -> int         property, exact after every operation
        ref_count(id) -> int         current count for one block

    Design notes rather than code:

    - The free list is a *stack* — here a `deque` used LIFO (`append`/`pop`),
      which is O(1) both ways and hands back the most recently touched blocks,
      the ones most likely still warm in cache. A plain `list` would be the
      marginally faster stack (no block-linked structure, better locality); the
      `deque` earns its keep only if the free list ever needs FIFO reuse or
      cheap `popleft`. A `set` is O(1) too but throws the ordering away for
      nothing. See `docs/bites/`.
    - Both mutating methods are **all-or-nothing**: they validate fully before
      touching any state. A half-applied allocate leaks blocks; a half-applied
      free hands one block to two sequences.
    - `ref_count` is a plain list indexed by block id, sized `num_blocks`.
      In M3 it is always 0 or 1 (locked decision 6) — the structure exists so
      that M6's prefix sharing is an extension, not a rewrite.
    - Exhaustion raises `BlockExhausted`, and does so *before* mutating
      anything: a partially-served `allocate(n)` that then throws would leak
      blocks that no block table records, and nothing would ever free them.
    - Double-free is a bug, not a no-op. Detect it and raise loudly; a silently
      double-freed block ends up in the free list twice and gets handed to two
      live sequences at once, which is the single worst failure mode in this
      milestone and produces plausible-looking wrong tokens.
    """

    def __init__(self, num_blocks: int):
        self.free_list = deque(range(num_blocks))
        self.ref_counter = [0] * num_blocks

    def allocate(self, n: int) -> list[int]:
        # Capacity check BEFORE any mutation: a partially-served allocate that
        # then raises would leak blocks no block table records, and nothing
        # would ever free them.
        if len(self.free_list) < n:
            raise BlockExhausted(f"requested {n} blocks, only {len(self.free_list)} free")
        handout_list = []
        for _ in range(n):
            temp_id = self.free_list.pop()
            self.ref_counter[temp_id] = 1
            handout_list.append(temp_id)
        return handout_list

    def free(self, ids: list[int]) -> None:
        # Same all-or-nothing discipline as allocate(), for the same reason.
        # Validating inside the mutation loop would let free([live, bogus])
        # reclaim `live` and then raise, leaving it in the free list AND in a
        # running sequence's block table.
        #
        # Counted, not just membership-tested: `ids` may legitimately name the
        # same block twice once M6 shares blocks between sequences, and a block
        # with ref_count 2 can absorb exactly two frees, not three.
        for block_id, wanted in Counter(ids).items():
            if self.ref_counter[block_id] < wanted:
                raise DoubleFreeError(
                    f"block {block_id} freed {wanted}x but has ref_count "
                    f"{self.ref_counter[block_id]}"
                )

        for block_id in ids:
            # Decrement, not assign-zero. Identical in M3 where counts are only
            # ever 0 or 1 -- and the whole point of carrying ref counts now is
            # that M6's shared prefix blocks must survive one owner finishing.
            self.ref_counter[block_id] -= 1
            if self.ref_counter[block_id] == 0:
                self.free_list.append(block_id)

    @property
    def num_free(self) -> int:
        return len(self.free_list)

    def ref_count(self, block_id: int) -> int:
        return self.ref_counter[block_id]          

# =============================================================================
# Phase 1 — review notes
# =============================================================================
#
# 1. One pre-allocated pool, never per-block GPU allocation.
#    `block_id` is a plain index into dim 0, so `pool[block_id]` is one
#    contiguous run and "allocating a block" is an integer leaving a list.
#    cudaMalloc/cudaFree on the hot path would put the allocator's own cost
#    inside the benchmark this project exists to produce.
#
# 2. `BlockExhausted` is normal control flow, not an error.
#    Running out of blocks is the *expected* steady state of a well-utilised
#    engine -- it is the signal the scheduler admits against. Phase 4 catches
#    it; it must never reach a traceback.
#
# 3. Allocation is all-or-nothing: validate the whole request before mutating
#    anything. A partially-granted request would leak blocks on the failure
#    path and leave a half-grown sequence, and it is precisely the failure path
#    that gets least testing. This discipline is what makes Phase 2's growth
#    rule free -- `_ensure_capacity` needs no rollback of its own.
#
# 4. The free list is LIFO (a stack, not a queue).
#    A just-freed block is the one most likely still resident in cache, and
#    order is otherwise meaningless here. Free order is also *not* an ownership
#    check: ref counts are, and `DoubleFreeError` exists because a block
#    returned twice is a silent aliasing bug -- two sequences writing the same
#    physical address, producing plausible wrong tokens.

# --- Phase 2: address translation and the block table -----------------------
#
# The three functions below are the *whole* mechanism, and they are written
# once, here, rather than inlined at the admission site, the decode site and
# the gather -- where two of the three would eventually disagree.


def logical_block(pos: int, block_size: int = BLOCK_SIZE) -> int:
    """Which entry of a block table holds token position `pos`.

    The block table is a single-level page table: `logical_block` is the index
    into it, `slot` is the offset within the frame. One level, not OSTEP's
    four, because a sequence's logical address space is dense and bounded --
    positions 0..n with no holes -- rather than a sparse 2^48 in which a flat
    table would be mostly empty.
    """
    return (pos // block_size)


def slot(pos: int, block_size: int = BLOCK_SIZE) -> int:
    """Which offset *within* that block holds token position `pos`.

    Always in `[0, block_size)`, and `logical_block(p) * block_size + slot(p)`
    round-trips to `p` for every p. Phase 3's scatter writes to
    `pool[block_table[logical_block(p)], :, layer, :, slot(p), :]`.
    """
    return pos % block_size


def blocks_needed(num_tokens: int, block_size: int = BLOCK_SIZE) -> int:
    """How many blocks it takes to hold `num_tokens` token positions.

    `blocks_needed(0) == 0`: a sequence with no cached tokens owns no blocks,
    and admission must not hand it one. `blocks_needed(n) ==
    logical_block(n - 1) + 1` for every n >= 1 -- the block holding the last
    position is the last block needed. A test pins that, because the two
    pieces of arithmetic are individually plausible and jointly wrong in the
    off-by-one case.

    `(n + b - 1) // b` over `math.ceil(n / b)`: integer arithmetic throughout,
    so it cannot lose a bit to float rounding once n grows large. `-(-n // b)`
    is equally exact and equally fine; this one just reads as what it is.
    """
    return (num_tokens+block_size-1)//block_size


class BlockManager:
    """Owns the allocator and is the only thing that knows the growth rule.

    Why this class exists rather than the scheduler calling `BlockAllocator`
    directly (ADR-0011, Q4 follow-up): the growth rule -- "does this sequence
    need another block?" -- otherwise gets expressed once at admission and
    again in the decode path, and those two copies drift.

    **Capacity, not content.** Every method here is about how many blocks a
    sequence owns. None of them touches `num_cached_tokens`, because that
    counts positions whose KV has actually been *written*, and the write
    happens later, inside the forward pass. The caller advances it after a
    successful step. Consequence, and it is the invariant to hold onto:

        0 <= seq.num_cached_tokens <= len(seq.block_table) * BLOCK_SIZE

    -- capacity runs one step ahead of content, never behind.

    All three public methods are thin wrappers over `_ensure_capacity`, so the
    growth rule is written down exactly once.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int = BLOCK_SIZE):
        self.allocator = allocator
        self.block_size = block_size

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    def _ensure_capacity(self, seq, num_tokens: int) -> None:
        """Grow `seq.block_table` until it can hold `num_tokens` positions.

        The growth rule, stated once. Both callers below are this function with
        a different `num_tokens`, which is the point: expressed twice, the
        admission copy and the decode copy drift.

        Shrinking is deliberately not a case. Blocks are released only by
        `free`, all at once; a sequence never gives back a partially-filled
        tail block, because the tokens in it are still live.
        """
        want = blocks_needed(num_tokens, self.block_size)
        have = len(seq.block_table)
        if want <= have:
            return
        # allocate() is all-or-nothing, so a BlockExhausted here leaves the
        # table exactly as it was -- no half-grown sequence, nothing leaked.
        seq.block_table.extend(self.allocator.allocate(want - have))

    def allocate_for_prefill(self, seq, num_tokens: int | None = None) -> None:
        """Give `seq` enough blocks for the tokens it will prefill, at admission.

        `num_tokens` defaults to the prompt length. Phase 4 passes an explicit
        count for a sequence readmitted after **preemption by recomputation**:
        it re-prefills its prompt *plus* everything it had already generated,
        so admission must size for both. The alternative -- folding the output
        into `prompt_token_ids` on preemption -- would make
        `len(output_token_ids)` stop meaning "tokens generated", which is the
        `max_new_tokens` counter.

        Prompt only, no headroom for generation. Reserving `max_new_tokens`
        up front would be reserve-max wearing a block table -- exactly the
        M2 behaviour paging exists to kill -- at the price of never failing
        mid-generation. Paying that cost is Phase 4's decision (preempt vs.
        refuse), not admission's.

        Raises `BlockExhausted` if the pool can't cover it, which the scheduler
        reads as "don't admit this one yet", not as an error.
        """
        if num_tokens is None:
            num_tokens = len(seq.prompt_token_ids)
        self._ensure_capacity(seq, num_tokens)

    def append_slot(self, seq) -> None:
        """Make room for one more token, called before each decode step.

        `num_cached_tokens` positions are written, so the next write lands at
        position `num_cached_tokens` -- and capacity for position p means
        capacity for p+1 *tokens*. Hence `+ 1`, and hence the whole thing is
        `_ensure_capacity`, not an unconditional append.

        Usually a no-op: 15 of every 16 decode steps allocate nothing. It is
        also idempotent, because it asks for capacity rather than consuming
        it -- calling it twice without a token in between grows nothing, since
        the second call finds `want == have`.
        """
        self._ensure_capacity(seq, seq.num_cached_tokens + 1)

    def free(self, seq) -> None:
        """Return every block `seq` owns, and reset it to holding no KV.

        Called from the scheduler's eviction path when a sequence reaches
        FINISHED -- and, in Phase 4, from the preemption path for a sequence
        that is *not* finished, which is why it resets `num_cached_tokens`
        rather than assuming the sequence is done. The token ids are untouched:
        preemption-by-recomputation drops the KV and keeps the tokens, so the
        sequence stays re-admittable.

        Order matters. The table is cleared only after the free succeeds, so a
        DoubleFreeError leaves the sequence still owning what it owned instead
        of orphaning live blocks.
        """
        self.allocator.free(seq.block_table)
        seq.block_table = []
        seq.num_cached_tokens = 0


def internal_fragmentation(seqs) -> int:
    """Token-slots owned but not holding KV, summed over `seqs`.

    The entire remaining waste after paging, and bounded by `BLOCK_SIZE - 1`
    per sequence rather than by the workload -- which is the claim M3 makes and
    therefore the claim it has to measure. Report it in the Phase 6 benchmark;
    do not assume it is zero.
    """
    return sum(len(s.block_table) * BLOCK_SIZE - s.num_cached_tokens for s in seqs)


# =============================================================================
# Phase 2 — review notes
# =============================================================================
#
# 1. Capacity is not content.
#    `BlockManager` grants *room*; it never advances `num_cached_tokens`. The
#    caller does that, and only after a forward pass has actually succeeded.
#    The invariant is
#
#        0 <= num_cached_tokens <= len(block_table) * BLOCK_SIZE
#
#    and the gap between the two sides is capacity running ahead of content --
#    which is exactly what a decode step consumes. Advancing the counter inside
#    the manager would make a failed forward pass leave a sequence claiming KV
#    that was never written: silent wrong tokens on the next step.
#
# 2. `append_slot` is idempotent, with no guard needed.
#    It asks for capacity rather than consuming it, so calling it twice between
#    two forward passes is a no-op. That property falls out of (1) rather than
#    being defended by a flag, which is why it cannot rot.
#
# 3. Prefill allocates for the prompt only, with no generation headroom.
#    Reserving max_new_tokens up front would be reserve-max wearing a block
#    table -- it reintroduces exactly the reservation waste M3 exists to
#    remove, while paying the paging overhead anyway. Growth happens one block
#    at a time, on demand, in `_ensure_capacity`.
#
# 4. `_ensure_capacity` never shrinks.
#    Freeing a trailing block when a sequence's tail empties would be correct
#    and useless -- sequences only grow until they finish, and `free()` returns
#    everything at once. A shrink path would be untested code guarding an
#    impossible state.
#
# 5. `(n + b - 1) // b`, not `math.ceil(n / b)`.
#    Integer arithmetic throughout: `ceil` on a float divide is exact for the
#    magnitudes here but stops being exact somewhere above 2**53, and the
#    integer form has no such cliff to reason about.
#
# 6. `free()` clears the block table only *after* the allocator call returns.
#    If the allocator raises (a double free, say), the sequence keeps its table
#    and the bug is still diagnosable. Clearing first would destroy the
#    evidence of the thing that just went wrong.
