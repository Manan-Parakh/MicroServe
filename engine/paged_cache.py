"""The object the model is handed as `past_key_values` when running paged.

`engine/kv_cache.py` is bookkeeping over integers; this module is the only
place that actually *reads and writes the pool*, and therefore the only place
in the paged path that imports `transformers` (ADR-0011, Q4).

What a `Cache` has to be, in transformers 5.15.1
------------------------------------------------
`Cache` is nearly a pure container: `Cache.update(k, v, layer_idx, ...)`
forwards straight to `self.layers[layer_idx].update(k, v)`. So all the work
lives on a `CacheLayerMixin` subclass, and the contract is five members --
`lazy_initialization`, `update`, `get_mask_sizes`, `get_seq_length`,
`get_max_length`. Shapes crossing the boundary are
`[batch, num_kv_heads, seq_len, head_dim]`.

Two traps the Phase 0 probe found, both encoded below:

- **`get_mask_sizes` is a promise made before `update` runs.** The 4-D causal
  mask is built from its answer. Return a width that disagrees with the keys
  you later hand back and you get a shape mismatch at the attention op --
  loud, which is the good outcome.
- **Pre-create the layers.** `Cache.get_mask_sizes` short-circuits to
  `(query_length, 0)` while `self.layers` is still short, so with
  `layer_class_to_replicate` our declared width is ignored *exactly on the
  prefill call*. `PagedCache` therefore builds all N layers up front.

Why a batch descriptor
----------------------
A `StaticCache` knows its own contents. A paged cache does not: the mapping
from row to blocks lives on the `Sequence` objects the scheduler picked for
*this* step, and it changes every step. So the caller sets a `PagedBatch`
before each forward pass and every layer reads the same one. Nothing about a
batch survives the step that used it -- only the pool does.

The cost, said out loud
-----------------------
`update()` gathers this row's blocks into a contiguous scratch tensor, once
per layer, once per step. That copy is real memory traffic M2's rectangular
`StaticCache` does not pay; it is the price of the concurrency win, and
Phase 6 has to report it rather than hide it. vLLM avoids the copy with a
custom kernel that reads scattered blocks directly. Not doing that is a
locked decision, not an oversight.

And note what paging does *not* remove: the gathered buffer is still a padded
rectangle, because SDPA needs one. What goes away is *reservation* waste in
the pool -- memory held for tokens that may never exist.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from engine.kv_cache import BLOCK_SIZE, PagedKVPool, blocks_needed


class PagedBatch:
    """Which sequences are in this step, and where their KV lives.

    Built fresh for every forward pass and shared by all layers. Deliberately
    plain: the three lists are the entire per-step state, and every tensor
    below is derived from them.

    - `block_tables[r]` -- row r's logical->physical map (the `Sequence`'s own
      list, not a copy)
    - `past_lens[r]`    -- positions already holding valid KV before this pass
    - `q_lens[r]`       -- *real* new tokens in this pass, which for a padded
      prefill batch is less than the padded query width

    Rows are **right-padded** (Q2, Phase 3): row r's real tokens sit at query
    indices `[0, q_lens[r])` and its KV at pool positions `[0, past+q)`. M2's
    path left-pads instead, so that `logits[:, -1, :]` reads every row's next
    token with one slice. That trick cannot survive here -- a row's KV starts
    at logical position 0 whatever the padding, so left-padded queries and
    0-based KV would put the causal mask and the block table one against the
    other. The cost is `last_logit_index`, below; the benefit is that the
    scatter indices are the same arithmetic as everything else in M3.
    """

    def __init__(
        self,
        block_tables: list[list[int]],
        past_lens: list[int],
        q_lens: list[int],
        device: torch.device | str = "cpu",
        block_size: int = BLOCK_SIZE,
    ):
        if not (len(block_tables) == len(past_lens) == len(q_lens)):
            raise ValueError("block_tables, past_lens and q_lens must describe the same rows")
        self.block_tables = block_tables
        self.past_lens = past_lens
        self.q_lens = q_lens
        self.device = torch.device(device)
        self.block_size = block_size

        for r, (bt, past, q) in enumerate(zip(block_tables, past_lens, q_lens)):
            needed = blocks_needed(past + q, block_size)
            if needed > len(bt):
                # Capacity is the scheduler's job (BlockManager.append_slot).
                # Discovering the shortfall here means a step was built that
                # cannot be written -- fail before corrupting another row.
                raise ValueError(
                    f"row {r} needs {needed} blocks for {past + q} tokens but owns {len(bt)}"
                )

    @property
    def batch_size(self) -> int:
        return len(self.block_tables)

    @property
    def max_blocks(self) -> int:
        """Widest block table in the batch; sets the gathered rectangle's width."""
        return max((len(bt) for bt in self.block_tables), default=0)

    @property
    def kv_length(self) -> int:
        """Width of the keys `update()` will return, and so what `get_mask_sizes`
        must promise. Block-padded, per Phase 0 finding 3: the tail is junk and
        the mask covers it."""
        return self.max_blocks * self.block_size

    def block_table_tensor(self) -> torch.Tensor:
        """`[batch, max_blocks]` of physical block ids, for the gather.

        Short rows are padded by repeating their own last block id. Any valid
        id would do -- the padding is masked out of attention either way -- but
        repeating an owned block keeps the read inside memory this sequence
        already touched, and avoids inventing a sentinel id that a future
        `free()` bug could then hand to the allocator.
        """
        width = self.max_blocks
        rows = [bt + [bt[-1]] * (width - len(bt)) for bt in self.block_tables]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def attention_mask(self) -> torch.Tensor:
        """`[batch, kv_length]`, 1 where a row's KV is real.

        This is what stops row r attending to the block-padded tail *and* to
        the taller rows' extra positions. Causality alone would nearly do it --
        a query at position p never looks past p -- but HF builds the causal
        component from a single `cache_position` shared by the whole batch, and
        in a paged decode step the rows sit at different absolute positions.
        The per-row mask is what carries that difference.
        """
        valid = torch.tensor(
            [p + q for p, q in zip(self.past_lens, self.q_lens)], device=self.device
        )
        positions = torch.arange(self.kv_length, device=self.device)
        return (positions.unsqueeze(0) < valid.unsqueeze(1)).long()

    def position_ids(self, padded_q_len: int) -> torch.Tensor:
        """`[batch, padded_q_len]` absolute positions, for RoPE.

        Row r's j-th query is at absolute position `past_lens[r] + j`. Padded
        columns get a harmless in-range placeholder; they are masked out of
        attention and their logits are discarded.
        """
        offsets = torch.tensor(self.past_lens, device=self.device).unsqueeze(1)
        ids = offsets + torch.arange(padded_q_len, device=self.device).unsqueeze(0)
        return ids.clamp_(min=0)

    def last_logit_index(self) -> torch.Tensor:
        """`[batch]` query index holding each row's next-token prediction.

        The price of right-padding: `logits[:, -1, :]` is only correct for the
        longest row, so read `logits[arange(B), last_logit_index()]` instead.
        """
        return torch.tensor([q - 1 for q in self.q_lens], device=self.device)

    def __repr__(self) -> str:
        return (
            f"PagedBatch(rows={self.batch_size}, kv_length={self.kv_length}, "
            f"past={self.past_lens}, q={self.q_lens})"
        )


# --- the index math ----------------------------------------------------------
#
# Everything above and below this section is plumbing. These two functions are
# the milestone: one turns "row r's j-th new token" into a physical address,
# the other reassembles scattered blocks into the rectangle SDPA demands.
#
# Both are written with tensor ops rather than a Python loop over rows. A loop
# is correct and much easier to read, but it runs once per layer per step --
# 24x per forward pass -- and would put the allocator's own overhead inside the
# very benchmark this project exists to produce.
#
# `docs/handouts/m03-phase3-paged-cache.md` has the argument; the worked shapes
# are in `scratch/m03_paged_kv/04_block_table_and_gather.py` and
# `docs/snippets/gather_worked_examples.md`.


def scatter_index(
    batch: PagedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where each new token's K/V must be written.

    Returns four parallel 1-D `long` tensors of the same length N, where N is
    the total number of *real* new tokens in the batch (`sum(q_lens)`, not
    `batch_size * padded_q_len` -- padded query columns must not be written, or
    they would clobber a neighbour's KV or the tail of a live block):

        rows[i]   -- which batch row token i came from
        cols[i]   -- which query column of `key_states` it sits in
        blocks[i] -- the physical block id it belongs in
        slots[i]  -- the offset within that block

    So that `update()` can do, for one layer, a single vectorised write:

        pool[blocks, 0, layer, :, slots, :] = key_states[rows, :, cols, :]

    The construction is the standard ragged-batch idiom. `repeat_interleave`
    turns the per-row counts into a row label per token; subtracting each row's
    exclusive-prefix-sum from a flat `arange` restarts the column counter at 0
    on every row boundary. That gives `cols` without ever materialising the
    padded rectangle.

    Then the only line that is specific to paging:

        pos = past_lens[rows] + cols

    Row r's j-th new token lands at absolute position `past + j`, not `j` --
    using `j` would silently rewrite the first tokens of the prompt on every
    decode step, and the model would keep generating fluent, wrong text.
    Address translation from Phase 2 takes over from there.
    """
    device = batch.device
    q_lens = torch.tensor(batch.q_lens, dtype=torch.long, device=device)
    past_lens = torch.tensor(batch.past_lens, dtype=torch.long, device=device)
    total = int(q_lens.sum())

    rows = torch.repeat_interleave(
        torch.arange(batch.batch_size, dtype=torch.long, device=device), q_lens
    )
    # Exclusive prefix sum: where each row's run starts in the flat 0..N range.
    # Subtracting it makes `cols` count 0,1,2,... within each row.
    starts = torch.cumsum(q_lens, dim=0) - q_lens
    cols = torch.arange(total, dtype=torch.long, device=device) - starts.repeat_interleave(q_lens)

    positions = past_lens.repeat_interleave(q_lens) + cols

    # `block_table_tensor` pads short rows by repeating their last block id, so
    # a padded column could only be selected by an out-of-range logical block --
    # which PagedBatch.__init__ already refused to construct.
    logical = positions // batch.block_size
    slots = positions % batch.block_size
    blocks = batch.block_table_tensor()[rows, logical]

    return rows, cols, blocks, slots


def gather_kv(
    pool: torch.Tensor, block_table: torch.Tensor, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reassemble one layer's scattered blocks into a contiguous rectangle.

        pool         `[num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]`
        block_table  `[batch, max_blocks]` physical ids, in logical order
        returns      two tensors of `[batch, num_kv_heads, max_blocks*block_size, head_dim]`

    Three decisions, each of which has a wrong version that still runs:

    1. **The layer is indexed inside the same op that picks the blocks.**
       `pool[block_table][..., layer_idx, :, :, :]` gives identical numbers but
       materialises all 24 layers first and discards 23 -- 24x the memory
       traffic on the hottest path in the engine.

    2. **K and V come out of one gather, not two.** Leaving dim 1 as a slice
       reads each block's keys and values together, and they are adjacent in
       the pool by construction. Two separate gathers would walk the same block
       ids twice and touch the same cache lines twice.

    3. **permute before reshape.** After indexing, the axes are
       `[kv, batch, nblk, heads, block, dim]`: the head axis sits *between* the
       two axes we want to fold together. `reshape` walks memory row-major, so
       folding straight from there interleaves heads into time -- every value
       present, every shape checking out, and the tokens wrong. The permute
       moves heads out of the way first.

    The result is block-padded to `max_blocks * block_size` rather than trimmed
    to any row's true length (Phase 0 finding 3). Causality plus the per-row
    attention mask cover the junk tail, so trimming would buy nothing and cost
    a second copy.
    """
    # Advanced indices at dims 0 and 2 are separated by a slice, so the
    # broadcast result of `block_table` does NOT stay in place -- numpy/torch
    # move it to the front. Hence [batch, nblk, ...] leading, kv in the middle.
    gathered = pool[block_table, :, layer_idx]  # [B, nblk, 2, H, block, dim]
    gathered = gathered.permute(2, 0, 3, 1, 4, 5)  # [2, B, H, nblk, block, dim]

    batch_size, num_heads, num_blocks, block_size, head_dim = gathered.shape[1:]
    flat = gathered.reshape(2, batch_size, num_heads, num_blocks * block_size, head_dim)
    return flat[0], flat[1]


# --- the transformers glue: mine ---------------------------------------------


class PagedLayer(CacheLayerMixin):
    """One layer's view of the pool. Owns no storage of its own.

    Every `CacheLayerMixin` in `transformers` keeps `self.keys` / `self.values`
    tensors; this one keeps `None` for both, because the storage is the shared
    pool and the "cache" for a layer is just an index into it. That is the
    whole idea of paging, expressed in the framework's own vocabulary.
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, pool: PagedKVPool, layer_idx: int, batch_ref: dict):
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx
        # A one-entry dict shared with the PagedCache, so `set_batch` reaches
        # every layer without the cache having to walk them. Rebinding an
        # attribute on 24 layers per step is the kind of bookkeeping that
        # quietly goes stale on the one path that forgot to update it.
        self._batch_ref = batch_ref
        self.is_initialized = True

    @property
    def batch(self) -> PagedBatch:
        batch = self._batch_ref.get("batch")
        if batch is None:
            raise RuntimeError(
                "PagedCache.set_batch() must be called before every forward pass: "
                "a paged cache cannot know which sequences are in the step."
            )
        return batch

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # Nothing to learn: the pool's dtype and device were fixed at
        # construction, and a mismatch here is a bug worth catching loudly
        # rather than silently up-casting on the hot path.
        if key_states.dtype != self.pool.tensor.dtype:
            raise TypeError(
                f"model produced {key_states.dtype} but the pool holds "
                f"{self.pool.tensor.dtype}; size the pool with the model's dtype"
            )

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter this step's K/V into the pool, then gather the whole history back."""
        self.lazy_initialization(key_states, value_states)
        batch = self.batch

        rows, cols, blocks, slots = scatter_index(batch)
        layer = self.layer_idx
        self.pool.tensor[blocks, 0, layer, :, slots, :] = key_states[rows, :, cols, :]
        self.pool.tensor[blocks, 1, layer, :, slots, :] = value_states[rows, :, cols, :]

        return gather_kv(self.pool.tensor, batch.block_table_tensor(), layer)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """The promise. Must equal the width `update` will return, exactly."""
        return self.batch.kv_length, 0

    def get_seq_length(self) -> int:
        return max(self.batch.past_lens, default=0)

    def get_max_length(self) -> int:
        return -1  # bounded by the pool, not by this layer

    def reset(self) -> None:
        # Blocks are reclaimed through BlockManager.free(seq), never here: the
        # pool outlives any one batch, and zeroing it would wipe every other
        # running sequence's KV.
        self._batch_ref["batch"] = None


class PagedCache(Cache):
    """`past_key_values` for the paged path. A view over the pool, not a store.

    Layers are built eagerly rather than via `layer_class_to_replicate` --
    Phase 0 finding 2: `Cache.get_mask_sizes` short-circuits while `self.layers`
    is short, which would silently ignore our declared width on the prefill
    call and only the prefill call.

    One `PagedCache` can live for the whole run. `set_batch` is what changes
    per step; nothing here is per-sequence state.
    """

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self._batch_ref: dict = {"batch": None}
        layers = [PagedLayer(pool, i, self._batch_ref) for i in range(pool.spec.num_layers)]
        super().__init__(layers=layers)

    def set_batch(self, batch: PagedBatch) -> None:
        """Declare which sequences this forward pass covers. Call before every one."""
        self._batch_ref["batch"] = batch

    @property
    def batch(self) -> PagedBatch | None:
        return self._batch_ref["batch"]

    def __repr__(self) -> str:
        return f"PagedCache(layers={len(self.layers)}, batch={self.batch})"


# =============================================================================
# Phase 3 — review notes
# =============================================================================
#
# The three decisions in this file that have a wrong version which still runs.
# Each is silent: no exception, no obviously-bad output. They are collected
# here rather than left only in the docstrings because they are the things to
# re-derive when this file is next touched.
#
# 1. permute before reshape (`gather_kv`)
#    After indexing, the axes are [kv, batch, nblk, heads, block, dim] -- the
#    head axis sits *between* the two we fold together. `reshape` walks memory
#    row-major, so folding straight from there interleaves heads into time.
#    Every value is present, every shape checks out, and the tokens are wrong.
#    The permute moves heads out of the way first.
#
# 2. Advanced indices separated by a slice broadcast to the FRONT
#    `pool[block_table, :, layer_idx]` with block_table of [B, nblk] yields
#    [B, nblk, 2, H, block, dim] -- NOT [2, B, nblk, ...]. NumPy and PyTorch
#    move the broadcast dimensions to the front whenever a `slice` separates
#    two advanced indices; they stay in place only when the advanced indices
#    are adjacent. This is why the permute is (2, 0, 3, 1, 4, 5) and not
#    something more obvious, and it governs the write in `update()` equally.
#
# 3. `past_lens[r] + j`, never `j` (`scatter_index`)
#    Row r's j-th new token lands at an *absolute* position. Using `j` would
#    make every decode step rewrite the first tokens of the prompt. The model
#    would not crash; it would keep generating fluent, wrong text, and only a
#    token-level golden comparison would catch it. Hence the Tier 2 gate.
#
# And three cost decisions, which are correctness-neutral but are the reason
# this milestone produces a number worth quoting:
#
# 4. One gather for K and V, not two. Leaving dim 1 as a slice reads each
#    block's keys and values together; they are adjacent in the pool by
#    construction, so one pass touches each cache line once.
#
# 5. The layer is indexed in the *same* advanced-indexing op as the blocks.
#    `pool[block_table][..., layer_idx, :, :, :]` gives identical numbers while
#    materialising all num_layers layers and discarding all but one -- that
#    factor of 24, on the hottest path in the engine.
#
# 6. The padded rectangle is never materialised (`scatter_index`). The ragged
#    idiom means a padded query column has no index at all, which is a stronger
#    guarantee than masking one: a padded column of a short row, if it existed,
#    would write into a *neighbour's* block, not into dead space.
#
# What Phase 3 did NOT remove: padding. The gathered buffer is still a
# rectangle padded to the batch's widest row, because SDPA needs one. What
# paging removes is *reservation* waste in the pool -- memory held against
# tokens that may never exist. "Paging eliminates padding" is the natural wrong
# summary and it would make Phase 6's numbers read as an underperformance.
