"""Iteration-level scheduling where the binding constraint is *memory*.

M2's `Scheduler` admits on one condition: is there a slot under
`max_batch_size`. This one admits on two -- a slot, **and** enough free blocks
for the prompt -- and it has a failure mode that has never existed in this
codebase before: block availability can run out *mid-generation*. Every running
sequence is still growing, and a step can need a block that does not exist.

Deliberately a separate class from `engine/scheduler.py` rather than a
generalization of it (ADR-0013). M5's headline deliverable runs all four
engines side by side; M2's scheduler has to stay exactly as it was measured.

Three things here are worth understanding before changing anything.

**Prefill-priority, one batch kind per step.** A step is *either* a prefill of
newly admitted sequences *or* a decode of the running set, never both.
Prefill and decode as separate forward calls is a locked decision; what this
adds is that a step with admissions spends itself entirely on them, so the
already-running sequences skip a decode that iteration. That is a real cost --
it trades TPOT for TTFT, and it is head-of-line blocking by another name. vLLM
solves it with chunked prefill, which is out of scope here. Naming the trade is
the point; hiding it would make the M5 latency curves unreadable.

**No resync.** M2 rebuilds its entire `StaticCache` on every membership change
(ADR-0007), because a single shared write-position cannot track per-row
positions. A paged cache has a block table per sequence, so each row's position
is its own: one sequence finishing touches nobody else's KV. That is a genuine
secondary win of paging and `test_paged_scheduler.py` pins it rather than
assuming it -- the plan's instruction was to not assume it in either direction.

**Preemption by recomputation** (ADR-0012). On `BlockExhausted` mid-step, the
most recently admitted running sequence is evicted: its blocks are freed, it
goes back to the front of the waiting queue, and its KV is discarded. Its
tokens are not -- readmission re-prefills prompt plus everything it had
generated, and picks up exactly where it left off.
"""

from collections import deque

from engine.kv_cache import (
    BlockAllocator,
    BlockExhausted,
    BlockManager,
    PagedKVPool,
    blocks_needed,
)
from engine.model_runner import ModelRunner
from engine.paged_cache import PagedBatch, PagedCache
from engine.sequence import Sequence, SequenceStatus


def all_token_ids(seq: Sequence) -> list[int]:
    """Every token whose KV a fresh prefill of `seq` would have to compute.

    For a never-run sequence this is just the prompt. For one readmitted after
    preemption it is prompt + generated, which is precisely the recomputation
    cost ADR-0012 accepts.
    """
    return seq.prompt_token_ids + seq.output_token_ids


class PagedScheduler:
    def __init__(
        self,
        max_batch_size: int,
        model_runner: ModelRunner,
        pool: PagedKVPool,
        allocator: BlockAllocator | None = None,
    ):
        self.max_batch_size = max_batch_size
        self.model_runner = model_runner
        self.pool = pool
        self.allocator = allocator or BlockAllocator(pool.num_blocks)
        self.manager = BlockManager(self.allocator, block_size=pool.spec.block_size)
        # One cache for the whole run. It holds no per-sequence state -- only a
        # view of the pool plus whichever PagedBatch is current -- so unlike
        # M2's StaticCache it never needs rebuilding.
        self.cache = PagedCache(pool)

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        """Counted, not logged. Preemption is wasted prefill compute, so the
        M5 chart has to be able to say how much of the paged engine's cost is
        recomputation rather than paging itself."""

    # --- admission ----------------------------------------------------------

    def add_sequence(self, seq: Sequence) -> None:
        """Scheduler owns the state machine, not the caller."""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def _can_admit(self, seq: Sequence) -> bool:
        need = blocks_needed(len(all_token_ids(seq)), self.manager.block_size)
        return self.manager.num_free >= need

    def _admit(self) -> list[Sequence]:
        """Move as many WAITING sequences into `running` as slots and blocks allow.

        Strictly FIFO, and it **stops** at the first sequence that does not
        fit rather than looking past it for a smaller one. Skipping ahead would
        raise utilization slightly and starve long prompts indefinitely, which
        is the one pathology a memory-bound scheduler is most likely to hide:
        a long request that never runs never appears in the latency numbers
        either.
        """
        admitted: list[Sequence] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting[0]
            if not self._can_admit(seq):
                break
            self.waiting.popleft()
            # Cannot raise: _can_admit just checked, and nothing between the
            # two lines can consume a block.
            self.manager.allocate_for_prefill(seq, num_tokens=len(all_token_ids(seq)))
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    # --- preemption ---------------------------------------------------------

    def _preempt_newest(self) -> Sequence:
        """Evict the most recently admitted running sequence to free its blocks.

        Newest-first is what makes this terminate. The oldest sequences keep
        their blocks and keep making progress, so the pool drains steadily;
        evicting the oldest instead would repeatedly destroy the most
        recomputation work and could livelock a saturated pool.

        Requeued at the *front*, so it is the next thing readmitted once room
        appears -- it has the most sunk cost and would otherwise be overtaken
        by every request that arrived while it was running.
        """
        victim = self.running.pop()
        self.manager.free(victim)  # also resets num_cached_tokens to 0
        victim.status = SequenceStatus.WAITING
        self.waiting.appendleft(victim)
        self.num_preemptions += 1
        return victim

    def _grow_running(self) -> None:
        """Give every running sequence room for one more token, preempting as needed.

        Restarts the walk after each preemption rather than continuing it: the
        victim is gone from `self.running`, and `append_slot` is idempotent
        (it asks for capacity rather than consuming it), so re-walking the
        survivors costs nothing and mutating a list mid-iteration costs
        correctness.
        """
        while True:
            try:
                for seq in self.running:
                    self.manager.append_slot(seq)
                return
            except BlockExhausted:
                if len(self.running) <= 1:
                    # Nothing left to evict but the sequence that needs the
                    # block. Preempting it would free its blocks and readmit it
                    # into the same shortage forever, so this is a sizing error,
                    # not a scheduling one, and it should be loud.
                    raise BlockExhausted(
                        f"pool of {self.pool.num_blocks} blocks cannot hold a single "
                        f"sequence at length {self.running[0].num_cached_tokens + 1}; "
                        "no other sequence remains to preempt"
                    ) from None
                self._preempt_newest()

    # --- the step ------------------------------------------------------------

    def _prefill(self, seqs: list[Sequence]) -> None:
        """One batched prefill over the just-admitted sequences.

        `past_lens` is 0 for all of them, including readmitted ones: preemption
        discarded their KV, so recomputation genuinely starts from position 0
        even though the sequence is mid-generation.
        """
        token_lists = [all_token_ids(s) for s in seqs]
        batch = PagedBatch(
            [s.block_table for s in seqs],
            [0] * len(seqs),
            [len(t) for t in token_lists],
            device=self.model_runner.device,
            block_size=self.manager.block_size,
        )
        next_token_ids = self.model_runner.paged_prefill(self.cache, batch, token_lists)
        for tok, seq, tokens in zip(next_token_ids, seqs, token_lists):
            # Content advances only now, after the forward pass actually wrote.
            seq.num_cached_tokens = len(tokens)
            seq.output_token_ids.append(tok)

    def _decode(self) -> None:
        """One token for every running sequence."""
        self._grow_running()
        batch = PagedBatch(
            [s.block_table for s in self.running],
            [s.num_cached_tokens for s in self.running],
            [1] * len(self.running),
            device=self.model_runner.device,
            block_size=self.manager.block_size,
        )
        last_token_ids = [s.last_token_id for s in self.running]
        next_token_ids = self.model_runner.paged_decode_step(self.cache, batch, last_token_ids)
        for tok, seq in zip(next_token_ids, self.running):
            seq.num_cached_tokens += 1
            seq.output_token_ids.append(tok)

    def _retire(self) -> list[Sequence]:
        """Finish any sequence that hit EOS or its budget, and return its blocks.

        Freeing here rather than at the caller's convenience is what makes the
        next step's admission see the room. Note what does *not* happen: no
        cache rebuild, no invalidation of anyone else's KV. That is the paging
        win over ADR-0007's resync.
        """
        eos_id = self.model_runner.tokenizer.eos_token_id
        survivors, finished = [], []
        for seq in self.running:
            hit_eos = seq.output_token_ids and seq.output_token_ids[-1] == eos_id
            hit_length_cap = len(seq.output_token_ids) >= seq.max_new_tokens
            if hit_eos or hit_length_cap:
                seq.status = SequenceStatus.FINISHED
                self.manager.free(seq)
                finished.append(seq)
            else:
                survivors.append(seq)
        self.running = survivors
        return finished

    def step(self) -> list[Sequence]:
        """Advance by one iteration. Returns sequences that finished this step.

        Prefill-priority: a step that admits anything spends itself on their
        prefill, and the already-running sequences wait an iteration. See the
        module docstring for why that trade is stated rather than smoothed
        over.
        """
        if not self.waiting and not self.running:
            return []
        admitted = self._admit()
        if admitted:
            self._prefill(admitted)
        elif self.running:
            self._decode()
        else:
            # Everything is waiting and nothing fits. Only reachable if the
            # pool cannot hold the head of the queue with nothing running,
            # which _grow_running's guard would also catch -- but the queue
            # would otherwise spin forever returning [].
            head = self.waiting[0]
            raise BlockExhausted(
                f"pool of {self.pool.num_blocks} blocks cannot admit sequence "
                f"{head.seq_id} ({len(all_token_ids(head))} tokens) with an empty pool"
            )
        return self._retire()

    # --- introspection for the benchmark -------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return self.manager.num_free

    @property
    def num_used_blocks(self) -> int:
        return self.pool.num_blocks - self.manager.num_free


# =============================================================================
# Phase 4 — review notes
# =============================================================================
#
# The four things here that are policy rather than mechanism, and what each
# one would break if flipped. Full argument in ADR-0012 (preemption) and
# ADR-0013 (structure and step shape).
#
# 1. Newest-first eviction is what makes preemption terminate.
#    Not a fairness choice. The oldest sequences keep their blocks and keep
#    making progress, so the pool drains steadily. Evicting the oldest instead
#    would repeatedly destroy the most accumulated work and can livelock a
#    saturated pool: every victim is the one closest to freeing its blocks.
#    The complementary half is requeuing at the *front* -- the victim has the
#    most sunk cost and must not be overtaken by every request that arrived
#    while it was running.
#
# 2. Recomputation, not swapping (ADR-0012).
#    The two trade the same currency in opposite directions: recomputation
#    spends compute to avoid moving bytes; swapping spends PCIe bandwidth and
#    a pile of new machinery to avoid recomputing. On a laptop the compute is
#    the abundant side -- prefill is the one high-arithmetic-intensity phase of
#    this workload -- and the host link is the scarce one. Swapping would also
#    add a host pool, a host allocator, a GPU-to-host block map, and a transfer
#    in the step() critical path: the least-tested code in the engine holding
#    the most invariants. Revisit for long-context workloads or a fast
#    interconnect, not before.
#
# 3. Preemption keeps the tokens and drops only the KV.
#    Folding output into `prompt_token_ids` on preemption would be simpler by
#    one line and would make `len(output_token_ids)` stop meaning "tokens
#    generated" -- which is the `max_new_tokens` budget. The sequence would
#    silently be granted a fresh budget every time it was preempted.
#
# 4. Prefill-priority costs TPOT to buy TTFT (ADR-0013).
#    A step that admits spends itself on that prefill, so running sequences
#    skip an iteration. Head-of-line blocking by another name, and under M5's
#    Poisson arrivals a burst will show as a TPOT spike. Chunked prefill is the
#    real fix and is deliberately out of scope. Say this in the M5 writeup
#    rather than filing it as noise.
#
# And one property that is mechanism, not policy, and is easy to lose:
#
# 5. No resync (ADR-0007's most expensive operation, gone).
#    M2 rebuilds its whole StaticCache whenever membership changes, because one
#    shared write-position cannot track per-row positions. Here every sequence
#    owns a block table, so a sequence finishing touches nobody else's KV. This
#    is a *separable* win of M3 -- report it apart from the paging number, or
#    the two get conflated.
