"""Phase 4: admission gated by memory, and preemption when it runs out.

Two tiers, same split as `test_paged_cache.py`.

**Tier 1 uses a fake runner and a deliberately tiny pool**, so the exhaustion
and preemption paths execute on *every* run rather than only when somebody
happens to configure a small pool. That is the whole reason for the fake: a
preemption path that is only reachable under memory pressure is a preemption
path that is never tested, and it is the one piece of Phase 4 that has no
equivalent anywhere earlier in this codebase.

The fake runner does no attention at all -- it returns a deterministic token
per row. Every assertion here is about *bookkeeping*: who is running, who owns
which blocks, whether a preempted sequence resumes where it left off. The
numerical question ("do paged tokens match HF?") was settled in Phase 3 and is
re-asked at the end of this file with the real model.
"""

import pytest

from engine.kv_cache import (
    BLOCK_SIZE,
    BlockAllocator,
    BlockExhausted,
    KVSpec,
    PagedKVPool,
    blocks_needed,
)
from engine.paged_scheduler import PagedScheduler, all_token_ids
from engine.sequence import Sequence, SequenceStatus

# Non-square on purpose, and tiny: a transpose cannot typecheck past a shape
# assertion, and the pool is small enough that exhaustion is the normal case.
TINY = KVSpec(num_layers=2, num_kv_heads=2, head_dim=4, dtype=None, block_size=BLOCK_SIZE)

EOS = 999
NEXT_TOKEN = 7


class FakeTokenizer:
    eos_token_id = EOS


class FakeRunner:
    """Records what it was asked to do; computes nothing.

    `paged_prefill` and `paged_decode_step` have the same signatures as the
    real `ModelRunner`, and both are handed a real `PagedBatch` -- so its
    capacity validation still runs, and a scheduler that builds an impossible
    step still fails here rather than silently.
    """

    device = "cpu"

    def __init__(self, eos_after: dict[int, int] | None = None):
        self.tokenizer = FakeTokenizer()
        self.prefill_calls: list[list[list[int]]] = []
        self.decode_calls: list[list[int]] = []
        # seq_id -> number of tokens after which this sequence emits EOS
        self.eos_after = eos_after or {}
        self._batch_rows: list[Sequence] = []

    def _tokens_for(self, seqs, lengths):
        out = []
        for seq, n in zip(seqs, lengths):
            limit = self.eos_after.get(seq.seq_id)
            out.append(EOS if limit is not None and n >= limit else NEXT_TOKEN)
        return out

    def paged_prefill(self, cache, batch, token_id_lists):
        self.prefill_calls.append([list(t) for t in token_id_lists])
        return self._tokens_for(self._batch_rows, [len(t) + 1 for t in token_id_lists])

    def paged_decode_step(self, cache, batch, last_token_ids):
        self.decode_calls.append(list(last_token_ids))
        return self._tokens_for(self._batch_rows, [p + 1 for p in batch.past_lens])


class RecordingScheduler(PagedScheduler):
    """Tells the fake runner which sequences are in the batch.

    The real runner never needs this -- it reads everything from the
    `PagedBatch` -- but the fake needs seq ids to decide who emits EOS.
    """

    def _prefill(self, seqs):
        self.model_runner._batch_rows = seqs
        super()._prefill(seqs)

    def _decode(self):
        self.model_runner._batch_rows = list(self.running)
        super()._decode()


def make_scheduler(num_blocks: int, max_batch_size: int = 64, eos_after=None):
    pool = PagedKVPool.__new__(PagedKVPool)  # no tensor: nothing here touches storage
    pool.num_blocks = num_blocks
    pool.spec = TINY
    runner = FakeRunner(eos_after=eos_after)
    sched = RecordingScheduler(max_batch_size, runner, pool, allocator=BlockAllocator(num_blocks))
    sched.cache = None  # the fake runner ignores it
    return sched


def seq(seq_id: int, prompt_len: int, max_new_tokens: int = 4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(100, 100 + prompt_len)),
        max_new_tokens=max_new_tokens,
    )


def run_to_completion(sched, limit: int = 500) -> list[Sequence]:
    finished = []
    for _ in range(limit):
        if not sched.waiting and not sched.running:
            break
        finished.extend(sched.step())
    else:
        raise AssertionError("scheduler did not drain within the step limit")
    return finished


# --- admission ----------------------------------------------------------------


def test_admission_needs_blocks_not_just_a_slot():
    """The whole difference from M2 in one assertion: a free slot is no longer
    sufficient. Two 32-token prompts need 2 blocks each; the pool holds 3."""
    sched = make_scheduler(num_blocks=3)
    sched.add_sequence(seq(0, prompt_len=32))
    sched.add_sequence(seq(1, prompt_len=32))
    sched.step()
    assert [s.seq_id for s in sched.running] == [0]
    assert [s.seq_id for s in sched.waiting] == [1]


def test_admission_is_fifo_and_does_not_skip_ahead():
    """A long prompt that does not fit must block the queue rather than being
    passed over. Skipping ahead raises utilization and starves it forever --
    and a request that never runs never shows up in the latency numbers."""
    sched = make_scheduler(num_blocks=3)
    sched.add_sequence(seq(0, prompt_len=48))  # 3 blocks: fits alone
    sched.add_sequence(seq(1, prompt_len=1))  # 1 block: would fit if skipped to
    sched.step()
    assert [s.seq_id for s in sched.running] == [0]
    assert [s.seq_id for s in sched.waiting] == [1]


def test_admission_allocates_exactly_the_blocks_the_prompt_needs():
    sched = make_scheduler(num_blocks=8)
    s = seq(0, prompt_len=33)  # 3 blocks: 16 + 16 + 1
    sched.add_sequence(s)
    sched.step()
    assert len(s.block_table) == blocks_needed(33, BLOCK_SIZE) == 3
    assert sched.num_free_blocks == 5


def test_admitted_sequences_own_disjoint_blocks():
    """The failure this guards is the worst one in the milestone: two live
    sequences writing the same physical address, producing plausible wrong
    tokens rather than a crash."""
    sched = make_scheduler(num_blocks=16)
    for i in range(4):
        sched.add_sequence(seq(i, prompt_len=20))
    sched.step()
    owned = [b for s in sched.running for b in s.block_table]
    assert len(owned) == len(set(owned))


# --- the step: prefill priority ------------------------------------------------


def test_a_step_with_admissions_is_a_prefill_step_only():
    """Prefill and decode are separate forward calls (locked decision), and a
    step that admits spends itself on the prefill. The running sequences skip
    an iteration -- TPOT traded for TTFT, stated rather than smoothed over."""
    sched = make_scheduler(num_blocks=32)
    sched.add_sequence(seq(0, prompt_len=8))
    sched.step()
    assert len(sched.model_runner.prefill_calls) == 1
    assert sched.model_runner.decode_calls == []

    sched.add_sequence(seq(1, prompt_len=8))
    sched.step()
    assert len(sched.model_runner.prefill_calls) == 2
    assert sched.model_runner.decode_calls == []  # seq 0 did NOT decode this step


def test_prefill_advances_cached_tokens_to_the_full_prompt():
    sched = make_scheduler(num_blocks=32)
    s = seq(0, prompt_len=20)
    sched.add_sequence(s)
    sched.step()
    assert s.num_cached_tokens == 20
    assert len(s.output_token_ids) == 1


def test_decode_advances_one_position_per_step():
    sched = make_scheduler(num_blocks=32)
    s = seq(0, prompt_len=20, max_new_tokens=10)
    sched.add_sequence(s)
    sched.step()  # prefill
    for expected in (21, 22, 23):
        sched.step()
        assert s.num_cached_tokens == expected


def test_capacity_never_lags_content():
    """The Phase 2 invariant, asserted continuously through a real run rather
    than at one moment: capacity runs ahead of content, never behind."""
    sched = make_scheduler(num_blocks=32)
    s = seq(0, prompt_len=30, max_new_tokens=20)
    sched.add_sequence(s)
    while not s.is_finished:
        sched.step()
        assert 0 <= s.num_cached_tokens <= len(s.block_table) * BLOCK_SIZE


def test_growth_allocates_only_at_a_block_boundary():
    """15 of every 16 decode steps allocate nothing. If this fails, the growth
    rule has become an unconditional append and the pool drains 16x too fast."""
    sched = make_scheduler(num_blocks=32)
    s = seq(0, prompt_len=16, max_new_tokens=20)  # exactly full: next token grows
    sched.add_sequence(s)
    sched.step()
    assert len(s.block_table) == 1
    sched.step()  # position 16 -> needs block 2
    assert len(s.block_table) == 2
    for _ in range(5):
        sched.step()
        assert len(s.block_table) == 2


# --- retirement, and the resync that isn't -------------------------------------


def test_finishing_returns_blocks_to_the_pool():
    sched = make_scheduler(num_blocks=32)
    s = seq(0, prompt_len=20, max_new_tokens=3)
    sched.add_sequence(s)
    run_to_completion(sched)
    assert s.status == SequenceStatus.FINISHED
    assert s.block_table == []
    assert sched.num_free_blocks == 32


def test_one_sequence_finishing_does_not_disturb_another():
    """The paging win over ADR-0007's resync, pinned rather than assumed.

    M2 rebuilds its entire StaticCache when membership changes, because one
    shared write-position cannot track per-row positions. Here each sequence
    owns its own block table, so an eviction must leave every other sequence's
    blocks and cached-token count exactly as they were -- and no prefill call
    may be issued on their behalf.
    """
    sched = make_scheduler(num_blocks=32, eos_after={1: 22})
    long_seq = seq(0, prompt_len=20, max_new_tokens=30)
    short_seq = seq(1, prompt_len=20, max_new_tokens=30)
    sched.add_sequence(long_seq)
    sched.add_sequence(short_seq)
    sched.step()  # admit + prefill seq 0
    sched.step()  # admit + prefill seq 1

    prefills_before = len(sched.model_runner.prefill_calls)
    blocks_before = list(long_seq.block_table)
    while short_seq.status != SequenceStatus.FINISHED:
        sched.step()

    assert long_seq.block_table[: len(blocks_before)] == blocks_before
    assert len(sched.model_runner.prefill_calls) == prefills_before  # no resync


# --- preemption by recomputation (ADR-0012) ------------------------------------


def test_exhaustion_mid_generation_preempts_the_newest():
    """The failure mode that has never existed in this codebase before.

    Both sequences are admitted legally, then both grow until the pool cannot
    serve the next block. The newest is evicted so the oldest keeps making
    progress -- that ordering is what makes the scheme terminate.
    """
    sched = make_scheduler(num_blocks=4)
    a = seq(0, prompt_len=16, max_new_tokens=40)
    b = seq(1, prompt_len=16, max_new_tokens=40)
    sched.add_sequence(a)
    sched.add_sequence(b)
    sched.step()  # admit + prefill a  (1 block)
    sched.step()  # admit + prefill b  (1 block)
    for _ in range(40):
        if sched.num_preemptions:
            break
        sched.step()
    assert sched.num_preemptions >= 1
    assert b.status == SequenceStatus.WAITING
    assert a.status == SequenceStatus.RUNNING


def test_preemption_frees_the_victims_blocks_and_keeps_its_tokens():
    """Recomputation, precisely: the KV goes, the tokens stay. A sequence that
    lost its output tokens too would restart generation, and `max_new_tokens`
    would stop being a budget."""
    sched = make_scheduler(num_blocks=4)
    a = seq(0, prompt_len=16, max_new_tokens=40)
    b = seq(1, prompt_len=16, max_new_tokens=40)
    sched.add_sequence(a)
    sched.add_sequence(b)
    sched.step()
    sched.step()
    for _ in range(40):
        if sched.num_preemptions:
            break
        generated_before = len(b.output_token_ids)
        sched.step()

    assert b.block_table == []
    assert b.num_cached_tokens == 0
    assert len(b.output_token_ids) >= generated_before  # tokens survived


def test_a_preempted_sequence_is_requeued_at_the_front():
    """It has the most sunk cost. FIFO from the back would let every request
    that arrived while it ran overtake it."""
    sched = make_scheduler(num_blocks=4)
    for i in range(2):
        sched.add_sequence(seq(i, prompt_len=16, max_new_tokens=40))
    sched.step()
    sched.step()
    sched.add_sequence(seq(2, prompt_len=16, max_new_tokens=40))
    for _ in range(40):
        if sched.num_preemptions:
            break
        sched.step()
    assert sched.waiting[0].seq_id == 1  # the victim, ahead of the newcomer


def test_recomputation_reprefills_prompt_plus_generated():
    """The cost ADR-0012 accepts, made visible: the readmission prefill is over
    more tokens than the original one, and that difference is the wasted work."""
    # 16 + 40 tokens is exactly 4 blocks, so one sequence fits the pool alone
    # and the deadlock guard stays out of the way -- but two do not, so the
    # preemption path is forced.
    sched = make_scheduler(num_blocks=4)
    a = seq(0, prompt_len=16, max_new_tokens=40)
    b = seq(1, prompt_len=16, max_new_tokens=40)
    sched.add_sequence(a)
    sched.add_sequence(b)
    run_to_completion(sched)

    assert sched.num_preemptions >= 1
    prefills_for_b = [c for c in sched.model_runner.prefill_calls if len(c) == 1]
    lengths = [len(c[0]) for c in prefills_for_b]
    assert max(lengths) > 16  # at least one prefill was a recomputation


def test_a_workload_larger_than_the_pool_still_drains():
    """Phase 4's done-when. Eight sequences, a pool that can hold two of them
    at full length: every one must finish, none may be lost, and the pool must
    come back empty."""
    sched = make_scheduler(num_blocks=6)
    for i in range(8):
        sched.add_sequence(seq(i, prompt_len=16, max_new_tokens=20))
    finished = run_to_completion(sched)

    assert sorted(s.seq_id for s in finished) == list(range(8))
    assert all(len(s.output_token_ids) == 20 for s in finished)
    assert sched.num_free_blocks == 6


def test_a_pool_too_small_for_one_sequence_fails_loudly():
    """The deadlock guard. With nothing left to preempt but the sequence that
    needs the block, preempting it would readmit it into the same shortage
    forever. That is a sizing error, not a scheduling one, so it raises."""
    sched = make_scheduler(num_blocks=1)
    sched.add_sequence(seq(0, prompt_len=16, max_new_tokens=40))
    sched.step()  # prefill fills the single block exactly
    with pytest.raises(BlockExhausted, match="cannot hold a single sequence"):
        sched.step()


def test_a_prompt_larger_than_the_pool_fails_loudly():
    sched = make_scheduler(num_blocks=2)
    sched.add_sequence(seq(0, prompt_len=100))
    with pytest.raises(BlockExhausted, match="cannot admit sequence"):
        sched.step()


def test_all_token_ids_is_what_a_fresh_prefill_would_cost():
    s = seq(0, prompt_len=5)
    assert all_token_ids(s) == s.prompt_token_ids
    s.output_token_ids = [1, 2, 3]
    assert all_token_ids(s) == s.prompt_token_ids + [1, 2, 3]


# --- Tier 2: the real gate -----------------------------------------------------
#
# CPU + eager, per ADR-0010, same as every other correctness gate in this repo.

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GATE_KWARGS = {"device": "cpu", "attn_implementation": "eager"}


@pytest.fixture(scope="module")
def engine_factory():
    from engine.paged_engine import PagedEngine

    def _make(num_blocks: int, max_new_tokens: int = 12):
        return PagedEngine(
            MODEL_ID,
            max_new_tokens=max_new_tokens,
            num_blocks=num_blocks,
            **GATE_KWARGS,
        )

    return _make


def drain(engine, limit: int = 400) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for _ in range(limit):
        if not engine.scheduler.waiting and not engine.running:
            break
        for s in engine.step():
            out[s.seq_id] = list(s.output_token_ids)
    else:
        raise AssertionError("engine did not drain")
    return out


def test_paged_engine_matches_hf_generate(engine_factory):
    """One sequence through the whole stack -- engine, scheduler, manager,
    pool, cache -- must emit exactly what HF `generate()` does."""
    engine = engine_factory(num_blocks=64)
    prompt = "The capital of France is"
    s = engine.add_request(prompt)
    got = drain(engine)[s.seq_id]

    ids = engine.model_runner.encode(prompt)
    golden = engine.model_runner.generate(ids, 12)[len(ids) :]
    assert got == golden


def test_preemption_does_not_change_the_tokens(engine_factory):
    """The claim recomputation rests on: a preempted-and-recomputed sequence
    produces the *same* tokens as one that ran straight through. If this fails,
    preemption is not transparent and every M5 number is suspect.

    Two sequences share a pool sized so the second gets evicted mid-decode.
    """
    max_new = 16
    reference = engine_factory(num_blocks=64, max_new_tokens=max_new)
    prompts = [
        "The capital of France is",
        "Operating systems use virtual memory because",
    ]
    goldens = []
    for p in prompts:
        ids = reference.model_runner.encode(p)
        goldens.append(reference.model_runner.generate(ids, max_new)[len(ids) :])

    # Two blocks = 32 token positions. Either sequence fits alone (~22 and ~24
    # tokens), neither pair does, so exactly the mid-generation shortage this
    # test exists for.
    engine = engine_factory(num_blocks=2, max_new_tokens=max_new)
    seqs = [engine.add_request(p) for p in prompts]
    results = drain(engine)

    assert engine.num_preemptions >= 1
    for s, golden in zip(seqs, goldens):
        assert results[s.seq_id] == golden
