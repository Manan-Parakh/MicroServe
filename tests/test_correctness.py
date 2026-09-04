import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.continuous_engine import ContinuousEngine
from engine.engine import Engine
from engine.kv_cache import BLOCK_SIZE
from engine.model_runner import ModelRunner
from engine.paged_engine import PagedEngine
from engine.sequence import Sequence
from engine.static_engine import StaticEngine

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
MAX_NEW_TOKENS = 20
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]
STAGGERED_MAX_NEW_TOKENS = [8, 20]  # different lengths per request -- forces a
# real early-finish, not just a same-length batch


# The correctness gate runs on CPU + eager attention, unconditionally -- not a
# preference, a requirement. See ADR-0010: SDPA's fused kernels are not
# numerically invariant to KV-cache buffer shape or query-count, on either
# device, and GPU's cuBLAS backend additionally picks different reduction
# strategies per input shape with no way to disable that. CPU + eager is the
# one configuration (of a full CPU/GPU x sdpa/eager matrix, tested) where a
# resync-interrupted decode and an uninterrupted one provably match, for
# every prompt tested. Benchmarks stay on GPU + SDPA (the fast, real path) --
# only this gate is pinned.
GATE_KWARGS = {"device": "cpu", "attn_implementation": "eager"}


@pytest.fixture(scope="module")
def golden_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    return tokenizer, model


@pytest.fixture(scope="module")
def model_runner():
    return ModelRunner(MODEL_ID, **GATE_KWARGS)


@pytest.fixture(scope="module")
def naive_engine():
    return Engine(MODEL_ID, max_new_tokens=MAX_NEW_TOKENS, **GATE_KWARGS)


@pytest.fixture(scope="module")
def continuous_engine():
    return ContinuousEngine(
        MODEL_ID, max_new_tokens=MAX_NEW_TOKENS, max_batch_size=2, **GATE_KWARGS
    )


@pytest.fixture(scope="module")
def static_engine():
    return StaticEngine(MODEL_ID, max_new_tokens=MAX_NEW_TOKENS, max_batch_size=2, **GATE_KWARGS)


# 64 blocks x 16 tokens = 1024 token positions: comfortable for the shared
# multi-sequence cases, so the gate below tests *indexing*, not preemption.
# The preemption path has its own gate in tests/test_paged_scheduler.py.
PAGED_NUM_BLOCKS = 64


@pytest.fixture(scope="module")
def paged_engine():
    return PagedEngine(
        MODEL_ID,
        max_new_tokens=MAX_NEW_TOKENS,
        max_batch_size=8,
        num_blocks=PAGED_NUM_BLOCKS,
        **GATE_KWARGS,
    )


def _golden_tokens(golden_model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> list[int]:
    tokenizer, model = golden_model
    torch.manual_seed(SEED)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,  # true greedy argmax; see ADR-0008
            pad_token_id=tokenizer.eos_token_id,
        )
    return output_ids[0].tolist()


@pytest.mark.parametrize("prompt", PROMPTS)
def test_model_runner_matches_golden(golden_model, model_runner, prompt):
    golden = _golden_tokens(golden_model, prompt)
    torch.manual_seed(SEED)
    candidate = model_runner.generate(model_runner.encode(prompt), MAX_NEW_TOKENS)
    assert candidate == golden


@pytest.mark.parametrize("prompt", PROMPTS)
def test_naive_engine_matches_golden(golden_model, naive_engine, prompt):
    tokenizer, _ = golden_model
    golden = _golden_tokens(golden_model, prompt)
    golden_new_tokens = golden[len(tokenizer(prompt)["input_ids"]) :]

    torch.manual_seed(SEED)
    naive_engine.add_request(prompt)
    result = naive_engine.step()

    assert result.output_token_ids == golden_new_tokens


def _forward_step_generate(model_runner, prompt_token_ids_list: list[list[int]]) -> list[list[int]]:
    """Drive `prefill_batch` + repeated `decode_step` for a batch of sequences at
    once -- the M2 continuous/static engines' actual code path, unlike `generate()`.
    """
    max_cache_len = max(len(ids) for ids in prompt_token_ids_list) + MAX_NEW_TOKENS
    next_ids, cache, mask = model_runner.prefill_batch(prompt_token_ids_list, max_cache_len)
    generated = [[t] for t in next_ids]
    for _ in range(MAX_NEW_TOKENS - 1):
        last_ids = [row[-1] for row in generated]
        next_ids, mask = model_runner.decode_step(cache, last_ids, mask)
        for row, t in zip(generated, next_ids):
            row.append(t)
    return generated


@pytest.mark.parametrize("prompt", PROMPTS)
def test_forward_step_single_sequence_matches_golden(golden_model, model_runner, prompt):
    """Phase 1's forward-step primitive, one sequence at a time (no padding
    involved), must reproduce the same tokens as `.generate()` -- confirms the
    prefill/decode split and StaticCache usage are correct on their own before
    layering batching on top."""
    golden = _golden_tokens(golden_model, prompt)
    golden_new_tokens = golden[len(model_runner.encode(prompt)) :]

    generated = _forward_step_generate(model_runner, [model_runner.encode(prompt)])

    assert generated[0] == golden_new_tokens


def test_forward_step_batched_matches_golden(golden_model, model_runner):
    """All PROMPTS run through one shared batched prefill_batch/decode_step call
    (left-padded to a common length, different prompt lengths on purpose) and
    each row must still match its own independent golden reference exactly --
    this is the test that would catch a row reading another row's padding or
    KV state."""
    prompt_token_ids_list = [model_runner.encode(prompt) for prompt in PROMPTS]
    golden_new_tokens = [
        _golden_tokens(golden_model, prompt)[len(ids) :]
        for prompt, ids in zip(PROMPTS, prompt_token_ids_list)
    ]

    generated = _forward_step_generate(model_runner, prompt_token_ids_list)

    assert generated == golden_new_tokens


@pytest.mark.parametrize("engine_fixture", ["continuous_engine", "static_engine", "paged_engine"])
def test_engine_matches_golden_multi_sequence(golden_model, request, engine_fixture):
    """All three batching engines, run with 2 concurrent sequences of different
    prompt lengths AND different max_new_tokens (one finishes before the
    other), must reproduce the same tokens as `.generate()` per sequence --
    the real end-to-end test that engine + scheduler/batch-loop + batched
    forward-step primitives all compose correctly, including each engine's
    different early-finish handling (eviction vs. discard-and-keep-computing).

    `paged_engine` joins here in M3 Phase 5. It passes for a *different* reason
    than the other two -- there is no shared cache to resync, because every
    sequence owns its own block table -- which is exactly why it belongs in the
    same parametrization rather than in a test of its own."""
    engine = request.getfixturevalue(engine_fixture)
    torch.manual_seed(SEED)
    seqs = [
        engine.add_request(prompt, max_new_tokens=n)
        for prompt, n in zip(PROMPTS, STAGGERED_MAX_NEW_TOKENS)
    ]

    total_finished = 0
    while total_finished < len(seqs):
        total_finished += len(engine.step())

    for prompt, n, seq in zip(PROMPTS, STAGGERED_MAX_NEW_TOKENS, seqs):
        golden = _golden_tokens(golden_model, prompt, max_new_tokens=n)
        golden_new_tokens = golden[len(seq.prompt_token_ids) :]
        assert seq.output_token_ids == golden_new_tokens


# --- M3 Phase 5: cases designed against the block machinery -------------------
#
# The parametrized case above uses short prompts and max_new_tokens of 8 and
# 20. With BLOCK_SIZE = 16 that barely crosses one boundary -- it would pass
# while the indexing was substantially wrong. These cases are chosen against
# the mechanism: block boundaries exactly, block boundaries plus one, several
# boundaries in a row, blocks that necessarily interleave between two live
# sequences, and blocks reused after a death.
#
# All of them assert token equality with HF `generate()`. Every block-order,
# slot, mask-width and stale-KV bug ends here as a token mismatch, which is the
# only failure signal that cannot be argued with.

# Long enough to slice 3 * BLOCK_SIZE tokens out of, with room to spare -- the
# longest case below needs 48. Checked by _ids_of_length rather than trusted.
BOUNDARY_SOURCE = (
    "Operating systems use virtual memory so that each process sees a private, "
    "contiguous address space while physical memory stays fragmented and shared. "
    "The page table performs the translation on every access, and the hardware "
    "caches recent translations in a lookaside buffer so that the walk is not "
    "repeated for every load and store. Demand paging means a page is brought "
    "in only when it is first touched, which is the same idea a paged key-value "
    "cache applies to attention state rather than to program memory."
)


def _ids_of_length(model_runner, n: int) -> list[int]:
    """A real token-id sequence of exactly `n` tokens.

    Sliced at the id level rather than the string level on purpose: no string
    of characters is guaranteed to tokenize to a chosen length, and the whole
    point of these cases is to hit `n = BLOCK_SIZE` and `n = BLOCK_SIZE + 1`
    exactly.
    """
    ids = model_runner.encode(BOUNDARY_SOURCE)
    assert len(ids) >= n, "BOUNDARY_SOURCE is too short for the requested length"
    return ids[:n]


def _golden_from_ids(golden_model, token_ids: list[int], max_new_tokens: int) -> list[int]:
    """Golden reference for a prompt given as ids rather than a string."""
    tokenizer, model = golden_model
    torch.manual_seed(SEED)
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0].tolist()[len(token_ids) :]


def _add_by_ids(engine, token_ids: list[int], max_new_tokens: int):
    """Queue a request whose prompt is given as ids.

    `add_request` takes a string, and re-encoding a decoded slice does not
    round-trip reliably -- which would silently change the prompt length these
    tests are built around.
    """
    seq = Sequence(
        seq_id=engine._next_seq_id,
        prompt_token_ids=list(token_ids),
        max_new_tokens=max_new_tokens,
    )
    engine._next_seq_id += 1
    engine.scheduler.add_sequence(seq)
    return seq


def _drain(engine, limit: int = 2000) -> None:
    for _ in range(limit):
        if not engine.scheduler.waiting and not engine.running:
            return
        engine.step()
    raise AssertionError("paged engine did not drain")


@pytest.mark.parametrize("prompt_len", [BLOCK_SIZE, BLOCK_SIZE + 1, 2 * BLOCK_SIZE])
def test_paged_prompt_lengths_at_and_around_a_block_boundary(
    golden_model, model_runner, paged_engine, prompt_len
):
    """A prompt of exactly BLOCK_SIZE fills its last block with no slack, so the
    very first decode token must trigger a growth. BLOCK_SIZE + 1 puts a single
    token in a fresh block, the case where an off-by-one in `blocks_needed` or
    `logical_block` shows up. Both are invisible to a round-numbered prompt.
    """
    ids = _ids_of_length(model_runner, prompt_len)
    seq = _add_by_ids(paged_engine, ids, max_new_tokens=12)
    _drain(paged_engine)
    assert seq.output_token_ids == _golden_from_ids(golden_model, ids, 12)


def test_paged_generation_crosses_several_block_boundaries(
    golden_model, model_runner, paged_engine
):
    """A prompt three tokens short of a boundary, then 40 tokens of generation:
    the sequence grows a block roughly every sixteenth step, mid-decode, with
    real history already in the pool each time. Growth-during-decode is the
    path a single short case never reaches."""
    ids = _ids_of_length(model_runner, 2 * BLOCK_SIZE - 3)
    seq = _add_by_ids(paged_engine, ids, max_new_tokens=40)
    _drain(paged_engine)
    assert len(ids) + 40 > 4 * BLOCK_SIZE  # genuinely multi-block
    assert seq.output_token_ids == _golden_from_ids(golden_model, ids, 40)


def test_paged_concurrent_sequences_with_interleaved_blocks(
    golden_model, model_runner, paged_engine
):
    """Two sequences admitted a step apart, so their blocks necessarily
    interleave in the pool as both grow. Each must still reproduce its own
    golden output exactly.

    This is the case a contiguous cache cannot even express, and the one where
    a row reading a neighbour block would look like slightly-off text rather
    than a crash.
    """
    ids_a = _ids_of_length(model_runner, BLOCK_SIZE + 5)
    ids_b = _ids_of_length(model_runner, 2 * BLOCK_SIZE + 7)[BLOCK_SIZE:]

    seq_a = _add_by_ids(paged_engine, ids_a, max_new_tokens=30)
    paged_engine.step()  # admit + prefill A only (prefill-priority, ADR-0013)
    seq_b = _add_by_ids(paged_engine, ids_b, max_new_tokens=30)
    _drain(paged_engine)

    assert seq_a.output_token_ids == _golden_from_ids(golden_model, ids_a, 30)
    assert seq_b.output_token_ids == _golden_from_ids(golden_model, ids_b, 30)


def test_paged_blocks_interleave_while_both_sequences_are_live(model_runner, paged_engine):
    """The structural half of the test above, asserted mid-flight.

    Checked while both are running, because block tables are cleared on
    retirement -- after the fact there is nothing left to inspect. Without this
    the test above could pass while each sequence happened to own one
    contiguous run, testing nothing about interleaving.
    """
    ids_a = _ids_of_length(model_runner, BLOCK_SIZE + 5)
    ids_b = _ids_of_length(model_runner, BLOCK_SIZE + 5)

    seq_a = _add_by_ids(paged_engine, ids_a, max_new_tokens=40)
    paged_engine.step()
    seq_b = _add_by_ids(paged_engine, ids_b, max_new_tokens=40)
    for _ in range(BLOCK_SIZE + 4):
        paged_engine.step()

    assert len(seq_a.block_table) >= 2 and len(seq_b.block_table) >= 2
    # Disjoint -- the aliasing check -- and genuinely interleaved, not two runs.
    assert not set(seq_a.block_table) & set(seq_b.block_table)
    combined = sorted(seq_a.block_table + seq_b.block_table)
    a_positions = [combined.index(b) for b in sorted(seq_a.block_table)]
    assert a_positions != list(range(len(a_positions)))

    _drain(paged_engine)


def test_paged_reuses_a_dead_sequences_blocks_without_stale_kv(
    golden_model, model_runner, paged_engine
):
    """The nastiest bug this milestone can produce, and the one most likely to
    look like "the model got a bit worse" rather than like a failure.

    A sequence finishes and its blocks return to the free list still holding
    its KV -- `free()` deliberately does not zero them, because a reused block
    is always fully overwritten before it is read, and zeroing on the hot path
    would buy nothing. The next sequence is handed those same physical blocks.
    If the scatter, the block table, or `num_cached_tokens` is off by anything
    at all, the new sequence attends to a dead one keys and produces fluent,
    wrong text.

    Reuse is asserted rather than assumed -- the LIFO free list makes it
    certain today, but an allocator change could quietly break the premise and
    leave this test passing while testing nothing.
    """
    assert paged_engine.scheduler.num_free_blocks == PAGED_NUM_BLOCKS, "pool not clean"

    ids_first = _ids_of_length(model_runner, 3 * BLOCK_SIZE)
    first = _add_by_ids(paged_engine, ids_first, max_new_tokens=8)
    paged_engine.step()
    used_by_first = set(first.block_table)
    assert len(used_by_first) >= 3
    _drain(paged_engine)
    assert paged_engine.scheduler.num_free_blocks == PAGED_NUM_BLOCKS  # all returned

    ids_second = _ids_of_length(model_runner, 3 * BLOCK_SIZE)[BLOCK_SIZE // 2 :]
    second = _add_by_ids(paged_engine, ids_second, max_new_tokens=20)
    paged_engine.step()
    assert set(second.block_table) & used_by_first, "no reuse -- the premise failed"
    _drain(paged_engine)

    assert second.output_token_ids == _golden_from_ids(golden_model, ids_second, 20)
