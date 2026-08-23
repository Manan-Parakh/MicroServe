import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.continuous_engine import ContinuousEngine
from engine.engine import Engine
from engine.model_runner import ModelRunner
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


@pytest.mark.parametrize("engine_fixture", ["continuous_engine", "static_engine"])
def test_engine_matches_golden_multi_sequence(golden_model, request, engine_fixture):
    """Both M2 engines (Phase 3), run with 2 concurrent sequences of different
    prompt lengths AND different max_new_tokens (one finishes before the
    other), must reproduce the same tokens as `.generate()` per sequence --
    the real end-to-end test that engine + scheduler/batch-loop + batched
    forward-step primitives all compose correctly, including each engine's
    different early-finish handling (eviction vs. discard-and-keep-computing)."""
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
