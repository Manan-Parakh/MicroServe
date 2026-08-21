import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.engine import Engine
from engine.model_runner import ModelRunner

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
MAX_NEW_TOKENS = 20
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def golden_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


@pytest.fixture(scope="module")
def model_runner():
    return ModelRunner(MODEL_ID)


@pytest.fixture(scope="module")
def naive_engine():
    return Engine(MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)


def _golden_tokens(golden_model, prompt: str) -> list[int]:
    tokenizer, model = golden_model
    torch.manual_seed(SEED)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
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
