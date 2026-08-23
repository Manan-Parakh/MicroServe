"""StaticEngine-only tests: fake ModelRunner, no real model.

Regression coverage for the max_cache_len bug found running the M2 benchmark
(mismatched prompt lengths + mismatched max_new_tokens within one static
batch overflowed the StaticCache -- see engine/static_engine.py's
_start_batch comment for why the fix isn't simply
`max(prompt_len + max_new_tokens)` per sequence).
"""

from collections import deque

from engine.sequence import Sequence
from engine.static_engine import StaticEngine


def make_seq(seq_id: int, prompt_len: int, max_new_tokens: int) -> Sequence:
    return Sequence(
        seq_id=seq_id, prompt_token_ids=list(range(prompt_len)), max_new_tokens=max_new_tokens
    )


class FakeModelRunner:
    """Records the max_cache_len it was called with instead of running a model."""

    def __init__(self):
        self.tokenizer = type("FakeTokenizer", (), {"eos_token_id": -1})()
        self.last_max_cache_len = None

    def prefill_batch(self, token_id_lists, max_cache_len):
        self.last_max_cache_len = max_cache_len
        cache = f"cache/{len(token_id_lists)}"
        attention_mask = [1] * len(token_id_lists)
        return [0] * len(token_id_lists), cache, attention_mask

    def decode_step(self, cache, last_token_ids, attention_mask):
        return [0] * len(last_token_ids), attention_mask


def test_start_batch_sizes_cache_for_longest_prompt_plus_longest_budget():
    """The bug: a sequence with a SHORT prompt but the batch's LONGEST
    max_new_tokens still gets left-padded to the batch's longest prompt, then
    the whole batch steps for as many iterations as that longest budget
    needs -- so the cache must hold (longest prompt) + (longest budget), not
    max(each sequence's own prompt + its own budget)."""
    # Bypass __init__ (it constructs a real ModelRunner) and set up the
    # state it would have produced, with a fake model_runner instead.
    engine = StaticEngine.__new__(StaticEngine)
    engine.model_runner = FakeModelRunner()
    engine.max_new_tokens = 64
    engine.max_batch_size = 3
    engine.waiting = deque()
    engine.running = []
    engine.cache = None
    engine.attention_mask = None
    engine._next_seq_id = 0
    fake = engine.model_runner

    # Short prompt, huge budget; longer prompt, tiny budget. The old buggy
    # formula (per-sequence prompt+budget max) would size for 6+60=66; the
    # correct one needs the longer prompt (13) + the longer budget (60) = 73.
    engine.waiting.append(make_seq(seq_id=1, prompt_len=6, max_new_tokens=60))
    engine.waiting.append(make_seq(seq_id=2, prompt_len=13, max_new_tokens=8))

    engine._start_batch()

    assert fake.last_max_cache_len == 13 + 60
