"""Scheduler-only tests: fake ModelRunner, no real model — pure admit/decode/evict
bookkeeping in isolation from generation correctness, per the M2 plan's Phase 2 spec.
"""

from engine.scheduler import Scheduler
from engine.sequence import Sequence, SequenceStatus


def make_seq(seq_id: int, max_new_tokens: int = 3, prompt_len: int = 2) -> Sequence:
    return Sequence(
        seq_id=seq_id, prompt_token_ids=list(range(prompt_len)), max_new_tokens=max_new_tokens
    )


class FakeModelRunner:
    """Deterministic stand-in: always emits token 0, and counts calls so tests can
    assert *which* forward-step primitive the scheduler chose (resync vs. decode)."""

    def __init__(self):
        self.tokenizer = type("FakeTokenizer", (), {"eos_token_id": -1})()
        self.prefill_calls = 0
        self.decode_calls = 0

    def prefill_batch(self, token_id_lists, max_cache_len):
        self.prefill_calls += 1
        cache = f"cache-for-{len(token_id_lists)}-rows"
        attention_mask = [1] * len(token_id_lists)
        return [0] * len(token_id_lists), cache, attention_mask

    def decode_step(self, cache, last_token_ids, attention_mask):
        self.decode_calls += 1
        return [0] * len(last_token_ids), attention_mask


# --- _admit() ---


def test_admit_under_capacity():
    sched = Scheduler(max_batch_size=4, model_runner=FakeModelRunner())
    sched.add_sequence(make_seq(1))
    sched.add_sequence(make_seq(2))

    admitted = sched._admit()

    assert admitted == 2
    assert len(sched.waiting) == 0
    assert len(sched.running) == 2


def test_admit_respects_cap():
    sched = Scheduler(max_batch_size=3, model_runner=FakeModelRunner())
    for i in range(5):
        sched.add_sequence(make_seq(i))

    admitted = sched._admit()

    assert admitted == 3
    assert len(sched.waiting) == 2
    assert len(sched.running) == 3


def test_admit_accounts_for_existing_running():
    sched = Scheduler(max_batch_size=3, model_runner=FakeModelRunner())
    sched.running = [make_seq(100), make_seq(101)]
    for i in range(4):
        sched.add_sequence(make_seq(i))

    admitted = sched._admit()

    assert admitted == 1
    assert len(sched.running) == 3


def test_admit_fifo_order():
    sched = Scheduler(max_batch_size=2, model_runner=FakeModelRunner())
    sched.add_sequence(make_seq(1))
    sched.add_sequence(make_seq(2))
    sched.add_sequence(make_seq(3))

    sched._admit()

    assert [seq.seq_id for seq in sched.running] == [1, 2]
    assert [seq.seq_id for seq in sched.waiting] == [3]


def test_admit_sets_running_status():
    sched = Scheduler(max_batch_size=2, model_runner=FakeModelRunner())
    sched.add_sequence(make_seq(1))

    sched._admit()

    assert sched.running[0].status == SequenceStatus.RUNNING


def test_admit_zero_budget_noop():
    sched = Scheduler(max_batch_size=1, model_runner=FakeModelRunner())
    sched.running = [make_seq(1)]
    sched.add_sequence(make_seq(2))

    admitted = sched._admit()

    assert admitted == 0
    assert len(sched.waiting) == 1
    assert len(sched.running) == 1


# --- step() ---


def test_step_on_idle_scheduler_is_a_noop():
    fake = FakeModelRunner()
    sched = Scheduler(max_batch_size=2, model_runner=fake)

    finished = sched.step()

    assert finished == []
    assert fake.prefill_calls == 0
    assert fake.decode_calls == 0


def test_first_step_triggers_resync_not_decode():
    fake = FakeModelRunner()
    sched = Scheduler(max_batch_size=2, model_runner=fake)
    sched.add_sequence(make_seq(1, max_new_tokens=5))

    sched.step()

    assert fake.prefill_calls == 1
    assert fake.decode_calls == 0


def test_steady_state_uses_decode_step():
    fake = FakeModelRunner()
    sched = Scheduler(max_batch_size=2, model_runner=fake)
    sched.add_sequence(make_seq(1, max_new_tokens=5))

    sched.step()  # resync (first ever call)
    sched.step()  # membership unchanged -> cheap decode

    assert fake.prefill_calls == 1
    assert fake.decode_calls == 1


def test_new_admission_forces_resync_mid_run():
    fake = FakeModelRunner()
    sched = Scheduler(max_batch_size=2, model_runner=fake)
    sched.add_sequence(make_seq(1, max_new_tokens=5))
    sched.step()  # admits seq 1, resync

    sched.add_sequence(make_seq(2, max_new_tokens=5))
    sched.step()  # seq 2 admitted -> forces resync again, not decode

    assert fake.prefill_calls == 2
    assert fake.decode_calls == 0


def test_eviction_at_max_new_tokens_then_goes_idle():
    fake = FakeModelRunner()
    sched = Scheduler(max_batch_size=2, model_runner=fake)
    sched.add_sequence(make_seq(1, max_new_tokens=2))

    finished = sched.step()  # 1st token: not finished yet
    assert finished == []
    assert sched.running[0].status == SequenceStatus.RUNNING

    finished = sched.step()  # 2nd token: hits the cap, evicted
    assert len(finished) == 1
    assert finished[0].status == SequenceStatus.FINISHED
    assert sched.running == []

    finished = sched.step()  # nothing waiting or running -> no-op, no crash
    assert finished == []
    assert fake.prefill_calls == 1


def test_eviction_forces_resync_for_remaining_sequences():
    """The scenario ADR-0007 exists for: one sequence finishing must not let a
    plain decode_step run on the survivor with a now-stale cache."""
    fake = FakeModelRunner()
    sched = Scheduler(max_batch_size=2, model_runner=fake)
    sched.add_sequence(make_seq(1, max_new_tokens=1))  # finishes after 1 token
    sched.add_sequence(make_seq(2, max_new_tokens=5))  # keeps running

    sched.step()  # resync admits both; seq 1 hits its cap and gets evicted
    assert fake.prefill_calls == 1

    sched.step()  # seq 1's eviction must force a resync for seq 2 alone
    assert fake.prefill_calls == 2
    assert fake.decode_calls == 0
