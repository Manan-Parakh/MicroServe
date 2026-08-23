from collections import deque

import torch
from transformers.cache_utils import StaticCache

from engine.model_runner import ModelRunner
from engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, max_batch_size: int, model_runner: ModelRunner):
        self.max_batch_size = max_batch_size
        self.model_runner = model_runner
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.cache: StaticCache | None = None  # None == never built yet, or invalidated
        self.attention_mask: torch.Tensor | None = None
        self.membership_changed: bool = False  # did the last eviction leave `cache` stale?

    def add_sequence(self, seq: Sequence) -> None:
        """Scheduler owns the state machine, not the caller."""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def _admit(self) -> int:
        """Pull WAITING sequences into `running`, up to the batch cap. Returns count admitted."""
        budget = min(len(self.waiting), self.max_batch_size - len(self.running))
        for _ in range(budget):
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        return budget

    def _prefill_or_decode(self) -> list[int]:
        """Resync (ADR-0007) if membership changed this step or since the last
        call; otherwise one cheap decode step reusing the existing cache."""
        admitted = self._admit()
        if admitted > 0 or self.membership_changed or self.cache is None:
            max_cache_len = max(
                len(seq.prompt_token_ids) + seq.max_new_tokens for seq in self.running
            )
            token_id_lists = [seq.prompt_token_ids + seq.output_token_ids for seq in self.running]
            next_token_ids, self.cache, self.attention_mask = self.model_runner.prefill_batch(
                token_id_lists, max_cache_len
            )
            self.membership_changed = False
        else:
            last_token_ids = [seq.last_token_id for seq in self.running]
            next_token_ids, self.attention_mask = self.model_runner.decode_step(
                self.cache, last_token_ids, self.attention_mask
            )
        return next_token_ids

    def _append_tokens(self) -> None:
        next_token_ids = self._prefill_or_decode()
        for tok, seq in zip(next_token_ids, self.running):
            seq.output_token_ids.append(tok)

    def _check_evictions(self) -> list[Sequence]:
        """Evict any sequence that hit EOS or its max_new_tokens budget."""
        eos_id = self.model_runner.tokenizer.eos_token_id
        survivors = []
        finished = []
        for seq in self.running:
            hit_eos = seq.output_token_ids and seq.output_token_ids[-1] == eos_id
            hit_length_cap = len(seq.output_token_ids) >= seq.max_new_tokens
            if hit_eos or hit_length_cap:
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)
                self.membership_changed = True
            else:
                survivors.append(seq)
        self.running = survivors
        return finished

    def step(self) -> list[Sequence]:
        """Advance by one iteration. Returns sequences that finished this step."""
        if not self.waiting and not self.running:
            return []
        self._append_tokens()
        return self._check_evictions()
