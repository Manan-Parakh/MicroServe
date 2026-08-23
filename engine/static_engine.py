from collections import deque

import torch
from transformers.cache_utils import StaticCache

from engine.continuous_engine import DEFAULT_MAX_BATCH_SIZE
from engine.engine import DEFAULT_MAX_NEW_TOKENS
from engine.model_runner import ModelRunner
from engine.sequence import Sequence, SequenceStatus


class StaticEngine:
    """Fixed-size-batch comparison baseline (ADR-0006): once a batch of up to
    `max_batch_size` sequences starts, no new sequence is admitted until every
    member of that batch has finished -- even sequences that finish early stay
    in the batch, still getting a real forward pass every step, purely because
    the batch can't shrink. That wasted compute is the failure mode continuous
    batching (Phase 2's `Scheduler`) exists to fix; Phase 4's benchmark compares
    the two directly.

    Unlike `Scheduler`, a batch here never resyncs mid-lifetime: membership is
    fixed from the moment `prefill_batch` builds the cache until the whole
    batch clears, so one `StaticCache` covers the batch's entire run.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        device: str | None = None,
        attn_implementation: str | None = None,
    ):
        self.model_runner = ModelRunner(
            model_id, device=device, attn_implementation=attn_implementation
        )
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.cache: StaticCache | None = None
        self.attention_mask: torch.Tensor | None = None
        self._next_seq_id = 0

    def add_request(self, prompt: str, max_new_tokens: int | None = None) -> Sequence:
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=self.model_runner.encode(prompt),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        self._next_seq_id += 1
        self.waiting.append(seq)
        return seq

    def _start_batch(self) -> list[int]:
        n = min(len(self.waiting), self.max_batch_size)
        self.running = [self.waiting.popleft() for _ in range(n)]
        for seq in self.running:
            seq.status = SequenceStatus.RUNNING

        # NOT max(prompt_len + max_new_tokens) per sequence: every row is
        # left-padded to the batch's LONGEST prompt before decoding starts,
        # and then every row keeps stepping in lockstep -- including
        # already-finished ones (class docstring) -- until the batch's
        # slowest member (its own longest max_new_tokens) is done. The cache
        # has to hold the longest padded prompt PLUS the longest generation
        # budget in the batch, not the largest per-sequence sum of the two.
        max_cache_len = max(len(seq.prompt_token_ids) for seq in self.running) + max(
            seq.max_new_tokens for seq in self.running
        )
        token_id_lists = [seq.prompt_token_ids for seq in self.running]
        next_token_ids, self.cache, self.attention_mask = self.model_runner.prefill_batch(
            token_id_lists, max_cache_len
        )
        return next_token_ids

    def _decode_batch(self) -> list[int]:
        last_token_ids = [seq.last_token_id for seq in self.running]
        next_token_ids, self.attention_mask = self.model_runner.decode_step(
            self.cache, last_token_ids, self.attention_mask
        )
        return next_token_ids

    def step(self) -> list[Sequence]:
        """Advances by one iteration. Returns sequences that finished this step
        (possibly empty). A finished sequence stops recording output but keeps
        occupying its batch slot -- see class docstring."""
        if not self.running:
            if not self.waiting:
                return []
            next_token_ids = self._start_batch()
        else:
            next_token_ids = self._decode_batch()

        eos_id = self.model_runner.tokenizer.eos_token_id
        newly_finished = []
        for seq, tok in zip(self.running, next_token_ids):
            if seq.is_finished:
                continue  # already done earlier in this batch's lifetime -- discard
            seq.output_token_ids.append(tok)
            if tok == eos_id or len(seq.output_token_ids) >= seq.max_new_tokens:
                seq.status = SequenceStatus.FINISHED
                newly_finished.append(seq)

        if all(seq.is_finished for seq in self.running):
            self.running = []
            self.cache = None
            self.attention_mask = None

        return newly_finished
