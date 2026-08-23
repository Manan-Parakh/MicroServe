from collections import deque

from engine.model_runner import ModelRunner
from engine.sequence import Sequence, SequenceStatus

DEFAULT_MAX_NEW_TOKENS = 64


class Engine:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
        attn_implementation: str | None = None,
    ):
        self.model_runner = ModelRunner(
            model_id, device=device, attn_implementation=attn_implementation
        )
        self.max_new_tokens = max_new_tokens
        self._queue: deque[Sequence] = deque()
        self._next_seq_id = 0

    def add_request(self, prompt: str, max_new_tokens: int | None = None) -> Sequence:
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=self.model_runner.encode(prompt),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        self._next_seq_id += 1
        self._queue.append(seq)
        return seq

    def step(self) -> Sequence | None:
        """Runs the next queued sequence to completion. No interleaving — the naive baseline."""
        if not self._queue:
            return None
        seq = self._queue.popleft()
        full_ids = self.model_runner.generate(seq.prompt_token_ids, seq.max_new_tokens)
        seq.output_token_ids = full_ids[len(seq.prompt_token_ids) :]
        seq.status = SequenceStatus.FINISHED
        return seq
