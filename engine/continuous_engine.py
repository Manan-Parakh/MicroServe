from engine.engine import DEFAULT_MAX_NEW_TOKENS
from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from engine.sequence import Sequence

DEFAULT_MAX_BATCH_SIZE = 8


class ContinuousEngine:
    """Thin shell over `Scheduler`: `add_request` turns a prompt into a `Sequence`
    and hands it off; `step()` just forwards to `Scheduler.step()`. All batching
    policy lives in `Scheduler`, not here — see docs/plans/m02-continuous-batching-plan.md.
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
        self.scheduler = Scheduler(max_batch_size, self.model_runner)
        self.max_new_tokens = max_new_tokens
        self._next_seq_id = 0

    def add_request(self, prompt: str, max_new_tokens: int | None = None) -> Sequence:
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=self.model_runner.encode(prompt),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        self._next_seq_id += 1
        self.scheduler.add_sequence(seq)
        return seq

    def step(self) -> list[Sequence]:
        """Advances by one iteration. Returns sequences that finished this step
        (possibly empty) -- unlike M1's `Sequence | None`, more than one sequence
        can finish in the same iteration now that many run concurrently."""
        return self.scheduler.step()

    @property
    def running(self) -> list[Sequence]:
        """Forwards to the scheduler's running batch -- lets callers (e.g. the
        benchmark) inspect current batch occupancy the same way for both
        ContinuousEngine and StaticEngine, without reaching into internals."""
        return self.scheduler.running
