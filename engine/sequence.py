from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def last_token_id(self) -> int:
        """The token to feed as input for this sequence's next forward call."""
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
