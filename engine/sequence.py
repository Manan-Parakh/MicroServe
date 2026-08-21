from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    DONE = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
