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

    # --- paged KV cache (M3 Phase 2). Unused by the M1/M2 engines. ---

    block_table: list[int] = field(default_factory=list)
    """Logical -> physical block mapping. Entry `i` is the physical block id
    holding token positions `[i*BLOCK_SIZE, (i+1)*BLOCK_SIZE)`, so the list
    index *is* the logical block number -- a single-level page table.

    Total capacity is `len(block_table) * BLOCK_SIZE`, which is >= the number
    of positions actually holding KV; the tail of the last block is internal
    fragmentation.
    """

    num_cached_tokens: int = 0
    """How many token positions currently hold valid KV in the pool.

    Deliberately NOT derived from `len(prompt) + len(output)`: preemption by
    recomputation (Phase 4) drops the KV while keeping the tokens, and prefix
    sharing (M6) gives a sequence KV for tokens it never computed. Those are
    the two states this field exists to express.

    Invariant: `0 <= num_cached_tokens <= len(block_table) * BLOCK_SIZE`.
    """

    @property
    def last_token_id(self) -> int:
        """The token to feed as input for this sequence's next forward call."""
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
