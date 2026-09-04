"""The fourth engine: continuous batching over a paged KV cache.

Same shell as `ContinuousEngine` -- `add_request(prompt)` and `step()`, all
policy in the scheduler -- with one addition that is the whole milestone: this
engine has to be told how much memory its KV cache may use, because for the
first time that number, and not `max_batch_size`, is what bounds concurrency.

`max_batch_size` survives as an upper bound only. If it binds, the milestone is
un-measured: `peak_concurrent_sequences` goes back to reporting its own
configuration, which is exactly the caveat M2's benchmark README had to carry.
Set it well above what the pool can hold.
"""

from engine.engine import DEFAULT_MAX_NEW_TOKENS
from engine.kv_cache import KVSpec, PagedKVPool, blocks_for_gib
from engine.model_runner import ModelRunner
from engine.paged_scheduler import PagedScheduler
from engine.sequence import Sequence

DEFAULT_MAX_BATCH_SIZE = 64
"""Deliberately far above M2's 8. In M3 the pool is meant to be the binding
constraint; this is a guard rail against a runaway batch, not a policy."""

DEFAULT_KV_CACHE_GIB = 1.0


class PagedEngine:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        num_blocks: int | None = None,
        kv_cache_gib: float | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
    ):
        """`num_blocks` and `kv_cache_gib` are two ways to say the same thing.

        Give `num_blocks` when the pool size is the independent variable (the
        tests, and the M5 sweep that shows concurrency rising with it). Give
        `kv_cache_gib` when the hardware is the independent variable, which is
        how a real deployment sizes it. Neither: 1 GiB, which on the dev model
        is ~5,400 blocks and far more than the tiny benchmark needs.
        """
        if num_blocks is not None and kv_cache_gib is not None:
            raise ValueError("give num_blocks or kv_cache_gib, not both")

        self.model_runner = ModelRunner(
            model_id, device=device, attn_implementation=attn_implementation
        )
        model = self.model_runner.model
        spec = KVSpec.from_config(model.config, dtype=model.dtype)
        if num_blocks is None:
            num_blocks = blocks_for_gib(
                kv_cache_gib if kv_cache_gib is not None else DEFAULT_KV_CACHE_GIB, spec
            )
        self.pool = PagedKVPool(num_blocks, spec, device=self.model_runner.device)
        self.scheduler = PagedScheduler(max_batch_size, self.model_runner, self.pool)
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
        """Advance by one iteration. Returns sequences that finished this step."""
        return self.scheduler.step()

    @property
    def running(self) -> list[Sequence]:
        """Same accessor shape as ContinuousEngine and StaticEngine, so the
        benchmark can measure batch occupancy without reaching into internals."""
        return self.scheduler.running

    @property
    def num_preemptions(self) -> int:
        """Recomputation is wasted prefill compute. The M5 chart has to be able
        to separate it from the cost of paging itself."""
        return self.scheduler.num_preemptions

    @property
    def kv_utilization(self) -> float:
        """Fraction of pool blocks currently owned by a sequence.

        The number M2 measured at 33%. Not the same as *useful* utilization --
        the tail of each sequence's last block is internal fragmentation, which
        `kv_cache.internal_fragmentation` reports separately.
        """
        return self.scheduler.num_used_blocks / self.pool.num_blocks


# =============================================================================
# Milestone 3 — see it run
# =============================================================================
#
#     python -m engine.paged_engine
#
# Five scenes, each printing the thing the corresponding phase built. This is a
# demonstration, not a test -- the tests are in `tests/`, and they are what
# actually gates the milestone. This exists so the mechanism is visible rather
# than only asserted: block tables filling, physical ids that are not in order,
# a sequence being preempted and coming back with the same tokens.
#
# Runs on CPU + eager by default (ADR-0010), so scene 5's golden comparison is
# a real comparison rather than a coin flip on kernel reduction order.


def _demo() -> None:  # pragma: no cover - a demonstration, not a test
    import torch

    from engine.kv_cache import (
        BLOCK_SIZE,
        blocks_needed,
        internal_fragmentation,
        logical_block,
        slot,
    )

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    rule = "-" * 78

    def header(n: int, title: str) -> None:
        print(f"\n{rule}\n{n}. {title}\n{rule}")

    # --- Scene 1: the pool -----------------------------------------------
    header(1, "The pool (Phase 1) -- one tensor, allocated once")
    engine = PagedEngine(
        model_id,
        max_new_tokens=24,
        max_batch_size=8,
        num_blocks=48,
        device="cpu",
        attn_implementation="eager",
    )
    pool, spec = engine.pool, engine.pool.spec
    print(f"  shape          {tuple(pool.shape)}")
    print("                  [blocks, K/V, layers, kv_heads, block_size, head_dim]")
    print(f"  dtype          {spec.dtype}   block_size {spec.block_size}")
    per_token = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * spec.dtype.itemsize
    print(
        f"  bytes/token    {per_token / 1024:.0f} KiB"
        f"   ->  {pool.nbytes / 1024**2:.2f} MiB for {pool.num_blocks} blocks"
        f" = {pool.capacity_tokens} token slots"
    )
    print(f"  free blocks    {engine.scheduler.num_free_blocks}")

    # --- Scene 2: address translation ------------------------------------
    header(2, "Address translation (Phase 2) -- a single-level page table")
    print(f"  position -> logical_block, slot      (BLOCK_SIZE = {BLOCK_SIZE})")
    for pos in (0, 15, 16, 17, 33):
        print(
            f"    pos {pos:>3}  ->  block {logical_block(pos):>2}, slot {slot(pos):>2}"
            f"   | blocks_needed({pos + 1:>3}) = {blocks_needed(pos + 1)}"
        )

    # --- Scene 3: block tables as they fill -------------------------------
    header(3, "Block tables (Phases 2-3) -- physical ids are NOT in order")
    seq_a = engine.add_request("Operating systems use virtual memory because", 24)
    engine.step()  # prefill-priority: admits + prefills A only (ADR-0013)
    seq_b = engine.add_request("The capital of France is", 24)
    engine.step()  # admits + prefills B
    for _ in range(BLOCK_SIZE + 2):
        engine.step()
    for tag, s in (("A", seq_a), ("B", seq_b)):
        print(f"  seq {tag}: {s.num_cached_tokens:>3} tokens cached, block_table = {s.block_table}")
    print(f"  interleaved and disjoint: {not set(seq_a.block_table) & set(seq_b.block_table)}")
    print(
        f"  internal fragmentation: {internal_fragmentation([seq_a, seq_b])} slots "
        f"(bound is {BLOCK_SIZE - 1} per sequence = {2 * (BLOCK_SIZE - 1)} here -- "
        "the only waste paging leaves, and it does not grow with the workload)"
    )
    print(f"  KV utilization of the pool: {engine.kv_utilization:.0%}")
    while engine.running or engine.scheduler.waiting:
        engine.step()

    # --- Scene 4: preemption ---------------------------------------------
    header(4, "Preemption by recomputation (Phase 4, ADR-0012)")
    tight = PagedEngine(
        model_id,
        max_new_tokens=16,
        max_batch_size=8,
        num_blocks=2,
        device="cpu",
        attn_implementation="eager",
    )
    print(f"  pool: {tight.pool.num_blocks} blocks = {tight.pool.capacity_tokens} token slots")
    print("  two sequences, either of which fits alone -- neither pair does")
    prompts = ["The capital of France is", "Operating systems use virtual memory because"]
    victims = [tight.add_request(p) for p in prompts]
    step_no = 0
    seen = 0
    while tight.running or tight.scheduler.waiting:
        tight.step()
        step_no += 1
        if tight.num_preemptions > seen:
            seen = tight.num_preemptions
            evicted = tight.scheduler.waiting[0]
            print(
                f"    step {step_no:>3}: preempted seq {evicted.seq_id} -- "
                f"blocks freed, {len(evicted.output_token_ids)} generated tokens kept, "
                f"KV discarded"
            )
    print(f"  total preemptions: {tight.num_preemptions}")

    # --- Scene 5: the only claim that matters -----------------------------
    header(5, "Correctness (Phase 5) -- preempted output still matches HF exactly")
    runner = tight.model_runner
    for seq_obj, prompt in zip(victims, prompts):
        ids = runner.encode(prompt)
        with torch.no_grad():
            golden = runner.generate(ids, 16)[len(ids) :]
        match = seq_obj.output_token_ids == golden
        print(
            f"  seq {seq_obj.seq_id}  match={match}   {runner.decode(seq_obj.output_token_ids)!r}"
        )
        assert match, "paged output diverged from the golden reference"

    print(f"\n{rule}")
    print("  A sequence was evicted mid-generation, lost its KV, was recomputed from")
    print("  scratch, and emitted byte-identical tokens. That is what makes preemption")
    print("  a cost rather than a behaviour change -- and it is why the M5 numbers can")
    print("  be quoted at all.")
    print(rule)


if __name__ == "__main__":
    _demo()
