"""06 — Why paging wins: the same arithmetic, three allocation policies.

No model needed. This is pure counting, and it is the number M3 exists to
move: KV-cache memory UTILIZATION, which converts directly into peak
concurrent sequences.

Three policies over the same batch of sequences:

  reserve-max   what M2's StaticCache does today: one rectangle,
                batch x max_over_batch(prompt + max_new_tokens).
                Wastes on two axes at once -- unused generation budget AND
                other rows' padding.
  per-seq-exact a hypothetical perfect contiguous allocator: each sequence
                gets exactly prompt + max_new_tokens. Still reserves the
                generation budget it may never use.
  paged         ceil(current_len / BLOCK_SIZE) blocks per sequence, grown on
                demand. Only waste is the tail of the last block.

Run:  uv run python scratch/m03_paged_kv/06_fragmentation.py
"""

import random

BLOCK_SIZE = 16


def analyse(seqs, label):
    """seqs: list of (prompt_len, max_new_tokens, actually_generated)."""
    used = sum(p + a for p, _, a in seqs)

    reserve_max = len(seqs) * max(p + m for p, m, _ in seqs)
    per_seq_exact = sum(p + m for p, m, _ in seqs)
    blocks = sum(-(-(p + a) // BLOCK_SIZE) for p, _, a in seqs)
    paged = blocks * BLOCK_SIZE

    print(f"--- {label} ({len(seqs)} sequences) ---")
    print(f"  tokens actually holding live KV : {used:,}")
    print()
    print("  policy         slots held    utilization   wasted")
    for name, held in (
        ("reserve-max", reserve_max),
        ("per-seq-exact", per_seq_exact),
        ("paged", paged),
    ):
        print(f"  {name:<14} {held:>10,}    {used / held:>9.1%}   {held - used:>8,}")
    print()
    print(f"  paged uses {blocks:,} blocks of {BLOCK_SIZE};"
          f" internal fragmentation = {paged - used:,} slots")
    print(f"  = {(paged - used) / len(seqs):.1f} per sequence"
          f" (bound is {BLOCK_SIZE - 1}, expectation ~{BLOCK_SIZE / 2:.0f})")
    print(f"  headroom vs reserve-max: {reserve_max / paged:.1f}x more sequences"
          " in the same VRAM")
    print()


def main():
    random.seed(0)

    print("=== A. Uniform batch: paging looks almost pointless ==================")
    analyse([(128, 128, 128)] * 8, "every sequence identical, budget fully used")
    print("  Right -- when every sequence is the same length and runs to its full")
    print("  budget, a rectangle IS the optimal shape. Paging buys nothing here.")
    print("  This is why the benchmark workload must NOT be uniform.")
    print()

    print("=== B. Skewed lengths, budgets mostly unused: the real world =========")
    seqs = []
    for _ in range(32):
        prompt = random.choice([16, 24, 40, 64, 120, 300, 800])
        budget = 256
        generated = min(budget, max(4, int(random.expovariate(1 / 60))))
        seqs.append((prompt, budget, generated))
    analyse(seqs, "ShareGPT-ish: long-tailed prompts, early EOS")
    print("  Two independent wastes stack up under reserve-max:")
    print("   1. every row is padded out to the LONGEST row's total")
    print("   2. every row reserves max_new_tokens it usually does not spend")
    print("  Paging removes both. It cannot remove the last-block tail -- but")
    print("  that tail is bounded by BLOCK_SIZE, not by the workload.")
    print()

    print("=== C. Choosing BLOCK_SIZE ==========================================")
    print("  Same batch as B, sweeping the block size:")
    print()
    print("  BLOCK_SIZE   blocks   slots held   utilization   frag/seq")
    used = sum(p + a for p, _, a in seqs)
    for bs in (1, 4, 8, 16, 32, 64, 128):
        blocks = sum(-(-(p + a) // bs) for p, _, a in seqs)
        held = blocks * bs
        print(f"  {bs:>10}   {blocks:>6}   {held:>10,}   {used / held:>10.1%}"
              f"   {(held - used) / len(seqs):>8.1f}")
    print()
    print("  Utilization alone says BLOCK_SIZE = 1. It is not free:")
    print("   - block_table length grows 16x -> more Python bookkeeping per step")
    print("   - the gather does 16x more, 16x smaller reads -> worse memory")
    print("     locality, and this path is already memory-bandwidth-bound")
    print("   - allocator churn per decode step goes from occasional to constant")
    print("  16 is vLLM's default because it sits where the utilization curve has")
    print("  flattened but a block is still a large contiguous read.")
    print("  It is a TUNABLE CONSTANT in this project, not a magic number --")
    print("  the table above is the experiment you can rerun for real later.")


if __name__ == "__main__":
    main()
