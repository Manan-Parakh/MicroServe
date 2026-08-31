"""04 — Block table, address translation, and the gather.

This is the heart of M3. Per claude.md the real implementation is YOURS to
write, so this file deliberately stops short: it shows the *shapes* and the
tensor vocabulary on toy data with marker values you can verify by eye, and
leaves the composition as exercises at the bottom.

Mental model (it is not an analogy -- it is the same data structure):

    OS                          MicroServe
    ---------------------------------------------------
    virtual address        ->   token position in a sequence
    page                   ->   block (BLOCK_SIZE tokens)
    physical frame         ->   pool[block_id]
    page table             ->   Sequence.block_table (list[int])
    free frame list        ->   BlockAllocator's free list
    MMU does translation   ->   we do it in Python, then COPY,
                                because SDPA needs a contiguous rectangle

Run:  uv run python scratch/m03_paged_kv/04_block_table_and_gather.py
"""

import torch

BLOCK_SIZE = 4  # toy value; the real one is 16


def main():
    torch.set_printoptions(linewidth=140, precision=1, sci_mode=False)

    # ---------------------------------------------------------------- part 1
    print("=== 1. Address translation ==========================================")
    # A block table is just this: logical block index -> physical block id.
    # Note the ids are NOT contiguous and NOT sorted. That is the entire point.
    block_table = [7, 2, 9]
    seq_len = 10
    print(f"  BLOCK_SIZE  = {BLOCK_SIZE}")
    print(f"  block_table = {block_table}   (logical block i lives in pool[block_table[i]])")
    print(f"  seq_len     = {seq_len}")
    print()
    print("  formula:  logical_block = pos // BLOCK_SIZE")
    print("            slot          = pos %  BLOCK_SIZE")
    print("            physical      = block_table[logical_block]")
    print()
    print("   pos | logical | slot | physical block")
    print("  -----+---------+------+---------------")
    for pos in range(seq_len):
        lb, slot = pos // BLOCK_SIZE, pos % BLOCK_SIZE
        marker = "   <- new block starts here" if slot == 0 else ""
        print(f"   {pos:3} | {lb:7} | {slot:4} | {block_table[lb]:>7}{marker}")
    print()
    n_blocks_needed = -(-seq_len // BLOCK_SIZE)  # ceil div
    capacity = n_blocks_needed * BLOCK_SIZE
    print(f"  {seq_len} tokens needs ceil({seq_len}/{BLOCK_SIZE}) = {n_blocks_needed} blocks"
          f" = {capacity} slots")
    print(f"  -> {capacity - seq_len} slots wasted in the last block. That is INTERNAL")
    print(f"     FRAGMENTATION: bounded by BLOCK_SIZE-1 per sequence, ~{BLOCK_SIZE // 2}")
    print("     on average. Compare to reserving max_new_tokens up front, which")
    print("     wastes however much the sequence did not use. See 06.")
    print()

    # ---------------------------------------------------------------- part 2
    print("=== 2. A pool with marker values ====================================")
    # Marker scheme: the value stored is  block_id * 100 + slot,  so you can
    # read any gathered tensor and immediately see where it came from.
    num_blocks, num_layers, num_kv_heads, head_dim = 12, 2, 2, 3
    pool = torch.zeros((num_blocks, 2, num_layers, num_kv_heads, BLOCK_SIZE, head_dim))
    for b in range(num_blocks):
        for s in range(BLOCK_SIZE):
            pool[b, :, :, :, s, :] = b * 100 + s
    print(f"  pool {tuple(pool.shape)}, filled so that value = block_id*100 + slot")
    print(f"  e.g. pool[7, K=0, layer=0, head=0] slots 0..3 ="
          f" {pool[7, 0, 0, 0, :, 0].tolist()}")
    print()

    # ---------------------------------------------------------------- part 3
    print("=== 3. The gather, one layer, one sequence ==========================")
    layer = 0
    ids = torch.tensor(block_table)

    step1 = pool[ids]                      # advanced indexing on dim 0
    print(f"  a) pool[block_ids]           {tuple(step1.shape)}")
    print("       [n_blocks, K/V, layers, kv_heads, block, dim]")

    step2 = step1[:, 0, layer]             # keys, this layer
    print(f"  b) [:, K=0, layer]           {tuple(step2.shape)}")
    print("       [n_blocks, kv_heads, block, dim]")

    step3 = step2.permute(1, 0, 2, 3)      # heads first, blocks in time order
    print(f"  c) .permute(1, 0, 2, 3)      {tuple(step3.shape)}")
    print("       [kv_heads, n_blocks, block, dim]   <- blocks now adjacent in time")

    step4 = step3.reshape(num_kv_heads, len(block_table) * BLOCK_SIZE, head_dim)
    print(f"  d) .reshape(...)             {tuple(step4.shape)}")
    print("       [kv_heads, n_blocks*block, dim]    <- one flat time axis")

    step5 = step4.unsqueeze(0)             # batch of 1
    print(f"  e) .unsqueeze(0)             {tuple(step5.shape)}")
    print("       [batch, kv_heads, kv_len, head_dim]  <- what update() returns")
    print()
    print("  Sanity check -- head 0, dim 0, across the flat time axis:")
    print(f"    {step5[0, 0, :, 0].tolist()}")
    print(f"    reads {block_table[0]}xx, {block_table[1]}xx, {block_table[2]}xx in order.")
    print("    Scattered blocks, contiguous result. THAT is PagedAttention's trick,")
    print("    done with a copy instead of a custom kernel (the M3 guardrail).")
    print()
    print(f"  Note kv_len = {step5.shape[2]} = 3 blocks x {BLOCK_SIZE}, not {seq_len}.")
    print("  Phase 0 proved the block-padded length is FINE: declare it from")
    print("  get_mask_sizes and causality masks the tail. No trimming needed.")
    print()
    print("  WHY permute *then* reshape: reshape reads memory in row-major order.")
    print("  Reshaping [n_blocks, heads, block, dim] directly would interleave")
    print("  heads into the time axis. Try it -- the sanity check goes garbage:")
    wrong = step2.reshape(len(block_table) * BLOCK_SIZE, num_kv_heads, head_dim)
    print(f"    wrong-order first 6 values: {wrong[:, 0, 0].tolist()[:6]}")
    print()

    # ---------------------------------------------------------------- part 4
    print("=== 4. The scatter (writing new tokens in) ==========================")
    print("  Decode writes ONE token per sequence per step. For a sequence at")
    print(f"  position {seq_len} with block_table {block_table}:")
    pos = seq_len
    lb, slot = pos // BLOCK_SIZE, pos % BLOCK_SIZE
    print(f"    pos {pos} -> logical block {lb}, slot {slot}"
          f" -> physical block {block_table[lb]}")
    print(f"    pool[{block_table[lb]}, :, :, :, {slot}, :] = new_kv")
    print()
    print("  The trap: doing that in a Python for-loop over the batch means")
    print("  batch_size * num_layers small copies per step. Look up")
    print("  Tensor.index_copy_ and advanced indexing with a tensor of")
    print("  (block_id, slot) pairs to do the whole batch in one op.")
    print()

    # ---------------------------------------------------------------- part 5
    print("=== 5. Exercises (this is the milestone -- yours to write) ==========")
    print("  E1. BlockAllocator: free list as a LIFO stack, allocate(n) / free(ids)")
    print("      / num_free, and a CATCHABLE exhaustion error (not IndexError")
    print("      from .pop() on an empty list -- Phase 4 treats exhaustion as")
    print("      normal control flow).")
    print("  E2. Sequence.block_table + the growth rule: when does a decode step")
    print("      need a NEW block? (hint: exactly when pos % BLOCK_SIZE == 0)")
    print("  E3. Batched gather: part 3 did ONE sequence. A batch has different")
    print("      block tables of different lengths. What is the kv_len of the")
    print("      returned rectangle, and what goes in the shorter rows?")
    print("  E4. Batched scatter: part 4's one-op version.")
    print("  E5. Block reuse: sequence A finishes, its blocks are freed, sequence")
    print("      B allocates them. B must never see A's KV. Which mechanism")
    print("      guarantees that -- zeroing on free, or something cheaper?")


if __name__ == "__main__":
    main()
