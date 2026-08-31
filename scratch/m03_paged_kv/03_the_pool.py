"""03 — The pool: one big tensor, and what a block id actually indexes.

Locked decision: ONE pre-allocated tensor, shape

    [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
     ^dim 0       ^K/V

A `block_id` is just an integer index into dim 0. `pool[block_id]` is a
contiguous slice -- that is the whole reason num_blocks goes first.

Run:  uv run python scratch/m03_paged_kv/03_the_pool.py
"""

import torch

BLOCK_SIZE = 16

# Toy dims first, so the printouts are readable. Real dims at the bottom.
TOY = dict(num_blocks=8, num_layers=2, num_kv_heads=2, block_size=4, head_dim=3)
REAL = dict(num_blocks=1024, num_layers=24, num_kv_heads=2, block_size=16, head_dim=64)


def make_pool(num_blocks, num_layers, num_kv_heads, block_size, head_dim, dtype=torch.float32):
    # torch.ZEROS, never torch.empty. Uninitialised memory can contain inf/NaN,
    # and a NaN in a masked-out slot still poisons the output (0 * inf = NaN).
    # A *reused* block always holds a dead sequence's finite KV, so zeroing once
    # at creation is enough -- free() never has to zero.
    return torch.zeros(
        (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim), dtype=dtype
    )


def describe(pool, label):
    n_blocks, two, n_layers, n_heads, blk, hd = pool.shape
    per_block = pool[0].numel() * pool.element_size()
    print(f"--- {label} ---")
    print(f"  pool.shape       {tuple(pool.shape)}")
    print("                    blocks  K/V  layers  kv_heads  block  dim")
    print(f"  total elements   {pool.numel():,}")
    print(f"  total bytes      {pool.numel() * pool.element_size() / 1024**2:,.1f} MiB")
    print(f"  ONE block        {per_block / 1024:,.1f} KiB  ({blk} tokens x all layers, K and V)")
    print(f"  pool[block_id]   {tuple(pool[0].shape)}  <- one block, all layers")
    print(f"  contiguous?      {pool[0].is_contiguous()}  <- a block is one flat run of memory")
    print()


def main():
    torch.set_printoptions(linewidth=140)

    toy = make_pool(**TOY)
    describe(toy, "TOY POOL (readable numbers)")

    print("--- slicing the pool: which axis is which ---------------------------")
    print("  pool[block_id]                 -> one block, everything in it")
    print(f"     shape {tuple(toy[3].shape)}   [K/V, layers, kv_heads, block, dim]")
    print("  pool[block_id, 0]              -> the KEYS of one block")
    print(f"     shape {tuple(toy[3, 0].shape)}      [layers, kv_heads, block, dim]")
    print("  pool[block_id, 0, layer]       -> the keys of one block, one layer")
    print(f"     shape {tuple(toy[3, 0, 1].shape)}         [kv_heads, block, dim]")
    print("  pool[block_id, 0, layer, :, slot] -> ONE TOKEN's keys in that layer")
    print(f"     shape {tuple(toy[3, 0, 1, :, 2].shape)}            [kv_heads, dim]")
    print()

    print("--- writing one token, then reading it back -------------------------")
    block_id, layer, slot = 3, 1, 2
    toy[block_id, 0, layer, :, slot, :] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"  wrote keys for (block={block_id}, layer={layer}, slot={slot}):")
    print(f"    {toy[block_id, 0, layer, :, slot, :].tolist()}")
    print(f"  neighbouring slot {slot + 1} is still zero: "
          f"{toy[block_id, 0, layer, :, slot + 1, :].tolist()}")
    print(f"  a different block ({block_id + 1}) is untouched: "
          f"{toy[block_id + 1, 0, layer, :, slot, :].tolist()}")
    print()

    print("--- why num_blocks is dim 0 -----------------------------------------")
    print("  Because a gather is then `pool[some_list_of_block_ids]`, one op,")
    print("  and each gathered block is a contiguous run the copy engine likes.")
    print("  Put num_layers first instead and a single sequence's blocks become")
    print("  a strided mess across the whole tensor.")
    print()
    print(f"  pool.stride() = {toy.stride()}")
    print(f"  -> stepping block_id by 1 jumps {toy.stride()[0]} elements;")
    print("     everything inside one block is packed together after that.")
    print()

    real = make_pool(**REAL, dtype=torch.bfloat16)
    describe(real, "REAL POOL (Qwen2.5-0.5B, 1024 blocks, bf16)")
    print(f"  capacity: {REAL['num_blocks'] * REAL['block_size']:,} tokens total,")
    print("  shared across every running sequence -- that sharing is the point.")


if __name__ == "__main__":
    main()
