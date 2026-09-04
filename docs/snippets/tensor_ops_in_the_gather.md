# Tensor operations in `04_block_table_and_gather.py`

Every operation in this file is one of two kinds:

- **Metadata edits** — rewrite the shape/stride tuple, touch zero bytes. Free.
- **Materializing ops** — actually move data. These are what cost you.

The gather chain in part 3 runs five operations. Two of them move memory. Knowing
which two is the whole point of reading this file carefully.

```
a) pool[ids]              (3, 2, 2, 2, 4, 3)   COPY      advanced indexing
b) [:, 0, layer]          (3, 2, 4, 3)         view      basic slicing
c) .permute(1,0,2,3)      (2, 3, 4, 3)         view      stride reorder
d) .reshape(...)          (2, 12, 3)           COPY      forced by (c)
e) .unsqueeze(0)          (1, 2, 12, 3)        view      rank fix
```

---

## Advanced indexing — `pool[ids]`

**What it does.** `ids` is a tensor, not an int or a slice. PyTorch reads it as
"give me these rows of dim 0, in this order" and builds a new tensor whose first
axis has length `len(ids)`.

**What happens.** `ids = [7, 2, 9]` selects pool rows 7, 2, 9 and stacks them as
rows 0, 1, 2 of the result. The output is a fresh allocation — there is no stride
pattern that expresses "row 7, then row 2, then row 9", so it cannot be a view.
Repeats are legal (`[7, 7, 2]` works), which is what makes prefix sharing possible
later.

**Why it's here.** This *is* the address translation. The block table is a Python
list of physical block ids; turning it into a tensor and indexing with it performs
every lookup in one op instead of a Python loop. The scattered blocks land
adjacent in the output — that reordering is the entire trick.

Contrast with basic slicing (`pool[7:10]`), which is a view: a contiguous range
has a stride, so no copy is needed.

---

## Basic slicing — `[:, 0, layer]`

**What it does.** Integer indices drop an axis; `:` keeps one. Two ints here, so
rank goes from 6 to 4.

**What happens.** `0` picks K (not V) and `layer` picks one layer, both by
adjusting the base pointer and dropping their entries from the shape/stride tuples.
No data moves.

**Why it's here.** The pool holds K and V for all layers in one tensor. Attention
is computed per layer, on K and V separately, so you have to narrow down before
anything else. Doing it early also means steps c–e operate on a tensor ~4x smaller.

---

## `permute` — reorder axes

**What it does.** Rearranges the axes. Purely a stride-tuple shuffle; element
`[i,j,k]` of the result is element `[j,i,k]` of the input, found by arithmetic, not
by moving it.

**What happens.** `(n_blocks, heads, block, dim) -> (heads, n_blocks, block, dim)`.
The result is **non-contiguous**: walking it in row-major order no longer walks
memory in increasing address order.

**Why it's here.** It sets up the reshape. `reshape` flattens whichever axes are
currently adjacent, so this line decides *what gets merged next*. Skip it and
`reshape` merges blocks with heads instead of blocks with slots — the file
demonstrates exactly this, and the sanity check reads `700, 702, 700, 702, ...`:
the same block twice, half the slots invisible.

`permute` doesn't move data. It changes what "adjacent" means.

---

## `reshape` — merge the block and slot axes

**What it does.** Reinterprets the same elements under a new shape, reading in
row-major order.

**What happens.** `(heads, n_blocks, block, dim) -> (heads, n_blocks*block, dim)`.
Because the input is non-contiguous after `permute`, no stride tuple can express
the requested shape, so `reshape` silently allocates and copies. (`view` would
raise `RuntimeError` here instead — that's the only difference between them.)

**Why it's here.** SDPA wants one flat time axis. Blocks and slots are two axes
describing the same thing — position in the sequence — and this collapses them.

This is the second and last copy in the chain. Merging two axes into one always
costs a copy unless the input is already contiguous.

---

## `unsqueeze` — insert a length-1 axis

**What it does.** Adds an axis of extent 1. Nothing moves; the new axis has exactly
one valid index, so its stride is a placeholder.

**What happens.** `(2, 12, 3) -> (1, 2, 12, 3)`.

**Why it's here.** Two distinct reasons, and it's worth keeping them separate in
your head because they show up in different places:

**1. Satisfying a rank contract.** `scaled_dot_product_attention` requires 4D
`(batch, heads, seq, head_dim)`. The gather produced rank 3 because it handled one
sequence. `unsqueeze(0)` declares "batch of one." The position `0` is forced: SDPA
treats the trailing two axes as the matmul dims and everything left of them as
batch. `unsqueeze(1)` would not error — it would compute attention against the
wrong axes.

**2. Controlling broadcast alignment.** Broadcasting aligns shapes **from the
right**; dims are compatible if equal or if one is 1, and size-1 dims stretch for
free at stride 0. The placement rule:

> Put a `1` on every axis along which the tensor is **constant**.

Not "along which it's missing" — along which it doesn't vary. That reframing makes
placement mechanical. This is what you'll use constantly in E3's mask:

```python
mask = pos[None, :] < ctx_lens[:, None]   # (1, L) vs (S, 1) -> (S, L)
mask = mask[:, None, None, :]             # (S, 1, 1, L) for SDPA
```

`pos` is constant across sequences, so `1` goes on the sequence axis. `ctx_lens` is
constant across positions, so `1` goes on the position axis. Then the padding mask
is constant across heads and across query positions, so both get a `1`. Swap any of
these and you get a shape error, or — when two dims coincidentally match — a
silently transposed mask.

---

## The scatter (part 4) — `index_copy_` / advanced indexing

**What it does.** Writes into the pool at computed positions. Advanced indexing on
the left side of an assignment is a scatter, the mirror of the gather in part 3.

**What happens.** `pool[block_ids, :, :, :, slots, :] = new_kv` routes each
sequence's new K/V to its own physical block and slot in a single op.

**Why it's here.** Decode writes exactly one token per sequence per step. The trap
the file names is real: a Python loop over the batch issues
`batch_size * num_layers` tiny CUDA launches per step, and launch overhead will
dominate your benchmark numbers before the model does. Compute the
`(block_id, slot)` pairs on the CPU once per step, ship them as one tensor, do one
op.

---

## `expand` — the one you'll need next (GQA)

**What it does.** Stretches a size-1 axis to size *n* by setting its stride to 0 —
every index along that axis reads the same memory. Free, and read-only.

**Why you'll need it.** If the real model uses grouped-query attention, you'll have
`num_q_heads = n * num_kv_heads` and SDPA needs them equal. The standard idiom:

```python
k = k.unsqueeze(2)                     # (S, kv_h, 1, L, D)   free
k = k.expand(S, kv_h, n, L, D)         # (S, kv_h, n, L, D)   free, stride 0
k = k.reshape(S, kv_h * n, L, D)       # COPY — merging axes
```

`unsqueeze` makes the group axis, `expand` fills it for free, `reshape` pays.
Same pattern as part 3: stride edits are free right up until you flatten.

---

## The rule to carry forward

| Free (metadata) | Costs memory |
|---|---|
| basic slicing | advanced indexing (`pool[ids]`) |
| `permute`, `transpose` | `reshape` on non-contiguous input |
| `unsqueeze`, `squeeze` | `contiguous()` |
| `expand` | any axis merge |

The gather-then-attend approach pays two copies per layer per step. That is the
M3 guardrail working as intended — the real PagedAttention kernel pays zero,
because it reads `pool[block_id]` directly inside the attention loop and never
builds the contiguous rectangle at all. You're buying correctness with a copy, and
the README should say so.
