# Worked examples: the block table gather

Every snippet below is runnable and every output shown is real. `BLOCK_SIZE = 4`
throughout, matching the toy value in `04_block_table_and_gather.py`.

Setup used by most examples:

```python
import torch
torch.set_printoptions(linewidth=140, precision=1, sci_mode=False)
B = 4
```

---

## W1 — Shape is a lie, stride is the truth

Every tensor is a flat block of memory plus a **shape** and a **stride**. The
stride says how many elements to step to advance one index along each axis. Most
of what looks like "moving data" is just editing that tuple.

```python
x = torch.arange(24.).reshape(2, 3, 4)
print(tuple(x.shape), x.stride(), x.is_contiguous())

y = x.permute(1, 0, 2)
print(tuple(y.shape), y.stride(), y.is_contiguous())
print("same storage:", x.data_ptr() == y.data_ptr())

z = x.unsqueeze(0)
print(tuple(z.shape), z.stride())
```

```
(2, 3, 4) (12, 4, 1) True
(3, 2, 4) (4, 12, 1) False
same storage: True
(1, 2, 3, 4) (24, 12, 4, 1)
```

Read the permute line carefully. The **shape** swapped to `(3, 2, 4)` and the
**stride** swapped to match, `(4, 12, 1)`. Not one byte moved — `data_ptr` is
identical. "Contiguous" just means the strides are descending and multiply out
correctly; after a permute they don't, hence `False`.

`unsqueeze` prepended a `1` to the shape and a `24` to the stride. That stride is
never used (there's only one valid index on that axis), so it's a placeholder.

> **Takeaway.** `is_contiguous()` is the flag that decides whether your next
> `reshape` is free or is a hidden copy.

---

## W2 — Basic slicing vs advanced indexing

This is the single most important distinction in the file.

```python
pool = torch.arange(60.).reshape(12, 5)

v = pool[7:10]                          # basic: a slice
print("slice shares storage:", v.data_ptr() == pool[7].data_ptr())

ids = torch.tensor([7, 2, 9])
g = pool[ids]                           # advanced: a tensor index
print("gather shape", tuple(g.shape), "shares storage:", g.data_ptr() == pool.data_ptr())
print("rows:", g[:, 0].tolist())

print("repeat ids [7,7,2]:", pool[torch.tensor([7, 7, 2])][:, 0].tolist())
```

```
slice shares storage: True
gather shape (3, 5) shares storage: False
rows: [35.0, 10.0, 45.0]
repeat ids [7,7,2]: [35.0, 35.0, 10.0]
```

`pool[7:10]` is a contiguous range, expressible as a base pointer plus a stride, so
it's a view. `pool[ids]` with `ids = [7, 2, 9]` is not — there is no stride pattern
meaning "row 7, then row 2, then row 9." PyTorch must allocate and copy.

The last line matters more than it looks. **Repeated ids are legal.** Two logical
blocks can point at the same physical block, and the gather happily duplicates it.
That is exactly the mechanism Milestone 6's prefix sharing runs on — the gather
code needs no changes at all when refcounted blocks arrive.

> **Takeaway.** Tensor index on dim 0 = copy, always. It's the price of arbitrary
> reordering, and arbitrary reordering is the whole point of a block table.

---

## W3 — The marker pool

The marker scheme makes every later result verifiable by eye: value =
`block_id * 100 + slot`.

```python
num_blocks, num_layers, num_kv_heads, head_dim = 12, 2, 2, 3
pool = torch.zeros((num_blocks, 2, num_layers, num_kv_heads, B, head_dim))
for b in range(num_blocks):
    for s in range(B):
        pool[b, :, :, :, s, :] = b * 100 + s

print(pool[7, 0, 0, 0, :, 0].tolist())
```

```
[700.0, 701.0, 702.0, 703.0]
```

So a `7xx` in any output means "came from physical block 7," and the last digit is
the slot within it. Build debugging aids like this before you build the thing
you're debugging.

---

## W4 — The five-step chain, annotated

```python
block_table = [7, 2, 9]
ids = torch.tensor(block_table)
layer = 0

s1 = pool[ids]
s2 = s1[:, 0, layer]
s3 = s2.permute(1, 0, 2, 3)
s4 = s3.reshape(num_kv_heads, len(block_table) * B, head_dim)
s5 = s4.unsqueeze(0)
```

```
a pool[ids]      (3, 2, 2, 2, 4, 3)  contig True   COPY
b [:,0,layer]    (3, 2, 4, 3)        contig False  view
c permute        (2, 3, 4, 3)        contig False  view
d reshape        (2, 12, 3)          contig True   COPY
e unsqueeze      (1, 2, 12, 3)                     view

head0 dim0 timeline: [700, 701, 702, 703, 200, 201, 202, 203, 900, 901, 902, 903]
```

Two copies out of five operations. Note step b already reports `contig False` —
picking `K=0` and one layer out of the middle of a 6D tensor leaves gaps, which is
why step d has to copy even though the permute alone might not have forced it.

Read the timeline: `7xx` then `2xx` then `9xx`, slots 0–3 in order within each.
Scattered blocks, contiguous result. That's the whole milestone in one line of
output.

---

## W5 — Why permute must come before reshape

The file's own demo, run:

```python
wrong = s2.reshape(len(block_table) * B, num_kv_heads, head_dim)
print(wrong[:, 0, 0].tolist())
```

```
[700, 702, 700, 702, 200, 202, 200, 202, 900, 902, 900, 902]
```

Diagnose it from the markers. Head 0's supposed timeline reads block 7 slot 0,
block 7 slot **2**, then block 7 slot 0 **again**. Slots 1 and 3 have vanished
entirely and everything appears twice.

Cause: `reshape` flattens in row-major order, merging whichever axes are currently
adjacent. In `(n_blocks, heads, block, dim)` the two leftmost axes are blocks and
**heads**, so it interleaved heads into the time axis. `permute` doesn't move data
— it changes what "adjacent" means, so the following `reshape` merges the axes you
actually meant.

Also worth seeing what `view` does in the same spot:

```python
s3.view(num_kv_heads, 12, head_dim)
# RuntimeError: view size is not compatible with input tensor's size and stride...
```

`view` refuses on non-contiguous input; `reshape` silently copies instead. That is
the only difference between them. **Use `view` while developing** — it turns a
hidden allocation into a loud error, which is exactly what you want when you're
trying to find out where your copies are.

---

## W6 — Building the mask (the two-unsqueeze idiom)

The rule for placing a `1`:

> Put a `1` on every axis along which the tensor is **constant** — not "along which
> it's missing," along which it doesn't vary.

```python
S = 3
ctx = torch.tensor([10, 5, 12])
L = 12
pos = torch.arange(L)

print(tuple(pos[None, :].shape), tuple(ctx[:, None].shape))
mask = pos[None, :] < ctx[:, None]
print(mask.int())
```

```
(1, 12) (3, 1)
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```

`pos` is the same for every sequence → `1` on the sequence axis → `[None, :]`.
`ctx` is one number per sequence, constant across positions → `1` on the position
axis → `[:, None]`. Broadcasting stretches both to `(3, 12)` and compares. Row 1
has 5 valid entries, matching `ctx[1] = 5`.

Get the two backwards and you get a `(12, 3)` tensor — usually a loud shape error
downstream, but silently wrong if `S` happens to equal `L`.

Then for SDPA:

```python
m4 = mask[:, None, None, :]
print(tuple(m4.shape), m4.stride())
```

```
(3, 1, 1, 12) (12, 12, 12, 1)
```

Two more `1`s: the padding mask is constant across heads (every head sees the same
padding) and across query positions. Note the strides — the two inserted axes both
have stride 12, never used, no memory allocated for the "expansion."

---

## W7 — Batched gather with ragged block tables (E3)

Sequences have different lengths, so their block tables do too. You pad to
`max_blocks`.

```python
bts = [[7, 2, 9], [3], [5, 1]]
maxb = max(len(b) for b in bts)
padded = torch.zeros(len(bts), maxb, dtype=torch.long)
for i, bt in enumerate(bts):
    padded[i, :len(bt)] = torch.tensor(bt)

G = pool[padded]                       # index tensor is 2D now
K = (G[:, :, 0, layer]
       .permute(0, 2, 1, 3, 4)
       .reshape(len(bts), num_kv_heads, maxb * B, head_dim))
print("seq0 head0:", K[0, 0, :, 0].tolist())
print("seq1 head0:", K[1, 0, :, 0].tolist())
```

```
pool[padded] -> (3, 3, 2, 2, 2, 4, 3)
K            -> (3, 2, 12, 3)
seq0 head0: [700, 701, 702, 703, 200, 201, 202, 203, 900, 901, 902, 903]
seq1 head0: [300, 301, 302, 303,   0,   1,   2,   3,   0,   1,   2,   3]
```

Two things to notice.

**The index tensor's shape becomes the output's leading shape.** `padded` is
`(3, 3)`, so `pool[padded]` is `(3, 3, ...)` — the gather generalizes to batches
for free, and the permute just gains a leading `0`.

**Look at seq1's tail: `0, 1, 2, 3`, twice.** The pad value `0` is a *real block
id*, so the gather dutifully fetched physical block 0. This is not zeros and it is
not garbage — it is another sequence's live KV cache. `kv_len = 12` for all rows
even though seq1 only has 4 valid tokens.

The mask is the only thing standing between that and a silent correctness bug:

```python
ctxb = torch.tensor([10, 4, 7])
mb = torch.arange(maxb * B)[None, :] < ctxb[:, None]
print(mb.sum(1).tolist())        # [10, 4, 7]
```

> **Answering E3 directly.** `kv_len = max_blocks * BLOCK_SIZE`. Shorter rows
> contain whatever physical block the padding id points at. Correctness comes
> entirely from the mask, never from the data.

---

## W8 — Feeding SDPA, and proving the mask works

```python
import torch.nn.functional as F

q = torch.randn(len(bts), num_kv_heads, 1, head_dim)   # decode: q_len = 1
V = K.clone()
out = F.scaled_dot_product_attention(q, K, V, attn_mask=mb[:, None, None, :])
print(tuple(q.shape), "->", tuple(out.shape))
```

```
(3, 2, 1, 3) -> (3, 2, 1, 3)
```

Now verify by hand for seq1, which has only 4 valid tokens out of 12:

```python
sc = (q[1, 0, 0] @ K[1, 0].T) / head_dim**0.5
sc = sc.masked_fill(~mb[1], float('-inf'))
w = sc.softmax(-1)
print("matches sdpa:", torch.allclose(w @ V[1, 0], out[1, 0, 0], atol=1e-5))
print("weight on padded slots:", w[4:].abs().max().item())
```

```
matches sdpa: True
weight on padded slots: 0.0
```

Exactly zero, not small — `exp(-inf) = 0` is exact in IEEE floating point. The
stale KV from block 0 contributes nothing at all. **Make this assertion a test**,
because it's the one that catches an off-by-one in the mask, and an off-by-one
there produces plausible-looking output rather than a crash.

Note the boolean mask convention: `True` = attend, `False` = masked. A float mask
means something different (it's *added* to the scores), so don't mix them.

---

## W9 — Batched scatter (E4)

```python
pos_list = [10, 4, 7]
bidx = torch.tensor([bts[i][p // B] for i, p in enumerate(pos_list)])
sidx = torch.tensor([p % B for p in pos_list])
print(list(zip(pos_list, bidx.tolist(), sidx.tolist())))

new_kv = torch.full((3, num_layers, num_kv_heads, head_dim), -1.)
pool[bidx, 0, :, :, sidx, :] = new_kv
```

```
[(10, 9, 2), (4, 3, 0), (7, 1, 3)]
before 902.0  ->  after -1.0
neighbour slot untouched: 903.0
```

One op writes all three sequences at all layers. The `bidx`/`sidx` tensors are
paired elementwise — position 0 of each names one target — so this is three
independent scattered writes issued as a single kernel launch.

The trap the file warns about is real. A Python loop here issues
`batch_size × num_layers` tiny launches every decode step; launch overhead is
~5–10 µs each, so at batch 32 and 32 layers that's over a thousand launches and
several milliseconds of pure overhead per token. Your benchmark would be measuring
Python, not the model.

Compute `bidx`/`sidx` on the CPU once per step, transfer once, index once.

---

## W10 — `expand` and GQA

`expand` stretches a size-1 axis by setting its stride to **0**: every index along
it reads the same memory.

```python
k = torch.arange(float(2*2*8*3)).reshape(2, 2, 8, 3)   # (S, kv_heads, L, D)
n = 3                                                   # q_heads per kv_head

ke = k.unsqueeze(2)
ke2 = ke.expand(2, 2, n, 8, 3)
ke3 = ke2.reshape(2, 2*n, 8, 3)
```

```
unsqueeze(2)  (2, 2, 1, 8, 3)  stride (48, 24, 24, 3, 1)  view
expand        (2, 2, 3, 8, 3)  stride (48, 24,  0, 3, 1)  view
reshape       (2, 6, 8, 3)                                COPY
q-head grouping: [0.0, 0.0, 0.0, 24.0, 24.0, 24.0]
```

The `0` in the stride is the entire mechanism. Three "copies" of each kv head exist
in the shape and none exist in memory.

The grouping check confirms the layout: q-heads 0,1,2 all read kv-head 0 (value
`0.0`), q-heads 3,4,5 all read kv-head 1 (value `24.0`). That's the correct
grouping — contiguous runs, not interleaved. `repeat` instead of `expand` here
would give you the interleaved order and wrong attention.

The final `reshape` copies because merging two axes into one cannot be expressed by
any stride tuple. If you'd rather not pay it, `enable_gqa=True` on
`scaled_dot_product_attention` handles the broadcast internally — but write the
explicit version once so you know what it's doing.

---

## W11 — Why you never zero on free (E5)

```python
p2 = torch.full((4, B), 9.)
p2[2] = torch.tensor([111., 222., 333., 444.])     # sequence A's KV
# A finishes; block 2 returns to the free list; sequence B allocates it
print(p2[2].tolist())
```

```
[111.0, 222.0, 333.0, 444.0]
```

Still there. Freeing a block is a `list.append(2)` — it does not touch the data.

The mechanism that protects B is masking, and it's exact:

```python
sc = torch.randn(B)
sc[2:] = float('-inf')
print(sc.softmax(-1).tolist())
```

```
[0.79, 0.21, 0.0, 0.0]
```

Zeroing on free would mean writing `2 × num_layers × num_kv_heads × B × head_dim`
floats per freed block, at exactly the moment you're trying to admit a new request.
Masking costs a comparison you were already doing.

> **Answering E5.** Something cheaper: the mask. Correctness comes from
> `context_len`, which is per-sequence state you already maintain, not from the
> contents of memory.

---

## W12 — The growth rule (E2)

```python
for p in range(10):
    print(f"pos {p:2} -> lb {p//B} slot {p%B}" + ("  NEW BLOCK" if p % B == 0 else ""))
```

```
pos  0 -> lb 0 slot 0  NEW BLOCK
pos  1 -> lb 0 slot 1
pos  2 -> lb 0 slot 2
pos  3 -> lb 0 slot 3
pos  4 -> lb 1 slot 0  NEW BLOCK
...
pos  8 -> lb 2 slot 0  NEW BLOCK
pos  9 -> lb 2 slot 1
```

A sequence of length `n` about to write token at position `n` needs a new block
exactly when `n % B == 0`. Equivalently `len(block_table) == n // B`.

The consequence for the scheduler: **allocation can fail mid-decode.** Every
running sequence may demand a new block on any step, and the pool is finite. That's
why E1 insists on a catchable exhaustion error rather than `IndexError` from
`.pop()` on an empty list — running out of blocks is normal control flow that
triggers preemption, not a bug.

---

## Reference card

| Operation | Cost | Why |
|---|---|---|
| `x[7:10]` basic slice | free | contiguous range has a stride |
| `x[ids]` tensor index | **copy** | arbitrary order has no stride |
| `x[ids] = v` | in-place | scatter, mirror of the gather |
| `permute` / `transpose` | free | stride reorder, leaves non-contiguous |
| `unsqueeze` / `squeeze` | free | shape edit, placeholder stride |
| `expand` | free | stride 0, read-only |
| `reshape` (contiguous input) | free | acts as `view` |
| `reshape` (non-contiguous) | **copy** | silent fallback |
| `view` (non-contiguous) | **error** | use this while developing |
| `contiguous()` | **copy** | explicit version of the above |

Rules that follow:

1. Narrow with basic slicing before you `reshape`.
2. `permute` decides what `reshape` merges. Order is never optional.
3. Broadcasting aligns from the right; put a `1` where the tensor is constant.
4. Padded rows contain live data from other sequences. The mask is load-bearing.
5. Anything per-sequence in a Python loop becomes per-step launch overhead.

The gather-then-attend path pays two copies per layer per step. That's the M3
guardrail working as designed — the real PagedAttention kernel pays zero, reading
`pool[block_id]` inside the attention loop and never building the rectangle. Note
that deviation in the README.
