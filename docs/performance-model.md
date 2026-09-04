# What moves the numbers, and why

A mechanism-by-mechanism account of what this engine does to performance.
Every figure is from `docs/benchmarks/2026-09-03_15-57-52-m3-paged-vs-contiguous/`
(RTX 3050 Ti Laptop, Qwen2.5-0.5B-Instruct bf16, commit `cceca99`) or from the
M2 run it composes with. Numbers without provenance are not quotable.

---

## 0. The physics the metrics sit on

Two facts explain most of what follows.

**Decode is memory-bandwidth-bound.** A decode step reads every model weight
once, whatever the batch size. Useful work scales with batch; weight traffic
does not. Arithmetic intensity is therefore roughly proportional to batch size,
and a batch of 1 is the worst possible use of a GPU.

**Prefill is compute-bound.** A prompt of N tokens does N tokens' worth of
matmul in one pass. It is the phase with work to spare, which is why
recomputation is cheap and a host round trip is not (ADR-0012).

The four metrics map onto these directly:

| metric | what it physically measures |
|---|---|
| **peak concurrent sequences** | how much KV fits in the budget |
| **throughput** | concurrency × per-sequence decode rate |
| **TTFT** | prefill compute **+ queueing behind other work** |
| **TPOT** | one decode step's latency, divided by nobody |

**Peak concurrency is upstream of throughput.** That is why M3's headline is a
capacity number and not a tokens/sec number.

---

## 1. KV bytes per token — the constant everything divides by

```
2 (K and V) x num_layers x num_kv_heads x head_dim x itemsize
2 x 24 x 2 x 64 x 2 = 12,288 bytes = 12 KiB per token
```

**`num_kv_heads`, not `num_heads`.** Qwen2.5-0.5B is GQA: 14 attention heads,
**2** KV heads. Grouped-query attention is a **7x** divisor on KV memory for
this model — the single largest constant in the budget. The roadmap's pool
shape said `num_heads`; taken literally it would have allocated a 7x oversized
pool and quietly invalidated every memory number the milestone produced.

Everything downstream is this constant divided into a budget. At 64 MiB:
5,461 token slots, total, to be shared by every live sequence.

---

## 2. Reserve-max: why the contiguous engine stops at 26

`StaticCache` is a rectangle. Every row is sized for
`max(prompt + max_new_tokens)` across the running batch — 203 tokens in the
validation workload.

```
per row : 203 x 12 KiB = 2.38 MiB
64 MiB  / 2.38 MiB     = 26 rows
```

The waste is `max_len − actual_len` **per sequence**, so it scales with how
long-tailed the workload is. Measured **reserved-slot occupancy: 42%**. Nearly
three-fifths of the memory paid for is held against tokens that never arrive.

This is not an implementation flaw. A rectangle cannot express "this row is
done at 40 tokens and that one needs 190" — reserving for the worst case is the
only correct thing a rectangle can do.

---

## 3. Paged allocation: 4.92x concurrency

Blocks of 16 tokens, handed out on demand, addressed through a per-sequence
block table. The waste term changes character entirely:

| | waste per sequence | scales with |
|---|---|---|
| contiguous | `max_len − actual_len` | the workload's length distribution |
| paged | `≤ BLOCK_SIZE − 1` (15 slots) | nothing |

**That bound is the claim M3 actually makes.** Occupancy 42% → **90%**;
concurrency 26 → **128**, a **4.92x** ratio.

**It is a floor.** Paged peaked at 128 — every request in the workload — with
**zero preemptions**, holding 4,077 of 5,456 slots. It never hit the wall, so
its real ceiling at 64 MiB is unknown. Contiguous was genuinely saturated.

**Occupancy explains concurrency; it does not corroborate it.** They are one
fact seen from two sides.

---

## 4. Concurrency → throughput, and the tax on the way

163 → 466 tok/s is **2.86x**, not 4.92x. The gap is the gather. The whole
result factorizes:

```
system throughput   =   concurrency   x   per-sequence rate
     2.86x          =      4.92x      x        0.58x
```

Paging bought 4.92x the sequences and gave back 42% of each sequence's decode
rate. Quoting either factor alone misrepresents the trade.

The 4.92x is worth far more than the 0.58x costs *because decode is
bandwidth-bound*: five times the batch is close to five times the useful work
for the same weight traffic. On a compute-bound workload this trade would not
pay.

---

## 5. Gather-then-attend: where the 0.58x goes

SDPA requires a contiguous, rectangular K/V. A paged sequence's KV is neither.
So every layer, every step, the blocks are copied into a scratch rectangle.

**Cost, measured with the concurrency win removed** (sweep rows where both
engines peak at 6 sequences):

| | contiguous | paged | ratio |
|---|---|---|---|
| TPOT p50 @ 6 seqs | 0.053 s | 0.090 s | **1.68x worse** |
| throughput @ 6 seqs | 55.9 tok/s | 32.4 tok/s | **0.58x** |

**Paging is a straight loss below roughly 19 concurrent sequences and a large
win above it.** The crossover exists because the two curves have different
shapes:

| concurrency | contiguous TPOT | paged TPOT |
|---|---|---|
| 6 | 0.053 s | 0.090 s |
| 26 | 0.112 s | — |
| 128 | — | 0.098 s |

**Contiguous step latency grows steeply with batch (2.1x from 6 to 26). Paged
step latency is nearly flat (+9% from 6 to 128).** Caveat: part of contiguous's
growth is the resync of §6, not attention, so this is a crossover between
*these two engines*, not gather-vs-nothing.

**Why the trade is taken anyway:** the alternative is a kernel that reads
scattered blocks directly, which is the guardrail's exact prohibition and a
much larger project. Gather-then-attend proves the paging idea without it.

**Where the gather cost would actually bite: long context.** The gathered
rectangle grows with history, so its cost scales with *total live tokens*. At
203 tokens it is small next to weight traffic. At 8K it would not be. **"Paging
is free" is not supported by this data.**

### The unexplained part — flagged, not fixed

A paged step does **21x the work** going from 6 to 128 sequences for **+9%
time**. Nearly-flat step latency means the step is dominated by something that
is not the work. At batch 6 the gather moves ~98K elements per layer — nothing
— yet costs ~37 ms/step against contiguous. **That is not bandwidth. It is
overhead.** Four candidates, ranked:

1. `block_table_tensor()` rebuilds a padded Python list and does a **pageable
   host→device copy every step** (which synchronizes).
2. `scatter_index` does two more pageable H2D transfers per step (`q_lens`,
   `past_lens`) plus ~8 small device ops with fixed launch cost.
3. `gather_kv` **copies twice per layer** — advanced indexing materializes one,
   then `.permute().reshape()` breaks contiguity and forces a second. 48 copies
   per step, not 24.
4. 24 separate small per-layer gathers, each with its own launch overhead.
   (Per-layer was chosen deliberately: peak scratch is 1/24th, and scratch
   competes with the pool for the VRAM this milestone exists to reclaim.)

**This matters most at M4.** Quantization shrinks the weight traffic that this
fixed cost is currently hiding behind, so an unprofiled ~90 ms/step will
surface there as a disappointing quantization result.

---

## 6. Why TTFT fell 84x — and why it is not paging

**38.6 s → 0.46 s p99, the largest number in the run.** The mechanism has
nothing to do with block tables making prefill faster.

`StaticCache` has **one shared write position** for the whole batch (ADR-0007).
When batch membership changes — any sequence finishing, any new one admitted —
the surviving rows' positions can no longer be expressed, so M2's engine
**re-prefills the entire running batch**.

```
cost of one membership change = full prefill of all 26 running sequences
128 requests through a 26-slot batch = ~128 membership changes
```

A newly admitted request's first token therefore queues behind repeated
full-batch prefills. That queue is the 38.6 s.

A paged cache has a **block table per sequence**; each row's position is its
own. A neighbour retiring is a free-list push and touches nothing. Membership
changes cost nothing.

**So the win is not faster prefill — it is the removal of ~128 redundant
re-prefills.** Same work, done once instead of many times. Pinned by
`test_one_sequence_finishing_does_not_disturb_another`, which asserts that **no
prefill call is issued** for a survivor when a neighbour retires.

**Report this separately from the paging result.** It is a real and separable
win of M3; crediting it to paging would overstate paging by an order of
magnitude.

---

## 7. Prefill-priority scheduling: TTFT bought with TPOT

One batch kind per step (ADR-0013). If any waiting sequence can be admitted,
that step is a prefill step and nothing decodes.

- **Buys:** a new arrival waits at most one decode step before its prefill.
  This is most of the 0.46 s TTFT.
- **Costs:** head-of-line blocking. A long prefill stalls every running
  sequence's decode for that step. It lands in TPOT p99.
- **The fix, out of scope:** chunked prefill — split a long prefill across
  steps and co-schedule it with decode. Named rather than smoothed over.

---

## 8. Preemption by recomputation: the cost of not crashing

Fires only under genuine pressure — **the 1 MiB sweep row is the only place in
the entire run where it fired** (4 times).

- **Cost:** re-prefill of prompt + already-generated tokens. Grows with how far
  along the victim was.
- **Victim = newest.** Two reasons: least sunk cost, and it *guarantees
  termination* — the oldest sequence can always eventually run alone. Requeued
  at the front of the waiting deque.
- **Rejected alternative — swap to host memory** (ADR-0012): preserves the
  work, pays PCIe. Recomputation and swapping trade the same currency in
  opposite directions; on a laptop the link is the scarce resource and prefill
  compute is the abundant one. Swapping also needs a host pool, a host
  allocator, a GPU→host block map and a transfer in the `step()` critical path
  — the least-tested code holding the most invariants.
- **Effect on metrics:** raises the victim's time-to-completion and total
  tokens computed; keeps the engine from crashing. Invisible in the validation
  run (0 preemptions).
- **Correctness:** a preempted sequence loses its KV, is recomputed, and emits
  **byte-identical** tokens. That is what makes preemption a cost rather than a
  behaviour change — and it is why these numbers can be quoted at all.

---

## 9. Block size = 16: the knob nobody turned

The genuine tension in the design, and it is unmeasured.

| | smaller blocks | larger blocks |
|---|---|---|
| internal fragmentation | less (`≤ BS−1` per seq) | more |
| block table length | longer | shorter |
| gather index work | **more** | less |

At the validation point, internal fragmentation peaked at **1,440 token slots
≈ 16.9 MiB of a 64 MiB budget**. That is the dominant residual waste and the
reason `kv_utilization` of the *budget* peaks at 75% while occupancy of what
live sequences hold is 90%.

Halving `BLOCK_SIZE` to 8 would roughly halve that waste — and double block
table length, which feeds straight into the overhead suspected in §5. **The two
levers pull against each other, and the profile in §5 has to come first.**

---

## 10. Two configuration traps

**`max_batch_size` must not bind.** In M2 it *was* the binding constraint, so
`peak_concurrent_sequences` merely reported it back. In M3 memory must bind;
setting `max_batch_size` low enough to bind again silently un-measures the
entire milestone.

**"Equal VRAM" is not a setting either engine has.** `ContinuousEngine` takes
`max_batch_size`, `PagedEngine` takes `num_blocks`. A byte budget is converted
into each engine's currency, and the contiguous derivation is *verified*
against the tensors `StaticCache` actually allocates rather than trusted
(ADR-0014).

---

## Summary: what moves what

| lever | peak concurrent | throughput | TTFT | TPOT | mechanism |
|---|---|---|---|---|---|
| GQA (2 KV heads vs 14) | **7x ↑** | ↑ | — | — | 7x fewer KV bytes per token |
| paged vs. reserve-max | **4.92x ↑** | 2.86x ↑ | — | — | waste bound becomes `BLOCK_SIZE−1`, not `max_len−actual` |
| gather-then-attend | — | **0.58x ↓** | — | **1.68x ↓** | a contiguous copy per layer per step, for SDPA |
| per-sequence block tables | — | ↑ | **84x ↑** | ↑ | no full-batch resync on membership change |
| prefill-priority steps | — | — | **↑** | ↓ (p99) | head-of-line blocking, one batch kind per step |
| preemption (recompute) | enables >pool | ↓ slightly | ↓ for victim | — | discards KV, replays prefill |
| smaller `BLOCK_SIZE` | ↑ | ? | — | ? | less fragmentation, more index work |

## What to move next, ranked

1. **Profile the ~90 ms fixed step cost** (§5). Largest known unexplained
   quantity, and it distorts M4.
2. **Chunked prefill** — removes the TPOT p99 cost of §7.
3. **Quantization (M4)** — shrinks weights, freeing VRAM for KV, raising
   concurrency again. Will also expose (1).
4. **`BLOCK_SIZE` sweep** — only after (1), since the two interact.
5. **A paged-attention kernel** — would remove §5 entirely. Guardrailed out of
   this project by design; a separate and much larger one.

## Standing caveats

- The paged benchmark ran **GPU + SDPA**; the correctness gate that licenses
  quoting it runs **CPU + eager** (ADR-0010). Two configurations by design, and
  Phase 0's block-padded-gather findings still need re-verifying under SDPA.
- Never reached genuine whole-GPU VRAM pressure — a 0.5B model on a 4 GiB
  laptop GPU would need thousands of concurrent long sequences. **The ratios
  transfer; the absolute concurrency figures do not.**
- **Paging did not remove padding.** The buffer handed to SDPA is still a
  rectangle padded to the batch's longest row. What went away is *reservation*
  waste in the pool.
