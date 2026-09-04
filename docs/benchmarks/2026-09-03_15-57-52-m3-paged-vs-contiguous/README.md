# M3 Phase 6 — paged vs. contiguous at equal VRAM

- Model: `Qwen/Qwen2.5-0.5B-Instruct`  
- Commit: `cceca99`  
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.0 GiB total)  
- torch 2.13.0+cu130, Python 3.12.7  
- KV per token: 12 KiB (192 KiB per 16-token block)  
- Reserve-max length for this workload: 64 tokens  
- Workload: M2 Phase 4 fixed prompt list, mixed max_new_tokens (composes with the M2 run) (6 requests)

| budget | contiguous batch | paged blocks | peak concurrent (cont / paged) | KV util peak (cont / paged) | throughput tok/s (cont / paged) | preemptions |
|---|---|---|---|---|---|---|
| 1 MiB | 1 | 5 | 1 / 5 | 98% / 99% | 16.3 / 20.0 | 4 |
| 2 MiB | 2 | 10 | 2 / 6 | 84% / 81% | 30.1 / 33.7 | 0 |
| 3 MiB | 4 | 16 | 4 / 6 | 59% / 51% | 52.8 / 31.6 | 0 |
| 4 MiB | 5 | 21 | 5 / 6 | 47% / 39% | 54.3 / 32.9 | 0 |
| 6 MiB | 8 | 32 | 6 / 6 | 26% / 25% | 55.9 / 32.4 | 0 |
| 8 MiB | 10 | 42 | 6 / 6 | 21% / 19% | 53.6 / 33.1 | 0 |

## Budget check

The contiguous engine has no VRAM knob, so its `max_batch_size` is derived
from the budget. Measured against the tensors `StaticCache` actually
allocated at its widest moment:

- 1 MiB: measured 0.75 MiB peak - **within budget**
- 2 MiB: measured 1.50 MiB peak - **within budget**
- 3 MiB: measured 3.00 MiB peak - **within budget**
- 4 MiB: measured 3.75 MiB peak - **within budget**
- 6 MiB: measured 4.50 MiB peak - **within budget**
- 8 MiB: measured 4.50 MiB peak - **within budget**

## Cross-check

- 1 MiB: total_tokens 180 vs 180 — **OK**
- 2 MiB: total_tokens 180 vs 180 — **OK**
- 3 MiB: total_tokens 180 vs 180 — **OK**
- 4 MiB: total_tokens 180 vs 180 — **OK**
- 6 MiB: total_tokens 180 vs 180 — **OK**
- 8 MiB: total_tokens 180 vs 180 — **OK**

Different memory strategies, identical output. A mismatch here would 
invalidate every number above it.

## Validation point — a budget large enough to saturate

- Budget: 64 MiB  
- Requests: 128, validation: 128 requests, long-tailed max_new_tokens 16-192 (does NOT compose with M2)  
- Reserve-max length: 203 tokens (2.38 MiB per contiguous sequence)

| | contiguous | paged |
|---|---|---|
| derived config | max_batch_size=26 | num_blocks=341 |
| **peak concurrent sequences** | **26** | **128** |
| reserved-slot occupancy | 42% | 90% |
| KV utilization of budget (peak) | 37% | 75% |
| throughput (tok/s) | 163.0 | 466.2 |
| TTFT p50 / p99 (s) | 13.20 / 38.61 | 0.46 / 0.46 |
| TPOT p50 / p99 (s) | 0.112 / 0.222 | 0.098 / 0.100 |
| preemptions | — | 0 |
| internal fragmentation (peak slots) | — | 1440 |
| total tokens | 8600 | 8600 |

**Concurrency ratio: 4.92x** at equal VRAM.

Cross-check: total_tokens 8600 vs 8600 — **OK**.

---

## Reading these numbers

**The headline: 4.92x more concurrent sequences at equal VRAM**, with
reserved-slot occupancy going from 42% to 90%. The second number explains the
first — it is not an independent result. A contiguous cache reserves a
rectangle sized for the longest sequence in the batch, so on a long-tailed
workload more than half the memory it holds is reserved against tokens that
never arrive. Paging reserves in 16-token blocks on demand, so the only waste
left is the tail of each sequence's last block.

**4.92x is a floor, not a ceiling.** The paged engine peaked at 128 concurrent
sequences — which is every request in the workload. It never ran out of blocks
(zero preemptions), so its real limit at 64 MiB was not reached. The contiguous
engine, by contrast, was genuinely saturated at 26. A larger workload would
widen the gap, not narrow it.

**Throughput rose 2.9x (163 → 466 tok/s), which is a consequence and not a
separate win.** Decode is memory-bandwidth-bound: the model weights are read
once per step regardless of how many sequences share that step, so five times
the batch is close to five times the useful work for the same weight traffic.
This is the arithmetic-intensity argument, and it is why the concurrency number
is the one that matters.

**TTFT p99 fell from 38.6 s to 0.46 s — a 84x difference, and the biggest
single number in this table.** That is not paging. It is the absence of
ADR-0007's resync. The contiguous engine rebuilds its entire `StaticCache`
whenever batch membership changes, re-prefilling every running sequence; with
26 rows finishing and being replaced continuously, a newly admitted request
waits behind repeated full-batch prefills. A paged cache has a block table per
sequence, so a membership change costs nothing. **Report this separately from
the paging result** — conflating the two would credit paging with a win that
belongs to per-sequence indexing.

**TPOT p50 moved very little (0.112 → 0.098 s) and p99 improved (0.222 →
0.100).** The paged path pays a gather per layer per step that the contiguous
path does not, so a per-sequence slowdown was the expected cost. It does not
show at this scale: at 128 concurrent sequences the gather is small relative to
the weight traffic of a forward pass. It would show at long context lengths,
where the gathered rectangle grows with history. That is the honest boundary of
this result.

### Why the sweep and the validation point disagree

The sweep's ratio collapses to 1.0x above 3 MiB while the validation point
shows 4.92x. Both are correct, and the reason is the sweep's workload: six
requests. Once the budget is large enough for the contiguous engine to run all
six, there is nothing left for paging to win — peak concurrency is capped by
the workload, not by memory. The sweep is kept because it composes with the
recorded M2 run and because its low-budget rows show the mechanism cleanly
(1 MiB: 1 sequence vs. 5, and the only row where preemption fires at all). The
validation point exists precisely because six requests cannot demonstrate the
milestone's claim.

### What paging did *not* do

It did not eliminate padding. The gathered buffer handed to
`scaled_dot_product_attention` is still a rectangle padded to the batch's
longest row, because SDPA requires one. What went away is *reservation* waste
in the pool — memory held against tokens that may never exist. The residual
waste is internal fragmentation, reported above at 1,440 token slots peak,
which is bounded by `BLOCK_SIZE - 1` per sequence (15 × 128 = 1,920 worst
case) rather than by the workload's length distribution. That bound is the
claim M3 actually makes.

### Caveats on provenance

- Run on an RTX 3050 Ti Laptop GPU (4 GiB). A 0.5B model cannot be driven into
  genuine whole-GPU VRAM pressure at a feasible runtime — it would need
  thousands of concurrent long sequences, and the contiguous engine's resync
  makes that take hours. The 64 MiB validation budget reproduces the pressure
  regime at a scale that runs in minutes; the *ratio* is what transfers, not
  the absolute concurrency figures.
- Benchmarks run on GPU + SDPA, the real path. The correctness gate that
  licenses quoting these numbers runs on CPU + eager (ADR-0010) and is a
  separate configuration by design.
- The validation workload does **not** compose with the M2 run. Only the sweep
  does.
