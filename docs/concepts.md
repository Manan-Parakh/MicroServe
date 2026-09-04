# Concepts Glossary

A terse, growing glossary of systems/inference concepts as they enter the
codebase. Per `claude.md`: add an entry when a new concept first gets *built*
(not just read about) — pointing to where it's used in the code and to the
source that explains it properly. Keep entries short; this is an index, not a
tutorial.

Format:

```
### <term>
<one or two sentences — the working definition, not the textbook one>
**Used in:** `<file path>`
**Source:** <paper / book chapter / essay this came from>
```

---

### Greedy decoding
Always take the argmax token at each decode step instead of sampling from the
logits distribution. Makes generation a pure function of (weights, prompt) —
no RNG stream to reproduce — which is what makes an exact token-for-token
correctness comparison possible at all.
**Used in:** `tests/test_correctness.py`, `engine/model_runner.py`
**Source:** see `docs/bites/testing-and-correctness/correctness-harness.md`
for the deeper write-up.

### Naive/static serving
One request processed to full completion before the next one starts — no
batching, no interleaving. Deliberately bad: it's the "before" number every
scheduler-based improvement (M2 onward) is measured against.
**Used in:** `engine/engine.py`
**Source:** `docs/roadmap.md` Milestone 1.

### Throughput vs. latency
Throughput is system-wide tokens/sec; latency is how long one request takes
end to end. A naive engine can have "fine" latency for the one request
running and still have terrible throughput the moment a second request has
to wait its turn — they're different axes, not two names for the same thing.
**Used in:** `bench/benchmark.py`
**Source:** `claude.md` — "The four metrics."

### Sequence lifecycle (OS process-state analogy)
A request moves through `WAITING` → `RUNNING` → `FINISHED` as a scheduler
admits and evicts it — the same shape as an OS's ready/running/terminated
process states, minus a "blocked on I/O" equivalent. `RUNNING` means a
scheduler is actively stepping this sequence's generation.
**Used in:** `engine/sequence.py`
**Source:** OS process scheduling (owner's own coursework); applied here per
`docs/plans/m02-continuous-batching-plan.md` Phase 0.

### Prefill vs. decode
Prefill processes a sequence's whole prompt in one parallel forward
pass — compute-bound, since every prompt token's attention can be computed
simultaneously. Decode generates one token at a time — memory-bandwidth
bound, since each step still has to read the full model weights from GPU
memory for just one new token's worth of compute. This is *why* they're
separate forward calls, and why decode is the one that benefits from
batching multiple sequences together (it amortizes that same memory read
across more useful work).
**Used in:** `engine/model_runner.py` (`prefill_batch` / `decode_step`)
**Source:** Roadmap Two, Phase 2; `docs/plans/m02-continuous-batching-plan.md`.

### True greedy decoding vs. a checkpoint's decoding defaults
`model.generate(..., do_sample=False)` is greedy *sampling* — it still
applies whatever logits processors the checkpoint's `generation_config.json`
sets by default (e.g. repetition penalty), which are deterministic but not
part of "argmax the raw logits." A hand-rolled decode loop that skips those
processors computes a genuinely different function, even though both are
reproducible. Worth checking a checkpoint's `generation_config` before
trusting it as ground truth for a from-scratch decode loop.
**Used in:** `tests/test_correctness.py`
**Source:** discovered debugging `docs/adr/0008-pin-repetition-penalty-for-true-greedy.md`.

### Iteration-level scheduling (continuous batching)
A scheduler re-decides who's in the running batch on *every* decode step, not
once per batch: a finished sequence is evicted and its slot freed the instant
it stops, and a waiting sequence can be admitted the very next step — instead
of a fixed batch running at the pace of its slowest member. The mechanical
consequence for a KV-cache implementation that only supports one shared
write-position per batch (like `StaticCache`): membership changing means the
cache has to be rebuilt via a resync prefill (ADR-0007), not patched in place.
**Used in:** `engine/scheduler.py`
**Source:** Orca paper, via the Anyscale continuous-batching writeup;
`docs/plans/m02-continuous-batching-plan.md` Phase 2.

### Floating-point non-associativity across recomputation paths
Two mathematically-equivalent computations (a bulk-reprocessed KV history vs.
an incrementally-cached one; a padded `StaticCache` buffer vs. an
exactly-sized `DynamicCache`) can round differently and, at a genuine
near-tied logit, flip which token wins greedy argmax — deterministic on
either path alone, just not equal to each other. Not a MicroServe bug: caused
by GPU's cuBLAS shape-dependent GEMM algorithm selection and SDPA's
fused-kernel shape sensitivity, neither of which any PyTorch flag disables
without losing the acceleration built on top of it.
**Used in:** `tests/test_correctness.py` (`GATE_KWARGS`), `engine/model_runner.py`
(`attn_implementation` parameter)
**Source:** `docs/adr/0010-correctness-gate-cpu-plus-eager-attention.md`;
deep-dive in `docs/bites/gpu-and-hardware/floating-point-determinism.md`.

### Static batching's collapse under variable output length
A fixed-size batch can't admit new work until every one of its members is
done — a sequence that finishes early doesn't free its slot, it just sits
there still getting a real forward pass every step, its output discarded.
Measured here as an exact `wasted_compute_fraction` (steps spent on an
already-finished row ÷ total steps) rather than a wall-clock idle-time
estimate — deterministic and reproducible where a timer wouldn't be. In the
M2 benchmark: continuous batching's fraction was 0% (it evicts and admits
immediately); static's was 46.3%, on the same workload, same total tokens.
**Used in:** `bench/benchmark.py` (`run_scheduled`)
**Source:** Orca paper's stated failure mode for static batching;
`docs/benchmarks/2026-08-23_09-18-14-m2-continuous-vs-static-vs-naive/summary.md`.

### Free list (free-frame list)
A stack of unused block ids, maintained instead of searched for. Turns "find a
free block" from an O(num_blocks) scan into an O(1) pop — which matters because
allocation happens on admission *and* every time a running sequence crosses a
block boundary during decode. LIFO rather than FIFO so the block handed out is
the most recently touched one, and therefore the warmest; flipping to
`popleft()` is a one-character debugging aid that maximises the window before a
freed block is reused, turning use-after-free from silent corruption into
visible staleness. The free list is *derived* — `ref_count` is the source of
truth about whether a block is allocated, which is why a `set` free list (whose
`add()` silently ignores duplicates) would make double-free undetectable.
**Used in:** `engine/kv_cache.py` (`BlockAllocator.free_list`)
**Source:** OSTEP Ch 18–19 free-frame list; deep-dive in
`docs/bites/data-structures/free-lists-and-block-allocation.md`.

### Reference counting for block reclamation
One integer per block: incremented on acquire, decremented on release, returned
to the free list only at zero. Always 0 or 1 in M3 (a block has one owning
sequence), and the mechanism exists this early so M6's prefix sharing is an
extension rather than a rewrite. Chosen over tracing GC because reclamation must
be *deterministic* — a pool run at 98% utilization has no room for "eventually" —
and ref counting's one real weakness, uncollectable cycles, is structurally
impossible here since blocks never reference other blocks. The counter is also
the only thing that makes double-free and leaks detectable; the invariant worth
asserting is `num_free + count(ref_count > 0) == num_blocks`.
**Used in:** `engine/kv_cache.py` (`BlockAllocator.ref_counter`, `free()`)
**Source:** OSTEP Ch 39 (inode/fd counts), Linux `struct page._refcount`;
deep-dive in `docs/bites/data-structures/reclamation-strategies.md`.

### Scatter / gather

Two halves of one idea: writing values to *non-contiguous* addresses computed
from an index vector (scatter), and reading them back into a contiguous buffer
(gather). In MicroServe the scatter writes a step's new K/V into whichever
blocks a sequence happens to own; the gather rebuilds its history into the
rectangle SDPA requires.
**Used in:** `engine/paged_cache.py` — `scatter_index()` and `gather_kv()`.
**Source:** Horace He, *Making Deep Learning Go Brrrr From First Principles*
(memory-bandwidth framing); `scratch/m03_paged_kv/04_block_table_and_gather.py`
for the worked shapes.

### Gather-then-attend

The no-kernel realization of PagedAttention: rather than teaching the attention
kernel to read scattered blocks (vLLM's approach), copy the blocks into a
contiguous scratch tensor first and call stock `scaled_dot_product_attention`.
Proves the paging idea without kernel authorship, at the cost of one extra copy
per layer per step — real memory traffic that a rectangular cache does not pay,
and the trade M3 has to state rather than hide.
**Used in:** `engine/paged_cache.py` — `PagedLayer.update()`.
**Source:** PagedAttention paper §4 (for what the copy replaces); locked
decision 3 in `CLAUDE.md` for why we don't write the kernel.

### Memory traffic vs. memory capacity

Two distinct and opposed memory costs. Capacity is how much fits — the thing
paging improves, and what caps concurrent sequences. Traffic is how many bytes
move per step — the thing the gather makes *worse*. M3 spends the second to buy
the first, which is only a good trade because decode is bandwidth-bound on the
*weights*, and the KV gather is small beside them until batch sizes get large.
**Used in:** the Phase 6 benchmark's framing; `engine/paged_cache.py` module
docstring.
**Source:** Horace He, *Making Deep Learning Go Brrrr*.

### Ragged-batch indexing

Turning per-row counts into flat index vectors without ever materialising the
padded rectangle. `repeat_interleave(arange(B), counts)` gives a row label per
element; `arange(N) - (cumsum(counts) - counts).repeat_interleave(counts)`
restarts a column counter at 0 on each row boundary. The pair is the standard
way to address a variable-length batch, and here it is what guarantees a padded
query column is never written into the pool.
**Used in:** `engine/paged_cache.py` — `scatter_index()`.
**Source:** the same idiom underlies `nested_tensor` and FlashAttention's
`cu_seqlens` (cumulative sequence lengths) argument.

### Advanced indexing with a slice in the middle

When a tensor is indexed with two advanced (tensor/integer) indices separated
by a `slice`, NumPy and PyTorch move the broadcast result dimensions to the
*front* rather than leaving them in place. `pool[block_table, :, layer_idx]`
with `block_table` of `[B, nblk]` therefore yields `[B, nblk, 2, H, ...]`, not
`[2, B, nblk, H, ...]`. Getting this backwards produces a tensor of the right
size and the wrong meaning.
**Used in:** `engine/paged_cache.py` — `gather_kv()`, and the write in
`PagedLayer.update()`.
**Source:** NumPy docs, *Advanced indexing — combining advanced and basic
indexing*.

### Preemption by recomputation

Making room in a full KV cache by evicting a running sequence entirely: free
its blocks, discard its KV, keep its tokens, and re-prefill prompt + generated
when it is readmitted. The alternative is *swapping* — copying its blocks to
host memory and back. They trade opposite currencies: recomputation spends
compute to avoid moving bytes, swapping spends interconnect bandwidth and a
good deal of new machinery to avoid recomputing.
**Used in:** `engine/paged_scheduler.py` — `_preempt_newest`, `_grow_running`.
**Source:** ADR-0012; PagedAttention paper, §4.5 (scheduling and preemption).

### Head-of-line blocking

A queue where the item at the front holds up everything behind it regardless of
whether they could proceed. Here it is prefill-priority scheduling: a step
spent on a newly admitted sequence's prefill is a step every running sequence
does not decode, so arrivals raise TTFT for themselves at the cost of TPOT for
everyone already in flight. Chunked prefill is the standard fix.
**Used in:** `engine/paged_scheduler.py` — `step()`.
**Source:** ADR-0013; the term is from networking (switch input queues).

### Memory-pressure-bound admission

Admitting requests against free memory rather than against a fixed concurrency
cap. The consequential difference from a slot cap is that memory availability
can stop being true *after* admission, because running sequences keep growing —
so it requires a preemption policy, which a slot cap never does.
**Used in:** `engine/paged_scheduler.py` — `_can_admit`, `_admit`.
**Source:** ADR-0012, ADR-0013.

### Reserve-max

Sizing a KV cache row for the longest sequence it could ever hold -- prompt
plus `max_new_tokens` -- and holding that memory whether or not a token ever
lands there. It is what a rectangular `StaticCache` does by construction, and
it is the specific waste paging removes. On a long-tailed workload it leaves
occupancy around 40%; the paged engine measures 90% on the same requests.
**Used in:** `bench/paged_benchmark.py` -- `plan_for_budget`,
`reserved_slot_occupancy_mean`; the contrast is the point of M3.
**Source:** ADR-0014; PagedAttention paper, Figure 2.

### KV-cache memory utilization

Token slots holding a real token divided by token slots the engine is holding.
The denominator is the whole question: against a *fixed budget* it answers "how
much of the VRAM I paid for is working", against *what the engine reserved for
its live sequences* it answers "how wasteful is the reservation strategy". Both
are reported in M3's benchmark because one number cannot carry both questions.
**Used in:** `bench/paged_benchmark.py` -- `kv_utilization_*` and
`reserved_slot_occupancy_mean`.
**Source:** ADR-0014; PagedAttention paper, §6.2.

### Adversarial test design

Choosing test inputs against the *mechanism* rather than against typical usage.
M3's gate needed prompts of exactly `BLOCK_SIZE` and `BLOCK_SIZE + 1`, and a
sequence reusing a dead one's blocks -- none of which a realistic prompt would
hit reliably, and all of which are where block indexing actually breaks. A test
suite made of representative inputs measures the common case; one made of
boundary cases measures the code.
**Used in:** `tests/test_correctness.py` -- the M3 Phase 5 section.
**Source:** the plan's Phase 5 rationale.
