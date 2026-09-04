# Progress Log

A dated, terse record of what happened and what was learned. Per `claude.md`:
one entry per non-trivial change, 1–2 lines, tagged with the concept name(s),
linking to an ADR or plan if one applies. No entry for trivial changes — an
entry for any real decision or any moment something worth recalling happened.

Newest entries at the top.

---

## 2026-08-23 — Milestone 2 complete: Phase 4 benchmark, `StaticEngine` cache bug found and fixed

`bench/benchmark.py` rewritten for M2: a mixed-`max_new_tokens` workload (6
prompts, `[8, 60, 12, 50, 10, 40]`) run through all three engines, so static
batching's collapse is actually visible (some requests finish early, some
don't). Records throughput, TTFT/TPOT p50/p99, peak concurrent sequences,
and `wasted_compute_fraction` — an exact slot-step count standing in for
"GPU idle time," chosen over a wall-clock estimate for reproducibility.
Running it for real caught a genuine bug: `StaticEngine._start_batch`'s
`max_cache_len` formula undercounted whenever a short-prompt/long-budget
sequence shared a batch with a longer-prompt/short-budget one (every row
gets left-padded to the batch's longest prompt, then the whole batch steps
for as many iterations as the longest budget needs — not
`max(prompt_len + max_new_tokens)` per sequence). Caught as a hard CUDA
`index_copy_` assertion, not a silent wrong answer; fixed, and
`tests/test_static_engine.py` added as a regression test (fake `ModelRunner`,
no real model). `Engine.add_request` and `ContinuousEngine` also gained
small additive changes (`max_new_tokens` override on the naive engine;
`.running` property) so the benchmark could drive all three engines
identically. Result: continuous batching beats static on both throughput
(47.4 vs. 27.2 tok/s) and wasted compute (0% vs. 46.3%), on identical total
tokens generated — the roadmap's M2 Definition of Done, met.
**Concepts:** static batching's collapse under variable output length.
**Plan:** `docs/plans/m02-continuous-batching-plan.md` Phase 4. **Benchmark:**
`docs/benchmarks/2026-08-23_09-18-14-m2-continuous-vs-static-vs-naive/`.
**Milestone 2 is now complete** — `docs/roadmap.md` checkbox ticked.

## 2026-08-23 — Correctness gate: pinned to CPU + eager attention (ADR-0009 → ADR-0010)

Phase 3's new multi-sequence engine test surfaced a real floating-point
non-associativity issue, not a logic bug: `StaticCache` (our engine's only
cache) is not bit-exact against `.generate()`'s `DynamicCache` in every case,
because SDPA's fused attention kernel is sensitive to padded-vs-exact buffer
shape, and separately, GPU's cuBLAS backend picks different GEMM algorithms
for a bulk-reprocessed history vs. an incrementally-decoded one. First
attempt (ADR-0009, "pin to CPU") was accepted after checking only one
scenario and immediately failed on a *different* prompt for a *different*
reason once run for real — walked back and superseded rather than silently
amended. Full CPU/GPU × sdpa/eager × prompt matrix isolated two distinct
causes: cache-shape sensitivity under SDPA (fixed by eager attention) and
cuBLAS's own shape-dependent algorithm selection on GPU (not fixed by
attention implementation at all — a lower-level, unavoidable GPU trade).
CPU + eager attention was the one configuration where both vanished for
every case tested (ADR-0010, accepted). `ModelRunner`/`Engine`/
`ContinuousEngine`/`StaticEngine` gained an `attn_implementation` parameter
(same pattern as the existing `device` param); `tests/test_correctness.py`
now pins both via a `GATE_KWARGS` dict, applied to every fixture including
the golden reference. Benchmarks untouched — GPU + SDPA remains the
production path. 21/21 tests passing.
**Concepts:** floating-point non-associativity across recomputation paths.
**Bite:** `docs/bites/gpu-and-hardware/floating-point-determinism.md`.
**ADRs:** ADR-0009 (superseded), ADR-0010 (accepted).

## 2026-08-23 — Milestone 2, Phase 2: `Scheduler` — iteration-level admit/decode/evict

`engine/scheduler.py` built — the milestone's core mechanism, and per
`claude.md` not handed over solo: the owner wrote every line themselves in a
Socratic pair-programming session (a separate Claude conversation, handed off
with full project context), then brought it back for review. Two real issues
found at review that the owner's own build log hadn't caught: `self.model_runner`
was bound to the `ModelRunner` class instead of the instance (same
"annotation vs. value" bug family recurring a fourth time), and `step()` on
an idle scheduler with a stale `membership_changed` flag would crash calling
`max([])` on an empty running batch — fixed with an explicit no-op guard.
`tests/test_scheduler.py` added — 12 tests against a `FakeModelRunner` (no
real model needed) verifying admit/evict bookkeeping and, specifically,
*which* forward-step primitive (resync vs. decode) gets called across
admission, steady-state, and eviction transitions — the direct test of
ADR-0007's reason for existing. 19/19 tests passing project-wide. Q1
(admission cap) resolved as a `max_batch_size` constant. Full build/bug/
decision log kept in `docs/achievements/m02-scheduler-build-session.md` —
Python fundamentals revised (dataclass annotations vs. real attributes,
Enum members vs. string literals, mutate-while-iterating, `deque` semantics)
alongside the scheduler-specific design reasoning.
**Concepts:** iteration-level scheduling / continuous batching. **Plan:**
`docs/plans/m02-continuous-batching-plan.md` Phase 2. **Achievement log:**
`docs/achievements/m02-scheduler-build-session.md`.

## 2026-08-22 — Milestone 2, Phases 0–1: `Sequence` states + batched forward-step primitives

`docs/plans/m02-continuous-batching-plan.md` drafted and Phases 0–1 built.
`engine/sequence.py`: `SequenceStatus` gains `RUNNING` (replacing `DONE`
with `FINISHED`), `max_new_tokens` moved per-request onto `Sequence`, plus
`last_token_id`/`is_finished` as derived properties. `engine/model_runner.py`
gains `prefill_batch`/`decode_step` — real batched forward passes via
`transformers.StaticCache`, replacing `.generate()` for every engine from
here on. Two things surfaced mid-build that changed the plan: `StaticCache`
tracks one write-position for its whole batch, so a batch whose membership
changes over time needs a fresh cache each time rather than one long-lived
per-sequence buffer (ADR-0007); and the correctness harness's golden
reference silently depended on Qwen2.5-Instruct's default
`repetition_penalty=1.1`, invisible until an engine stopped calling
`.generate()` — fixed by pinning `repetition_penalty=1.0` everywhere greedy
is claimed (ADR-0008). `tests/test_correctness.py` extended with two new
forward-step tests (single-sequence and batched, different prompt lengths,
to guard against cross-row KV contamination); 7/7 passing.
**Concepts:** sequence lifecycle (OS process-state analogy), prefill vs.
decode as working code, true greedy decoding vs. a checkpoint's decoding
defaults. **Plan:** `docs/plans/m02-continuous-batching-plan.md`. **ADRs:**
ADR-0005, ADR-0006 (design decisions before Phase 0), ADR-0007, ADR-0008
(discovered during Phase 1).

## 2026-08-21 — Fix: benchmark output dirs collided on same-day reruns

`bench/benchmark.py` named run folders by date only
(`YYYY-MM-DD-m1-naive-baseline`), so a second run on the same day silently
overwrote the first's `results.json`. Switched to a full timestamp
(`YYYY-MM-DD_HH-MM-SS-m1-naive-baseline`) so every run gets its own folder.
Updated `docs/benchmarks/README.md`'s naming convention to match.
**Concepts:** benchmark provenance (`claude.md`'s "numbers without provenance
are not quotable").

## 2026-08-19 — Milestone 1: naive baseline engine + benchmark

Built `engine/sequence.py` (`Sequence`, `SequenceStatus`), `engine/engine.py`
(`Engine.add_request()` / `Engine.step()` — one request to completion at a
time, no scheduler), and `bench/benchmark.py` (bare sequential timing script,
per ADR-0004). `tests/test_correctness.py` extended to check the naive
engine's output against the golden reference in addition to `ModelRunner`'s.
First baseline numbers recorded in `docs/benchmarks/`.
**Concepts:** naive/static serving, throughput vs. latency. **Plan:**
`docs/plans/m01-naive-baseline-plan.md`. **ADR:** ADR-0004 (benchmark scope).

## 2026-08-19 — Milestone 0: repo setup, tiny model, correctness harness

`uv` + `ruff` chosen for tooling (ADR-0001); `pyproject.toml`, `.gitignore`
already in place, first `uv sync` pulled CPU-only `torch` by default — fixed
by pinning an explicit `pytorch-cu130` index in `[tool.uv.sources]` to match
the machine's RTX 3050 / driver-reported CUDA 13.1. Picked
Qwen2.5-0.5B-Instruct as the dev model (ADR-0002). Built
`engine/model_runner.py` (thin HF `generate()` wrapper) and
`tests/test_correctness.py` — fixed seed, greedy decode, exact token-match
against a golden HF reference, greedy-only scope per ADR-0003. `LICENSE`
(MIT) added; pre-commit/CI stub deferred, per the plan's own note that it can
be without harm at this stage.
**Concepts:** greedy decoding. **Plan:** `docs/plans/m00-setup-plan.md`.
**ADRs:** ADR-0001, ADR-0002, ADR-0003.

## 2026-08-13 — Project scaffolding: docs reorganized, roadmap written

Restructured `docs/` to match `claude.md`'s repo layout (`adr/`, `plans/`,
`benchmarks/`, plus a new `reference/` for the two source PDFs). Wrote
`docs/roadmap.md` — the merged, checkboxed master plan from
`inference-engineering-roadmap.pdf` and `microserve-build-plan.pdf`. Replaced
the two ad hoc template files with `docs/adr/template.md` and
`docs/plans/template.md`, adapted to this project. Added root `.gitignore` and
`.claudeignore`.
**Concepts:** none yet — no code written. **Next:** break Milestone 0 (setup)
into an implementation plan in `docs/plans/` before writing any code.

## 2026-08-30 — M3 Phase 1: pool, sizing arithmetic, and the allocator's spec

Resolved the two questions gating Phase 1 (ADR-0011): `engine/kv_cache.py`
holds the pool and `BlockAllocator` with **no `transformers` import**, while
Phase 3's HF `Cache` glue goes in a separate `engine/paged_cache.py` — the
split falls on the ownership line and the risk line at once, and it makes
`tests/test_kv_cache.py` model-free by import graph rather than by discipline.
Pool size is an explicit `num_blocks` with `blocks_for_gib()` as a sizing aid;
a vLLM-style VRAM fraction was rejected because Phase 6's sweep holds VRAM
fixed, and a self-sizing pool would make a recorded number unreproducible.
Landed `KVSpec` (the six numbers + byte arithmetic — 12 KiB/token, 192 KiB per
16-token block for Qwen2.5-0.5B bf16), `PagedKVPool` (one `torch.zeros`,
never `empty`), and `tests/test_kv_cache.py`: 11 green, 15 red against a
`NotImplementedError` plus a 2000-step fuzz that re-checks `num_free` and every
ref count after every operation. **`BlockAllocator` itself is unwritten and
deliberately so** — handout at `docs/handouts/m03-phase1-block-allocator.md`.
**Concepts:** free-frame list, arena allocation, internal fragmentation,
benchmark provenance. **Plan:** `docs/plans/m03-paged-kv-cache-plan.md`.
**ADRs:** ADR-0011.

## 2026-08-30 — M3 Phase 1 complete: `BlockAllocator` written and reviewed

Owner implemented `BlockAllocator` (`deque` used LIFO as the free list, plain
list for ref counts, capacity checked before mutation in `allocate`, ref count
as the double-free detector). All 28 spec tests green on first run. Review
found two holes the tests hadn't covered, both now fixed and both now tested:
`free()` validated inside its mutation loop, so `free([live, bogus])` reclaimed
the live block and *then* raised — the same partial-mutation flaw `allocate()`
correctly guards against, and the more dangerous direction, since it puts a
running sequence's block back in circulation. And `free()` assigned `0` rather
than decrementing, which is indistinguishable while counts are 0-or-1 and
silently reclaims a shared prefix block the moment its first owner finishes in
M6 — the exact scenario the ref counter exists to survive. Validation is now
`Counter`-based so duplicate ids within one call can't both pass against the
same pre-mutation count. Verified the new tests fail against the original
implementation before accepting them.
**Concepts:** free list, reference counting (both added to `concepts.md`).
**Bites:** new `docs/bites/data-structures/` category — five bites on
technique→structure selection (index, free lists, scheduler queues, indexing
structures, reclamation strategies).
**Next:** Phase 2, `Sequence.block_table` and the growth rule.

## 2026-08-30 — M3 Phase 2 scaffolded: block table, BlockManager, growth-rule spec

Two interface decisions taken before writing anything. A thin `BlockManager`
owns the allocator and the growth rule, rather than the scheduler calling
`BlockAllocator` directly — otherwise "does this sequence need another block?"
gets expressed once at admission and again in the decode path, and those two
copies drift. And `Sequence` gains an explicit `num_cached_tokens` rather than
deriving cached length from `len(prompt) + len(output)`: preemption by
recomputation (Phase 4) drops the KV while keeping the tokens, and prefix
sharing (M6) gives a sequence KV for tokens it never computed — neither state
is expressible without the field. Settled the consequence explicitly:
`BlockManager`'s methods are about **capacity, not content**, so the caller
advances the counter after a successful forward pass and the invariant is
`0 <= num_cached_tokens <= len(block_table) * BLOCK_SIZE`. `append_slot` is
therefore idempotent, and `free()` doubles as the preemption path.
Scaffolded `logical_block` / `slot` / `blocks_needed` / `BlockManager` as
stubs and `internal_fragmentation()` as a real metric; wrote 26 tests
(29 Phase 1 tests still green). Handout at
`docs/handouts/m03-phase2-block-table.md`.
**Concepts:** block table as a single-level page table; logical vs. physical
address space; internal fragmentation.
**Next:** owner writes the six functions.

### 2026-08-31 — M3 Phase 2 complete: address translation and the growth rule

All six functions written; `tests/test_kv_cache.py` is 55/55 and the suite is
77/77, so the M2 correctness gate survived adding two fields to `Sequence`.
The growth rule lives in exactly one place — `BlockManager._ensure_capacity`
— with `allocate_for_prefill` and `append_slot` as the same call at different
token counts, which is the fix for the drift the plan warned about. Two
consequences worth remembering: `append_slot` needs `num_cached_tokens + 1`
(capacity for position *p* means capacity for *p+1* tokens), and idempotence
falls out for free rather than needing a guard. `allocate_for_prefill`
reserves the prompt only, no generation headroom — headroom would be
reserve-max wearing a block table. Cleared a stray `from tarfile import
BLOCKSIZE` next to `BLOCK_SIZE`.
**Concepts:** address translation; single-level page table; internal
fragmentation.
**Next:** Phase 3 — the HF `Cache` subclass (scatter/gather against the pool).
Phase 4 is still gated on Q1 and Q3.

### 2026-08-31 — M3 Phase 3 scaffolded: `PagedCache` and the two index functions

Four decisions, then the glue. Per-layer gather over all-layers-once (peak
scratch 1/24th, and scratch competes with the pool for exactly the VRAM M3
reclaims); batched prefill over one-at-a-time (so M3-vs-M2 compares memory
strategy, not prefill batching); right-padded rows in the paged path, because
a paged sequence's KV starts at logical position 0 whatever the query padding
— left-padding would set the block table against the causal mask, and the
price is `PagedBatch.last_logit_index()` replacing `logits[:, -1, :]`; and a
two-tier gate.

Wrote `engine/paged_cache.py` — `PagedBatch` (the per-step descriptor: block
tables, past lens, real query lens, and the mask/position-id/logit-index
tensors derived from them), `PagedLayer`, `PagedCache` — plus
`ModelRunner.paged_prefill` / `paged_decode_step`. `scatter_index` and
`gather_kv` are stubs; 16 of 24 tests are red on them.

Verified the harness before handing it over: with a throwaway reference
implementation of the two stubs, single-sequence *and* batched paged
generation are token-identical to HF `generate()`. So a red test now is index
math, not plumbing. The batched check also de-risks Phase 4 early — the
per-row attention mask is what carries rows sitting at different absolute
positions, since HF builds its causal component from one shared
`cache_position`.
**Concepts:** scatter/gather; gather-then-attend; memory traffic vs. memory
capacity; the `Cache`/`CacheLayerMixin` contract.
**Next:** owner writes `scatter_index` and `gather_kv`. Handout at
`docs/handouts/m03-phase3-paged-cache.md`.

## 2026-09-03 — M3 Phase 3 complete: scatter_index and gather_kv

Wrote both index functions; **101 tests green**, including both Tier 2 gate
prompts and the untouched M1/M2 correctness gate. Ruff clean.

Split-of-work change, recorded deliberately: the plan reserved these two
functions for the owner as the milestone's core mechanism. The owner handed
them over — time pressure, coding skills already re-sharpened — swapping
authorship for review. `docs/handouts/m03-phase3-paged-cache.md` is superseded
but kept: it is the written argument for why these two functions *are*
PagedAttention-without-a-kernel, and so the right thing to read before
reviewing them.

Three implementation choices worth knowing:
- `scatter_index` is the ragged-batch idiom (`repeat_interleave` for the row
  label, `arange` minus an exclusive prefix sum for the column), so the padded
  rectangle is never materialised and padded columns cannot be written.
- `gather_kv` leaves dim 1 as a slice, so **K and V come out of one gather**,
  not two — they are adjacent in the pool by construction, so one pass touches
  each cache line once. The layer is indexed in the *same* advanced-indexing
  op as the blocks; `pool[bt][..., layer]` would materialise all 24 layers and
  discard 23.
- The permute precedes the reshape, for the reason the whole phase turns on:
  reshape walks row-major, so folding the block axis into time while heads sit
  between them interleaves heads into time. Silent — every value present,
  every shape correct, the tokens wrong.
**Concepts:** scatter/gather; gather-then-attend; ragged-batch indexing.
**Next:** Phase 4 — scheduler integration. Two open questions gate it (Q1:
separate `PagedScheduler` vs. generalizing M2's; Q3: block exhaustion
mid-generation), and ADR-0012 follows Q3.

## 2026-09-03 — M3 Phase 4 complete: admission gated by memory, preemption by recomputation

`PagedScheduler` + `PagedEngine`, the fourth engine. **122 tests green**, ruff
clean. Two ADRs written: **ADR-0012** (preemption) and **ADR-0013** (scheduler
structure and step shape).

The new thing in this codebase: a constraint that can stop being true *after*
admission. M2's `max_batch_size` check, once passed, stays passed. Block
availability does not — every running sequence is still growing, and one decode
step in sixteen needs a new block.

- **Preempt by recomputation, newest-first** (ADR-0012). The hardware settled
  it: recomputation and swapping trade the same currency in opposite
  directions, and on a laptop compute is the abundant side while the PCIe link
  is the scarce one. Prefill is the one high-arithmetic-intensity phase of this
  workload; a host round trip is pure data movement on the slowest link, with
  no arithmetic to amortise it. Swapping would also add a host pool, a host
  allocator, a GPU→host block map, and a transfer in the `step()` critical
  path — the least-tested code in the engine holding the most invariants.
  Newest-first eviction is not fairness, it is what makes the scheme
  *terminate*: the oldest keep their blocks and keep draining the pool.
- **Separate `PagedScheduler`, not a generalized `Scheduler`** (ADR-0013). M5's
  four-way chart needs M2's scheduler byte-identical to the version its numbers
  came from — an already-measured engine is closer to a recorded artifact than
  to live code. And the abstraction did not fit: preemption mutates the running
  set mid-step, so it is control flow, not an admission predicate.
- **Prefill-priority steps.** A step either admits-and-prefills or decodes,
  never both. It trades TPOT for TTFT — head-of-line blocking — and under M5's
  Poisson arrivals a burst will show as a TPOT spike. Chunked prefill is the
  real fix and is out of scope; the trade is named rather than smoothed over.
- **ADR-0007's resync is gone**, and confirmed by test rather than assumed:
  `test_one_sequence_finishing_does_not_disturb_another` asserts no prefill
  call is issued on a survivor's behalf when a neighbour retires. A *separable*
  win of M3 — report it apart from the paging number or the two get conflated.

Tier 1 uses a fake runner and a 4-block pool so the preemption path runs on
every test, not only under memory pressure. Tier 2 shows a
preempted-and-recomputed sequence matching HF `generate()` exactly — which is
what makes preemption a cost rather than a behaviour change.

Per-phase **review notes** sections now live at the bottom of `kv_cache.py`
(Phases 1 and 2), `paged_cache.py` (Phase 3) and `paged_scheduler.py`
(Phase 4): the decisions with a wrong version that still runs.
**Concepts:** preemption; recomputation vs. swapping; head-of-line blocking;
memory-pressure-bound admission.
**Next:** Phase 5 — the correctness gate on the paged path.

## 2026-09-03 — M3 Phase 5 complete: the correctness gate, aimed at the mechanism

`PagedEngine` added to the parametrized multi-sequence case, plus five cases
written *against* the block machinery. **17 tests in `test_correctness.py`.**

The existing gate used short prompts and `max_new_tokens` of 8 and 20 — with
`BLOCK_SIZE = 16` that barely crosses one boundary, so it would have passed
while the indexing was substantially wrong. The new cases: a prompt of exactly
`BLOCK_SIZE`; one of `BLOCK_SIZE + 1`; generation crossing several boundaries;
two sequences whose blocks provably interleave (asserted mid-flight, because
block tables are cleared on retirement).

The one that matters most is
`test_paged_reuses_a_dead_sequences_blocks_without_stale_kv`. A finished
sequence's blocks go back to the free list **still holding its KV** — `free()`
deliberately does not zero them, since a reused block is always fully
overwritten before it is read. The next sequence is handed those same physical
blocks and must still be exactly golden. That is the nastiest bug this
milestone can produce and the one most likely to look like "the model got a bit
worse" rather than like a failure. Reuse is *asserted*, not assumed, so an
allocator change cannot leave the test passing while testing nothing.
**Concepts:** adversarial test design against a known failure mode.

## 2026-09-03 — M3 Phase 6 complete: 4.92x concurrent sequences at equal VRAM

**The milestone's headline. 122 tests green.** Numbers in
`docs/benchmarks/2026-09-03_15-57-52-m3-paged-vs-contiguous/`, methodology in
**ADR-0014**.

| | contiguous | paged |
|---|---|---|
| peak concurrent sequences (64 MiB) | 26 | **128** |
| reserved-slot occupancy | 42% | **90%** |
| throughput | 163 tok/s | 466 tok/s |
| TTFT p99 | 38.6 s | 0.46 s |

The occupancy number *explains* the concurrency number — it is not independent
corroboration. A contiguous cache reserves a rectangle sized for the longest
sequence in the batch, so on a long-tailed workload more than half of what it
holds is reserved against tokens that never arrive.

Three things recorded so they cannot be misquoted later:

- **4.92x is a floor.** The paged engine peaked at 128 — every request in the
  workload — with zero preemptions, so its real ceiling at 64 MiB was never
  reached. The contiguous engine *was* genuinely saturated at 26.
- **The TTFT result is not paging.** 38.6 s → 0.46 s is the absence of
  ADR-0007's resync: the contiguous engine re-prefills its whole batch on every
  membership change, so a new arrival queues behind repeated full-batch
  prefills. Separable win, separately reported.
- **Flat TPOT does not mean the gather is free.** At 128 concurrent sequences
  the per-layer gather is small next to weight traffic. It would show at long
  context, where the gathered rectangle grows with history.

Two methodology findings, both in ADR-0014. "Equal VRAM" is not a setting
either engine has, so a byte budget is converted into each engine's own
currency — and the contiguous derivation is *verified* against the tensors
`StaticCache` actually allocates rather than trusted. And the plan's workload
could not exhibit the effect the plan wanted to measure: six requests cap peak
concurrency at six, so above 3 MiB the ratio collapses to 1.0x. That is a fact
about the workload, not about paging. Hence two workloads — the sweep for
continuity with M2, a 128-request validation point for the actual claim.
**Concepts:** KV-cache memory utilization; reserve-max; capacity vs. bandwidth.

## 2026-09-03 — Milestone 3 complete

Every phase done, all six roadmap build boxes and both Definition-of-Done boxes
ticked. Theory boxes left deliberately unticked — those record having read
something, which is not mine to claim.

`python -m engine.paged_engine` runs a five-scene demonstration: the pool,
address translation, two sequences' block tables interleaving, a live
preemption, and the preempted sequence's output matching HF exactly. Written
because the mechanism should be *visible*, not only asserted in a test file.

ADRs from this milestone: **0011** (module split, pool sizing), **0012**
(preemption by recomputation over swapping — the laptop's PCIe link is the
scarce resource, not compute), **0013** (separate `PagedScheduler`,
prefill-priority steps), **0014** (equal-VRAM benchmark methodology).
**Next:** M4 — quantization. Two things to carry forward: re-verify Phase 0
findings 3 and 4 under SDPA before quoting any GPU number from the paged path
(they were established on CPU + eager), and the headline 7B-class model is
chosen at M4, not before.

## 2026-09-03 — M3 plan annotated with results; one suspected defect flagged

`docs/plans/m03-paged-kv-cache-plan.md` now carries a **How to run this
milestone** section (tests, the `python -m engine.paged_engine` demo, the exact
benchmark command) and a **Phase 6 — results, read honestly** section that
tabulates the run and reads it against what the plan predicted.

The prediction held with its sign: **2.86x system throughput = 4.92x
concurrency x 0.58x per-sequence rate.** Paging bought sequences and gave back
per-sequence decode rate, exactly as Phase 6's preamble said it would.

Two things the run README under-reported, now written down. (1) At *equal*
concurrency the sweep isolates paging's cost: **TPOT 1.6-1.7x worse, throughput
0.58-0.62x.** "Flat TPOT" was true only at the validation point. Paging is a
straight loss below ~19 concurrent sequences. (2) **A paged step does 21x the
work from 6 to 128 sequences for +9% time** — nearly-flat step latency means
fixed overhead dominates, ~37 ms/step of it at batch 6 where the gather moves
almost no bytes. Four candidates named (pageable H2D per step in
`block_table_tensor` and `scatter_index`; the double copy in `gather_kv` from
`permute().reshape()`; 24 small per-layer gathers) with the profile that would
settle it. **Not fixed** — flagged for a decision. It matters for M4:
quantization shrinks the weight traffic this overhead currently hides behind.
**Concepts:** arithmetic intensity; fixed overhead vs. bandwidth.

## 2026-09-03 — `docs/performance-model.md`: the numbers, connected to mechanisms

One document tying every measured figure to the mechanism that produced it,
because the run README and four ADRs each held a piece and none held the chain.
Ten sections, each a lever: KV bytes per token (GQA is a **7x** divisor, the
largest constant in the budget), reserve-max, paged allocation,
gather-then-attend, the resync, prefill-priority, preemption, `BLOCK_SIZE`, and
two configuration traps. Ends with a what-moves-what table and a ranked list of
what to move next.

The two questions it exists to answer plainly. **Why TTFT fell 84x:**
`StaticCache` has one shared write position, so any membership change forces a
re-prefill of the whole batch — ~128 of them across the run. Per-sequence block
tables remove the *redundancy*, not the prefill. **What gather-then-attend
costs:** with the concurrency win removed, TPOT 1.68x worse and throughput
0.58x — paging is a straight loss below ~19 concurrent sequences and a large
win above it.
**Concepts:** arithmetic intensity; reserve-max; capacity vs. bandwidth.

