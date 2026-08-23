# M02 — Continuous batching — Implementation Plan

**Status:** ✅ Complete — all four phases built, verified, and benchmarked.
**Milestone:** M2 (see `docs/roadmap.md` for the milestone's build spec and
guardrail).
**Scope:** Real `Sequence` states (waiting/running/finished), a `Scheduler`
implementing iteration-level scheduling, prefill and decode as separate
forward calls, and a simple per-sequence contiguous KV buffer (no paging
yet).
**Guardrail (copied from the roadmap, so it's visible while building):** Do
not implement paged or block-based memory yet — accepting fragmentation here
is correct, not lazy; it isolates the scheduling problem so Milestone 3 can
isolate the memory problem. Do not attempt to fuse prefill and decode into
one kernel call.

---

## Reading required before this milestone

From `docs/roadmap.md`'s Milestone 2 theory gate — read these before Phase 0,
not before (per `claude.md`'s "just in time" rule):

1. **Prefill vs. decode** (Roadmap Two, Phase 2). Prefill processes the whole
   prompt in one parallel pass — compute-bound. Decode generates one token at
   a time — memory-bandwidth bound. This is *why* they get separate code
   paths in the scheduler, not a stylistic choice.
2. **The KV cache size formula, worked by hand**: `2 × num_layers ×
   num_heads × head_dim × seq_len × batch_size × bytes_per_param`. Run it for
   a 7B, 13B, and 70B model at 4K/32K context, batch 1/32, fp16 — watch the
   numbers stop being "large" and become "larger than the GPU." This is the
   number the per-sequence KV buffer in Phase 1 is a concrete instance of.
3. **The Orca paper**, via the Anyscale continuous-batching writeup
   (`anyscale.com/blog/continuous-batching-llm-inference`) — iteration-level
   scheduling, and why static batching collapses under variable output
   lengths. This is the paper Phase 2's `Scheduler` and Phase 4's benchmark
   both exist to reproduce the finding of.

---

## The core mental model this plan is built around

> A GPU sitting idle while it waits on one sequence's decode step is wasted
> compute another queued sequence could have used. Continuous batching
> interleaves many sequences' decode steps and re-evaluates who's in the
> batch at *every single iteration* — not once per batch — so a finished
> sequence's slot frees the instant it finishes, not at the end of the
> slowest sequence sharing its batch.

Everything below either builds this idea (the `Scheduler`) or exists to let
it be measured honestly (the static-batch comparison engine, the mixed-length
benchmark workload). Nothing else belongs in this milestone.

---

## Locked decisions this plan honors

1. `engine.add_request(prompt)` / `engine.step()` stays the public API
   surface (locked in M1). `step()`'s *meaning* shifts, though — M1's plan
   already flagged this: from M2 onward, one `step()` call does one decode
   iteration across possibly many sequences, not a full generation. That
   shift lands in this milestone.
2. `engine/` stays synchronous and framework-free — no async, no serving
   layer (Act Two, out of scope here).
3. No paged or block-based memory yet (this milestone's guardrail, above).
4. Prefill and decode run as two separate forward calls, never fused (this
   milestone's guardrail, above) — note this simplification in the README
   once built, per the roadmap's build spec.
5. **The KV cache is owned and pre-allocated, not delegated to
   `transformers`'s auto-growing `DynamicCache`.** *Why, and why now:* this
   was an open design fork this session — the alternative (each sequence
   holding its own HF `DynamicCache`) is less code today but defers "does
   this look like M3's block table" entirely to M3, which would then likely
   need to rewrite this layer rather than extend it. See ADR-0005. **Refined
   during Phase 1 (ADR-0007):** "owned per sequence" turned out to mean
   *per running batch*, not one buffer object per `Sequence` — `transformers`'s
   `StaticCache` (the pre-allocated class matching ADR-0005's intent) tracks
   one write-position shared across its whole batch, so it's rebuilt fresh
   whenever the running batch's membership changes (an admission or an
   eviction), covering every currently-running sequence's full history at
   once. Between membership changes, decode is one real batched call using
   that `StaticCache` unmodified.
6. **Three engines exist after this milestone**: naive (M1, untouched),
   static-batch (new, small, built this milestone), continuous-batch (new,
   the actual deliverable). *Why a real static-batch engine, not reusing
   M1:* M1's naive engine runs one request at a time — it was never
   batching at all, so comparing continuous batching against it would
   overstate the win and wouldn't be the comparison Orca's paper (or this
   milestone's own DoD) actually means by "static batching." A fixed-size
   batch that waits for its slowest member before admitting new work is the
   real point of contrast, and it shares almost all its machinery with the
   continuous engine (same forward-step primitive from Phase 1), so it's a
   small add, not a second milestone. See ADR-0006.

---

## What already exists (so we don't redo it)

- `engine/sequence.py` — bare `Sequence` (`WAITING`/`DONE` only, from M1).
  This milestone replaces the status enum and adds scheduler-facing fields.
- `engine/engine.py` — M1's naive one-at-a-time engine. Left untouched by
  this milestone; it becomes the "before" line on the M2 comparison chart.
- `engine/model_runner.py` — currently only exposes `encode`/`decode`/
  `generate()`, where `generate()` wraps HF's own full-generation loop and
  internal cache. M2 cannot drive interleaved multi-sequence stepping through
  `generate()` — it needs a lower-level single-forward-pass primitive
  (Phase 1).
- `tests/test_correctness.py` — the gate. Currently exercises only M1's
  single-sequence, one-at-a-time path; this milestone extends it to the two
  new engines with multiple concurrent sequences.
- `bench/benchmark.py` — M1's bare sequential timing script, run against a
  small fixed prompt set with uniform `max_new_tokens`. That workload can't
  demonstrate what continuous batching fixes (nothing would ever finish
  early enough to matter) — Phase 4 needs a workload with mixed output
  lengths.

---

## Phase 0 — `Sequence` state machine  ✅

**Goal:** `Sequence` carries the state a scheduler needs to make an
admit/evict decision every iteration.

**Why this phase:** M1's `Sequence` was deliberately bare — WAITING/DONE
was itself evidence of what M1 was missing (no notion of "paused mid-
generation to let another sequence run"). This phase is that missing piece.

- ✅ `SequenceStatus`: `WAITING`, `RUNNING`, `FINISHED` — replacing
  `WAITING`/`DONE`. Maps directly onto the OS process-state model (ready /
  running / terminated) the owner already knows from OS coursework — same
  idea, one fewer state than a full OS scheduler needs since there's no
  "blocked on I/O" equivalent here.
- ✅ `max_new_tokens` moved from `Engine`-wide onto each `Sequence` (owner's
  call — every request can now ask for its own generation length, via
  `add_request(prompt, max_new_tokens=...)` once Phase 3 wires it up).
- ✅ `last_token_id` and `is_finished` as *properties*, not stored fields —
  derived from `output_token_ids`/`status` rather than duplicated state a
  scheduler could accidentally let drift out of sync.
- **No KV-buffer field** — Phase 1 (ADR-0007) found the buffer belongs to
  the running *batch*, not to an individual sequence, so there's nothing for
  `Sequence` to hold a handle to.

**Files:** `engine/sequence.py`. `engine/engine.py` needed two
mechanical, behavior-preserving edits so M1's naive engine still compiles
against the new `Sequence` (`DONE` → `FINISHED`; pass
`max_new_tokens=self.max_new_tokens` when constructing a `Sequence`) —
M1's own logic is unchanged.
**Concepts:** sequence lifecycle as a state machine; the OS process-states
analogy (name it explicitly — don't re-derive it from scratch)
**Done when:** a `Sequence` can move through all three states and a
scheduler could read its state to decide what to do with it next.
**Test:** none needed on its own — exercised by Phase 3's correctness tests.
**Verified:** `tests/test_correctness.py` (4/4 at the time) still passes
against M1's naive engine after the mechanical edits — confirms the state
rename didn't change M1's behavior.

---

## Phase 1 — `ModelRunner` batched forward-step primitives  ✅

**Goal:** primitives that run one real forward pass — a batched prefill/
resync, or a batched single-token decode step — across multiple sequences at
once, replacing `model.generate()`. Both engines in Phase 3 are built on
this.

**Why this phase:** neither new engine can use `model.generate()` anymore —
interleaving requires stopping after every single token to let the scheduler
re-evaluate who's in the batch. This turned out to be more than plumbing —
see ADR-0007 for why plain per-sequence buffers don't batch correctly, and
ADR-0008 for a correctness-harness bug this phase's debugging surfaced.

- ✅ `ModelRunner.prefill_batch(token_id_lists, max_cache_len)` — left-pads a
  batch of full token histories to a common width, runs one forward call,
  returns each row's greedy next token, a fresh `StaticCache` sized for this
  batch, and the attention mask to extend on later calls. This is the
  resync call from ADR-0007: called on every batch membership change,
  covering *every* currently-running sequence's full history so far, not
  just the newly-admitted ones.
- ✅ `ModelRunner.decode_step(cache, last_token_ids, attention_mask)` — one
  batched forward call, exactly one new token per row, `cache` updated in
  place by `StaticCache` itself. Cheap; used for every iteration between
  membership changes.
- ✅ Left-padding + position-id helpers (`_left_pad`,
  `_position_ids_from_mask`) so real tokens are numbered correctly
  regardless of how much padding precedes them, and padded columns are
  masked out of attention.

**Two things discovered while building this, not in the original plan:**
1. **ADR-0007** — `transformers` 5.15.1 requires `past_key_values` to be a
   `Cache` object; the obvious pre-allocated choice (`StaticCache`) tracks
   one write-position for its *whole* batch, so it can't be reused
   as-is across a batch whose membership changes over time (a naive
   sequence's real content would land at the wrong column). Resolved by
   rebuilding `StaticCache` fresh on every membership change instead of
   writing custom per-row indexing.
2. **ADR-0008** — the correctness harness's golden reference silently
   depended on Qwen2.5-Instruct's default `repetition_penalty=1.1`
   (`.generate()` applies it even with `do_sample=False`). Invisible through
   M0–M1 because `model_runner.generate()` also just calls `.generate()`;
   would have silently broken every future milestone's correctness
   comparison once engines stopped calling `.generate()` at all. Fixed by
   pinning `repetition_penalty=1.0` in both the golden reference and
   `ModelRunner.generate()`.

**Files:** `engine/model_runner.py`. No new module and no `Sequence` field
needed — see Open Question Q2, resolved by ADR-0007's finding that the
buffer belongs to the batch, not the sequence. `tests/test_correctness.py`,
`engine/model_runner.py` (2 lines) for ADR-0008's fix.
**Concepts:** prefill vs. decode as genuinely different forward calls, now
as working code; the KV cache size formula as a concrete `StaticCache`
allocation; true greedy decoding vs. a checkpoint's default logits
processors (ADR-0008)
**Done when:** a single sequence can be prefilled and then decoded one token
at a time through this primitive, producing the same tokens `generate()`
would have.
**Test:** `tests/test_correctness.py` gained
`test_forward_step_single_sequence_matches_golden` (single-sequence,
parametrized over both prompts) and `test_forward_step_batched_matches_golden`
(both prompts run through one shared, left-padded batched call — the test
that would catch a row reading another row's padding or KV state).
**Verified:** all 7 tests pass (`uv run pytest tests/test_correctness.py -v`)
— 4 from M0/M1 plus the 3 new Phase 1 cases, all exact token-for-token
matches against the (corrected) golden reference.

---

## Phase 2 — `Scheduler`: iteration-level admission and eviction  ✅

**Goal:** a `Scheduler` owning the WAITING/RUNNING/FINISHED bookkeeping,
deciding on every `step()` call: which `WAITING` sequences get admitted this
iteration, which `RUNNING` sequences get one more decode step, and which
`RUNNING` sequences just finished and get evicted immediately — not at the
end of some larger shared batch.

**This is the milestone's core mechanism**, and per `claude.md` it was not
handed over solo: the owner wrote every line of `Scheduler` themselves in a
Socratic pair-programming session (a separate Claude conversation, since this
was heavy on back-and-forth token-wise), then brought it back here for
review. See `docs/achievements/m02-scheduler-build-session.md` for the full
build log — decisions made, bugs hit, and the reasoning behind each fix.

Responsibilities, as built:
- ✅ Admission policy: FIFO, up to `max_batch_size` (Q1, resolved — a
  constant, not uncapped).
- ✅ **If membership changed this step** (an admission just happened, or the
  *previous* step's eviction left the cache stale): call
  `ModelRunner.prefill_batch` (Phase 1) with every currently-running
  sequence's full history so far — the resync from ADR-0007, rebuilding the
  shared `StaticCache` from scratch.
- ✅ **If membership is unchanged:** call `ModelRunner.decode_step` (Phase 1)
  — one cheap batched call reusing the existing `StaticCache`.
- ✅ After that step, check each running sequence's stop condition — EOS
  token (`model_runner.tokenizer.eos_token_id`) or `max_new_tokens` reached.
- ✅ Evict anything that just finished — mark `FINISHED`, remove from
  `running` immediately (which sets `membership_changed`, forcing next
  step's resync) rather than waiting for the end of a shared batch.
- ✅ An explicit "nothing to do" guard in `step()` (both `waiting` and
  `running` empty) — found at review time, not in the original design sketch:
  without it, the scheduler going idle with `membership_changed` still `True`
  from the eviction that emptied it would call `prefill_batch` on an empty
  batch and crash on `max([])`.

**Files:** `engine/scheduler.py` (new), `tests/test_scheduler.py` (new),
`docs/achievements/m02-scheduler-build-session.md` (new — the build/learning
log).
**Concepts:** iteration-level scheduling (Orca); continuous batching; the OS
preemptive-scheduling analogy.
**Done when:** manually driving the scheduler through a few `step()` calls
with 2-3 fake sequences of staggered lengths produces the expected
admit/run/evict sequence. **Met** — see `test_eviction_forces_resync_for_remaining_sequences`.
**Test:** `tests/test_scheduler.py` — 12 tests against a `FakeModelRunner`
(no real model, no GPU) verifying admit/evict bookkeeping and, critically,
*which* forward-step primitive gets called when (resync vs. decode) across
admission, steady-state, and eviction transitions.
**Verified:** `uv run pytest tests/ -v` — 19/19 passing (7 from M0/M1/Phase 1
plus these 12).

---

## Phase 3 — Continuous-batch engine + static-batch engine  ✅

**Goal:** two new engine entry points, both built on Phase 1's forward-step
primitive: one driven by Phase 2's `Scheduler` (evicts/admits every
iteration), one driven by a static variant (admits a fixed-size batch, does
not admit anything new until every member of that batch has finished).

**Why both, and why static batching gets a real (if small) implementation:**
locked decision 6, above — this is the actual comparison Milestone 2 exists
to produce.

- ✅ Continuous engine (`engine/continuous_engine.py`, `ContinuousEngine`): a
  thin shell — `add_request` builds a `Sequence` and hands it to
  `Scheduler.add_sequence`; `step()` just forwards to `Scheduler.step()`.
  **API shape change, as flagged:** return type is `list[Sequence]`, not
  M1's `Sequence | None`, since zero or more sequences may finish in the
  same iteration.
- ✅ Static-batch engine (`engine/static_engine.py`, `StaticEngine`, Q3
  resolved — its own class, not a mode flag): fixed-size admission, and a
  finished sequence stays in its batch slot (still getting a real forward
  pass, output just discarded) until every member of that batch is done —
  the actual wasted-compute failure mode, not a shortcut approximation of
  it. A batch's `StaticCache` never resyncs mid-lifetime (membership is
  fixed once formed), unlike the continuous engine's `Scheduler`.
- ✅ Both stay behind the `add_request()`/`step()` shape locked in M1.

**Files:** `engine/continuous_engine.py` (new), `engine/static_engine.py`
(new). `engine/engine.py` (M1's naive engine) untouched, as locked.
**Concepts:** static batching's GPU-idle failure mode under variable output
lengths.
**Done when:** correctness test passes on both engines. **Met.**
**Test:** `tests/test_correctness.py::test_engine_matches_golden_multi_sequence`,
parametrized over both engines — 2 concurrent sequences, staggered
`max_new_tokens` (`[8, 20]`) so one genuinely finishes before the other,
exercising each engine's different early-finish handling for real.
**Verified:** `uv run pytest tests/ -v` — 22/22 passing (the 22nd is Phase
4's regression test, added below — 21 at the time Phase 3 itself was
verified). Getting here surfaced a real floating-point correctness-gate
issue, not a logic bug — see ADR-0009 (superseded) → ADR-0010 (accepted) and
`docs/bites/gpu-and-hardware/floating-point-determinism.md`. The correctness
gate now runs on CPU + eager attention unconditionally; benchmarks stay on
GPU + SDPA.

---

## Phase 4 — Benchmark: continuous vs. static vs. naive  ✅

**Goal:** the actual comparison the milestone's Definition of Done asks for.

**Why this phase, and why the workload needs to change:** M1's benchmark
used a small fixed prompt set with uniform `max_new_tokens` — a workload
where nothing ever finishes early. That can't demonstrate what continuous
batching fixes; the whole point (per the Orca reading) only shows up when
output lengths vary within a shared batch.

- ✅ `bench/benchmark.py`'s `WORKLOAD`: 6 prompts with deliberately mixed
  `max_new_tokens` (`[8, 60, 12, 50, 10, 40]`) — short and long requests
  interleaved, so short ones finish early while long ones keep running,
  which is the exact condition static batching handles badly. (Not M5's
  Poisson-arrival, ShareGPT-length workload generator — that's a separate,
  later build; every request here "arrives" at t=0.)
- ✅ **GPU idle time redefined as an exact, deterministic count instead of a
  wall-clock estimate** — a real design call made along the way, not in the
  original plan. "Wasted slot-steps": a step where an already-`FINISHED`
  sequence still occupies a running-batch slot (only possible for
  `StaticEngine` — `ContinuousEngine` evicts immediately). A wall-clock idle
  estimate would be noisier and less reproducible than a count derived
  straight from each engine's own bookkeeping. Recorded as
  `wasted_compute_fraction` (`wasted_slot_steps / total_slot_steps`)
  alongside throughput, TTFT p50/p99, TPOT p50/p99, and peak concurrent
  sequences, matching `docs/benchmarks/README.md`'s four-metrics convention
  as closely as M2's design allows (naive has no per-token timestamps
  without added plumbing, per ADR-0004's original scope call — TTFT/TPOT/
  wasted-compute are continuous/static-only).
- ✅ Saved to `docs/benchmarks/2026-08-23_09-18-14-m2-continuous-vs-static-vs-naive/`
  with commit + hardware + a readable `summary.md`, per `claude.md`'s
  provenance rule.
- 🐛 **Found and fixed a real bug getting here**: `StaticEngine._start_batch`'s
  original `max_cache_len` formula (`max(prompt_len + max_new_tokens)` per
  sequence) undercounted whenever a short-prompt/long-budget sequence shared
  a batch with a longer-prompt/short-budget one — every row is left-padded
  to the batch's *longest prompt*, then the whole batch steps in lockstep
  for as many iterations as the batch's *longest budget* needs, so the cache
  must hold `max(prompt_len) + max(max_new_tokens)`, not a per-sequence sum.
  Caught by actually running the benchmark (a real `index_copy_` CUDA
  assertion, not a silent wrong-answer) — see `engine/static_engine.py`'s
  `_start_batch` comment and `tests/test_static_engine.py`'s regression test.

**Files:** `bench/benchmark.py`, `engine/continuous_engine.py` (`.running`
property added for uniform benchmark introspection across both scheduled
engines), `engine/engine.py` (`add_request` gained an optional
`max_new_tokens` override, matching the other two engines, so the naive
baseline can run the *same* mixed workload for a fair comparison),
`engine/static_engine.py` (bug fix), `tests/test_static_engine.py` (new),
`docs/benchmarks/2026-08-23_09-18-14-m2-continuous-vs-static-vs-naive/`.
**Concepts:** GPU idle time as a directly measurable consequence of static
batching's collapse — measured here as an exact wasted-compute fraction
rather than a wall-clock estimate.
**Done when:** numbers exist showing continuous batching's throughput/
idle-time advantage over static batching under the mixed-length workload.
**Met** — continuous: 47.4 tok/s, 0% wasted compute. static: 27.2 tok/s,
46.3% wasted compute. naive: 13.8 tok/s (no concurrency, the "before"
number). All three generated the same 180 total tokens.
**Test:** `tests/test_static_engine.py` — the one real bug this phase
surfaced got a regression test; the benchmark run itself is measurement, not
correctness (per the original plan), covered instead by Phase 3's
correctness gate having already passed on both engines before this phase
started.

---

## Open questions / decisions to resolve (owner decides)

**Q1 — Admission batch-size cap. RESOLVED.** A `max_batch_size` constant,
passed into `Scheduler.__init__` — the constant, per the original
recommendation.

**Q2 — Where the per-sequence KV buffer lives. RESOLVED by ADR-0007.** There
is no per-sequence buffer — it belongs to the running batch as a whole
(a `StaticCache` rebuilt on membership change), owned by whichever engine
code drives the `Scheduler`, not by `Sequence` or a new module.
`engine/kv_cache.py` stays unused/reserved for M3's real `BlockAllocator`,
as originally recommended.

**Q3 — Static-batch engine structure. RESOLVED.** Its own class/module —
`StaticEngine` in `engine/static_engine.py`, not a mode flag on `Engine`.

---

## ADRs written alongside

- ✅ **ADR-0005 — KV cache ownership model for M2.** Own, pre-allocated KV
  cache vs. reusing `transformers`'s `DynamicCache` — decided before Phase 0
  (locked decision 5). A real alternative existed, and M3's `BlockAllocator`
  builds directly on whichever way this goes.
- ✅ **ADR-0006 — Static-batching baseline built for real, not reused from
  M1.** Decided before Phase 0 (locked decision 6) — why M1's engine
  wouldn't have been an honest comparison point.
- ✅ **ADR-0007 — Resync-prefill on batch membership change.** Discovered
  during Phase 1 — refines ADR-0005 once `StaticCache`'s shared write-
  position turned out to require a new cache per batch membership, not a
  literal per-sequence object.
- ✅ **ADR-0008 — Pin `repetition_penalty=1.0` for true greedy decoding.**
  Discovered during Phase 1 while debugging a correctness mismatch — a
  cross-cutting fix to the harness every milestone depends on, not
  M2-specific, but surfaced here.

---

## Concepts for `docs/concepts.md` (as they arise)

- ✅ **Sequence lifecycle / OS process-state analogy** — WAITING/RUNNING/
  FINISHED → `engine/sequence.py`.
- ✅ **Prefill vs. decode** — prefill processes the whole prompt in one
  parallel, compute-bound pass; decode generates one token at a time,
  memory-bandwidth-bound → `engine/model_runner.py`.
- ✅ **True greedy decoding vs. a checkpoint's default decoding behavior** —
  `.generate()`'s convenience defaults (like repetition penalty) aren't part
  of "greedy," even though they're deterministic → `tests/test_correctness.py`,
  ADR-0008.
- ✅ **Iteration-level scheduling** — admission/eviction decisions made every
  single decode step, not once per batch → `engine/scheduler.py` (Phase 2).
  Source: Orca paper, via the Anyscale continuous-batching writeup.
- ✅ **Static batching's collapse under variable output length** — a fixed
  batch runs at the pace of its slowest member; sequences that finish early
  stay in the batch (still computed on, output discarded) until the whole
  batch clears → `bench/benchmark.py` (Phase 4), measured as
  `wasted_compute_fraction`.
- ✅ **Floating-point non-associativity across recomputation paths** —
  discovered debugging Phase 3 → `docs/adr/0010-correctness-gate-cpu-plus-eager-attention.md`,
  `docs/bites/gpu-and-hardware/floating-point-determinism.md`.

---

## Definition of done (whole milestone)

- ✅ All phases above ticked.
- ✅ `tests/test_correctness.py` passes against both the continuous-batch and
  static-batch engines, multi-sequence.
- ✅ A benchmark demonstrating continuous batching beating static batching on
  throughput and GPU idle time, under a workload with variable output
  lengths, recorded in `docs/benchmarks/` with commit + hardware.
- ✅ `docs/progress.md` updated.
- ✅ ADR-0005 through ADR-0010 written and linked (ADR-0009 superseded by
  ADR-0010, kept per this project's own rule on wrong-but-reasoned decisions).
- ✅ `docs/roadmap.md` Milestone 2 checkbox ticked.

---

## Review summary (for the owner)

**Phases:** 0 `Sequence` state machine (waiting/running/finished, the
bookkeeping a scheduler needs) → 1 per-sequence KV buffer + a raw
forward-step primitive on `ModelRunner` (replaces `generate()` for
interleaved stepping) → 2 the `Scheduler` itself (iteration-level
admit/run/evict — the milestone's actual core mechanism, design-only in this
doc, built together) → 3 two engines wired to that primitive (continuous,
and a real static-batch comparison engine) → 4 the benchmark that produces
the milestone's headline comparison.

Nothing is implemented yet. Phases 0–1 are mostly routine scaffolding
(data-structure fields, a framework-boundary wrapper) and can move quickly.
Phase 2 is deliberately left as a design sketch, not code — that's where we
sit down together. Phase 3 wires two engines to what Phase 1/2 produce, and
Phase 4 is measurement.

**Decisions already made this session:** own contiguous per-sequence KV
buffer (not HF's `Cache`), and a real static-batch engine as the comparison
baseline (not reusing M1). Both recorded as locked decisions above, with
ADRs queued.

**Status now: Milestone 2 is complete.** All four phases built and verified
(see each phase's **Verified** line, the Run Guide below, and
`docs/achievements/m02-scheduler-build-session.md` for Phase 2's build log).
Q1, Q2, Q3 all resolved. Along the way: a real correctness-gate issue
(floating-point non-associativity across recomputation paths — ADR-0009
superseded → ADR-0010 accepted) and a real bug in `StaticEngine`'s cache
sizing (found running the actual benchmark, fixed, regression-tested). The
benchmark itself meets the roadmap's Definition of Done: continuous batching
beats static batching on both throughput (47.4 vs. 27.2 tok/s) and wasted
compute (0% vs. 46.3%) under a mixed-output-length workload.

Next: Milestone 3 — paged KV cache (`BlockAllocator`, block tables,
gather-then-attend). Per `claude.md`, its reading (Modal GPU Glossary,
*Making Deep Learning Go Brrrr*, OSTEP Ch 13/15/18–22, the full PagedAttention
paper) is next-milestone reading, not now — and it's the milestone that
matters most, so worth starting a fresh planning discussion rather than
rushing in.

---

## Run Guide

Covers everything built across all four phases — Milestone 2 is complete.
Assumes M0's one-time `uv sync` is already done.

### Running everything

```bash
uv run pytest tests/ -v
```

**What to expect:** `22 passed` in roughly a minute. Breakdown:
- `tests/test_correctness.py` (9 tests) — the real-model gate. Loads
  `Qwen/Qwen2.5-0.5B-Instruct` **on CPU with eager attention**, always,
  regardless of what GPU is on the machine — see "Why CPU + eager attention"
  below before assuming this is slow-by-accident.
- `tests/test_scheduler.py` (12 tests) — `Scheduler` bookkeeping only, no
  real model, `FakeModelRunner` instead — this file alone runs in well under
  a second.
- `tests/test_static_engine.py` (1 test) — same fake-model pattern, a
  regression test for the cache-sizing bug Phase 4 found.

If you only want the fast, no-model-download subset while iterating on
`Scheduler` or `StaticEngine` logic:

```bash
uv run pytest tests/test_scheduler.py tests/test_static_engine.py -v
```

### Why CPU + eager attention for the correctness gate

Short version (full story: ADR-0009 → ADR-0010, and the bite at
`docs/bites/gpu-and-hardware/floating-point-determinism.md`): our
`StaticCache`-based `prefill_batch`/`decode_step` primitive is
*mathematically* equivalent to `.generate()`'s own decode loop, but not
*numerically* bit-identical to it in every case — different cache shapes and
different attention-kernel choices can round differently, and at a genuine
near-tied logit that can flip which token wins greedy argmax. CPU + eager
attention was the one configuration (out of a full CPU/GPU × sdpa/eager
matrix) where this never happened, for every case tested — so it's what the
correctness gate uses, permanently, not something worth re-toggling. If a
correctness test ever fails again, this is the *last* thing to suspect, not
the first — the harness is already pinned specifically to rule this class of
issue out.

### What each piece in `engine/sequence.py` does (Phase 0)
- `SequenceStatus` (`WAITING` / `RUNNING` / `FINISHED`) — where a request is
  in its lifecycle. A `Scheduler` (Phase 2) will read and change this.
- `Sequence.max_new_tokens` — per-request now, not a single engine-wide
  value; each request can ask for a different generation length.
- `Sequence.last_token_id` — a property, not a stored field: reads
  `output_token_ids[-1]` (or the last prompt token, before anything's been
  generated). This is what gets fed as input to the *next* decode step.
- `Sequence.is_finished` — a small readability property over `status`.

### What each piece in `engine/model_runner.py` does (Phase 1)

(`encode`/`decode`/`generate` are unchanged from M0/M1 — only the additions
below are new.)
- `_left_pad(token_id_lists)` — takes a batch of token-id lists of
  *different* lengths and pads the shorter ones on the *left* with the
  tokenizer's EOS id, so every row's real content ends at the same rightmost
  column. Returns the padded `input_ids` and a matching `attention_mask`
  (`0` = padding, `1` = real token).
- `_position_ids_from_mask(attention_mask)` — turns that mask into position
  ids: real tokens are numbered `0, 1, 2, ...` regardless of how much
  padding precedes them; padded columns get a harmless placeholder (they're
  masked out of attention anyway, so their position doesn't matter).
- `prefill_batch(token_id_lists, max_cache_len)` — the **resync/prefill**
  call (ADR-0007). Runs one forward pass over each sequence's *full* token
  history at once (a fresh prompt, or a resync's prompt + everything
  generated so far), builds a brand-new `StaticCache` sized to
  `max_cache_len`, and returns each row's next greedy token plus the cache
  and mask to carry into `decode_step`. Call this whenever the running
  batch's membership changes.
- `decode_step(cache, last_token_ids, attention_mask)` — the cheap,
  steady-state call: one new token per row, one forward pass, `cache`
  updated in place. Call this every step where membership hasn't changed.

**Try it directly** (e.g. in a scratch script or REPL) — this is the manual
version of what Phase 2's `Scheduler` will automate:

```python
from engine.model_runner import ModelRunner

runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")

prompts = ["The capital of France is", "def fibonacci(n):"]
token_id_lists = [runner.encode(p) for p in prompts]
max_cache_len = max(len(ids) for ids in token_id_lists) + 20  # room for 20 new tokens each

next_ids, cache, mask = runner.prefill_batch(token_id_lists, max_cache_len)
generated = [[t] for t in next_ids]
for _ in range(19):
    last = [row[-1] for row in generated]
    next_ids, mask = runner.decode_step(cache, last, mask)
    for row, t in zip(generated, next_ids):
        row.append(t)

for prompt, row in zip(prompts, generated):
    print(prompt, "->", runner.decode(row))
```

**Why the correctness harness matters more than usual here:** every M2+
engine drives the model through `prefill_batch`/`decode_step`, never
`.generate()` again — so `tests/test_correctness.py` is the only thing
standing between "looks plausible" and "silently computing the wrong
tokens." ADR-0008 exists because that gap was real: debugging it is what
found the harness had an unstated dependency on this model's default
`repetition_penalty`.

### What each piece in `engine/scheduler.py` does (Phase 2)

- `Scheduler.__init__(max_batch_size, model_runner)` — owns `waiting`
  (a `deque`, FIFO), `running` (a `list`, order = the shared cache's row
  order), `cache`/`attention_mask` (`None` until the first step),
  `membership_changed` (did the *previous* step's eviction leave the cache
  stale?).
- `add_sequence(seq)` — the only way into `waiting`; force-sets
  `status = WAITING` so the scheduler is authoritative over the state
  machine regardless of what a caller constructed.
- `_admit()` — pulls `WAITING` sequences into `running`, up to
  `max_batch_size - len(running)`. Returns how many it admitted.
- `_prefill_or_decode()` — the resync-vs-decode decision from ADR-0007: if
  this step admitted anyone, or the *last* step evicted anyone, or there's
  no cache yet → resync (`prefill_batch` over every running sequence's full
  history). Otherwise → one cheap `decode_step`.
- `_check_evictions()` — EOS or `max_new_tokens` reached → `FINISHED`,
  removed from `running` immediately, returned to the caller.
- `step()` — the four above, composed: admit → prefill-or-decode → append
  tokens → evict. Returns whatever finished *this* step (possibly empty).

**Try it directly** — driving the scheduler by hand with a real model, no
`Engine` wrapper, watching resync vs. decode happen:

```python
from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from engine.sequence import Sequence

runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")
scheduler = Scheduler(max_batch_size=2, model_runner=runner)

seq_a = Sequence(seq_id=0, prompt_token_ids=runner.encode("The capital of France is"), max_new_tokens=6)
seq_b = Sequence(seq_id=1, prompt_token_ids=runner.encode("def fibonacci(n):"), max_new_tokens=20)
scheduler.add_sequence(seq_a)
scheduler.add_sequence(seq_b)

step_num = 0
while not (seq_a.is_finished and seq_b.is_finished):
    finished = scheduler.step()
    step_num += 1
    print(f"step {step_num}: finished={[s.seq_id for s in finished]}, running={[s.seq_id for s in scheduler.running]}")

print(runner.decode(seq_a.output_token_ids))
print(runner.decode(seq_b.output_token_ids))
```

**What to expect:** step 1 admits both (a resync, since `cache is None`).
Steps 2–6 are cheap decode steps. At step 6, `seq_a` (6-token budget)
finishes and gets evicted — printed as `finished=[0]`. Step 7 is forced back
into a resync (`membership_changed` from step 6's eviction), now covering
`seq_b` alone. Steps 8 onward decode `seq_b` until it hits its own 20-token
budget.

### What each piece in `engine/continuous_engine.py` / `static_engine.py` does (Phase 3)

`ContinuousEngine` is a thin shell — it owns a `Scheduler` and does nothing
`Scheduler` doesn't already do:

```python
from engine.continuous_engine import ContinuousEngine

engine = ContinuousEngine("Qwen/Qwen2.5-0.5B-Instruct", max_batch_size=2)
seq = engine.add_request("The capital of France is", max_new_tokens=10)
while not seq.is_finished:
    finished = engine.step()  # list[Sequence] -- zero or more, not M1's Sequence | None
print(engine.model_runner.decode(seq.output_token_ids))
```

`StaticEngine` has the same `add_request`/`step()` shape but its own
admission logic — no `Scheduler` involved, since its policy ("wait for the
whole batch") is genuinely different, not a variant of iteration-level
scheduling:

```python
from engine.static_engine import StaticEngine

engine = StaticEngine("Qwen/Qwen2.5-0.5B-Instruct", max_batch_size=2)
seqs = [
    engine.add_request("The capital of France is", max_new_tokens=5),
    engine.add_request("def fibonacci(n):", max_new_tokens=20),
]
while not all(s.is_finished for s in seqs):
    engine.step()
    print([len(s.output_token_ids) for s in seqs], [s.is_finished for s in seqs])
```

**What to expect:** watch the first sequence's token count stop growing at 5
while the second keeps climbing to 20 — the first sequence is `FINISHED`
but its slot is still occupied and `engine.step()` is still doing real work
for it every iteration (its printed length just stops changing). This is
`StaticEngine`'s failure mode made visible one line at a time, before Phase
4 measures it as a number.

### Running the M2 benchmark (Phase 4)

```bash
uv run python -m bench.benchmark --engine all
```

(`--engine {naive,continuous,static,all}`, `--max-batch-size N` — default 3
— and `--model-id` are also available.) Takes under 30 seconds on a small
GPU; saves `docs/benchmarks/<timestamp>-m2-continuous-vs-static-vs-naive/results.json`
and prints the same JSON to stdout.

**What to expect, and how to read each field** (see
`docs/benchmarks/2026-08-23_09-18-14-m2-continuous-vs-static-vs-naive/summary.md`
for the actual recorded run):

| Field | Meaning | What "good" looks like |
|---|---|---|
| `throughput_tokens_per_sec` | total output tokens ÷ wall time | continuous > static > naive |
| `total_tokens` | sum of every sequence's generated tokens | **identical across all three engines** — if not, something is generating the wrong number of tokens somewhere; check this before trusting any other number in the same run |
| `ttft_p50_s` / `ttft_p99_s` | time from submission to each sequence's *first* token | continuous's p99 should be lower than static's when `max_batch_size` < number of prompts, since static makes late-queued requests wait for an entire earlier batch to clear, not just a slot |
| `tpot_p50_s` / `tpot_p99_s` | average time per output token *after* the first, per sequence | should be similar between continuous and static — both do one real decode step per token once a sequence is actually running; this metric isn't where the difference shows up |
| `peak_concurrent_sequences` | max simultaneous `RUNNING` sequences observed | naive: always 1. continuous/static: equal to `max_batch_size` if enough requests were queued to fill it. **Capped by `max_batch_size`, not real memory pressure — that's Milestone 3's story, not this one** (ADR-0004 flagged this same caveat for peak-concurrency back at M1) |
| `wasted_slot_steps` / `total_slot_steps` / `wasted_compute_fraction` | naive/continuous-only field is `n/a`; for static, counts steps spent computing on an already-`FINISHED` row still occupying a batch slot | **continuous: 0.0 always, by construction** (it evicts immediately — if this is ever nonzero, that's a real bug in `Scheduler`, not noise). static: nonzero whenever output lengths are staggered — this *is* the number the roadmap's "GPU idle time" Definition of Done asks for |

**The one sanity check worth running by eye before trusting a result:**
`total_tokens` must match across all three engines in the same run. It's the
cheapest possible correctness cross-check on a benchmark script itself —
three different scheduling strategies did three different amounts of *work
along the way* (wasted or not), but they must produce the same amount of
*real output*, since they're running the identical workload.
