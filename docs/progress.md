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
