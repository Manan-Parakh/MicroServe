# M01 — Naive baseline — Implementation Plan

**Status:** Built and verified — all phases complete, correctness gate passing, baseline recorded.
**Milestone:** M1 (see `docs/roadmap.md`).
**Scope:** A minimal `engine.py` that handles one request at a time via HF's
own generation loop and cache — the deliberately-bad "before" number every
later milestone is measured against.
**Guardrail (verbatim):** Resist building the scheduler or block allocator
here even if it's tempting to get ahead — this milestone exists solely to
produce a comparison baseline; added complexity now just makes that baseline
murkier.

---

## Reading required before this milestone

**Nothing new** — everything needed already exists from Milestone 0
(roadmap.md Master Map, row 1).

---



## The core mental model this plan is built around

> One request in, one request fully generated, before the next one starts.
> No batching, no scheduling — the point is to measure how bad that is.

This milestone is not "build a small feature of the real engine" — it's
"build the control group." Every metric from M2 onward (continuous batching,
paging, quantization) is only meaningful relative to this number. If this
baseline is accidentally *good* (e.g. because some scheduling sneaks in), the
whole project's headline comparisons get quieter than they should. The
guardrail above exists specifically to protect that: bad-on-purpose,
measured honestly.

---



## Locked decisions this plan honors

1. `engine.add_request(prompt)` + `engine.step()` is the public API surface.
  *Why this shape, and why now:* this two-method interface is what every
   future engine version (naive, batched, paged, quantized) will implement
   identically — `add_request` enqueues work, `step()` advances the engine by
   one iteration. Fixing that shape here, in the simplest possible
   implementation, means M2's scheduler and M3's block allocator get to slot
   in *behind* an interface that doesn't change, instead of every milestone
   inventing its own calling convention. (Serving layer — HTTP, streaming —
   is explicitly Act Two and does not touch this API's shape.)
2. `engine/` stays synchronous, framework-free — same reasoning as M0: this
  layer is the thing being studied, not wrapped in infrastructure yet.

---



## What already exists

- `engine/model_runner.py` (from M0) — wraps the HF model forward/generate
call; this milestone calls into it rather than talking to `transformers`
directly, so the model-loading/generation boundary stays in one place.
- `tests/test_correctness.py` (from M0) — the gate this milestone must keep
passing. Since M1 doesn't change *how* tokens are computed (still plain HF
`generate()`, just wrapped), passing this test should be closer to
"confirm nothing broke" than a hard bar to clear — a genuinely useful
signal if it does fail here, since it means the wrapper itself introduced a
bug, not the model.

---



## Phase 0 — Minimal `Sequence`  ✅

**Goal:** A `Sequence` object that just tracks id, prompt tokens, output
tokens, and status — no scheduler-facing state yet.

**Why this phase, and why it's intentionally bare:** `Sequence` is the object
that represents "one request" as it moves through the engine, and it will
grow substantially over the project — M2 adds waiting/running/finished state
for the scheduler, M3 adds a `block_table` for paged memory. Building only
the bare fields needed *right now* (not the fields M2/M3 will eventually
want) is deliberate: guessing ahead at what the scheduler or allocator will
need risks baking in the wrong shape before those milestones' actual
requirements are known. This phase exists to give M1's engine loop something
concrete to hold onto per request — nothing more.

- ✅ `Sequence`: id, input token ids, generated token ids, status
(waiting/done). Two states only, because M1 has no concept of a sequence
being paused mid-generation to let another one run — it's either not
started yet or it's done. That binary is itself evidence of what M1 is
missing, which M2 exists to fix.

**Files:** `engine/sequence.py`
**Concepts:** what a "sequence" means in an inference engine (one request's
full lifecycle)
**Done when:** a `Sequence` can be constructed and holds its tokens.
**Test:** none needed, exercised by Phase 1.
**Verified:** a `@dataclass` with `seq_id: int`, `prompt_token_ids: list[int]`,
`output_token_ids: list[int]` (defaults empty), `status: SequenceStatus`
(`WAITING`/`DONE`, an `Enum`) — exactly the fields above, nothing extra.

---



## Phase 1 — Naive engine loop  ✅

**Goal:** `engine.add_request(prompt)` + `engine.step()` that processes one
`Sequence` at a time to completion via HF `generate()`.

**Why this phase, and why "no scheduler" is the point, not a shortcut:**
this is the actual deliverable of the milestone. A request that arrives while
another is generating simply waits its turn — no interleaving, no shared GPU
time between concurrent requests. That's exactly what production LLM serving
gets wrong if you don't batch: GPU sits idle waiting on one slow decode step
at a time. Building this badly-behaved version *correctly* (i.e., without
secretly optimizing it) is what makes the M2 comparison honest.

- ✅ `add_request(prompt)` enqueues a new `Sequence` — this is the entire
"admission" logic for this milestone: append to a list. (Compare to M2,
where admission becomes a scheduling decision.)
- ✅ `step()` runs the next queued sequence's full generation to completion
(one request at a time, no interleaving) — note this is a different notion
of "step" than later milestones will use. Here, one `step()` call may do an
entire multi-token generation; from M2 onward, one `step()` call does
*one* decode iteration across possibly many sequences (iteration-level
scheduling). Flagging that now so the API doesn't silently mean two
different things without anyone noticing.
- ✅ No scheduler, no custom KV cache — HF's own cache (managed internally by
`generate()`) is used as-is. *Why:* this milestone measures what you get
for free from the framework, before this project's own scheduling and
memory-management code enters the picture at all.

**Files:** `engine/engine.py`
**Concepts:** naive/static serving — the "before" every scheduler-based
improvement is compared against
**Done when:** two sequential requests each produce correct output.
**Test:** `tests/test_correctness.py` run against this engine path.
**Verified:** `test_naive_engine_matches_golden` in
`tests/test_correctness.py` drives the real `Engine.add_request()`/`step()`
path (not just `ModelRunner` directly) and asserts exact match against the
golden reference, for both test prompts — 4/4 tests passing overall.

---



## Phase 2 — Baseline benchmark numbers  ✅

**Goal:** Record throughput and latency for this engine so later milestones
have something to beat.

**Why this phase, and why it's part of M1 rather than deferred entirely to
M5:** an implementation without a recorded number isn't a baseline yet — it's
just code. The whole point of building this milestone badly-on-purpose is
lost if the "how bad" never gets written down. This phase is intentionally
the smallest possible measurement, not the real benchmark harness (see Q1) —
just enough to produce a comparable number before M2 changes the engine
underneath it.

- ✅ `bench/benchmark.py --engine naive` (minimal version — full workload
harness is Milestone 5) — a thin script, not the Poisson-arrival workload
generator that M5 builds; this milestone only needs "run N prompts,
sequentially, and time them," since there's no concurrency to model yet.
- ✅ Record tokens/sec (**throughput** — system-wide, how much work the
engine gets through) and per-request latency (**how long one request
waits**, p50/p99) for a small fixed set of prompts — two of the four
metrics named in `claude.md`. TTFT/TPOT/peak-concurrency are deliberately
*not* measured yet — see ADR-0004's scope note for why (no per-token
timestamp exists until `step()` is iteration-level at M2, and peak
concurrency is trivially 1 by construction here).
- ✅ Save numbers to `docs/benchmarks/` with commit + hardware — per
`claude.md`, a number without provenance (which commit, which GPU/CPU) is
not quotable; this is what makes the M1 baseline something M2 can actually
cite later, instead of a number nobody can reproduce or trust. **Caveat:**
recorded `commit: "uncommitted"` since no git commit exists yet in this
repo — re-run (or manually patch the field) once the first commit lands.

**Files:** `bench/benchmark.py`, `docs/benchmarks/`
**Concepts:** throughput vs. latency as distinct metrics
**Done when:** a numbers file exists and is referenced from `docs/progress.md`.
**Test:** none — this is measurement, not correctness.
**Verified:** `docs/benchmarks/2026-08-19-m1-naive-baseline/` — 16.15
tokens/sec, p50 latency 3.93s, p99 4.27s, on the RTX 3050 Ti Laptop GPU, 5
fixed prompts, 64 max new tokens each. See that folder's `README.md` for the
full header.

---



## Open questions / decisions to resolve (owner decides)

**Q1 — How much of** `bench/benchmark.py` **to build now vs. defer to M5.
RESOLVED: bare sequential timing script.** See ADR-0004.

---



## Concepts for `docs/concepts.md` (as they arise)

- **Naive/static serving** — one request processed to completion before the
next starts; no batching → `engine/engine.py`
- **Throughput vs. latency** — throughput is system-wide tokens/sec, latency
is per-request time; a fast engine can still feel slow to one user →
`bench/benchmark.py`

---



## Definition of done (whole milestone)

- ✅ All phases above ticked.
- ✅ `pytest tests/test_correctness.py` passes against this engine.
- ✅ Baseline throughput/latency numbers recorded in `docs/benchmarks/`.
- ✅ `docs/progress.md` updated.
- ✅ `docs/roadmap.md` Milestone 1 checkbox ticked.

---



## Review summary (for the owner)

**Phases:** 0 minimal `Sequence` (a request's identity and state, deliberately
bare) → 1 naive engine loop (one request at a time, the actual "before"
behavior) → 2 baseline benchmark numbers (writing the "how bad" down so M2
onward has something concrete to beat).

All three built and verified — see the **Verified** line under each phase
above. Q1 resolved via ADR-0004.

**Outstanding, worth your attention:** the recorded baseline
(`docs/benchmarks/2026-08-19-m1-naive-baseline/`) is tagged
`commit: "uncommitted"` because this repo has no git commits yet — per this
project's own git-safety rule, commits are never made without an explicit
ask. Once you make the first commit, re-run
`uv run python -m bench.benchmark --engine naive` so the quoted baseline
carries real provenance.

---



## Run Guide

Assumes M0's one-time `uv sync` has already been done (see
`docs/plans/m00-setup-plan.md`'s Run Guide).

**Run the naive engine's correctness check** (part of the same gate M0 set up
— `test_naive_engine_matches_golden` in `tests/test_correctness.py` exercises
this milestone's `Engine` specifically):

```bash
uv run pytest tests/test_correctness.py -v
```

**Run the M1 baseline benchmark:**

```bash
uv run python -m bench.benchmark --engine naive
```

Prints the result JSON to stdout and saves it to
`docs/benchmarks/<today>-m1-naive-baseline/results.json` (a fresh run today
creates a fresh dated folder rather than overwriting a prior one — that's
deliberate, per `docs/benchmarks/README.md`'s "one folder per run"
convention).

**Use the engine directly** (e.g. in a scratch script or REPL):

```python
from engine.engine import Engine

engine = Engine("Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=64)
engine.add_request("The capital of France is")
result = engine.step()          # runs the request to completion, one at a time
print(engine.model_runner.decode(result.output_token_ids))
```

**Quick sanity note:** the first call to any of the above on a fresh machine
will download the Qwen2.5-0.5B-Instruct checkpoint (~1GB) from the Hugging
Face Hub; subsequent runs are fast (cached).