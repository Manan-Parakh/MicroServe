# M00 — Setup — Implementation Plan

**Status:** Built and verified — all phases complete, correctness gate passing.
**Milestone:** M0 (see `docs/roadmap.md`).
**Scope:** Repo skeleton, dependency/lint tooling, tiny model running via plain
HF `generate()`, and the correctness harness every later milestone is judged
against.
**Guardrail (verbatim):** Don't install vLLM, Triton, or TensorRT-LLM yet.
Don't pick the eventual 7B model yet — that's Milestone 4's call.

---

## Reading required before this milestone

**Nothing new** — pure engineering setup (roadmap.md Master Map, row 0).

Optional, non-gating context: **A2 — memory-mapped files** (`man mmap` + short
blog explainer, ~15 min) — why `.safetensors` loads fast: the file is mapped
into the process's virtual address space and its pages fault in lazily as
layers are actually read, instead of the whole checkpoint being read into RAM
up front. Not required to finish M0, but it explains something you'll
literally watch happen the first time a multi-GB checkpoint loads almost
instantly and then "warms up" as generation starts.

---

## The core mental model this plan is built around

> Get the correctness test right once, here, so every later milestone can be
> trusted against it without re-deriving what "correct" means.

Every later milestone rewrites *how* tokens get computed (batching, paging,
quantization). None of them are allowed to change *what* gets computed. M0's
whole job is building the one artifact — the correctness harness — that lets
every future rewrite be checked against a fixed, known-good answer instead of
"it looks right." Everything else in this milestone (repo, tooling, model
loading) exists only in service of getting to that test.

---

## Locked decisions this plan honors

1. **Dev model is tiny** (Qwen2.5-0.5B or Llama-3.2-1B) — owner's choice
   between the two. *Why it's locked:* the point of a "dev model" is fast
   iteration while debugging the engine's own logic, not realism — a 7B model
   would make every test run and every debug cycle painfully slow for no
   benefit at this stage. The realistic model is deliberately deferred to M4.
2. **`engine/` stays synchronous, framework-free** — no `async def`, no
   FastAPI anywhere in it. *Why:* this project is the engine, not the server
   around it. Mixing an async event loop or a web framework into the engine
   layer now would couple two concerns (scheduling logic vs. request
   I/O) that the project deliberately wants to keep separable — serving is
   explicitly "Act Two."
3. **CPU is fine for correctness tests if no usable GPU is present.**
   *Why:* correctness (does the engine emit the right tokens) and performance
   (how fast) are different questions. Gating the *correctness* harness on
   GPU availability would mean the project's most important safety net stops
   working the moment you're on a machine without one — so it has to run
   anywhere. Benchmark numbers, in contrast, are GPU-only and always tagged
   with which GPU.

---

## What already exists

- `docs/` structure (roadmap, ADRs, plans, concepts, benchmarks) already in
  place — the "how we work" scaffolding predates any code.
- Nothing under `engine/`, `bench/`, or `tests/` yet — this plan is what
  creates them.

---

## Phase 0 — Repo & tooling  ✅

**Goal:** A repo that runs `pytest -v` clean with zero tests, before any
engine code exists.

**Why this phase, and why first:** none of the engineering-hygiene questions
here (dependency manager, lint config, pinned versions) are specific to
inference engines — they're the same setup any serious Python project needs,
and getting them right *before* code exists means the first real code
(Phase 1) lands into a repo that already has reproducible installs and a
working test runner, instead of retrofitting tooling around code that's
already there.

- ⬜ `git init`, first commit — `git init` done; first commit intentionally
  left to the owner (never committed without an explicit ask).
- ✅ Dependency manager chosen: **uv** (ADR-0001). This determines how
  `pyproject.toml`/lockfiles get written in every later phase.
- ✅ `pyproject.toml` with **pinned** versions (`torch`, `transformers`,
  `accelerate`; dev group `pytest`, `ruff`), resolved into `uv.lock`. *Why
  pinned:* this project's correctness test compares against a "golden"
  reference; an unpinned `transformers` version could silently change
  generation behavior on a fresh install months from now, making the golden
  reference wrong instead of the code.
- ✅ `pytest` configured via `[tool.pytest.ini_options]` in `pyproject.toml`
  (`testpaths = ["tests"]`, `pythonpath = ["."]` so `engine`/`bench` import
  cleanly without a src-layout or installed package).
- ✅ Lint/format tool: **ruff** (ADR-0001), configured in `pyproject.toml`
  (`line-length = 100`, `select = ["E", "F", "I", "UP"]`).
- ✅ `README.md` at repo root — front door for anyone opening the repo cold;
  Status and Running It sections updated to reflect M0/M1 completion.
- ✅ `LICENSE` — MIT (no strong reason to pick otherwise for a solo learning
  project; trivially changeable later).

**Files:** repo root, `pyproject.toml`, `README.md`, `LICENSE`
**Concepts:** none new
**Done when:** `pytest -v` runs clean with zero collected tests.
**Test:** none needed.
**Verified:** `uv sync` resolves and locks; `uv run ruff check .` passes
clean. (First `uv sync` silently installed CPU-only `torch` — PyPI's default
Windows wheel; fixed by pinning an explicit `pytorch-cu130` index in
`[tool.uv.sources]`/`[[tool.uv.index]]` to match the RTX 3050 Ti's
driver-reported CUDA 13.1. Worth knowing: the default index is not
GPU-aware, and this project's whole point depends on the GPU actually being
used from here on.)

---

## Phase 1 — Tiny model smoke test  ✅

**Goal:** The chosen tiny model loads and generates through plain HF
`generate()`.

**Why this phase, and why before the harness:** you can't write a
correctness test against a model you haven't confirmed actually loads and
generates coherent text on this machine. This phase is deliberately *manual
and throwaway* — the point is to surface environment problems (missing CUDA,
wrong `transformers` version, tokenizer mismatches) cheaply, with a human
eyeballing the output, before any of that becomes a mysterious test failure
in Phase 2 with no context to debug it from.

- ✅ `engine/`, `bench/`, `tests/` package skeletons — the directory
  structure fixed by `claude.md`'s repo layout, created empty so Phase 2
  onward has somewhere to put files.
- ✅ Load Qwen2.5-0.5B-Instruct (ADR-0002) via `transformers` — confirms the
  environment (Python version, `torch`, CUDA/CPU) can actually instantiate
  the model, independent of anything this project writes.
- ✅ One manual generation call, printed output, sanity-checked by eye — a
  human judgment call ("does this look like real language") that a script
  can't yet make, because there's no golden reference to compare against
  until Phase 2 exists.

**Files:** `engine/model_runner.py` (minimal), scratch script
**Concepts:** none new
**Done when:** the model produces coherent text for a test prompt.
**Test:** none needed, exercised by Phase 2.
**Verified:** `engine/model_runner.py`'s `ModelRunner` loads the model and
tokenizer once and exposes `encode`/`generate`/`decode` — every later engine
version calls through this one object instead of `transformers` directly.

---

## Phase 2 — Correctness harness  ✅

**Goal:** `tests/test_correctness.py` exists and is the gate every future
milestone must pass.

**Why this phase, and why it's the milestone's real deliverable:** this is
the artifact everything above was building toward. Once this test exists,
every later milestone (naive → batched → paged → quantized) has a mechanical,
automated answer to "did I just break generation" instead of a human
eyeballing output like Phase 1 did. `claude.md`'s rule — no benchmark number
gets quoted until this test passes on that code path — only has teeth once
this file exists.

- ✅ **Fixed random seed** — removes one axis of run-to-run variation so a
  failing test can be attributed to a real code difference, not luck.
- ✅ **Greedy decoding** — always take the highest-probability token instead
  of sampling. *Why greedy specifically:* sampling adds a second source of
  randomness (the RNG stream) on top of the model's computation, which would
  make a test failure ambiguous — did the *code* produce a different result,
  or did it just sample a different token from an equivalent distribution?
  Greedy collapses that ambiguity: the same prompt through the same weights
  should produce the same argmax token, full stop, so any mismatch is a real
  signal. Scope locked to greedy-only per ADR-0003.
- ✅ **Golden reference:** HF `generate()` output tokens for a fixed prompt,
  generated once per test module (a `scope="module"` pytest fixture) — this
  is the "known-good" oracle every future engine variant gets diffed against,
  not re-derived each run.
- ✅ Test asserts the engine's own output tokens match the golden reference
  **exactly** (token-for-token, not "looks similar") — an exact match is what
  makes this test able to catch the specific bug class this project is most
  exposed to later: a batching or block-indexing bug that changes *which*
  token gets picked, which a fuzzy/statistical comparison could easily miss.

**Files:** `tests/test_correctness.py`
**Concepts:** greedy decoding, deterministic generation
**Done when:** `pytest tests/test_correctness.py` passes.
**Test:** is itself the test.
**Verified:** `uv run pytest -v` — 4 passed (2 prompts × `ModelRunner` +
naive `Engine`, the latter added once M1's engine existed; see M1's plan for
why the same file covers both).

---

## Open questions / decisions to resolve (owner decides)

**Q1 — Correctness harness scope. RESOLVED: greedy-only.** See ADR-0003.

**Q2 — Python tooling. RESOLVED: uv + ruff.** See ADR-0001.

**Q3 — Tiny model choice. RESOLVED: Qwen2.5-0.5B-Instruct.** See ADR-0002.

---

## Concepts for `docs/concepts.md` (as they arise)

- **Greedy decoding** — always pick the highest-probability next token; makes
  generation deterministic → `tests/test_correctness.py`

---

## Definition of done (whole milestone)

- ✅ All phases above ticked (except the first-commit half of Phase 0's
  `git init` bullet — intentionally left to the owner).
- ✅ `pytest tests/test_correctness.py` passes.
- ✅ `docs/progress.md` updated.
- ✅ `docs/roadmap.md` Milestone 0 checkbox ticked.

---

## Review summary (for the owner)

**Phases:** 0 repo/tooling (reproducible environment) → 1 tiny model smoke
test (confirm the environment can actually run the model, by eye) → 2
correctness harness (the automated oracle every later milestone answers to).

All three built and verified — see the **Verified** line under each phase
above. Decisions Q1–Q3 are resolved via ADR-0001–0003.

---

## Run Guide

Everything below assumes `uv` is installed and you're in the repo root.

**One-time setup:**
```bash
uv sync                    # creates .venv, installs pinned deps (torch, transformers, accelerate, pytest, ruff)
```
The first `uv sync` on a fresh machine will download `torch` (a few GB for
the CUDA build) and, on first test/benchmark run, the ~1GB Qwen2.5-0.5B-Instruct
checkpoint from the Hugging Face Hub (cached under the default HF cache dir
after that — see `.gitignore`'s `huggingface_cache/` entry).

**Run the correctness gate** (the thing every milestone must pass):
```bash
uv run pytest -v
uv run pytest tests/test_correctness.py -v   # just the gate, same thing here since it's the only test file
```

**Lint/format check:**
```bash
uv run ruff check .
uv run ruff format .       # auto-fixes formatting; run check again after
```

**Verify the GPU is actually being used** (worth confirming after any fresh
`uv sync`, since the default PyPI index silently serves CPU-only `torch` —
see Phase 0's Verified note):
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

See `docs/plans/m01-naive-baseline-plan.md`'s Run Guide for running the
engine itself and the benchmark script.
