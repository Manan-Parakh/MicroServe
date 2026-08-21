# Progress Log

A dated, terse record of what happened and what was learned. Per `claude.md`:
one entry per non-trivial change, 1–2 lines, tagged with the concept name(s),
linking to an ADR or plan if one applies. No entry for trivial changes — an
entry for any real decision or any moment something worth recalling happened.

Newest entries at the top.

---

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
