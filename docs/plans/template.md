# M<NN> — <Milestone name> — Implementation Plan

**Status:** Draft for review — do NOT start implementation until approved.
**Milestone:** M<NN> (see `docs/roadmap.md` for the milestone's build spec and
guardrail).
**Scope:** <one or two sentences — what this plan builds, stated as narrowly as
the milestone's own BUILD spec in `docs/roadmap.md`.>
**Guardrail (copied from the roadmap, so it's visible while building):**
<paste the milestone's guardrail here verbatim — it is as binding as the spec.>

This is a working checklist. Each phase should be small and independently
testable. Tick boxes as you go. This file is the sub-phase breakdown for one
milestone — the milestone itself, and its place in the wider project, lives in
`docs/roadmap.md`.

---

## The core mental model this plan is built around

> One or two sentences: the single idea this milestone exists to prove or
> build. If a phase below doesn't serve that idea, it doesn't belong here.

---

## Locked decisions this plan honors

Decisions already made — either in `claude.md`'s "Locked decisions" section,
or in an ADR written before this plan. List them so the plan doesn't quietly
re-litigate them:

1. <decision — link the ADR if one exists, e.g. "see ADR-00NN">
2. <decision>

---

## What already exists (so we don't redo it)

- <file/module that already exists and this plan builds on, with the exact
  path>
- <a decision or constant already fixed elsewhere in the codebase>

---

## Phase 0 — <name>  ⬜

**Goal:** <the smallest, most isolated thing this phase proves.>

- ⬜ <concrete step>
- ⬜ <concrete step>

**Files:** <paths this phase touches>
**Concepts:** <systems/inference concept(s) this phase is teaching — these
become `docs/concepts.md` entries once the phase lands>
**Done when:** <the observable condition — a test passing, a number printed,
a specific behavior reproduced>
**Test:** <what `tests/` file/case covers this, or "none needed, exercised by
later phases">

---

## Phase 1 — <name>  ⬜

**Goal:** <...>

- ⬜ <concrete step>

**Files:**
**Concepts:**
**Done when:**
**Test:**

---

<!-- repeat Phase N sections as needed — keep each one small enough to build
and verify in a single sitting -->

---

## Open questions / decisions to resolve (owner decides)

Per `claude.md`'s decision protocol: present 2–3 real options, one paragraph
each, with a recommendation — then stop and wait.

**Q1 — <question>.** <2-3 options with tradeoffs>. **Recommendation:** <...>,
because <...>.

---

## ADRs to write alongside (if any)

- **ADR-00NN — <title>.** <one-line context for why this decision needs a
  written record — a real alternative existed and the reasoning is worth
  keeping.>

---

## Concepts for `docs/concepts.md` (as they arise)

- **<term>** — <terse definition> → `<file path where it's used>`

---

## Definition of done (whole milestone)

- ⬜ All phases above ticked.
- ⬜ `tests/test_correctness.py` passes (or the milestone's equivalent gate).
- ⬜ Benchmark numbers (if any) recorded in `docs/benchmarks/` with commit +
  hardware.
- ⬜ `docs/progress.md` updated.
- ⬜ Any ADRs from this plan written and linked.
- ⬜ `docs/roadmap.md` checkbox for this milestone ticked.

---

## Review summary (for the owner)

**Phases:** <one line per phase, in order — the whole plan at a glance.>

**The decisions I most want your call on before I build:** <the 1-3 open
questions that actually shape the interfaces, not the small ones.>

Nothing is implemented yet — this is the plan. Say go (or redirect) and I'll
start at Phase 0.
