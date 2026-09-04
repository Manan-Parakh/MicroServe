# M<NN> — <Milestone name> — Implementation Plan

<!--
HOW TO USE THIS TEMPLATE
========================
This file has two lives, and the template covers both:

  1. **As a proposal**, before any code exists. Fill in everything down to
     "Review summary", leave the POST-BUILD sections as stubs, and stop. Do
     not start building until the owner says go.
  2. **As a build log**, annotated *in place* as each phase lands and
     reconciled against the measured result at the end. Tick boxes, append
     "→ Resolved:" lines to the open questions, and fill the POST-BUILD
     sections.

The rule that makes this work: **annotate, never rewrite.** When reality
diverges from the plan, leave the pre-build reasoning standing and say what
actually happened next to it. Per `claude.md`, the rejected alternatives and
the wrong predictions are the part worth keeping — a plan edited to look
correct in hindsight teaches nothing.

Delete this comment block when you fill the template in.
-->

**Status:** Draft for review — do NOT start implementation until approved.
<!-- POST-BUILD, replace with: **Status: COMPLETE — YYYY-MM-DD.** <what
landed, test count, benchmark recorded, ADRs written.> Nothing here is
pending approval. -->

**Milestone:** M<NN> (see `docs/roadmap.md` for the milestone's build spec and
guardrail).

**Scope:** <one or two sentences — what this plan builds, stated as narrowly as
the milestone's own BUILD spec in `docs/roadmap.md`.>
<!-- POST-BUILD: relabel to "**Scope, as built:**" and correct it to what was
actually built, if that differs. -->

**Guardrail (copied from the roadmap, so it's visible while building):**
<paste the milestone's guardrail here verbatim — it is as binding as the spec.>
<!-- POST-BUILD: relabel to "**Guardrail — held.**" (or "— crossed", with the
superseding ADR) and name the specific place where it pulled hardest. The
guardrail's near-misses are worth recording: they are where the milestone
wanted to become a different project. -->

**The decisions, and where they are recorded:**

| | question | resolution | ADR |
|---|---|---|---|
| Q1 | <short form of the question> | <blank until resolved> | |
| Q2 | | | |

<!-- POST-BUILD, insert here, right under the title:

**This document is a build log, not a proposal.** <one paragraph: written as
a plan, annotated as each phase landed, reconciled against the results.>
Where reality diverged from the plan, the divergence is called out rather
than edited away. There are <N> such divergences and they are the most useful
paragraphs in the file:

1. **<divergence>** — <one line, with a pointer to the section>.
2. ...

**Where to start reading.** If you want to *run* it: **How to run this
milestone**. If you want to know whether it worked: **Phase <N> — results,
read honestly**. Everything before those two sections is the design and its
record.
-->

---

## The reading gate

Per `claude.md`'s just-in-time rule, this is read immediately before Phase 0
and **not earlier** — theory read weeks ahead of application evaporates. The
milestone's list lives in `docs/roadmap.md`; restate it here with *why each
item matters for this specific milestone*, because that is what makes it land.

**The roadmap's theory boxes are the owner's to tick, not mine** — a theory box
records having read the thing.

1. **<source>** — <what to take from it, and what to skip>. Why it matters
   here specifically: <the phase it maps onto>.
2. **<source>** — <...>

**What is deliberately NOT on this list:** <the adjacent-and-tempting reading
the guardrail excludes>. If an item above makes any of that look tempting,
that is the guardrail's exact failure mode — note it as a "bite" in
`docs/bites/` and move on.

---

## The core mental model this plan is built around

> One or two sentences: the single idea this milestone exists to prove or
> build, stated as a mechanism rather than a goal. If a phase below doesn't
> serve that idea, it doesn't belong here.

The measurement that matters is **<the one number>** — not <the number it will
be confused with>.

---

## Locked decisions this plan honors

Decisions already made — either in `claude.md`'s "Locked decisions" section,
or in an ADR written before this plan. List them so the plan doesn't quietly
re-litigate them:

1. <decision — link the ADR if one exists, e.g. "see ADR-00NN">
2. <decision>

<!-- If a locked decision or roadmap statement is *wrong* — not a decision to
revisit, factually wrong, like a tensor shape or a constant — correct it here
and label it: "Not a decision — a fix." Say what following it literally would
have cost. -->

---

## What exists before this milestone (the seams it cuts into)

For each existing module this plan touches: what it does today, and **what
this milestone needs from it that it doesn't currently give**. The second half
is the point — it is where the integration risk lives.

- `<path>` — <what it does today>. **What M<NN> needs from it:** <...>
- `<path>` — **does not exist yet.** <which phase creates it>

---

## Ownership: who writes what

Per `claude.md`'s prime directive — the milestone's core mechanism is the
owner's to write; scaffolding, plumbing, and docs are mine.

| Component | Who |
|---|---|
| <the core mechanism — the thing the milestone exists to teach> | **You.** This is the milestone. |
| <the second core mechanism / the bug-farm> | **You.** |
| <framework boilerplate around it> | Me. Plumbing, not the idea. |
| Test scaffolding, fakes, fixtures | Me. |
| Benchmark plumbing, metrics, output shape | Me. |
| Docs, ADRs, concepts entries | Me, from your decisions. |

<!-- POST-BUILD: if this table did not survive contact, say so plainly and say
what replaced it. Where authorship was handed over, what the rule was
protecting has to be protected another way — name it (e.g. review notes in the
source files, listing the non-obvious invariants each function is positioned
against) and list the two or three that most repay reading. -->

---

## Phase 0 — <name>  ⬜

**Goal:** <the smallest, most isolated thing this phase proves.>

<!-- For any phase carrying real integration or design risk, add a short
"**Why this phase exists — the honest risk:**" paragraph. Name the specific
failure mode it buys insurance against, and what finding it later would cost.
A spike phase that writes no production code is a legitimate phase. -->

- ⬜ <concrete step>
- ⬜ <concrete step — and where a choice was made, one clause on what the
  alternative was and why it loses here>

**Files:** <paths this phase touches>
**Concepts:** <systems/inference concept(s) this phase is teaching — these
become `docs/concepts.md` entries once the phase lands>
**Done when:** <the observable condition — a test passing, a number printed,
a specific behavior reproduced>
**Test:** <what `tests/` file/case covers this, or "none committed — this is a
spike; its output is knowledge, written up below">

<!-- POST-BUILD per phase: flip ⬜ → ✅ in the heading and on each step, and
append to "Done when" what was actually met (test counts, the number printed).
If the phase was a spike, add a "### Phase N findings" subsection — those
findings are the *spec* for a later phase and should be written as one. -->

---

## Phase 1 — <name>  ⬜

**Goal:** <...>

- ⬜ <concrete step>

**Files:**
**Concepts:**
**Done when:**
**Test:**

---

<!-- Repeat Phase N sections as needed — each small enough to build and verify
in a single sitting. A shape that has worked:
     Phase 0  spike the risky seam; write no production code
     Phase 1  the core data structure, no framework in scope
     Phase 2  the second mechanism / the index math
     Phase 3  the framework integration
     Phase 4  the engine or policy that uses it
     Phase 5  the correctness gate, extended adversarially
     Phase 6  the measurement
-->

---

## Open questions / decisions to resolve (owner decides)

Per `claude.md`'s decision protocol: present 2–3 real options, one paragraph
each, with a recommendation — then stop and wait. Do not present a fake choice
where one option is obviously correct, and do not implement the recommendation
while waiting for an answer.

**These are kept verbatim once answered.** Head each with its outcome and ADR;
do not edit the original argument, even where the recommendation was taken and
it now reads as a retrospective justification. It was written before the
answer was known, and that is what makes it worth keeping.

### Q1 — <question>?
<!-- POST-BUILD: **→ Resolved: <option>. ADR-00NN.** <the deciding argument —
especially if it was one the question only half-anticipated.> -->

*(a) <option>* — <what it is, what it costs, what it buys>.
*(b) <option>* — <...>.
**Recommendation: <option>**, because <...>.

### Q2 — <question>?

*(a)* ... *(b)* ...
**Recommendation:** ...

---

## ADRs to write alongside

- **ADR-00NN — <title>.** <one-line context for why this decision needs a
  written record — a real alternative existed and the reasoning is worth
  keeping.>

<!-- POST-BUILD: retitle to "## ADRs written", tick each, and record any
divergence in numbering or subject rather than tidying it away. An ADR the
plan did not foresee is itself a finding: say why it turned out to be needed. -->

---

## Concepts for `docs/concepts.md` (as they arise)

- **<term>** — <terse definition> → `<file path where it's used>`. Source:
  <the reading that explains it properly>.

<!-- POST-BUILD: tick the ones that landed, and add the ones that only appeared
once the numbers were in — those are usually the best entries. -->

---

## How to run this milestone  <!-- POST-BUILD -->

Filled in as the milestone lands. A future reader — including the owner in
three months — should be able to run everything from here without reading the
code.

### The gate — run this before quoting any number

```bash
uv run pytest -q                              # whole suite, <N> tests, ~<T>
uv run pytest tests/test_correctness.py -v    # the gate alone
uv run ruff check .
```

<!-- Note which test files need no model and no GPU — those are the fast inner
loop, and saying so is how the loop stays fast. -->

### See the mechanism run

```bash
uv run python -m engine.<module>
```

<!-- If the milestone's mechanism can be *shown*, show it. A demonstration
block that prints the mechanism doing its thing is worth more than a passing
test for understanding, and costs little. Say what each scene demonstrates, and
have the last one assert, so the demo fails loudly if the claim stops being
true. It is a demonstration, not a test — `tests/` gates the milestone. -->

### Reproduce the benchmark

```bash
uv run python -m bench.<module> <the exact flags of the recorded run>
```

Output lands in `docs/benchmarks/<timestamp>-<slug>/`. <GPU-only? CPU-safe?>

<!-- Provenance rule: the recorded numbers must be reproducible from the
command written here, including any flag whose default differs from what the
recorded run used. If they are not, say so explicitly as a wart to fix. -->

---

## Phase <N> — results, read honestly  <!-- POST-BUILD -->

Full data: `docs/benchmarks/<dir>/`. Methodology: ADR-00NN. <hardware, model,
dtype, commit>.

### Did the milestone achieve what it set out to?

<The headline, with the number, against the Definition of Done. Then the parts
that are less flattering. This section is where the plan earns its keep:>

- **What must not be credited to this milestone.** <effects that came from
  something else and would be misattributed.>
- **The cost, stated plainly.** <what got worse — every real change trades
  something. If nothing got worse, the measurement is probably not isolating
  the change.>
- **Where the effect collapses, and why.** <the regime in which the claim stops
  holding, and whether that is a fact about the mechanism or about the
  workload.>
- **Limits on the headline.** <what would have to be true for the number to
  generalize.>
- **Suspected defects — flagged, not fixed.** <ranked candidates, and the
  profile or experiment that would settle it. Flagging honestly beats fixing
  silently.>
- **Key pointers for a future reader.** <`file:line` for the two or three
  places that carry the real logic.>

---

## Definition of done (whole milestone)

- ⬜ All phases above ticked.
- ⬜ `tests/test_correctness.py` passes (or the milestone's equivalent gate).
- ⬜ Benchmark numbers (if any) recorded in `docs/benchmarks/` with commit +
  hardware.
- ⬜ `docs/progress.md` updated.
- ⬜ Any ADRs from this plan written and linked.
- ⬜ `docs/roadmap.md` build boxes and DoD for this milestone ticked (theory
  boxes are the owner's).
- ⬜ `README.md` notes any standing simplification this milestone introduces.

---

## Review summary (for the owner)  <!-- PRE-BUILD; superseded by the Retrospective -->

**Phases:** <one line per phase, in order — the whole plan at a glance.>

**The decisions I most want your call on before I build:** <the 1–3 open
questions that actually shape the interfaces, not the small ones.>

Nothing is implemented yet — this is the plan. Say go (or redirect) and I'll
start at Phase 0.

---

## Retrospective  <!-- POST-BUILD -->

**Did it work?** <yes/no, with the number, and a pointer to the honest reading
above.>

**What the plan got right.** <predictions that held — especially ones that held
*including their sign* — and phases whose insurance paid for itself.>

**What the plan got wrong, in order of how much it cost.**

1. **<the costliest wrong assumption>** — <what it was, what it cost, how it
   was fixed. If the fix was additive rather than a replacement, say so.>
2. **<...>**

**What is still open, carried into M<NN+1>.**

- <the flagged-not-fixed items, and why they matter *specifically* in the next
  milestone.>
- <anything verified under conditions the next milestone will change.>
