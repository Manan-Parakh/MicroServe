Write a deep, placement-ready bite for each topic in: $ARGUMENTS

`$ARGUMENTS` is a single topic or a comma-separated list (e.g. `virtual memory, continuous
batching, roofline model`). Split on commas, trim whitespace, produce **one self-contained bite
per topic**. If a topic is ambiguous, pick the reading most relevant to GPU/inference systems
engineering and note the assumption in one line.

## Who this is for

Per `claude.md`: the reader already knows ML/DL, LLMs, and RAG well, and has recently covered OS
fundamentals (processes, threads, scheduling, syscalls, virtual memory) and async Python.
**Do not re-explain these at the ML/LLM level** (what a model, embedding, or RAG pipeline is). The
systems/GPU/inference layer is the weak spot and growth target — they're going deep there to be
placement-ready for **ML Systems / Inference / GPU Infra Engineer** roles: what a thing is, the
mechanics under the hood, how it behaves under load/failure, where it slots into an inference
engine, and how it shows up in interviews.

**Scope discipline:** go deep on what an inference-systems engineer actually needs — mechanics,
trade-offs, failure modes, how it's used in a real serving stack, how it's asked about in
interviews. Respect this project's own guardrails (`claude.md`'s Locked Decisions and each
milestone's guardrail): no custom CUDA/Triton kernel-authoring depth, no PTX/SASS/occupancy
tuning, no distributed inference or training, no RAG/vector-index internals, no FastAPI/serving-
layer depth. If a milestone guardrail caps a topic in the codebase, cap the bite the same way — a
bite can still *name* what's out of scope without going deep on it. If a topic is huge (e.g. "GPU
memory hierarchy"), scope to the slice MicroServe actually touches and say so.

## Step 1 — Pick (or create) a category

Bites live directly under `docs/bites/`, organized into category sub-folders (never loose in the
root, and never behind an extra `concepts/` wrapper). Current categories:

| Folder | Scope |
|--------|-------|
| `gpu-and-hardware/` | GPU compute & memory internals — SM, warp, CUDA core, register file, HBM/SRAM, roofline model, Tensor Cores |
| `os-and-memory/` | OS primitives that MicroServe's design borrows from — virtual memory, paging, page tables, mmap, processes/threads/scheduling |
| `inference-serving/` | The serving-engine core — prefill vs. decode, continuous batching, KV cache, PagedAttention, block allocation, quantization |
| `benchmarking-and-metrics/` | Throughput, TTFT, TPOT, latency percentiles, Poisson workload generation, how to read a benchmark honestly |
| `python-systems/` | Concurrency model as it actually shows up in `engine/` — sync vs. async, GIL, why the engine loop is plain synchronous Python |

Choose the single best-fitting folder. **If none fits, create a new sub-category** — short,
lowercase, kebab-case (e.g. `quantization/`, `attention-kernels/`). A new folder is cheaper than a
bad taxonomy. If you create one, register it in `docs/bites/README.md`'s category table too.

Name the file `docs/bites/{category}/{kebab-case-topic}.md` (the slug). Cross-link related bites
in the body with `[[slug]]` wiki-links (slug only, no path — links are path-independent so files
can move between categories without breaking).

**Relationship to `docs/concepts.md`:** `concepts.md` stays terse and is only added to once a
concept is actually *built* in code. A bite can exist well before that — it's fine (expected,
even) to write a bite for a milestone's *reading* before any code lands. If a bite's topic later
gets its own `concepts.md` entry, cross-reference the bite from there (`**See also:**
docs/bites/{category}/{slug}.md`) instead of duplicating the depth.

## Step 2 — Write each file in this exact structure

Drop a section only if it genuinely doesn't apply (e.g. no Comparison Table if there's nothing
confusable to compare) — otherwise keep all of them. Keep the whole file readable in a few
minutes; depth over breadth, but don't pad.

---

# {Topic Name}

> One-sentence definition a smart non-expert could understand.

---

## The Problem

Why this concept exists — what pain, limitation, or trade-off in systems/inference engineering
made it necessary. Ground it in a real, specific scenario (ideally one MicroServe itself hits or
will hit), not abstract theory. A tight paragraph, not an essay.

---

## Mental Model

2–4 lines (or a small ASCII diagram) for how to picture it. The core abstraction — what you hold
in your head when reasoning about this under pressure.

---

## How It Works

The mechanics that matter: data flow, lifecycle, key internals. Enough to reason about it, not
reimplement it. **If the topic has multiple genuinely competing approaches** (e.g. eviction
policies, batching strategies), structure this section as one sub-header per approach instead,
each covering: what it is, when to use it, trade-offs (latency, complexity, failure modes, memory
cost), and a real system that uses it (vLLM, TensorRT-LLM, HF `generate()`, etc.). Don't force
multiple approaches where only one mechanism exists.

---

## Usage

A minimal, runnable example — code block, real syntax (Python/CLI/config, whatever fits). Show the
80% common path, then the one or two flags/options that matter most. Prefer a 5-line example over
a paragraph describing the API.

---

## Where This Fits in MicroServe

Concrete placement in *this* codebase: which module it lives in or motivates
(`engine/sequence.py`, `engine/kv_cache.py`, `engine/scheduler.py`,
`engine/model_runner.py`, `engine/engine.py`, `bench/`), and which milestone in
`docs/roadmap.md` it belongs to. If it's read-only background (not something MicroServe builds —
e.g. Tensor Core internals), say that plainly instead of forcing a fake tie-in.

---

## Gotchas & Trade-offs

Bulleted list of what trips people up — the stuff that isn't in the docs. Include: common
misuses, interaction with other concepts, things that look right but aren't, failure modes,
footguns, memory/latency/throughput trade-offs, scaling cliffs.

---

## Comparison Table

Only include when there are easily-confused siblings (variants, competing systems, similar
flags). Columns: option, what it is, when to pick it. Omit this section entirely if nothing fits.

---

## Interview & Career Angle

What mastering this unlocks: the kinds of ML-systems/inference-infra interviews, system-design
questions, or senior engineering decisions that require fluency in it. Then list 2–4 adjacent
concepts worth learning next, each with a one-line reason.

---

## Hard Questions I Should Be Able to Answer

3–5 **scenario** questions that go *beyond definition* — the real test of whether the knowledge
is yours. Tailored to someone who has built (or is building) an LLM inference engine — scheduler,
KV cache, batching, model runner — and is prepping for ML systems / inference engineering
interviews. Every question must force a **decision, trade-off, debugging judgment, or design
call** — plain recall ("what is X?") is banned; "throughput craters under load, how do you find
and fix it?" is the bar. Mix across these angles (pick the 3–5 that genuinely matter most for this
concept, not one of each mechanically):

- **Practical / real-world**: something you'd actually hit building or operating an inference
  engine like this one.
- **Interview / system-design**: the senior-level question this concept shows up as, phrased as an
  interviewer would.
- **Engine-internals**: how it surfaces when building *the engine itself* — scheduler policy,
  allocator design, model-runner boundary — not "how do I call a framework," since MicroServe has
  no framework wrapping it.
- **Debugging / "it's on fire"**: a judgment call under a production incident, where relevant.

Phrase each as a prompt to yourself. **Do NOT answer them in this section.**

---

## Answers

Answer each question above, numbered to match. A tight paragraph or a few bullets each: the model
answer, grounded in real practice, naming the trade-off or the concrete steps. Be honest where
there's no single right answer — show the reasoning, not just the verdict.

---

## Rules of Thumb

2–4 `>` blockquote heuristics for real decisions ("default to X unless Y").

---

## TL;DR

Three bullet points. The only things you must remember about this concept.

---

**Jargon callouts:** when a bite leans on *another* tech-heavy term the reader likely doesn't know
yet — a GPU/OS term, a sibling technique, an acronym (e.g. `SM`, `HBM`, `mmap`, `PTX`) — drop a
one-line callout the first time it appears:

> 🔹 **term** — plain-English meaning.

Keep it to a single sentence; it's a heads-up, not a detour. Don't gloss terms already assumed
known (model, embedding, tokenization, attention, transformer, RAG) or the term the bite itself is
about.

## Step 3 — Update `docs/bites/README.md` (REQUIRED, every time)

After writing the file(s), you MUST update `docs/bites/README.md` so the map and backlog stay
accurate:

1. **Add to the Concept Map**: add an **unchecked checklist item** for each new file under its
   category heading, in the exact format
   `- [ ] **[{slug}.md](./{category}/{slug}.md)** — {short topic blurb}`, matching the style of
   existing rows (never pre-check it — `[x]` is the reader's to flip). If you created a **new
   category**, add a new `### {emoji} {folder}/ — {scope} · \`0 / 0\`` heading + checklist, and add
   the folder to the Categories table.

   **Do NOT hand-edit any counts, bars, %, or the heading `· \`N / M\`` tokens** — those are
   *derived* and regenerated by `docs/bites/update_progress.py`. After adding the checklist
   item(s), run `python docs/bites/update_progress.py` once; it recomputes every number (including
   each heading's `· \`N / M\`` token and the overall progress bar) from the checkboxes.

2. **Remove from "Proposed Next Concepts"**: if the topic you just wrote appears there (exact or
   near match), delete that bullet — it's now written and lives in the map.

3. **Append newly-surfaced proposals**: from the Interview & Career Angle "adjacent concepts" list
   in the file you just wrote, add any that are **not already** written (not in the map) and **not
   already** in Proposed Next Concepts, as `◦ **Name** — one-line reason` bullets under the right
   category.

Keep edits surgical and match existing formatting exactly. The README is the single index for
tracking this reading — it must never drift from what's actually under `docs/bites/`.

## Saving

Write directly to `docs/bites/{category}/{slug}.md` and update `docs/bites/README.md` — no need
to ask first, this is the standing workflow for this folder. (This is a note-taking/reading aid,
distinct from `claude.md`'s decision protocol for the engine's own code — writing a bite is never
a "significant decision" requiring an ADR.)
