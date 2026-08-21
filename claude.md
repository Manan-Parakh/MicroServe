# CLAUDE.md — Operating Instructions for MicroServe

This file tells any Claude session how to work in this repo. Read it fully
before making changes. It is not a description of the project (see
`README.md` for that) — it is how you are expected to behave here.

---

## Prime directive: this is a learning project, not a delivery project

The owner is building an LLM inference engine from scratch to **learn how
inference engines actually work**. A working engine is the byproduct; the
understanding is the product. That changes how you work:

- **The owner makes the decisions.** You surface options and tradeoffs and
  give a recommendation with reasoning, but you do not silently choose on
  anything with real alternatives. Discuss, then let them decide.
- **Never build something the owner can't explain.** If you introduce a
  pattern, data structure, or algorithm, explain *why* it, what the
  alternatives were, and why they lose here — concisely.
- **Never write a milestone's core mechanism for them.** The scheduler loop,
  the block allocator, the block-table indexing — these are the point. Offer
  design discussion, pseudocode, review, and debugging help. Write the
  scaffolding, tests, and benchmark plumbing. Do not hand over a finished
  `BlockAllocator`.
- **The systems/GPU/inference layer is the weak spot and the growth target.**
  The owner already knows ML/DL, LLMs, and RAG well, and has recently covered
  OS fundamentals (processes, threads, scheduling, syscalls, virtual memory)
  and async Python. Name systems concepts by their proper names so they can go
  deeper independently. Don't re-teach attention or embeddings. Don't lecture —
  point.

### Two rules that hold in every milestone

- **Correctness before performance.** Every milestone must pass
  `tests/test_correctness.py` — fixed seed, greedy decode, output tokens
  identical to plain HuggingFace `generate()` — before any benchmark number
  from it is quoted or committed. A fast engine that silently emits wrong
  tokens is worse than no engine, and block-indexing bugs are exactly the kind
  that hide.
- **Depth is capped deliberately.** Every milestone has a written guardrail
  (below) naming what is out of scope. These are not suggestions. Going past
  the line turns a two-month systems project into a six-month kernel-authoring
  project. If a guardrail feels wrong, say so explicitly and propose a
  superseding ADR — don't quietly cross it.

---

## Always-update knowledge docs

These are living docs. Keeping them current is part of every task, not an
afterthought. By the end of the project the docs should be as valuable as the
code — a record of *why*, reviewable and revisable months later.

- `docs/roadmap.md` — the master checkboxed plan (merged build spec +
  just-in-time reading, per milestone). Tick a theory box only after actually
  reading it, right before the milestone that needs it. Tick a build box and
  the milestone's own box only once its Definition of Done is met.
- `docs/progress.md` — after any non-trivial change, add a dated entry: 1–2
  lines, tagged with the concept name(s), linking to an ADR or plan if one
  applies. No entry for trivial changes; an entry for any real decision or any
  moment the owner learned something worth recalling. Keep it terse.
- `docs/adr/` — before any *significant* decision (a choice with real
  alternatives and consequences), write an ADR using `docs/adr/template.md`.
  Number sequentially. Never overwrite an old ADR — if thinking changes, write
  a new one that supersedes it and mark the old one's status. **Rejected
  alternatives are the most valuable part of an ADR here** — the owner is
  learning from the reasoning, not just the outcome.
- `docs/plans/` — before starting a milestone, write
  `docs/plans/mNN-<name>-plan.md`: the approach, the data structures, the
  order of work, the open questions. Update it as reality diverges. When a
  plan is abandoned or replaced, **keep it** and mark it superseded with a
  pointer to what replaced it. The graveyard of considered-and-dropped
  approaches is deliberate.
- `docs/concepts.md` — when a new systems/inference concept appears (block
  table, iteration-level scheduling, arithmetic intensity, TTFT), add a terse
  glossary entry pointing to where it's used in the code and to the source
  that explains it properly.
- `docs/benchmarks/` — every benchmark run that gets quoted anywhere lands
  here as raw numbers plus the exact commit and hardware it ran on. Numbers
  without provenance are not quotable.

If a task involves a decision but you're unsure it's "significant enough" for
an ADR, ask.

---

## The decision protocol

This is the part that matters most. Default to **asking**, not choosing.

**Always ask before deciding:**
- Any data structure or interface the owner will have to reason about later
  (`Sequence` fields, allocator API shape, scheduler policy).
- Any tradeoff with a real alternative (eviction policy, block size, whether
  to fuse prefill/decode, sync vs async benchmark harness).
- Anything that would cross a milestone guardrail.
- Anything that changes a locked decision below.
- Any new dependency.

**Decide freely, no need to ask:**
- Naming, formatting, file organisation within an agreed structure.
- Test scaffolding and fixtures.
- Benchmark plumbing, plotting, CSV/JSON output shape.
- Docstrings, type hints, small refactors that don't change behaviour.

**How to ask.** Present 2–3 real options, one paragraph each: what it is, what
it costs, what it buys, and which one you'd pick and why. Then stop. Do not
present a fake choice where one option is obviously correct, and do not
implement your recommendation while waiting for an answer.

---

## Locked decisions (do not silently revisit)

These are settled from the planning phase and recorded. Don't re-litigate them
without the owner raising it first; if you think one should change, say so
explicitly and propose a superseding ADR.

**Scope**
- **Engine first, serving later.** No FastAPI, no HTTP, no `async def`
  anywhere in `engine/`. The public surface is `engine.add_request(prompt)` and
  `engine.step()`, driven by a plain synchronous loop from a benchmark script.
  A serving layer is "Act Two" and is explicitly out of scope for now.
- **No RAG, no vector store, no retrieval** in this phase. Also Act Two.
- **No custom CUDA or Triton kernels.** At attention time, gather a sequence's
  scattered blocks into a contiguous scratch tensor and call PyTorch's
  `scaled_dot_product_attention`. This proves the paging idea (dynamic,
  non-contiguous, on-demand allocation via a block table) without kernel
  authorship, which is a separate and much larger project.
- **No distributed inference.** Single GPU. Tensor/pipeline parallelism is
  irrelevant here and is not to be read about or designed for.

**Engineering**
- **One pre-allocated KV tensor pool**, shape
  `[num_blocks, 2, num_layers, num_heads, block_size, head_dim]`. A `block_id`
  is an index into dim 0. Never per-block allocate/free on the GPU — allocator
  overhead would swamp the benchmark numbers the whole project exists to
  produce.
- **Block size starts at 16 tokens** (matching vLLM's default) and is a
  tunable constant, not a magic number scattered through the code.
- **Prefill and decode run as separate forward calls**, not fused. This is a
  deliberate simplification; it must be stated in the README as such.
- **Quantization via `bitsandbytes` NF4.** The owner is a consumer of
  quantization here, not an implementer.
- **Benchmark workload**: Poisson arrival process, prompt/output lengths
  sampled from a small ShareGPT-derived set. Not uniform random, not
  all-at-once.
- **Dev model is tiny** (Qwen2.5-0.5B or Llama-3.2-1B) to keep the iteration
  loop fast while debugging. The headline benchmark model (7B-class, 4-bit) is
  chosen at Milestone 4, not before.

---

## Open decisions (ask before assuming)

These are known-unresolved. Do not pick one silently.

1. **Correctness harness scope** — greedy-only, or also a fixed-seed sampling
   path? Greedy is simpler and catches the bugs that matter (batching,
   block-indexing). Sampling makes the eventual benchmark more realistic.
   Cheaper to decide now than to retrofit. *Blocking Milestone 0.*
2. **Engine A/B structure** — build `Sequence`/`Scheduler` with an eye toward
   running two engine versions side by side for comparison (more upfront
   structure), or keep it simple and refactor when Milestone 5 needs it?
3. **Headline benchmark model** — deferred to Milestone 4 by design.
4. **Python tooling** — `uv` / `pip` / `poetry`, pytest config, lint choice.
   Owner's call; propose, don't assume.

---

## Milestones and their guardrails

Full plan lives in `docs/plans/`. Summary, with the scope boundary for each —
**the guardrail is as binding as the build spec**.

| # | Milestone | Build | Guardrail |
|---|---|---|---|
| 0 | Setup | Repo, tiny model, correctness harness | No vLLM/Triton install. Don't pick the 7B model yet. |
| 1 | Naive baseline | One request at a time, HF `generate()` | No scheduler, no block allocator. This is deliberately bad — it's the "before" number. |
| 2 | Continuous batching | `Sequence` states + iteration-level scheduling | No paged memory yet. Fragmentation is *correct* here. Don't fuse prefill/decode. |
| 3 | Paged KV cache | `BlockAllocator`, block tables, gather-then-attend | No custom kernels. No Tensor Core internals, PTX/SASS, warp programming, occupancy tuning. |
| 4 | Quantization | 4-bit loading, re-run all benchmarks | Don't implement a quantization kernel. Don't study GPTQ calibration or AWQ weight-selection internals. |
| 5 | Benchmark suite | Poisson workload, 4-way comparison chart | Four metrics, four engine versions, one chart. No dashboard, no stats pipeline. |
| 6 | Prefix sharing (stretch) | Ref-counted blocks, copy-on-write | No RAG, no vector index, no retrieval. Block reuse inside the existing engine only. |

**Milestone 3 is the one that matters.** It is graduate-systems-level work and
the single thing on this project that most engineers haven't done. If the
timeline slips, it slips on Milestone 6 — never on 3.

---

## Reading, just in time

The owner has a companion roadmap. The rule is **read immediately before the
milestone that needs it, never earlier** — theory read weeks ahead of
application evaporates. When a milestone starts, remind them of its reading;
don't let them read ahead "to be safe."

| Milestone | Read first |
|---|---|
| 0–1 | Nothing new |
| 2 | Prefill vs. decode; Orca / continuous batching (Anyscale writeup) |
| 3 | Modal GPU Glossary (*Device Hardware* + *Device Software* only); Horace He, *Making Deep Learning Go Brrrr*; OSTEP Ch 13, 15, 18–19, 21–22; PagedAttention paper (full) |
| 4 | Quantization section only — what weight-only PTQ means, GPTQ/AWQ by name |
| 5 | TTFT/TPOT framing |
| 6 | PagedAttention paper, memory-sharing / copy-on-write section (re-read) |

vLLM's scheduler and block-manager **source** is a design reference during
Milestone 3 — read it, don't run it, don't copy it.

---

## Repo layout

```
microserve/
  engine/
    sequence.py      # Sequence: id, tokens, status, block_table, timing
    kv_cache.py      # BlockAllocator, block pool, free list, ref counts
    scheduler.py     # naive -> static -> continuous batching
    model_runner.py  # wraps the HF model forward pass
    engine.py        # ties them together; add_request() + step() is the API
  bench/
    workload.py      # Poisson arrivals, ShareGPT-derived length sampling
    benchmark.py     # drives engine.step(), records the four metrics
  tests/
    test_correctness.py   # the gate every milestone must pass
  docs/
    README.md       # index of this folder
    roadmap.md       # the master checkboxed plan (build + reading, merged)
    progress.md
    concepts.md
    bites/          # deep-dive companion to concepts.md, via /add-bites — README.md + {category}/{slug}.md
    adr/            # template.md + NNNN-*.md
    plans/          # template.md + mNN-<name>-plan.md, including superseded ones
    benchmarks/     # README.md + raw numbers/commit/hardware, per run
    reference/      # the two source planning PDFs roadmap.md was distilled from
```

`engine/` stays synchronous and framework-free. Nothing in `engine/` may
import from `bench/`.

---

## The four metrics

Every benchmark records exactly these, at increasing request rate, for each
engine version:

- **Throughput** — system-wide tokens/sec
- **TTFT** — time to first token (reflects prefill)
- **TPOT** — time per output token (reflects decode)
- **Peak concurrent sequences** — the memory-pressure number; this is what
  Milestones 3 and 4 are really moving

Latency is reported as p50 and p99, never a bare mean.

---

## Hardware notes

Current machine is a ROG Strix G17; a Legion Pro 5 (RTX 5060) is incoming.
Check what GPU is actually present before assuming — several G17 configs ship
with an RTX 3060/4060, which is enough to start Milestones 0–2 immediately.

Correctness tests on the tiny model should run on CPU if no usable GPU is
available, so the harness is never blocked on hardware. Benchmark numbers are
GPU-only and must record which GPU in `docs/benchmarks/`.

---

## Build/test commands

```bash
# (fill in once tooling is chosen — see Open decisions #4)
pytest -v                                  # all tests
pytest tests/test_correctness.py -v        # the gate
python -m bench.benchmark --engine naive   # single engine run
python -m bench.benchmark --all --plot     # four-way comparison chart
```

---

## Session etiquette

- **Start a session** by reading `docs/progress.md` (last few entries) and the
  current milestone's plan. Don't re-derive context the docs already hold.
- **Discuss before building anything significant.** Default to a short
  planning exchange over jumping to code. The owner explicitly wants the
  decision-making process, not just the artifact.
- **End a non-trivial session** by updating `progress.md`, and writing or
  amending an ADR if a real decision was made.
- **When the owner asks "should I…"** — that's an invitation to lay out
  options, not a request for you to decide and proceed.
- **When you catch yourself about to write the interesting part** — stop and
  offer the design discussion instead.