# MicroServe — Project Roadmap

Merged from `docs/reference/microserve-build-plan.pdf` (the build plan) and
`docs/reference/inference-engineering-roadmap.pdf` (the theory roadmap,
"Roadmap Two"). This file is the single checkboxed source of truth for *what
gets built and what gets read, in what order*. The reasoning for *why* it's
shaped this way, the decision protocol, and the locked/open decisions all live
in `claude.md` — this file doesn't repeat them.

**How to use this checklist:**
- Check a **theory** box only after actually reading it — and only right
  before the milestone that needs it. Reading ahead "to be safe" defeats the
  point (see `claude.md`'s "Reading, just in time").
- Check a **build** box only once it's done and passing the milestone's
  correctness gate. Build boxes here are milestone-level; the step-by-step
  breakdown for whichever milestone is *current* lives in its own
  `docs/plans/mNN-<name>-plan.md`, written just before that milestone starts.
- Tick the milestone's own box in the Master Map once its Definition of Done
  is fully met.

---

## Master map

| # | Milestone | Build | Theory gate |
|---|---|---|---|
| ✅ 0 | Setup | Repo, tiny model, correctness harness | Nothing new |
| ✅ 1 | Naive baseline | One-request-at-a-time engine | Nothing new |
| ⬜ 2 | Continuous batching | Scheduler, iteration-level scheduling | Phase 2 (prefill/decode), Orca paper |
| ⬜ 3 | Paged KV cache | Block allocator, block table, gather-then-attend | Phase 1 (GPU memory), Part A1 (paging), PagedAttention paper |
| ⬜ 4 | Quantization | 4-bit loading, re-benchmark | Phase 3 quantization section only |
| ⬜ 5 | Benchmark suite | Poisson workload, 4-way comparison chart | Phase 6 TTFT/TPOT framing only |
| ⬜ 6 | Prefix sharing (stretch) | Reference-counted blocks, copy-on-write | PagedAttention §4 (re-read) |

**The two rules that hold in every milestone** (full text in `claude.md`):
correctness before performance, and depth is capped deliberately by each
milestone's guardrail. If the timeline slips, it slips on Milestone 6 —
never on Milestone 3.

---

## Deferred entirely — not read or built at any point in this plan

Each has a specific future trigger. Not cut permanently, just out of scope for
this project's six milestones.

| Topic | Trigger that would bring it back |
|---|---|
| Phase 5 — distributed inference (tensor/pipeline parallelism) | A future project with a model too large for one GPU |
| Part A3 — vector indexes (HNSW/IVF), Redis-style KV caching | Act Two, when a RAG layer goes on top of this engine |
| Phase 4 (of Roadmap Two) — installing/running vLLM as a black box | Repurposed, not skipped — see M3's theory list below |
| CUDA/Triton kernel authoring, Tensor Core internals, PTX/SASS, occupancy tuning | A future moment where profiling reveals a bottleneck that genuinely needs a custom kernel |
| FastAPI/HTTP serving layer, streaming, cancellation (Phase 6 of Roadmap Two) | Act Two — a thin wrapper once the engine is done and benchmarked |
| Distributed *training* (ZeRO, FSDP, gradient sync) | Different problem, different bottlenecks — not this project |
| Full DBMS theory (normalization, query optimization, transaction isolation) | Only if building the *application* around the model later |
| Compiler-level optimization (`torch.compile` internals, Triton-the-language, XLA) | A future specialization, not this project |

---

## Part A — Supplementary foundations (read opportunistically, not gated)

Three short OS/DBMS ideas that resurface constantly in this project. Unlike
the milestone theory below, these aren't strictly gated to one milestone —
read each when it naturally comes up.

- ⬜ **A1 — Virtual memory & paging.** OSTEP Ch 13, 15, 18–19, 21–22
  (`pages.cs.wisc.edu/~remzi/OSTEP/`). Skim the mechanics; keep the concept:
  physical memory is chopped into fixed-size pages, a page table maps logical
  addresses to wherever those pages actually live. **This is the exact idea
  PagedAttention borrows — read this immediately before Milestone 3, not
  before.**
- ⬜ **A2 — Memory-mapped files.** `man mmap` + a short blog explainer
  (~15 min). Why `.safetensors` loads fast: it's memory-mapped, not read —
  pages fault in lazily as layers are used. Relevant context for Milestone 0
  (loading the tiny model) and worth having before Milestone 4 (loading a
  real 7B checkpoint), but not a hard gate on either.
- **A3 — Indexing & key-value stores (HNSW/IVF, Redis).** Deferred entirely —
  see table above. Not read at any point in this plan.

---

## Milestone 0 — Setup and the correctness harness  ✅

*Everything after this milestone is judged against the test written here —
get it right once, trust it for the rest of the project.*

**Theory:** ✅ Nothing new — this milestone is pure engineering setup.

**Build (project code):**
- ✅ Repo skeleton: `engine/`, `bench/`, `tests/`, `docs/` (docs already in
  place as of this reorganization).
- ✅ Tiny model running through plain HF `generate()` — Qwen2.5-0.5B-Instruct
  (ADR-0002; model choice itself was settled by the build plan for iteration
  speed, not realism).
- ✅ Correctness test: fixed random seed, greedy decoding, output tokens
  stored as a golden reference.

**Build (environment/SE hygiene — "what a good software project has"):**
This is the part of Milestone 0 driven by general best practice rather than
the build plan PDF, per the owner's explicit request that setup come first
and be done properly:
- ⬜ `git init`, first commit. `git init` is done; the first commit is
  deliberately left to the owner (commits are never made without an explicit
  ask, per this project's own tooling rules).
- ✅ Python dependency management chosen and configured — `uv` (ADR-0001).
- ✅ `pyproject.toml` pinning real versions (`torch`, `transformers`,
  `accelerate`; dev group `pytest`, `ruff`), locked via `uv.lock`.
- ✅ `pytest` configured (`[tool.pytest.ini_options]` in `pyproject.toml`).
- ✅ Lint/format tooling chosen and configured — `ruff` (ADR-0001).
- ✅ `README.md` at repo root: what the project is, how to run it, current
  status.
- ⬜ `.env.example` — not added; no config needs secrets yet (no API keys in
  `engine/`), matching this item's own "likely none at this stage" note.
- ✅ `LICENSE` — MIT.
- ⬜ Pre-commit hook or CI stub — deferred, per this item's own "could be
  deferred without harm" note.

**Guardrail:** Don't install vLLM, Triton, or TensorRT-LLM yet. Don't pick
the eventual 7B model yet — that decision belongs to Milestone 4, once
quantization is in place and it actually fits the available VRAM.

**Definition of done:** ✅ `pytest tests/test_correctness.py` passes,
comparing the engine's output tokens to HF's reference output, identical
seed, identical prompt.

---

## Milestone 1 — Naive baseline  ✅

*Deliberately bad, on purpose — this is the "before" number every later
milestone is measured against.*

**Theory:** ✅ Nothing new — everything needed already exists from Milestone 0.

**Build:**
- ✅ `engine.py` v0: a minimal `Sequence` object, one request handled at a
  time, wrapping HF's own generation loop and cache.
- ✅ No scheduler, no custom KV management yet — resisted, per guardrail.

**Guardrail:** Resist building the scheduler or block allocator here even if
it's tempting to get ahead — this milestone exists solely to produce a
comparison baseline; added complexity now just makes that baseline murkier.

**Definition of done:** ✅ Correctness test passes. ✅ Throughput and latency
recorded and saved (`docs/benchmarks/2026-08-19-m1-naive-baseline/`) — this
number appears on every chart from Milestone 5 onward. Note: recorded as
`commit: uncommitted` since no git history exists yet — re-run once the first
commit lands so the number carries real provenance (see
`docs/benchmarks/README.md`'s "numbers without provenance are not quotable").

---

## Milestone 2 — Continuous batching  ⬜

*The scheduling half of what makes vLLM fast — isolated here, deliberately,
before memory management enters the picture in Milestone 3.*

**Theory:**
- ⬜ **Prefill vs. decode** (Phase 2 of Roadmap Two). Prefill processes the
  whole prompt in one parallel pass — compute-bound. Decode generates one
  token at a time, memory-bandwidth bound. This distinction is *why* prefill
  and decode get separate code paths in the scheduler.
- ⬜ **KV cache formula, in numbers**: `2 × num_layers × num_heads ×
  head_dim × seq_len × batch_size × bytes_per_param`. Exercise: run it for a
  7B, 13B, and 70B model at 4K/32K context, batch 1/32, fp16 — watch the
  numbers stop being "large" and become "larger than the GPU."
- ⬜ **Orca paper** (via the Anyscale continuous batching writeup) —
  iteration-level scheduling, and why static batching collapses under
  variable output lengths.
  `anyscale.com/blog/continuous-batching-llm-inference`

**Build:**
- ⬜ A real `Sequence` with waiting/running/finished state.
- ⬜ A `Scheduler` implementing iteration-level scheduling — evicting
  finished sequences and admitting new ones every decode step, not every
  full batch.
- ⬜ Prefill (new sequences) and decode (running sequences) run as two
  separate forward calls, not fused — note this simplification in the
  README.
- ⬜ Each sequence still reserves a simple contiguous KV buffer;
  fragmentation is expected and correct here.

**Guardrail:** Do not implement paged or block-based memory yet — accepting
fragmentation here is correct, not lazy; it isolates the scheduling problem
so Milestone 3 can isolate the memory problem. Do not attempt to fuse
prefill and decode into one kernel call.

**Definition of done:** ⬜ Correctness test still passes. ⬜ A benchmark
demonstrating continuous batching beating static batching on throughput and
GPU idle time under a workload with variable output lengths.

---

## Milestone 3 — Paged KV cache · THE BIG ONE  ⬜

*Budget the most time here. This is the one line on the resume almost nobody
else building a portfolio project has actually implemented.*

**Theory:**
- ⬜ **Phase 1 — GPU memory hierarchy.** Modal GPU Glossary
  (`modal.com/gpu-glossary`) — **only** the *Device Hardware* (SM, CUDA core,
  warp scheduler, register file) and *Device Software* (thread, warp, block,
  grid, memory hierarchy) sections. Skip the host-side API pages.
- ⬜ **Horace He, "Making Deep Learning Go Brrrr From First Principles"**
  (`horace.io/brrr_intro.html`) — the roofline model: every op is
  compute-bound or memory-bandwidth-bound; most deep learning ops are
  memory-bound. Exercise: classify a matmul, a ReLU, a layernorm, an
  elementwise add, and softmax before checking the essay's own answers.
- ⬜ **Part A1 — virtual memory and paging** (see above) — not an analogy
  exercise; PagedAttention's block table *is* an OS page table.
- ⬜ **PagedAttention paper** (Kwon et al., 2023, `arxiv.org/abs/2309.06180`)
  — read fully. Short, and every design choice in this milestone traces back
  to it. Memory utilization goes from ~40% to over 90%, which converts
  directly into more concurrent requests per GPU.
- ⬜ **vLLM's scheduler and block-manager source** — read as a design
  reference during this milestone. Read it, don't run it, don't copy it.
  (This is Roadmap Two's Phase 4, repurposed rather than skipped.)

**Build:**
- ⬜ `BlockAllocator` managing one pre-allocated KV tensor pool, shape
  `[num_blocks, 2, num_layers, num_heads, block_size, head_dim]`, with a
  free-block stack and reference counts.
- ⬜ Each `Sequence` gets a `block_table` mapping logical position → physical
  block id.
- ⬜ At attention time: gather the sequence's scattered blocks into a
  contiguous scratch tensor, then call `scaled_dot_product_attention` on it
  — no custom kernel.
- ⬜ Block size as a tunable constant, starting at 16 tokens/block (vLLM's
  default).

**Guardrail — the one most worth reading twice:** Do **not** write a custom
CUDA or Triton kernel that reads scattered blocks directly from memory —
gather-then-attend with PyTorch's own `scaled_dot_product_attention` proves
the same idea without kernel authorship, which is a separate, much larger
project. Do **not** go into Tensor Core instruction-level behavior,
PTX/SASS, warp-level programming, or occupancy tuning.

**Definition of done:** ⬜ Correctness test passes on the paged code path.
⬜ A memory-utilization benchmark showing higher achievable concurrent batch
size at the same VRAM budget, contiguous allocation vs. paged.

---

## Milestone 4 — Quantization  ⬜

*The interesting result isn't "uses less memory" — it's how many more
concurrent sequences the paged allocator now fits in the same VRAM.*

**Theory:**
- ⬜ **Phase 3 quantization section only** (Roadmap Two): what weight-only
  post-training quantization means; the names GPTQ and AWQ, at the level of
  knowing what `bitsandbytes` is doing and *why* reducing bytes-per-parameter
  speeds up a memory-bandwidth-bound workload specifically. Not their
  internal calibration math.

**Build:**
- ⬜ 4-bit weight loading via `bitsandbytes` (NF4).
- ⬜ Re-run every benchmark from Milestones 1–3 under quantization.
- ⬜ Pick and load the real target model here — a 7B-class model, 4-bit
  (this is where the eventual headline model gets chosen, per `claude.md`
  Open Decision #3).

**Guardrail:** Do not implement a custom quantization kernel. Do not study
GPTQ's calibration procedure or AWQ's activation-aware weight-selection
algorithm in depth — consumer of these techniques here, not implementer.

**Definition of done:** ⬜ A chart showing memory footprint before/after
quantization, and the resulting increase in maximum concurrent sequences at
fixed VRAM — the number that ties Milestones 3 and 4 together.

---

## Milestone 5 — Benchmark suite  ⬜

*This is the artifact people actually look at. A clean chart earns more
credibility than any paragraph describing the project.*

**Theory:**
- ⬜ **Phase 6 TTFT/TPOT framing only** (Roadmap Two): time-to-first-token
  reflects prefill; time-per-output-token reflects decode. That's the whole
  reading requirement — the rest of Phase 6 (FastAPI/serving) isn't needed
  until serving becomes Act Two.

**Build:**
- ⬜ `bench/workload.py` — Poisson-process request arrivals (the standard
  choice in the Orca and vLLM papers), prompt/output lengths sampled from a
  small ShareGPT-derived set, not uniform random.
- ⬜ `bench/benchmark.py` — drives `engine.step()` under this workload,
  records throughput, TTFT, TPOT, p50/p99 latency.
- ⬜ One chart: all four metrics across all four engine versions (naive →
  batched → paged → quantized) at increasing request rate.

**Guardrail:** Don't build a full statistical analysis pipeline or a
dashboard. Four metrics, four engine versions, one chart — that's what a
reviewer will actually look at.

**Definition of done:** ⬜ A README-ready comparison chart plus the raw
numbers behind it, committed alongside the benchmark script that produced
them, filed in `docs/benchmarks/`.

---

## Milestone 6 — Prefix / block sharing (stretch)  ⬜

*Still core-engine work, not serving — the natural extension of Milestone
3's block allocator, and PagedAttention's other headline idea.*

**Theory:**
- ⬜ **Re-read PagedAttention §4** (memory-sharing / copy-on-write) —
  already read the paper in Milestone 3; this is a focused re-read, not new
  material.

**Build:**
- ⬜ Reference-counted blocks: sequences sharing an identical prompt prefix
  point at the same physical blocks instead of duplicating them.
- ⬜ Copy-on-write when a shared block needs to diverge.
- ⬜ A simple prefix-matching check on incoming requests to detect the
  sharing opportunity.

**Guardrail:** Do not build the RAG layer, a vector index, or any retrieval
logic here — that's Act Two. This milestone is only about block reuse inside
the engine that already exists.

**Definition of done:** ⬜ A measured reduction in prefill time on a workload
with repeated/overlapping prompt prefixes, compared to Milestone 4's
non-sharing baseline.

---

## Concepts read but not built (context only, no milestone owns these)

Two Phase 3 (Roadmap Two) techniques come up in reading but are **not**
implemented as their own milestone — know the shape of the idea, nothing more:

- **FlashAttention** — tiles attention computation so Q/K/V blocks stay in
  fast on-chip SRAM instead of materializing the full N×N matrix in slow
  HBM. Direct exploit of the SRAM/HBM bandwidth gap from Phase 1. MicroServe
  gets this for free via PyTorch's `scaled_dot_product_attention` in
  Milestone 3 — not hand-implemented.
- **Speculative decoding** — a small draft model proposes tokens ahead, the
  large model verifies them in one parallel pass. Answers decode's
  sequential-latency problem. Know the shape; not built in this project.

---

## Quick links

| Resource | URL |
|---|---|
| OSTEP | pages.cs.wisc.edu/~remzi/OSTEP/ |
| Modal GPU Glossary | modal.com/gpu-glossary |
| Making DL Go Brrrr — Horace He | horace.io/brrr_intro.html |
| FlashAttention paper | arxiv.org/abs/2205.14135 |
| PagedAttention / vLLM paper | arxiv.org/abs/2309.06180 |
| vLLM source (reference during M3) | github.com/vllm-project/vllm |
| Continuous batching — Anyscale | anyscale.com/blog/continuous-batching-llm-inference |
| NPTEL — DBMS (IIT Kharagpur, A3 only, deferred) | nptel.ac.in/courses/106105175 |

---

## Next step

Per `claude.md`'s working approach: before touching code, Milestone 0 gets
broken into its own sub-phase implementation file at
`docs/plans/m00-setup-plan.md` (using `docs/plans/template.md`), reviewed and
approved, *then* built step by step together.
