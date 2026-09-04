# M04 — Quantization — Implementation Plan

**Status:** Draft for review — do NOT start implementation until approved.
Five questions below need the owner's call first, and **Q1 (the headline
model) blocks Phase 3 onward**, because the answer decides whether this
milestone runs on the machine that is here today.

**Milestone:** M4 (see `docs/roadmap.md` for the build spec and guardrail).

**Scope:** 4-bit weight-only loading via `bitsandbytes` NF4, plumbed through
the one `ModelRunner` seam all four engines already call; the memory
accounting that converts freed weight bytes into KV-pool blocks; the real
target model, chosen and loaded here for the first time; and a re-run of the
M1–M3 benchmarks under quantization producing the one chart that ties M3 and
M4 together.

**Guardrail (copied from the roadmap, so it's visible while building):**
> Do not implement a custom quantization kernel. Do not study GPTQ's
> calibration procedure or AWQ's activation-aware weight-selection algorithm
> in depth — consumer of these techniques here, not implementer.

This guardrail is the loosest one in the project and it will pull in exactly
one place: when the NF4 numbers come back and the obvious next question is
"why is dequant costing this much?" That question ends in a CUDA kernel. It is
M4's version of M3's fused-gather temptation — note it, don't build it.

**The decisions, and where they are recorded:**

| | question | resolution | ADR |
|---|---|---|---|
| Q1 | which headline model, on 4 GiB of VRAM? | | |
| Q2 | what is the golden reference for a quantized engine? | | |
| Q3 | exact-match, or prefix-match with a divergence point? | | |
| Q4 | what does the M4 chart compare — same model bf16 vs NF4, or dev vs headline? | | |
| Q5 | `bitsandbytes` version pin and Windows wheel | | |

---

## The reading gate

Read immediately before Phase 0, not earlier. M4's list is short by design —
this is the milestone where the project is a **consumer** of a technique, and
the roadmap's own theory box says so.

**The roadmap's theory box is the owner's to tick, not mine.**

1. **Roadmap Two, Phase 3 — quantization section only.** What weight-only
   post-training quantization (PTQ) means: the weights are stored at 4 bits
   and dequantized to the compute dtype on the fly, block by block, as each
   matmul needs them. Activations and the KV cache stay at bf16. **The thing
   to carry into Phase 2 is that last sentence** — it is the whole reason M4's
   Definition of Done is phrased as "more concurrent sequences at fixed VRAM"
   rather than "a smaller KV cache."
2. **`bitsandbytes` NF4, at the level of what it stores.** NF4 = 4-bit
   NormalFloat: a 16-level quantization grid whose levels are placed at the
   quantiles of a normal distribution, on the empirical observation that
   trained weight blocks are approximately Gaussian. Per-block scale factors
   (block size 64 by default) with optional double quantization of the scales
   themselves. You need to be able to say what is stored and what is
   recomputed at runtime — not how the grid was derived.
3. **Why fewer bytes-per-parameter speeds up a memory-bandwidth-bound
   workload specifically.** This is Horace He's roofline framing from M3's
   reading, applied a second time and in the opposite direction. M3 *added*
   memory traffic (the gather) to buy capacity. M4 *removes* memory traffic
   (4 bits per weight instead of 16) and buys both capacity and bandwidth.
   Decode is memory-bandwidth-bound and reads every weight once per step, so
   ~4× less weight traffic per step is the mechanism by which a *slower*
   arithmetic path can still be a faster engine. Re-read the roofline section
   with that specific claim in mind, and predict the sign before Phase 6
   measures it.
4. **GPTQ and AWQ, by name only.** One sentence each. GPTQ: layer-wise
   second-order error compensation against a calibration set. AWQ: scale the
   salient channels identified from activation statistics before rounding.
   You want to know they exist, that they are *also* weight-only PTQ, and that
   they trade a calibration pass for better fidelity than NF4's
   calibration-free grid. **Stop there** — the guardrail names both by name
   specifically so that naming them is enough.

**What is deliberately NOT on this list:** GPTQ's Hessian approximation, AWQ's
search procedure, CUDA dequant kernels, FP8/INT8 activation quantization,
KV-cache quantization (that is a real and tempting technique — it is not on
this project's roadmap at all), and SmoothQuant. If item 3 makes a dequant
kernel look tempting, that is the guardrail's exact failure mode — note it as
a bite in `docs/bites/` and move on.

---

## The core mental model this plan is built around

> A GPU's VRAM holds three things: **weights**, the **KV cache**, and
> transient activations. M3 made the KV cache pack efficiently, but it could
> only pack into whatever was left after the weights. Weight-only NF4 shrinks
> the first term by ~4×, and **every byte it frees is handed to the block
> pool** — so the milestone's result is not "the model is smaller", it is
> "`num_blocks` went up by the freed bytes, and concurrency went up with it."
> M4 does not make the KV cache smaller. It makes there be more room for it.

The measurement that matters is **peak concurrent sequences at a fixed total
VRAM budget, bf16 weights vs NF4 weights, on the paged engine** — not weight
footprint, which is arithmetic anyone can do on paper, and not tokens/sec,
which will move for two reasons at once.

There is a second, weaker claim worth measuring while we are here: decode is
memory-bandwidth-bound, so cutting weight traffic 4× should *also* improve
TPOT — unless dequant overhead eats it. **Predict the sign before measuring**
(see Phase 6's preamble); M3's plan did this and it was the most useful
paragraph in the file.

---

## Locked decisions this plan honors

1. **Quantization via `bitsandbytes` NF4.** The owner is a consumer of
   quantization here, not an implementer (`claude.md`, locked decisions). The
   dependency itself is therefore pre-authorized; only its *version pin* is
   an open question (Q5).
2. **No custom CUDA or Triton kernels.** Unchanged from M3, and it binds the
   dequant path exactly as it bound the gather path.
3. **One pre-allocated KV tensor pool**, explicit `num_blocks`, sized by the
   caller (ADR-0011). M4 changes *what number is passed*, never the mechanism.
4. **Prefill and decode stay separate forward calls.**
5. **`engine.add_request` / `engine.step()` stays the public surface**;
   `engine/` stays synchronous and framework-free and imports nothing from
   `bench/`.
6. **The paged engine is not replaced.** M5 needs all four engines running for
   its four-way chart. Quantization is a *load-time option* on the existing
   engines, not a fifth engine. This is the single most important structural
   decision in the milestone and it follows directly from M3's locked decision
   7 — see Phase 1.
7. **Block size stays 16.** NF4 does not touch the KV cache and therefore has
   no bearing on block size.

**A statement in the roadmap that needs correcting before it costs a run.**
The M4 build spec says "re-run every benchmark from Milestones 1–3 under
quantization." Taken literally that includes M1's naive baseline on the
*headline* model, which on 4 GiB may not be runnable at all, and M3's
128-request validation point, which takes real time. Q4 asks which comparison
the chart actually needs; the answer will make some of those re-runs
unnecessary rather than merely slow. Not a decision to re-litigate — a scope
question the roadmap left implicit.

---

## What exists before this milestone (the seams it cuts into)

- **`engine/model_runner.py`** — `ModelRunner.__init__(model_id, device,
  attn_implementation)` calls `AutoModelForCausalLM.from_pretrained(...)` and
  then `.to(self.device)`. **What M4 needs from it:** a `quantization_config`
  path — and note that a bnb-quantized model **cannot be `.to(device)`-moved
  after loading**; it must be placed by `device_map` at load time. That one
  line is the whole integration, and it is why Phase 0 exists. This is the
  single seam all four engines call through, which is what makes M4 cheap.
- **`engine/kv_cache.py` — `KVSpec.from_config(config, dtype=model.dtype)`,
  called from `PagedEngine.__init__` (`engine/paged_engine.py:53`).** **What
  M4 needs from it:** `model.dtype` under a bnb-quantized model is not
  obviously the KV dtype. The KV cache must stay at the **compute** dtype
  (bf16), and if `model.dtype` reports something else — `uint8`, `int8`, or a
  storage dtype — the pool is silently allocated at the wrong element size and
  **every memory number in the milestone is invalid**. This is precisely the
  shape of M3's `num_heads` / `num_key_value_heads` bug: a plausible-looking
  value that quietly scales the pool. Phase 0's first job is to print it.
- **`engine/paged_engine.py`** — takes `num_blocks` *or* `kv_cache_gib`,
  never both. **What M4 needs from it:** a third way to say it — "here is a
  total VRAM budget; measure the weights, give the remainder to the pool."
  That helper is Phase 2 and it is where the milestone's claim is actually
  computed.
- **`tests/test_correctness.py`** — the gate. 130 tests, pinned to **CPU +
  eager** (ADR-0009, ADR-0010) precisely because GPU kernel reduction order
  makes bitwise-identical greedy decoding unreliable. **What M4 needs from it,
  and cannot have:** `bitsandbytes` NF4 requires CUDA. **The quantized path
  cannot run on the gate as currently constituted.** This is the structural
  problem of the milestone and it is Q2 and Q3.
- **`bench/benchmark.py`** — M1/M2 metrics, `--model-id`, `--max-batch-size`.
  **What M4 needs:** a `--quantize` flag and the weight footprint recorded
  into the output JSON.
- **`bench/paged_benchmark.py`** — the equal-VRAM sweep and the 128-request
  validation point (ADR-0014). **What M4 needs:** the budget conversion in
  ADR-0014 currently splits a *KV* byte budget. M4's claim is about a *total*
  VRAM budget, of which weights are a term. Extending it is Phase 2; whether
  that supersedes ADR-0014 or extends it is a judgement call I'd make while
  writing it, unless you'd rather decide.
- **Carried in from M3, explicitly** (`docs/progress.md`, 2026-09-03): the
  **suspected fixed-overhead defect** in the paged step — ~37 ms/step at batch
  6 where the gather moves almost no bytes, with four ranked candidates.
  Flagged, not fixed. It matters here specifically because quantization
  removes ~4× of the weight traffic this overhead is currently hiding behind,
  so it will be a *larger fraction* of every M4 number. Phase 5.
- **Also carried from M3:** Phase 0 findings 4 and 5 were established on **CPU
  + eager**. The block-padded gather has never been verified under SDPA on
  GPU. Every M4 number comes from the GPU. Phase 0 handles this.

---

## Ownership: who writes what

**The M3 waiver stands.** Partway through M3 the owner elected to hand the
core mechanisms over, trading authorship for review. That is recorded as a
permanent override, not a one-off, so this table is written under it — but the
table matters less this milestone than any before it, for a reason worth
saying out loud:

**M4 has no core mechanism to hand over.** The guardrail defines this
milestone as consuming someone else's technique. The 4-bit path is a
`BitsAndBytesConfig` object and a `device_map` argument. What is genuinely
this project's own work here is *the accounting* — Phase 2's weights-vs-KV
budget split — and *the honest reading* of what the numbers mean, which is
Phase 6. Those are the two places to spend attention.

| Component | Who |
|---|---|
| The VRAM budget split: measure weights, convert the remainder to blocks | **You**, if you want the one piece of real reasoning in this milestone. It is small and it is where the DoD number comes from. |
| Deciding what the chart compares (Q4) and reading the result | **You.** This is the milestone's judgement. |
| `BitsAndBytesConfig` plumbing through `ModelRunner` | Me. Framework plumbing, and the guardrail says so. |
| Correctness gate restructuring for a GPU-only path | Me, from your call on Q2/Q3. |
| Benchmark flags, re-run harness, chart | Me. |
| Docs, ADRs, concepts entries | Me, from your decisions. |

---

## Phase 0 — Spike: does NF4 load, run, and report its dtype honestly?  ⬜

**Goal:** before writing any accounting on top of a quantized model, establish
five facts by printing them. No production code.

**Why this phase exists — the honest risk:** two of the five facts below can
silently invalidate the entire milestone rather than crash it. If
`model.dtype` under bnb is not the compute dtype, `KVSpec` allocates the pool
at the wrong element size and every memory number is wrong while every test
still passes — the exact failure mode of M3's `num_heads` bug, and the exact
reason M3's Phase 0 paid for itself twice. Twenty minutes here; days in Phase
6 otherwise.

- ⬜ Load `Qwen2.5-0.5B-Instruct` with `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
  bnb_4bit_use_double_quant=True)` on CUDA. Confirm it loads at all on this
  box: Windows, `torch 2.13.0+cu130`, RTX 3050 Ti.
- ⬜ **Print `model.dtype`, and separately print the dtype of an actual
  `q_proj.weight` and of a `LayerNorm` weight.** Record all three. Decide from
  the evidence what `KVSpec` must be handed — do not infer it.
- ⬜ Measure the weight footprint honestly: `torch.cuda.memory_allocated()`
  before and after load, and `sum(p.numel() * p.element_size() for p in
  model.parameters())`. **These two will disagree** (bnb `Params4bit` packs two
  weights per byte; `element_size()` on a packed tensor is not the story).
  Record both and note which one Phase 2 should trust.
- ⬜ Confirm `.to(device)` after a 4-bit load fails or is a no-op, and that
  `device_map` is the required path. Write down the exact error text — it is
  the thing a future reader will search for.
- ⬜ Greedy-generate 16 tokens twice from the same prompt on the same model
  and confirm the two runs agree — NF4 dequant must be deterministic, and if
  it is not, Q3 answers itself.
- ⬜ **The M3 carry-over:** run the *bf16* paged engine on GPU under SDPA (not
  eager) and confirm the block-padded gather still matches HF `generate()`.
  M3's Phase 0 findings 4 and 5 were established on CPU + eager only. Do this
  before quantization enters the picture, so a failure has one cause and not
  two.

**Files:** scratchpad only, plus a findings note written into this file.
**Concepts:** weight-only PTQ; NF4; compute dtype vs. storage dtype; packed
tensors and why `element_size()` lies about them.
**Done when:** the five facts are written into "Phase 0 findings" below, and
the bf16-SDPA-on-GPU check has passed.
**Test:** none committed — this is a spike. Its output is knowledge and the
spec for Phase 1 and Phase 2.

---

## Phase 1 — `ModelRunner` learns to load quantized  ⬜

**Goal:** one new argument on the one seam, and all four engines can run
quantized without any of them knowing what quantization is.

**Why the seam holds:** `Engine`, `StaticEngine`, `ContinuousEngine` and
`PagedEngine` all construct a `ModelRunner` and touch `transformers` nowhere
else. That was true by design since M1 and this is the milestone that collects
on it. If this phase needs to touch a fourth file, something is wrong with the
change, not with the file.

- ⬜ `ModelRunner.__init__(..., quantization: str | None = None)` — `None`
  (today's behaviour, unchanged) or `"nf4"`. A string, not a bool: `"int8"`
  is a plausible later value and a bool would have to be widened.
- ⬜ Build `BitsAndBytesConfig` in one place, with the compute dtype named as
  a module constant, not inline. Load via `device_map` per Phase 0's finding;
  keep the unquantized path's `.to(device)` exactly as it is.
- ⬜ Thread the argument through all four engine constructors as a
  pass-through keyword. **No engine may branch on it.** If an engine needs to
  know, that is a finding worth stopping on.
- ⬜ `ModelRunner.weight_bytes` — the measured weight footprint, using
  whichever measurement Phase 0 says to trust. Phase 2 consumes this.
- ⬜ `PagedEngine` must hand `KVSpec` the **compute** dtype, from Phase 0's
  finding — not `model.dtype`, unless Phase 0 proves those are the same thing.
  Whichever it is, say so in a comment naming the failure it prevents.
- ⬜ Add `bitsandbytes` to `pyproject.toml` and lock it (Q5).

**Files:** `engine/model_runner.py`, the four engine constructors,
`pyproject.toml`, `uv.lock`.
**Concepts:** the single-seam property, cashed in.
**Done when:** all four engines construct and generate under
`quantization="nf4"` on GPU, and the full existing test suite still passes
unquantized.
**Test:** the existing suite is the regression check (it must be untouched by
this phase). New quantized cases wait for Phase 4 — the gate needs its Q2/Q3
answer first.

---

## Phase 2 — The VRAM budget: weights measured, remainder to blocks  ⬜

**Goal:** the arithmetic that turns "4 GiB of VRAM" into "this many blocks",
with the weight term measured rather than assumed. This is where M4's headline
number is actually produced.

**This is the phase with real thinking in it** — see *Ownership*. Everything
else in this milestone is plumbing around it.

- ⬜ A sizing helper that takes a **total** VRAM budget, subtracts measured
  weights, subtracts a named activation/workspace reserve, and converts what
  is left through the existing `blocks_for_bytes` (ADR-0011). The reserve must
  be an explicit, named, tunable constant — a fudge factor with a name is
  honest engineering; the same number inlined is a lie waiting to be found.
- ⬜ Decide and document where it lives. It reads a `ModelRunner` and produces
  a `num_blocks`, so `engine/kv_cache.py` next to `blocks_for_gib` is the
  natural home — but it must not import `bench/`, and `PagedEngine` must keep
  accepting a plain `num_blocks` so the sweep stays reproducible (ADR-0011's
  whole point: pool size never depends on ambient GPU state unless the caller
  asks for it).
- ⬜ Print the accounting as a table for the dev model at bf16 and NF4:
  weights, reserve, KV budget, `num_blocks`, token slots. **The ratio of the
  last column between the two rows is the milestone's claim**, computable
  before a single benchmark runs. Write the predicted number down here, then
  let Phase 6 check it.
- ⬜ Note explicitly, in code and in the concepts entry: the *per-token* KV
  cost is **identical** in both rows. Only the number of slots changes. A
  reader who misses this will attribute the win to the wrong mechanism.

**Files:** `engine/kv_cache.py`, `engine/paged_engine.py`.
**Concepts:** VRAM budget decomposition (weights / KV / activations);
weight-only quantization as *indirect* KV capacity.
**Done when:** the table prints, and the predicted concurrency ratio is
written into this plan before Phase 6 runs.
**Test:** `tests/test_kv_cache.py` — pure arithmetic, no model, no GPU. Feed
it fake weight byte counts and assert the block count, including the
degenerate cases (weights exceed budget → a catchable error naming the
shortfall, not a negative block count).

---

## Phase 3 — The headline model  ⬜

**Goal:** pick the real target model, load it 4-bit, prove it fits and
generates. **Blocked on Q1** — the answer decides whether this phase runs on
the machine that is here today.

**The constraint, stated numerically so Q1 can be answered on evidence.** This
box has an **RTX 3050 Ti Laptop with 3.999 GiB** of VRAM (verified, not
assumed). A 7B-class model at NF4 is ~3.5–4.0 GiB of weights *before* any KV
cache, activations, or CUDA context. **A 7B model does not fit on this GPU**,
4-bit or otherwise, and no amount of paging changes that. `claude.md` notes a
Legion Pro 5 (RTX 5060) as incoming; whether it has arrived is the pivot.

- ⬜ Record the chosen model and the reasoning as an ADR (Open decision #3 has
  been deferred since M0 specifically to be answered here, which makes it
  worth a written record even though the choice may look forced).
- ⬜ Load it 4-bit, generate greedily, confirm the output is coherent — a
  model that loads and emits fluent text is the bar here; token-identity
  against a bf16 reference is *not* expected and not the goal.
- ⬜ Run Phase 2's accounting table on it. Weights, reserve, KV budget, blocks,
  token slots — this is the number the milestone reports.
- ⬜ Confirm `KVSpec.from_config` derives correctly for it. Any GQA model with
  a different head/KV-head ratio is a fresh chance for M3's bug to reappear
  in a new costume. Assert the derived per-token KV bytes against a
  hand-computed figure in the ADR.

**Files:** an ADR, `bench/` defaults, `README.md`.
**Concepts:** why model size is chosen by VRAM arithmetic rather than by
preference.
**Done when:** the chosen model loads 4-bit, generates, and its accounting
table is recorded.
**Test:** a smoke test marked GPU-only and skipped without CUDA — this model
is far too slow for the gate's inner loop.

---

## Phase 4 — The correctness gate under quantization  ⬜

**Goal:** a defensible answer to "is the quantized engine correct?", given
that the existing gate's core premise does not survive the milestone.

**The problem, plainly.** The gate compares engine output to HF `generate()`
token-for-token, pinned to **CPU + eager** because GPU kernel reduction order
makes bitwise-identical greedy decoding unreliable (ADR-0009, ADR-0010).
`bitsandbytes` NF4 **requires CUDA**. So the quantized path cannot run under
the conditions that make the gate trustworthy, and the two constraints are
mutually exclusive rather than merely inconvenient.

The way out is to be precise about *what property is being tested*. The gate
has never tested that the model is good; it tests that **our engine's
batching, scheduling, and block indexing do not change the model's output**.
That property is still testable under quantization — against a quantized
golden. Quantization fidelity (does NF4 track bf16?) is a different question,
it belongs to `bitsandbytes` and not to us, and the guardrail says so. **Do
not let the two merge.** See Q2 and Q3.

- ⬜ Implement the Q2 answer: a quantized golden fixture, GPU-only, skipped
  cleanly without CUDA so the CPU inner loop stays green on any machine.
- ⬜ Implement the Q3 answer: the match criterion, whatever it turns out to
  be, in **one** helper — not per-test, where two of them will eventually
  disagree.
- ⬜ Re-run M3's adversarial paged cases (block boundaries, block reuse,
  interleaved sequences, a live preemption) under quantization. **These are
  the tests that matter here.** A quantization bug that is really a KV-dtype
  bug will show up as a block-boundary failure, not as gibberish.
- ⬜ Keep the whole existing CPU gate untouched and passing. The unquantized
  path is still the milestone's control.
- ⬜ Record in `README.md` and the ADR what the quantized gate does and does
  not prove.

**Files:** `tests/test_correctness.py`, an ADR.
**Concepts:** what a correctness gate is actually asserting; determinism as a
property of the *kernel selection*, not of the model.
**Done when:** the quantized cases pass on GPU, the CPU gate is unchanged and
green, and the suite still passes on a machine with no GPU.
**Test:** this phase is the test.

---

## Phase 5 — The carried-over overhead defect  ⬜

**Goal:** settle M3's flagged fixed-overhead finding before it contaminates
M4's numbers. Either fix it, or measure it and quote it as a known constant —
**both are acceptable outcomes; leaving it unexamined is not.**

**Why it belongs here and not in M3.** From `docs/progress.md`: a paged step
does 21× the work from 6 to 128 sequences for +9% time, implying ~37 ms/step
of fixed overhead at batch 6. Four candidates were ranked: pageable H2D copies
per step in `block_table_tensor` and `scatter_index`; the double copy in
`gather_kv` from `permute().reshape()`; and 24 small per-layer gathers. NF4
cuts weight traffic ~4×, which is the traffic this overhead currently hides
behind — so the same absolute cost becomes a much larger share of every M4
number, and an unexamined constant would be silently attributed to
quantization.

- ⬜ Profile a paged step at small batch with `torch.profiler`, quantized and
  not. One profile, four hypotheses, ranked — the experiment M3 already
  specified.
- ⬜ **Decide with the owner before fixing anything.** The cheap fixes (pinned
  buffers, hoisting the H2D copies out of the per-layer loop) are in scope.
  Anything that ends in a fused kernel is not — that is M3's guardrail, still
  binding.
- ⬜ Whatever is decided, record the before/after with provenance. If it is
  fixed, M3's recorded numbers do not change — they were measured on M3's
  commit and stay quotable as such; say so rather than re-running them.

**Files:** `engine/paged_cache.py`, `engine/paged_scheduler.py`, a scratch
profiling script.
**Concepts:** fixed overhead vs. bandwidth-proportional cost; why a
nearly-flat latency curve is a symptom and not a triumph.
**Done when:** the profile names the dominant term, and the fix-or-quote
decision is recorded here.
**Test:** the existing gate must still pass if anything is touched — this
phase edits the most invariant-dense code in the repo.

---

## Phase 6 — The chart, and the honest read  ⬜

**Goal:** the roadmap's Definition of Done — memory footprint before/after
quantization, and the resulting increase in maximum concurrent sequences at
fixed VRAM. **Blocked on Q4**, which decides what the two bars actually are.

**The prediction, written before measuring** (M3's Phase 6 preamble did this
and it was the most valuable paragraph in that file — the result confirmed it
*including the sign*):

- **Weight footprint:** ~3.5–4× smaller. Not 4× exactly — NF4 stores per-block
  scales, and `LayerNorm`, embeddings, and the LM head are typically left at
  bf16 by `bitsandbytes` defaults. **Predict the gap and check whether the
  measurement lands in it.**
- **Concurrency at fixed total VRAM:** up, by roughly (freed weight bytes) ÷
  (KV bytes per sequence). Phase 2 computes this on paper first; Phase 6
  either confirms or explains the discrepancy. A discrepancy is a *finding*,
  not a failure.
- **TPOT:** the interesting one. Two effects in opposition — ~4× less weight
  traffic per decode step (faster, since decode is bandwidth-bound) against
  per-matmul dequant overhead (slower, and it is arithmetic on the critical
  path). Literature says the traffic wins at small batch. **At a 0.5B model on
  a laptop GPU I would not be surprised to see it lose**, because the weights
  are small enough that the workload may not be bandwidth-bound in the first
  place. Whichever way it goes, the *explanation* is the deliverable.
- **Throughput:** rises if TPOT holds and concurrency rises. If concurrency
  rises and throughput does not, the bottleneck is elsewhere — Phase 5's fixed
  overhead is the first suspect, which is exactly why Phase 5 precedes this.

- ⬜ Extend the existing harnesses with `--quantize` rather than writing a new
  one. M4 adds a dimension to M3's measurement; it is not a new measurement.
- ⬜ Record `sys.argv` into every results JSON. **This is M3's outstanding
  provenance wart** — its recorded run used `--validate-requests 128` against
  a default of 48, so running the tool with its own defaults does not
  reproduce the recorded numbers. One line. Fix it here.
- ⬜ Produce the one chart the DoD asks for. One chart. The roadmap says four
  metrics and one chart at M5; M4's is a single before/after.
- ⬜ Write "Phase 6 — results, read honestly" into this file, per the template:
  what must not be credited to quantization, what got worse, where the effect
  collapses, and any suspected defect flagged rather than quietly fixed.

**Files:** `bench/benchmark.py`, `bench/paged_benchmark.py`,
`docs/benchmarks/<timestamp>-m4-quantized/`.
**Concepts:** bytes-per-parameter as a bandwidth term; dequant overhead as
arithmetic traded for traffic.
**Done when:** the chart exists, the numbers are recorded with commit and
hardware, and the honest read is written.
**Test:** measurement, not correctness — Phase 4 is what licenses quoting it.

---

## Open questions / decisions to resolve (owner decides)

### Q1 — Which headline model, given 3.999 GiB of VRAM?

This is `claude.md` Open decision #3, deferred since M0 to be answered exactly
here. The roadmap says "a 7B-class model, 4-bit". **The measured VRAM on this
box makes that impossible**: 7B at NF4 is ~3.5–4.0 GiB of weights alone,
before the KV cache, activations, or the ~300 MiB CUDA context. There would be
no room for the block pool, which would leave M4 unable to demonstrate its own
Definition of Done.

*(a) Drop to a 3B-class model on this GPU — `Qwen2.5-3B-Instruct` at NF4,
~1.8–2.0 GiB.* Leaves roughly 1.5 GiB for the pool, which is a real budget:
enough for a meaningful sweep and enough for the bf16 comparison arm to be
genuinely constrained (3B at bf16 is ~6 GiB and **does not fit at all**, which
is itself a striking result — "quantization is what makes this model runnable"
is a stronger headline than a ratio). Same family as the dev model, so the GQA
shape and tokenizer are familiar and M3's `num_key_value_heads` lesson carries
over cleanly. Costs: "7B-class" was the stated target since M0, and it is the
size the outside world recognizes.

*(b) Wait for the Legion Pro 5 / RTX 5060 and do 7B there.* 8 GiB makes 7B NF4
comfortable — ~4 GiB weights, ~3 GiB of pool — and the bf16 arm is impossible
there too, so the headline is the same in kind but at the recognizable size.
Costs: M4 is blocked on hardware that may or may not be here, and `claude.md`
says to *check* what GPU is present rather than assume. Phases 0–2 and 4–5 do
not depend on this and can proceed on the dev model regardless; only Phase 3
and Phase 6's headline arm block.

*(c) Keep Qwen2.5-0.5B as the headline and report only ratios.* Fastest, and
every mechanism is demonstrated. Costs the milestone its point: at 0.5B the
weights are ~1 GiB bf16 and freeing 750 MiB of that is a real but small
absolute win, and the "quantization is what makes it fit" story disappears
entirely. The dev model stays the dev model regardless — this option is about
whether it is *also* the headline.

**Recommendation: (a) now, with (b) as an upgrade if the Legion arrives before
Phase 6.** The plan is written so this is cheap: Phases 0–2, 4 and 5 all run
on the dev model and are hardware-independent, so nothing is blocked in the
meantime. (a) also produces the better story — an unquantized 3B *cannot run
at all* on this GPU, which makes quantization's contribution categorical
rather than a percentage. **What I need from you:** has the Legion arrived, or
is there a date? That single fact decides between (a) and (b), and it is not
something I can check.

### Q2 — What is the golden reference for a quantized engine?

The gate compares against HF `generate()` on the same model, on CPU + eager
(ADR-0010). NF4 requires CUDA. Both halves cannot hold.

*(a) A quantized golden: HF `generate()` on the same NF4 model on GPU, engine
output compared to it token-for-token.* Tests exactly the property the gate
has always tested — that *our* batching, scheduling, and block indexing do not
change the model's output — and holds the quantization constant on both sides
so it cancels. Costs: GPU-only, so it is skipped on any machine without CUDA,
and it re-enters the GPU non-determinism that ADR-0010 was written to escape
(which is Q3).

*(b) Compare NF4 engine output against the bf16 golden with a tolerance —
perplexity, or an n-token agreement threshold.* Tests something the project
does not own: how faithfully NF4 tracks bf16. That is `bitsandbytes`'s
property, the guardrail explicitly makes us a consumer of it, and a threshold
here would be an arbitrary number we would then have to defend. It also
*hides* our own bugs — a genuine block-indexing error and normal quantization
drift both look like "some tokens differ."

*(c) Don't gate the quantized path; rely on the bf16 gate plus a smoke test.*
Cheapest and not indefensible — quantization is a load-time option and the
engine logic is provably identical. But M3's own experience argues against it:
the KV pool dtype is derived from the model, quantization changes what the
model reports about itself, and that is exactly the class of bug that passes
every test while corrupting every number.

**Recommendation: (a), plus (c)'s honesty about scope.** Same-quantization
golden, GPU-only, skipped without CUDA, with the existing CPU gate untouched
as the control. And state in the ADR that this gate **does not** assert NF4
tracks bf16 — that claim is not ours to make.

### Q3 — Exact match, or prefix match with a reported divergence point?

Q2(a) reintroduces the problem ADR-0010 solved by fleeing to the CPU: on GPU,
a batched forward and a batch-of-one forward may reduce in different orders,
so two mathematically identical computations can differ in the last bit, and
greedy `argmax` turns that into a different token that then diverges forever.
Quantization does not cause this — it just means we can no longer avoid it.

*(a) Demand exact match and see if it holds.* The strongest assertion, and it
may simply work: the engine and the golden run the same kernels on the same
device, and M3's paged cases already pass exactly on CPU. If it holds, take
it. Risk: a flaky gate is worse than a weaker one, and flakiness here would
train us to ignore the gate.

*(b) Assert an exact match on the first N tokens (N ≈ 8–16), and on divergence
report the position and both tokens.* Weaker in principle, far more
diagnostic in practice: a block-indexing bug diverges immediately and
reproducibly, while float-reduction noise diverges late and at a varying
position. **The divergence position is itself the signal** distinguishing the
two, which the exact-match form throws away.

*(c) Keep the paged/batched cases exact on CPU (unquantized, as today) and run
the quantized cases only as prefix checks.* Belt and braces — the sharp
assertion stays where it is reliable, and the quantized path gets the weaker
check appropriate to its noisier environment.

**Recommendation: (a) first, falling back to (c).** Try exact; it costs one
run to find out. If it is flaky, take (c) and record *why* in the ADR — "we
tried exact and it was not stable, here is the observed divergence position
distribution" is a much better record than choosing the weaker bar up front.
This one I can resolve empirically in Phase 0 and report back rather than
having you decide blind, if you prefer.

### Q4 — What does the M4 chart compare?

The DoD asks for "memory footprint before/after quantization, and the
resulting increase in maximum concurrent sequences at fixed VRAM." Two
readings, and they produce different charts.

*(a) Same model, bf16 vs NF4, on the paged engine at a fixed total VRAM
budget.* A controlled experiment: one variable changes, and the concurrency
ratio is attributable to quantization alone. Composes directly with M3's
sweep and ADR-0014's methodology. Costs realism — on the dev model both arms
fit easily, so it measures the mechanism rather than the situation.

*(b) Dev model bf16 vs headline model NF4.* The realistic, product-shaped
comparison — "here is what the machine can actually serve." But two variables
move at once (model size and precision), so no number in it is attributable.
As a *headline* it is fine; as the milestone's measurement it is not.

*(c) Both: (a) as the measurement, (b) as one additional line in the write-up.*
M3 ended up doing exactly this — a controlled sweep for continuity plus a
separate validation point for the real claim — after discovering the plan's
single workload could not exhibit the effect the plan wanted. That was M3's
costliest planning error and it is cheap to not repeat.

**Recommendation: (c).** (a) is the measurement and (b) is the sentence anyone
will actually quote; they answer different questions and the write-up should
say which is which. This also settles the roadmap's "re-run every benchmark
from M1–M3" literally-read scope problem: the M1 naive baseline on a headline
model is a number nobody needs, and (a) makes clear which re-runs earn their
compute.

### Q5 — `bitsandbytes` version and platform

The dependency is pre-authorized by `claude.md`'s locked decisions, so this is
a pin, not a choice of library — but it is a new dependency on a Windows box
with `torch 2.13.0+cu130`, and `bitsandbytes` has historically been the
project's most fragile install on Windows. Options are: pin a known-good
version now, or take the latest and pin whatever works. I would **take the
latest that resolves against this torch/CUDA pair, verify in Phase 0, then pin
exactly** — and record the working combination in the ADR, because "it built
on 2026-09-03 with these three versions" is the kind of provenance that saves
an afternoon later. Flagging it rather than asking: say the word if you'd
rather choose the version yourself.

---

## ADRs to write alongside

- **ADR-0015 — The headline model** (Q1). `claude.md` Open decision #3, open
  since M0 and closed here. Worth a record even if the VRAM arithmetic makes
  the choice look forced, because the arithmetic *is* the reasoning and it is
  reusable when the hardware changes.
- **ADR-0016 — The correctness gate under quantization** (Q2 + Q3).
  Supersedes nothing but **amends ADR-0009 and ADR-0010**, whose CPU pin was
  written before a CUDA-only code path existed. Must state explicitly what the
  quantized gate does not prove.
- **ADR-0017 — Total-VRAM budget accounting** (Phase 2). Extends ADR-0014's
  equal-VRAM methodology from a KV budget to a total budget with weights as a
  term. Whether this supersedes 0014 or extends it is a call I'd make while
  writing it — say if you want it decided first.
- **Possibly ADR-0018** — if Phase 5 changes the paged step's hot path, the
  fix and its measured effect need a record, and M3's recorded numbers need an
  explicit statement that they stand as measured on their own commit.

---

## Concepts for `docs/concepts.md` (as they arise)

- **Weight-only PTQ** — weights stored at reduced precision, dequantized to
  the compute dtype at use; activations and KV cache unchanged → the
  `quantization` path in `engine/model_runner.py`.
- **NF4** — 4-bit NormalFloat: 16 levels at the quantiles of a normal
  distribution, per-block scales, calibration-free. Source: the QLoRA paper's
  quantization section.
- **Double quantization** — quantizing the per-block scale factors themselves;
  a small further saving, on by default.
- **Compute dtype vs. storage dtype** — what a quantized model *reports* about
  itself versus what its matmuls run in; the KV pool must follow the compute
  dtype → `engine/kv_cache.py`.
- **VRAM budget decomposition** — weights + KV cache + activations; the
  identity that makes quantization an *indirect* KV capacity win → Phase 2's
  sizing helper.
- **Dequant overhead** — arithmetic added to the critical path in exchange for
  removed memory traffic; the roofline trade, run in the opposite direction
  from M3's gather.

---

## How to run this milestone  <!-- POST-BUILD -->

Filled in as the milestone lands.

---

## Phase 6 — results, read honestly  <!-- POST-BUILD -->

Filled in once the benchmark runs. Per the template: what must not be credited
to quantization, what got worse, where the effect collapses, limits on the
headline, defects flagged rather than quietly fixed.

---

## Definition of done (whole milestone)

- ⬜ All phases above ticked.
- ⬜ `tests/test_correctness.py` passes — the existing CPU gate unchanged and
  green, plus the quantized GPU cases from Phase 4.
- ⬜ A chart showing memory footprint before/after quantization and the
  resulting increase in maximum concurrent sequences at fixed VRAM, recorded
  in `docs/benchmarks/` with commit + hardware.
- ⬜ M3's provenance wart fixed: `sys.argv` recorded into every results JSON.
- ⬜ `docs/progress.md` updated.
- ⬜ ADR-0015 onward written and linked.
- ⬜ `docs/roadmap.md` M4 build boxes and DoD ticked (theory box is the
  owner's), and `claude.md` Open decision #3 marked resolved.
- ⬜ `README.md` records the headline model, that quantization is weight-only,
  and that the KV cache remains bf16.

---

## Review summary (for the owner)  <!-- PRE-BUILD; superseded by the Retrospective -->

**Phases:**

0. Spike — prove NF4 loads on this box, and find out what a quantized model
   reports as its dtype before that number silently sizes the KV pool. Also
   clears M3's outstanding "verified on CPU + eager only" carry-over.
1. `ModelRunner` takes a `quantization` argument; all four engines inherit it
   for free through the single seam. No engine branches on it.
2. The VRAM budget split — measure weights, hand the remainder to the block
   pool. **The one piece of real reasoning in this milestone**, and where the
   headline number comes from.
3. The headline model: chosen, loaded 4-bit, accounted for. Blocked on Q1.
4. The correctness gate under quantization — the structurally hard part, since
   NF4 needs CUDA and the gate is pinned to CPU for good reasons.
5. M3's flagged fixed-overhead defect: profile and decide. Before Phase 6, so
   it doesn't get silently attributed to quantization.
6. The chart, with the prediction written down before measuring.

**The decisions I most want your call on before I build:**

- **Q1 — the headline model.** A 7B model does not fit in this GPU's 3.999
  GiB at any precision. I need to know whether the RTX 5060 machine has
  arrived or has a date; that fact alone decides between "3B on this box now"
  and "7B on the new box later." Everything except Phase 3 and Phase 6's
  headline arm proceeds regardless.
- **Q2/Q3 — the quantized correctness gate.** NF4 requires CUDA; the gate is
  pinned to CPU + eager by ADR-0009/0010. My proposal is a same-quantization
  golden on GPU, testing what the gate has always tested and explicitly not
  claiming NF4 tracks bf16. Q3's exact-vs-prefix match I can settle
  empirically in Phase 0 and report back, if you'd rather not decide blind.
- **Q4 — what the chart compares.** Same model bf16-vs-NF4 as the measurement,
  plus the dev-vs-headline line as the quotable sentence. This is the M3
  mistake I most want to not repeat: M3's plan specified a workload that could
  not exhibit the effect the plan wanted to measure.

Nothing is implemented yet — this is the plan. Say go (or redirect) and I'll
start at Phase 0, which is hardware-independent and answers three of the
questions above with evidence.

---

## Retrospective  <!-- POST-BUILD -->

Filled in at the end of the milestone.
