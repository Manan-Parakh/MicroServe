# ADR-0010: Correctness gate runs on CPU *and* eager attention — two distinct floating-point non-associativity sources, not one CPU-vs-GPU issue

- **Status**: Accepted
- **Date**: 2026-08-23
- **Milestone**: M2 (discovered), cross-cutting (the correctness-gate policy
  every future milestone runs under) — **supersedes ADR-0009**
- **Concepts**: floating-point determinism / reduction order on GPU and CPU;
  `scaled_dot_product_attention` kernel selection; the exact-match correctness
  bar from ADR-0003. See also
  `docs/bites/gpu-and-hardware/floating-point-determinism.md` for the
  underlying mechanics, written as a study reference alongside this ADR.

## Context — the assumptions made, in order, and what each one cost

This ADR exists because ADR-0009's conclusion ("pin to CPU") was reached
after testing exactly one scenario, was accepted, and then immediately failed
differently the moment the fixed suite ran for real. Recording the assumption
chain here — including the wrong turns — is the point: each one looked
reasonable when made, and each one's blind spot is exactly what the next
investigation step exposed.

**Assumption 1 (implicit, since M0): a golden-token comparison is
device-agnostic.** `tests/test_correctness.py` never pinned a device;
`ModelRunner` auto-selects CUDA when available. Consequence: every M0–M2
"verified" claim (Phase 0/1/3's own **Verified** notes) was only ever checked
on whatever GPU happened to be present on this machine. Nothing forced a
CPU run until this investigation, so a CPU-specific failure mode had been
sitting unexercised the whole time.

**Assumption 2: the Phase 3 test failure was caused by ADR-0007's resync
specifically.** Grounds: the failure only appeared for the sequence that
survived past the other's eviction, i.e., the one whose decode was
interrupted by a resync. Consequence of trusting this too quickly: the fix
search started and stayed scoped to "resync," when the actual scope needed
to be "any recomputation-path change," a strictly bigger category resync is
only one member of.

**Assumption 3 (ADR-0009's core claim): CPU reproduces `.generate()` bit-
for-bit where GPU doesn't, so CPU is "the deterministic, correct one."**
Verified against exactly one prompt/scenario (`"def fibonacci(n):"`,
resync case) before being accepted. Consequence: false in general. The very
next thing that happened — running the *other* test prompt
(`"The capital of France is"`) through a *different*, resync-free scenario,
on CPU — failed. CPU is internally deterministic (same code, same device,
same result, every run — that part holds and is explained properly in the
bite above), but "internally deterministic" is not the same claim as
"matches `.generate()`," and conflating the two is exactly the mistake
ADR-0009 made.

**Assumption 4 (this ADR, tested before being trusted): there are two
distinct, separately-caused divergence sources, not one.** This is the
assumption that finally held up under a full CPU/GPU × sdpa/eager × prompt ×
scenario matrix — see below.

## The corrected investigation

Two genuinely different phenomena were being conflated as "the GPU problem":

**Phenomenon 1 — cache-implementation numerics.** `StaticCache` (our
engine's only cache — ADR-0005/0007) allocates a fixed-size, padded buffer.
`DynamicCache` (`.generate()`'s own default) grows to exactly the needed
size, no padding. Isolated with zero engine code involved — 100% HF's own
`model.generate()`, once with its default `DynamicCache`, once with an
explicit `StaticCache` passed in:

| device | attention | `"capital"` prompt | `"fibonacci"` prompt |
|---|---|---|---|
| CPU | sdpa (default) | **mismatch** | match |
| CPU | eager | match | match |
| GPU | sdpa (default) | match | match |
| GPU | eager | match | match |

Forcing `attn_implementation="eager"` makes `StaticCache` and `DynamicCache`
produce bit-identical logits on CPU, where SDPA didn't. GPU never showed this
particular mismatch for either test prompt (it does not follow that it never
will — see Consequences).

**Phenomenon 2 — bulk-reprocess vs. incremental-decode ("the resync
question"), with `StaticCache` held constant on both sides.** Same
uninterrupted-generation-vs-resync-at-token-8 comparison from ADR-0009,
re-run across the full matrix:

| device | attention | `"capital"` prompt | `"fibonacci"` prompt |
|---|---|---|---|
| CPU | sdpa | mismatch | match |
| CPU | eager | **match** | **match** |
| GPU | sdpa | mismatch | mismatch |
| GPU | eager | mismatch | mismatch |

Eager attention fixes this on CPU for both prompts. It does **not** fix it on
GPU, for either prompt, under either attention implementation. This is the
important negative result: whatever makes a bulk 9-token prefill numerically
diverge from nine sequential 1-token decodes on GPU is not an attention-
kernel-selection question at all — eager attention is a plain, unfused
matmul+softmax+matmul, and GPU still doesn't agree with itself across these
two shapes. The mechanism is one level lower: cuBLAS (the GEMM library
`torch.matmul` calls into on CUDA, used by *both* attention implementations)
selects a matrix-multiplication algorithm based on the operands' shape, and
different algorithms accumulate in different orders — this is documented
NVIDIA behavior, not a MicroServe- or PyTorch-specific quirk, and there is no
supported flag that makes cuBLAS shape-invariant the way `eager` made
attention-kernel-choice irrelevant. (`torch.use_deterministic_algorithms`
does not address this either — see ADR-0009's Option B, which already ruled
it out for the right underlying reason, ahead of this fuller picture.)

**Why CPU and GPU actually differ here, precisely stated:** it is not that
CPU is "more correct." It is that CPU + eager happens to compute attention
via a path that is *shape-invariant* for every case tested — the same
sequence of scalar operations regardless of how many query positions are
processed in one call, or how much padding surrounds the real content in a
cache buffer. GPU's accelerated GEMM path is shape-*variant* by design — that
variance is *how* it gets fast, since different algorithms suit different
matrix shapes — and nothing in PyTorch's public API removes that variance
without also removing the GPU acceleration built on top of it.

## Options considered

### Option A — CPU + eager attention for the correctness gate, unconditionally
Both settings pinned together, permanently, for `tests/test_correctness.py`
only. Benchmarks (`docs/benchmarks/`) stay on GPU + SDPA — the real,
production-shaped path — unchanged.
- **Buys**: the only configuration in the tested matrix where both known
  divergence sources vanish, for every prompt and scenario tested, including
  the one (resync) that's actually central to this milestone. A defensible,
  mechanistic reason, not a lucky pass: eager attention is explicit,
  unfused, shape-invariant math; CPU's GEMM path didn't exhibit cuBLAS's
  shape-dependent algorithm selection for any case tested here.
- **Costs**: the correctness gate no longer runs the exact code path
  benchmarks use (already true of the CPU-only decision in ADR-0009; now
  also true of the attention implementation). Slightly slower test runs —
  eager attention is not fused, and this is CPU already; still seconds for a
  0.5B model.

### Option B — Chase a GPU-side fix (deterministic algorithms, disabling TF32, pinning a specific cuBLAS algorithm)
Keep the gate on GPU; try to force shape-invariant GEMM behavior via PyTorch/
CUDA configuration flags.
- **Buys**: if it worked, the gate would run the real deployment path.
- **Costs**: no PyTorch-exposed flag targets *cross-shape* GEMM-algorithm
  consistency (`use_deterministic_algorithms` targets *run-to-run* repeat-
  ability of the same op, already ruled out in ADR-0009 for the right
  reason). This would mean pinning a specific cuBLAS algorithm by hand, which
  is exactly the kind of "spend an evening chasing a flag" the correctness-
  harness bite already warns against for thin-margin flips, and is
  unsupported/fragile across CUDA versions. Also directly against this
  session's explicit instruction not to keep re-tuning the harness
  indefinitely — this option has no natural stopping point.

### Option C — Accept GPU divergence as expected and stop gating on it
Keep the gate on GPU, and treat a resync-adjacent mismatch as inherently
untestable — skip or `xfail` those specific cases.
- **Buys**: no test-infrastructure change at all.
- **Costs**: quietly removes exact-match coverage from precisely the
  scenario (resync) that's this milestone's actual point, for the actual
  reason `claude.md`'s prime directive treats correctness as non-negotiable.
  A real logic bug hiding behind an `xfail` in exactly this code path is the
  single most dangerous place for one to hide undetected.

## Decision

**Option A.** `tests/test_correctness.py` pins every `ModelRunner`/engine
fixture, and the golden-reference model, to `device="cpu"` and
`attn_implementation="eager"` — a single `GATE_KWARGS` dict, applied once,
not re-litigated per test. `ModelRunner.__init__` and all three engine
constructors (`Engine`, `ContinuousEngine`, `StaticEngine`) gained an
optional `attn_implementation: str | None = None` parameter (same pattern as
the existing `device` parameter) so this is purely additive — unset, HF's own
default (`sdpa`) still applies everywhere else, including benchmarks.

## Why the others lose

Option B chases a fix that doesn't exist in PyTorch's public surface for the
actual mechanism (cross-shape GEMM algorithm selection, not run-to-run
nondeterminism), and has no natural stopping point — exactly the "keep
changing the harness" pattern this investigation was explicitly asked to
stop doing. Option C is the more dangerous of the two: it resolves the
symptom by removing the one test most likely to catch a real bug in the
milestone's actual core mechanism, for a reason (rare floating-point ties)
that's easy to misuse as an excuse for something that isn't actually
floating-point noise the next time a test fails here.

## Consequences

- `tests/test_correctness.py`: a module-level `GATE_KWARGS = {"device": "cpu",
  "attn_implementation": "eager"}`, spread into every fixture's constructor
  call; `golden_model` also loads with `attn_implementation="eager"` (still
  CPU by default, unchanged). All 21 tests pass under this configuration.
- `ModelRunner`, `Engine`, `ContinuousEngine`, `StaticEngine` all gained
  `attn_implementation: str | None = None`, threaded straight through to
  `AutoModelForCausalLM.from_pretrained(...)`. No default behavior changed
  for any caller that doesn't pass it.
- Benchmarks remain untouched: GPU, default `sdpa`, no `attn_implementation`
  passed. `claude.md`'s "benchmark numbers are GPU-only" split now has a
  precise counterpart on the correctness side: "correctness numbers are
  CPU+eager-only."
- **This does not prove GPU+SDPA is bug-free** — it proves the *specific*
  divergences found are attributable to well-understood, well-documented
  numerical properties of GEMM/attention kernels, not to MicroServe's own
  scheduling or masking logic (confirmed by reproducing both phenomena with
  zero custom code, using HF's own `generate()` alone). A future GPU-only
  mismatch should still be investigated with the same rigor as this one,
  not assumed away by citing this ADR.
- M3's paged attention (gather scattered blocks → contiguous scratch tensor →
  `scaled_dot_product_attention`) changes the attention computation's input
  shape on every gather, structurally identical to what triggered Phenomenon
  1 here (buffer-shape sensitivity). Re-run this exact matrix methodology
  (not necessarily this exact conclusion) once M3's gather-then-attend path
  exists, rather than assuming CPU+eager is automatically still sufficient.
- `claude.md`'s Hardware notes line should be updated from "CPU if no usable
  GPU is available" to reflect this ADR's stricter, permanent policy —
  flagged for the owner, not edited silently (project-instruction text).

## What I learned

*(owner's to fill in, optional)*
