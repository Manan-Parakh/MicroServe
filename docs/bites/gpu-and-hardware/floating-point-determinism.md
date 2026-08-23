# Floating-Point Determinism & Reduction Order on GPU (and CPU)

> The same mathematically-correct computation, run two structurally different
> ways, can produce a different low-order bit — and at a genuine near-tie,
> that bit is enough to flip which token wins an argmax.

---

## The Problem

`(a + b) + c` is not guaranteed to equal `a + (b + c)` in finite-precision
floating-point arithmetic — addition and multiplication stop being
associative once you round after every step. A matmul, a softmax, an
attention computation are all just large trees of adds and multiplies; *how*
that tree is shaped — what gets summed first, what gets batched together,
what kernel a library picks for a given tensor shape — changes the rounding
path, and therefore can change the last few bits of the answer, even though
every implementation is "doing the same math."

This stopped being an abstract footnote the moment MicroServe's M2 correctness
gate hit it directly. `engine/model_runner.py`'s `prefill_batch`/`decode_step`
(built on `transformers.StaticCache`, a fixed-size padded buffer) is
mathematically equivalent to `.generate()`'s own decode loop (built on
`DynamicCache`, which grows to exactly the needed size) — same causal
attention, same weights, same greedy argmax. But "mathematically equivalent"
and "bit-identical" are different claims, and a correctness harness that
requires exact token-for-token match (`docs/adr/0003-correctness-harness-scope.md`)
is implicitly requiring the second, stronger one. Two independent
divergences surfaced from exactly this gap — see
`docs/adr/0009-pin-correctness-gate-to-cpu.md` (superseded) and
`docs/adr/0010-correctness-gate-cpu-plus-eager-attention.md` (the corrected,
accepted account) for the full investigation this bite summarizes.

---

## Mental Model

Two *different* things get lumped together as "GPU nondeterminism," and only
one of them is actually about randomness:

```
1. RUN-TO-RUN NONDETERMINISM
   same op, same shape, same device, same inputs, run twice
   -> can give different results, because some CUDA kernels (atomic-add-
      based scatter/reduce ops) don't fix a summation order at all
   -> torch.use_deterministic_algorithms(True) targets exactly this

2. CROSS-IMPLEMENTATION NUMERICAL NON-EQUIVALENCE
   same math, computed two structurally different ways (different tensor
   shape, different cache class, different fused-kernel choice)
   -> each way is internally perfectly reproducible on its own
   -> but the two ways don't have to agree with each other
   -> no flag "fixes" this -- it's not a bug, it's rounding-path divergence
```

MicroServe hit *only* category 2. Every failing case reproduced identically
on repeat runs (same device, same code) — the divergence was between two
different code paths computing the "same" thing, not between two runs of one
path.

---

## How It Works

### Where the rounding-path divergence actually comes from

**GPU: shape-dependent GEMM algorithm selection.** `torch.matmul` on CUDA
calls into cuBLAS, which picks a matrix-multiplication algorithm/tiling
strategy based on the operands' shape (and, historically, on `TF32` mode for
`float32` matmuls on Ampere+ GPUs — a reduced-mantissa fast path enabled by
default unless explicitly turned off). Different algorithms accumulate
partial sums in different orders. This is documented, expected NVIDIA/cuBLAS
behavior — a deliberate performance trade, not an oversight. It means a
9-token bulk attention computation and nine sequential 1-token attention
computations, even though mathematically identical, are *not* guaranteed to
route through the same summation order, because they're different shapes.

> 🔹 **TF32** — a 19-bit floating-point format Ampere-and-later NVIDIA GPUs
> silently substitute for `float32` matmuls by default, trading mantissa
> precision for throughput. Not the cause investigated here (this project's
> model runs in `bfloat16`), but the same *family* of "faster path, slightly
> different numbers" trade-off.

**Attention-kernel choice: fused (SDPA) vs. explicit (eager).**
`scaled_dot_product_attention` (`sdpa`) is a fused kernel — it doesn't
materialize the full attention-score matrix as a separate step, and its
internal tiling/blocking strategy can depend on the KV tensor's actual shape,
*including* how much of a pre-allocated buffer is real content versus padding
that's masked out afterward. `eager` attention is the naive, unfused version:
explicit `Q @ K.T`, explicit masked softmax, explicit `@ V`. Mathematically
identical to `sdpa` (masked positions are `-inf` before softmax, contributing
exactly `0` after it). Numerically, `eager`'s explicit steps are far less
sensitive to how much "extra" allocated-but-masked capacity surrounds the
real content, because there's no fused kernel making an internal decision
based on total buffer size.

**Why this matters for a paged/pre-allocated KV cache specifically.**
`StaticCache` allocates `max_cache_len` slots up front and writes real
content into a prefix of them; `DynamicCache` allocates exactly what's used,
every step. Two runs that logically hold the *same* keys/values can
therefore sit in tensors of *different total shape* — exactly the kind of
difference GEMM/fused-attention algorithm selection is sensitive to. This
generalizes past `StaticCache`: M3's block allocator gathers scattered
physical blocks into a contiguous scratch tensor before attending — a shape
transformation with the same structural profile as the one investigated
here.

### What the actual investigation found (summarized from ADR-0010)

Two genuinely separate divergence sources were conflated as "one GPU
problem" before being tested apart:

| | Cause | Fixed by eager attention? |
|---|---|---|
| **Phenomenon 1** | `StaticCache` (padded) vs. `DynamicCache` (exact) under `sdpa`, on CPU | Yes — confirmed, both test prompts |
| **Phenomenon 2** | Bulk multi-token prefill vs. sequential single-token decode (same cache class), on GPU | **No** — persisted under `eager` too |

Phenomenon 1 is an attention-kernel-selection question — eager attention's
explicit, unfused math is provably shape-invariant, so it disappears.
Phenomenon 2 is one level lower, in cuBLAS's own GEMM algorithm selection —
`eager` still calls `torch.matmul` under the hood, so it inherits whatever
shape-dependent behavior cuBLAS has, and no attention-implementation choice
reaches down far enough to fix it. The one configuration where *both*
phenomena vanished, across every prompt and scenario tested: **CPU + eager
attention** — not because CPU is inherently "more correct," but because
neither phenomenon's specific triggering condition (cuBLAS's GPU algorithm
selection; SDPA's fused-kernel shape sensitivity) applies there for the
cases tested.

---

## Usage

```python
from engine.model_runner import ModelRunner

# Fast path (production / benchmarks): whatever the hardware defaults to.
runner = ModelRunner("Qwen/Qwen2.5-0.5B-Instruct")

# Correctness-gate path: explicit, deterministic-for-this-purpose config.
gate_runner = ModelRunner(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
    attn_implementation="eager",
)
```

`attn_implementation` is a standard `transformers` `from_pretrained` kwarg
(`"sdpa"`, `"eager"`, `"flash_attention_2"` where available) — MicroServe
just threads it through `ModelRunner.__init__` alongside the existing
`device` parameter, so nothing about the default (fast) path changes unless
a caller opts in.

---

## Where This Fits in MicroServe

- `engine/model_runner.py` — `ModelRunner.__init__` gained
  `attn_implementation: str | None = None`, passed straight to
  `AutoModelForCausalLM.from_pretrained`.
- `tests/test_correctness.py` — the correctness gate's `GATE_KWARGS =
  {"device": "cpu", "attn_implementation": "eager"}`, applied to every
  fixture including the golden reference itself.
- `docs/adr/0009-pin-correctness-gate-to-cpu.md` (superseded) and
  `docs/adr/0010-correctness-gate-cpu-plus-eager-attention.md` (accepted) —
  the decision record this bite is the study companion to.
- Read-ahead for M3: paged attention's gather-scattered-blocks-into-a-
  scratch-tensor step is the same *kind* of shape transformation that caused
  Phenomenon 1 here — worth re-running this investigation's methodology
  (not assuming its exact conclusion) once that code exists.

---

## Gotchas & Trade-offs

- **`torch.use_deterministic_algorithms(True)` does not fix cross-shape
  divergence.** It targets category-1 nondeterminism (same op, same shape,
  different runs) by disabling nondeterministic atomic-based kernels. It has
  no effect on category-2 divergence (same math, different shape/
  implementation), which is what near-tied-argmax flips across a cache
  resync or a batching-shape change actually are.
- **A passing CPU+eager gate does not prove the GPU+SDPA path is bug-free.**
  It proves the *specific* divergence found is attributable to well-
  documented kernel/algorithm-selection behavior, verified by reproducing it
  with zero custom code (HF's own `.generate()` alone, in both failing
  configurations). A *new* GPU-only mismatch still deserves the same
  isolation process below, not an automatic "floating-point, ignore it."
- **This is a genuine engineering trade-off, not a bug to keep chasing.**
  There is no PyTorch-exposed flag that makes cuBLAS shape-invariant without
  giving up the GPU acceleration built on that shape-dependent algorithm
  selection. Pin a stable, defensible test configuration once (the actual
  fix here), rather than re-tuning device/kernel flags indefinitely hoping
  for a green run — that path has no natural stopping point.
- **Only a genuine near-tie flips.** Both phenomena found here only changed
  the *winning* token where the top-1/top-2 logit margin was already thin.
  A wide-margin decision is essentially immune — which is exactly why 19 of
  20 tokens in the failing sequences still matched: the model was confident
  everywhere except the one position where two roughly-equally-good
  continuations existed.
- **`bfloat16` (this project's model dtype) has a short mantissa (7 bits)**,
  which makes near-ties considerably more common than `float32` would for
  the same model and prompt — worth remembering before assuming a
  low-precision benchmark run and a correctness-gate run "should" agree
  token-for-token even absent any cache/shape difference at all.

---

## Comparison Table

| Mechanism | What causes it | Same-run reproducible? | Fixed by `use_deterministic_algorithms`? | Fixed by `eager` attention? |
|---|---|---|---|---|
| Atomic-op nondeterminism (category 1) | Nondeterministic reduction order in specific CUDA kernels (scatter/atomic-add ops) | No — varies run to run | Yes | N/A |
| SDPA shape sensitivity (Phenomenon 1) | Fused-kernel behavior depends on real-vs-padded buffer shape | Yes | No | Yes |
| cuBLAS GEMM algorithm selection (Phenomenon 2) | Matmul algorithm choice depends on operand shape | Yes | No | No |
| TF32 reduced-mantissa matmul | Default fast path for `float32` on Ampere+ | Yes | No (separate flag: `torch.backends.cuda.matmul.allow_tf32`) | N/A (dtype-level, not kernel-level) |

---

## Interview & Career Angle

This is one of the fastest ways to separate "has read about floating point"
from "has actually debugged it in a serving system." Comes up directly when
discussing why a serving engine's output can differ from a reference
implementation's, why A/B-testing two inference backends needs a tolerance
band instead of exact match, and in any conversation about reproducibility
claims for GPU-accelerated ML systems generally.

Adjacent concepts worth learning next:
- **Mixed-precision training/inference numerics** — how `bfloat16`/`float16`
  accumulation strategies (accumulate in fp32, round at the end vs. round at
  every step) change how often near-ties actually occur.
- **cuBLAS/cuDNN algorithm heuristics** — the broader family this project's
  Phenomenon 2 belongs to; relevant for anyone tuning GPU kernels for
  throughput and wondering why a "faster" configuration changed output.
- **Flash Attention's tiling strategy** — a fused kernel whose whole design
  is built around a specific (and shape-dependent) reduction/accumulation
  order, directly related to why `sdpa`'s fused path here behaved
  differently from `eager`'s explicit one.

---

## Hard Questions I Should Be Able to Answer

1. **Practical:** your correctness gate passes on CPU but fails on GPU for
   one specific prompt, 19 of 20 tokens matching. What's the first thing you
   check, and what would make you confident it's floating-point rounding
   rather than a real bug?
2. **Interview / system-design:** "Our inference engine's benchmark
   configuration (batched, GPU, fused kernels) occasionally produces
   different greedy tokens than our single-request reference implementation.
   Is this acceptable? How would you decide, and how would you communicate
   it to a team that wants 'reproducible' output?"
3. **Engine-internals:** you're about to add a KV-cache implementation that
   changes buffer shape depending on batch composition (sound familiar?).
   What test would you write *before* trusting it, given everything in this
   bite?
4. **Debugging / it's on fire:** `torch.use_deterministic_algorithms(True)`
   is already set, and your model still gives different greedy tokens across
   two runs with the exact same code, same shape, same device. What do you
   suspect now, since the category-1 fix didn't work?
5. **Practical:** you need a correctness gate that's both fast (so it runs on
   every commit) and trustworthy. Given that CPU+eager was the slowest
   configuration in this investigation's matrix, how do you justify using it
   anyway?

---

## Answers

1. Check whether the mismatch starts at a position where the *golden* run's
   own top-1/top-2 logit margin was already thin (log it during golden
   generation). If yes, that's strong evidence for rounding noise, not a
   bug — confirm by re-running the same failing prompt through the exact
   same code path on the *other* device or attention implementation; if the
   result changes without any logic change, that seals it. A real bug
   (wrong mask, wrong index) reproduces the same wrong answer everywhere,
   not just on one specific hardware/kernel combination.
2. Acceptable, but only if it's an explicit, documented policy decision, not
   a silently-loosened assertion. Communicate it the way this project's ADRs
   do: name the exact mechanism (which kernel/algorithm choice), show the
   evidence it's not a logic bug (reproduce with the reference
   implementation's own code, zero custom code involved), and pick one
   stable test configuration that's exact-match reliable — don't promise
   "the GPU path is provably bit-exact," promise "the GPU path is provably
   *not* running different logic," which is the claim that's actually true
   and actually defensible.
3. A test that runs the same logical KV content through *two different
   buffer shapes* (e.g., tightly-sized vs. padded with slack) and asserts
   identical output — exactly the "tight vs. padded `max_cache_len`"
   experiment this investigation ran. If that test is flaky on the fast
   attention path but stable under `eager`, you've already found your
   correctness-gate configuration before shipping the feature, not after.
4. Suspect a shape/implementation difference somewhere in the two runs that
   *looks* identical but isn't — different batch composition, different
   cache state left over from a previous call, different code path taken
   for a different total sequence length (e.g., one run happened to trigger
   a resync-equivalent recomputation and the other didn't). Category-1 fixes
   only ever address genuine run-to-run randomness in one fixed computation
   graph; if the flag didn't help, the computation graph itself likely
   isn't as identical between the two runs as it appears.
5. Because "fast" and "trustworthy" are being asked of the same test suite
   for two different purposes that don't have to share a configuration.
   Every millisecond spent on the correctness gate buys confidence the
   *logic* is right; every millisecond spent on a benchmark buys a real
   throughput number. Splitting them (CPU+eager for the gate, GPU+SDPA for
   benchmarks — already this project's practice for device, extended here to
   attention implementation) means neither purpose has to compromise for the
   other's sake, and a 0.5B dev model makes the "slow" gate configuration
   take single-digit seconds regardless.

---

## Rules of Thumb

> A near-tied argmax flip and a real logic bug look identical from the
> outside (wrong token, everything else fine) — the only way to tell them
> apart is to check whether the *reference* implementation's own logit
> margin at that exact position was already thin.

> "Internally deterministic" (same code, same device, same result every run)
> is not the same claim as "matches a different, mathematically-equivalent
> implementation." Don't let a passing repeat-run fool you into trusting a
> cross-implementation comparison.

> No configuration flag makes a fused, hardware-accelerated kernel
> shape-invariant without giving up the acceleration it exists for — pick a
> deliberately slower, explicit configuration for correctness testing rather
> than chasing a flag that doesn't exist.

---

## TL;DR

- Floating-point addition/multiplication isn't associative — two
  mathematically-equivalent computations (different cache shape, different
  batching, different fused-vs-explicit kernel) can legitimately produce
  different rounding, and at a genuine logit near-tie, that can flip which
  token wins greedy argmax.
- This project hit two *distinct* causes at once and initially mistook them
  for one "GPU problem": `StaticCache`-vs-`DynamicCache` shape sensitivity
  under SDPA (fixed by switching to eager attention) and cuBLAS's own
  shape-dependent GEMM algorithm selection on GPU (not fixed by any
  attention-implementation choice — a lower-level, unavoidable trade for GPU
  speed).
- The permanent fix was picking one stable, explicit, defensible correctness-
  gate configuration (CPU + eager attention) rather than continuing to
  toggle device/kernel flags — see `docs/adr/0010-correctness-gate-cpu-plus-eager-attention.md`
  for the full decision record.
