# ADR-0009: Pin the correctness gate to CPU — GPU kernel-path floating-point non-associativity can flip a near-tied argmax across a recomputation boundary

- **Status**: Superseded by ADR-0010
- **Date**: 2026-08-23
- **Milestone**: M2 (discovered), cross-cutting (the correctness-gate policy
  every future milestone runs under)

> **Superseded 2026-08-23.** Pinning to CPU alone was tested against only one
> resync scenario before being accepted, and turned out to be incomplete: a
> *different* test (single-sequence, no resync at all) started failing on CPU
> the moment the suite actually ran there, for a completely different reason
> (`StaticCache` vs. `DynamicCache` numerics under SDPA, not a GPU issue).
> ADR-0010 documents the fuller investigation and the actual permanent fix
> (CPU **and** eager attention). Kept here, not deleted, per this project's
> own rule that a wrong-but-reasoned decision is worth re-reading later.
- **Concepts**: floating-point determinism / reduction order on GPU (refines
  the "GPU kernels aren't perfectly associative" gotcha already flagged in
  `docs/bites/testing-and-correctness/correctness-harness.md`); the
  exact-match correctness bar from ADR-0003

## Context

Phase 3's new multi-sequence engine test (`ContinuousEngine`, two sequences
with staggered `max_new_tokens=[8, 20]`) failed at the very last token of the
20-token sequence: token `320` where the golden reference said `955`. The
divergence appeared only for the continuous-batch engine, and only for the
sequence that survived past the other one's eviction — i.e., only for a
sequence whose decode was interrupted by a resync (ADR-0007) partway through.

Isolating the cause, in order:

1. Reproduced it directly with `ModelRunner`: an uninterrupted prefill+decode
   run vs. one that hits a mid-generation resync at token 8 (bulk-prefilling
   `prompt + already-generated tokens` in one forward pass, then continuing
   to decode) — same divergence, same position, same wrong token.
2. Ruled out the resync *concept* itself: replayed the identical "interrupt
   and restart" scenario through HF's own `.generate()` with zero custom
   code — one call for 8 tokens, then a fresh `.generate()` call fed the
   result as `input_ids`. It matched the uninterrupted run bit-for-bit.
   "Resuming from accumulated history via a fresh forward pass" is not
   inherently unsafe.
3. Ruled out `StaticCache` specifically, and ruled out `max_cache_len`
   (buffer size): manually replicating the exact resync computation (same
   mask, same position ids, same cache size) matched golden — but only on
   CPU. The same code, unchanged, produces the wrong token on GPU.
4. Confirmed with an A/B test on `ModelRunner` itself: forcing
   `device="cpu"` end to end (both the uninterrupted and the resync-interrupted
   run) makes the two match bit-for-bit. On `device="cuda"` (the machine's
   default — a GPU is available), they diverge at exactly this one token.

The mechanism: bulk-prefilling an already-partially-generated sequence in one
forward pass, versus reaching the same point by decoding it one token at a
time, are mathematically equivalent but not computed the same way — different
input shapes route to different CUDA kernels / different reduction orders in
the attention matmuls. This is floating-point non-associativity: `(a+b)+c` is
not guaranteed to equal `a+(b+c)` in finite-precision arithmetic, and GPU
matmul/softmax reductions sum in whatever order the selected kernel uses, not
a fixed one. At a genuine near-tie between the top-2 logits, two
mathematically-equivalent-but-differently-ordered computations can rarely
pick different winners. CPU execution does not exhibit this here because it
does not switch kernels/reduction strategy across these two shapes the way
the CUDA backend does.

This is not specific to MicroServe's code — it is exactly why real serving
engines (vLLM included) do not promise bit-identical greedy output across
different batching/scheduling strategies on GPU, only "the same distribution,
up to numerical noise." It is also going to recur: M3's paged attention
(gather scattered blocks into a scratch tensor, then attend) changes the
computation's shape/path from a contiguous KV read in exactly the same way a
resync does, so the same class of rare divergence should be expected there
too if the correctness gate stays on GPU.

## Options considered

### Option A — Pin the correctness gate to CPU unconditionally
Force every `ModelRunner`/engine instance in `tests/test_correctness.py` onto
`device="cpu"`, regardless of whether a GPU is available on the machine
running the tests.
- **Buys**: a genuinely deterministic gate — the same code, same inputs,
  same result, every run, on any machine. Verified directly: CPU reproduced
  bit-identical output across the resync boundary where GPU did not.
- **Costs**: the correctness gate no longer exercises the exact GPU kernel
  path benchmarks will run under. Slightly slower test runs (CPU inference on
  a 0.5B model — seconds, not a real cost).

### Option B — Force deterministic CUDA algorithms (`torch.use_deterministic_algorithms`)
Enable PyTorch's deterministic-algorithms mode (plus e.g.
`CUBLAS_WORKSPACE_CONFIG`) and keep the gate on GPU.
- **Buys**: if it worked, would keep testing the real GPU path while fixing
  reproducibility.
- **Costs**: targets the wrong problem. That flag exists to make *repeated
  runs of the same operation* reproducible (pinning down otherwise
  nondeterministic atomic-op ordering); it does not make *two different
  operations that happen to be mathematically equivalent* (a 9-token bulk
  prefill vs. 8 sequential 1-token decodes) produce identical results — those
  are legitimately different computation graphs, each internally
  deterministic on its own. Confirmed this isn't the failure mode here: the
  divergence was consistent and repeatable within a single process, not
  flip-flopping run to run, which is what `use_deterministic_algorithms`
  actually targets. Also carries a real throughput cost, which matters for
  the milestone this project is building toward measuring.

### Option C — Relax the exact-match bar for tokens generated after a resync
Keep the gate on GPU; for any token generated after a membership-change
resync, allow a tolerance (e.g. compare top-2 logit margins, or accept a
mismatch if the golden run's own margin at that position was already thin)
instead of requiring bit-identical ids.
- **Buys**: tests the real GPU path continuous batching will actually run on.
- **Costs**: quietly moves the "when do we stop requiring exact-match"
  decision earlier than planned. The project's own correctness-harness
  reasoning (`docs/bites/testing-and-correctness/correctness-harness.md`)
  is explicit that exact-match is the bar through Milestone 3, and that
  Milestone 4 (quantization) is where it deliberately changes — introducing
  a second, GPU-specific fuzziness rule now blurs that boundary and makes it
  harder to trust a future "it doesn't match, but that's expected" call is
  actually expected and not a real bug hiding behind this excuse.

## Decision

**Option A — pin the correctness gate to CPU unconditionally.** Not "CPU if
no GPU is available" (the current wording in `claude.md`'s Hardware notes) —
CPU always, for this specific gate, regardless of what hardware the machine
running it has.

## Why the others lose

Option B solves a different problem than the one actually observed — verified
this by checking that the divergence was deterministic and repeatable, not
flaky, which rules out the atomic-op-ordering nondeterminism that flag is
built for. Option C is the more tempting one, because it keeps testing the
real deployment path, but it does so by weakening exactly the guarantee this
project has leaned on since ADR-0003 — and the project already has a
principled place for "exact-match stops being the right bar" (quantization,
M4), which this isn't. Borrowing that exception early, for a different
reason, makes the eventual real exception (M4) harder to reason about
cleanly.

## Consequences

- `tests/test_correctness.py`'s fixtures now construct every `ModelRunner`
  and engine with `device="cpu"` explicitly, so the whole suite is
  deterministic on any machine — GPU present or not — rather than silently
  depending on what hardware happens to be available.
- GPU is now used only for `docs/benchmarks/` runs, never for the
  correctness gate — sharpening a split `claude.md` already states
  ("Benchmark numbers are GPU-only") but hadn't yet applied to correctness.
- A passing CPU gate no longer proves the GPU code path is bit-exact — but
  this is an acceptable trade specifically *because* the divergence is
  proven (via the CPU/GPU A/B test above) to be a hardware floating-point
  property of the computation, not a logic difference. A real logic bug
  (wrong mask, wrong index, wrong position id) reproduces deterministically
  on CPU every time; this class of bug does not.
- `claude.md`'s Hardware notes line ("Correctness tests on the tiny model
  should run on CPU if no usable GPU is available") is now stricter than
  written — worth updating that line to say "always," so a future session
  doesn't read it as GPU-preferred-when-present. Flagged for the owner
  rather than edited silently, since it's project-instruction text.
- This is very likely to recur at M3: paged attention's gather-scattered-
  blocks-then-attend also changes the computation's shape/path relative to a
  contiguous KV read, the same structural cause as this ADR's resync case.
  Worth remembering this ADR's finding then, rather than re-running the same
  investigation from scratch.

## What I learned

*(owner's to fill in, optional)*
