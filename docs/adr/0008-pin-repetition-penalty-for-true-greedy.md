# ADR-0008: Pin `repetition_penalty=1.0` in the correctness harness — "greedy" means raw argmax, not `.generate()`'s defaults

- **Status**: Accepted
- **Date**: 2026-08-22
- **Milestone**: M2 (discovered), cross-cutting (fixes the harness every
  milestone depends on)
- **Concepts**: greedy decoding, logits processors (refines ADR-0003)

## Context

Building M2 Phase 1's `ModelRunner.prefill_batch`/`decode_step` (a manual
argmax loop over raw logits, replacing `model.generate()`) turned up a
correctness mismatch against the existing golden reference in
`tests/test_correctness.py`. Debugging it (comparing bfloat16 vs. float32,
`StaticCache` vs. `DynamicCache`, single-sequence vs. batched) ruled out
precision noise and cache-implementation bugs — every path gave the exact
same, deterministic, *wrong* answer starting a few tokens in.

The actual cause: `Qwen/Qwen2.5-0.5B-Instruct`'s checkpoint ships a default
`generation_config` with `repetition_penalty: 1.1`. `model.generate(...,
do_sample=False)` still applies it — `do_sample=False` only disables
*sampling*, not the logits processors (like repetition penalty) that run
before the final argmax/sampling step. `_golden_tokens()` in
`tests/test_correctness.py` calls `.generate()` without overriding this, so
the "golden" reference has always implicitly included repetition penalty.

This was invisible through M0–M1 because `model_runner.generate()` also just
calls `.generate()` — same implicit default on both sides of the comparison,
so it always matched trivially. It only becomes visible once an engine stops
calling `.generate()` and does its own raw-logits argmax, which is exactly
what every engine from M2 onward does. Left unfixed, the correctness gate
would have been quietly meaningless for every remaining milestone: not
because the new engine code was wrong, but because the reference it was
being checked against was computing a different, undocumented function.

## Options considered

### Option A — Pin `repetition_penalty=1.0` everywhere greedy decoding is claimed
Explicitly pass `repetition_penalty=1.0` in `_golden_tokens()` and in
`ModelRunner.generate()`, so "greedy" means exactly what ADR-0003 said it
should mean: raw argmax over the model's own logits, a pure function of
(weights, prompt), nothing else layered on top.
- **Buys**: the golden reference becomes the actual ground truth for what
  every hand-rolled engine (M2 onward) is supposed to compute — no implicit,
  model-specific defaults smuggled in through `.generate()`'s convenience
  behavior. Stays true to ADR-0003's own reasoning without reopening its
  greedy-vs-sampling decision.
- **Costs**: none functionally — `repetition_penalty=1.0` is a no-op
  multiplier, so M0/M1's existing passing tests are expected to keep passing
  unchanged.

### Option B — Reimplement repetition penalty in the engine's manual decode loop
Make `ModelRunner.decode_step`/`prefill_batch` apply the same logits
processor `.generate()` uses by default, so the engine matches the
checkpoint's actual recommended decoding behavior.
- **Buys**: output would resemble what a real deployment of this model
  would produce (repetition penalty exists because raw greedy decoding on
  real prompts tends to loop).
- **Costs**: scope creep with no milestone asking for it — implementing
  logits processors isn't in any milestone's build spec, and it turns every
  future engine variant (static-batch, continuous-batch, paged, quantized)
  into needing to replicate it too, for a decoding-quality concern this
  project isn't studying.

## Decision

**Option A — pin `repetition_penalty=1.0` in both the golden reference and
`ModelRunner.generate()`.**

## Why the others lose

Option B answers a question this project isn't asking. `claude.md`'s prime
directive is understanding *how inference engines schedule and manage
memory*, not decoding-quality tuning — repetition penalty is a text-quality
knob, orthogonal to everything M1–M6 actually build. Pinning it off is also
the more honest reading of ADR-0003's own stated reasoning ("a pure function
of weights and prompt... no RNG stream to reproduce") — a static, prompt-
independent penalty table doesn't add randomness, but it does add a second,
undocumented input (the checkpoint's `generation_config.json`) that ADR-0003
didn't intend to be part of the comparison.

## Consequences

- `tests/test_correctness.py`'s `_golden_tokens()` now passes
  `repetition_penalty=1.0` explicitly alongside `do_sample=False`.
- `ModelRunner.generate()` gets the same explicit `repetition_penalty=1.0`,
  so M1's naive engine (which calls it directly) stays consistent with the
  redefined golden reference.
- Every M2+ engine's hand-rolled argmax loop (no logits processors at all)
  is now checked against a golden reference that actually matches what it's
  supposed to compute — the gap this ADR closes is exactly the kind of thing
  that would otherwise have silently made every future milestone's
  correctness test either meaningless (if it happened to still pass by
  coincidence) or a source of confusing, unrelated failures (if it didn't).
- Any other checkpoint MicroServe ever loads should be checked for a
  non-default `generation_config` before trusting its golden reference the
  same way — this bit because Qwen2.5-Instruct ships repetition penalty on
  by default, not because of anything specific to this codebase.

## What I learned

*(owner's to fill in, optional)*
