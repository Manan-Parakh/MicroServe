# ADR-0002: Dev model for Milestones 0–3 — Qwen2.5-0.5B-Instruct

- **Status**: Accepted
- **Date**: 2026-08-19
- **Milestone**: M0
- **Concepts**: none new (model selection, not an inference-systems concept)

## Context

`claude.md`'s locked decisions fix the *category* of dev model (tiny, for
fast iteration) but leave the specific choice between Qwen2.5-0.5B and
Llama-3.2-1B open, deferring the eventual 7B-class headline model to M4. M0
needs one picked before `engine/model_runner.py` can load anything.

## Options considered

### Option A — Qwen2.5-0.5B-Instruct
0.5B params, smallest widely-used instruction-tuned model in this size class.
- **Buys**: fastest load/generate cycle of the two — matters most while
  debugging engine internals (scheduler in M2, block allocator in M3), where
  the model is reloaded and re-run constantly and its own compute time is
  pure overhead on top of what's actually being tested.
- **Costs**: smaller model, less representative of real deployment behavior
  (shorter effective context handling in practice, less "interesting"
  generations for eyeballing correctness during Phase 1's manual smoke test).

### Option B — Llama-3.2-1B-Instruct
1B params, roughly 2x Option A.
- **Buys**: closer architecture family to a plausible eventual 7B-class
  Llama-family headline model, so behavior observed now (attention pattern
  shapes, tokenizer quirks) generalizes slightly better to M4.
- **Costs**: slower iteration loop than Option A, for a generalization
  benefit that mostly doesn't matter — M4's headline model choice is made
  fresh, independent of this decision, per `claude.md`.

## Decision

**Qwen2.5-0.5B-Instruct.**

## Why the others lose

Llama-3.2-1B loses because the thing it buys (architecture-family similarity
to a future 7B model) isn't actually load-bearing: `claude.md` explicitly
decouples the M4 headline-model choice from this one, so there's no
compounding benefit to picking the "closer" family now. Its cost (slower
iteration) is real and paid on every single debug cycle through M0–M3 — the
exact opposite of what a dev model is for.

## Consequences

`tests/test_correctness.py` and `bench/benchmark.py` both hardcode
`Qwen/Qwen2.5-0.5B-Instruct` as the model id for now. Swapping later (if the
0.5B model turns out too small to exercise some behavior meaningfully, e.g.
during M3's paging work) means changing one constant in each file — cheap to
reverse.

## What I learned

*(owner's to fill in, optional)*
