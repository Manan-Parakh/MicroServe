# Correctness Harness (Golden-Output Regression Testing for Inference Engines)

> A test that proves your "faster" inference engine still says the exact same
> thing as a known-correct reference, token for token.

*Scope note: MicroServe's plan for this is exactly one thing —
`tests/test_correctness.py`, fixed seed + greedy decode, output tokens
compared against plain HuggingFace `generate()`. This bite goes one layer
deeper than that single test: it covers why exact-match is the right bar
*right now*, and where that bar has to change later in the project.*

---

## The Problem

Every optimization MicroServe makes after Milestone 1 — batching sequences
together, paging the KV cache into non-contiguous blocks, quantizing the
weights — is a rewrite of *how* tokens get computed, done in the hope that
*what* gets computed doesn't change. That hope is not automatically true.
Batching introduces padding and attention-mask logic that's easy to get
subtly wrong; paging introduces a block table whose indexing can silently
read the wrong block; quantization changes the actual numbers weights hold.
A scheduler bug or a block-table off-by-one doesn't usually crash — it just
makes the model quietly attend to the wrong tokens and emit *plausible-looking
wrong text*. Without an automated oracle, that kind of bug ships.

---

## Mental Model

Treat one known-good code path as an **oracle**: HF's own `generate()`, run
single-sequence, no batching, no custom cache. Every other engine variant
(naive wrapper → batched → paged → quantized) has to reproduce the oracle's
output token-for-token under the same seed and prompt. The test isn't "does
this look right" — it's a diff against ground truth.

```
prompt + seed → [oracle: HF generate()]   → golden token ids
prompt + seed → [engine under test]       → candidate token ids
                        assert golden == candidate
```

---

## How It Works

**Exact-match golden comparison (what MicroServe uses through Milestone 3).**
Fix the random seed, decode greedily (always take `argmax` of the logits —
removes sampling as a variable entirely), run the oracle once per prompt, and
store its output ids. Every engine variant re-runs the same prompt and asserts
identical ids. Cheap, deterministic, and — critically — it catches the exact
bug class this project is most exposed to: batching and block-indexing
mistakes, because those change *which* logits get argmax'd, not just their
precision.

**Perplexity/likelihood-threshold comparison (needed from Milestone 4 on).**
Once weights are quantized (NF4), the model's actual numbers change, and nudging
a near-tied logit past its neighbor is expected, not a bug. Exact-match stops
being the right bar. Instead, compare the candidate's log-likelihood of the
*golden* sequence (or perplexity on a held-out set) against a tolerance band —
"close enough" is now a number you choose and record, not zero.

**Differential testing against a second independent implementation.** Run the
same prompt through vLLM or another engine and diff. Gives you an oracle that
isn't your own code, which exact-match-against-your-own-baseline can't. Not
used in MicroServe — the guardrail is "read vLLM's source, don't run it, don't
install it as a dependency" (M3 guardrail in `claude.md`).

**Property-based / invariant testing.** Instead of checking final tokens,
assert internal invariants directly — no block is referenced by two live
sequences, ref counts never go negative, every logical position in a
`block_table` resolves to an allocated block. Complementary to golden-output
testing, not a replacement: it catches allocator bugs *before* they manage to
produce a wrong-but-plausible token.

---

## Usage

```python
# tests/test_correctness.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"
SEED = 0


def golden_tokens():
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    out = model.generate(ids, do_sample=False, max_new_tokens=20)  # greedy
    return out[0].tolist()


def test_naive_engine_matches_golden():
    from engine.engine import Engine

    engine = Engine(MODEL_ID, seed=SEED)
    engine.add_request(PROMPT)
    candidate = engine.step_until_done()
    assert candidate == golden_tokens()
```

`do_sample=False` is the whole trick — it makes `generate()` deterministic
(always argmax) so the golden reference doesn't itself vary run to run.

---

## Where This Fits in MicroServe

- `tests/test_correctness.py` — the file itself; per `claude.md`'s "Two rules
  that hold in every milestone," this must pass **before any benchmark number
  from that milestone is quoted or committed.**
- Milestone 0's Definition of Done *is* this test passing against the golden
  reference.
- Re-run (not re-written) after every milestone that touches how tokens get
  computed: M1 (naive wrapper), M2 (batching), M3 (paging) all owe it an
  exact-match pass; M4 (quantization) is where the bar has to change — see
  Gotchas below.
- `docs/plans/m00-setup-plan.md` Q1 leaves open whether this harness ever
  grows a fixed-seed *sampling* path in addition to greedy — still an open
  decision, not resolved by this bite.

---

## Gotchas & Trade-offs

- **Quantization breaks exact-match, and that's expected, not a regression.**
  NF4 genuinely changes the numbers the model computes. Milestone 4 needs a
  different pass/fail bar (perplexity/likelihood threshold), decided and
  written down *before* the first quantized benchmark is quoted — otherwise
  "correctness before performance" has no teeth for that milestone.
- **A single-sequence test doesn't exercise batching or paging bugs at all.**
  If every golden-comparison prompt runs alone, continuous batching and the
  block allocator's multi-sequence interleaving never actually get tested.
  The gate needs at least one test that runs *multiple concurrent sequences*
  through the batched/paged path and checks each one independently matches
  its own single-sequence golden output.
- **GPU kernels aren't perfectly associative.** Batched matmuls can sum
  values in a different order than a single-sequence forward pass, producing
  tiny floating-point differences. Usually harmless for greedy decoding
  because the top logit has a comfortable margin — but a prompt whose top-2
  logits are nearly tied can flip which token wins purely from reduction-order
  noise, with no actual bug involved. If that happens, it's a flaky-prompt
  problem, not a block-table problem — don't spend an evening debugging the
  allocator over it.
- **CPU vs. GPU correctness runs can legitimately diverge** for the same
  reason. `claude.md` allows CPU for correctness tests specifically so the
  harness isn't blocked on hardware — but the golden reference and the
  candidate should be generated on the *same* device when you're chasing a
  suspected bug, or you're debugging two different noise sources at once.

---

## Comparison Table

| Approach | What it checks | When MicroServe uses it |
|---|---|---|
| Exact-match golden tokens | Same argmax token sequence, bit-for-bit | M0–M3 (fp16/bf16 paths) |
| Perplexity/likelihood threshold | "Close enough" under a chosen tolerance | M4 onward (quantized) |
| Differential testing (2nd engine) | Independent oracle, not your own baseline | Not used — guardrail forbids installing vLLM as a dependency |
| Property-based / invariant testing | Internal structures (block table, ref counts) never enter an illegal state | Complementary, from M3 onward |

---

## Interview & Career Angle

This is one of the first things a senior ML-infra interviewer probes for once
you claim to have built or optimized an inference engine: *"how do you know
your fast path is still correct?"* It separates "I made it faster" from "I
made it faster and can prove it didn't break." Comes up directly in
system-design interviews for serving infrastructure, and in any conversation
about shipping a quantized or batched model to production.

Adjacent concepts worth learning next:
- **Differential testing** — comparing two independent implementations
  instead of one implementation against itself; catches bugs a self-referential
  golden test structurally can't.
- **Floating-point determinism / reduction order on GPU** — why the same
  matmul can produce different low-order bits depending on batch size or
  kernel choice (TF32, cuBLAS algorithm selection); the root cause behind the
  "flaky near-tied logit" gotcha above.
- **Property-based testing (e.g. Hypothesis)** — generating many random inputs
  to check an invariant holds, rather than hand-picking test cases; a natural
  fit for allocator/block-table logic.

---

## Hard Questions I Should Be Able to Answer

1. **Practical:** after wiring up continuous batching, `test_correctness.py`
   starts failing on one specific prompt but passes on the other nineteen —
   where do you look first, and how do you tell a real batching bug apart
   from floating-point reduction noise?
2. **Interview / system-design:** "We're optimizing an LLM inference engine
   and want confidence that each change doesn't silently corrupt output. Design
   the test suite — what runs pre-merge vs. what only runs as part of a full
   benchmark, and why the split?"
3. **Engine-internals:** your block allocator has an off-by-one in the
   block-table index for a sequence's *last* block only. Would single-sequence
   greedy-decode correctness testing catch this? If not, what test would?
4. **Debugging / it's on fire:** a benchmark run shows a nice throughput win
   after your latest change, but you didn't re-run the correctness gate first
   — what's the instinct here, and why is skipping the gate specifically
   dangerous for an inference engine (as opposed to, say, a stateless web
   service)?
5. **Practical:** after adding 4-bit quantization, exact-token match fails on
   3 of 20 prompts. Bug, or expected? How do you decide, and what do you
   replace the exact-match assertion with once you're confident it's expected?

---

## Answers

1. First check whether the failing prompt's top-1 and top-2 logits were
   already close under the *un-batched* path (log the margin in the golden
   generation step). If the margin was already thin, suspect floating-point
   reduction-order noise from batching, not a logic bug — rerun that one
   prompt at batch size 1 through the new code path; if it now matches,
   that's confirmation. If it still doesn't match even alone, it's a real
   bug in the new code, most likely in attention-mask/position-id handling
   for padded sequences.
2. Pre-merge: fast, deterministic, single-and-multi-sequence exact-match
   golden tests (seconds to run) — this is the correctness *gate*, blocking.
   Full benchmark run: throughput/latency/TTFT/TPOT under realistic load —
   answers "is it fast," assumes correctness already established, never used
   to paper over a correctness gate that wasn't run. The split exists because
   correctness and performance are orthogonal claims and conflating them lets
   a fast-but-wrong change slip through on a good benchmark chart.
3. No — if only the *last* block is misindexed, a short single test prompt
   whose full sequence fits in one block would never touch the bug, and even
   longer prompts might silently reference a block that happens to hold
   compatible-looking garbage without crashing. Two things would actually
   catch it: (a) golden-token tests using prompts deliberately long enough to
   span multiple blocks and land unevenly in the last one, and (b) a
   property-based/invariant test that walks every sequence's `block_table`
   and asserts every logical position maps to a distinct, allocated physical
   block.
4. The instinct should be "the number doesn't count yet" — per `claude.md`'s
   own rule, no benchmark number is quotable until the correctness gate has
   passed on that exact code path. It's specifically dangerous for an
   inference engine because a broken scheduler or allocator very often
   produces *fluent, plausible, wrong* text rather than a crash — the kind of
   bug a human skimming output won't notice, but that a golden-token diff
   catches immediately.
5. Expected, most likely — NF4 quantization genuinely perturbs the computed
   logits, and near-tied argmax decisions are exactly where that perturbation
   flips the winner. Confirm by checking whether the mismatches cluster on
   prompts with thin top-1/top-2 margins in the fp16 golden run (same
   diagnostic as Q1). Once confident it's expected, replace bare
   `assert golden == candidate` with a likelihood/perplexity-threshold check —
   candidate's log-probability of the golden sequence must stay above a
   tolerance you pick and record, so "still correct" for the quantized path
   has an explicit, written bar instead of a vibe.

---

## Rules of Thumb

> Never quote a benchmark number from a code path that hasn't just passed the
> correctness gate — "faster" is not a claim you get to make first.

> Greedy decode + fixed seed kills sampling noise, not GPU kernel
> non-determinism — trust exact-match only where logit margins are
> comfortably wide, and treat thin-margin flips as a signal to check the
> margin, not the allocator, first.

> When you introduce approximation on purpose (quantization), swap the
> pass/fail bar explicitly — don't quietly loosen an assertion without writing
> down what "still correct" now means.

> A correctness suite that only ever runs one sequence at a time will never
> catch a batching or paging bug — it's testing the oracle, not the thing you
> actually changed.

---

## TL;DR

- A correctness harness makes one known-good path (HF `generate()`, greedy,
  fixed seed) the oracle every faster engine variant must reproduce
  token-for-token.
- Exact-match is the right bar through Milestone 3; quantization at Milestone
  4 requires switching to a likelihood/perplexity-threshold bar, decided in
  advance, not discovered by a flaky test.
- The bug class this project is most exposed to — batching and block-table
  indexing errors — only gets caught if the test suite actually exercises
  multi-sequence, multi-block code paths, not just a lone prompt run alone.
