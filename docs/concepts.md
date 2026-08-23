# Concepts Glossary

A terse, growing glossary of systems/inference concepts as they enter the
codebase. Per `claude.md`: add an entry when a new concept first gets *built*
(not just read about) — pointing to where it's used in the code and to the
source that explains it properly. Keep entries short; this is an index, not a
tutorial.

Format:

```
### <term>
<one or two sentences — the working definition, not the textbook one>
**Used in:** `<file path>`
**Source:** <paper / book chapter / essay this came from>
```

---

### Greedy decoding
Always take the argmax token at each decode step instead of sampling from the
logits distribution. Makes generation a pure function of (weights, prompt) —
no RNG stream to reproduce — which is what makes an exact token-for-token
correctness comparison possible at all.
**Used in:** `tests/test_correctness.py`, `engine/model_runner.py`
**Source:** see `docs/bites/testing-and-correctness/correctness-harness.md`
for the deeper write-up.

### Naive/static serving
One request processed to full completion before the next one starts — no
batching, no interleaving. Deliberately bad: it's the "before" number every
scheduler-based improvement (M2 onward) is measured against.
**Used in:** `engine/engine.py`
**Source:** `docs/roadmap.md` Milestone 1.

### Throughput vs. latency
Throughput is system-wide tokens/sec; latency is how long one request takes
end to end. A naive engine can have "fine" latency for the one request
running and still have terrible throughput the moment a second request has
to wait its turn — they're different axes, not two names for the same thing.
**Used in:** `bench/benchmark.py`
**Source:** `claude.md` — "The four metrics."

### Sequence lifecycle (OS process-state analogy)
A request moves through `WAITING` → `RUNNING` → `FINISHED` as a scheduler
admits and evicts it — the same shape as an OS's ready/running/terminated
process states, minus a "blocked on I/O" equivalent. `RUNNING` means a
scheduler is actively stepping this sequence's generation.
**Used in:** `engine/sequence.py`
**Source:** OS process scheduling (owner's own coursework); applied here per
`docs/plans/m02-continuous-batching-plan.md` Phase 0.

### Prefill vs. decode
Prefill processes a sequence's whole prompt in one parallel forward
pass — compute-bound, since every prompt token's attention can be computed
simultaneously. Decode generates one token at a time — memory-bandwidth
bound, since each step still has to read the full model weights from GPU
memory for just one new token's worth of compute. This is *why* they're
separate forward calls, and why decode is the one that benefits from
batching multiple sequences together (it amortizes that same memory read
across more useful work).
**Used in:** `engine/model_runner.py` (`prefill_batch` / `decode_step`)
**Source:** Roadmap Two, Phase 2; `docs/plans/m02-continuous-batching-plan.md`.

### True greedy decoding vs. a checkpoint's decoding defaults
`model.generate(..., do_sample=False)` is greedy *sampling* — it still
applies whatever logits processors the checkpoint's `generation_config.json`
sets by default (e.g. repetition penalty), which are deterministic but not
part of "argmax the raw logits." A hand-rolled decode loop that skips those
processors computes a genuinely different function, even though both are
reproducible. Worth checking a checkpoint's `generation_config` before
trusting it as ground truth for a from-scratch decode loop.
**Used in:** `tests/test_correctness.py`
**Source:** discovered debugging `docs/adr/0008-pin-repetition-penalty-for-true-greedy.md`.

### Iteration-level scheduling (continuous batching)
A scheduler re-decides who's in the running batch on *every* decode step, not
once per batch: a finished sequence is evicted and its slot freed the instant
it stops, and a waiting sequence can be admitted the very next step — instead
of a fixed batch running at the pace of its slowest member. The mechanical
consequence for a KV-cache implementation that only supports one shared
write-position per batch (like `StaticCache`): membership changing means the
cache has to be rebuilt via a resync prefill (ADR-0007), not patched in place.
**Used in:** `engine/scheduler.py`
**Source:** Orca paper, via the Anyscale continuous-batching writeup;
`docs/plans/m02-continuous-batching-plan.md` Phase 2.

### Floating-point non-associativity across recomputation paths
Two mathematically-equivalent computations (a bulk-reprocessed KV history vs.
an incrementally-cached one; a padded `StaticCache` buffer vs. an
exactly-sized `DynamicCache`) can round differently and, at a genuine
near-tied logit, flip which token wins greedy argmax — deterministic on
either path alone, just not equal to each other. Not a MicroServe bug: caused
by GPU's cuBLAS shape-dependent GEMM algorithm selection and SDPA's
fused-kernel shape sensitivity, neither of which any PyTorch flag disables
without losing the acceleration built on top of it.
**Used in:** `tests/test_correctness.py` (`GATE_KWARGS`), `engine/model_runner.py`
(`attn_implementation` parameter)
**Source:** `docs/adr/0010-correctness-gate-cpu-plus-eager-attention.md`;
deep-dive in `docs/bites/gpu-and-hardware/floating-point-determinism.md`.

### Static batching's collapse under variable output length
A fixed-size batch can't admit new work until every one of its members is
done — a sequence that finishes early doesn't free its slot, it just sits
there still getting a real forward pass every step, its output discarded.
Measured here as an exact `wasted_compute_fraction` (steps spent on an
already-finished row ÷ total steps) rather than a wall-clock idle-time
estimate — deterministic and reproducible where a timer wouldn't be. In the
M2 benchmark: continuous batching's fraction was 0% (it evicts and admits
immediately); static's was 46.3%, on the same workload, same total tokens.
**Used in:** `bench/benchmark.py` (`run_scheduled`)
**Source:** Orca paper's stated failure mode for static batching;
`docs/benchmarks/2026-08-23_09-18-14-m2-continuous-vs-static-vs-naive/summary.md`.
