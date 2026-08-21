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
