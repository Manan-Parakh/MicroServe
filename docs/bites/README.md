# docs/bites/

Deep-dive companion to `docs/concepts.md`. Where `concepts.md` is a terse,
code-anchored glossary entry, a **bite** is a full placement-ready write-up of
one systems/GPU/inference concept — problem, mechanics, usage, MicroServe
placement, gotchas, interview questions. Written via the `/add-bites` command
(`.claude/commands/add-bites.md`).

**Progress:** `0 / 2` (0%)
`[----------------------------------------]`

Counts and bars above are derived — never hand-edit them. Run
`python docs/bites/update_progress.py` after adding or checking off entries.

## Categories

| Folder | Scope |
|--------|-------|
| `gpu-and-hardware/` | GPU compute & memory internals — SM, warp, CUDA core, register file, HBM/SRAM, roofline model, Tensor Cores |
| `os-and-memory/` | OS primitives MicroServe's design borrows from — virtual memory, paging, page tables, mmap, processes/threads/scheduling |
| `inference-serving/` | The serving-engine core — prefill vs. decode, continuous batching, KV cache, PagedAttention, block allocation, quantization |
| `benchmarking-and-metrics/` | Throughput, TTFT, TPOT, latency percentiles, Poisson workload generation |
| `python-systems/` | Concurrency model as it shows up in `engine/` — sync vs. async, GIL, why the engine loop is plain synchronous Python |
| `testing-and-correctness/` | Proving an optimization didn't silently break output — golden-output regression testing, determinism, differential testing |

## Concept Map

### 🖥️ gpu-and-hardware/ — GPU compute & memory internals · `0 / 1`

- [ ] **[floating-point-determinism.md](./gpu-and-hardware/floating-point-determinism.md)** — why a mathematically-equivalent recomputation (different cache shape, different batching, fused vs. explicit attention) can flip a near-tied greedy-decode token, and why CPU+eager was the fix here, not GPU-vs-CPU per se

### 🧠 os-and-memory/ — virtual memory, paging, OS primitives · `0 / 0`

### ⚙️ inference-serving/ — scheduling, KV cache, batching, quantization · `0 / 0`

### 📊 benchmarking-and-metrics/ — throughput, latency, workload design · `0 / 0`

### 🐍 python-systems/ — concurrency model in `engine/` · `0 / 0`

### 🧪 testing-and-correctness/ — proving an optimization didn't silently break output · `0 / 1`

- [ ] **[correctness-harness.md](./testing-and-correctness/correctness-harness.md)** — golden-output greedy-decode regression testing; why it's MicroServe's gate before every benchmark, and why quantization forces a different pass/fail bar

## Proposed Next Concepts

- **testing-and-correctness/** ◦ **Differential testing** — comparing two independent implementations instead of one implementation against itself
- **testing-and-correctness/** ◦ **Property-based testing (e.g. Hypothesis)** — generating many random inputs to check an invariant holds; a natural fit for allocator/block-table logic
- **gpu-and-hardware/** ◦ **Mixed-precision accumulation strategies** — how bf16/fp16 kernels choose to accumulate in fp32 vs. round at every step, and why that changes how often near-ties occur
- **gpu-and-hardware/** ◦ **cuBLAS/cuDNN algorithm heuristics** — the broader family behind this project's GEMM shape-selection finding; relevant for anyone tuning GPU kernels for throughput
- **gpu-and-hardware/** ◦ **Flash Attention's tiling strategy** — a fused kernel built around a specific, shape-dependent accumulation order; directly related to why SDPA and eager attention diverged here
