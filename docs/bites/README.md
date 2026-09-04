# docs/bites/

Deep-dive companion to `docs/concepts.md`. Where `concepts.md` is a terse,
code-anchored glossary entry, a **bite** is a full placement-ready write-up of
one systems/GPU/inference concept — problem, mechanics, usage, MicroServe
placement, gotchas, interview questions. Written via the `/add-bites` command
(`.claude/commands/add-bites.md`).

**Progress:** `0 / 7` (0%)
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
| `data-structures/` | Technique → structure selection for systems code — free lists, queues, indexing, reclamation, and the access patterns that pick them |

## Concept Map

### 🖥️ gpu-and-hardware/ — GPU compute & memory internals · `0 / 1`

- [ ] **[floating-point-determinism.md](./gpu-and-hardware/floating-point-determinism.md)** — why a mathematically-equivalent recomputation (different cache shape, different batching, fused vs. explicit attention) can flip a near-tied greedy-decode token, and why CPU+eager was the fix here, not GPU-vs-CPU per se

### 🧠 os-and-memory/ — virtual memory, paging, OS primitives · `0 / 0`

### ⚙️ inference-serving/ — scheduling, KV cache, batching, quantization · `0 / 0`

### 📊 benchmarking-and-metrics/ — throughput, latency, workload design · `0 / 0`

### 🐍 python-systems/ — concurrency model in `engine/` · `0 / 0`

### 🧱 data-structures/ — technique → structure selection for systems code · `0 / 5`

- [ ] **[choosing-a-data-structure.md](./data-structures/choosing-a-data-structure.md)** — the index bite: how the hot-path access pattern (not the data) selects the structure, and the recurring technique→structure pairings a systems engineer meets
- [ ] **[free-lists-and-block-allocation.md](./data-structures/free-lists-and-block-allocation.md)** — stack vs. queue vs. set vs. bitmap vs. intrusive list for arena/slab/block allocators; why interchangeable fixed-size slots make a LIFO stack optimal, and why a `set` hides double-free
- [ ] **[queues-and-priority-in-schedulers.md](./data-structures/queues-and-priority-in-schedulers.md)** — why FCFS needs only a deque, when a heap actually earns its O(log n), ring buffers for bounded handoff, and how preemption policy picks the structure
- [ ] **[indexing-structures.md](./data-structures/indexing-structures.md)** — direct-indexed arrays vs. hash maps vs. sorted arrays vs. radix trees; why a block table is a single-level page table while an OS needs four
- [ ] **[reclamation-strategies.md](./data-structures/reclamation-strategies.md)** — ref counting vs. scope-based vs. tracing GC vs. epoch reclamation; why a pool at 98% utilization can't tolerate "eventually", and why `-= 1` beats `= 0`

### 🧪 testing-and-correctness/ — proving an optimization didn't silently break output · `0 / 1`

- [ ] **[correctness-harness.md](./testing-and-correctness/correctness-harness.md)** — golden-output greedy-decode regression testing; why it's MicroServe's gate before every benchmark, and why quantization forces a different pass/fail bar

## Proposed Next Concepts

- **testing-and-correctness/** ◦ **Differential testing** — comparing two independent implementations instead of one implementation against itself
- **testing-and-correctness/** ◦ **Property-based testing (e.g. Hypothesis)** — generating many random inputs to check an invariant holds; a natural fit for allocator/block-table logic
- **gpu-and-hardware/** ◦ **Mixed-precision accumulation strategies** — how bf16/fp16 kernels choose to accumulate in fp32 vs. round at every step, and why that changes how often near-ties occur
- **gpu-and-hardware/** ◦ **cuBLAS/cuDNN algorithm heuristics** — the broader family behind this project's GEMM shape-selection finding; relevant for anyone tuning GPU kernels for throughput
- **gpu-and-hardware/** ◦ **Flash Attention's tiling strategy** — a fused kernel built around a specific, shape-dependent accumulation order; directly related to why SDPA and eager attention diverged here
- **data-structures/** ◦ **Slab allocation** — the Linux kernel's fixed-size object cache; the direct ancestor of the block pool
- **data-structures/** ◦ **Buddy allocator** — the classic answer to "I need *contiguous* blocks", and why paging made that need disappear here
- **data-structures/** ◦ **Copy-on-write** — the M6 mechanism; two block tables pointing at one ref-counted block, and `fork()` as the ancestor
- **data-structures/** ◦ **Lock-free queues (SPSC/MPMC)** — the ring buffer that survives concurrency; relevant the moment a serving layer exists
- **os-and-memory/** ◦ **TLB and translation caching** — why a multi-level page-table walk is affordable at all, and what "TLB reach" means
- **os-and-memory/** ◦ **MLFQ** — how real schedulers approximate SJF without predicting job length
- **inference-serving/** ◦ **Preemption: recompute vs. swap** — the M3 Phase 4 decision; what happens when the pool fills mid-generation
- **inference-serving/** ◦ **Chunked prefill** — splitting one huge prefill across iterations so it stops blocking decode
- **inference-serving/** ◦ **Prefix caching (APC)** — vLLM's production feature built entirely on ref-counted shared blocks
- **benchmarking-and-metrics/** ◦ **Little's Law** — relates queue depth, arrival rate and latency; the fastest sanity check on a scheduler under load
