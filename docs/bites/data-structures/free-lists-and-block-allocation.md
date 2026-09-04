# Free Lists and Block Allocation

> When every free slot is interchangeable, allocation stops being a search
> problem and becomes a pop from a stack.

---

## The Problem

MicroServe pre-allocates one KV tensor of shape
`[num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]` and never calls
the CUDA allocator again — because `cudaMalloc` synchronises the device and would
land its cost inside the very benchmark this project exists to produce. That
converts a memory-allocation problem into a *bookkeeping* problem: which of the
5,461 indices into dim 0 are currently in use?

Naively that's a search: scan for a free one. At 64 concurrent sequences crossing
a block boundary every 16 decoded tokens, plus a burst of `ceil(prompt_len/16)`
allocations on every admission, a linear scan makes pool size a performance
parameter. It also gets worse exactly when the pool is nearly full — the
condition you're deliberately engineering the system to sit in, since utilization
is the whole point of paging.

The free list turns the search into an O(1) pop, permanently, by maintaining the
answer instead of computing it.

> 🔹 **slab allocator** — the Linux kernel's fixed-size object pool; the direct
> ancestor of this design. Same trick: uniform objects, so a free list beats a
> search.

---

## Mental Model

The pool is a hotel with identical rooms. The free list is the key rack behind the
desk. You never walk the corridors looking for an empty room; you take whatever
key is on top of the rack, and returned keys go back on top.

```
pool:      [0][1][2][3][4][5][6][7]      <- the actual memory, never moves
free_list: 7 6 5 4 3 2 1 0               <- just integers, the top is the right end
ref_count: 0 0 0 0 0 0 0 0               <- separate, and it's the source of truth
```

Two facts that matter: the free list holds *ids*, not memory, so it costs
`num_blocks` integers total; and it is a **derived** structure — `ref_count` is
what actually knows whether a block is allocated.

---

## How It Works — the candidate structures

### Stack (LIFO) — `list` or `deque`, append/pop

O(1) both ways, and the block you just freed is the one still resident in L2 and
in the GPU's page tables. MicroServe uses this. In OSTEP terms it's the
free-frame list, and the kernel makes the same choice for the same reason.

- **Buys**: fastest possible; best temporal locality; trivially correct.
- **Costs**: reuses blocks immediately, so a use-after-free is *maximally* likely
  to silently succeed by reading plausible data. LIFO is the worst order for
  catching dangling block ids.
- **Used by**: Linux page allocator per-CPU free lists, most slab allocators,
  vLLM's block manager.

### Queue (FIFO) — `deque`, append/popleft

Same complexity, opposite recency. A freed block waits until every other free
block has been handed out before coming back.

- **Buys**: a use-after-free has the longest possible window before the data is
  overwritten, so bugs surface as *stale but self-consistent* KV rather than
  another sequence's KV — much easier to spot. Debug builds of real allocators
  often switch to FIFO for exactly this.
- **Costs**: worst cache locality; you touch a cold block every allocation.
- **When to pick**: debugging a suspected block-table bug. It's a one-line change
  (`pop()` → `popleft()`) and worth knowing you can make it.

### Set

Membership testing in O(1), plus `pop()` for an arbitrary element.

- **Buys**: makes "is this block free?" a direct query.
- **Costs**: throws away ordering for nothing, ~3-4x the memory of a list of
  ints, and — the real killer — **`add()` on a present element is a silent
  no-op**, which makes double-free undetectable at the free list. It also gives
  no locality guarantee at all.
- **When to pick**: essentially never here. If you want membership testing, that's
  what `ref_count` gives you already.

### Bitmap

One bit per block; allocation is a scan for a set bit, usually accelerated with
word-at-a-time tricks and `__builtin_ctz`.

- **Buys**: tiny (5,461 blocks = 683 bytes); and it is the *only* structure here
  that can answer **"give me n contiguous blocks"**, because adjacency in id space
  is visible in the layout.
- **Costs**: allocation is O(num_blocks/64) not O(1); needs a "next free hint" to
  stay fast; more code.
- **When to pick**: when contiguity matters. MicroServe's whole thesis is that it
  *doesn't* — gather-then-attend removes the need for adjacent blocks — so the
  bitmap's one advantage is precisely the thing paging made irrelevant. That's
  the argument for the stack, stated properly.

### Intrusive linked list

Store the "next free" pointer *inside* the free object itself, so the free list
costs zero extra memory.

- **Buys**: zero overhead; the classic kernel and embedded-allocator trick.
- **Costs**: pointer chasing, so bad locality; and it requires the free object's
  memory to be writable and reinterpretable, which is a C idea. In MicroServe the
  free slots are GPU tensor memory — writing link pointers into VRAM to save 22 KB
  of host RAM would be absurd.
- **When to pick**: C/C++/kernel allocators where the metadata-to-object ratio
  actually matters. Worth *knowing* because it's a standard interview reference
  point; not worth building here.

### Buddy allocator

Power-of-two-sized blocks, split and coalesced on demand.

- **Buys**: variable-size allocation with bounded fragmentation and O(log n)
  coalescing.
- **Costs**: substantial complexity, and it exists to solve variable-size
  allocation — a problem fixed-size blocks define out of existence.
- **When to pick**: not here. Name it in an interview to show you know why fixed
  size was the better trade.

---

## Usage

MicroServe's, in full — the structure is four lines, and the discipline around it
is the rest:

```python
class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.free_list = deque(range(num_blocks))   # LIFO: append / pop
        self.ref_counter = [0] * num_blocks         # the source of truth

    def allocate(self, n: int) -> list[int]:
        if len(self.free_list) < n:                 # check BEFORE mutating
            raise BlockExhausted(...)
        ids = [self.free_list.pop() for _ in range(n)]
        for b in ids:
            self.ref_counter[b] = 1
        return ids
```

Switching to FIFO to hunt a use-after-free is one character: `pop()` →
`popleft()`.

---

## Where This Fits in MicroServe

`engine/kv_cache.py`, Milestone 3 Phase 1 — `BlockAllocator.free_list` and
`ref_counter`, sitting under `PagedKVPool`. Consumed by `Sequence.block_table`
(Phase 2) and by the scheduler's admission and preemption path (Phase 4), where
`BlockExhausted` stops being an error and becomes a scheduling signal.

The OS equivalent is OSTEP Ch 18–19's free-frame list: the kernel maintains a
list of free physical frames, hands them out on page fault, and reclaims them on
process exit. Identical structure, identical reasoning, different nouns.

---

## Gotchas & Trade-offs

- **The free list is not the source of truth; the ref counts are.** Any check that
  reads only the free list can be fooled. `free()` must validate against
  `ref_counter`, not against `block_id in free_list` (which is O(n) anyway).
- **Partial mutation on failure leaks blocks forever.** `allocate(n)` that pops
  three then raises has orphaned three ids that no block table records, so nothing
  will ever free them. The pool shrinks silently over hours of running. Validate
  fully, then mutate — in *both* directions: a half-applied `free` is worse, since
  it puts a live block back in circulation.
- **Duplicate ids inside one `free()` call.** `free([x, x])` from a block table
  that somehow recorded x twice will pass a naive per-id check, because both
  entries are validated against the same pre-mutation count. Count the ids
  (`Counter`), don't just test membership.
- **Exhaustion is not an error.** In a paged engine, running out of blocks is the
  *expected* steady state under load — it's what triggers preemption. Raise a
  named catchable type, never let an `IndexError` escape from `pop()`, or the
  scheduler cannot distinguish "we're full" from "there's a bug."
- **`allocate(0)` and `free([])` must be no-ops, not special cases.** Both happen
  routinely: a decode step where no sequence crossed a block boundary frees
  nothing.
- **Don't zero on free.** It's tempting as a hygiene measure and it costs a full
  block memset (192 KiB) per free on the GPU. It's unnecessary here because the
  pool was zeroed once at construction and a reused block always holds a dead
  sequence's *finite* KV — which is harmless, since causality masks the padded
  tail. (Uninitialised memory would not be harmless: `inf` in a masked slot gives
  `0 * inf = NaN` and silently destroys the output.)

---

## Comparison Table

| Structure | Alloc | Free | Contiguous runs? | Locality | Pick when |
|---|---|---|---|---|---|
| Stack (LIFO) | O(1) | O(1) | no | best | default — interchangeable slots |
| Queue (FIFO) | O(1) | O(1) | no | worst | debugging use-after-free |
| Set | O(1) | O(1) | no | none | never (hides double-free) |
| Bitmap | O(n/64) | O(1) | **yes** | good | you need adjacent blocks |
| Intrusive list | O(1) | O(1) | no | poor | C/kernel, metadata size matters |
| Buddy | O(log n) | O(log n) | yes | medium | variable-size allocation |

---

## Interview & Career Angle

"Design a memory allocator" is a standard systems interview, and the paged-KV
version of it is now a standard *ML*-systems interview. The differentiator isn't
knowing what a free list is — it's articulating why fixed-size blocks let you use
the simplest possible structure, and why that simplification was bought by the
gather-then-attend design decision upstream. Interviewers probe: what happens when
it's full? (preemption policy, not an exception) and how do you catch a double
free? (a separate counter, because the free list structurally cannot).

Adjacent concepts worth learning next:

- **[[reclamation-strategies]]** — ref counting vs. scope vs. tracing, and why
  ref counting fits an inference engine.
- **Slab allocation** — the kernel's fixed-size object cache; same idea, richer
  per-CPU structure.
- **Buddy allocator** — the answer when you *do* need contiguity.
- **Memory pool fragmentation analysis** — internal vs. external, and why paging
  trades one for the other.

---

## Hard Questions I Should Be Able to Answer

1. Your engine has been up 12 hours. `num_free` reports 400 blocks free, but the
   sum of all live sequences' block tables accounts for only 4,800 of 5,461. Where
   did 261 blocks go, and what code path produces that?
2. An interviewer asks why you didn't use a bitmap. Answer in a way that shows you
   know what a bitmap buys, not just that a stack is simpler.
3. You suspect a block-table off-by-one is causing rare garbage output under load,
   but you can't reproduce it in a test. What one-line change to the allocator
   makes the bug more likely to produce an *obvious* symptom instead of a subtle
   one, and why does it work?
4. Sizing: with a 6 GiB KV budget on a 7B 4-bit model, would you rather have
   `BLOCK_SIZE = 16` or `BLOCK_SIZE = 64`, and what measurement would settle it?
5. Under sustained load, `allocate()` raises `BlockExhausted` several times per
   second and throughput is fine. Is this a bug?

---

## Answers

**1.** Leaked blocks — allocated, never recorded, never freed. The classic source
is a `allocate(n)` that mutated before it validated: it popped some ids, hit the
capacity check or an exception partway, and unwound without returning them. The
second source is an admission path that allocates blocks for a sequence and then
rejects the sequence (queue full, bad request, exception during prefill) without a
`free()` in the failure path — a missing `try/finally` in effect. Both are
invisible to every test that only does happy-path allocate/free. The diagnostic is
exactly the sum you just computed: maintain the invariant `num_free + sum(1 for c
in ref_counter if c > 0) == num_blocks` and assert it periodically; when it breaks,
the id set difference tells you which blocks leaked, and their ids often cluster
around the failing path.

**2.** A bitmap is one bit per block, so it's 8x smaller than a list of int8s and
680 bytes for the whole pool — and more importantly it's the only representation
in which *adjacency is visible*, so it can answer "give me 8 contiguous blocks"
with a word-scan. That capability is the reason kernels use them for physical
frame allocation, where DMA and huge pages need contiguity. MicroServe needs the
exact opposite: PagedAttention's whole premise is that a sequence's KV does *not*
have to be contiguous, because we gather scattered blocks into a scratch tensor
before attending. So the bitmap's unique advantage is the property we engineered
away, and what's left is an O(n/64) scan versus an O(1) pop. The stack wins on
what remains.

**3.** Change `pop()` to `popleft()` — LIFO to FIFO. Under LIFO, a block freed by
sequence A is immediately handed to sequence B, so a dangling reference in A's
block table reads B's freshly written, perfectly plausible KV, and the output is
subtly wrong in a way that looks like a model quality issue. Under FIFO the block
sits at the back of a 5,000-deep queue, so A's stale read returns A's *own* old
KV for a long window — self-consistent, stale, and much more likely to produce a
visibly repeating or frozen output. You've converted a heisenbug into a loud one
by maximising the time between free and reuse. Same trick as a debug allocator's
delayed-free quarantine.

**4.** 16, and the measurement is a fragmentation-vs-locality sweep, not a
guess. Internal fragmentation is bounded by `BLOCK_SIZE - 1` per sequence, so 64
wastes up to 63 slots per sequence against 15 — at 64 concurrent sequences that's
roughly 3,000 wasted token-slots vs 500, and on a 7B model each token-slot is far
more expensive than the 12 KiB it costs on the 0.5B dev model. Against that, 64
gives 4x fewer, 4x larger reads in the gather, which is a memory-bandwidth-bound
path. The settling measurement is running the real workload at both settings and
reporting peak concurrent sequences *and* TPOT: 64 should show higher TPOT
improvement only if the gather is actually bandwidth-limited, and lower peak
concurrency for certain. If TPOT doesn't move, fragmentation decides it and 16
wins.

**5.** Not necessarily — that's the system working. In a paged engine the steady
state under sustained load *is* a full pool, because you sized the pool to be the
binding constraint and the scheduler admits until it binds. `BlockExhausted`
becomes the backpressure signal: stop admitting, or preempt a low-priority
sequence. It's a bug only if (a) throughput is degrading, which suggests
preemption thrashing — sequences being preempted and recomputed faster than they
make progress — or (b) it's raised from the *decode* path rather than admission,
meaning a running sequence can't grow and you're now forced to preempt mid-flight,
which is the case a per-sequence block cap at admission is supposed to prevent.
Log `num_free` alongside scheduling decisions so you can tell those apart.

---

## Rules of Thumb

> Interchangeable fixed-size slots → LIFO stack. Reach for anything fancier only
> when you need contiguity or variable sizes.

> Never let the free list be the source of truth. Keep a per-slot count and
> validate against it; the free list is a cache of "which ones are free", not the
> record of it.

> Both `allocate` and `free` are all-or-nothing. Validate completely, then mutate.

> Flip LIFO to FIFO when hunting a use-after-free. It's one character and it turns
> silent corruption into visible staleness.

---

## TL;DR

- Fixed-size interchangeable blocks turn allocation from a search into a stack
  pop; that simplification was bought upstream by deciding blocks need not be
  contiguous.
- The free list is derived. `ref_count` is the truth, and it's what makes double
  free and leaks detectable — a `set` free list would hide both.
- Exhaustion is a normal, expected, *typed* event in a paged engine, because it's
  what triggers preemption.
