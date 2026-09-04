# Choosing a Data Structure (Systems Edition)

> The structure isn't chosen from the data you're storing — it's chosen from the
> operation you do a million times a second.

*Scope note: this is the index bite for the `data-structures/` category. It gives
the decision rule; the four sibling bites go deep on each family —
[[free-lists-and-block-allocation]], [[queues-and-priority-in-schedulers]],
[[indexing-structures]], [[reclamation-strategies]].*

---

## The Problem

In application code the data structure barely matters — a `dict` is fast enough
for everything and the bottleneck is the network. In systems code the structure
*is* the design. MicroServe's `BlockAllocator` runs on every admitted sequence
and every block boundary crossed during decode; at 64 concurrent sequences
generating 256 tokens each that is thousands of allocate/free calls per second,
interleaved with the actual model forward pass. If the allocator is O(n) in the
pool size, the pool size becomes a performance parameter — and worse, the
allocator's own cost lands *inside* the benchmark numbers the project exists to
produce. You'd be measuring your bookkeeping, not your engine.

The trap is that the obvious structure is usually the wrong one, and it *works*.
A `set` for the free list is correct. A `dict` for the block table is correct. A
sorted list scanned linearly for the next sequence to schedule is correct. They
all pass the tests and then quietly cap your throughput.

---

## Mental Model

Don't ask "what am I storing?" Ask, in this order:

```
1. What is the ONE operation on the hot path?        -> that must be O(1)
2. Is the key space dense and small?                 -> array, not hash map
3. Does order matter, and which end?                 -> stack / queue / heap
4. What's the invariant I'd hate to violate silently? -> pick the structure
                                                        that makes it checkable
```

Step 4 is the one people skip, and it is the one that matters most in an
allocator. See [[free-lists-and-block-allocation]].

---

## How It Works — the recurring pairings

### Reusable identical slots → **stack (LIFO free list)**

Slab and arena allocators, page-frame allocators, connection pools, and
MicroServe's `BlockAllocator`. Every slot is interchangeable, so there is no
"best" one to return — which means you can pick the *cheapest* one, and LIFO is
cheapest twice over: O(1) push and pop, and the block you just freed is the one
still resident in cache.

> 🔹 **arena allocator** — carve one big allocation up yourself instead of asking
> the OS for each piece. The whole point of MicroServe's pool.

### Bounded producer/consumer handoff → **ring buffer**

Fixed capacity known up front, single producer, single consumer, no allocation
per item. Audio buffers, NIC descriptor rings, CUDA stream command queues,
lock-free work queues. If you're allocating a node per item to implement a queue
whose maximum depth you already know, you've picked wrong.

### "Who's next?" where next is defined by a *policy* → **heap**

Priority scheduling, timer wheels, deadline queues, EDF. O(log n) push/pop, and
critically it maintains only the *minimum*, not a total order — which is all a
scheduler ever asks for. Full sorting is strictly more work than the question
requires.

### "Who's next?" where next means *oldest* → **deque**

FCFS needs no heap. If arrival order *is* priority order, the queue is already
sorted by construction and a deque gives you O(1) at both ends. MicroServe's
`Scheduler.waiting` is exactly this. See
[[queues-and-priority-in-schedulers]].

### Dense integer key → object → **direct-indexed array**

Block id → ref count. Logical block index → physical block id. Page number →
frame number. When the key is already a small dense integer, hashing it is
strictly wasted work: you compute a hash to recover the index you started with.
See [[indexing-structures]].

### Sparse or non-integer key → **hash map**

Request id → `Sequence`, especially once ids are UUIDs or monotonic counters
that never get reused. The moment key space is sparse, the array becomes mostly
holes and the hash map wins on memory.

### Shared ownership with no single owner → **reference counting**

Copy-on-write pages, shared prefix blocks (MicroServe M6), file descriptors,
`shared_ptr`. Deterministic reclamation, no pause, and the counter doubles as a
correctness assertion. See [[reclamation-strategies]].

### Variable-size objects in a fixed pool → **don't. Fix the size.**

Variable-size allocation gives you external fragmentation and a best-fit/first-fit
policy question with no good answer. Fixed-size blocks trade that for *internal*
fragmentation, which is bounded by the block size and therefore predictable. This
is the entire argument for paging, in OSTEP and in PagedAttention alike.

---

## Usage

The two-line version of each pairing, as it appears in `engine/`:

```python
# dense integer key, hot path -> direct-indexed array
self.ref_counter = [0] * num_blocks          # ref_counter[block_id]

# interchangeable slots, want the warmest -> stack
self.free_list = deque(range(num_blocks))    # append / pop  (LIFO)

# arrival order IS priority order -> deque, no heap needed
self.waiting: deque[Sequence] = deque()      # append / popleft (FIFO)

# small n, filtered wholesale each step -> plain list, rebuilt
self.running = [s for s in self.running if not s.is_finished]
```

That last one is worth noticing: `running` is O(n) per step and that is *correct*,
because n is capped at `max_batch_size` (~64) and the rebuild is one pass over a
contiguous list. Optimising it would be measurable in neither direction.

---

## Where This Fits in MicroServe

- `engine/kv_cache.py` — `free_list` (stack), `ref_counter` (direct-indexed array).
  Milestone 3, Phase 1.
- `engine/scheduler.py` — `waiting` (deque/FIFO), `running` (list, rebuilt).
  Milestone 2, and revisited in M3 Phase 4 when block exhaustion adds a
  preemption policy and possibly a real priority order.
- `engine/sequence.py` — `block_table: list[int]`, a direct-indexed array where
  the list index *is* the logical block number. Milestone 3, Phase 2.
- `bench/` — latency percentiles need a sorted array, not a heap: you compute p50
  and p99 once, offline, over the whole sample.

---

## Gotchas & Trade-offs

- **"O(1) amortised" is not O(1) on a latency budget.** A Python `list` append
  that triggers a realloc, or a hash map that rehashes, is a rare long tail. It
  doesn't matter for the free list; it matters a great deal in a per-token path
  where you're reporting p99.
- **Big-O hides the constant, and at these sizes the constant is everything.**
  `deque` and `list` are both O(1) as stacks; `list` is meaningfully faster
  because it's a flat array of pointers with no block bookkeeping. At
  `num_blocks = 5461` neither is measurable. Know which fact you're relying on.
- **Choosing a structure that can't express the bug.** A `set` free list makes
  double-free *invisible* — inserting an already-present element is a silent
  no-op. A list free list makes it invisible in a different way: the id is just
  in there twice. Neither detects it; the separate `ref_counter` does. Pick
  structures so your invariants are checkable, not just your operations fast.
- **Premature `dict`.** Reaching for a dict when the key is `range(n)` costs a
  hash per access, ~3x the memory, and loses the ability to iterate in index
  order for free.
- **Ignoring cache locality.** Two structures with identical complexity can
  differ 10x in wall clock because one is a contiguous array and the other is a
  pointer chase. This is why intrusive linked lists (nodes stored *inside* the
  objects) beat external linked lists in kernels, and why arrays beat both when
  they fit.

---

## Comparison Table

| Access pattern | Structure | Why |
|---|---|---|
| Any free slot, want warmest | Stack (`list`/`deque` LIFO) | O(1) both ends; recency = cache warmth |
| Any free slot, want fairest reuse | Queue (`deque` FIFO) | Delays reuse; useful for use-after-free detection |
| Free slot, need *contiguous run* of them | Bitmap / buddy allocator | Only these can answer "n adjacent" |
| Next by arrival | `deque` | Already ordered; heap is wasted |
| Next by policy score | Heap | Maintains min only, O(log n) |
| Bounded handoff, known depth | Ring buffer | Zero per-item allocation |
| Dense int key → value | Array | Index *is* the hash |
| Sparse/opaque key → value | Hash map | Array would be mostly holes |
| Sparse key, need range queries | Sorted array / B-tree / radix | Hash map can't answer "next after k" |

---

## Interview & Career Angle

This is the difference between "I used a dict" and "the key space is dense and
the operation is on the decode path, so it's an array." ML-systems interviews
rarely ask you to invert a binary tree; they ask you to design a KV-cache
allocator or a request scheduler and then probe *why* each structure. The
strongest signal you can give is naming the access pattern before naming the
structure, and naming what you gave up.

Adjacent concepts worth learning next:

- **Cache-oblivious data structures** — why layout beats complexity at small n.
- **Lock-free queues (MPMC/SPSC)** — the version of the ring buffer that survives
  concurrency; relevant the moment a serving layer exists.
- **Buddy allocator** — the classic answer to "I need *contiguous* blocks", which
  MicroServe deliberately doesn't need.
- **Slab allocation** — the Linux kernel's answer to fixed-size object pools; the
  direct ancestor of the block pool.

---

## Hard Questions I Should Be Able to Answer

1. Your paged engine's `BlockAllocator` uses a `set` for its free list. Everything
   passes. What class of bug have you just made undetectable, and what would the
   symptom look like three weeks later in production?
2. M6 adds prefix sharing: a block can now be owned by several sequences and you
   must be able to ask "which sequences share block 412?" — which structures
   change, and which stay?
3. An interviewer says: "design the memory manager for an LLM inference server."
   Sketch it in five minutes, naming each structure and the access pattern that
   selected it.
4. Your scheduler's `running` list is O(n) to filter and n is 64. A colleague
   wants to make it a dict for O(1) removal. Argue the other side, then say what
   would change your mind.
5. Profiling shows the allocator is 8% of step time. Where do you look first, and
   what would make you conclude the structure is the problem versus the calling
   pattern?

---

## Answers

**1.** Double free. `set.add()` on an element already present is a silent no-op,
so the free list stays consistent and `num_free` is *wrong* — it under-counts,
because the block only appears once no matter how many times it was freed. The
allocator then hands that block to a second sequence while the first still has it
in its block table. Both write KV into the same physical block, each clobbering
the other's history at whatever offsets overlap. There is no crash and no NaN:
attention reads real, well-formed floats that belong to a different conversation.
The symptom is occasional incoherent output under high concurrency — precisely
when block reuse is most frequent — and it will not reproduce on a single
sequence, which is where you'll try to debug it. The structural fix is not a
better free list; it's the separate `ref_counter`, which makes "was this block
allocated?" a question with an answer.

**2.** The free list is unchanged — a block is still free or not, and reclamation
is still LIFO. `ref_counter` is unchanged in *shape* but its values now exceed 1,
which retires every `assert ref_count in (0, 1)` you wrote in M3 and turns
`free()` into a genuine decrement (which is why it should already be one). The
block table is unchanged: it's per-sequence and still direct-indexed. What's
*new* is the reverse query — block → owners — and you should resist adding a
structure for it. Reference counting deliberately doesn't maintain the owner set,
because maintaining it costs a set insert/remove on every allocate and free to
answer a question the engine never asks on the hot path. If you need it for
debugging, walk the sequences; it's O(total blocks) and you're not in a hot path.
Prefix sharing additionally needs prefix hash → block id, which *is* a new
structure and *is* a hash map, because the key is a content hash and therefore
sparse by construction.

**3.** One pre-allocated pool tensor, fixed-size blocks, block id = index into
dim 0 — fixed size because variable-size gives external fragmentation with no
good fit policy, and internal fragmentation is bounded by block size instead.
Free list as a LIFO stack: slots are interchangeable so take the warmest, O(1)
both ways. Ref counts as a dense array indexed by block id, because block ids are
`range(num_blocks)` and hashing them is recovering an index you already have.
Per-sequence block table as a list where the index is the logical block number —
same reasoning, one level up. Admission queue as a FIFO deque under FCFS; swap it
for a heap only when you introduce a priority the arrival order doesn't already
encode. Then say the thing that separates senior from mid: exhaustion is *normal*
here, not exceptional, so the allocator's failure is a typed catchable event and
the scheduler owns the preemption policy.

**4.** At n = 64 a list rebuild is one linear pass over 64 contiguous pointers —
a few hundred nanoseconds, against a model forward pass measured in
milliseconds. It's four orders of magnitude below the noise floor. The dict costs
you deterministic iteration order, which matters because the batch's row order
must line up with the cache's row order; you'd be trading a free invariant for an
unmeasurable win. What would change my mind: `max_batch_size` rising to the
thousands (a real serving system), or removal moving from once-per-step to
once-per-sequence-per-step. Measure first — if it doesn't show in the profile, it
isn't a problem.

**5.** First: is it 8% of a step that's dominated by a tiny model on CPU? On the
real benchmark model the forward pass grows by orders of magnitude and the
allocator doesn't, so 8% may be an artifact of the dev model. If it's real, look
at *call frequency* before structure: allocating one block at a time in a Python
loop over the batch is `batch_size` calls per step where one batched call would
do, and that's a calling-pattern problem no structure fixes. Structure is the
culprit only if a single call is slow — which for an O(1) stack means you've
accidentally made it not-O(1) (list `.pop(0)`, a set being rebuilt, a sort
hiding in a comprehension). The tell is scaling: time the allocator against
`num_blocks`. Flat means the structure is fine and your call pattern isn't.

---

## Rules of Thumb

> Name the hot-path operation before naming the structure. If you can't say
> which operation must be O(1), you're not ready to choose.

> If the key is a small dense integer, it's an array. Every time. A hash map is
> what you use when you *can't* index directly.

> Prefer the structure that makes your invariant checkable over the one that's
> marginally faster. At n in the thousands, correctness is the only axis that
> differs measurably.

> Fixed-size blocks over variable-size allocation: swap unbounded external
> fragmentation for bounded internal fragmentation.

---

## TL;DR

- The access pattern selects the structure, not the data. One question: what runs
  on the hot path, and how often?
- Dense integer keys → arrays; sparse keys → hash maps; interchangeable slots →
  LIFO stacks; policy-ordered work → heaps; arrival-ordered work → deques.
- At MicroServe's sizes the structures are all fast enough. Choose the one that
  makes double-free, leaks, and off-by-ones *visible*.
