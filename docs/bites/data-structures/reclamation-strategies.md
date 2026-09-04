# Reclamation Strategies: Ref Counting, Scope, and Tracing

> Every resource pool needs an answer to "when is this safe to reuse?" — and the
> answer determines whether your bugs are loud or silent.

---

## The Problem

A KV block is safe to reuse when no sequence will ever read it again. In
Milestone 3 that's easy: a block belongs to exactly one sequence, so it's free
when that sequence finishes. Milestone 6 breaks the easy case — prefix sharing
means two sequences with the same system prompt point at the *same* physical
blocks, and now "sequence A finished" no longer implies "A's blocks are free."

Get this wrong in the safe direction and you leak: blocks are never returned, the
pool drains over hours, and admission slowly grinds to a halt for no visible
reason. Get it wrong in the unsafe direction and a live sequence's KV gets
overwritten by a new one — no crash, no NaN, just attention reading well-formed
floats from someone else's conversation. That's the worst bug class in the
project, and which strategy you pick determines how detectable it is.

---

## Mental Model

Three answers to "when is it safe?", in increasing order of what they can express:

```
scope-based   : "when the owner's scope ends"    -- one owner, statically known
ref counting  : "when the last owner lets go"    -- many owners, dynamically known
tracing (GC)  : "when nothing can reach it"      -- arbitrary graphs, incl. cycles
```

You move right only when the left one can't express your ownership. MicroServe
sits at ref counting from M3 onward, deliberately, before it needs the second
column — because retrofitting ownership semantics into a working allocator is a
rewrite, and adding a counter that's always 1 is a line of code.

---

## How It Works — the three strategies

### Scope-based / single-owner

Ownership is structural: one owner, freed when that owner dies. RAII in C++,
`Drop` in Rust, `with` blocks and context managers in Python, `try/finally`.

- **Buys**: zero runtime cost, zero metadata, and — the real prize — *the compiler
  or the language structure enforces it*. You can't forget.
- **Costs**: cannot express sharing at all. The moment two things need the same
  resource you're forced to copy, or to escape into a different scheme.
- **In MicroServe**: this is effectively what M3 is. Blocks are owned by exactly
  one `Sequence` and freed in the scheduler's eviction path when it reaches
  `FINISHED`. The `try/finally` discipline matters here: a sequence that fails
  during prefill must still return its blocks, or you leak.

### Reference counting

A per-resource integer. Increment on acquire, decrement on release, reclaim at
zero.

- **Buys**: **deterministic** reclamation — the block is free at a known
  instruction, not at some future collection. No pauses, which matters when your
  p99 latency is a reported metric. And the counter doubles as an assertion: "is
  this block allocated?" has a real answer, which is what makes double-free and
  leaks detectable at all.
- **Costs**: metadata per resource (trivial here: one int per block); an
  increment/decrement on every share; and — the famous one — **it cannot collect
  cycles**. If A refers to B and B refers to A, both counts stay at 1 forever.
- **In MicroServe**: `BlockAllocator.ref_counter`, always 0 or 1 in M3 by locked
  decision, exceeding 1 in M6 when prefixes are shared. Cycles are structurally
  impossible — blocks don't reference other blocks — so the one real weakness of
  ref counting doesn't apply. That's why it's the right fit and not just a
  convenient one.
- **Used by**: `shared_ptr`, CPython's own object model, POSIX file descriptors
  and inodes, Linux page frames (`struct page._refcount`), vLLM's block manager.

### Tracing garbage collection

Periodically walk from a root set, mark everything reachable, sweep the rest.

- **Buys**: handles arbitrary object graphs including cycles; no per-operation
  cost; the programmer thinks about nothing.
- **Costs**: reclamation is *non-deterministic in time*, which means memory stays
  held for an unbounded interval after last use, and collection introduces pauses.
  Both are disqualifying for a fixed-size pool where the whole design goal is to
  keep utilization at 98% — you cannot run near capacity when reclamation is
  "eventually."
- **In MicroServe**: never, for the pool. Python's own GC is of course running
  underneath for the host objects, and it's worth noticing the split: the
  *tensor pool* is manually managed with ref counts while the `Sequence` objects
  around it are garbage collected. That's the standard shape of a systems Python
  program.

### Epoch / deferred reclamation (worth naming)

Free into a quarantine, reclaim only once all readers that could hold a reference
have passed a barrier. RCU in the Linux kernel, hazard pointers, epoch-based
reclamation in lock-free structures.

- **Buys**: safe reclamation without per-access refcount traffic in concurrent
  code.
- **Costs**: delayed reclamation, so memory high-water mark rises; real complexity.
- **When it'd matter here**: only once a serving layer introduces genuine
  concurrency over the pool — explicitly out of scope. Know the name; don't build
  it.

---

## Usage

Ref counting done properly — the two lines that people get wrong:

```python
def free(self, ids: list[int]) -> None:
    # 1. validate EVERYTHING before mutating anything.
    for block_id, wanted in Counter(ids).items():
        if self.ref_counter[block_id] < wanted:
            raise DoubleFreeError(...)

    # 2. DECREMENT, don't assign zero. Identical while counts are 0/1 --
    #    and the entire reason the counter exists is that they won't be.
    for block_id in ids:
        self.ref_counter[block_id] -= 1
        if self.ref_counter[block_id] == 0:
            self.free_list.append(block_id)
```

`self.ref_counter[b] = 0` passes every M3 test and silently reclaims a shared
block the moment its *first* owner finishes in M6.

---

## Where This Fits in MicroServe

`engine/kv_cache.py`, Milestone 3 Phase 1 — `ref_counter` as a plain list indexed
by block id. Locked decision: always 0 or 1 in M3. The structure is present early
specifically so Milestone 6 (prefix sharing, copy-on-write) is an *extension*
rather than a rewrite.

Where `free()` is *called* from is the more interesting design question, and it's
a scheduler concern: a `FINISHED` sequence's blocks are returned in the eviction
path, which is the one place that already knows the sequence's lifecycle. Putting
it in `Sequence.__del__` would be the Python-idiomatic move and the wrong one —
you'd be handing pool reclamation to the garbage collector, reintroducing exactly
the non-determinism ref counting exists to avoid.

The OS equivalents: `struct page._refcount` in Linux, inode and file-descriptor
counts (OSTEP Ch 39), and copy-on-write `fork()` (Ch 5, Ch 23), which is the
direct ancestor of M6's block sharing.

---

## Gotchas & Trade-offs

- **`= 0` instead of `-= 1`.** The single highest-value line in this bite. It
  passes every test in a system where counts are only ever 0 or 1, and it is a
  correctness bug the day sharing arrives.
- **Ref counting doesn't collect cycles.** True in general, irrelevant here
  (blocks don't point at blocks). Say both halves in an interview; saying only the
  first sounds like a memorised weakness.
- **Missing decrements leak; extra decrements corrupt.** They are not symmetric.
  A leak is a slow, visible resource exhaustion you can find with an invariant
  check. An over-decrement hands a live block to a second sequence and produces
  plausible wrong text. Bias your error handling toward leaking.
- **Failure paths are where leaks live.** Happy-path allocate/free is easy; the
  leak is in the admission that allocated blocks and then rejected the request, or
  the prefill that raised halfway. Every allocation needs an owner *before* the
  next thing that can fail.
- **Increment before you publish the reference, decrement after you drop it.**
  The window matters even single-threaded once preemption can interleave.
- **Don't zero the memory on free.** Reclamation is a *bookkeeping* operation. A
  192 KiB memset per freed block would be pure cost, and it's unnecessary because
  a reused block always holds finite (if meaningless) KV, which causality masks
  away. Zeroing once at pool creation is what actually matters — uninitialised
  memory can hold `inf`, and `0 * inf = NaN` silently destroys the output.
- **The invariant worth asserting:**
  `num_free + count(ref_counter[i] > 0) == num_blocks`, checked periodically. It
  catches both leak and over-decrement, and it's three lines.

---

## Comparison Table

| Strategy | Reclaims when | Determinism | Cycles | Cost | Pick when |
|---|---|---|---|---|---|
| Scope / RAII | owner's scope ends | exact | n/a | zero | exactly one owner, statically known |
| Reference counting | last owner releases | exact | ✗ leaks them | one int + inc/dec | many owners, no cycles |
| Tracing GC | "eventually" | none | ✓ | pauses, high-water | arbitrary graphs, latency not critical |
| Epoch / RCU | after a grace period | bounded delay | n/a | complexity | lock-free concurrent readers |

---

## Interview & Career Angle

"How do you know when a KV block can be reused?" is the natural follow-up to any
PagedAttention question, and it's where prefix caching gets designed live. The
strong answer connects three things: ref counting is chosen because reclamation
must be deterministic in a pool running at 98% utilization; its cycle weakness
doesn't apply because the object graph is flat; and the counter is *also* the
mechanism that makes double-free detectable, which no free-list structure can do
on its own. Copy-on-write over ref-counted blocks is then a direct restatement of
`fork()`, and saying so is worth a lot.

Adjacent concepts worth learning next:

- **Copy-on-write** — the M6 mechanism, and the reason ref counts exist now.
- **[[free-lists-and-block-allocation]]** — the structure the counts arbitrate.
- **Prefix caching / automatic prefix caching (APC)** — vLLM's production feature
  built entirely on this foundation.
- **RCU and epoch-based reclamation** — how this problem is solved once readers
  are concurrent.

---

## Hard Questions I Should Be Able to Answer

1. M6 lands. Sequence A and B share blocks `[7, 2, 9, 4]`. A finishes. Walk
   through exactly what `free()` does, and name the one-line bug in M3's
   implementation that would corrupt B here.
2. Your pool loses ~30 blocks an hour and you can't find a missing `free()`. Where
   do you look, and what invariant would have caught it on day one?
3. An interviewer asks why you didn't use Python's garbage collector for the KV
   pool, given you're already in Python.
4. Sequence A shares a partially-filled block with B and generates its next token.
   What has to happen, when exactly does it have to happen, and what breaks if you
   defer it by one token?
5. You must choose between a bug that leaks blocks and a bug that double-frees
   them. Which do you ship, and what does that tell you about how to write the
   failure paths?

---

## Answers

**1.** `free([7, 2, 9, 4, ...A's own blocks])` decrements each: 7, 2, 9, 4 go from
2 to 1 and are **not** returned to the free list, because they're still B's;
A's exclusive blocks go from 1 to 0 and are appended. The M3 bug that breaks this
is `self.ref_counter[block_id] = 0` instead of `-= 1` — it drops the shared blocks
straight to zero and pushes them onto the free list while B is still reading them.
The next admitted sequence gets block 7, writes its prompt's KV there, and B's
attention now reads a stranger's keys and values for its first 16 positions. No
exception, no NaN, just B producing fluent text that ignores its own system
prompt. This is why the decrement is worth writing correctly in M3 even though M3
can never exercise it.

**2.** Leaks live in failure paths, not the happy path. Look at every place that
allocates and then does something that can raise before an owner is recorded:
admission that reserves `ceil(prompt_len/16)` blocks and then rejects the request
(queue full, invalid input, tokenizer error); prefill that raises partway; a
preemption path that drops a sequence's block table without freeing it; and any
`allocate()` that mutated before it validated. Thirty an hour is a low-frequency
error path, so it's likely an exception case, not the mainline. The invariant that
catches it immediately is
`num_free + sum(1 for c in ref_counter if c > 0) == num_blocks`, asserted every N
steps — and when it fails, the set difference between "blocks with count > 0" and
"blocks named in some live block table" gives you the leaked ids directly, which
usually cluster around one code path.

**3.** Two disqualifying properties. First, non-determinism in *time*: the GC frees
when it decides to, so a block stays held for an unbounded interval after its last
use. The entire point of paging the KV cache is running the pool at 98%
utilization, and you cannot run near capacity when reclamation is "eventually" —
admission would fail while blocks sit collectable-but-uncollected. Second, pauses:
p99 latency is one of the four reported metrics, and a collection pause lands
directly in it. There's also a mechanical reason — the pool is one big tensor, so
the GC has nothing to collect anyway; individual blocks aren't Python objects,
they're index ranges. The right framing is that the pool is *manually* managed
memory living inside a garbage-collected host program, which is exactly how
systems Python is normally structured.

**4.** Copy-on-write, and it must happen *before* the write, not after. A shares
block 4 with B and block 4 is partially filled — say 10 of 16 slots used by the
common prefix. A's next token wants slot 10 of block 4, which is physically inside
a block B also owns. So: allocate a fresh block, copy block 4's contents into it,
decrement block 4's ref count to 1, patch `A.block_table[3]` to the new id, then
write. Deferring by one token means A writes its token into B's block, so B's
position 10 — which B believes is unwritten padding, or which B will itself write
next step — now holds A's KV. Whichever of them reads it second gets the other's
data. Only the last shared block can ever need this, because the earlier ones are
full, which is what makes CoW cheap: at most one copy per divergence.

**5.** Ship the leak. A leak is bounded in badness and *discoverable*: the pool
drains, admission slows, an invariant check or a `num_free` gauge shows it, and no
output is ever wrong. A double free produces silently incorrect generation that
looks like a model quality problem, doesn't reproduce single-sequence, and can run
in production for weeks. That asymmetry should shape how you write failure paths:
when you're not certain an error path still owns a block, *don't* free it. Prefer
"we might have leaked it" over "we might have handed it to someone else." It also
argues for the periodic invariant check, which converts the leak from a mystery
into a pointer at the responsible code path — you can afford that trade only
because the leak is the safe failure.

---

## Rules of Thumb

> Decrement, never assign zero. The day counts exceed 1 is the day that shortcut
> becomes silent corruption, and you won't be looking at this file then.

> Ref counting for pooled resources with deterministic lifetimes; tracing GC only
> where reclamation latency doesn't matter. A pool at 98% utilization is the
> definition of "it matters."

> Bias failure paths toward leaking. A leak is discoverable and harmless to
> output; an over-decrement is silent and corrupts it.

> Assert `num_free + live_count == num_blocks` periodically. Three lines, and it
> catches both directions of the bug.

---

## TL;DR

- Ref counting fits because reclamation must be deterministic (98% utilization
  leaves no room for "eventually") and because blocks can't form cycles — the one
  real weakness doesn't apply.
- `-= 1`, not `= 0`. Indistinguishable in M3, corrupting in M6.
- Leaks are recoverable and visible; over-frees are silent and produce plausible
  wrong text. Write failure paths accordingly.
