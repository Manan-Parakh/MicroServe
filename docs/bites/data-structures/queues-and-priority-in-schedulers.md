# Queues and Priority in Schedulers

> A scheduler's data structure is its policy made concrete — if arrival order is
> your policy, a deque *is* your scheduler.

---

## The Problem

MicroServe's scheduler answers one question every iteration: which sequences run
in this batch? Under FCFS that's "the oldest waiting ones, up to the cap." It's
tempting to reach for a priority queue immediately — schedulers have priorities,
priorities mean heaps — but under FCFS the queue is *already sorted by
construction*, because things arrive in arrival order. A heap would maintain an
ordering that append-order gives you free, at O(log n) per operation, forever.

The reverse mistake is worse and more common: a linear scan. `min(waiting,
key=arrival_time)` is O(n) per step and looks harmless at n = 8. When M3 Phase 4
adds preemption and the waiting queue starts holding preempted sequences too, n
grows and the scan runs on every single decode iteration — one of the two hot
loops in the entire engine.

The structure question is really the policy question wearing a disguise. Decide
the policy, and the structure falls out.

---

## Mental Model

```
policy                     structure           cost per step
-----------------------------------------------------------
FCFS / arrival order    -> deque              O(1)   (already sorted)
LIFO / newest first     -> stack              O(1)
by a score              -> heap               O(log n)
by a score + updates    -> indexed heap       O(log n) + O(1) locate
round-robin, fixed set  -> ring buffer        O(1), zero allocation
"all of them, filtered" -> list, rebuilt      O(n), fine when n is capped
```

The heap earns its keep only when the ordering is **not** the insertion order and
**not** cheap to recompute.

> 🔹 **heap** — a tree kept in an array that maintains only the *minimum* at the
> root, not a full sort. That partial guarantee is exactly what "who's next?"
> needs, and it's why it beats sorting.

---

## How It Works — by policy

### FCFS → `deque`

MicroServe's `Scheduler.waiting` is `deque[Sequence]`: `append` on arrival,
`popleft` on admission. O(1) both ends, no comparisons, no keys, no tie-breaking
rules to get wrong.

- **Buys**: simplest possible; naturally fair; no starvation by construction.
- **Costs**: no priority, so one enormous prompt at the head blocks a hundred
  short ones behind it — **head-of-line blocking**, the defining weakness of FCFS.
- **Used by**: vLLM's default scheduler is FCFS. This is not a toy choice.

> 🔹 **head-of-line blocking** — one item at the front of a queue delaying
> everything behind it, even when the rest could have proceeded.

### Shortest-job-first / priority → binary heap (`heapq`)

Order by predicted output length, deadline, or a fairness score.

- **Buys**: dramatically better mean latency when job sizes are skewed — and
  inference job sizes are extremely skewed.
- **Costs**: **starvation** — a long sequence can be perpetually outranked. Needs
  an aging term. And SJF needs a length *prediction*, which for LLM generation is
  genuinely hard (you don't know when EOS lands). Predicting badly is worse than
  FCFS.
- **Used by**: research schedulers and some production systems for TTFT
  optimisation; also the standard OSTEP Ch 7–8 material.

### Priority with changing keys → indexed heap / heap + lazy deletion

If a sequence's priority changes while it waits (aging, a preemption penalty), a
plain heap can't find it to update.

- **Buys**: correct dynamic priorities.
- **Costs**: either an id→heap-position map maintained on every sift (fiddly), or
  *lazy deletion* — push the new entry, mark the old stale, skip stale entries on
  pop. Lazy deletion is far simpler and usually right; the heap just grows
  somewhat larger than the live set.

### Round-robin over a fixed set → ring buffer

Fixed capacity, no per-item allocation, wraps with a mask when the size is a power
of two.

- **Buys**: zero allocation, cache-friendly, trivially bounded.
- **Costs**: capacity is fixed at construction; overflow is a policy decision you
  must make (block, drop, or grow).
- **Used by**: NIC descriptor rings, audio buffers, GPU command queues, lock-free
  SPSC queues. It's the structure you name when someone says "bounded queue."

### "The whole running set, filtered" → plain list, rebuilt

MicroServe's `running` is a `list[Sequence]` rebuilt each step as the survivors of
an eviction check.

- **Buys**: contiguous, deterministic iteration order — which *matters*, because
  the batch's row order must match the KV cache's row order.
- **Costs**: O(n) per step. Irrelevant at `max_batch_size = 64`; reconsider in the
  thousands.

---

## Usage

```python
from collections import deque
import heapq

# FCFS: the queue is sorted by construction
waiting: deque[Sequence] = deque()
waiting.append(seq)                      # arrival
seq = waiting.popleft()                  # admission

# priority: only if the order is NOT arrival order
heapq.heappush(pq, (-priority, seq_id, seq))   # seq_id breaks ties
                                               # (Sequence isn't comparable)
_, _, seq = heapq.heappop(pq)
```

That `seq_id` tie-breaker is not cosmetic: with equal priorities Python falls
through to comparing the third element and raises `TypeError` on a dataclass with
no `__lt__`. It's the single most common bug when someone first swaps a deque for
a heap.

---

## Where This Fits in MicroServe

`engine/scheduler.py`, Milestone 2 — `waiting: deque` (FIFO) and `running: list`
(rebuilt). Iteration-level scheduling means this runs every forward pass, not
once per request, which is why the constant factors are worth a thought at all.

Milestone 3 Phase 4 is where it gets interesting: block exhaustion introduces
**preemption**, and a preempted sequence has to go somewhere. If it goes back to
the head of `waiting`, you need a deque (`appendleft`) and you've implicitly
chosen "preempted sequences resume first." If it goes to the tail, you've chosen
fairness at the cost of already-spent prefill work. That's a policy decision the
plan flags as ADR-worthy, and the structure follows from it — not the reverse.

The OS equivalent is OSTEP Ch 7–8 (FIFO, SJF, STCF, round-robin) and Ch 9 (MLFQ,
which is literally an array of queues plus an aging rule). MLFQ is the canonical
answer to "FCFS starves short jobs, SJF starves long ones."

---

## Gotchas & Trade-offs

- **A heap where a deque would do is a real cost and a real bug surface.**
  O(log n) per push/pop, plus tie-break comparators, plus non-deterministic order
  among equal keys — which breaks reproducibility in a benchmark.
- **Non-deterministic iteration order breaks correctness, not just style.** The
  row order of the batch must line up with the KV cache's rows. Anything that
  reorders `running` silently corrupts output. This is why it's a list, not a set
  or dict.
- **Starvation is invisible in p50.** SJF looks fantastic at p50 and catastrophic
  at p99, because the starved long requests are exactly the tail. Always report
  both — `claude.md`'s "p50 and p99, never a bare mean" exists for this.
- **Preemption interacts with the queue choice.** Repeatedly preempting and
  re-admitting the same sequence is *thrashing*: it makes no progress while
  burning full prefill recomputation each round. A per-sequence block cap at
  admission, or a "cannot be preempted twice in a row" rule, is the guard.
- **Aging is not optional in a priority scheduler.** Without it, SJF under
  sustained load will starve the longest request indefinitely, and it will be the
  one a customer is watching.
- **Don't build a timer wheel here.** It's the right structure for millions of
  timeouts; at MicroServe's scale a heap or a sorted check is fine. Know the name.

---

## Comparison Table

| Policy | Structure | Best case | Worst case |
|---|---|---|---|
| FCFS | `deque` | fair, O(1), no tuning | head-of-line blocking |
| LIFO | stack | great for bursty warm work | brutal unfairness |
| SJF / SRTF | heap | best mean latency on skewed sizes | starvation; needs length prediction |
| Priority + aging | heap (lazy delete) | tunable fairness | two knobs to tune, easy to get wrong |
| Round-robin | ring buffer | bounded, zero-alloc | fixed capacity, overflow policy needed |
| MLFQ | array of deques | adapts without prediction | most complex; boost interval to tune |

---

## Interview & Career Angle

"Design the request scheduler for an inference server" is a top-tier ML-systems
system-design question, and the expected answer is *not* a clever heap — it's
FCFS with a clear statement of why, plus the specific conditions that would make
you upgrade. Interviewers are checking whether you know that continuous batching
already extracts most of the win, that job-size prediction is the blocker for SJF
in LLM serving, and that preemption policy (recompute vs. swap) is a separate
axis from queue order.

Adjacent concepts worth learning next:

- **Iteration-level (continuous) batching** — the scheduling idea that makes the
  queue matter every forward pass rather than every request.
- **MLFQ** — how real schedulers get SJF-like behaviour without predicting job
  length.
- **Preemption: recompute vs. swap** — the M3 Phase 4 decision; what you do when
  the pool is full mid-generation.
- **Little's Law** — relates queue depth, arrival rate, and latency; the fastest
  way to sanity-check a scheduler's behaviour under load.

---

## Hard Questions I Should Be Able to Answer

1. Under FCFS, one 8,000-token prompt arrives ahead of fifty 40-token prompts.
   Describe what happens to TTFT p50 and p99, and give two fixes that don't
   require predicting output length.
2. You swap `waiting` from a deque to a heap keyed by prompt length. Throughput
   goes up, and a customer complains. What did you break, and how would you have
   caught it before shipping?
3. M3 Phase 4: the pool is full and a running sequence needs one more block. Walk
   through the decision — who gets preempted, where do they go in the queue, and
   what stops this from thrashing?
4. An interviewer asks: "why does vLLM default to FCFS when SJF is provably
   better for mean latency?"
5. Your scheduler's queue depth grows without bound at a request rate the engine
   should handle. Where do you look?

---

## Answers

**1.** TTFT p50 barely moves for the fifty short prompts if — and only if —
continuous batching admits them alongside the long one; iteration-level
scheduling means they don't wait for the long prompt to *finish*, just for it to
be prefilled. p99 takes the hit, driven by the long prompt's own prefill, which is
a single large forward pass that blocks the step it runs in. Two fixes without
length prediction: **chunked prefill**, which splits the 8,000-token prefill
across several iterations so decode work interleaves and the short prompts don't
stall behind one giant forward pass; and an **admission cap on prefill tokens per
step**, which is the cruder version of the same idea. Both attack the fact that
the long *prompt* is known at arrival, unlike output length, so you can act on it
without predicting anything.

**2.** Starvation, almost certainly of the customer with the longest prompt.
Throughput rose because you're now packing short, cheap sequences preferentially,
which raises tokens/sec system-wide while some individual request sits at the back
of the heap forever. You'd have caught it by reporting p99 TTFT and *max* wait
time, not throughput and p50 — the metric set in `claude.md` exists precisely
because throughput alone rewards this failure. The fix is an aging term: priority
decays with wait time so anything eventually reaches the front, which is MLFQ's
periodic boost in miniature.

**3.** First check whether preemption is even the right response: if the sequence
needing a block is the *newest* admitted one, the cheaper fix is to not have
admitted it — a per-sequence block reservation at admission (prompt blocks plus
some headroom) makes mid-flight exhaustion rare. When you must preempt, the victim
should be the sequence whose work is cheapest to redo, which under
preempt-by-recomputation means the one with the *fewest generated tokens* — you're
throwing away its KV and will re-prefill it later. It goes back to the head of
`waiting` (`appendleft`), not the tail, because it already waited once and
tail-insertion makes repeated preemption unboundedly unfair. The anti-thrash guard
is a rule that a just-preempted sequence cannot be preempted again immediately on
re-admission, or equivalently a minimum-progress quantum: it must generate at
least k tokens before becoming eligible as a victim. Without that, under sustained
pressure you can preempt and re-admit the same sequence forever while it makes
zero net progress and burns full prefill cost each round.

**4.** Because the premise is shakier than it sounds. SJF is provably optimal for
mean completion time *when you know the job sizes*, and in LLM serving you don't —
output length is unknown until EOS, and prompt length is a poor proxy for it. A
mispredicting SJF is worse than FCFS, not better. Second, continuous batching has
already captured most of the available win: sequences don't wait for a batch to
drain, so queue order matters much less than it does in a static-batching system.
Third, FCFS is starvation-free by construction and has zero tuning knobs, which
matters enormously for a system other people operate. The honest summary is that
SJF's theoretical advantage is real but small here, and it's paid for with a
prediction you can't make and a fairness failure you can't bound.

**5.** Queue depth growing without bound means arrival rate exceeds service rate —
Little's Law says the queue is the integral of that gap, so the question is which
side moved. Check service rate first: is the batch actually full? If `running` is
below `max_batch_size` while `waiting` is deep, admission is being blocked by
something — most likely block exhaustion, so log `num_free` next to every
scheduling decision. If the batch *is* full and throughput is at its usual level,
the arrival rate genuinely exceeds capacity and the answer is backpressure or more
hardware, not a scheduler change. If the batch is full and throughput has *fallen*,
suspect preemption thrashing or a cache-invalidation path being hit every step
(in MicroServe, a membership change forcing a resync-prefill every iteration would
do it) — both burn work without producing tokens.

---

## Rules of Thumb

> If arrival order is your policy, the deque already sorted your queue. Don't pay
> a heap for an ordering you got free.

> Report p99 and max wait whenever you change queue policy. Throughput alone
> rewards starvation.

> Decide the preemption policy before the queue structure — the structure is the
> policy's shadow, not the other way round.

> Any priority scheme without an aging term will starve someone. Add it when you
> add the heap, not after the complaint.

---

## TL;DR

- FCFS + continuous batching captures most of the win; a heap buys less than it
  looks like and costs starvation plus a prediction you can't make.
- `deque` for arrival order, heap only when the order isn't insertion order, ring
  buffer for bounded fixed-capacity handoff, plain list when n is capped and
  iteration order is load-bearing.
- The interesting scheduler question in a paged engine isn't queue order — it's
  what happens when the pool is full.
