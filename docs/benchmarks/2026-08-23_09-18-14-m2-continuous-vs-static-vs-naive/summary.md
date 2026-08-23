# M2 benchmark — continuous vs. static vs. naive

**Commit:** ee76afb · **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU ·
**Model:** Qwen/Qwen2.5-0.5B-Instruct · **Workload:** 6 fixed prompts, mixed
`max_new_tokens` per prompt (`[8, 60, 12, 50, 10, 40]`) — see
`bench/benchmark.py`'s `WORKLOAD`. `max_batch_size=3` for continuous/static.
Raw numbers in `results.json` (same folder).

| Engine | Throughput (tok/s) | Wasted-compute fraction | Peak concurrent | Wall time (s) |
|---|---|---|---|---|
| naive | 13.8 | n/a (no concurrency) | 1 | 13.0 |
| continuous | **47.4** | **0.0%** | 3 | 3.8 |
| static | 27.2 | **46.3%** | 3 | 6.6 |

All three engines generated the same 180 total tokens (a correctness check
in itself — total work done doesn't depend on scheduling strategy).

**This is the milestone's Definition of Done, met:** continuous batching
beats static batching on both throughput (47.4 vs. 27.2 tok/s, +74%) and the
wasted-compute metric that stands in for "GPU idle time" here (0% vs. 46.3%
of slot-steps spent recomputing on an already-finished sequence, because a
static batch can't shrink until every member is done — see
`bench/benchmark.py`'s `run_scheduled` docstring for why slot-steps was
chosen over a wall-clock idle estimate).
