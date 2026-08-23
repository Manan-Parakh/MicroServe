# Achievement — M2 Phase 2: building `Scheduler` yourself

**Date:** 2026-08-23 · **Milestone:** M2, continuous batching · **Component:**
`engine/scheduler.py`

This isn't a design doc (see `docs/plans/m02-continuous-batching-plan.md` for
that) — it's a record of a specific learning session: you wrote every line of
`Scheduler` yourself, in a Socratic pair-programming session with a fresh
Claude conversation, after saying you were rusty on Python. This file exists
so the mistakes and the reasoning behind the fixes stay legible later, not
just the final code.

---

## What got built

`engine/scheduler.py` — `Scheduler`, owning the WAITING/RUNNING/FINISHED
bookkeeping and the resync-vs-decode decision from ADR-0007. `add_sequence`,
`_admit`, `_prefill_or_decode`, `_append_tokens`, `_check_evictions`, and
`step()` — all written by you; `step()` was the one piece assembled at review
time since it's a one-line composition of the other four, not new logic.

`tests/test_scheduler.py` — 12 tests, all against a `FakeModelRunner` (no real
model, no GPU, ~instant) that just counts how many times `prefill_batch` vs.
`decode_step` got called. That's deliberate: `Scheduler` never touches a
tensor directly, so a fake that returns plain Python lists is enough to
verify its logic completely — a concrete demonstration of the layering
(`Engine` → `Scheduler` → `ModelRunner`) actually paying off in testability.

---

## Two things fixed at review time that your own log hadn't caught

1. **`self.model_runner = ModelRunner`** (the class, not the `model_runner`
   parameter) had crept back into the live file, even though your own handoff
   doc's "done and correct" snippet showed the right version
   (`self.model_runner = model_runner`). Same bug family as the ones below —
   worth noticing it recurred even after you'd already found and fixed it
   three times elsewhere.
2. **A real crash case your test plan didn't cover**: if `running` and
   `waiting` are both empty but `membership_changed` is still `True` from the
   eviction that just emptied `running`, `_prefill_or_decode()` would call
   `max([...])` over an empty list. This isn't an edge case — it's the normal
   state the scheduler is in the instant after the last sequence in a batch
   finishes with nothing queued behind it. Fixed with an explicit "nothing to
   do" guard at the top of `step()`, before `_prefill_or_decode()` is ever
   reached.

Everything else in your own bug log below was found and fixed by you, before
review — this file is documenting it, not crediting me for it.

---

## Design decisions you made — reviewed, all held up

Re-checked every entry in your own decisions table against the actual
requirements. All of them are sound; nothing needed walking back:

- `cache is None` doubling as both "never built" and "invalidated" — clean,
  no separate flag needed for that case.
- `Scheduler.add_sequence()` force-setting `status = WAITING` rather than
  trusting the caller's `Sequence` construction — defensive ownership of the
  state machine, the right instinct.
- `waiting` as a `deque`, mutated only through `add_sequence()` — the
  encapsulation point that stops a caller from breaking FIFO order.
- `_admit()` computing and returning its own `budget` rather than taking it
  as a parameter or having the caller diff `len(running)` before/after — the
  three-iteration path you took to get here (return `None` + diff → pass
  budget in → compute-and-return-directly) is worth remembering: the
  "diff before/after" option *looked* more decoupled but actually made the
  caller infer a fact from a side effect, which is more coupling, not less.
- **The `max_cache_len` algebra** — you derived `len(prompt_token_ids) +
  max_new_tokens` by expanding the handoff's literal formula
  (`current_history_length + (max_new_tokens - tokens_generated_so_far)`)
  and noticing `output_token_ids`' length cancels out. I re-derived it
  independently and it's correct: a per-sequence ceiling that's constant for
  the sequence's whole lifetime, which matches "worst-case cache size" not
  needing to shrink or grow as generation proceeds. This is the KV-cache-size
  reading from the plan doc, made concrete in your own code rather than left
  as a formula on paper.
- Building `token_id_lists` with `seq.prompt_token_ids + seq.output_token_ids`
  (new list) instead of `.append`/`.extend` (mutates in place) — correctly
  identified that `prompt_token_ids` gets re-read on every future resync, so
  mutating it once would corrupt every sequence's history permanently and
  silently.
- `survivors = []` (new list, one pass) instead of mutating `self.running`
  while iterating it for eviction — see the bug log below for what this
  avoided.

---

## Bugs you hit, and the pattern behind each — the material worth re-reading

Nearly every bug fell into one of three shapes. Reusing your own framing
because it's the right one:

### 1. Declaring something without it existing at runtime

- Bare `name: type` class-level annotations (no `@dataclass`, no `=`) —
  populates `__annotations__` only; no actual attribute exists.
- `self.cache = StaticCache | None` — the right-hand side evaluates to a
  *type object* (a union type), not a value. No crash; `self.cache is None`
  would just be `False` forever.
- `self.membership_changed = bool` — same shape, the class instead of `True`/
  `False`.
- A function with a return type hint (`-> list[int]`) and a body that never
  actually hits a `return` statement — always returns `None` regardless of
  which branch ran.
- **The habit this leaves behind:** after writing a line that "sets" or
  "declares" something, ask *"if I ran this right now and inspected the
  object, would this actually be there?"*

### 2. A value that's the wrong *kind* of thing, without crashing

- `seq.status = "WAITING"` (a string) instead of `SequenceStatus.WAITING` (an
  Enum member) — `"WAITING" == SequenceStatus.WAITING` is just `False`,
  always, silently. Recurred later as a commented-out EOS check comparing a
  token id to the literal string `"end_token_we_dont_know_exactly"`.
- `self.running = list[Sequence]` (the generic alias, a type) instead of `[]`
  or `list()` (an actual empty list). `list[]` isn't even valid syntax.
- `self.waiting.pop(0)` — `deque.pop()` takes no index and removes from the
  *back*; `.pop(0)` on a `deque` doesn't mean what it means on a `list`.
  Needed `popleft()` for FIFO.
- `budget -= budget` inside the admission loop — sets the counter to zero on
  the very first iteration regardless of its starting value, so the loop
  always ran exactly once no matter how large `budget` was. Traced by hand
  with `budget = 3`: one iteration, then `3 - 3 = 0`. Needed `budget -= 1` —
  and once `_admit()` also had to *return* the original count, the loop
  counter had to be a separate variable (`n`) from the value being returned.
- `if self.cache == None:` instead of `is None` — `is` is the semantically
  correct identity check for a singleton; avoids any theoretical custom
  `__eq__` surprise.
- **The habit this leaves behind:** when something "should" be failing but
  isn't, suspect a silent type mismatch before assuming the logic itself is
  wrong.

### 3. Mutating something whose value depends on staying unchanged

- `seq.prompt_token_ids.append(seq.output_token_ids)` inside a comprehension
  meant to build a *new* list — two stacked problems: `.append()` returns
  `None` (so the comprehension collects `None`s, not token lists), and it
  **permanently mutates `prompt_token_ids` in place**. Since this runs on
  every resync, a second resync would append onto an already-corrupted list —
  nothing crashes, the corruption just compounds silently.
- The classic mutate-while-iterating footgun during eviction:
  ```python
  survivors = self.running        # same list object, not a copy
  for seq in self.running:
      if <finished>:
          survivors.pop(seq)      # wrong method too — .pop() takes an index
  ```
  Verified by hand with a 4-element list that removing an item mid-iteration
  causes the *next* item to slide into the just-vacated index and get
  silently skipped by the loop's position counter — not a crash, a silent
  skip. Fixed by building a brand-new `survivors` list in a single pass and
  reassigning `self.running` at the end, never mutating the list being
  iterated.
- A stray `.seq.` chained attribute lookup (`seq.seq.prompt_token_ids`) plus
  an assignment statement placed where a list comprehension needed an
  expression — not valid syntax, but the root cause (typing an assignment
  where a value-producing expression belongs) is the same shape as the
  `.append()`-returns-`None` bug above: reaching for a statement when an
  expression is what the position actually needs.
- **The habit this leaves behind:** before calling any mutating method
  (`.append`, `.pop`, `.remove`), ask *"does anything else still depend on
  this object being exactly what it was a moment ago?"*

---

## Python/tooling concepts revised this session

- `@dataclass` vs. bare class annotations — annotations alone are metadata,
  not storage; `@dataclass` is what turns them into `__init__` parameters and
  real attributes.
- Generic aliases (`list[Sequence]`) as *type hints only* — meaningless (or a
  syntax error) as a runtime value.
- `deque` vs. `list` semantics for queue operations — `.popleft()` vs. `.pop()`
  vs. `.pop(0)`, and why a `deque` was chosen over a `list` for `waiting` in
  the first place (O(1) pops from the front instead of O(n)).
- `is None` vs. `== None` for identity checks on singletons.
- List comprehensions require an *expression* per element, not a statement —
  assignments don't belong inside `[... for ...]`.
- Mutating vs. non-mutating list operations (`.append`/`.extend` vs. `+`) and
  aliasing (`x = y` binds two names to the *same* object; it is not a copy).
- The mutate-while-iterating hazard — verified concretely by hand-tracing a
  small example rather than taking it on faith.
- Enum members vs. string literals that merely look like the member's name.

---

## Open items carried into the next session

- **EOS stop condition** — was an explicit documented gap in the handoff
  (`# TODO: no EOS id available anywhere in this codebase yet`). Resolved at
  review time: `self.model_runner.tokenizer.eos_token_id` was already
  available (the same source `ModelRunner.generate()`'s `pad_token_id` and
  `_left_pad`'s padding use) — `_check_evictions()` now checks it alongside
  the `max_new_tokens` cap. Not a new design decision, just plumbing a value
  that already existed elsewhere in the codebase through to where it was
  needed.
- **`step()` itself** was written at review time, not by you — it's a
  four-line composition (`if idle: return []` / `_append_tokens()` /
  `return _check_evictions()`) with no new logic beyond the empty-scheduler
  guard from item 2 above, so it didn't warrant a separate build session.
  Worth reading it once to confirm it matches how you were already picturing
  the four helper methods fitting together.
- Manual, non-fake correctness testing (real model, real tokens) for the
  scheduler happens in Phase 3, once `Scheduler` is wired into an actual
  `Engine` — this session's tests intentionally stay at the bookkeeping
  level, per the plan doc's own scope for Phase 2.

---

## Questions for you before Phase 3

- The eviction/resync interaction test
  (`test_eviction_forces_resync_for_remaining_sequences`) is the one that
  most directly exercises ADR-0007's reason for existing — worth walking
  through it once by hand if the `prefill_calls`/`decode_calls` counts
  weren't immediately obvious when you read the test.
- Ready to move to Phase 3 (wiring `Scheduler` into a real continuous-batch
  `Engine`, plus the static-batch comparison engine), or want to spend more
  time first confirming you can explain every line of `scheduler.py` back
  without notes?
