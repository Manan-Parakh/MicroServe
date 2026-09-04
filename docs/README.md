# docs/

| Path | What it is |
|---|---|
| `roadmap.md` | The master checkboxed plan — what gets built and read, in what order. Start here. |
| `progress.md` | Dated log of non-trivial changes and decisions. |
| `concepts.md` | Terse glossary of systems/inference concepts, added as they enter the code. |
| `performance-model.md` | What actually moves throughput / TTFT / TPOT / concurrency, and the mechanism under each. Read alongside `benchmarks/`. |
| `bites/` | Deep-dive companion to `concepts.md` — one placement-ready write-up per systems/GPU/inference concept, written via `/add-bites`. |
| `adr/` | Architecture Decision Records. `template.md` + numbered `NNNN-*.md`, never overwritten — superseded ones stay, marked as such. |
| `plans/` | Per-milestone implementation plans (`mNN-<name>-plan.md`), written just before that milestone starts. `template.md` is the skeleton to copy. |
| `benchmarks/` | Raw benchmark numbers, one folder/file per run, each tagged with commit + hardware. |
| `handouts/` | Design write-ups produced while building a milestone — the argument for *why* a mechanism is the point, kept even once superseded. |
| `snippets/` | Small worked examples and scratch explanations (tensor-op walkthroughs, attention sketches) referenced from plans and concepts. |
| `achievements/` | Long-form session write-ups worth keeping whole. |
| `reference/` | The two source planning documents this project's roadmap was distilled from — kept for provenance, not meant to be re-read routinely. |

See `claude.md` at the repo root for the *how we work* rules (decision
protocol, locked decisions, milestone guardrails) — this folder is the
*what and when*, not the *why*.
