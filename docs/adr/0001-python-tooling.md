# ADR-0001: Python dependency and lint tooling — uv + ruff

- **Status**: Accepted
- **Date**: 2026-08-19
- **Milestone**: M0
- **Concepts**: none (pure engineering hygiene, no systems/inference concept)

## Context

`claude.md` Open Decision #4 leaves Python tooling to the owner. M0's Phase 0
needs a dependency manager and a lint/format tool settled before any real
dependency (`torch`, `transformers`) gets added, so every later `uv add`/lock
step is consistent from the first commit onward.

## Options considered

### Option A — uv + ruff
A single Rust-based tool handles venv creation, dependency resolution,
locking, and running scripts; ruff handles lint + format in one binary.
- **Buys**: fast installs/resolution (matters when `torch` alone is a
  multi-GB download iterated on repeatedly), one lockfile format, one tool to
  learn instead of several.
- **Costs**: newer tool, smaller (though fast-growing) ecosystem familiarity
  than pip/poetry.

### Option B — poetry + ruff
Poetry manages dependencies/locking/venvs via its own resolver; ruff for
lint/format.
- **Buys**: mature, widely used, already installed on this machine.
- **Costs**: slower dependency resolution and installs than uv, particularly
  painful for large ML packages like `torch` that get re-resolved often
  during early iteration.

### Option C — pip + requirements.txt + ruff
Plain `pip install -r requirements.txt`, versions pinned by hand.
- **Buys**: simplest, most universally understood, zero new tooling concepts.
- **Costs**: no lockfile — reproducibility depends on remembering to pin
  every transitive version by hand; slowest installs of the three.

## Decision

**uv + ruff.**

## Why the others lose

Poetry loses here specifically because of install/resolution speed on large
packages — this project reinstalls/upgrades `torch` repeatedly across
milestones (M4 adds `bitsandbytes`, dependency changes are frequent), and
uv's speed advantage compounds every time that happens. It's the right choice
under different constraints (e.g. a team already standardized on Poetry's
plugin ecosystem), just not here.

Plain pip loses on reproducibility, not speed — without a lockfile, "works on
my machine" drift becomes a real risk for a project whose whole premise is a
correctness harness comparing outputs across environments (CPU dev machine
vs. GPU benchmark machine). It's the right choice for a tiny,
rarely-touched script, which this project is not.

## Consequences

Every dependency addition from here on goes through `uv add`, and `uv.lock`
is tracked in git (per `.gitignore`'s existing note that lockfiles are never
ignored). Contributors (or future-you on a new machine) run `uv sync` once to
reproduce the exact environment. Nothing about this choice is hard to
reverse — swapping to poetry or pip later would mean regenerating a lockfile
from `pyproject.toml`, not rewriting the project.

## What I learned

*(owner's to fill in, optional)*
