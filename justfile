check: schema-check lint typecheck test

ci: schema-check lint typecheck test-cov

fmt: fmt-py fmt-ts fmt-sql fmt-md

lint: lint-py lint-ts lint-sql lint-md

fmt-py:
    uv run ruff check --fix .
    uv run ruff format .

lint-py:
    uv run ruff check .
    uv run ruff format --check .

fmt-sql:
    uv run sqlfluff fix packages/rbtr-legacy

lint-sql:
    uv run sqlfluff lint packages/rbtr-legacy

fmt-md *FILES:
    uv run rumdl check --fix {{ if FILES == "" { "." } else { FILES } }}

lint-md *FILES:
    uv run rumdl check {{ if FILES == "" { "." } else { FILES } }}

fmt-ts:
    bunx @biomejs/biome check --fix packages/pi-rbtr

lint-ts:
    bunx @biomejs/biome check packages/pi-rbtr

typecheck: typecheck-py typecheck-ts

typecheck-py:
    uv run mypy

typecheck-ts:
    cd packages/pi-rbtr && bunx tsc --noEmit

# Regenerate the pi-rbtr TypeScript protocol types from the
# Python models (via `rbtr schema-dump`).  The generated file is
# committed, so CI (and local `just check`) runs this before
# `git diff --exit-code` fails on any drift.
schema-check:
    cd packages/pi-rbtr && bun run scripts/gen-types.ts
    git diff --exit-code packages/pi-rbtr/extensions/rbtr/generated/protocol.ts

test: test-rbtr test-legacy test-ts

test-legacy:
    uv run pytest packages/rbtr-legacy/src/tests

test-rbtr:
    uv run pytest packages/rbtr/src/tests

test-ts:
    cd packages/pi-rbtr && bunx vitest run

test-cov:
    uv run pytest packages/rbtr-legacy/src/tests --cov --cov-report=term --cov-report=markdown-append:cov-append.md

build:
    uv build --package rbtr-legacy

# ── dead code detection (requires `uv sync --group debug`) ──

dead-code:
    uv run --group debug vulture

# ── profiling (requires `uv sync --group debug`) ──
# Benchmark indexing + query latency (no embedding).

# Usage: just bench [repo-path] [base-ref] [head-ref]
bench *ARGS:
    uv run packages/rbtr/scripts/bench_index.py {{ ARGS }}

# Mine real search queries from session history and replay them.
# Usage: just bench-search [path/to/sessions.db]
bench-search *ARGS:
    uv run packages/rbtr/scripts/bench_search.py {{ ARGS }}

# Measure the contribution of docstrings to code search quality.
# Clones four repos (rbtr, django, pi-mono, uv), indexes each
# twice (default and --strip-docstrings), replays docstring-
# derived queries, and writes BENCHMARKS.md.

# Usage: just bench-docstrings [--dry-run] [--cache-dir DIR]
bench-docstrings *ARGS:
    uv run packages/rbtr/scripts/bench_docstrings.py {{ ARGS }}

# Evaluate search quality against curated queries (rbtr repo only).

# Usage: just eval-search [ref]
eval-search *ARGS:
    uv run packages/rbtr/scripts/eval_search.py {{ ARGS }}

# Tune search fusion weights via grid search (rbtr repo only).

# Usage: just tune-search [--step 0.05]
tune-search *ARGS:
    uv run packages/rbtr/scripts/tune_search.py {{ ARGS }}

# Run bench_index.py under scalene (line-level CPU + memory).

# Usage: just bench-scalene [repo-path] [base-ref] [head-ref]
bench-scalene *ARGS:
    uv run --group debug python -m scalene run -o .rbtr/scalene-bench.json packages/rbtr/scripts/bench_index.py {{ ARGS }}

# View a scalene profile in browser (defaults to bench profile).

# Usage: just scalene-view [path-to-json]
scalene-view *ARGS:
    uv run --group debug python -m scalene view {{ ARGS }}

# ── release ──

# Get the current version from pyproject.toml

current_version := `uvx bump-my-version show current_version`
is_dev := if current_version =~ '.*dev.*' { 'true' } else { 'false' }



# Bump version, branch, commit, tag, and push.
# The branch name determines which workflow runs (release-* or pre-release-*).

[group('release')]
bump-pre-release *FLAGS:
    #!/usr/bin/env sh
    set -x -e
    if [ '{{ is_dev }}' = 'true' ]; then
      # 2026.2.1-dev0 -> 2026.2.1-dev1
      uvx bump-my-version bump pre_release {{ FLAGS }}
    else
      # 2026.2.0 -> 2026.2.1-dev0
      uvx bump-my-version bump num {{ FLAGS }}
    fi

[group('release')]
bump-release *FLAGS:
    #!/usr/bin/env sh
    set -x -e
    # Can we get a stable version by bumping just the pre_label?
    # This will fail when pre_label bumping cannot be done (already stable),
    # and will include "dev" if it is bumping the calver section.
    NEXT=$(uvx bump-my-version show --increment pre_label new_version 2>/dev/null || echo 'invalid')
    if echo "$NEXT" | grep -qvE 'dev|invalid'; then
      # 2026.2.1-dev0 -> 2026.2.1
      uvx bump-my-version bump pre_label {{ FLAGS }}
    else
      # 2026.2.0 -> 2026.2.1
      # or 2026.2.0-dev0 -> 2026.2.1
      uvx bump-my-version bump num {{ FLAGS }}
      uvx bump-my-version bump pre_label --allow-dirty {{ FLAGS }}
    fi

[private]
_branch-and-push prefix:
    #!/usr/bin/env sh
    set -x -e
    # Re-read version from disk — just variables are evaluated at
    # load time, before the bump modifies pyproject.toml.
    NEW_VERSION=$(uvx bump-my-version show current_version)
    git checkout -b "{{ prefix }}-v${NEW_VERSION}"
    git add -A
    git commit -m "v${NEW_VERSION}"
    git tag "v${NEW_VERSION}"
    git push -u origin HEAD
    git push origin tag "v${NEW_VERSION}"

[group('release')]
pre-release *FLAGS: (bump-pre-release FLAGS) (_branch-and-push "pre-release")

[group('release')]
release *FLAGS: (bump-release FLAGS) (_branch-and-push "release")
