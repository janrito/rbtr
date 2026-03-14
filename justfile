check: lint typecheck test

fmt: fmt-py fmt-sql fmt-md

lint: lint-py lint-sql lint-md

fmt-py:
    uv run ruff check --fix .
    uv run ruff format .

lint-py:
    uv run ruff check .
    uv run ruff format --check .

fmt-sql:
    uv run sqlfluff fix .

lint-sql:
    uv run sqlfluff lint .

fmt-md *FILES:
    uv run rumdl check --fix {{ if FILES == "" { "." } else { FILES } }}

lint-md *FILES:
    uv run rumdl check {{ if FILES == "" { "." } else { FILES } }}

typecheck:
    uv run mypy

test:
    uv run pytest

build:
    uv build

# ── profiling (requires `uv sync --group debug`) ──
# Benchmark indexing + query latency (no embedding).

# Usage: just bench [repo-path] [base-ref] [head-ref]
bench *ARGS:
    uv run scripts/bench_index.py {{ ARGS }}

# Mine real search queries from session history and replay them.
# Usage: just bench-search [path/to/sessions.db]
bench-search *ARGS:
    uv run scripts/bench_search.py {{ ARGS }}

# Evaluate search quality against curated queries (rbtr repo only).

# Usage: just eval-search [ref]
eval-search *ARGS:
    uv run scripts/eval_search.py {{ ARGS }}

# Tune search fusion weights via grid search (rbtr repo only).

# Usage: just tune-search [--step 0.05]
tune-search *ARGS:
    uv run scripts/tune_search.py {{ ARGS }}

# Run bench_index.py under scalene (line-level CPU + memory).

# Usage: just bench-scalene [repo-path] [base-ref] [head-ref]
bench-scalene *ARGS:
    uv run --group debug python -m scalene run -o .rbtr/scalene-bench.json scripts/bench_index.py {{ ARGS }}

# View a scalene profile in browser (defaults to bench profile).

# Usage: just scalene-view [path-to-json]
scalene-view *ARGS:
    uv run --group debug python -m scalene view {{ ARGS }}

# Get the current version from pyproject.toml

current_version := `uvx bump-my-version show current_version`
[private]
_is_current_dev := if current_version =~ '.*dev.*' { 'true' } else { 'false' }

# find if we can bump just the label
# this will fail when pre_label bumping cannot be done (i.e. it's already stable)
# this will succeed but include dev if it is bumping the calver section

[private]
_next_pre_label_version := `uvx bump-my-version show --increment pre_label new_version 2>/dev/null || echo 'invalid'`
[private]
_can_bump_pre_label := if _next_pre_label_version =~ '.*dev.*|invalid' { 'false' } else { 'true' }

[group('release')]
pre-release *FLAGS: && build
    #!/usr/bin/env sh
    set -x -e
    if [ '{{ _is_current_dev }}' = 'true' ]; then
      # 2026.2.1-dev0 -> 2026.2.1-dev1
      uvx bump-my-version bump pre_release {{ FLAGS }}
    else
      # 2026.2.0 -> 2026.2.1-dev0
      uvx bump-my-version bump num {{ FLAGS }}
    fi

[group('release')]
release *FLAGS: && build
    #!/usr/bin/env sh
    set -x -e
    if [ '{{ _can_bump_pre_label }}' = 'true' ]; then
      # 2026.2.1-dev0 -> 2026.2.1
      uvx bump-my-version bump pre_label {{ FLAGS }}
    else
      # 2026.2.0 -> 2026.2.1
      # or 2026.2.0-dev0 -> 2026.2.1
      uvx bump-my-version bump num {{ FLAGS }}
      uvx bump-my-version bump pre_label --allow-dirty {{ FLAGS }}
    fi
