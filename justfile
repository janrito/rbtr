check: lint typecheck test

fmt:
    uv run ruff check --fix .
    uv run ruff format .

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run mypy

test:
    uv run pytest

build:
    uv build

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
