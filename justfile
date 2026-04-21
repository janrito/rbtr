check: schema-check lint typecheck test test-ts

ci: schema-check lint typecheck test-cov test-ts

fmt: fmt-py fmt-ts fmt-sql fmt-md

lint: lint-py lint-ts lint-sql lint-md

fmt-py:
    uv run ruff check --fix
    uv run ruff format

lint-py:
    uv run ruff check
    uv run ruff format --check

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

test:
    uv run pytest

test-ts:
    cd packages/pi-rbtr && bunx vitest run

test-cov:
    uv run pytest --cov --cov-report=term --cov-report=markdown-append:cov-append.md

# ── build ──

build: build-rbtr build-ext

build-rbtr:
    uv build --package rbtr --out-dir dist

# Build every pi-* extension under packages/ into dist/ as an npm tarball.
build-ext:
    #!/usr/bin/env sh
    set -e
    mkdir -p dist
    for dir in packages/pi-*; do
      [ -d "$dir" ] || continue
      (cd "$dir" && bunx tsc --noEmit && npm pack --pack-destination ../../dist)
    done

# ── dead code detection (requires `uv sync --group debug`) ──

dead-code:
    uv run --group debug vulture

# Run the rbtr-eval pipeline (clone -> extract -> merge -> measure).
# Run a single stage with `cd packages/rbtr-eval && uv run dvc repro <stage>`.
eval:
    cd packages/rbtr-eval && uv run dvc repro

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
