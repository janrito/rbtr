# rbtr

Monorepo for rbtr.

## Packages

| Package                                | Language   | Description                              |
| -------------------------------------- | ---------- | ---------------------------------------- |
| [`rbtr`](packages/rbtr/)               | Python     | Structural code index (CLI + library)    |
| [`pi-rbtr`](packages/pi-rbtr/)         | TypeScript | Pi extension for the code index          |
| [`rbtr-eval`](packages/rbtr-eval/)     | Python     | Benchmark and tuning pipeline (DVC)      |

## Skills

| Skill                                            | Description        |
| ------------------------------------------------ | ------------------ |
| [`review-github-pr`](skills/review-github-pr/)   | PR review workflow |

## Install

```bash
uv tool install rbtr        # the code index CLI
pi install npm:@rbtr/pi     # the pi extension
```

See each package's README for usage:
[`rbtr`](packages/rbtr/) and [`pi-rbtr`](packages/pi-rbtr/).

## Architecture

rbtr is split across three packages and one skill:

- **[`rbtr`](packages/rbtr/)** — the Python CLI and daemon that
  builds and serves the structural index.
- **[`pi-rbtr`](packages/pi-rbtr/)** — a thin TypeScript pi
  extension that talks to the daemon and surfaces its queries
  as agent tools.
- **[`review-github-pr`](skills/review-github-pr/)** — a skill
  that drives GitHub PR reviews via `gh api graphql`.
- **[`rbtr-eval`](packages/rbtr-eval/)** — an internal DVC
  pipeline for search-quality benchmarking; not shipped.

The extension talks to the daemon over ZMQ: a REP socket for
request/response and a PUB socket for progress, ready, and
error notifications. See
[`packages/rbtr/ARCHITECTURE.md`](packages/rbtr/ARCHITECTURE.md)
and
[`packages/pi-rbtr/ARCHITECTURE.md`](packages/pi-rbtr/ARCHITECTURE.md)
for detail.

## Development

```bash
just setup                  # uv sync + bun install
just check                  # schema-check + lint + typecheck + test
just fmt                    # auto-fix (Python, TypeScript, SQL, Markdown)
```

| Recipe              | What it does                                      |
| ------------------- | ------------------------------------------------- |
| `just test`         | pytest (all Python packages)                      |
| `just test-ts`      | vitest (pi-rbtr)                                  |
| `just typecheck`    | mypy + tsc                                        |
| `just lint`         | ruff + biome + sqlfluff + rumdl                   |
| `just schema-check` | regenerate `protocol.ts`, fail on drift           |
| `just eval`         | run the rbtr-eval DVC pipeline                    |
| `just build`        | build Python wheels + npm tarballs                |

| Skill                                            | Description               |
| ------------------------------------------------ | ------------------------- |
| [`rbtr-data`](.agents/skills/rbtr-data/)         | Data handling conventions |
| [`rbtr-testing`](.agents/skills/rbtr-testing/)   | Testing conventions       |
