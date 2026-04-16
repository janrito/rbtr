# pi-rbtr

A [pi] extension package that gives the LLM access to
rbtr's structural code index. The agent can search by name,
keyword, or concept, read symbol source, list file
structure, trace dependency edges, and compare structural
changes between git refs — without constructing shell
commands or parsing raw output.

[pi]: https://github.com/badlogic/pi-mono

## Install

The extension requires the `rbtr` CLI. Install both:

```bash
# Install rbtr (the code index)
uv tool install rbtr

# Install the pi extension (project-local)
pi install -l ./packages/pi-rbtr
```

For development from a local clone (no global install):

```bash
# In .pi/rbtr-index.json
{ "command": "uvx --from ./packages/rbtr" }
```

## What the agent gets

Seven tools, registered automatically on session start:

| Tool                   | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `rbtr_search`          | Search by name, keyword, or concept (BM25 + semantic + name fusion) |
| `rbtr_read_symbol`     | Read a symbol's full source by name                                 |
| `rbtr_list_symbols`    | Structural table of contents for a file                             |
| `rbtr_find_refs`       | Find references via the dependency graph (imports, tests, docs)     |
| `rbtr_changed_symbols` | Symbols that changed between two git refs                           |
| `rbtr_index`           | Index the repository (background, incremental)                      |
| `rbtr_status`          | Check whether the index exists and how many symbols it contains     |

The extension also injects a system prompt note so the agent
knows the index is available without being told.

### When to use which tool

- **Concept query** ("how does authentication work") →
  `rbtr_search`. More precise than grep for semantic queries.
- **Known symbol** ("read the source of `fuse_scores`") →
  `rbtr_read_symbol`. Faster than finding the file and reading it.
- **File structure** ("what's in `config.py`?") →
  `rbtr_list_symbols`. One-line-per-symbol TOC with line ranges.
- **Exact string match** ("find all `TODO` comments") →
  `grep`. The index is structural, not textual.
- **Who calls X?** → `rbtr_find_refs`. Follows import, test,
  and doc edges in the dependency graph.
- **What changed?** → `rbtr_changed_symbols`. Function-level
  diff between two refs, not line-level.

## Commands

Three user-facing commands (no LLM involved):

| Command          | Description                           |
| ---------------- | ------------------------------------- |
| `/rbtr-status`   | Show index status (chunk count, path) |
| `/rbtr-index`    | Trigger a background indexing         |
| `/rbtr-settings` | View and toggle extension settings    |

## Configuration

Settings are read from JSON config files. Project-local
overrides global:

| File                          | Scope         |
| ----------------------------- | ------------- |
| `~/.pi/agent/rbtr-index.json` | Global        |
| `.pi/rbtr-index.json`         | Project-local |

```json
{
  "command": "rbtr",
  "autoIndex": true
}
```

| Key         | Default  | Description                                      |
| ----------- | -------- | ------------------------------------------------ |
| `command`   | `"rbtr"` | How to invoke the CLI (see below)                |
| `autoIndex` | `true`   | Auto-index on session start when no index exists |

### CLI invocation modes

The `command` setting determines how `rbtr` is called:

| Value                 | Invocation                            | Use case                                    |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| `"rbtr"`              | `rbtr --json <cmd>`                   | Installed globally (`uv tool install rbtr`) |
| `"uvx"`               | `uvx rbtr --json <cmd>`               | Published on PyPI, no global install        |
| `"uvx --from <path>"` | `uvx --from <path> rbtr --json <cmd>` | Local development from a directory          |

The extension validates the command on session start and
shows an error with install instructions if it fails.

## How it works

The extension shells out to the `rbtr` CLI via `pi.exec()`.
All commands use `--json` for machine-readable NDJSON output.
There is no daemon — each tool call spawns a process. Builds
are incremental (blob-SHA dedup) so repeated calls are fast.

### Session lifecycle

1. **`session_start`** — load settings, resolve CLI command,
   run `rbtr status`. If the index exists, show the symbol
   count in the footer. If not and `autoIndex` is true,
   start a background indexing.
2. **`before_agent_start`** — append a note to the system
   prompt telling the agent the index is available.
3. **Tool calls** — each tool runs `rbtr --json <subcommand>`
   with a 30-second timeout (10 minutes for indexing).

### Background indexing

`rbtr_index` and `/rbtr-index` launch indexing as a
background process. The footer shows "building…" and updates
to the symbol count on completion. If the LLM searches
while a build is running, it gets an error asking it to
retry after the build completes. Concurrent build requests
are deduplicated.

### Output truncation

All query tool results are truncated to 50 KB / 2000 lines
(pi's default limits) to avoid overflowing the LLM context.
Truncation is noted in the output with a suggestion to use
`--limit` or `rbtr_read_symbol` for details.

## Development

```bash
bun install               # install dependencies
just check                # full check (Python + TypeScript)
just lint-ts              # biome lint
just fmt-ts               # biome format
just typecheck-ts         # tsc --noEmit
```

### Project structure

```text
packages/pi-rbtr/
├── package.json              # pi manifest, npm dependencies
├── tsconfig.json             # strict TypeScript config
└── extensions/
    └── rbtr-index/
        ├── index.ts          # extension entry point
        ├── exec.ts           # CLI resolution, pi.exec() helpers
        ├── render.ts         # custom TUI renderers for all tools
        └── settings.ts       # config load/save, settings UI
```

### Architecture reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the extension
is structured, CLI integration details, and rendering design.
