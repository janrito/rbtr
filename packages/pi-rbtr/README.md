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

# Install the pi extension
pi install npm:@rbtr/pi
```

For development from a local clone (no global install):

```bash
# Install the extension from the local checkout
pi install -l ./packages/pi-rbtr

# Point the extension at the local rbtr source
# (in .pi/rbtr-index.json)
{ "command": "uvx --from ./packages/rbtr" }
```

## What the agent gets

Seven tools, registered automatically on session start:

| Tool                   | Description                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `rbtr_search`          | Search by name, keyword, or concept (BM25 + semantic + name fusion). Optional `keywords`/`variants` for query expansion. |
| `rbtr_read_symbol`     | Read a symbol's full source by name                                                                                      |
| `rbtr_list_symbols`    | Structural table of contents for a file                                                                                  |
| `rbtr_find_refs`       | Find references via the dependency graph (imports, tests, docs)                                                          |
| `rbtr_changed_symbols` | Symbols that changed between two git refs                                                                                |
| `rbtr_index`           | Index the repository (background, incremental)                                                                           |
| `rbtr_status`          | Check whether the index exists and how many symbols it contains                                                          |

The extension also injects a system prompt note so the agent
knows the index is available without being told.

### When to use which tool

- **Concept query** ("how does authentication work") →
  `rbtr_search` with `keywords` and `variants`. More precise
  than grep for semantic queries. The LLM generates expansion
  terms automatically via `promptGuidelines`.
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

### Tool examples

**`rbtr_search`** — query in, scored results out:

```json
{"kind": "search", "results": [
  {"name": "fuse_scores", "kind": "function", "file_path": "src/rbtr/index/search.py",
   "line_start": 298, "score": 0.49, "lexical": 0.0, "semantic": 0.81, "name_match": 0.0}
]}
```

**`rbtr_read_symbol`** — symbol name in, full source out:

```json
{"kind": "read_symbol", "chunks": [
  {"name": "fuse_scores", "kind": "function", "file_path": "src/rbtr/index/search.py",
   "line_start": 298, "line_end": 380, "content": "def fuse_scores(...):\n    ..."}
]}
```

**`rbtr_list_symbols`** — file path in, TOC out:

```json
{"kind": "list_symbols", "chunks": [
  {"name": "_name_score_expr", "kind": "function", "line_start": 44, "line_end": 86},
  {"name": "fuse_scores", "kind": "function", "line_start": 298, "line_end": 380}
]}
```

**`rbtr_find_refs`** — symbol name in, dependency edges out:

```json
{"kind": "find_refs", "edges": [
  {"source_id": "abc123", "target_id": "def456", "kind": "imports"}
]}
```

**`rbtr_changed_symbols`** — two refs in, changed symbols out:

```json
{"kind": "changed_symbols", "chunks": [
  {"name": "resolveCommand", "kind": "function", "file_path": "exec.ts", "line_start": 34}
]}
```

### Footer

The extension shows index state in the pi footer:

- **Building:** `rbtr: ⟳ parsing 42/177` (spinner with
  phase and progress).
- **Ready:** `rbtr: ● 1.2k` (symbol count).
- **Error:** `rbtr: not installed` or
  `rbtr: disconnected (cli)` when the daemon is down.

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

The extension talks to the rbtr daemon via ZMQ. If the
daemon is unavailable it falls back to the CLI. The
daemon is auto-started on first use and version-checked
on each session start. Indexing runs in the background;
results are truncated to fit the LLM context.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the session
lifecycle, reconnection model, and rendering design.

## Development

```bash
bun install               # install dependencies
just check                # full check (Python + TypeScript)
just lint-ts              # biome lint
just fmt-ts               # biome format
just typecheck-ts         # tsc --noEmit
```

### Architecture reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the extension
is structured, CLI integration details, and rendering design.
