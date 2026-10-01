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
pi install npm:@janrito/pi-rbtr
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

Nine tools, registered automatically on session start. `rbtr_status` and `rbtr_gc` start
inactive; `rbtr_index_tools` loads them when the agent needs them:

| Tool                   | Description                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `rbtr_search`          | Search by name, keyword, or concept (BM25 + semantic + name fusion). Returns a 20-line preview per hit, flagged when clipped. |
| `rbtr_read_symbol`     | Read a symbol's full source by name                                                                                           |
| `rbtr_list_symbols`    | Structural table of contents for a file                                                                                       |
| `rbtr_find_refs`       | Find references via the dependency graph (imports, docs)                                                                      |
| `rbtr_changed_symbols` | Symbols that changed between two git refs                                                                                     |
| `rbtr_watch`           | Watch refs and keep them indexed (background, incremental)                                                                    |
| `rbtr_index_tools`     | Load `rbtr_status` and `rbtr_gc`, which start inactive                                                                        |
| `rbtr_status`          | Check whether the index exists and how many symbols it contains                                                               |
| `rbtr_gc`              | Reclaim index storage. **Destructive**; previews as a dry run unless told otherwise                                           |

The extension also adds a one-line note to the system prompt, so
the agent knows the index is there without being told: rbtr tools
to find code by meaning, structure or name, grep for exact
strings. Each tool's description leads with what it is used
instead of, and the tools that find code are listed first.

### When to use which tool

- **Concept query** ("how does authentication work") →
  `rbtr_search` with `keywords` and `variants`. More precise
  than grep for semantic queries. The model writes the expansion
  terms itself, prompted by the tool's description and parameters.
- **Known symbol** ("read the source of `fuse_scores`") →
  `rbtr_read_symbol`. Faster than finding the file and reading it.
- **File structure** ("what's in `config.py`?") →
  `rbtr_list_symbols`. One-line-per-symbol TOC with line ranges.
- **Exact string match** ("find all `TODO` comments") →
  `grep`. The index is structural, not textual.
- **Who calls X?** → `rbtr_find_refs`. Follows import and doc
  edges in the dependency graph.
- **What changed?** → `rbtr_changed_symbols`. Function-level
  diff between two refs, not line-level.

### Tool examples

Shapes, not fixtures — the line numbers and scores below move
with the code they describe.

Every tool replies with one JSON object, rbtr's own response, in
the shape its output schema declares; codemode scripts receive the
same object as a structured value. An empty result has the same
shape with an empty list. A read names the snapshot it read as
`resolved` (its SHA, and whether that was the ref asked for, `head`,
the dirty working tree, or the latest indexed commit because the
one asked for is not indexed yet), and echoes its scoping paths,
repo-relative.

**`rbtr_search`** — query in, scored results out:

```json
{"kind": "search", "results": [
  {"name": "fuse_scores", "kind": "function", "file_paths": ["src/rbtr/index/search.py"],
   "line_start": 298, "line_end": 380, "score": 0.49,
   "preview": {"text": "def fuse_scores(...):\n    ...", "clipped": true, "total_lines": 83}}
], "resolved": {"sha": "4a6a6ffc…", "source": "head"}}
```

A hit carries the first 20 lines of the symbol. A longer one
is `clipped` and reports `total_lines`; `rbtr_read_symbol`
returns the whole body.

The per-signal ranking breakdown is omitted by default; pass
`explain: true` to include a nested `signals` object.

**`rbtr_read_symbol`** — symbol name in, full source out:

```json
{"kind": "read_symbol", "chunks": [
  {"name": "fuse_scores", "kind": "function", "file_path": "src/rbtr/index/search.py",
   "line_start": 298, "line_end": 380, "content": "def fuse_scores(...):\n    ..."}
], "resolved": {"sha": "4a6a6ffc…", "source": "head"}, "file_paths": null}
```

**`rbtr_list_symbols`** — file path in, TOC out:

```json
{"kind": "list_symbols", "chunks": [
  {"name": "_name_score_expr", "kind": "function", "line_start": 44, "line_end": 86},
  {"name": "fuse_scores", "kind": "function", "line_start": 298, "line_end": 380}
], "resolved": {"sha": "4a6a6ffc…", "source": "head"}, "file_path": "src/rbtr/index/search.py"}
```

**`rbtr_find_refs`** — symbol name in, referring symbols out:

```json
{"kind": "find_refs", "refs": [
  {"name": "from rbtr.index.store import IndexStore", "kind": "import",
   "file_path": "src/rbtr/daemon/watcher.py", "line_start": 30, "edge": "imports"}
], "resolved": {"sha": "4a6a6ffc…", "source": "head"}, "file_paths": null}
```

**`rbtr_changed_symbols`** — two refs in, changed symbols out:

```json
{"kind": "changed_symbols", "changes": [
  {"chunk": {"name": "resolveCommand", "kind": "function", "file_path": "exec.ts", "line_start": 34, "line_end": 70},
   "change": "modified"}
], "base_sha": "5ba5a78c…", "head_sha": "4a6a6ffc…", "file_paths": null}
```

**`rbtr_watch`** — refs in, the watch set out:

```json
{"kind": "watch_set", "watched": [
  {"ref": "HEAD", "sha": "4a6a6ffc…", "indexed": true},
  {"ref": "main", "sha": "9e51a463…", "indexed": false}
]}
```

An unindexed ref is built in the background; the footer shows
progress.

**An error** — rbtr's code and message, flagged as an error:

```json
{"kind": "error", "code": "index_not_built", "message": "Ref 'main' is not indexed — run rbtr watch first"}
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

| Command          | Description                                                                             |
| ---------------- | --------------------------------------------------------------------------------------- |
| `/rbtr-status`   | Show index status (chunk count, path)                                                   |
| `/rbtr-index`    | Index the repository in the background, or the given refs: `/rbtr-index main feature-x` |
| `/rbtr-settings` | View and toggle extension settings                                                      |

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
  "autoIndex": true,
  "annotate": true
}
```

| Key         | Default  | Description                                                                             |
| ----------- | -------- | --------------------------------------------------------------------------------------- |
| `command`   | `"rbtr"` | How to invoke the CLI (see below)                                                       |
| `autoIndex` | `true`   | Auto-index on session start when no index exists                                        |
| `annotate`  | `true`   | Append index facts to `bash` searches and to the first read of a large file (see below) |

### Index facts in `bash` and `read` output

Agents often keep to `bash` and `read` even with the rbtr
tools available, so with `annotate` on, the extension
appends what the index knows to those tools' output, in a
block headed `[rbtr index]` of at most ten lines:

```text
$ grep -rn "def union" django
django/db/models/query.py:939:    def union(self, *other_qs, all=False):

[rbtr index]
used by: django/db/models/sql/compiler.py, django/db/models/__init__.py
```

- After a `grep`, `rg`, `ag` or `git grep` for a name (`union`,
  `QuerySet.union`, `def union`): where it is defined, unless
  the output already shows it, and which files import it.
- After a search for a phrase: the index's three best
  matches.
- After the first read of a file of 200 lines or more: its
  outline, with line ranges.

Nothing is appended when the index has nothing to add, when
the tool failed, or when the daemon is not running.

### CLI invocation modes

The `command` setting determines how `rbtr` is called:

| Value                 | Invocation                            | Use case                                    |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| `"rbtr"`              | `rbtr --json <cmd>`                   | Installed globally (`uv tool install rbtr`) |
| `"uvx"`               | `uvx rbtr --json <cmd>`               | Published on PyPI, no global install        |
| `"uvx --from <path>"` | `uvx --from <path> rbtr --json <cmd>` | Local development from a directory          |

The extension validates the command on session start and
shows an error with install instructions if it fails.

## Development

Development requires Node.js 22.19 or later and npm. The full
repository checks also require Python 3.13, uv, and just.

```bash
npm install               # install dependencies
just check                # full check (Python + TypeScript)
just lint-ts              # biome lint
just fmt-ts               # biome format
just typecheck-ts         # tsc --noEmit
```

### Architecture reference

The extension talks to the daemon over ZMQ and falls back to
the CLI when none is reachable, starting one on first use.
[ARCHITECTURE.md](ARCHITECTURE.md) covers the session
lifecycle, the reconnection model, and rendering.
