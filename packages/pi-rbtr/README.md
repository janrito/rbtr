# pi-rbtr

A [pi] extension package that gives the LLM access to
rbtr's structural code index. The agent can search by name,
keyword, or concept, read symbol source, list file
structure, trace dependency edges, and compare structural
changes between git refs — without constructing shell
commands or parsing raw output.

[pi]: https://pi.dev

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

Nine tools, registered when pi loads the extension. `rbtr_status` and `rbtr_gc` start
inactive; `rbtr_index_tools` loads them when the agent needs them:

| Tool                   | Description                                                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `rbtr_search`          | Search by name, keyword, or concept (BM25 + semantic + name fusion). Returns the first 8 lines of each hit, flagged when clipped. |
| `rbtr_read_symbol`     | Read a symbol's full source by name                                                                                               |
| `rbtr_list_symbols`    | Structural table of contents for a file                                                                                           |
| `rbtr_find_refs`       | Find references via the dependency graph (imports, docs)                                                                          |
| `rbtr_changed_symbols` | Symbols that changed between two git refs                                                                                         |
| `rbtr_watch`           | Watch refs and keep them indexed (background, incremental)                                                                        |
| `rbtr_index_tools`     | Load `rbtr_status` and `rbtr_gc`, which start inactive                                                                            |
| `rbtr_status`          | Check whether the index exists and how many symbols it contains                                                                   |
| `rbtr_gc`              | Reclaim index storage. **Destructive**; previews as a dry run, and deletes only after you confirm                                 |

Inside a git repository, the extension also adds a one-line note to the system prompt,
so the agent knows the index is there without being told: rbtr tools to find code by
meaning, structure or name, grep for exact strings. Each tool's description leads with
what it is used instead of, and the tools that find code are listed first.

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

Every tool's reply is rbtr's own response, in the shape its output
schema declares. codemode scripts receive that object as a
structured value. The model reads it as text: one line per symbol
as `path:line kind name`, each file once, and a method named with
its scope (`EngineArgs::add_cli_args`, as `rbtr_read_symbol` takes
it). A read first names the snapshot it read: its SHA, and whether
that was the ref asked for, `head`, the dirty working tree, or the
latest indexed commit because the one asked for is not indexed yet.

**`rbtr_search`** — query in, ranked hits out, each with the first
8 lines of its source:

```text
read at 4a6a6ffc1b2e (head): 2 result(s)
match: src/rbtr/index/search.py:298 function fuse_scores
  def fuse_scores(
      frame: pl.DataFrame,
  … 75 more lines
match: src/rbtr/index/search.py:44 function _name_score_expr
  def _name_score_expr(query: str) -> pl.Expr:
```

A hit that ends in `… N more lines` is longer; `rbtr_read_symbol`
returns it whole.

**`rbtr_read_symbol`** — symbol name in, full source out:

```text
read at 4a6a6ffc1b2e (head)
defined: src/rbtr/index/search.py:298-380 function fuse_scores
def fuse_scores(
    ...
```

**`rbtr_list_symbols`** — file path in, outline out, without
imports and comments:

```text
read at 4a6a6ffc1b2e (head)
outline of src/rbtr/index/search.py:
  _name_score_expr (function) lines 44-86
  fuse_scores (function) lines 298-380
```

**`rbtr_find_refs`** — symbol name in, referring symbols out:

```text
read at 4a6a6ffc1b2e (head): 1 ref(s)
imports: src/rbtr/daemon/watcher.py:30 import from rbtr.index.index import Index
```

**`rbtr_changed_symbols`** — two refs in, changed symbols out:

```text
changes from 5ba5a78c0d1e to 4a6a6ffc1b2e: 1 change(s)
modified: exec.ts:34 function resolveCommand
```

The replies that report on the index rather than code are rbtr's
JSON, as are errors.

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

### With codemode

pi's `codemode` tool runs model-written JavaScript that calls other
tools. The rbtr tools declare their output schemas, so a script gets
each reply as a typed value — `refs`, `resolved`, or an error's
`code` — without parsing text, and can chain lookups and send the
model only what it needs. codemode is off by default; to turn it on:

```json
{ "defaultTools": ["+codemode"] }
```

in `~/.pi/agent/settings.json`, or a project's `.pi/settings.json`.
While it is on, each rbtr tool's description carries its script
declaration, about 1,000 tokens across the tools.

### Footer

The extension shows index state in the pi footer:

- **Starting:** `rbtr: ⠋ starting…` while the extension
  finds and starts the daemon.
- **Building:** `rbtr: ⠋ parsing 42/177 · 12s…` (spinner
  with phase, progress and elapsed time; a static `⟳`
  outside pi's terminal UI).
- **Collecting:** `rbtr: ⠋ collecting 3/10…` while garbage
  collection runs.
- **Ready:** `rbtr: ● 1.2k symbols`. `○` with
  `· 42% embedded` or `· not embedded` while embedding is
  incomplete, and `· not running` when the CLI serves the
  tools.
- **Not running:** `rbtr: not running`, when no daemon answers
  and the index holds nothing of this repository.
- **No index:** `rbtr: no index — /rbtr-index`, when the index
  holds no repository.
- **Repo not indexed:** `rbtr: repo not indexed — /rbtr-index`,
  when the index holds other repositories but not this one.
- **Ref not indexed:** `rbtr: ref not indexed`, when the
  repository's other refs are indexed but not the one reads use;
  reads use the latest indexed commit.
- **Not a repository:** `rbtr: not a git repository`, when pi runs
  outside a git repository; nothing is indexed.
- **Unavailable:** `rbtr: index unavailable`, when rbtr could not
  read the index status.
- **Error:** `rbtr: cannot run`, `rbtr: indexing failed`,
  or `rbtr: ✗ error`.

## Commands

Three user-facing commands (no LLM involved):

| Command          | Description                                                                             |
| ---------------- | --------------------------------------------------------------------------------------- |
| `/rbtr-status`   | Show index status (chunk count, path)                                                   |
| `/rbtr-index`    | Index the repository in the background, or the given refs: `/rbtr-index main feature-x` |
| `/rbtr-settings` | View and toggle extension settings                                                      |

## Configuration

Settings are read from JSON config files. Project-local
overrides global. The project file is read only when pi
trusts the project (`/trust`), like pi's own `.pi/` files:

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

| Key         | Default  | Description                                                                                |
| ----------- | -------- | ------------------------------------------------------------------------------------------ |
| `command`   | `"rbtr"` | How to invoke the CLI (see below)                                                          |
| `autoIndex` | `true`   | Index this repository when it is not indexed, at session start and when the daemon returns |
| `annotate`  | `true`   | Append index facts to `bash` searches and to the first read of a large file (see below)    |

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
- After a read of a file of 200 lines or more: its outline,
  the functions, classes and other named definitions with
  their line ranges, without imports or comments. A file is
  outlined once while the outline is in the model's context.

Nothing is appended when the index has nothing to add, when
the tool failed, when the daemon is not running, or when the
search looks only outside the repository (after a `cd`
elsewhere, or with paths outside it).

### CLI invocation modes

The `command` setting determines how `rbtr` is called:

| Value                 | Invocation                            | Use case                                    |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| `"rbtr"`              | `rbtr --json <cmd>`                   | Installed globally (`uv tool install rbtr`) |
| `"uvx"`               | `uvx rbtr --json <cmd>`               | Published on PyPI, no global install        |
| `"uvx --from <path>"` | `uvx --from <path> rbtr --json <cmd>` | Local development from a directory          |

The extension validates the command on session start and
shows a warning with install instructions if it fails.

## Development

Development requires Node.js 22.19 or later, npm, and pi: the
tests run against the installed pi. The full repository checks
also require Python 3.13, uv, and just.

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
[ARCHITECTURE.md](ARCHITECTURE.md) covers how a session
works and the choices behind it.
