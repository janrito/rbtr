# Pi Extension for rbtr Code Index

## Status: ✅ All phases complete

## Index

1. [Overview](#overview)
2. [Details](#details)
3. [Questions](#questions)
4. [Settings](#settings)
5. [CLI resolution](#cli-resolution)
6. [Assumptions](#assumptions)
7. [Architecture decisions](#architecture-decisions)
8. [Extension layout](#extension-layout)
9. [Tools](#tools)
10. [Commands](#commands)
11. [Events & lifecycle](#events--lifecycle)
12. [Custom rendering](#custom-rendering)
13. [Phases](#phases)

---

## Overview

Create a project-local pi extension (`.pi/extensions/rbtr-index/`)
that gives the LLM access to rbtr's structural code index. The
extension shells out to the `rbtr` CLI via `pi.exec()`, registers
tools for search and code navigation, shows indexing progress in
the footer, and renders results with custom TUI components.

This corresponds to Phase 2 of the grand migration plan, scoped
down: **no daemon, no daemon-client, no Unix socket** — just CLI
invocation. The daemon protocol will be added in a later phase
when `rbtr daemon` is implemented.

---

## Details

### What the extension provides to the LLM

The agent gains the ability to:

- **Search** the code index with ranked fusion (BM25 + semantic +
  name match). More precise than grep for concept queries like
  "retry logic" or "error handling in HTTP client".
- **Read symbols** by name — jump directly to a function/class
  definition without knowing the file path.
- **List symbols** in a file — structural table of contents
  (functions, classes, methods, imports) with line ranges.
- **Find references** to a symbol via the dependency graph
  (import edges, test edges, doc edges).
- **Show changed symbols** between two git refs — structural diff
  rather than line-level diff.
- **Build/rebuild** the index on demand.
- **Check status** — whether an index exists and how many symbols
  it contains.

### What the extension does automatically

- On `session_start`, checks index status. If no index exists,
  offers to build (or auto-builds — see [Questions](#questions)).
- Shows indexing progress in the footer via `ctx.ui.setStatus()`.
- Injects a one-liner in the system prompt via
  `before_agent_start` noting the index is available (so the LLM
  knows it can use the tools without being told).

### Interface: rbtr CLI

All communication with the index is via `pi.exec()` calling
`uv run --project <rbtr-root>/packages/rbtr rbtr --json <subcommand>`.
The `--json` flag ensures machine-readable NDJSON output
(one JSON object per line). The extension parses each line as a
JSON object.

CLI commands and their output shapes:

| Command | Output | Shape per line |
|---------|--------|----------------|
| `rbtr --json status` | Single JSON | `{ exists, db_path?, total_chunks? }` |
| `rbtr --json build` | Single JSON | `{ ref, stats: { total_chunks, total_edges, ... }, errors }` |
| `rbtr --json search QUERY` | NDJSON | `{ chunk: { id, file_path, kind, name, ... }, score, lexical, semantic, ... }` |
| `rbtr --json read-symbol NAME` | NDJSON | `{ id, file_path, kind, name, content, line_start, line_end, ... }` |
| `rbtr --json list-symbols FILE` | NDJSON | `{ id, file_path, kind, name, line_start, line_end, ... }` |
| `rbtr --json find-refs SYMBOL` | NDJSON | `{ source_id, target_id, kind }` |
| `rbtr --json changed-symbols --base REF --head REF` | NDJSON | `{ id, file_path, kind, name, ... }` |

### How it differs from `bash` + `rbtr`

The agent can already run `rbtr` via the `bash` tool. The
extension adds value by:

1. **Typed parameters** — the LLM gets schema-validated args
   instead of constructing shell commands.
2. **Custom rendering** — search results show scores, syntax-
   highlighted code previews, and file paths. Much better than
   raw JSON in the bash output.
3. **Output truncation** — large result sets are truncated to
   stay within context limits, with a note on how many results
   were omitted.
4. **Prompt integration** — `promptSnippet` and
   `promptGuidelines` teach the LLM when and how to use each
   tool without manual instructions.
5. **Auto-build** — the extension manages index lifecycle.
6. **Progress feedback** — footer status during indexing.

---

## Questions

1. **Auto-build on session start?** Should the extension
   automatically trigger `rbtr build` when the index is missing
   or stale, or should it just notify the user and let them
   decide?
   - Leaning: auto-build if no index exists, notify otherwise.
     Building is idempotent and skips unchanged files (fast).
   - **Answer:** Configurable via `rbtr-index.json` settings
     file. `autoBuild` boolean, default `true`. See
     [Settings](#settings) section. On session start: if
     `autoBuild` is true and no index exists, build
     automatically. If index exists, just report status. The
     `/rbtr-settings` command lets the user toggle this.

2. **`uv run` vs installed `rbtr`?** Should the extension
   assume `rbtr` is on `$PATH` (installed via `uv tool install`)
   or always use `uv run --project ...`?
   - Leaning: use `uv run --project <cwd>/packages/rbtr` since
     this is a monorepo development setup. Add a config option
     later for installed `rbtr`.
   - **Answer:** Configurable via `rbtr-index.json` with a
     `command` setting that supports three modes:
     - `"rbtr"` (default) — assume `rbtr` is on `$PATH`
       (installed via `uv tool install rbtr`). Recommended
       for end users.
     - `"uvx"` — run via `uvx rbtr`. One-shot from the
       published PyPI package without global install.
     - `"uvx --from <path>"` — run via `uvx --from <path> rbtr`.
       One-shot execution from a local directory. For
       monorepo development.

     Resolved at session start. The extension tries the
     configured command; if it fails, notifies the user with
     install instructions.

     See [Settings](#settings) and [CLI resolution](#cli-resolution)
     for details.

3. **Staleness detection?** How to detect if the index is stale
   (HEAD moved since last build)? The current `status` command
   doesn't report the indexed ref.
   - Leaning: defer staleness detection. Build is fast due to
     blob-SHA dedup. Can add a `--check` flag to `rbtr status`
     later.
   - **Answer:** Defer to a later phase. Requires `rbtr status`
     to report the indexed ref — a Python-side change. For now,
     build is fast due to blob-SHA dedup, so rebuilding on
     demand is acceptable.

4. **Search result limit in tool output?** What's the right
   default limit for search results sent to the LLM?
   - Leaning: default 10, matching the CLI default. The
     truncation utilities will handle cases where content is
     too large.

5. **Should `build` run in background?** Building can take
   seconds to minutes (embedding). Should the tool block or
   return immediately and report progress via status?
   - Leaning: block. `pi.exec()` supports timeout and signal.
     The LLM needs to know when the build finishes. Use
     `onUpdate` to stream progress if possible.
   - **Answer:** Background. The `rbtr_build` tool and
     `/rbtr-build` command both launch a build as a background
     `pi.exec()` call. Progress is reported via
     `ctx.ui.setStatus()` in the footer. The tool returns
     immediately with "build started" — when the build
     completes, a notification fires and the status updates.

     The daemon (future) is the proper long-term solution:
     the extension would send a `build` request over the
     Unix socket and receive progress notifications
     asynchronously. For now, we spawn a background process
     and poll/await completion.

     Implementation: the extension keeps a `buildPromise` in
     memory. If the LLM calls `rbtr_search` while a build is
     running, it gets a "build in progress" message. When the
     build finishes, subsequent searches work normally.

6. **Repo path?** Should the extension hardcode `.` (cwd) as
   the repo path, or allow the LLM to specify it?
   - Leaning: default to `ctx.cwd`, don't expose repo_path in
     the tool schema. This extension is project-scoped.

---

## Settings

Pi extensions don't have a built-in settings mechanism. The
established pattern (used by `preset.ts`, `subagent/agents.ts`)
is to read JSON config files from well-known locations and merge
them (project overrides user).

**Config file:** `rbtr-index.json`

| Location | Scope |
|----------|-------|
| `~/.pi/agent/rbtr-index.json` | Global (all projects) |
| `<cwd>/.pi/rbtr-index.json` | Project-local (overrides global) |

**Schema:**

```json
{
  "command": "rbtr",
  "autoBuild": true
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `command` | `string` | `"rbtr"` | How to invoke the rbtr CLI. See [CLI resolution](#cli-resolution). |
| `autoBuild` | `boolean` | `true` | Auto-build on session start when no index exists. |

**Loading:** Read once on `session_start`. Merge: defaults ←
global ← project. No live reload — change the file, then
`/reload` or start a new session.

**`/rbtr-settings` command:** Opens a `SettingsList` UI
(same pattern as `tools.ts`) to toggle `autoBuild` and view
the resolved command. Changes are written back to the
project-local config file.

---

## CLI resolution

The `command` setting determines how rbtr is invoked. The
extension resolves it at session start into an `{ executable,
baseArgs }` pair.

| `command` value | Executable | Base args | Use case |
|-----------------|------------|-----------|----------|
| `"rbtr"` (default) | `"rbtr"` | `["--json"]` | Installed globally via `uv tool install rbtr` |
| `"uvx"` | `"uvx"` | `["rbtr", "--json"]` | Published on PyPI, no global install |
| `"uvx --from <path>"` | `"uvx"` | `["--from", "<path>", "rbtr", "--json"]` | Local dev from a directory path |

Every tool call then appends subcommand + args to `baseArgs`:

```typescript
// Example: rbtr_search("retry logic", limit=5)
const result = await pi.exec(
  resolved.executable,
  [...resolved.baseArgs, "search", query, "--limit", String(limit)],
  { signal, timeout: 30_000 }
);
```

**Validation on session start:**
1. Run `<resolved command> status` with a 5s timeout.
2. If it succeeds — set status footer, done.
3. If it fails — notify with actionable error:
   - Command not found → "rbtr not installed. Run:
     `uv tool install rbtr` or set `command` to `"uvx"`
     in `.pi/rbtr-index.json`"
   - uvx fails → "rbtr not found. Check `command` setting
     in `.pi/rbtr-index.json`"

---

## Assumptions

1. **`rbtr` CLI is available** via `uv tool install` or
   configured invocation method. Resolved at session start.
2. **Index already has data** or will be built on first use.
   The extension doesn't manage embedding model downloads
   (rbtr handles that internally).
3. **Single repo per session.** The extension indexes the repo
   at `ctx.cwd`. No multi-repo support needed now.
4. **`--json` flag goes before the subcommand** (verified from
   the CLI parser: `rbtr --json search "query"`).
5. **No daemon.** All operations are synchronous CLI calls.
   The daemon will be a future enhancement.
6. **Extension is TypeScript** (pi's native extension language).
   No Python-side changes needed — the CLI is the interface.
7. **The extension is project-local** (`.pi/extensions/`) during
   development. Will be extracted into the `pi-review` npm
   package in a later phase.
8. **Node.js built-ins** (`node:child_process`, `node:path`) are
   available in extensions.
9. **`rbtr` is installed** via `uv tool install rbtr` by default.
   Configurable to `uvx` (published) or `uvx --from <path>`
   (local dev).

---

## Architecture decisions

### CLI invocation via `pi.exec()`

`pi.exec()` is the right primitive. It returns `{ stdout, stderr,
code, killed }`, supports `signal` for cancellation, and
`timeout`. All rbtr commands produce JSON on stdout and
diagnostic messages on stderr.

The `command` setting is resolved once on session start into a
`{ executable, baseArgs }` pair. See [CLI resolution](#cli-resolution)
for the three supported modes. Invocation pattern:

```typescript
// resolved = { executable: "rbtr", baseArgs: ["--json"] }
const result = await pi.exec(
  resolved.executable,
  [...resolved.baseArgs, "search", query, "--limit", String(limit)],
  { signal, timeout: 30_000 }
);
```

### Minimal in-memory state

The extension holds only transient runtime state:

- `resolvedCommand` — `{ executable, baseArgs }` resolved once
  on `session_start` from the `command` setting.
- `buildPromise` — a `Promise | null` tracking the currently
  running build. Prevents concurrent builds and lets
  `rbtr_search` report "build in progress".
- `settings` — loaded config from `rbtr-index.json`.

No `appendEntry` needed. The index state lives in DuckDB,
managed by the `rbtr` CLI. Session branching/forking doesn't
affect the extension because it has no session-dependent state.

### Truncation strategy

Search results can be large (10 results × multi-line code
snippets). The extension uses pi's `truncateHead` utility with
the default limits (50KB / 2000 lines). For search results,
each result is formatted as a compact summary; full source is
available via `rbtr_read_symbol`.

### Error handling

- `rbtr` not found → notify user with installation instructions.
- Non-zero exit code → throw (pi reports errors to the LLM).
- Parse failure → throw with stderr content.
- Timeout → pi handles via signal.

---

## Extension layout

```
.pi/extensions/rbtr-index/
├── index.ts          # Extension entry point
├── tools.ts          # Tool definitions (search, read-symbol, etc.)
├── render.ts         # Custom renderCall/renderResult functions
├── exec.ts           # CLI invocation + command resolution
└── settings.ts       # Settings types, load/save, SettingsList UI
```

Single-directory extension with `index.ts` as entry point. Split
into modules for clarity but no npm dependencies needed — all
imports are from pi's built-in packages (`@mariozechner/pi-coding-agent`,
`@sinclair/typebox`, `@mariozechner/pi-tui`).

---

## Tools

### `rbtr_search`

**Purpose:** Search the code index with ranked fusion.

**Parameters:**
- `query` (string, required) — Search query.
- `limit` (number, optional, default 10) — Max results.

**Prompt snippet:** `"Search the structural code index for symbols, functions, classes, and code patterns"`

**Prompt guidelines:**
- `"Use rbtr_search for conceptual queries like 'retry logic' or 'error handling'. For exact string matches, use grep instead."`
- `"Search results include score breakdown (lexical, semantic, name). Higher scores indicate better matches."`

**Output to LLM:** Formatted list of results with file path,
symbol name, kind, score, and a code preview (first 4 lines).
Full source available via `rbtr_read_symbol`.

### `rbtr_read_symbol`

**Purpose:** Read a symbol's full source by name.

**Parameters:**
- `symbol` (string, required) — Symbol name (e.g.
  `HttpClient.retry`, `fuse_scores`).

**Prompt snippet:** `"Read a symbol's full source code by name from the code index"`

**Prompt guidelines:**
- `"Use rbtr_read_symbol after rbtr_search to read the full source of a specific symbol. More precise than grep or read for navigating to a known symbol."`

**Output to LLM:** Full source code with file path and line
range. Multiple matches are returned if the name is ambiguous.

### `rbtr_list_symbols`

**Purpose:** List symbols in a file (structural TOC).

**Parameters:**
- `file` (string, required) — File path (relative to repo root).

**Prompt snippet:** `"List all symbols (functions, classes, methods) in a file as a structural table of contents"`

**Prompt guidelines:**
- `"Use rbtr_list_symbols to understand the structure of a file before reading specific parts. More informative than reading the whole file."`

**Output to LLM:** Compact list: `line_start-line_end  kind  name`.

### `rbtr_find_refs`

**Purpose:** Find references to a symbol via the dependency graph.

**Parameters:**
- `symbol` (string, required) — Symbol name.

**Prompt snippet:** `"Find references to a symbol via the dependency graph (imports, tests, docs)"`

**Prompt guidelines:**
- `"Use rbtr_find_refs to find where a symbol is imported, tested, or documented. Shows structural relationships, not just text matches."`

**Output to LLM:** List of edges: `source_id → target_id (kind)`.

### `rbtr_changed_symbols`

**Purpose:** Show symbols that changed between two git refs.

**Parameters:**
- `base` (string, required) — Base ref (e.g. `main`).
- `head` (string, required) — Head ref (e.g. `feature-branch`).

**Prompt snippet:** `"Show symbols that changed between two git refs (structural diff)"`

**Prompt guidelines:**
- `"Use rbtr_changed_symbols to understand what code changed structurally between branches. Shows function/class-level changes, not line-level diffs."`

**Output to LLM:** Compact list of changed symbols with file
path, kind, and name.

### `rbtr_build`

**Purpose:** Build or rebuild the code index.

**Parameters:**
- `ref` (string, optional, default "HEAD") — Git ref to index.

**Prompt snippet:** `"Build or update the structural code index for the repository"`

**Prompt guidelines:**
- `"Use rbtr_build when the user asks to index the codebase or when rbtr_search returns no results. Building is incremental — unchanged files are skipped."`

**Output to LLM:** Build summary: files parsed, chunks created,
edges inferred, elapsed time, any errors.

### `rbtr_status`

**Purpose:** Check index status.

**Parameters:** None.

**Prompt snippet:** `"Check whether the code index exists and how many symbols it contains"`

**Output to LLM:** Index exists/not, chunk count, DB path.

---

## Commands

### `/rbtr-status`

**Description:** Show index status in the TUI.

Uses `ctx.ui.notify()` to display the result. Quick check without
involving the LLM.

### `/rbtr-build`

**Description:** Trigger an index build from the command line.

Launches build in background. Shows progress in the footer via
`ctx.ui.setStatus()`. Notifies on completion. Doesn't involve
the LLM.

### `/rbtr-settings`

**Description:** View and toggle extension settings.

Opens a `SettingsList` component (same pattern as the `tools.ts`
example) with toggleable settings:

| Setting | Values | Description |
|---------|--------|-------------|
| `autoBuild` | on / off | Auto-build on session start |
| `command` | display only | Shows resolved invocation method |

Changes are written to `.pi/rbtr-index.json` (project-local).

---

## Events & lifecycle

### `session_start`

On session start:
1. Load settings from `rbtr-index.json` (merge global/project).
2. Resolve CLI command from `command` setting.
3. Validate by running `<command> status` with 5s timeout.
4. If CLI unavailable → notify with install instructions, bail.
5. If index exists → set footer: `"rbtr: 6660 symbols"`.
6. If no index and `autoBuild` is true → launch background
   build, set footer: `"rbtr: building…"`.
7. If no index and `autoBuild` is false → set footer:
   `"rbtr: no index — /rbtr-build to create"`.

### `before_agent_start`

Inject a brief note into the system prompt if the index is
available:

```
The rbtr code index is available for this repository.
Use rbtr_search for concept queries, rbtr_read_symbol to read
symbol source, rbtr_list_symbols for file structure.
```

This is appended to `event.systemPrompt`, not replacing it.

---

## Custom rendering

### `renderCall` (all tools)

Compact one-line header showing the tool name and key arguments:

```
rbtr_search "retry logic" (limit: 10)
rbtr_read_symbol HttpClient.retry
rbtr_list_symbols src/client.py
rbtr_find_refs HttpClient
rbtr_changed_symbols main..feature
rbtr_build HEAD
rbtr_status
```

### `renderResult`

**`rbtr_search`:**
- Collapsed: score, file path, symbol name, kind — one line per
  result. Up to 5 results shown, rest collapsed.
- Expanded: adds code preview (first 4 lines) for each result.

**`rbtr_read_symbol`:**
- Collapsed: file path, line range, kind, symbol name.
- Expanded: full source code (use pi's `highlightCode` for
  syntax highlighting).

**`rbtr_list_symbols`:**
- Always compact: `line_start-line_end  kind  name` per symbol.

**`rbtr_find_refs`:**
- Always compact: `source → target (kind)` per edge.

**`rbtr_changed_symbols`:**
- Always compact: `kind  name  file_path` per symbol.

**`rbtr_build`:**
- Single line: `✓ 177 files, 6660 chunks, 823 edges — 4.2s`
  or `✗ Build failed: <error>`.

**`rbtr_status`:**
- Single line: `✓ 6660 chunks` or `✗ No index found`.

---

## Phases

### Phase 1: Scaffold, settings & CLI resolution ✅

**Goal:** Extension loads, reads settings, resolves CLI command,
validates rbtr availability, shows status in footer.

**Steps:**
1. ✅ Create `.pi/extensions/rbtr-index/` directory.
2. ✅ Write `settings.ts` — types, load from JSON, merge
   global/project, write back.
3. ✅ Write `exec.ts` — resolve `command` setting into
   `{ executable, baseArgs }`. Helper to invoke
   `<command> --json <subcommand>` via `pi.exec()`. Parse
   NDJSON output into typed arrays. Error handling: not
   found, non-zero exit, parse failure.
4. ✅ Write `index.ts` — minimal extension entry point.
   - `session_start`: load settings, resolve CLI, validate
     with `status`, set footer.
   - Register `/rbtr-status` command.
   - Register `/rbtr-settings` command (SettingsList UI).
5. ✅ Manual tests passed.

**Verify:**
- ✅ Extension loads without errors (print, JSON, auto-discover).
- ✅ `rbtr --json status` returns valid JSON (6663 symbols).
- ✅ Bad command config → extension loads, no crash.
- ✅ `uvx --from ./packages/rbtr` config → resolves correctly.
- `/rbtr-status` and `/rbtr-settings` UI — verified loading,
  interactive TUI testing deferred to manual session.

### Phase 2: Build tool (background) & auto-build ✅

**Goal:** Index builds run in background with progress in the
footer. Auto-build on session start when configured.

**Steps:**
1. ✅ Implement `startBuild()` in `index.ts`:
   - Fires `pi.exec()` for `rbtr build` (10 min timeout).
   - Stores `buildPromise`, prevents concurrent builds.
   - On completion: notifies with stats, updates footer.
   - On error: notifies, sets footer to "build failed".
   - Null-safe: handles empty build output (re-checks status).
2. ✅ Register `rbtr_build` tool — returns immediately with
   "build started" or "build in progress".
3. ✅ Register `rbtr_status` tool — reports chunk count or
   "build in progress".
4. ✅ Register `/rbtr-build` command.
5. ✅ `session_start`: auto-builds if `autoBuild` is true and
   no index exists.

**Verify:**
- ✅ `rbtr_build` tool starts build, returns immediately.
- ✅ `rbtr_status` tool reports correct chunk count (6749).
- ✅ Concurrent build requests → second returns "in_progress".
- ✅ Build completion notification with stats.
- ✅ Null build output handled gracefully (re-checks status).

### Phase 3: Core tools (search, read-symbol, list-symbols) ✅

**Goal:** LLM can search the index and read symbol source.

**Steps:**
1. ✅ Register `rbtr_search` tool — passes raw NDJSON to LLM.
2. ✅ Register `rbtr_read_symbol` tool.
3. ✅ Register `rbtr_list_symbols` tool.
4. ✅ `promptSnippet` and `promptGuidelines` on all three.
5. ✅ `before_agent_start` handler injects index availability
   note into system prompt.
6. ✅ Added `requireReady()` guard — throws if CLI unavailable
   or build in progress.

**Verify:**
- ✅ `rbtr_search "search scoring"` returns ranked results.
- ✅ `rbtr_read_symbol "classify_query"` returns full source.
- ✅ `rbtr_list_symbols` shows structural TOC with line ranges.
- ✅ Agent uses `rbtr_search` unprompted when asked "How does
  search scoring work?" (system prompt injection works).

### Phase 4: Navigation tools (find-refs, changed-symbols) ✅

**Goal:** LLM can explore the dependency graph and structural
diffs.

**Steps:**
1. ✅ Register `rbtr_find_refs` tool.
2. ✅ Register `rbtr_changed_symbols` tool.

**Verify:**
- ✅ `rbtr_find_refs "fuse_scores"` returns edge references.
- ✅ `rbtr_changed_symbols HEAD~5 HEAD` returns changed symbols.

### Phase 5: Custom rendering ✅

**Goal:** Tool results render beautifully in the TUI.

**Steps:**
1. ✅ Write `render.ts` with `renderCall` + `renderResult` for
   all 7 tools.
2. ✅ Search: score-coloured, shortened path, kind, name.
   Collapsed = 5 results. Expanded = all + code preview.
3. ✅ Read symbol: path + line range. Expanded = source lines.
4. ✅ List symbols: compact `line-range  kind  name` table.
5. ✅ Find refs: `source → target (kind)` per edge.
6. ✅ Changed symbols: `kind  name  path` per symbol.
7. ✅ Build: ✓ started / ⏳ in progress.
8. ✅ Status: ✓ count / ✗ not found / ⏳ building.
9. ✅ Wired into all tool definitions in `index.ts`.

**Verify:**
- ✅ Extension loads with renderers, tools work in print mode.
- Interactive TUI rendering deferred to manual session.

### Phase 6: Output truncation & polish ✅

**Goal:** Large result sets don't overflow the context.

**Steps:**
1. ✅ `truncateOutput()` helper using pi's `truncateHead` with
   `DEFAULT_MAX_BYTES` (50KB) / `DEFAULT_MAX_LINES` (2000).
2. ✅ Applied to all query tools: search, read-symbol,
   list-symbols, find-refs, changed-symbols.
3. ✅ Truncation note appended when output is clipped.
4. ✅ Edge cases handled: empty results, not found, CLI
   unavailable, build in progress (`requireReady()`).
5. ✅ Prompt guidelines reviewed — clear and accurate.

**Verify:**
- ✅ All tools return results within context limits.
- ✅ Truncation message visible when output is clipped.
- ✅ Error messages are actionable.
