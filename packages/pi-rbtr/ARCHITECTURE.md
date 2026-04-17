# Architecture

Technical reference for contributors and maintainers.
For usage, see [README.md](README.md).

---

## Overview

pi-rbtr is a [pi extension package][pi-pkg] that bridges
the `rbtr` code index CLI with pi's tool system. The
extension registers tools that the LLM can call, handles
CLI invocation and output parsing, manages background
builds, and renders results in the TUI.

[pi-pkg]: https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/packages.md

The design prioritises three properties:

1. **Thin bridge.** The extension contains no indexing
   logic. All work is delegated to the `rbtr` CLI. The
   extension's job is invocation, parsing, and presentation.
2. **Stateless.** No session-dependent state. The index
   lives in DuckDB, managed by `rbtr`. Session branching,
   forking, and resuming have no effect on the extension.
3. **Graceful degradation.** If `rbtr` is not installed,
   the extension notifies the user and disables tools.
   If indexing is running, query tools report "in progress"
   instead of failing silently.

### Modules

```text
extensions/rbtr/
├── index.ts      Entry point — events, commands, tools
├── exec.ts       CLI resolution and invocation
├── render.ts     Custom TUI renderers
└── settings.ts   Config file loading and saving
```

`index.ts` is the extension factory. It registers
everything and holds the runtime state (resolved command,
index promise, settings). The other modules are pure
helpers with no side effects.

---

## CLI integration

### Command resolution

The `command` config setting is resolved once on
`session_start` into a `{ executable, baseArgs }` pair.
Every tool call appends its subcommand and arguments to
`baseArgs`:

```text
Setting               Executable    Base args
─────────             ──────────    ─────────
"rbtr"                rbtr          [--json]
"uvx"                 uvx           [rbtr, --json]
"uvx --from ./pkg"    uvx           [--from, ./pkg, rbtr, --json]
```

The `--json` flag is always present. It makes `rbtr`
output NDJSON (one JSON object per line) instead of
rich-formatted text. The extension never sees TTY output.

### Invocation

All CLI calls go through `pi.exec()`, which returns
`{ stdout, stderr, code, killed }`. Two helpers in
`exec.ts` wrap it:

- `runRbtr()` — runs a command, throws on non-zero exit
  or kill. Returns raw stdout/stderr.
- `runRbtrJson<T>()` — runs a command and parses stdout
  as NDJSON into a typed array.

### Error handling

| Condition         | Behaviour                                                                |
| ----------------- | ------------------------------------------------------------------------ |
| Command not found | `session_start` sets footer to error, notifies with install instructions |
| Non-zero exit     | `runRbtr` throws with stderr content                                     |
| Timeout           | `pi.exec()` kills the process; `runRbtr` throws                          |
| Empty output      | Callers handle gracefully (e.g. "no results")                            |

### Validation

On `session_start`, the extension runs `rbtr status` with
a 5-second timeout. If it fails, `cliAvailable` is set to
`false` and all tools throw on invocation. This single
check covers command-not-found, broken installs, and
permission errors.

---

## Tool design

### Parameter schemas

All tools use TypeBox schemas (`Type.Object(...)`) for
parameter validation. Parameters are minimal — most tools
take a single required string (query, symbol name, file
path). Optional parameters have sensible defaults.

### Prompt integration

Each tool defines two prompt fields:

- `promptSnippet` — one-line summary shown in the
  "Available tools" section of the system prompt.
- `promptGuidelines` — bullets appended to the
  "Guidelines" section, teaching the LLM when and how
  to use the tool.

The `before_agent_start` event appends a general note
about index availability to the system prompt, so the
LLM knows it can use the tools without being told.

### Output contract

Query tools (`search`, `read-symbol`, `list-symbols`,
`find-refs`, `changed-symbols`) follow the same pattern:

1. Call `runRbtr()` with the appropriate subcommand.
2. If stdout is empty, return a "not found" message.
3. Otherwise, truncate to pi's limits (50 KB / 2000 lines)
   and return the raw NDJSON as content.

The raw NDJSON is sent to the LLM as tool content. The LLM
parses JSON natively — no reformatting needed. The custom
renderers parse the same NDJSON for TUI display.

### The `requireReady` guard

A shared guard function checks two conditions before any
query tool executes:

- `cliAvailable` is true (validated on session start).
- `indexPromise` is null (no indexing in progress).

If either check fails, the guard throws an error that the
LLM receives as an `isError` tool result with an
actionable message.

---

## Index management

### Background indexing

`startIndexing()` fires a `pi.exec()` call for `rbtr index`
and stores the returned promise in `indexPromise`. The
function returns `false` if a indexing is already running
(promise is non-null).

The promise chain:

1. On success — parse index stats, update footer with
   symbol count, notify user.
2. On failure — set footer to "indexing failed", notify
   with error message.
3. Finally — clear `indexPromise` to null.

UI context (`setStatus`, `notify`, `theme`) is captured
by reference at indexing start so the callbacks can update
the TUI after the event handler returns.

### Auto-index

On `session_start`, if the index does not exist and
`autoIndex` is true, `startIndexing()` is called
automatically. Indexing runs in the background — the
session is immediately usable for other work.

### Timeout

Builds have a 10-minute timeout. A fresh build with
embeddings for a large repository can take several minutes
(embedding is the bottleneck). Incremental builds with
blob-SHA dedup typically complete in seconds.

---

## Rendering

### Architecture

`render.ts` exports paired `renderCall` / `renderResult`
functions for each tool. They are wired into tool
definitions as callbacks — pi calls them during TUI
rendering.

Both functions receive the pi `Theme` object for
consistent styling. `renderResult` receives an
`AgentToolResult<unknown>` — the same object returned by
the tool's `execute` function.

### Rendering strategy

The render functions parse the NDJSON content text from
`result.content` at render time. This avoids duplicating
parsed data in `details` and keeps the tool execute
functions simple.

The `tryParseNdjson<T>()` helper silently returns an empty
array on parse failure, so renderers fall back to raw text
display.

### Collapsed vs. expanded

Search results show up to 5 results in collapsed view
(one line each: score, path, kind, name) and all results
with code previews when expanded (Ctrl+O).

Read-symbol shows path and line range when collapsed,
full source when expanded.

Status, index, list-symbols, find-refs, and
changed-symbols always show their compact form — the
data fits in a few lines.

---

## Settings

### File locations

Settings are plain JSON files at two locations:

| Path                          | Scope                 |
| ----------------------------- | --------------------- |
| `~/.pi/agent/rbtr-index.json` | Global (all projects) |
| `<cwd>/.pi/rbtr-index.json`   | Project-local         |

### Merge order

Defaults ← global ← project. Project values override
global values for the same key. Unknown keys are
preserved on write.

### Loading

Settings are loaded once on `session_start`. Changes
require `/reload` or a new session. The `/rbtr-settings`
command writes to the project-local file only — global
settings are edited by hand.

---

## Design decisions

### Raw NDJSON as tool content

The extension sends `rbtr`'s raw JSON output directly to
the LLM rather than reformatting it. LLMs parse JSON
natively and can extract exactly the fields they need.
Reformatting would lose information or add size. The
custom renderers handle human-readable display
independently.

### No daemon

The extension spawns a `rbtr` process for each tool call.
This is simple and stateless — no socket management, no
connection lifecycle, no reconnection logic. Incremental
builds complete in seconds due to blob-SHA dedup, so
process overhead is negligible.

A daemon (Unix socket, JSON-RPC, server-push progress
notifications) is planned for a future phase of the rbtr
project. The extension will switch from `pi.exec()` to a
socket client when the daemon is available.

### `pi.exec()` over `child_process.spawn`

`pi.exec()` integrates with pi's abort signal and timeout
handling. It returns a simple `{ stdout, stderr, code }`
object. There is no need for streaming stdout (all `rbtr`
commands produce output only at the end), so `spawn` would
add complexity without benefit.

### Captured UI context for indexing callbacks

The `startIndexing()` function captures `ctx.ui.setStatus`,
`ctx.ui.notify`, and `ctx.ui.theme` by reference before
returning. This allows the promise callbacks to update the
TUI after the `session_start` handler has returned. The
captured references remain valid because pi's UI methods
are fire-and-forget operations on the TUI singleton.

### Project-local settings only for writes

The `/rbtr-settings` command writes only to
`.pi/rbtr-index.json`, never to the global file. This
avoids surprising side effects — project settings are
visible and version-controllable, global settings require
intentional manual editing.
