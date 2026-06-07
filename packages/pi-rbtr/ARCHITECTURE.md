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
2. **Minimal state.** The extension holds session-level
   transport state (`DaemonSession` caches RPC/PUB
   endpoints) but no conversation or index state. The
   index lives in DuckDB, managed by `rbtr`. Session
   branching, forking, and resuming have no effect on
   the extension.
3. **Graceful degradation.** If `rbtr` is not installed,
   the extension notifies the user and disables tools.
   If indexing is running, query tools report "in progress"
   instead of failing silently.

`index.ts` is the extension factory. It registers
tools, commands, and event handlers, and holds the
runtime state (daemon session, index promise, settings).
Daemon communication lives in `daemon-client.ts` and
`daemon-session.ts`; `exec.ts` provides the CLI fallback
when the daemon is unavailable. Rendering and settings
are pure helpers with no side effects.

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

`rbtr_search` accepts optional `keywords` and `variants`
parameters for query expansion. The session LLM generates
these inline when constructing the tool call, guided by
`promptGuidelines`. The daemon returns
`SearchResponse.expansion` reflecting the client-supplied
data.

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

### Daemon-first, CLI fallback

The extension connects to the rbtr daemon via ZMQ
(`DaemonSession`). If the daemon is unavailable, it falls
back to spawning `rbtr` as a subprocess via `pi.exec()`.
The daemon is auto-started on first use and reconciled on
each pi session start (version check, restart if stale).
See [Daemon protocol][proto] for the wire format, message
types, and error codes.

[proto]: ../rbtr/ARCHITECTURE.md#daemon-protocol

#### Stable IPC endpoints

The daemon binds to fixed paths inside `runtime_dir`
(`daemon.rpc`, `daemon.pub`). These paths are derived
from the directory, not from the PID, so they are stable
across restarts. This property is what makes both the
RPC retry in `DaemonSession.send()` and the ZMQ SUB
auto-reconnect work without endpoint re-discovery.

#### RPC reconnection

`DaemonSession.send()` retries once on transport failure:
nulls the cached status, calls `refresh()` to re-check
the daemon, and retries the request. Because the IPC path
is stable, the retry connects to the restarted daemon
transparently.

#### PUB reconnection

ZMQ SUB sockets auto-reconnect by default
(`reconnectInterval` = 100 ms). The `for await` async
iterator in `subscribe()` does not terminate on transport
failure — it blocks on `receive()` while ZMQ retries in
the background. When the daemon re-binds the same IPC
path, messages resume. The subscription never needs to
be torn down and recreated.

#### Health-check timer

ZMQ reconnection is transparent at the transport layer
but invisible at the UI layer — if the daemon dies, the
footer freezes at whatever it last showed. A 30-second
`setInterval` calls `session.detectTransition()` (which
wraps `refresh()`) to detect daemon death or return:

- **died:** footer → "disconnected (cli)".
- **returned:** queries index status, restores footer,
  starts the PUB subscription if it wasn't already active.
- **unchanged:** no-op.

The daemon publishes no startup notification for
already-indexed repos and no periodic heartbeat, so
active probing is the only way to detect lifecycle
transitions from the extension side.

#### `queryIndexStatus` is not a liveness check

`queryIndexStatus()` falls back to CLI when the daemon
is unavailable, so a non-null return does not mean the
daemon is alive. It also does not update
`session.available` when the daemon returns from a dead
state. Use `session.refresh()` (or `detectTransition()`)
for daemon liveness; reserve `queryIndexStatus()` for
reading index state after liveness is established.

### `pi.exec()` over `child_process.spawn`

`pi.exec()` integrates with pi's abort signal and timeout
handling. It returns a simple `{ stdout, stderr, code }`
object. There is no need for streaming stdout (all `rbtr`
commands produce output only at the end), so `spawn` would
add complexity without benefit.

### Footer class

`Footer` (`footer.ts`) encapsulates all status-bar
interaction: static text, animated spinners, and colour
tinting. It owns a single `setInterval` timer that
rotates through braille frames so the footer shows visible
motion during long-running daemon operations (model load,
orphan sweep). The extension creates one `Footer` at
session start and uses it throughout — indexing callbacks,
health-check transitions, and error display all go
through the same instance.

### Project-local settings only for writes

The `/rbtr-settings` command writes only to
`.pi/rbtr-index.json`, never to the global file. This
avoids surprising side effects — project settings are
visible and version-controllable, global settings require
intentional manual editing.
