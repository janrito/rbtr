# rbtr

An interactive, terminal-based PR review workbench powered by LLMs.
rbtr sits alongside you while you review a PR — it helps you reason
through changes, explore the code, and write clear feedback for the
author.

## Quick start

```bash
# Install (requires Python 3.13+)
uv tool install -e .

# Launch in any git repo
rbtr

# Or jump straight to a PR
rbtr 42
```

rbtr opens an interactive session in your terminal. Type plain text
to talk to the LLM, prefix with `/` for commands, or prefix with `!`
to run shell commands without leaving the session:

```text
you: explain the retry logic in src/client.py
claude/claude-sonnet-4-20250514: The retry logic uses exponential backoff…

you: !git diff HEAD~3 --stat
 src/client.py | 12 ++++++------
 1 file changed, 6 insertions(+), 6 deletions(-)

you: /review 42
Fetching PR #42…
```

## Providers

rbtr connects to LLMs through multiple providers. Use `/connect`
to authenticate:

| Provider   | Auth               | Command                                |
| ---------- | ------------------ | -------------------------------------- |
| Claude     | OAuth (Pro/Max)    | `/connect claude`                      |
| ChatGPT    | OAuth (Plus/Pro)   | `/connect chatgpt`                     |
| Google     | OAuth (free)       | `/connect google`                      |
| OpenAI     | API key            | `/connect openai sk-...`               |
| Fireworks  | API key            | `/connect fireworks fw-...`            |
| OpenRouter | API key            | `/connect openrouter sk-or-...`        |
| Endpoint   | URL + optional key | `/connect endpoint <name> <url> [key]` |

Multiple providers can be connected at the same time. Tab on
`/connect` autocompletes provider names.

### Endpoints

Any OpenAI-compatible API can be used as a custom endpoint:

```text
you: /connect endpoint deepinfra https://api.deepinfra.com/v1/openai di-...
you: /connect endpoint ollama http://localhost:11434/v1
you: /model deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct
```

Endpoints are first-class providers — they appear in `/model`
listings, support tab completion, and participate in the same
dispatch pipeline as builtin providers.

### GitHub

rbtr uses GitHub to fetch PRs and branches for review. Authenticate
with `/connect github` (device flow). This is separate from LLM
providers — it gives rbtr read access to your repositories.

## Models

Models use `<provider>/<model-id>` format:

```text
/model claude/claude-sonnet-4-20250514
/model chatgpt/gpt-5.2-codex
/model openai/gpt-4o
/model deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct
```

### Listing models

`/model` with no argument shows all models from connected providers,
marking the active one:

```text
you: /model
  claude:
    claude/claude-sonnet-4-20250514 ◂
    claude/claude-opus-4-20250514
  chatgpt:
    chatgpt/o3-pro
    chatgpt/gpt-5.2-codex
  deepinfra:
    deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct
```

Providers that don't expose a model listing show a hint instead:

```text
  ollama:
    /model ollama/<model-id>
```

The model list is fetched lazily on first Tab completion or `/model` command, and refreshed on every `/connect`.

### Switching models mid-conversation

Conversation history is preserved when you switch models. PydanticAI's
message format is provider-agnostic, so previous messages are converted
automatically when sent to the new provider:

```text
you: explain the retry logic in src/client.py
claude/claude-sonnet-4-20250514: The retry logic uses exponential backoff…

you: /model chatgpt/o3-pro
Model set to chatgpt/o3-pro

you: what do you think about adding jitter?
chatgpt/o3-pro: Based on the retry logic we discussed…
```

Only `/new` clears history (explicit user action). The active model is
persisted to `config.toml` across restarts. Conversation messages are
saved automatically to the session database (see Sessions below).

## Commands

| Command                | Description                                |
| ---------------------- | ------------------------------------------ |
| `/help`                | Show available commands                    |
| `/review`              | List open PRs and branches                 |
| `/review <id>`         | Select a PR or branch for review           |
| `/draft`               | View, sync, or post the review draft       |
| `/connect <service>`   | Authenticate with a service                |
| `/model`               | List available models from all providers   |
| `/model <provider/id>` | Set the active model                       |
| `/index`               | Index status, search, diagnostics, rebuild |
| `/compact`             | Summarise older context to free space      |
| `/compact reset`       | Undo last compaction (before new messages) |
| `/session`             | List, inspect, or delete sessions          |
| `/new`                 | Start a new conversation                   |
| `/quit`                | Exit (also `/q`)                           |

## Shell commands

Prefix any command with `!` to run it in a shell without leaving rbtr:

```text
you: !git log --oneline -5
you: !rg "TODO" src/
you: !cat src/client.py
```

Long output is truncated — press **Ctrl+O** to expand it.

## Tab completion

Tab works across all input modes:

| Context           | Example            | Completes                                      |
| ----------------- | ------------------ | ---------------------------------------------- |
| Slash commands    | `/rev` → `/review` | Command names                                  |
| Command arguments | `/connect c`       | Provider names, model IDs, draft subcommands   |
| Shell commands    | `!git ch`          | Bash programmable completion (branches, flags) |
| File paths        | `!cat ~/Doc`       | Directories and files (expands `~`)            |
| Executables       | `!my`              | Commands found in `PATH`                       |

A single match auto-accepts; multiple matches extend the common prefix
and show a menu (capped at 20 suggestions).

## Key bindings

| Key       | Action                                   |
| --------- | ---------------------------------------- |
| Enter     | Submit input                             |
| Alt+Enter | Insert newline (multiline input)         |
| Tab       | Autocomplete                             |
| Shift+Tab | Cycle thinking effort level              |
| Up/Down   | Browse history or navigate multiline     |
| Ctrl+C    | Cancel running task (double-tap to quit) |
| Ctrl+O    | Expand truncated shell output            |
| Ctrl+W    | Delete word backward                     |
| Ctrl+U    | Kill to start of line                    |
| Ctrl+K    | Kill to end of line                      |
| Ctrl+\_   | Undo                                     |

### Pasting

rbtr enables **bracketed paste** on the terminal, so pasted
content is never interpreted as keystrokes — newlines in pasted
text insert newlines into the prompt instead of submitting.

Large pastes are collapsed into an atomic marker so the input
line stays readable:

```text
> review this: [pasted 42 lines]
```

The marker is displayed in dim italic. The real content is
stored internally and expanded back when you press Enter —
the LLM receives the full pasted text, not the marker.

**Thresholds** (configurable in `config.toml` under `[tui]`):

| Setting                | Default | Triggers marker when…               |
| ---------------------- | ------- | ----------------------------------- |
| `paste_collapse_lines` | 4       | Paste has ≥ 4 lines                 |
| `paste_collapse_chars` | 200     | Single-line paste exceeds 200 chars |

Below the threshold, pasted text appears inline as-is.

**Markers are atomic:**

- Cursor skips over them — Left/Right arrows jump from one
  edge to the other.
- Backspace or Delete at a marker edge removes the entire
  marker and its content.
- Kill commands (Ctrl+W, Ctrl+U, Ctrl+K) that overlap a
  marker remove it whole.
- Multiple pastes produce independent markers; deleting one
  preserves the others.

Marker format: `[pasted 42 lines]` for multiline,
`[pasted 523 chars]` for long single-line. Duplicate counts
are disambiguated: `[pasted 42 lines #2]`.

### Cancellation recovery

Ctrl+C cancels the current LLM turn immediately. If the
model was mid-way through a tool-calling cycle, the model's
response (with tool calls) is already persisted to the session
database but some or all tool results may not be. Both
PydanticAI and upstream provider APIs (OpenAI, Anthropic)
reject conversations where a tool call has no matching result.

rbtr detects this on the next turn by scanning the entire
history for each `ToolCallPart` and checking whether a
matching `ToolReturnPart` exists anywhere. Any tool call
without a matching result gets a synthetic `(cancelled)`
return injected. This handles both cases:

- **No results saved** — all tool calls in the response are
  patched.
- **Partial results** — only the tool calls that were still
  in-flight when Ctrl+C fired are patched; completed results
  are preserved.

The synthetic results are persisted to the session database
immediately, so the repair is permanent — it won't re-trigger
on subsequent loads of the same session. A one-time warning
is shown:

```text
⚠ Previous turn was cancelled mid-tool-call (read_file, grep).
  Those tool results are lost — the model will continue without them.
```

The model sees the `(cancelled)` results and can decide
whether to retry the tool calls or proceed differently. No
manual intervention is needed.

## Usage display

After the first LLM response, the footer shows token usage and context
information:

```text
 owner/repo                                    claude/claude-sonnet-4-20250514
 PR #42 · feature-branch   |7|  12% of 200k  ↑ 24.3k  ↓ 1.2k  ↯ 18.0k  $0.0450
```

| Field     | Example | Description                                     |
| --------- | ------- | ----------------------------------------------- |
| `\|7\|`   |         | Messages in this conversation                   |
| `12%`     |         | Last request size as % of context window        |
| `of 200k` |         | Model's context window size                     |
| `↑ 24.3k` |         | Cumulative input tokens                         |
| `↓ 1.2k`  |         | Cumulative output tokens                        |
| `↯ 18.0k` |         | Cumulative cache-read tokens (hidden when zero) |
| `$0.0450` |         | Cumulative cost                                 |

**Colour signals help you manage context:**

- **Message count** — gray (≤ 25, fresh), yellow (26–50, consider `/new`),
  red (51+, very long).
- **Context %** — green (< 70%), yellow (70–89%), red (90%+).
- **Dimmed values** indicate the model didn't report pricing metadata
  (common with custom endpoints). Context window falls back to 128k
  and cost shows `$0.0000`.

`/new` resets all counters.

## Sessions

Every conversation is automatically saved to a local SQLite database
at `~/.config/rbtr/sessions.db`.

### How messages are stored

The database has a single `fragments` table. Each message is
stored as **1 + N rows**:

- **1 message row** — metadata (timestamps, model name, token
  counts, cost). Self-referencing (`message_id = id`),
  `fragment_index = 0`.
- **N content rows** — one per part (user text, assistant text,
  tool call, tool result, thinking, etc.), each serialised as
  JSON. Content rows point to the message row via `message_id`
  and have `fragment_index >= 1`.

Sessions are a `session_id` column, not a separate table.
Aggregates (message count, total cost) are computed via
`GROUP BY`.

Command and shell inputs (`/review`, `!ls`) are stored as
single self-referencing rows (no content fragments) so they
appear in input history.

#### Example: a simple turn

A user asks "list the files" and the model responds with a tool
call to `list_directory`, gets the result, then writes a text
answer. This produces 4 messages and ~10 rows:

```text
message row  "request"   fragment_index=0   ← user prompt metadata
  content    "user text" fragment_index=1   ← "list the files"

message row  "response"  fragment_index=0   ← model response metadata (tokens, cost)
  content    "tool call" fragment_index=1   ← list_directory({path: "."})

message row  "request"   fragment_index=0   ← tool result metadata
  content    "tool result" fragment_index=1 ← "src/ tests/ README.md ..."

message row  "response"  fragment_index=0   ← model response metadata
  content    "text"      fragment_index=1   ← "Here are the files: ..."
```

Each row carries denormalised metadata (`session_id`,
`model_name`, `repo_owner`, `repo_name`, `session_label`) for
efficient queries without joins.

### Streaming persistence

Model responses are persisted **as they stream in**, not
batch-saved after the turn:

1. An incomplete message row is inserted (`complete = 0`).
2. Each part is inserted as it starts streaming (`complete = 0`).
3. Each part is updated with the final data when it finishes.
4. The message row is marked `complete = 1` and cost is set.

Only complete rows (`complete = 1`) are returned by
`load_messages()`, so partially streamed responses are invisible.
User prompts and tool results are inserted complete immediately.

### Loading and reconstruction

`load_messages()` queries all non-compacted, complete fragments
for a session, groups them by `message_id`, merges each message
row with its content rows, and returns the reconstructed
conversation as a list — ready to pass to the model as history.

### Switching providers

The message format is **provider-agnostic**. A response from
Claude and one from GPT-4o have the same structure (text parts,
tool calls, etc.). History is preserved across `/model`
switches — only `/new` clears it.

Providers differ in how they encode tool exchanges, reasoning
metadata, and required message fields. PydanticAI translates
between formats, but edge cases in long or cross-provider
histories can produce structures that a provider rejects. rbtr
adds several repair layers on top of PydanticAI to keep sessions
usable across provider switches.

### History repair

LLM APIs require every tool call to have a matching tool result
(and vice versa), and require tool-call arguments to be valid
JSON. Providers also differ in wire format — tool-call ID
schemes, required fields, reasoning metadata — so history
produced by one provider can be structurally invalid for another.

PydanticAI's internal message format is provider-agnostic, but
providers embed provider-specific metadata that other providers
reject. rbtr adds repair layers on top to preserve cross-provider
session continuity.

#### Immutable history principle

Persisted messages are immutable. rbtr never modifies, deletes,
or injects messages into saved conversation history. All repairs
and adaptations are applied **transiently in memory** at load
time. The database always retains the original conversation
exactly as it happened.

This means:

- Switching to a provider that rejects thinking parts does not
  destroy the original thinking data. The next turn with a
  compatible provider uses the original `ThinkingPart`s.
- A session interrupted mid-tool-call retains the original
  dangling state. The synthetic `(cancelled)` tool returns
  exist only in the in-memory history passed to the provider.
- Any history transformation can be reviewed after the fact
  via incident records that describe what was changed and why.

When a repair or adaptation is applied, rbtr persists a
structured incident record (`LLM_HISTORY_REPAIR`) alongside
the conversation. Incident records are invisible to replay
(`load_messages` excludes them) but queryable for diagnostics.

#### On load

When a session is resumed, structural damage is repaired in
memory before the first API call. The database is not modified.

**Corrupt tool-call args.** A model can produce invalid JSON for
tool arguments during streaming (e.g. mixed XML/JSON fragments).
PydanticAI accepts any string at validation time, so the corrupt
data is saved to the database. The error only surfaces when the
provider adapter tries to parse the args. rbtr sets corrupt args
to `{}` in memory on every load — the message stays in history so
its matching tool results are not orphaned.

**Dangling tool calls.** Ctrl+C or a crash during tool execution
leaves tool calls with no matching result. Every provider rejects
unpaired tool calls. rbtr injects a synthetic tool result with
content `(cancelled)` for each unmatched call in memory. The
repair runs on every load and produces the same result — the
database retains the original dangling state.

#### On retry

When the first API call fails with a provider rejection, rbtr
retries once with simplified history. Both transforms below are
applied together in memory — the original messages remain in the
database unchanged. The next turn with a compatible provider uses
the original history.

**Thinking-part metadata.** Providers embed reasoning IDs in
thinking parts (`rs_*`, `reasoning_content`). A different
provider rejects them. rbtr converts thinking parts to plain text
wrapped in `<thinking>` tags — content preserved, metadata
stripped.

**Incompatible tool-exchange structure.** Each provider encodes
tool calls and results differently. Complex histories —
multi-turn chains, compaction boundaries, cross-provider
tool-call ID formats — can produce structures the target adapter
rejects. Some OpenAI-compatible endpoints also reject null
content on assistant messages that contain only tool calls, or
crash in the adapter on unexpected null values. rbtr converts
tool calls and results to plain text:
`[Repaired historical tool call -- tool_name(args)]`
and `[tool_name result]\n…`. All content survives as readable
text; only the machine-level pairing is removed.

The retry triggers on HTTP 400 errors matching format-rejection
patterns (unpaired tool calls, orphaned tool returns, missing
required fields, reasoning-ID mismatches), on `ValueError` from
malformed tool-call args that survived load-time repair, and on
`TypeError` from adapter crashes on unexpected null values.

#### On compaction

When history is split for context compaction (see
[Context compaction](#context-compaction)), two mechanisms prevent
the split from creating orphaned tool parts.

**Orphaned tool returns.** A tool result can end up in the kept
partition while its matching tool call was compacted away. rbtr
scans the kept partition for tool-result-only messages whose
tool-call ID has no match and moves them into the old partition.

**Split between call and result.** The split point can land
between a tool-call message and its immediately following
tool-result message. rbtr decrements the split index until the
boundary no longer separates a call/result pair.

### `/new` — starting fresh

`/new` clears the in-memory conversation (history, usage
counters) and generates a new session ID. The previous session's
data stays in the database — it can be listed with `/session` and
resumed with `/session resume`.

### `/session` command

```text
/session              List recent sessions (current repo)
/session all          List sessions across all repos
/session info         Show current session details
/session rename <n>   Rename the current session
/session resume <q>   Resume a session (ID prefix or label)
/session delete <id>  Delete a session by ID prefix
/session purge 7d     Delete sessions older than 7 days
```

**`/session rename`** changes the label on the current session.
Labels are set automatically when a review target is selected
(e.g. `acme/app — main → feature-x`), but you can override
them with any name.

**`/session resume`** accepts an ID prefix or a label substring
(case-insensitive).  ID prefix is tried first; if no match,
the label is searched.  When several sessions share a label the
most recent one is picked.

**`/session resume`** loads the target session's messages from the
database and switches the active session ID. The conversation
continues where it left off — the model sees the full history.
If the session had a review target (PR or branch), it's
automatically restored — rbtr re-runs `/review` to fetch fresh
metadata and rebuild the code index.
You can resume sessions from different repos or different models.

**`/session delete`** requires an exact ID prefix — no label
matching, to prevent accidental deletion.  Removes all fragments
for the session (cascading via foreign keys). You cannot delete
the active session — use `/new` first.

### Automatic behaviour

- **New session on startup.** Each `rbtr` invocation starts a fresh
  session, labelled with the current repo and branch. When you
  select a review target, the label updates to show the base and
  head branches (e.g. `acme/app — main → feature-x`).
- **Streaming persistence.** Parts are saved to the database as
  they arrive from the model, not batched after the turn.
- **Input history from the database.** Up/Down arrow browses input
  history across all sessions, deduplicated and sorted by recency.

### Pruning old sessions

Sessions accumulate over time. Use `/session purge` to clean up:

```text
/session purge 7d     Delete sessions older than 7 days
/session purge 2w     Delete sessions older than 2 weeks
/session purge 24h    Delete sessions older than 24 hours
```

Duration suffixes: `d` (days), `w` (weeks), `h` (hours). The
active session is never deleted. To remove a specific session,
use `/session delete <id>`.

### `/stats` command

```text
/stats                Current session statistics
/stats <id>           Stats for a specific session (prefix match)
/stats all            Aggregate stats across all sessions
```

Shows token usage (input, output, cache), cost, tool call
frequency, and — when the session has incidents — failure and
repair summaries.

#### Incident reporting

When an LLM call fails and rbtr auto-recovers, or when history
is manipulated to satisfy provider constraints, the event is
recorded as an **incident** in the session database (see
[History repair](#history-repair)). `/stats` surfaces these
when they exist:

```
  Failures (3)
    history_format                 2   recovered: 2
    overflow                       1   recovered: 1

  History repairs (5)
    repair_dangling                1   (cancelled_mid_tool_call)
    demote_thinking                2   (cross_provider_retry)
    flatten_tool_exchanges         2   (cross_provider_retry)

  Recovery rate       100%   3/3
```

**Failures** are grouped by kind (`history_format`, `overflow`,
`tool_args`, `type_error`, `effort_unsupported`) with
recovered/failed sub-counts. **History repairs** are grouped by
strategy with the reason that triggered them. The **recovery
rate** shows what percentage of failures were automatically
resolved.

Sessions with no incidents show no extra sections — the output
is identical to before.

## Context compaction

Long conversations accumulate tokens until the model's context
window fills up. rbtr compacts automatically — summarising older
messages while keeping recent turns intact.

### How compaction works

Compaction splits the conversation into **old** and **kept**
messages based on `keep_turns` (default 2). A _turn_ starts at a
user prompt and includes everything up to the next user prompt
(the model's responses, tool calls, tool results).

After splitting, any orphaned tool returns in **kept** — tool
results whose matching tool call is in **old** — are moved to
**old**. This prevents API errors from mismatched tool call IDs
after compaction.

The old messages are serialised to text and sent to the current
model with a summary prompt. The result:

- **Old rows** get `compacted_by` set to the summary's row ID.
  They become invisible to `load_messages()` but stay in the
  database for auditing. Cleaned up when the session is deleted
  (FK cascade).
- **Summary** — a single message containing the LLM-generated
  summary, prefixed with
  `[Context summary — earlier conversation was compacted]`.
  Timestamped to sort before the kept messages.
- **Kept messages** — untouched, stay as-is.

#### Example: compaction with `keep_turns = 2`

Before compaction (5 turns):

```text
turn 1:  user "set up the project"     → assistant "done, created package.json"
turn 2:  user "add authentication"     → assistant [tool calls] → "added auth middleware"
turn 3:  user "write tests for auth"   → assistant [tool calls] → "added 12 tests"
turn 4:  user "fix the failing test"   → assistant [tool calls] → "fixed assertion"
turn 5:  user "review the PR"          → assistant [reading files...] ← in progress
```

After compaction — turns 1–3 are summarised, turns 4–5 are kept:

```text
summary: "[Context summary] Set up project with package.json. Added auth
          middleware with JWT validation. Wrote 12 tests for auth module."
turn 4:  user "fix the failing test"   → assistant [tool calls] → "fixed assertion"
turn 5:  user "review the PR"          → assistant [reading files...] ← in progress
```

The summary is a plain text message with no provider-specific
parts, so it survives model switches.

#### What happens in the database

Before compaction, the fragments table has rows for all 5 turns
(message rows + content rows, `compacted_by = NULL`):

```text
id   message_id  kind              compacted_by  content
───  ──────────  ────              ────────────  ───────
m1   m1          request-message   NULL          ← "set up the project"
m1p  m1          user-prompt       NULL          "set up the project"
m2   m2          response-message  NULL          ← "done, created package.json"
m2p  m2          text              NULL          "done, created package.json"
...  (turns 2–3: same pattern)
m8   m8          request-message   NULL          ← "fix the failing test"
m8p  m8          user-prompt       NULL          "fix the failing test"
m9   m9          response-message  NULL          ← tool calls + "fixed assertion"
...  (turn 5)
```

After compaction, the old rows are marked and a summary is
inserted:

```text
id   message_id  kind              compacted_by  content
───  ──────────  ────              ────────────  ───────
m1   m1          request-message   S1            ← marked, invisible
m1p  m1          user-prompt       S1            ← marked, invisible
m2   m2          response-message  S1            ← marked, invisible
m2p  m2          text              S1            ← marked, invisible
...  (all turns 1–3 rows: compacted_by = S1)
S1   S1          request-message   NULL          ← NEW summary message
S1p  S1          user-prompt       NULL          "[Context summary] Set up project..."
m8   m8          request-message   NULL          ← turn 4, untouched
m8p  m8          user-prompt       NULL          "fix the failing test"
m9   m9          response-message  NULL          ← turn 4 response, untouched
...  (turn 5, untouched)
```

`load_messages()` filters `WHERE compacted_by IS NULL`, so it
returns: summary S1 → turn 4 → turn 5. The old rows stay in the
database until the session is deleted.

### When compaction triggers

Five paths, all using the same compaction logic:

**1. Post-turn** — After a successful LLM response, if context
usage exceeds `auto_compact_pct` (default 85%).

**2. Mid-turn** — During a multi-step tool-calling turn, after
each tool-call cycle. If the threshold is exceeded and the model
made tool calls (meaning more requests will follow), rbtr breaks
out of the agent loop, compacts, reloads history from the
database, and resumes the turn. The in-progress turn counts as
one of the `keep_turns` — so with `keep_turns = 2`, the current
turn and the previous turn are preserved, and everything older is
summarised. Mid-turn compaction fires at most once per turn.

**3. On overflow error** — When the API rejects a request with a
context-length error (HTTP 400, 413, etc.), rbtr compacts and
retries automatically.

**4. Manual** — `/compact` on demand, with optional extra
instructions for the summary:

```text
you: /compact
you: /compact Focus on the authentication changes
```

**5. Reset** — `/compact reset` undoes the latest compaction,
restoring the original messages to active context. The summary
message is deleted (its timestamp would interleave with restored
messages). Reset is only allowed if no messages were sent after
the compaction:

```text
you: /compact reset
Compaction reset — 42 fragments restored (26 active messages).
```

### Progress indicator

When compaction runs, two panels appear in the conversation
history:

```text
Compacting 42 messages …
Compacted 42 messages into ~1.2k tokens.
```

### Large conversations

When the messages to be summarised are too large to fit in a
single summary request, rbtr uses binary search to find the
largest prefix that fits, summarises that prefix, and pushes the
rest into the kept portion.

### Compaction settings

Settings in `config.toml` under `[compaction]`:

```toml
[compaction]
auto_compact_pct = 85       # trigger at this % of context window
keep_turns = 2              # recent turns to preserve
reserve_tokens = 16000      # tokens reserved for the summary response
summary_max_chars = 2000    # max chars per tool result in summary input
```

Token estimation uses `len(text) // 4` — no external tokenizer
dependency. After compaction the footer's context % stays at the
pre-compaction value until the next LLM call corrects it.

## Configuration

Two TOML files under `~/.config/rbtr/`:

### `config.toml` — preferences and endpoint URLs

```toml
model = "claude/claude-sonnet-4-20250514"

[endpoints.deepinfra]
base_url = "https://api.deepinfra.com/v1/openai"

[endpoints.ollama]
base_url = "http://localhost:11434/v1"
```

Config values can also be set via environment variables with
`RBTR_` prefix (e.g. `RBTR_MODEL`).

### `creds.toml` — tokens and API keys (0600 permissions)

```toml
github_token = "ghp_..."
openai_api_key = "sk-..."

[claude]
access_token = "..."
refresh_token = "..."
expires_at = 1739836725.0

[chatgpt]
access_token = "..."
refresh_token = "..."
expires_at = 1739836725.0
account_id = "..."

[endpoint_keys]
deepinfra = "..."
```

OAuth tokens (Claude, ChatGPT) are refreshed automatically when
they expire. You never need to edit `creds.toml` by hand — use
`/connect` instead.

## Prompt architecture

rbtr's instructions to the LLM are split into three templates,
each with a single responsibility:

| Template      | Scope           | Description                                    |
| ------------- | --------------- | ---------------------------------------------- |
| `system.md`   | Both agents     | Identity, authority, language style             |
| `review.md`   | Main agent only | Review context, principles, strategy, format    |
| `compact.md`  | Compact agent   | What to preserve/drop when summarising history  |

The main agent receives `system.md` + `review.md` on every
turn. The compaction agent (a separate, tool-less instance
that summarises older history) receives `system.md` +
`compact.md`. Both share the same identity and language rules.

### Customisation

You can customise the system prompt at three levels — user-wide
preferences, full replacement, and per-project instructions.

Loading order: built-in `system.md` (or `SYSTEM.md` override)
→ `APPEND_SYSTEM.md` → project instruction files.

### `APPEND_SYSTEM.md` — user-wide additions

Create `~/.config/rbtr/APPEND_SYSTEM.md` to append text to
the system prompt. Plain markdown, injected verbatim after the
built-in content. Applies to both the main and compaction
agents. Use this for personal review preferences, domain
context, or house style that applies to all repos:

```markdown
## My preferences

- Always check for missing error handling in HTTP handlers.
- Flag any use of `subprocess.run` without `check=True`.
- I prefer explicit `return None` over implicit returns.
```

### `SYSTEM.md` — system prompt replacement

Create `~/.config/rbtr/SYSTEM.md` to replace the built-in
`system.md` template. This replaces only the system prompt
(identity, authority, language) — review instructions
(`review.md`) and compaction instructions (`compact.md`) are
unaffected.

This is a Jinja template with the following variables:

| Variable               | Type  | Description                    |
| ---------------------- | ----- | ------------------------------ |
| `project_instructions` | `str` | Concatenated project files     |
| `append_system`        | `str` | Contents of `APPEND_SYSTEM.md` |

Minimal example:

```markdown
You are a code reviewer. Be direct.
{% if append_system %}
{{ append_system }}
{% endif %}
{% if project_instructions %}
{{ project_instructions }}
{% endif %}
```

### Project instructions

Project-specific rules are loaded from files in the repo root.
By default rbtr reads `AGENTS.md`. Configure the file list in
`config.toml`:

```toml
project_instructions = ["AGENTS.md"]
```

Multiple files are concatenated in list order:

```toml
project_instructions = ["AGENTS.md", "REVIEW.md", "docs/STYLE.md"]
```

Missing files are silently skipped. The concatenated content is
injected into the system prompt under a "Project instructions"
heading (in the built-in template) or via the
`project_instructions` template variable (in a custom
`SYSTEM.md`). Because the system prompt is shared, project
instructions apply to both the main and compaction agents.

Use project instruction files for coding standards, architecture
rules, review focus areas, or anything specific to the repo:

```markdown
# AGENTS.md

- Target Python 3.13+. Use modern features.
- All code must be type-annotated.
- No `Any` as a lazy type escape.
- Run `just check` after every change.
```

## Code index

When you start a review (`/review`), rbtr builds a structural index
of the repository in the background. The index gives the LLM tools
to search, navigate, and reason about the codebase — not just the
diff.

### What gets indexed

rbtr extracts **chunks** (functions, classes, methods, imports,
doc sections) from every file in the repo at the base commit, then
incrementally indexes the head commit. Each chunk records its name,
kind, file path, line range, and content.

Cross-file **edges** are inferred automatically:

- **Import edges** — structural (from tree-sitter import metadata)
  or text-search fallback for languages without an extractor.
- **Test edges** — `test_foo.py` → `foo.py` by naming convention
  and import analysis.
- **Doc edges** — markdown/RST sections that mention function or
  class names.

**Embeddings** are computed for semantic search using a local GGUF
model (bge-m3, quantized, runs on Metal/CPU — no API calls). The
structural index is usable immediately; embeddings fill in behind
it.

### Tools available to the LLM

rbtr gives the LLM 21 tools, conditionally hidden based on
what's available (repo only, or repo + index). Every tool that
reads state accepts a `ref` parameter — `"head"` (default),
`"base"`, or a raw commit SHA — so the LLM can inspect the
codebase at any point in time.

All paginated tools accept `offset` and a per-call limit
(`max_results`, `max_lines`, or `max_hits`) that defaults
to the configured cap. When output is truncated, a
`... limited (shown/total)` trailer tells the LLM how to
request the next page.

#### File tools (require repo)

Available as soon as a repository is connected. Read from the
git object store first; if a path is not found in git, fall
back to the local filesystem (covers `.rbtr/notes/` files
and other untracked files). Filesystem fallback respects
`.gitignore` and the `include`/`extend_exclude` config —
ignored paths (`.mypy_cache`, `node_modules`, etc.) are
never exposed to the LLM.

**`read_file(path, ref?, offset?, max_lines?)`** — Read
file contents with line numbers. Paginate with `offset`.
Binary files rejected.

**`grep(search, path?, ref?, offset?, max_hits?,
context_lines?)`** — Case-insensitive substring search. Three
modes: exact file (`path="src/app.py"`), directory prefix
(`path="src/"`), or repo-wide (no `path`). Matches shown
with surrounding context; nearby matches merged.

**`list_files(path?, ref?, offset?, max_results?)`** — List
files in the repo or a subdirectory. Sorted alphabetically.

**`changed_files(offset?, max_results?)`** — List files
changed between base and head (added, modified, deleted).
Starting point for review.

#### Search tools (require index)

Available once the code index finishes building. Always
search the head snapshot.

**`search(query, offset?, max_results?)`** — Unified code
search. Finds symbols by name, keywords, or concepts.
Fuses three signals internally — name matching, BM25
keyword search, and semantic similarity — into a single
ranked result list. Works for exact identifiers
(`IndexStore`), keyword queries (`retry timeout`), and
natural-language concepts (`how does auth work`).

The query is automatically classified as an identifier,
concept, or pattern, and fusion weights are adjusted
accordingly. Identifier queries favour name matching;
concept queries favour BM25 and semantic similarity.
Results include kind-boost (classes rank above imports)
and file-category penalty (source ranks above tests).

#### Index read tools (require index)

**`read_symbol(name, ref?)`** — Read the full source of a
symbol. Looks up by name (case-insensitive substring),
prefers code symbols (functions, classes, methods) over
tests or docs. Returns kind, scope, file path, line range,
and content.

**`list_symbols(path, ref?, offset?, max_results?)`** — List
the symbols in a file — structural table of contents with
line number, kind, scope, and name.

#### Dependency graph (require index)

**`find_references(name, kind?, ref?, offset?,
max_results?)`** — Find all symbols that reference a given
symbol via the dependency graph. Returns each referencing
symbol labelled by edge type. Filter by `kind`: `imports`,
`calls`, `inherits`, `tests`, `documents`, `configures`.

#### Git tools (require repo)

**`diff(path?, ref?, offset?, max_lines?)`** — Unified text
diff. Three modes: base→head (default), single-ref (one
commit), or range (`ref1..ref2`). Optional `path` filters
to a single file. Paginate with `offset`.

**`commit_log(offset?, max_results?)`** — Commit log between
base and head — SHA, author, and first line of commit
message.

**`changed_symbols(offset?, max_lines?)`** — List symbols
changed between base and head using the code index. Shows
added, removed, modified symbols plus stale docs, missing
tests, and broken edges.

#### Edit (always available)

**`edit(path, new_text, old_text?)`** — Create or modify
files matching the `editable_include` glob patterns
(default: `.rbtr/notes/*`, `.rbtr/AGENTS.md`). When
`old_text` is empty, creates the file or appends; when
set, replaces the exact match. Editable files are readable
by `read_file`, `grep`, and `list_files` via the
filesystem fallback.

#### PR discussion (require PR)

**`get_pr_discussion(offset?, max_results?)`** — Read
all existing discussion on the current PR. Returns reviews,
inline comments, and general comments sorted chronologically.
Includes bot comments (CI, linters) and emoji reactions.
Cached per session — fetched once, then read from memory.

#### Draft management (require PR)

**`add_draft_comment(path, anchor, body, suggestion?, ref?)`** —
Add an inline comment to the review draft. The `anchor` is
an exact code snippet from the file; the comment is placed on
its last line. Persisted to `.rbtr/drafts/<pr>.yaml`
immediately.

**`edit_draft_comment(path, comment, body?, suggestion?)`** —
Edit an existing comment. `comment` is a body substring
identifying which comment to edit.

**`remove_draft_comment(path, comment)`** — Remove a
comment by body substring.

**`set_draft_summary(summary)`** — Set the top-level
review body that appears at the top of the GitHub review.

**`read_draft()`** — Read the current draft (summary +
all comments with status indicators).

#### The `ref` parameter

Tools that accept `ref` return the **state of the codebase at
that snapshot**, not the changes introduced by it:

- `"head"` (default) — the PR head / feature branch tip.
- `"base"` — the base branch (e.g. `main`).
- Any raw commit SHA or git ref (e.g. `"abc1234"`, `"v2.1.0"`).

Change tools (`diff`, `changed_symbols`, `changed_files`) show
changes _between_ base and head — they don't accept `ref` in
the same sense.

#### Git-first with filesystem fallback

File tools (`read_file`, `grep`, `list_files`) look up paths
in the git object store first. If a path or prefix is not found
in git, they fall back to the local filesystem. This means:

- Repository files are always read from the git snapshot (at
  `ref`), ensuring reproducible reads.
- Workspace files (`.rbtr/notes/` files, untracked files)
  are accessible without committing them to git.
- When a prefix has files in both git and the filesystem, git
  wins — the filesystem is only tried when git has nothing.

The filesystem fallback applies the same three-layer filter as
the indexer: `include` globs override `.gitignore`, then
`.gitignore` patterns are applied, then `extend_exclude` globs.
This prevents build artifacts, caches, and database files from
leaking into tool results.

#### Tool availability

Tools are conditionally hidden from the LLM based on session
state — the LLM only sees tools it can actually use:

- **Repo tools** appear when a repository is connected
  (`/review`).
- **Index tools** appear when the code index is ready (built
  automatically in the background after `/review`).
- The LLM never sees a tool it can't call — no confusing
  error messages from missing prerequisites.

### Progress indicator

The footer shows indexing progress:

```text
⟳ parsing 42/177      (extracting chunks)
⟳ embedding 85/380    (computing vectors)
● 1.2k                (ready — 1,200 chunks indexed)
```

The review proceeds immediately — you don't have to wait for
indexing to finish.

### `/index` command

| Subcommand                   | Description                              |
| ---------------------------- | ---------------------------------------- |
| `/index`                     | Show index status (chunks, edges, size)  |
| `/index clear`               | Delete the index database                |
| `/index rebuild`             | Clear and re-index from scratch          |
| `/index prune`               | Remove orphan chunks not in any snapshot |
| `/index model`               | Show current embedding model             |
| `/index model <id>`          | Switch embedding model and re-embed      |
| `/index search <query>`      | Search the index and show ranked results |
| `/index search-diag <query>` | Search with full signal breakdown table  |

#### `/index status`

`/index status` (or just `/index`) shows four sections:

1. **Snapshots** — side-by-side comparison of base and head:
   file count, symbol count, and edge count per snapshot.
   Shows `-` for a snapshot that hasn't been indexed yet.
2. **Chunks** — per-kind breakdown (function, class, method,
   import, doc, etc.) of the head snapshot, with embedding
   coverage. Falls back to base when head isn't indexed yet.
3. **Edges** — per-kind breakdown (imports, tests, docs) of
   the same snapshot.
4. **Commits** — for each commit in `base..head`, the number
   of files it changed and how many indexed symbols live in
   those files at head. Helps identify high-impact commits vs
   trivial ones. Uses `commit_log_between` for the commit
   list, then diffs each commit against its first parent.
   Capped at 20 commits.

### Index configuration

Settings in `config.toml` under `[index]`:

```toml
[index]
enabled = true                                          # master toggle
db_dir = ".rbtr/index"                                  # DuckDB storage (relative to repo)
model_cache_dir = "~/.config/rbtr/models"               # GGUF model cache (shared)
embedding_model = "gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
max_file_size = 524288                                  # skip files > 512 KiB
chunk_lines = 50                                        # lines per plaintext chunk
chunk_overlap = 5                                       # overlap between chunks
embedding_batch_size = 32                               # chunks per embedding call
include = [".rbtr/notes/*"]                              # force-include globs (override .gitignore)
extend_exclude = [".rbtr/index"]                        # exclude globs (on top of .gitignore)
```

The `include` and `extend_exclude` settings apply to both the
indexer and file tool filesystem fallback. Literal patterns also
match children — `".rbtr/index"` excludes `.rbtr/index/data.db`.

The index is persistent — subsequent `/review` runs on the same
repo skip unchanged files (keyed by git blob SHA).

### Graceful degradation

- **No grammar installed** for a language → falls back to
  line-based plaintext chunking.
- **No embedding model** (missing GGUF, GPU init failure) →
  structural index works, semantic search signal is skipped
  (weight redistributed to name and keyword channels).
- **Slow indexing** → review starts immediately, index catches
  up in a background thread.

## Review draft

When reviewing a GitHub PR, rbtr helps you build a structured
review that can be posted back to GitHub. The LLM builds the
draft incrementally using tool calls; you inspect it locally,
sync it with GitHub, and post it when ready.

### Workflow

1. **Select a PR** — `/review 42` fetches the PR and pulls
   any existing pending review from GitHub into the local
   draft.

2. **Review with the LLM** — as you discuss the code, the
   LLM uses `add_draft_comment` and `set_draft_summary`
   to build a draft. Each change persists immediately to
   `.rbtr/drafts/42.yaml`.

3. **Inspect the draft** — `/draft` shows the current state:
   summary, numbered inline comments with sync status
   indicators, and suggestion markers.

4. **Sync** — `/draft sync` does a bidirectional sync with
   GitHub: pulls remote changes, merges them into the local
   draft, then pushes the result back as a pending review.

5. **Post** — `/draft post` submits the review to GitHub.
   `/draft post approve` and `/draft post request_changes`
   set the review event type (default is `COMMENT`).

### How the draft is stored

The draft lives at `.rbtr/drafts/<pr>.yaml` — a plain YAML
file updated atomically on every mutation. It's
human-readable and hand-editable:

```yaml
summary: Good PR overall, two issues.
github_review_id: 12345
summary_hash: a1b2c3d4e5f6g7h8
comments:
  - path: src/client.py
    line: 42
    body: "**blocker:** Retry without backoff."
    github_id: 98765
    comment_hash: f9e8d7c6b5a43210
  - path: src/config.py
    line: 8
    body: "**nit:** Unused import."
```

Top-level fields:

- **`summary`** — the review body (markdown).
- **`github_review_id`** — the PENDING review ID on
  GitHub. Absent until the draft has been synced.
- **`summary_hash`** — hash of the summary, frozen at last
  sync. Only updated when syncing, never on local edits.

Each comment has:

- **`path`, `line`, `body`, `suggestion`** — the review
  content. `suggestion` is optional replacement code
  (posted as a GitHub suggestion block).
- **`github_id`** — GitHub's comment ID, set after the
  comment has been pushed to or pulled from GitHub.
  Absent for locally-created comments that haven't been
  synced yet.
- **`comment_hash`** — hash of this comment's content,
  frozen at last sync. Only updated when syncing, never
  on local edits. Absent for new comments.

Hashes are short (16-char hex) SHA-256 digests. Comment
hashes cover `path`, `line`, `body`, and `suggestion`.
Because hashes are only written during sync — never when
the LLM or you edit a comment — any change to the content
makes the live hash diverge from the stored one, which is
how rbtr detects local modifications.

### Sync with GitHub

`/draft sync` performs a bidirectional sync between the local
draft and the user's pending review on GitHub.

#### GitHub API constraints

Two hard API limitations shape the entire sync design:

**1. No individual pending-comment updates.** You cannot
PATCH a single pending review comment — the endpoint
returns 404. You CAN individually DELETE a pending
comment, but the only way to _modify_ a pending review is
to delete the entire review and recreate it.

rbtr works within this constraint: every push deletes the
old pending review and creates a new one. To track changes
across this delete-and-recreate cycle, each comment's
`github_id` is recorded locally and re-established after
each push by re-fetching the new review's comments.

**2. No modern line data on the per-review endpoint.**
When writing reviews, rbtr sends the modern `line` + `side`
parameters (e.g. `line: 163, side: "RIGHT"`). GitHub
accepts these correctly.

However, the endpoint that reads review comments back
(`GET /repos/{o}/{r}/pulls/{n}/reviews/{id}/comments`)
returns `line: null`, `side: null`, and `subject_type: null`
for ALL review comments — pending AND submitted. The only
line-related data returned is the deprecated `position`
field (a 1-based offset into the diff hunk) and the
`diff_hunk` string (the hunk header through the commented
line).

The per-PR endpoint (`GET /pulls/{n}/comments`) does return
modern fields, but excludes pending comments entirely. The
individual comment endpoint (`GET /pulls/comments/{id}`)
returns 404 for pending comments.

rbtr works around this by walking the `diff_hunk` to
recover `(line, side)` from the deprecated `position`. The
conversion is deterministic — the same `diff_hunk` always
maps to the same file line number. This logic lives in
`_walk_hunk` / `_position_to_line` / `_line_to_position`
in `client.py`.

> **Verified 2026-02-27** with direct `httpx` calls using
> `Accept: application/vnd.github+json` and
> `X-GitHub-Api-Version: 2022-11-28`. Both pending and
> submitted reviews show `line: null` on the per-review
> endpoint. If GitHub fixes this, the `data.get("line")`
> path in `get_pending_review` already takes priority — the
> hunk-walking fallback only fires when `line` is null.

#### Matching: how local and remote comments are paired

When remote comments are fetched from GitHub, rbtr needs to
figure out which remote comment corresponds to which local
comment. This is done in two tiers:

**Tier 1 — `github_id` (exact match).** If a local
comment has a `github_id` from a previous sync, and a
remote comment has the same ID, they're the same comment.
This is the primary matching mechanism and handles edits,
deletions, and unchanged comments reliably.

**Tier 2 — `(path, line, formatted_body)` (content match).**
For local comments without a `github_id` (newly created by
the LLM, never synced), rbtr matches by exact content
against unmatched remote comments. This pairs them 1:1 —
if multiple remotes have identical content, none are
matched (ambiguity is not guessed at).

After both tiers:

- **Unmatched remote comments** are imported as new local
  comments. You'll see a warning:
  `"New remote comment imported: path:line"`.
- **Local comments with a stale `github_id`** (present in
  the sync snapshot but absent from the remote) were deleted
  on GitHub. They're removed locally with a warning:
  `"Comment on path:line was deleted on GitHub."`.
- **Local comments without a `github_id` and no content
  match** are kept as-is — they're new comments that will
  be included in the next push.
- **Tombstoned comments** (synced comments whose body was
  cleared locally via `remove_draft_comment`) are excluded
  from the push and dropped from the draft after sync.
  This prevents a deleted comment from being re-imported
  on the next pull. `/draft` shows them with a `✗`
  indicator and a note that they'll be deleted on next
  sync.

#### Three-way merge: detecting edits and conflicts

After matching, rbtr uses the synced content hash as a
common ancestor to detect what changed on each side since
the last sync. Each comment's current content is hashed
and compared to the stored hash:

| Synced | Local | Remote | Outcome                                |
| ------ | ----- | ------ | -------------------------------------- |
| A      | A     | A      | No change                              |
| A      | A     | B      | Remote edit → accept                   |
| A      | B     | A      | Local edit → keep, push on next sync   |
| A      | B     | B      | Both changed identically → no conflict |
| A      | B     | C      | **Conflict** → keep local, warn user   |

Conflicts always resolve in favour of the local draft — the
user chose to edit locally, so that takes priority. The
warning includes a preview of the remote body so you can
decide whether to incorporate it manually.

The same logic applies to the review summary (tracked via
`summary_hash` on the draft).

#### Sync protocol (step by step)

**Pull** (also runs automatically on `/review <n>`):

1. Fetch the user's PENDING review and its comments.
   Remote comments arrive with `line: null` — recover
   `(line, side)` from `diff_hunk` via `_position_to_line`.
2. Match remote comments to local using tiers 1 and 2.
3. Reconcile each matched pair (accept remote edits,
   detect conflicts).
4. Import unmatched remote comments, remove remote
   deletions.
5. Stamp `comment_hash` on each comment and `summary_hash`
   on the draft.

**Push** (runs as part of `/draft sync`):

1. Translate stale comments — if a comment's `commit_id`
   differs from the current PR head, translate its line
   number using diff hunk arithmetic. Comments targeting
   deleted lines are skipped with a warning.
2. Validate all comments against the current diff — skip
   comments whose `(path, line)` is no longer in a diff
   hunk.
3. Delete the existing PENDING review (if any).
4. Create a new PENDING review with valid comments, sending
   `line` + `side` + `commit_id` per the modern API.
5. Re-fetch the new review's comments to learn their
   `github_id`s (they change on every recreate). Again
   recover `(line, side)` from `diff_hunk`.
6. Match the returned comments to local by content
   (tier 2 — we just wrote them, so content is exact).
7. Store the new `github_id`s on local comments.
8. Merge pushed and skipped-stale comments back into the
   local draft.
9. Stamp `comment_hash` on each comment, set
   `github_review_id` and `summary_hash` on the draft.
10. Save the draft to disk.

#### Example: sync with a remote edit

```text
# Initial state: local and GitHub are in sync.
# Local comment (github_id=100, comment_hash=abc): "Fix the null check."

# Someone edits the comment on GitHub to "Fix the null check (line 42)."
# Meanwhile, local is unchanged (body still matches comment_hash).

you: /draft sync
Pulling remote pending review…
Pushing draft (1 comment)…
Draft synced (1 comment).

# Result: local body updated to "Fix the null check (line 42)."
# comment_hash updated to match the new content.
```

#### Example: conflict

```text
# comment_hash represents "Original text."
# Local body was edited to: "Better explanation."
# Remote body was edited to: "Different rewrite."

you: /draft sync
⚠ Conflict on src/client.py:42 — keeping local. Remote was: Different rewrite.
Pushing draft (1 comment)…
Draft synced (1 comment).

# Result: local "Better explanation." is pushed to GitHub.
```

#### Deleting synced comments locally

Simply removing a synced comment from the draft would cause
the next pull to re-import it from GitHub (it still exists
in the remote pending review). rbtr solves this with
_tombstones_.

When you ask the LLM to remove a comment that has already
been synced (`github_id` is set), `remove_draft_comment`
does not delete the comment outright. Instead it clears
the `body` and `suggestion` fields while keeping the
`github_id` and `comment_hash`. The comment stays in the
draft as a tombstone.

`/draft` shows tombstoned comments with a `✗` indicator:

```text
you: /draft
Summary: Good PR overall.
2 comments (1 pending deletion):
  ✓ 1. src/client.py:42 — **blocker:** Retry without backoff.
  ✗ 2. src/config.py:8 — (will be deleted on next sync)
```

On the next `/draft sync`:

1. **Pull** — the tombstone matches the remote comment by
   `github_id` (tier 1). Three-way merge sees the local
   side as dirty (body changed from the original to empty)
   and keeps the tombstone.
2. **Push** — tombstones are excluded from the pushed
   comments. The new pending review is created without
   the deleted comment.
3. **Save** — tombstones are dropped from the saved draft.
   The comment is gone from both local and remote.

Comments that have never been synced (no `github_id`) are
removed immediately — no tombstone needed since there is
nothing to re-import.

### Sync status indicators

`/draft` shows a status indicator next to each comment:

```text
you: /draft
Summary: Good PR overall.
2 comments:
  ✓ 1. src/client.py:42 — **blocker:** Retry without backoff.
  ★ 2. src/config.py:8 — **nit:** Unused import.
```

| Indicator | Meaning                                        |
| --------- | ---------------------------------------------- |
| `✓`       | **Synced** — matches the last-pushed snapshot  |
| `✎`       | **Modified** — changed locally since last sync |
| `★`       | **New** — never synced to GitHub               |
| `✗`       | **Deleted** — will be removed on next sync     |

### Draft commands

| Subcommand            | Description                           |
| --------------------- | ------------------------------------- |
| `/draft`              | Show the current draft with status    |
| `/draft sync`         | Bidirectional sync with GitHub        |
| `/draft post [event]` | Submit review to GitHub               |
| `/draft clear`        | Delete local draft and remote pending |

Tab completes subcommands and event types.

### Posting

`/draft post` submits the review to GitHub as a final,
visible review. Before posting it pulls the remote state
one last time to check for unsynced comments — if the
remote pending review has comments that aren't in your
local draft, the post is refused with a message to run
`/draft sync` first.

Any existing pending review is deleted before the
submitted review is created, so you don't end up with
both a pending and a submitted review.

Event types:

```text
/draft post                    # COMMENT (default)
/draft post approve            # APPROVE
/draft post request_changes    # REQUEST_CHANGES
```

After posting, the local draft file is deleted
automatically.

### Safety

- **Unsynced guard** — `/draft post` refuses if the remote
  pending review has comments not in your local draft. Run
  `/draft sync` first.
- **Atomic posting** — all comments are submitted in a
  single `create_review` API call. No partial reviews.
- **Crash-safe** — the draft is a TOML file on disk,
  updated on every mutation. If rbtr crashes, the draft
  survives. If the crash happens mid-sync (after deleting
  the old review but before creating the new one), the next
  sync detects the stale `review_id` (404) and recovers
  gracefully.
- **Human-editable** — `.rbtr/drafts/42.yaml` is plain
  YAML. You can edit it by hand — add comments,
  change bodies, or clear `comment_hash` / `summary_hash`
  to force a full re-sync. Sync state uses content
  hashes (not duplicated content), so editing a comment
  body is automatically detected as a local change.
- **Draft cleanup** — after a successful post, the local
  file is deleted automatically.

### Position ↔ line round-trip (GitHub API workaround)

The write path sends `line` + `side` (modern API). The
read path receives `position` + `diff_hunk` (deprecated
API). The conversion between them is deterministic:

```text
Write:  line=172, side=RIGHT  →  GitHub stores the comment
Read:   position=4, diff_hunk="@@ -169,6 +169,7 @@\n ..."  →  _position_to_line → (172, RIGHT)
```

`_walk_hunk(diff_hunk)` is the single source of truth. It
yields `(position, line, side)` for each diff line by
walking the hunk header's start-line counters and
classifying each line by its `+`/`-`/`` prefix:

- `+` line → `(new_line, "RIGHT")`, increment `new_line`
- `-` line → `(old_line, "LEFT")`, increment `old_line`
- context → `(new_line, "RIGHT")`, increment both
- `\` (no newline marker) → skip

`_position_to_line(diff_hunk)` returns the last yielded
triple (the commented line is always last in the hunk
GitHub returns). `_line_to_position(diff_hunk, line, side)`
finds the first matching triple and returns its position.

The content hash (`_comment_hash`) includes `line` because
the conversion is deterministic — the same `position`
always maps to the same `line`. `side` and `commit_id` are
excluded from the hash (resolution metadata not visible in
the GitHub UI).

### GitHub suggestions

When the LLM provides a `suggestion` parameter in
`add_draft_comment`, it's posted as a GitHub suggestion
block:

````markdown
Use exponential backoff.

```suggestion
time.sleep(2 ** attempt)
```
````

The author can apply suggestions with one click in the
GitHub UI.

Suggestion blocks are parsed back out when pulling from
GitHub — `body` and `suggestion` stay as separate fields
locally, even after a round-trip through the API.

## Tool-call limits

Each turn has a limit on how many tool calls the LLM can make
(default 25). This prevents runaway loops where the model keeps
searching without producing useful output.

When the limit is reached, rbtr does **not** error out. Instead
it asks the model to summarize what it accomplished and what
remains, so you can decide whether to continue:

```text
claude/claude-sonnet-4-20250514: I've reviewed 8 of the 12 changed files
so far. I found two issues in the retry logic and one missing null check.
The remaining files to review are: config.py, utils.py, client.py, and
the test file. Would you like me to continue?

you: yes, continue with the remaining files
```

The limit is configurable in `config.toml`:

```toml
[tools]
max_requests_per_turn = 25                                # default
notes_dir = ".rbtr/notes"                                 # hint for prompts
drafts_dir = ".rbtr/drafts"                               # draft YAML files
editable_include = [".rbtr/notes/*", ".rbtr/AGENTS.md"]   # edit tool globs
```

The `edit` tool can create and modify files matching the
`editable_include` glob patterns. `notes_dir` is a hint
used in prompts — editability is controlled solely by
`editable_include`. Draft files live in `drafts_dir` and
are managed exclusively by the draft tools.

If you change `editable_include`, update `index.include` to
match so the new paths are indexed:

```toml
[index]
include = [".rbtr/my-notes/*"]

[tools]
editable_include = [".rbtr/my-notes/*"]
```

## Development

```bash
uv sync        # Install dependencies
just check     # Lint + typecheck + test
just fmt       # Auto-fix and format
```

### Provider architecture

rbtr's provider layer handles authentication, credential
persistence, model construction, effort mapping, and
pricing — everything needed so that `/connect` + `/model`
is all it takes to use any supported service. The actual LLM
communication (streaming, tool calling, message formatting)
is [PydanticAI](https://ai.pydantic.dev/).

Each provider is a thin wrapper that knows how to
authenticate with its service and build the right PydanticAI
model instance. Every provider satisfies the same `Provider`
protocol — `is_connected`, `list_models`, `build_model`,
`model_settings`, `context_window` — so the engine never
deals with provider-specific logic. Custom endpoints
(`/connect endpoint`) satisfy the same protocol, so they
work identically to builtin providers.

**API-key providers** (OpenAI, Fireworks, OpenRouter) store
keys in `~/.config/rbtr/creds.toml`. Keys can also be set
via environment variables (`OPENAI_API_KEY`,
`FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`) — pydantic-settings
picks them up automatically.

**OAuth providers** (Claude, ChatGPT, Google) use PKCE with
a localhost callback. Tokens are stored in `creds.toml` and
refreshed automatically when they expire.

### Thinking effort

rbtr exposes a unified `thinking_effort` config (`low`/`medium`/
`high`/`max`) that maps to provider-specific settings. Users cycle
it with Shift+Tab; the footer shows the current level.

Each provider implements `model_settings(model_id, model, effort)`
from the `Provider` protocol. This returns a `ModelSettings`
subclass with the provider-specific effort parameter (e.g.
`AnthropicModelSettings(thinking=...)`,
`OpenAIModelSettings(reasoning=...)`) or `None` if the model
doesn't support thinking.

The engine calls `providers.model_settings(model_name, model,
effort)` — dispatch is handled by `_resolve`, same as
`build_model`. The caller (`engine/llm.py`) sets
`session.effort_supported` based on whether the provider
returned settings or `None`. If the model doesn't support
thinking, the footer shows `∴ off` in red.

### Token usage and pricing

Usage tracking lives in `engine/llm.py` → `_record_usage()`. After
each agent run it extracts:

- **Token counts** from pydantic-ai's `RunUsage` (input, output,
  cache read/write). These work for all providers automatically.

- **Cost and context window** from `ModelResponse.cost()`, which
  calls the `genai-prices` library internally. This works out of
  the box for models in the `genai-prices` database. Custom
  endpoints typically don't report pricing — the footer dims the
  cost and shows a fallback context window.

No rbtr code needs to change for new models to get usage tracking.
If `genai-prices` knows the model, cost appears automatically. If
not, tokens still display but cost shows as `$0.0000` (dimmed).

### Tool registration

Agent tools are defined in `engine/tools.py` using `@agent.tool`
decorators. Each tool receives `RunContext[AgentDeps]` and reads
session state (repo, index, review target). Tools are conditionally
hidden via `prepare` functions when their prerequisites aren't met
(e.g. index tools hidden when no index is loaded).

Tool call results appear in independent purple panels in the TUI.
Output is truncated to `tui.tool_max_lines` (default 15) with
Ctrl+O to expand.

Each turn is limited to `tools.max_requests_per_turn` model
requests (default 25). When the limit is hit, `_stream_agent`
catches `UsageLimitExceeded`, preserves accumulated messages,
and fires `_stream_summary` — a tool-free single-request
followup that asks the model to summarize progress so the user
can decide whether to continue.

### Language support

rbtr is language-agnostic. Language-specific analysis (tree-sitter
grammars, scope queries, import extraction) is provided by plugins
under `src/rbtr/plugins/`. Adding a language means adding a plugin
file — see `AGENTS.md` for the plugin contract.

### Index architecture

The code index lives in `src/rbtr/index/`:

```text
index/
├── git.py           file listing + diffing via pygit2
├── treesitter.py    language-agnostic tree-sitter extraction
├── chunks.py        plaintext and markdown chunking
├── edges.py         import, test, and doc edge inference
├── embeddings.py    GGUF embedding via llama-cpp-python
├── store.py         DuckDB storage (IndexStore)
├── search.py        score fusion, query classification, ranking
├── tokenise.py      code-aware tokenisation (camelCase, snake_case)
├── orchestrator.py  build_index, update_index, compute_diff
├── models.py        Chunk, Edge, IndexStats, enums
├── arrow.py         PyArrow table builders for bulk insert
├── languages.py     thin delegation to plugin manager
└── sql/             19 SQL files (sqlfluff-linted, DuckDB dialect)
```

**Data flow:** `git.py` lists files → `treesitter.py` or
`chunks.py` extracts chunks → `store.py` bulk-inserts via PyArrow
→ `edges.py` infers cross-file edges → `embeddings.py` computes
vectors → `store.py` batch-updates embeddings.

**Storage:** One DuckDB file per repo (`.rbtr/index/`). Three
tables: `file_snapshots` (commit→file→blob mapping), `chunks`
(extracted symbols with optional embeddings), `edges` (cross-file
relationships). All SQL is in separate `.sql` files — no inline
queries.

**Engine integration:** `engine/indexing.py` runs indexing in a
daemon thread, communicating via events (`IndexStarted`,
`IndexProgress`, `IndexReady`). `engine/tools.py` exposes index
data to the LLM via tool calls, conditionally hidden when no
index is loaded.

### Git reference handling

rbtr never modifies your working tree or local branches.
All reads go through the git object store.

When you select a PR, rbtr fetches the PR head and the
base branch from origin so that diffs, commit logs, and
changed-file lists reflect the actual PR scope — not stale
local refs. For PRs, the exact base and head SHAs come from
the GitHub API, so a local `main` that is behind
`origin/main` cannot pollute the results. For local branch
reviews, branch names are used directly.

Commit logs use `git log base..head` semantics — only
commits reachable from head but not from base. This is
correct even with merge histories (e.g. merging `main` into
a feature branch). Review comment validation diffs from the
merge base of the two refs, matching GitHub's three-dot
diff.

### Adding a language

1. Create `src/rbtr/plugins/<language>.py`.
2. Implement the plugin hooks: `detect_language`, `get_grammar`,
   `get_query`, optionally `get_import_extractor` and
   `get_scope_types`.
3. Register via the `rbtr.languages` pluggy entry point.

See existing plugins (e.g. `plugins/python.py`, `plugins/go.py`)
for examples. The `defaults.py` module registers grammar-only and
detection-only languages in bulk.

### Benchmarking

```bash
just bench                            # quick benchmark (current repo)
just bench -- /path/to/repo main      # custom repo
just bench -- . main feature          # with incremental update
just bench-scalene -- /path/to/repo   # line-level CPU + memory profiling
just scalene-view                     # view last scalene profile
```

Detailed results and optimization notes are in `PROFILING.md`.

### Search quality

The unified search system fuses three signals — name matching,
BM25 keyword search, and semantic (embedding) similarity — with
post-fusion adjustments for chunk kind and file category. Two
scripts measure and tune search quality:

```bash
just eval-search                      # evaluate against curated queries
just tune-search                      # grid-search fusion weights
just tune-search -- --step 0.05       # finer resolution (400 combos)
just bench-search                     # replay real queries (current dir)
just bench-search -- /path/to/repo    # replay for a specific repo
```

**`scripts/eval_search.py`** — Runs 24+ curated queries against
the rbtr repo, measuring recall@1, recall@5, and MRR across
three backends (name, BM25, unified). Queries are grouped by
the technique they test (tokenisation, IDF, kind scoring, file
category, name matching, query understanding, structural
signals). Results are tracked in `TODO-search.md`.

**`scripts/tune_search.py`** — Grid-searches over fusion weight
combinations for each query kind (identifier / concept).
Precomputes all channel scores once, then sweeps in-memory —
runs in ~1s. Reports the top 10 weight combos and the current
settings for comparison.

**`scripts/bench_search.py`** — Mines real search queries from
the session history database (`~/.config/rbtr/sessions.db`).
Pass a repo path (defaults to current directory), and the
script filters to events matching that repo by remote URL.
Extracts search→read pairs (search followed by `read_symbol`),
detects retry chains, classifies queries, and replays paired
queries through the current search pipeline. Reports R@1, R@5,
MRR, and per-query signal breakdowns for misranked results.

`eval_search` and `tune_search` are rbtr-specific — they
validate the repo identity via `pyproject.toml` before running.
`bench_search` works with any repo that has session history.

#### Search architecture

Search ranking is implemented in `src/rbtr/index/search.py`
(pure scoring functions) and orchestrated by
`IndexStore.search()` in `store.py`:

1. **Query classification** — `classify_query()` routes each
   query to identifier, concept, or pattern weights.
2. **Three channels** — BM25 on pre-tokenised content, cosine
   similarity on embeddings (when available), and name matching
   with token-level support.
3. **Score fusion** — min-max normalisation per channel, then
   weighted combination. Post-fusion multipliers for chunk kind
   (class=1.5×, import=0.3×) and file category (source=1.0×,
   test=0.5×).
4. **Code-aware tokenisation** — `tokenise_code()` splits
   camelCase/snake_case identifiers and emits both compound and
   parts. Applied at index time and query time.

All scoring helpers are pure functions tested in isolation (99
tests in `test_search.py`). The tokeniser has 140+ tests across
10 language conventions and keyword preservation.

## License

MIT
