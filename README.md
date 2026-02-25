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

| Provider | Auth               | Command                                |
| -------- | ------------------ | -------------------------------------- |
| Claude   | OAuth (Pro/Max)    | `/connect claude`                      |
| ChatGPT  | OAuth (Plus/Pro)   | `/connect chatgpt`                     |
| OpenAI   | API key            | `/connect openai sk-...`               |
| Endpoint | URL + optional key | `/connect endpoint <name> <url> [key]` |

Multiple providers can be connected at the same time. Tab on
`/connect` autocompletes provider names.

### Endpoints

Any OpenAI-compatible API can be used as a provider:

```text
you: /connect endpoint deepinfra https://api.deepinfra.com/v1/openai di-...
you: /connect endpoint ollama http://localhost:11434/v1
you: /model deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct
```

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

The model list is cached at startup and refreshed on every `/connect`.

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

| Command                | Description                              |
| ---------------------- | ---------------------------------------- |
| `/help`                | Show available commands                  |
| `/review`              | List open PRs and branches               |
| `/review <id>`         | Select a PR or branch for review         |
| `/draft`               | View, sync, or post the review draft     |
| `/connect <service>`   | Authenticate with a service              |
| `/model`               | List available models from all providers |
| `/model <provider/id>` | Set the active model                     |
| `/index`               | Show index status, clear, rebuild        |
| `/compact`             | Summarise older context to free space    |
| `/session`             | List, inspect, or delete sessions        |
| `/new`                 | Start a new conversation                 |
| `/quit`                | Exit (also `/q`)                         |

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
| Up/Down   | Browse history or navigate multiline     |
| Ctrl+C    | Cancel running task (double-tap to quit) |
| Ctrl+O    | Expand truncated shell output            |

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

The database has a single `fragments` table.  Each message is
stored as **1 + N rows**:

- **1 message row** — metadata (timestamps, model name, token
  counts, cost).  Self-referencing (`message_id = id`),
  `fragment_index = 0`.
- **N content rows** — one per part (user text, assistant text,
  tool call, tool result, thinking, etc.), each serialised as
  JSON.  Content rows point to the message row via `message_id`
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
answer.  This produces 4 messages and ~10 rows:

```
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

The message format is **provider-agnostic**.  A response from
Claude and one from GPT-4o have the same structure (text parts,
tool calls, etc.).  History is preserved across `/model`
switches — only `/new` clears it.

One exception: thinking parts carry provider-specific IDs that
some providers reject.  When this causes an API error, rbtr
converts thinking parts to plain text (wrapped in `<thinking>`
tags) and retries.  The conversion is read-time only — the
original parts stay in the database, so switching back to a
thinking-capable model recovers them.

### `/new` — starting fresh

`/new` clears the in-memory conversation (history, usage
counters) and generates a new session ID.  The previous session's
data stays in the database — it can be listed with `/session` and
resumed with `/session resume`.

### `/session` command

```text
/session              List recent sessions (current repo)
/session all          List sessions across all repos
/session info         Show current session details
/session resume <id>  Resume a previous session (prefix match)
/session delete <id>  Delete a session by ID (prefix match)
/session purge 7d     Delete sessions older than 7 days
```

**`/session resume`** loads the target session's messages from the
database and switches the active session ID.  The conversation
continues where it left off — the model sees the full history.
You can resume sessions from different repos or different models.

**`/session delete`** removes all fragments for a session
(cascading via foreign keys).  You cannot delete the active
session — use `/new` first.

### Automatic behaviour

- **New session on startup.** Each `rbtr` invocation starts a fresh
  session, labelled with the current repo and branch
  (e.g. `acme/app — main`).
- **Streaming persistence.** Parts are saved to the database as
  they arrive from the model, not batched after the turn.
- **Input history from the database.** Up/Down arrow browses input
  history across all sessions, deduplicated and sorted by recency.

### Pruning old sessions

Sessions accumulate over time.  Use `/session purge` to clean up:

```text
/session purge 7d     Delete sessions older than 7 days
/session purge 2w     Delete sessions older than 2 weeks
/session purge 24h    Delete sessions older than 24 hours
```

Duration suffixes: `d` (days), `w` (weeks), `h` (hours).  The
active session is never deleted.  To remove a specific session,
use `/session delete <id>`.


## Context compaction

Long conversations accumulate tokens until the model's context
window fills up.  rbtr compacts automatically — summarising older
messages while keeping recent turns intact.

### How compaction works

Compaction splits the conversation into **old** and **kept**
messages based on `keep_turns` (default 2).  A *turn* starts at a
user prompt and includes everything up to the next user prompt
(the model's responses, tool calls, tool results).

The old messages are serialised to text and sent to the current
model with a summary prompt.  The result:

- **Old rows** get `compacted_by` set to the summary's row ID.
  They become invisible to `load_messages()` but stay in the
  database for auditing.  Cleaned up when the session is deleted
  (FK cascade).
- **Summary** — a single message containing the LLM-generated
  summary, prefixed with
  `[Context summary — earlier conversation was compacted]`.
  Timestamped to sort before the kept messages.
- **Kept messages** — untouched, stay as-is.

#### Example: compaction with `keep_turns = 2`

Before compaction (5 turns):

```
turn 1:  user "set up the project"     → assistant "done, created package.json"
turn 2:  user "add authentication"     → assistant [tool calls] → "added auth middleware"
turn 3:  user "write tests for auth"   → assistant [tool calls] → "added 12 tests"
turn 4:  user "fix the failing test"   → assistant [tool calls] → "fixed assertion"
turn 5:  user "review the PR"          → assistant [reading files...] ← in progress
```

After compaction — turns 1–3 are summarised, turns 4–5 are kept:

```
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

```
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

```
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
returns: summary S1 → turn 4 → turn 5.  The old rows stay in the
database until the session is deleted.

### When compaction triggers

Four paths, all using the same compaction logic:

**1. Post-turn** — After a successful LLM response, if context
usage exceeds `auto_compact_pct` (default 85%).

**2. Mid-turn** — During a multi-step tool-calling turn, after
each tool-call cycle.  If the threshold is exceeded and the model
made tool calls (meaning more requests will follow), rbtr breaks
out of the agent loop, compacts, reloads history from the
database, and resumes the turn.  The in-progress turn counts as
one of the `keep_turns` — so with `keep_turns = 2`, the current
turn and the previous turn are preserved, and everything older is
summarised.  Mid-turn compaction fires at most once per turn.

**3. On overflow error** — When the API rejects a request with a
context-length error (HTTP 400, 413, etc.), rbtr compacts and
retries automatically.

**4. Manual** — `/compact` on demand, with optional extra
instructions for the summary:

```text
you: /compact
you: /compact Focus on the authentication changes
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
dependency.  After compaction the footer's context % stays at the
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

rbtr gives the LLM 14 tools, conditionally hidden based on
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
back to the local filesystem (covers `.rbtr/REVIEW-*` notes
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

**`search_symbols(name, offset?, max_results?)`** — Find
functions, classes, and methods by name substring
(case-insensitive). Returns kind, scope, file path, and line
number. Use short names — `MQ` not `lib.mq.MQ`.

**`search_codebase(query, offset?, max_results?)`** —
Full-text keyword search (BM25) across symbol names and
source content. Results ranked by relevance.

**`search_similar(query, offset?, max_results?)`** —
Semantic similarity search using embeddings. Finds
conceptually related code even without shared keywords.

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

#### Review notes (always available)

**`edit(path, new_text, old_text?)`** — Edit or create
review notes files in the `.rbtr/` workspace. Files must
be under `.rbtr/` with names starting with `REVIEW-`
(e.g. `.rbtr/REVIEW-plan.md`). When `old_text` is empty,
creates the file or appends; when set, replaces the exact
match. Review notes are readable by `read_file`, `grep`,
and `list_files` via the filesystem fallback.

#### PR discussion (require PR)

**`get_pr_discussion(offset?, max_results?)`** — Read
all existing discussion on the current PR. Returns reviews,
inline comments, and general comments sorted chronologically.
Includes bot comments (CI, linters) and emoji reactions.
Cached per session — fetched once, then read from memory.

#### Draft management (require PR)

**`add_review_comment(path, line, body, suggestion?)`** —
Add an inline comment to the review draft. Persisted to
`.rbtr/REVIEW-DRAFT-<pr>.toml` immediately.

**`edit_review_comment(index, body?, suggestion?)`** —
Edit an existing comment by 1-based index. Only provided
fields are changed.

**`remove_review_comment(index)`** — Remove a comment
by 1-based index.

**`set_review_summary(summary)`** — Set the top-level
review body that appears at the top of the GitHub review.

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
- Workspace files (`.rbtr/REVIEW-*` notes, untracked files)
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

| Subcommand          | Description                              |
| ------------------- | ---------------------------------------- |
| `/index`            | Show index status (chunks, edges, size)  |
| `/index clear`      | Delete the index database                |
| `/index rebuild`    | Clear and re-index from scratch          |
| `/index prune`      | Remove orphan chunks not in any snapshot |
| `/index model`      | Show current embedding model             |
| `/index model <id>` | Switch embedding model and re-embed      |

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
include = [".rbtr/REVIEW-*"]                            # force-include globs (override .gitignore)
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
  structural index works, only `search_similar` is degraded.
- **Slow indexing** → review starts immediately, index catches
  up in a background thread.

## Review draft

When reviewing a GitHub PR, rbtr helps you build a structured
review that can be posted back to GitHub. The LLM builds the
draft incrementally using tool calls; you control when and how
it's posted.

### Workflow

1. **Select a PR** — `/review 42` fetches the PR and
   auto-syncs any existing pending review from GitHub.

2. **Review with the LLM** — as you discuss the code, the
   LLM uses `add_review_comment` and `set_review_summary`
   to build a draft. Each change persists immediately to
   `.rbtr/REVIEW-DRAFT-42.toml`.

3. **Inspect the draft** — `/draft` shows the current state:
   summary, numbered inline comments, suggestion indicators.

4. **Sync** — `/draft sync` does a bidirectional sync: pulls
   any remote pending comments into the local draft, then
   pushes the merged result back to GitHub as a pending review
   (visible in the UI but not submitted).

5. **Post** — `/draft post` submits the review to GitHub.
   `/draft post approve` and `/draft post request_changes`
   set the review event type (default is COMMENT).

### Draft commands

| Subcommand            | Description                           |
| --------------------- | ------------------------------------- |
| `/draft`              | Show the current draft                |
| `/draft sync`         | Bidirectional sync with GitHub        |
| `/draft post [event]` | Submit review to GitHub               |
| `/draft clear`        | Delete local draft and remote pending |

Tab completes subcommands and event types.

### Safety

- **Unsynced guard** — `/draft post` refuses if the remote
  pending review has comments not in your local draft. Run
  `/draft sync` first.
- **Atomic posting** — all comments are submitted in a single
  `create_review` API call. No partial reviews.
- **Crash-safe** — the draft is a TOML file on disk, updated
  on every mutation. If rbtr crashes, the draft survives.
- **Human-editable** — `.rbtr/REVIEW-DRAFT-42.toml` is plain
  TOML with `[[comments]]` array-of-tables. You can edit it
  by hand.
- **Draft cleanup** — after a successful post, the local file
  is deleted automatically.

### GitHub suggestions

When the LLM provides a `suggestion` parameter in
`add_review_comment`, it's posted as a GitHub suggestion block:

````markdown
Use exponential backoff.

```suggestion
time.sleep(2 ** attempt)
```
````

The author can apply suggestions with one click in the GitHub UI.

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
max_requests_per_turn = 25    # default
workspace_prefix = "REVIEW-"  # filename prefix for writable .rbtr/ files
```

The `workspace_prefix` controls which files in `.rbtr/` the
LLM can create and edit (e.g. `REVIEW-plan.md`,
`REVIEW-findings.md`). Review drafts use
`<prefix>DRAFT-<pr>.toml` — with the default prefix,
`.rbtr/REVIEW-DRAFT-42.toml`. If you change the prefix,
update `index.include` to match:

```toml
[index]
include = [".rbtr/MY-PREFIX-*"]

[tools]
workspace_prefix = "MY-PREFIX-"
```

## Known issues

- **Cross-provider reasoning history:** PydanticAI stores
  provider-specific reasoning IDs (e.g. `rs_*`) in conversation
  history. Switching models can cause 400 errors when the new provider
  rejects those IDs. rbtr works around this by demoting thinking parts
  to plain text wrapped in `<thinking>` tags and retrying — revisit
  once PydanticAI fixes upstream.

## Development

```bash
uv sync        # Install dependencies
just check     # Lint + typecheck + test
just fmt       # Auto-fix and format
```

### Adding a new provider

Each provider lives in its own module under `src/rbtr/providers/`.
A provider needs:

1. **Auth flow** — implement authentication (OAuth, API key, etc.)
   and store credentials via `creds.update()`.

2. **`build_model(model_id)`** — return a pydantic-ai `Model`
   instance. Use the provider's async client and wrap it in the
   appropriate pydantic-ai model class (e.g. `AnthropicModel`,
   `OpenAIChatModel`). Any OpenAI-compatible API can reuse
   `OpenAIModel` with a custom `AsyncOpenAI` client — see
   `providers/endpoint.py` for an example.

3. **`list_models()`** — optional. Return available model IDs so
   `/model` and Tab completion work.

4. **Register in `providers/__init__.py`** — add the provider prefix
   to `BuiltinProvider` and wire up `_build_model_by_name()`.

Model IDs always use `<provider>/<model-id>` format everywhere.

### Thinking effort

rbtr exposes a unified `thinking_effort` config (`low`/`medium`/
`high`/`max`) that maps to provider-specific settings. Users cycle
it with Shift+Tab; the footer shows the current level.

All provider-specific dispatch lives in `providers/__init__.py`.
Model construction (`build_model`) and settings construction
(`build_model_settings`) are siblings — `engine/llm.py` calls
both without importing any provider module directly.

**To wire up effort for a new model type**, add a branch in
`build_model_settings()` (`providers/__init__.py`):

```python
from pydantic_ai.models.foo import FooModel

if isinstance(model, FooModel):
    from pydantic_ai.models.foo import FooModelSettings
    return FooModelSettings(foo_effort=effort)
```

The caller (`engine/llm.py`) sets `session.effort_supported`
based on whether `build_model_settings` returned settings or
`None`. If no branch matches, the footer shows `∴ off` in red.

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

## License

MIT
