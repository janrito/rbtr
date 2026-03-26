# rbtr

An agentic code review harness in the terminal. rbtr
indexes your repository, reads the diff, understands the
structure of the code — commit messages, PR description,
existing discussion — and helps you reason through the
changes. It writes structured review comments and posts
them to GitHub.

It connects to Claude, GPT, Gemini, and any OpenAI-compatible
endpoint. Conversations are saved automatically and survive
crashes. Long sessions compact themselves so the model never
loses context.

## Install

```bash
# Requires Python 3.13+
uv tool install -e .
```

## Your first review

```bash
rbtr                  # launch in a git repo
```

Connect a provider, select a PR, and ask the model to review:

```text
you: /connect claude
Opening browser to sign in with your Claude account…
Connected to Claude. LLM is ready.

you: /review 42
Fetching PR #42…
⟳ indexing 177 files…
```

The model has 20 tools — it reads files, searches the index,
inspects the diff, and follows references across the codebase.
Ask it anything about the changes:

```text
you: explain the nature of the changes, focus on error handling

claude: [reads diff, searches for retry logic, reads related files]

The PR adds retry logic to the HTTP client but has two issues:

1. **No backoff** — `src/client.py:42` retries immediately
   in a tight loop. Under load this will hammer the server.

2. **Unused import** — `src/client.py:89` imports `sleep`
   but the retry loop uses `time.sleep` directly.

I've added both as draft comments.

you: the first one is a blocker, mark it as such
```

Inspect and post the draft:

```text
you: /draft

## 2 comments

### src/client.py

  ★ 1. L:42
  > resp = session.get(url)
  **blocker:** Retries without backoff. Under load this
  hammers the server. Use exponential backoff.

  ★ 2. L:89
  > from time import sleep
  **nit:** Unused import — `sleep` is never called.

## Summary

Two issues in the retry logic. The missing backoff is a
blocker; the unused import is minor.

you: /draft post
Review posted.
```

You can also start without a PR — `rbtr` opens a plain
conversation in any git repo. Use `/review` later to select
a target.

### Snapshot review

Review code at a single point in time — no PR, no diff, no
GitHub. Useful for onboarding, architecture review, or audit.

```bash
rbtr v2.1.0               # launch with a tag
rbtr main                  # launch with a branch
rbtr HEAD                  # launch at current commit
```

Or from inside rbtr:

```text
you: /review v2.1.0
Reviewing snapshot: v2.1.0
⟳ indexing 177 files…

you: walk me through the auth module

you: /review main feature
Reviewing branch: main → feature
```

## Tools

The model has 20 tools for reading code, navigating the
codebase, writing review feedback, and running shell
commands. Tools appear and disappear based on session state
— the model never sees a tool it cannot use.

| Condition         | Tools available                          |
| ----------------- | ---------------------------------------- |
| Always            | `edit`, `remember`, `run_command`        |
| Repo + any target | `read_file`, `list_files`, `grep`        |
| Index ready       | `search`, `read_symbol`, `list_symbols`, |
|                   | `find_references`                        |
| PR or branch      | `diff`, `changed_files`, `commit_log`,   |
|                   | `changed_symbols`                        |
| PR only           | Draft tools, `get_pr_discussion`         |

In snapshot mode (`/review <ref>`) the model has file, index,
workspace, and shell tools. Diff and draft tools are hidden
— there is no base to compare against.

### Reading code

The model reads changed files, referenced modules, tests,
and config to understand context beyond the diff.

| Tool         | Description                                |
| ------------ | ------------------------------------------ |
| `read_file`  | Read file contents, paginated by lines     |
| `grep`       | Substring search, scoped by glob or prefix |
| `list_files` | List files, scoped by glob or prefix       |

### Understanding changes

The model starts here — what changed, how the work was
structured, and what the changes mean structurally.

| Tool              | Description                          |
| ----------------- | ------------------------------------ |
| `diff`            | Unified diff, scoped by glob or file |
| `changed_files`   | Files changed in the PR              |
| `commit_log`      | Commits between base and head        |
| `changed_symbols` | Symbols added, removed, or modified. |
|                   | Flags stale docs and missing tests   |

### Navigating the codebase

The code index lets the model search by name, keyword, or
concept — and follow references to check whether a change
breaks callers or misses related code.

| Tool              | Description                          |
| ----------------- | ------------------------------------ |
| `search`          | Find symbols by name, keyword, or    |
|                   | natural-language concept             |
| `read_symbol`     | Read the full source of a symbol     |
| `list_symbols`    | Table of contents for a file         |
| `find_references` | Find all symbols referencing a given |
|                   | symbol via the dependency graph      |

### Writing the review

The model builds the review incrementally — adding comments
on specific lines, editing them based on discussion, and
setting the overall summary.

| Tool                   | Description                      |
| ---------------------- | -------------------------------- |
| `add_draft_comment`    | Inline comment on a code snippet |
| `edit_draft_comment`   | Edit an existing comment         |
| `remove_draft_comment` | Remove a comment                 |
| `set_draft_summary`    | Set the top-level review body    |
| `read_draft`           | Read the current draft           |

### Context and memory

The model reads existing discussion to avoid duplicating
feedback, saves durable facts for future sessions, and
writes notes to the workspace.

| Tool                | Description                         |
| ------------------- | ----------------------------------- |
| `get_pr_discussion` | Existing reviews, comments, CI      |
| `remember`          | Save a fact for future sessions     |
| `edit`              | Create or modify `.rbtr/notes/` and |
|                     | `.rbtr/AGENTS.md`                   |

### Shell execution

The model can run shell commands when `tools.shell.enabled`
is `true` (the default).

| Tool          | Description                             |
| ------------- | --------------------------------------- |
| `run_command` | Execute a shell command, return output  |

The primary use is executing scripts bundled with skills.
The model is steered away from using it for codebase access
— files under review live in a different branch or commit,
so `read_file`, `grep`, and other bespoke tools are the
correct choice (they read from the git object store at the
right ref). The working tree is treated as read-only.

Output is streamed to the TUI via a head/tail buffer
(first 3 + last 5 lines, refreshed at ~30 fps). When the
command executes a skill script, the header shows the skill
name and source instead of raw JSON args
(e.g. `⚙ [brave-search · user] search.sh "query"`). The
full result returned to the model is truncated to
`tools.shell.max_output_lines` (default 2000).

```toml
[tools.shell]
enabled = true       # set false to disable entirely
timeout = 120        # default timeout in seconds (0 = no limit)
max_output_lines = 2000
```

`list_files`, `grep`, and `diff` accept a `pattern` parameter
that works like a git pathspec: a plain string is a directory
prefix or file path, glob metacharacters (`*`, `?`, `[`)
activate pattern matching, and `**` matches across
directories. For example, `pattern="src/**/*.py"` scopes to
Python files under `src/`.

Tools that read code accept a `ref` parameter — `"head"`
(default), `"base"`, or a raw commit SHA — so the model
can inspect the codebase at any point in time. File tools
read from the git object store first and fall back to the
local filesystem for untracked files (`.rbtr/notes/`,
drafts).

All paginated tools show a trailer when output is truncated.
Each turn is limited to 25 tool calls (configurable via
`max_requests_per_turn`). When the limit is reached, the
model summarises its progress and asks whether to continue.

## Commands

| Command                   | Description                                 | Context |
| ------------------------- | ------------------------------------------- | ------- |
| `/help`                   | Show available commands                     |         |
| `/review`                 | List open PRs and branches                  | ✓       |
| `/review <number>`        | Select a PR for review                      | ✓       |
| `/review <ref>`           | Snapshot review at a git ref                | ✓       |
| `/review <base> <target>` | Diff review between two refs                | ✓       |
| `/draft`                  | View, sync, or post the review draft        | ✓       |
| `/connect <service>`      | Authenticate with a service                 | ✓       |
| `/model`                  | List available models from all providers    |         |
| `/model <provider/id>`    | Set the active model                        | ✓       |
| `/index`                  | Index status, search, diagnostics, rebuild  | partial |
| `/compact`                | Summarise older context to free space       | ✓       |
| `/compact reset`          | Undo last compaction (before new messages)  | ✓       |
| `/session`                | List, inspect, or delete sessions           | partial |
| `/stats`                  | Show session token and cost statistics      | ✓       |
| `/memory`                 | List, extract, or purge cross-session facts | partial |
| `/skill`                  | List or load a skill                        | ✓       |
| `/reload`                 | Show active prompt sources                  |         |
| `/new`                    | Start a new conversation                    |         |
| `/quit`                   | Exit (also `/q`)                            |         |

The **Context** column shows which commands produce
[context markers](#context-markers) for the model.
`partial` means some subcommands emit markers (e.g.
`/index status` does, `/index search` does not).

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

The model list is fetched lazily on first Tab completion or
`/model` command, and refreshed on every `/connect`.

### Switching models mid-conversation

Conversation history is preserved when you switch models.
PydanticAI's message format is provider-agnostic, so previous
messages are converted automatically. When a provider rejects
history from a different provider (mismatched tool-call
formats, thinking metadata), rbtr repairs the history in
memory — the original messages are never modified:

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

## Terminal reference

### Shell commands

Prefix any command with `!` to run it in a shell:

```text
you: !git log --oneline -5
you: !rg "TODO" src/
```

Long output is truncated — press **Ctrl+O** to expand it.

### Tab completion

| Context        | Example            | Completes               |
| -------------- | ------------------ | ----------------------- |
| Slash commands | `/rev` → `/review` | Command names           |
| Command args   | `/connect c`       | Providers, models, etc. |
| Shell commands | `!git ch`          | Bash completion         |
| File paths     | `!cat ~/Doc`       | Directories and files   |

### Key bindings

| Key       | Action                                   |
| --------- | ---------------------------------------- |
| Enter     | Submit input                             |
| Alt+Enter | Insert newline (multiline input)         |
| Tab       | Autocomplete                             |
| Shift+Tab | Cycle thinking effort level              |
| Up/Down   | Browse history or navigate multiline     |
| Ctrl+C    | Cancel running task (double-tap to quit) |
| Ctrl+O    | Expand truncated shell output            |

### Pasting

Bracketed paste is enabled — pasted newlines insert into the
prompt instead of submitting. Large pastes collapse into an
atomic marker (`[pasted 42 lines]`) that expands on submit.

### Context markers

After a slash command or shell command, a context marker
appears above your input — a tag like `[/review → PR #42]`
or `[! git log — exit 0]`. On submit, markers expand into a
`[Recent actions]` block prepended to your message so the
model knows what you just did. Backspace at the start of the
input dismisses the last marker. Not every command produces
a marker — only those whose outcome is useful to the model.

## Usage display

The footer shows token usage and context after each response:

```text
 owner/repo                            claude/claude-sonnet-4-20250514
 PR #42 · feature-branch  |7| 12% of 200k ↑24.3k ↓1.2k ↯18.0k $0.045
```

`|7|` messages, `12%` context used, `↑` input tokens,
`↓` output tokens, `↯` cache-read tokens, `$` cost.
Colours shift from green to yellow to red as context fills.
`/new` resets all counters.

## Sessions

Every conversation is saved to a local SQLite database at
`~/.config/rbtr/sessions.db`. Messages are persisted as they
stream — if rbtr crashes, the conversation survives up to the
last received part. Requests are always persisted before their
responses, so resumed sessions load in the correct order.

### Persistence

The message format is provider-agnostic. History is preserved
across `/model` switches — only `/new` clears it. rbtr
repairs history automatically in memory on every turn — the
original messages are never modified. Preventive repairs
(sanitising cross-provider field values, patching cancelled
tool calls) run before each API call. When a provider still
rejects the history, escalating structural repairs retry
automatically. All repairs are recorded as incidents, visible
in `/stats`.

Ctrl+C during a tool-calling turn cancels immediately. Any tool
calls without results get synthetic `(cancelled)` returns so
the conversation can continue:

```text
⚠ Previous turn was cancelled mid-tool-call (read_file, grep).
  Those tool results are lost — the model will continue without them.
```

See [Conversation storage](ARCHITECTURE.md#conversation-storage)
and [Cross-provider history repair](ARCHITECTURE.md#cross-provider-history-repair)
in ARCHITECTURE.md.

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
(case-insensitive). ID prefix is tried first; if no match,
the label is searched. When several sessions share a label the
most recent one is picked.

**`/session resume`** loads the target session's messages from the
database and switches the active session ID. The conversation
continues where it left off — the model sees the full history.
If the session had a review target (PR or branch), it's
automatically restored — rbtr re-runs `/review` to fetch fresh
metadata and rebuild the code index.
You can resume sessions from different repos or different models.

**`/session delete`** requires an exact ID prefix — no label
matching, to prevent accidental deletion. Removes all fragments
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
ARCHITECTURE.md — History repair). `/stats` surfaces these
when they exist:

```text
  Failures (3)
    history_format                 2   recovered: 2
    overflow                       1   recovered: 1

  History repairs (6)
    repair_dangling                1   (cancelled_mid_tool_call)
    consolidate_tool_returns       2   (cross_provider_retry)
    demote_thinking                2   (cross_provider_retry)
    flatten_tool_exchanges         1   (cross_provider_retry)

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

## Cross-session memory

rbtr learns facts from conversations and remembers them across
sessions. Facts are durable knowledge — project conventions,
architecture decisions, user preferences, recurring patterns
discovered during review.

Static project instructions belong in `AGENTS.md`. Facts are
what the agent learns on its own.

### How facts are created

Facts are extracted automatically at two points:

- **During compaction** — the extraction agent runs
  concurrently with the summary agent, analysing the messages
  being compacted.
- **After posting a review** — `/draft post` triggers
  extraction on the full session, since completed reviews are
  the richest source of project knowledge.

You can also trigger extraction manually with
`/memory extract`, or teach the agent directly — it has a
`remember` tool that saves facts on demand during
conversation.

### Scopes

Each fact has a scope:

- **global** — applies everywhere (e.g. "prefers British
  English spelling")
- **repo** — specific to the current repository (e.g. "uses
  pytest with the mocker fixture, not unittest.mock")

Repo-scoped facts are keyed by `owner/repo` and only
injected when working in that repository.

### Injection

At session start, active facts for the current scopes are
loaded and injected into the system prompt. Two caps control
what's included:

- `max_injected_facts` (default 20) — maximum number of
  facts
- `max_injected_tokens` (default 2000) — token budget for
  injected facts

Facts are ordered by `last_confirmed_at` (most recently
confirmed first), so frequently re-observed facts take
priority.

### Deduplication

The extraction agent sees all existing facts and tags each
extraction as `new`, `confirm` (re-observed), or `supersede`
(replaces an outdated fact). Matching is content-based — no
opaque IDs are exposed to the LLM. When the LLM's reference
doesn't exactly match, a clarification retry corrects the
mismatch.

### `/memory` command

```text
/memory               List active facts by scope
/memory all           Include superseded facts
/memory extract       Extract facts from the current session
/memory purge 7d      Delete facts not confirmed in 7 days
/memory purge 2w      Delete facts older than 2 weeks
```

Purge uses `last_confirmed_at` — facts that are regularly
re-observed survive longer. Duration suffixes: `d` (days),
`w` (weeks), `h` (hours).

### Memory configuration

```toml
[memory]
enabled = true                    # Toggle the feature
max_injected_facts = 20           # Facts in system prompt
max_injected_tokens = 2000        # Token budget for injection
max_extraction_facts = 200        # Existing facts shown to extraction agent
fact_extraction_model = ""        # Override model (empty = session model)
```

## Skills

Skills are self-contained instruction packages — a markdown
file with optional bundled scripts — that teach the model
new capabilities. rbtr discovers skills automatically from
multiple directories and presents them in the system prompt.

rbtr scans the same skill directories as pi and Claude Code,
so existing skills work with zero configuration:

```text
~/.config/rbtr/skills/      # user-level rbtr skills
~/.claude/skills/           # Claude Code skills
~/.pi/agent/skills/         # pi skills
~/.agents/skills/           # Agent Skills standard
.rbtr/skills/               # project-level (any ancestor to git root)
.claude/skills/             # project-level Claude Code
.pi/skills/                 # project-level pi
.agents/skills/             # project-level Agent Skills
```

Skills use the [Agent Skills standard][agent-skills] format:
a markdown file with YAML frontmatter (`name`, `description`).
The model sees a catalog of available skills in its system
prompt and reads the full skill file on demand via `read_file`.

[agent-skills]: https://agentskills.io/specification

### `/skill` command

```text
/skill                         List discovered skills
/skill brave-search            Load a skill into context
/skill brave-search "query"    Load with a follow-up message
```

Tab-completes skill names. Skills marked with
`disable-model-invocation: true` are hidden from the prompt
catalog but still loadable via `/skill`.

### Configuration

```toml
[skills]
project_dirs = [".rbtr/skills", ".claude/skills", ".pi/skills", ".agents/skills"]
user_dirs = ["~/.config/rbtr/skills", "~/.claude/skills", "~/.pi/agent/skills", "~/.agents/skills"]
extra_dirs = []
```

Set `project_dirs = []` or `user_dirs = []` to disable
scanning. `extra_dirs` adds directories on top.

## Context compaction

Long conversations fill the context window. rbtr compacts
automatically — summarising older messages while keeping
recent turns intact.

Compaction splits the conversation by turn boundaries. A turn
starts at a user prompt and includes the model's responses,
tool calls, and tool results. The last `keep_turns` (default 2)
are preserved; everything older is serialised and sent to the
model for summarisation. The original messages stay in the
database for auditing.

### Example

Before (5 turns):

```text
turn 1:  user "set up the project"     → assistant "done"
turn 2:  user "add authentication"     → assistant [tools] → "added auth"
turn 3:  user "write tests for auth"   → assistant [tools] → "added 12 tests"
turn 4:  user "fix the failing test"   → assistant [tools] → "fixed assertion"
turn 5:  user "review the PR"          → assistant [reading files...]
```

After (turns 1–3 summarised, 4–5 kept):

```text
summary: "[Context summary] Set up project. Added auth
          middleware. Wrote 12 tests for auth module."
turn 4:  user "fix the failing test"   → assistant "fixed assertion"
turn 5:  user "review the PR"          → assistant [reading files...]
```

### When compaction triggers

- **Post-turn** — after a response, if context usage exceeds
  `auto_compact_pct` (default 85%).
- **Mid-turn** — during a tool-calling cycle, if the threshold
  is exceeded. Compacts once, reloads history, resumes the turn.
- **Overflow** — when the API rejects a request with a
  context-length error. Compacts and retries.
- **Manual** — `/compact`, with optional extra instructions:

```text
you: /compact
you: /compact Focus on the authentication changes
```

`/compact reset` undoes the latest compaction, restoring the
original messages. Only allowed before new messages are sent.

When the old messages are too large for a single summary
request, rbtr finds the largest prefix that fits, summarises
it, and pushes the rest into the kept portion.

### Settings

```toml
[compaction]
auto_compact_pct = 85       # trigger threshold (% of context)
keep_turns = 2              # recent turns to preserve
reserve_tokens = 16000      # reserved for the summary response
summary_max_chars = 2000    # max chars per tool result in input
```

See [Context compaction in ARCHITECTURE.md](ARCHITECTURE.md#context-compaction)
for the split algorithm, orphan handling, and reset mechanics.

## Configuration

User-level files live in `~/.config/rbtr/` (override with
`RBTR_USER_DIR`). A workspace overlay at `.rbtr/config.toml`
can override per-project settings — the nearest `.rbtr/`
walking from CWD to the git root wins (monorepo-friendly).

- **`config.toml`** — model, endpoints, feature settings.
- **`creds.toml`** — API keys and OAuth tokens (0600).

```toml
# config.toml
model = "claude/claude-sonnet-4-20250514"

[endpoints.deepinfra]
base_url = "https://api.deepinfra.com/v1/openai"
```

Config values can also be set via environment variables with
`RBTR_` prefix (e.g. `RBTR_MODEL`). OAuth tokens are managed
by `/connect` — you never need to edit `creds.toml` by hand.

### Customising the prompt

Three levels of customisation, loaded in order:

1. **`AGENTS.md`** (repo root) — project-specific rules.
   Configure the file list with
   `project_instructions = ["AGENTS.md"]`.
2. **`~/.config/rbtr/APPEND_SYSTEM.md`** — user-wide
   preferences appended to the system prompt.
3. **`~/.config/rbtr/SYSTEM.md`** — full system prompt
   replacement (Jinja template with `project_instructions`
   and `append_system` variables).

Example `AGENTS.md`:

```markdown
- Target Python 3.13+. Use modern features.
- All code must be type-annotated.
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

### Index configuration

```toml
[index]
enabled = true
embedding_model = "gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
include = [".rbtr/notes/*"]      # force-include (override .gitignore)
extend_exclude = [".rbtr/index"] # exclude on top of .gitignore
```

The index is persistent — subsequent `/review` runs skip
unchanged files (keyed by git blob SHA).

### Graceful degradation

- **No grammar installed** for a language → falls back to
  line-based plaintext chunking.
- **No embedding model** (missing GGUF, GPU init failure) →
  structural index works, semantic search signal is skipped
  (weight redistributed to name and keyword channels).
- **Slow indexing** → review starts immediately, index catches
  up in a background thread.

## Review draft

The LLM builds a structured review using draft tools
(`add_draft_comment`, `set_draft_summary`, etc.). The draft
persists to `.rbtr/drafts/<pr>.yaml` — crash-safe, human-
editable, and synced bidirectionally with GitHub.

### Workflow

1. `/review 42` — fetches the PR, pulls any existing
   pending review from GitHub.
2. The LLM adds comments and a summary as it reviews.
3. `/draft` — inspect the current state.
4. `/draft sync` — bidirectional sync with GitHub
   (three-way merge, conflicts resolve to local).
5. `/draft post` — submit the review. Optional event type:
   `approve`, `request_changes` (default `COMMENT`).

### Draft commands

| Subcommand            | Description                           |
| --------------------- | ------------------------------------- |
| `/draft`              | Show draft with sync status           |
| `/draft sync`         | Bidirectional sync with GitHub        |
| `/draft post [event]` | Submit review to GitHub               |
| `/draft clear`        | Delete local draft and remote pending |

### Status indicators

| Indicator | Meaning                                |
| --------- | -------------------------------------- |
| `✓`       | Synced — matches last-pushed snapshot  |
| `✎`       | Modified locally since last sync       |
| `★`       | New — never synced to GitHub           |
| `✗`       | Deleted — will be removed on next sync |

### Safety

- **Unsynced guard** — `/draft post` refuses if the remote
  has comments not in the local draft.
- **Atomic posting** — all comments are submitted in one
  API call.
- **Crash-safe** — YAML on disk, updated on every mutation.
  Mid-sync crashes recover on the next sync.
- **GitHub suggestions** — the LLM can provide replacement
  code; it's posted as a suggestion block that the author
  can apply with one click.

See [Review draft and GitHub integration][arch-draft]
in ARCHITECTURE.md for sync internals.

[arch-draft]: ARCHITECTURE.md#review-draft-and-github-integration

## Theme

rbtr defaults to a dark palette with semantically tinted panel
backgrounds. Switch to light mode or override individual
styles in `config.toml`:

```toml
[theme]
mode = "light"            # "dark" (default) or "light"

[theme.light]             # override fields for the active mode
bg_succeeded = "on #E0FFE0"
prompt = "bold magenta"
```

Text styles use ANSI names (`bold cyan`, `dim`, `yellow`, …)
that adapt to your terminal's colour scheme. Panel backgrounds
use hex values. Any string that Rich accepts as a
[style definition](https://rich.readthedocs.io/en/latest/style.html)
is valid.

Available fields are defined in `PaletteConfig` in
[`config.py`](src/rbtr/config.py).

## Development

```bash
uv sync        # Install dependencies
just check     # Lint + typecheck + test
just fmt       # Auto-fix and format
```

### Architecture reference

For details on how providers, tools, the index, language
plugins, session persistence, history repair, and GitHub sync
work internally, see [ARCHITECTURE.md](ARCHITECTURE.md).

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
commits reachable from head but not from base. Review
comment validation diffs from the merge base of the two
refs, matching GitHub's three-dot diff.

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

For search internals (fusion algorithm, scoring functions,
tokenisation), see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT
