# Plan

Project status and roadmap.

---

## Completed

### Step 1: PR Data Fetching & Context ✅

**Goal**: Fetch full PR/branch data and give the LLM context.

**What was built**:

- `models.py` — `ReviewTarget`, `PRTarget`, `BranchTarget`,
  `PRSummary`, `BranchSummary`. Union type `Target`.
- `github/client.py` — `list_open_prs()`, `list_unmerged_branches()`,
  `validate_pr_number()`.
- `repo.py` — `open_repo()`, `parse_github_remote()`,
  `default_branch()`, `list_local_branches()`,
  `require_clean()`.
- Engine: `/review` lists open PRs and branches as tables,
  or selects a review target by PR number or branch name.
- GitHub auth via device flow (`github/auth.py`), token stored
  in `creds.toml`.

### Step 2: LLM Agent ✅

**Goal**: PydanticAI agent with streaming output and multi-provider
support.

**What was built**:

- `engine/agent.py` — PydanticAI `Agent[AgentDeps, str]` with
  decorator-based instructions. Model provided at call time,
  not baked in.
- `engine/llm.py` — Streaming handler: iterates agent graph nodes,
  emits `TextDelta` and `ToolCallStarted`/`ToolCallFinished` events.
  Handles history format errors and thinking demotion.
- Providers: Claude (`providers/claude.py`), ChatGPT/Codex
  (`providers/openai_codex.py`), OpenAI API key
  (`providers/openai.py`), generic endpoint
  (`providers/endpoint.py`).
- `providers/__init__.py` — `build_model()` dispatches by
  `BuiltinProvider` enum prefix. `build_model_settings()` maps
  `ThinkingEffort` to provider-specific settings.
- OAuth/PKCE login flows for Claude and ChatGPT.
  `/connect` command handles all auth flows.
  `/model` lists and switches models.
- Conversation history preserved across model/provider switches.
  Only `/new` clears.
- Prompt templates in `prompts/` (Jinja via minijinja):
  `system.md`, `review.md`, `index_status.md`.

### Step 3: Repo Operations & Context Retrieval ✅

**Goal**: Rich tools for the agent to explore the codebase.

**What was built**:

- `engine/tools.py` — 14 tools registered on the agent.
  Conditionally hidden based on session state (repo, index, PR).
  Index tools, file tools, git tools, review notes, PR
  discussion, draft management.
- Prepare functions dynamically hide tools based on session state.

### Step 4: Code Index ✅

**Goal**: Semantic and structural search across the repo.

**Deviation from original plan**: DuckDB + pyarrow replaced LanceDB.
Local embeddings via llama-cpp-python replaced API-based embeddings.

**What was built**:

- `index/` package — `store.py` (DuckDB with three tables:
  `file_snapshots`, `chunks`, `edges`; full-text search, vector
  similarity, name search, diff queries), `orchestrator.py`
  (coordinates git reader, plugins, tree-sitter, chunking, edges,
  embeddings; incremental updates keyed by commit SHA),
  `chunks.py` (plaintext chunking), `treesitter.py` (language-agnostic
  extraction), `edges.py` (import/test/doc edge inference),
  `embeddings.py` (local GGUF via llama-cpp-python),
  `languages.py` (plugin bridge), `git.py` (file listing from
  tree objects), `arrow.py` (PyArrow bulk inserts),
  `models.py` (Chunk, Edge, ChunkKind, EdgeKind, ImportMeta,
  IndexStats), `sql/` (SQL files, sqlfluff-linted).
- `/index` command with subcommands: `status`, `clear`,
  `rebuild`, `prune`, `model`.
- Background indexing on `/review` with progress events
  (`IndexStarted`, `IndexProgress`, `IndexReady`).

### Step 5: Structural Analysis ✅ (revised approach)

**Deviation from original plan**: Kùzu graph database was dropped.
Structural queries (call chains, import graphs, impact analysis)
are implemented as SQL queries in DuckDB over the `edges` table,
exposed as agent tools.

**What was built**:

- Edge types: `IMPORTS`, `DEFINES`, `CALLS`, `TESTS`, `DOCUMENTS`.
- Tree-sitter import extraction per language plugin.
- Text search fallback for languages without tree-sitter extractors.

### Step 6: Configuration & UX ✅

**What was built**:

- `config.py` — Layered TOML config via pydantic-settings.
- `ThinkingEffort` enum mapped to provider-specific settings.
- Tool call events rendered in TUI panels.
- Token usage tracking (`usage.py`) with context window
  awareness and threshold warnings.

### Language Plugin System ✅

**What was built**:

- `plugins/` — pluggy-based plugin system.
- 10 language plugins: Python, JavaScript, Go, Rust, C, C++,
  Java, Ruby, Bash, plus grammars for JSON/YAML/TOML.
- Optional grammar packages in `[project.optional-dependencies]`.

### Review Draft & GitHub Write Path ✅

**What was built**:

- `engine/draft_cmd.py` — `/draft` command with `sync`, `post`,
  `clear` subcommands.
- Draft persistence to `.rbtr/REVIEW-DRAFT-<pr>.toml`.
- Bidirectional sync with GitHub pending reviews.
- Atomic review posting via `create_review` API.
- LLM tools: `add_review_comment`, `edit_review_comment`,
  `remove_review_comment`, `set_review_summary`.
- GitHub suggestion blocks (`suggestion` parameter).
- PR discussion tool (`get_pr_discussion`).

### Session Persistence ✅ (fragment-level rewrite)

**What was built** (see `TODO-parts.md` for full details):

- **Fragment-level schema** (`sessions/sql/schema.sql`): single
  `fragments` table. Each `ModelMessage` → 1 message-level row +
  N content rows (one per PydanticAI part). 14 SQL files.
  Schema v3 with `cache_read_tokens`/`cache_write_tokens` columns.
- **Serialisation** (`sessions/serialise.py`): `FragmentKind`
  StrEnum, `Fragment` frozen dataclass, `SessionContext`.
  No custom data models — `TypeAdapter[ModelMessage]` for headers,
  `TypeAdapter[AnyPart]` for parts. `reconstruct_message()` merges
  header + parts.
- **Store** (`sessions/store.py`): `SessionStore` with full public
  API — `save_messages`, `save_input`, `begin_response` →
  `ResponseWriter`, `compact_session`, `load_messages`,
  `load_message_ids`, `list_sessions`, `search_history`,
  `token_stats`, `delete_*`, `new_id`. Sync-only (SQLite WAL <1ms).
  All write methods accept `ModelMessage` objects — serialisation
  fully internal.
- **Streaming persistence**: `ResponseWriter` encapsulates the
  lifecycle. `add_part`/`finish_part`/`finish` — engine never sees
  fragment IDs or SQL. `finish()` accepts `cost`, `input_tokens`,
  `output_tokens`, `cache_read_tokens`, `cache_write_tokens`.
- **DB as sole source of truth**: `message_history` removed from
  `EngineState`. Every read goes through `store.load_messages()`.
  `demote_thinking` applied as read-time transformation via
  `simplify_history` parameter — DB keeps originals intact.
- **Engine integration**: `engine/save.py` deleted. All persistence
  via store's public API. `_sync_store_context()` pushes metadata
  at task start. Streaming via `ResponseWriter` in `_do_stream`.
- **`/session` command**: `list`, `all`, `info`, `resume`, `delete`,
  `purge`. Prefix-match on session IDs.
- **Input history from DB**: `search_history()` with prefix search
  across all sessions. Covers LLM prompts, `/commands`, `!shell`.

### Context Compaction ✅

**What was built** (see `TODO-parts.md` for full details):

- **Compaction logic** (`engine/compact.py`): `compact_history_async`
  (async) and `compact_history` (sync wrapper). Splits history
  via `split_history()`, serialises old turns, sends to LLM for
  summarisation.
- **DB integration**: `compact_session()` inserts summary, marks
  old messages with `compacted_by` FK. `load_messages()` filters
  `compacted_by IS NULL`. FK cascade on delete.
- **Four trigger paths**: post-turn auto (>85% context), mid-turn
  (breaks `agent.iter()`, compacts once, resumes), on overflow
  error, manual `/compact`.
- **Mid-turn compaction**: After `CallToolsNode` with tool calls,
  checks `context_used_pct >= auto_compact_pct`. Breaks loop,
  compacts, reloads from DB, resumes with `prompt=None`.
  At most once per turn.
- **TUI compaction panel**: `CompactionStarted`/`CompactionFinished`
  produce dedicated `"queued"` variant panels flushed to scrollback.
- **Binary search for large conversations**: Finds largest prefix
  that fits in context, pushes rest into kept portion.
- **Configurable**: `auto_compact_pct`, `keep_turns`,
  `reserve_tokens`, `summary_max_chars`.

### Token & Cost Stats Infrastructure ✅

**What was built**:

- **Token columns on all responses**: `input_tokens`, `output_tokens`,
  `cache_read_tokens`, `cache_write_tokens` populated on every
  `response-message` row — both batch-saved (via `save_messages`)
  and streamed (via `ResponseWriter.finish()`).
- **`tool_name` on retry-prompt rows**: `RetryPromptPart.tool_name`
  extracted to the `tool_name` column for per-tool failure stats.
- **`token_stats()` store method**: Single query returning
  `TokenStats` dataclass with lifetime vs active totals for all
  token types, cost, message counts, and compaction count.
  SQL in `session_token_stats.sql`.
- **Denormalised columns**: `session_label`, `repo_owner`,
  `repo_name`, `model_name`, `tool_name`, all token columns, `cost`
  on every row — most stats are single-query `GROUP BY` operations.

---

## What's next

### `/stats` command

Show session dashboard — tokens (total/active split), cost,
tool call frequency, file touches. See `TODO-stats.md` for full
design.

**Status**: Data infrastructure complete (Phase 1 done — token
columns, cache columns, tool_name on retries, `token_stats()`
method). Remaining: `tool_stats()` method, history-walking
helpers, command handler, TUI formatting. See `TODO-stats.md`
Phases 3–7.

### Multi-File Review Workflow

Currently `/review` selects a target and indexes it.
A guided review workflow could:

- Walk through changed files one by one.
- Summarise each file's changes before diving in.
- Track which files have been reviewed and which are pending.
- Aggregate findings into a structured review summary.

### Diff-Aware Context

The agent has `diff` and `changed_symbols` tools but lacks
tight integration between PR diffs and the index:

- Automatically feed the diff to the agent on `/review`.
- Map diff hunks to indexed chunks for precise context.
- Highlight which indexed symbols are affected by the change.

### Quality of Life

- **Clipboard integration** — `_copy_to_clipboard` exists but
  isn't wired to a command or keybinding.
- **Multiple review targets** — compare across branches or PRs.
- **File watching** — re-index on file changes during review.
- **Export** — CSV/JSON export of session stats and conversation
  history.

---

## Resolved Decisions

These were open questions in the original plan:

1. **LLM provider**: Multi-provider via `/connect`.
   Claude (OAuth), ChatGPT (OAuth), OpenAI (API key),
   generic endpoints. Model selection via `/model`.
   Config default via `config.model`.

2. **Embedding provider**: Local via llama-cpp-python with
   GGUF models from Hugging Face Hub. No API dependency.

3. **Storage backend**: DuckDB (not LanceDB). Handles chunks,
   edges, file snapshots, full-text search, and vector similarity
   in one embedded database.

4. **Graph database**: Not needed. Structural queries work well
   as SQL over the edges table in DuckDB.

5. **Session persistence**: Fragment-level schema in SQLite.
   Single `fragments` table, one row per PydanticAI part.
   Streaming persistence via `ResponseWriter`. DB is the sole
   source of truth — no in-memory cache. Schema v3.

6. **Compaction**: One tier (LLM summarisation). Four trigger
   paths (post-turn, mid-turn, overflow, manual). `compacted_by`
   FK marks old rows. Mid-turn compaction breaks `agent.iter()`.

7. **GitHub write path**: Bidirectional draft sync with atomic
   posting. TOML persistence for crash safety.
