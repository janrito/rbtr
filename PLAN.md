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

- `engine/tools.py` — 13 tools registered on the agent.
  Index tools (hidden when no index): `search_symbols`,
  `search_codebase`, `search_similar`, `get_dependents`,
  `get_callers`, `get_blast_radius`, `read_symbol`,
  `list_indexed_files`, `detect_language`, `semantic_diff`.
  Git tools (hidden when no repo): `diff`, `commit_log`.
- Prepare functions (`_require_index`, `_require_repo`) dynamically
  hide tools based on session state.

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
  IndexStats), `sql/` (20 SQL files).
- `/index` command with subcommands: `status`, `clear`,
  `rebuild`, `prune`, `model`.
- Background indexing on `/review` with progress events
  (`IndexStarted`, `IndexProgress`, `IndexReady`).

### Step 5: Structural Analysis ✅ (revised approach)

**Deviation from original plan**: Kùzu graph database was dropped.
Structural queries (call chains, import graphs, impact analysis)
are implemented as SQL queries in DuckDB over the `edges` table,
exposed as agent tools (`get_callers`, `get_dependents`,
`get_blast_radius`).

**What was built**:

- Edge types: `IMPORTS`, `DEFINES`, `CALLS`, `TESTS`, `DOCUMENTS`.
- Tree-sitter import extraction per language plugin.
- Text search fallback for languages without tree-sitter extractors.
- `semantic_diff` tool for comparing changes at the chunk level.

### Step 6: Configuration & UX ✅

**What was built**:

- `config.py` — Layered TOML config (class defaults → user →
  workspace `.rbtr/config.toml`) via pydantic-settings.
  Nested config models: `EndpointConfig`, `GithubConfig`,
  `IndexConfig`, `ToolsConfig`, `LogConfig`, `TuiConfig`,
  `OAuthConfig`, `ProvidersConfig` (Claude, ChatGPT, OpenAI,
  GitHub, Endpoint).
- `ThinkingEffort` enum (`low`/`medium`/`high`/`max`) mapped
  to provider-specific settings.
- Tool call events (`ToolCallStarted`, `ToolCallFinished`)
  rendered in the TUI.
- Token usage tracking (`usage.py`) with context window
  awareness and threshold warnings.

### Language Plugin System ✅

**What was built**:

- `plugins/` — pluggy-based plugin system.
  `hookspec.py` defines the interface, `manager.py` manages
  registration, `defaults.py` provides built-in defaults.
- 10 language plugins: Python, JavaScript, Go, Rust, C, C++,
  Java, Ruby, Bash, plus grammars for JSON/YAML/TOML.
- Each plugin declares: file detection, tree-sitter grammar,
  scope queries, import extraction, chunk kinds.
- Optional grammar packages in `[project.optional-dependencies]`.

---

## What's next

### GitHub Integration: Write Path

The read path is complete (list PRs, fetch metadata, read branches).
The write path is missing — rbtr cannot yet post review comments
back to GitHub.

- `github/client.py` — `create_review()`, `post_review_comment()`,
  `post_pr_comment()`.
- A `/submit` or `/post` command (or inline workflow) to let the
  reviewer push drafted comments to the PR.
- Inline comment drafting: the LLM drafts a comment for a specific
  file/line range, the reviewer edits or approves, then it's posted.

### Multi-File Review Workflow

Currently `/review` selects a target and indexes it.
A guided review workflow could:

- Walk through changed files one by one.
- Summarise each file's changes before diving in.
- Track which files have been reviewed and which are pending.
- Aggregate findings into a structured review summary.

### Diff-Aware Context

The agent has `diff` and `semantic_diff` tools but lacks
tight integration between PR diffs and the index:

- Automatically feed the diff to the agent on `/review`.
- Map diff hunks to indexed chunks for precise context.
- Highlight which indexed symbols are affected by the change.

### Embedding Model Configuration

Embeddings currently use a hardcoded local GGUF model.

- Configurable embedding model (local or API-based).
- `/index model` subcommand exists but could support more models.
- Re-embedding on model change (currently requires `/index rebuild`).

### Quality of Life

- **Clipboard integration** — `_copy_to_clipboard` exists but
  isn't wired to a command or keybinding.
- **Session persistence** — conversations are lost on exit.
  Save/restore sessions to `.rbtr/sessions/`.
- **Multiple review targets** — compare across branches or PRs.
- **File watching** — re-index on file changes during review.

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
   as SQL over the edges table in DuckDB. Avoids an extra
   dependency (Kùzu) without losing capability.
