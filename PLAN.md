# Plan

What to build next, in order.

## Step 1: PR Data Fetching & Context

**Goal**: When a PR is selected via `/review`, fetch full PR data
and give the LLM the context it needs.

**What to build**:

- `models.py` — Full models: `PRDetail` (extends PRSummary with body,
  base_branch, commits, changed_files, comments).
- `github/client.py` — `fetch_pr_detail()` — fetch PR metadata,
  commits list, file diffs, review comments.
- `repo.py` — `read_file_at_ref(repo, path, ref)` — read a file blob
  at any ref via pygit2 (no checkout).
  `list_files_at_ref(repo, ref)` — file tree at a ref.
  `get_diff(repo, base_ref, head_ref)` — unified diff between two refs.
- Engine: `/review` fetches and displays PR overview
  (title, author, description, stats, file list).
- Engine: register repo operations as available context for the LLM.

**Verify**: `/review 42` → see PR overview with file list and stats.
Read a file at the PR's head ref.

## Step 2: LLM Agent

**Goal**: PydanticAI agent with access to repo and GitHub operations
as tools.

**What to build**:

- `reviewer.py` — PydanticAI agent. System prompt includes PR context.
  Tools: read file at ref, get diff, list files, search symbols (grep).
  Streaming output to the event queue.
- Engine: plain text input → agent call → `MarkdownOutput` events
  streamed to UI.
- Configuration: model provider selection
  (env var or `.rbtr.toml` when that exists).

**Verify**: Select a PR, ask "what does this PR change?",
get a grounded answer that references actual files.

## Step 3: Repo Operations & Context Retrieval

**Goal**: Richer repo operations for the agent to use as tools.

**What to build**:

- `repo.py` — Import tracing (Python, regex-based),
  surrounding context (enclosing function/class for a diff hunk),
  symbol search (grep/regex for definitions).
- Register as agent tools.

**Verify**: Ask "what calls this function?" and get a grep-based answer.
Ask "show me the context around this change" and get the
enclosing function.

## Step 4: Vector Index (LanceDB)

**Goal**: Semantic search across the repo — when grep isn't enough,
find code by meaning.

**What to build**:

- `index.py` — Chunk files into meaningful units, embed with
  configurable provider, store in LanceDB table on disk.
  Keyed by commit sha, incremental updates.
  `semantic_search(query, top_k)` and `find_related(path, top_k)`.
- Add `lancedb` to dependencies.
- Register as agent tool.

**Verify**: Index a real repo, run semantic search, get relevant results.

## Step 5: Knowledge Graph (Kùzu)

**Goal**: Structural queries — call chains, inheritance,
module dependencies, change impact.

**What to build**:

- `graph.py` — Build graph with tree-sitter (Python first).
  Nodes: Module, Symbol.
  Edges: IMPORTS, DEFINES, CALLS, INHERITS.
  Queries: `get_call_chain`, `get_module_graph`, `get_change_impact`.
- Add `kuzu`, `tree-sitter`, `tree-sitter-python` to dependencies.
- Register as agent tools.

**Verify**: Build graph for a real repo, query call chains
and impact analysis.

## Step 6: Per-Repo Configuration

**Goal**: `.rbtr.toml` for per-repo settings,
`.rbtr/` for cached data.

**What to build**:

- `config.py` — Configuration loading from `.rbtr.toml` + env vars
  via pydantic-settings.
- `.rbtr/` directory creation and gitignore management.
- Move index and graph storage into `.rbtr/`.

**Verify**: Create `.rbtr.toml` with model settings,
verify they're picked up on launch.

## Open Questions

1. **LLM provider**: Which model? Need to configure pydantic-ai.
   Likely env var for now (`RBTR_MODEL`), `.rbtr.toml` later.
2. **Embedding provider**: API-based or local (sentence-transformers)?
   Deferred until Step 4.
