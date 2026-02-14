# rbtr — Interactive PR Review Workbench

## What It Is

A CLI tool where you point at a GitHub repo and interactively review PRs and branches, with an LLM as a research assistant. The LLM helps you understand changes, trace impact, and build a narrative — but you make all the decisions about what to comment on the PR.

## Architecture Principle: Shared Operations

The core operations (read a file at a ref, find usages, get a diff, etc.) are plain functions in `repo.py` and `github.py`. These same functions are:

- Called directly by CLI commands when the user invokes them interactively
- Registered as PydanticAI agent tools so the LLM can call them during a review

There is one implementation per operation. The agent tools are thin registrations that point at the real functions, not reimplementations.

## Configuration & Storage

Each repository gets a `.rbtr/` directory and a `.rbtr.toml` config file at the repo root.

**`.rbtr.toml`** — per-repo configuration, read via pydantic-settings `TomlConfigSettingsSource`.

**`.rbtr/`** — local cache directory (gitignored): index, graph, cache subdirectories.

## Execution Plan

### Step 0: Launch → List Available PRs & Branches

**Goal**: Running `rbtr` (no arguments) inside a cloned GitHub repo detects the repo, connects to GitHub, and lists open PRs and recent branches. The user picks one to start reviewing.

**What to build**:
- `rbtr/config.py` — Configuration loading from `.rbtr.toml` + env vars. Pydantic-settings `BaseSettings` with `TomlConfigSettingsSource`. Ensure `.rbtr/` directory exists and is gitignored.
- `rbtr/models.py` — Pydantic models: `PRSummary` (number, title, author, head branch, updated_at), `BranchSummary` (name, last commit sha, last commit message, updated_at).
- `rbtr/github.py` — Detect repo origin URL from pygit2, parse owner/repo, authenticate via token, list open PRs and recent branches via PyGithub.
- `rbtr/repo.py` — Open local repo via pygit2, extract remote URL, validate it's a GitHub repo.
- Update `rbtr/cli.py` — `rbtr` with no arguments lists PRs and branches. `rbtr review <pr_url_or_number>` goes directly to a specific PR.
- Display the list with rich (simple table/list, no TUI yet).

**Verify**: Run `rbtr` inside a cloned repo → see a numbered list of open PRs and recent branches. Select one by number.

### Step 1: PR Data Fetching & Display

**Goal**: Once a PR is selected, fetch full PR data and display an overview.

**What to build**:
- `rbtr/models.py` — Full models: `PRData`, `Commit`, `FileChange`, `Comment`, `Discussion`.
- `rbtr/github.py` — Fetch PR metadata, commits list, file diffs, review comments.
- Rich-formatted PR overview in the terminal: title, author, description, stats, commit list, file list.

**Verify**: Select a PR from Step 0 → see a rich overview with all metadata.

### Step 2: Repo Operations & Context Retrieval

**Goal**: Read files at any ref, trace imports, find symbols — the foundation for understanding changes in context.

**What to build**:
- `rbtr/repo.py` — File tree at a ref, file read at any ref (blob reads via pygit2, no checkout), import tracing (Python, regex-based), surrounding context (enclosing function/class for a diff hunk), symbol search (grep/regex for definitions).

**Verify**: Open a real repo, read files at base and head refs, trace imports for a changed file, find symbol definitions.

### Step 3: Vector Index (LanceDB)

**Goal**: Semantic search across the repo — when grep isn't enough, find code by meaning.

**What to build**:
- `rbtr/index.py` — Chunk files into meaningful units, embed with configurable provider, store in LanceDB table on disk. Keyed by commit sha, incremental updates. `semantic_search(query, top_k)` and `find_related(path, top_k)`.

**Verify**: Index a real repo, run semantic search, get relevant results.

### Step 4: Knowledge Graph (Kùzu)

**Goal**: Structural queries — call chains, inheritance, module dependencies, change impact.

**What to build**:
- `rbtr/graph.py` — Build graph with tree-sitter (Python first). Nodes: Module, Symbol, Commit, PRComment. Edges: IMPORTS, DEFINES, CALLS, INHERITS, MODIFIES, COMMENTS_ON. Queries: `get_call_chain`, `get_module_graph`, `get_change_impact`, `get_inheritance_tree`, `get_pr_overlay`.

**Verify**: Build graph for a real repo, query call chains and impact analysis.

### Step 5: LLM Agent

**Goal**: PydanticAI agent with access to all operations as tools.

**What to build**:
- `rbtr/reviewer.py` — Agent with tools from repo.py, github.py, index.py, graph.py. System prompt grounded in actual code. Called on-demand when user asks a question.

**Verify**: Ask a question about a real PR, get a grounded answer using semantic and structural context.

### Step 6: Interactive TUI (Textual)

**Goal**: Full interactive interface for navigating and reviewing a PR.

**What to build**:
- `rbtr/tui.py` — Textual app with views: PR Overview, Commit view, File view, Diff view (syntax-highlighted), Discussion view. LLM questions, semantic search, and graph queries available everywhere.

**Verify**: End-to-end on a real PR — navigate commits, view diffs, ask questions, search, graph queries.

## Open Questions

1. **LLM provider**: Which model? Need to configure pydantic-ai.
2. **Embedding provider**: API-based or local (sentence-transformers)?
3. **Local repo assumption**: Require the user to already have the repo cloned, or clone on the fly?
