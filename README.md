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
persisted to `config.toml` across sessions.

## Commands

| Command                | Description                             |
| ---------------------- | --------------------------------------- |
| `/help`                | Show available commands                 |
| `/review`              | List open PRs and branches              |
| `/review <id>`         | Select a PR or branch for review        |
| `/connect <service>`   | Authenticate with a service             |
| `/model`               | List available models from all providers|
| `/model <provider/id>` | Set the active model                    |
| `/index`               | Show index status, clear, rebuild       |
| `/new`                 | Start a new conversation                |
| `/quit`                | Exit (also `/q`)                        |

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

| Context            | Example             | Completes                                |
| ------------------ | ------------------- | ---------------------------------------- |
| Slash commands     | `/rev` → `/review`  | Command names                            |
| Command arguments  | `/connect c`        | Provider names, model IDs                |
| Shell commands     | `!git ch`           | Bash programmable completion (branches, flags) |
| File paths         | `!cat ~/Doc`        | Directories and files (expands `~`)      |
| Executables        | `!my`               | Commands found in `PATH`                 |

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

| Field        | Example    | Description                                            |
| ------------ | ---------- | ------------------------------------------------------ |
| `\|7\|`      |            | Messages in this conversation                          |
| `12%`        |            | Last request size as % of context window               |
| `of 200k`    |            | Model's context window size                            |
| `↑ 24.3k`    |            | Cumulative input tokens                                |
| `↓ 1.2k`     |            | Cumulative output tokens                               |
| `↯ 18.0k`    |            | Cumulative cache-read tokens (hidden when zero)        |
| `$0.0450`    |            | Cumulative cost                                        |

**Colour signals help you manage context:**

- **Message count** — gray (≤ 25, fresh), yellow (26–50, consider `/new`),
  red (51+, very long).
- **Context %** — green (< 70%), yellow (70–89%), red (90%+).
- **Dimmed values** indicate the model didn't report pricing metadata
  (common with custom endpoints). Context window falls back to 128k
  and cost shows `$0.0000`.

`/new` resets all counters.

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

Once the index is ready, the LLM can use these tools:

| Tool                | Description                                     |
| ------------------- | ----------------------------------------------- |
| `search_symbols`    | Find symbols by name (fuzzy)                    |
| `search_codebase`   | BM25 keyword search across all chunks           |
| `search_similar`    | Semantic embedding search                       |
| `get_callers`       | Find tests and docs that reference a symbol     |
| `get_dependents`    | Find what imports a symbol                      |
| `get_blast_radius`  | All inbound edges for a file                    |
| `read_symbol`       | Read the full source of a symbol                |
| `semantic_diff`     | Structural diff: added/removed/modified symbols |
| `diff`              | Unified text diff                               |
| `commit_log`        | Commit log between base and head                |
| `detect_language`   | Identify a file's language                      |

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

| Subcommand              | Description                              |
| ----------------------- | ---------------------------------------- |
| `/index`                | Show index status (chunks, edges, size)  |
| `/index clear`          | Delete the index database                |
| `/index rebuild`        | Clear and re-index from scratch          |
| `/index prune`          | Remove orphan chunks not in any snapshot |
| `/index model`          | Show current embedding model             |
| `/index model <id>`     | Switch embedding model and re-embed      |

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
include = []                                            # force-include globs
extend_exclude = [".rbtr"]                              # exclude globs (on top of .gitignore)
```

The index is persistent — subsequent `/review` runs on the same
repo skip unchanged files (keyed by git blob SHA).

### Graceful degradation

- **No grammar installed** for a language → falls back to
  line-based plaintext chunking.
- **No embedding model** (missing GGUF, GPU init failure) →
  structural index works, only `search_similar` is degraded.
- **Slow indexing** → review starts immediately, index catches
  up in a background thread.

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
