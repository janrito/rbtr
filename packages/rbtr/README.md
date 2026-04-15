# rbtr

A language-agnostic structural code index. rbtr decomposes
source files into functions, classes, methods, and imports,
connects them with a dependency graph, and makes them
searchable through name matching, keyword search, and
semantic similarity — fused into a single ranked result.

rbtr is a library and a CLI. The CLI builds and queries the
index. The library provides the same functionality as a
Python API for integration into editors, agents, and CI
pipelines.

## Install

```bash
pip install rbtr          # from PyPI
uv add rbtr               # as a project dependency
```

For development (from a local clone):

```bash
uv sync --extra languages
```

## Quick start

Build an index for a repository and search it:

```bash
cd /path/to/your/repo

# Build the index (tree-sitter extraction + embeddings)
rbtr build
# ref=HEAD  395/415 files (0 skipped)  6663 chunks  3001 edges  42.3s

# Search by name, keyword, or concept
rbtr search "retry logic"
rbtr search "HttpClient"
rbtr search "how does authentication work"
```

## Commands

### `rbtr build`

Build or update the index for a repository.

```bash
rbtr build                    # index HEAD
rbtr build --ref main         # index a specific ref
rbtr build --repo-path /path  # different repository
```

Output:

```text
ref=HEAD  395/415 files (0 skipped)  6663 chunks  3001 edges  42.3s
```

In a terminal, a progress bar shows parsing and embedding
phases. Subsequent builds are incremental — unchanged files
(by blob SHA) are skipped.

### `rbtr search <query>`

Search the code index by name, keyword, or natural-language
concept.

```bash
rbtr search "IndexStore"          # name match
rbtr search "retry timeout"       # keyword search
rbtr search "how does auth work"  # semantic search
rbtr search "config" --limit 5    # limit results
```

TTY output:

```text
packages/rbtr/src/rbtr/config.py:39-77  class  Config
  1.69  [c26a092e]
    class Config(BaseSettings):
        """Schema and defaults — no file sources."""

        model_config = SettingsConfigDict(env_prefix="RBTR_")
    … 39 lines total
```

JSON output (piped or `--json`):

```json
{"score":1.6875,"lexical":0.0,"semantic":0.0,"name":0.0,
 "kind_boost":1.0,"file_penalty":1.0,"importance":1.0,
 "proximity":1.0,
 "chunk":{"id":"c26a092e","file_path":"packages/rbtr/src/rbtr/config.py",
          "name":"Config","kind":"class","line_start":39,"line_end":77,
          "content":"class Config(BaseSettings):..."}}
```

Search fuses three channels:

- **Name matching** — case-insensitive token matching on
  chunk names. Finds identifiers like `IndexStore`,
  `build_index`.
- **BM25 keyword search** — full-text search over tokenised
  content. Code-aware tokenisation splits `camelCase` and
  `snake_case`.
- **Semantic similarity** — cosine distance between query
  and chunk embeddings (bge-m3, quantised GGUF, runs
  locally on Metal/CPU — no API calls).

Post-fusion adjustments:

- **Kind boost** — classes and functions rank above imports.
- **File category** — source files rank above tests.
- **Importance** — chunks with more inbound dependency edges
  rank higher.

### `rbtr read-symbol <name>`

Read the full source of a symbol by name.

```bash
rbtr read-symbol Config
rbtr read-symbol IndexStore.from_config
```

TTY output shows syntax-highlighted source with line
numbers. JSON output returns the full `Chunk` model.

### `rbtr list-symbols <file>`

Table of contents for a file — one line per symbol.

```bash
rbtr list-symbols src/rbtr/config.py
```

```text
   39-77   class      Config
   83-88   function   _toml_file
   94-121  class      RenderedConfig
   98-117  method     settings_customise_sources  (RenderedConfig)
  119-121  method     reload  (RenderedConfig)
```

### `rbtr find-refs <symbol>`

Find symbols that reference a given symbol via the
dependency graph (import edges, test edges, doc edges).

```bash
rbtr find-refs Config
```

### `rbtr changed-symbols --base <ref> --head <ref>`

Show symbols in files that changed between two refs.

```bash
rbtr changed-symbols --base main --head feature
```

### `rbtr status`

Show index status.

```bash
rbtr status
# ✓ 6663 chunks  .rbtr/index/index.duckdb
```

## Output modes

Every command outputs the **same data** in both modes — the
pydantic model is the single source of truth.

- **TTY** (interactive terminal): rich-formatted text with
  syntax highlighting, coloured scores, and progress bars.
- **Piped / `--json`**: one JSON object per line (NDJSON).

Force JSON in a terminal:

```bash
rbtr --json search "config"
```

Or via environment variable:

```bash
RBTR_JSON_OUTPUT=true rbtr search "config"
```

## Configuration

User-level config at `~/.rbtr/config.toml`. Workspace
overlay at `.rbtr/config.toml` (nearest to CWD wins,
monorepo-friendly). Environment variables with `RBTR_`
prefix override both.

```toml
# ~/.rbtr/config.toml
db_dir = "${WORKSPACE}/index"
max_file_size = 524288
chunk_lines = 50
chunk_overlap = 5
embedding_model = "gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
embedding_batch_size = 32
include = [".rbtr/notes/*", ".rbtr/AGENTS.md"]
extend_exclude = [".rbtr/"]
```

| Field                  | Default                                   | Description                         |
| ---------------------- | ----------------------------------------- | ----------------------------------- |
| `db_dir`               | `${WORKSPACE}/index`                      | DuckDB index directory              |
| `max_file_size`        | `524288`                                  | Skip files larger than this (bytes) |
| `chunk_lines`          | `50`                                      | Lines per plaintext chunk           |
| `chunk_overlap`        | `5`                                       | Overlap between adjacent chunks     |
| `embedding_model`      | `gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf` | HuggingFace model for embeddings    |
| `embedding_batch_size` | `32`                                      | Batch size for embedding inference  |
| `include`              | `[".rbtr/notes/*", ".rbtr/AGENTS.md"]`    | Extra globs to index                |
| `extend_exclude`       | `[".rbtr/"]`                              | Extra globs to exclude              |
| `json_output`          | `false`                                   | Force JSON output                   |

## Supported languages

rbtr uses [tree-sitter] for structural extraction. Languages
with full support get symbol-level chunks, import metadata
for the dependency graph, and scope detection for methods
inside classes. All other file types get line-based plaintext
chunking.

[tree-sitter]: https://tree-sitter.github.io/tree-sitter/

| Language   | Structural | Imports | Extensions             |
| ---------- | :--------: | :-----: | ---------------------- |
| Python     |     ✓      |    ✓    | `.py`, `.pyi`          |
| TypeScript |     ✓      |    ✓    | `.ts`, `.tsx`          |
| JavaScript |     ✓      |    ✓    | `.js`, `.jsx`, `.mjs`  |
| Go         |     ✓      |    ✓    | `.go`                  |
| Rust       |     ✓      |    ✓    | `.rs`                  |
| C          |     ✓      |    ✓    | `.c`, `.h`             |
| C++        |     ✓      |    ✓    | `.cpp`, `.cc`, `.hpp`  |
| Java       |     ✓      |    ✓    | `.java`                |
| Ruby       |     ✓      |    ✓    | `.rb`                  |
| Bash       |     ✓      |         | `.sh`, `.bash`, `.zsh` |
| Markdown   |  headings  |         | `.md`                  |
| C#         |  grammar   |         | `.cs`                  |
| Kotlin     |  grammar   |         | `.kt`, `.kts`          |
| Scala      |  grammar   |         | `.sc`, `.scala`        |
| Swift      |  grammar   |         | `.swift`               |
| CSS        |  grammar   |         | `.css`                 |
| HTML       |  grammar   |         | `.htm`, `.html`        |
| JSON       |  grammar   |         | `.json`                |
| TOML       |  grammar   |         | `.toml`                |
| YAML       |  grammar   |         | `.yaml`, `.yml`        |
| HCL        |  grammar   |         | `.hcl`, `.tf`          |

**Structural** = symbol extraction via tree-sitter queries.
**Imports** = structural import metadata for dependency graph edges.
**grammar** = tree-sitter grammar available but no query yet (line-based chunking).
**headings** = custom heading-hierarchy chunking.

### Adding a language

External plugins register via the `rbtr.languages` entry
point:

```toml
# your-plugin/pyproject.toml
[project.entry-points."rbtr.languages"]
kotlin = "rbtr_kotlin:KotlinPlugin"
```

See [ARCHITECTURE.md](ARCHITECTURE.md#language-plugins) for
the plugin API and progressive capability model.

## Graceful degradation

- **No grammar installed** for a language → line-based
  plaintext chunking (every file gets indexed).
- **No embedding model** (missing GGUF, GPU init failure) →
  structural index works, semantic search signal is skipped.
- **Empty repository** → `rbtr status` reports no index.

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [just](https://github.com/casey/just) for task running

### Setup

```bash
git clone <repo-url>
cd rbtr
uv sync --extra languages
```

### Running checks

```bash
just check        # lint + typecheck + test (all packages)
just test-rbtr    # tests for rbtr only
just test-legacy  # tests for rbtr-legacy only
just fmt          # auto-fix and format
just lint         # lint only
just typecheck    # mypy only
```

### Project structure

```text
packages/rbtr/
├── src/rbtr/
│   ├── cli/              # CLI entry point, output rendering
│   │   ├── __init__.py   # subcommands, root command, main()
│   │   ├── models.py     # CLI-specific output schemas
│   │   └── output.py     # emit(), rich formatting
│   ├── index/            # DuckDB index, search, orchestration
│   │   ├── store.py      # IndexStore — DuckDB operations
│   │   ├── orchestrator.py # build_index(), file routing
│   │   ├── search.py     # score fusion, ScoredResult
│   │   ├── models.py     # Chunk, Edge, IndexStats
│   │   ├── treesitter.py # tree-sitter symbol extraction
│   │   ├── chunks.py     # plaintext + markdown chunking
│   │   ├── edges.py      # dependency graph inference
│   │   ├── embeddings.py # local GGUF embedding model
│   │   ├── tokenise.py   # code-aware BM25 tokenisation
│   │   └── arrow.py      # PyArrow bulk insert helpers
│   ├── languages/        # pluggy language plugin system
│   │   ├── __init__.py   # LanguageManager singleton
│   │   ├── hookspec.py   # LanguageRegistration, plugin API
│   │   ├── defaults.py   # grammar-only registrations
│   │   ├── python.py     # Python plugin (full support)
│   │   ├── javascript.py # JS/TS plugin (full support)
│   │   └── ...           # go, rust, java, c, cpp, ruby, bash
│   ├── config.py         # pydantic-settings configuration
│   ├── git.py            # pygit2 wrappers (index-only)
│   ├── workspace.py      # .rbtr/ discovery, path resolution
│   └── errors.py         # RbtrError base exception
└── src/tests/            # 740+ tests
```

### Search quality

```bash
just eval-search                    # curated query evaluation
just tune-search                    # grid-search fusion weights
just bench-search                   # replay real queries from sessions
```

### Architecture reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for internals —
storage, search fusion, language plugins, dependency graph,
and design decisions.

## License

MIT
