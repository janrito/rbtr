# Architecture

Technical reference for contributors and maintainers.
For usage, see [README.md](README.md).

---

## Overview

rbtr is a structural code index — it decomposes source
files into named chunks (functions, classes, methods,
imports), connects them with a dependency graph, and makes
them searchable through three fused retrieval channels.

The design prioritises three properties:

1. **Language-agnostic.** Every file gets indexed. Languages
   with tree-sitter grammars get structural extraction;
   everything else gets line-based chunking. No language is
   special-cased outside the plugin system.
2. **Local and offline.** Embeddings run on-device (GGUF
   model on Metal/CPU). The index is a DuckDB file. No API
   calls, no network dependency.
3. **Incremental.** Builds are keyed by git blob SHA. Only
   changed files are re-extracted. Embeddings persist across
   builds.

### Layers

```text
┌──────────────────────────────────────────────┐
│  CLI (cli/)                                  │
│  pydantic-settings CliApp, rich output       │
├──────────────────────────────────────────────┤
│  Index (index/)                              │
│  DuckDB store, search fusion, orchestration  │
├──────────────────────────────────────────────┤
│  Languages (languages/)                      │
│  pluggy plugins, tree-sitter, chunking       │
├──────────────────────────────────────────────┤
│  Git (git.py)                                │
│  pygit2 object store, blob iteration         │
├──────────────────────────────────────────────┤
│  Config (config.py, workspace.py)            │
│  TOML + env, workspace discovery             │
└──────────────────────────────────────────────┘
```

Each layer depends only on the layers below it. The CLI
imports index and languages; the index imports languages
and git; languages and git import only config. There are
no upward dependencies.

### Data flow: `rbtr build`

```text
git.walk_tree(repo, ref)
  → FileEntry(path, blob_sha, content)
    → LanguageManager.detect_language(path)
      → tree-sitter extraction  OR  plaintext chunking
        → list[Chunk]
          → IndexStore.upsert_chunks()
            → edges.infer_edges(chunks)
              → IndexStore.upsert_edges()
                → embeddings.embed_missing(store)
```

### Data flow: `rbtr search`

```text
query
  → classify_query(query) → IDENTIFIER | CONCEPT | PATTERN
    → BM25 full-text search      → scored list
    → semantic cosine similarity  → scored list
    → name substring matching     → scored list
      → normalise + fuse (weighted combination)
        → post-fusion: kind boost, file penalty, importance
          → ranked list[ScoredResult]
```

---

## Index storage

The index lives in a single DuckDB file
(`.rbtr/index/index.duckdb`). DuckDB was chosen over
SQLite for three reasons: native full-text search (BM25),
PyArrow integration for bulk inserts, and array columns
for embeddings without serialisation overhead.

### Schema

Two primary tables and one metadata table:

**`chunks`** — one row per indexed symbol or text block.

| Column           | Type       | Purpose                          |
| ---------------- | ---------- | -------------------------------- |
| `id`             | VARCHAR PK | Content-addressed hash           |
| `commit_sha`     | VARCHAR    | Git commit this chunk belongs to |
| `blob_sha`       | VARCHAR    | Git blob SHA (dedup key)         |
| `file_path`      | VARCHAR    | Relative file path               |
| `kind`           | VARCHAR    | `function`, `class`, `method`, … |
| `name`           | VARCHAR    | Symbol name or heading text      |
| `scope`          | VARCHAR    | Enclosing scope (class name)     |
| `content`        | TEXT       | Full source text                 |
| `content_tokens` | TEXT       | Tokenised content (for BM25)     |
| `name_tokens`    | TEXT       | Tokenised name                   |
| `line_start`     | INTEGER    | Start line (1-indexed)           |
| `line_end`       | INTEGER    | End line (inclusive)             |
| `metadata`       | JSON       | Import metadata (module, names)  |
| `embedding`      | FLOAT[N]   | Semantic embedding vector        |

**`edges`** — directed relationships between chunks.

| Column      | Type    | Purpose                         |
| ----------- | ------- | ------------------------------- |
| `source_id` | VARCHAR | FK to chunks.id (the referrer)  |
| `target_id` | VARCHAR | FK to chunks.id (the referent)  |
| `kind`      | VARCHAR | `imports`, `tests`, `documents` |
| `commit_sha`| VARCHAR | Commit context                  |

**`meta`** — key-value pairs for schema version, embedding
model, and embedding format version.

### Blob deduplication

Chunks are keyed by a content-addressed hash of
`(blob_sha, file_path, line_start, kind, name)`. When two
files share the same blob SHA (e.g. after a rename with no
changes), chunks are extracted once. The orchestrator checks
`blob_sha` against existing chunks before extraction.

### FTS (full-text search)

DuckDB's FTS extension builds an inverted index over
`content_tokens` and `name_tokens`. The `content_tokens`
column contains code-aware tokenised text — `camelCase`
and `snake_case` are split into compound and parts
(see [Tokenisation](#tokenisation)).

A known DuckDB bug causes the FTS schema to persist across
connections while the index data does not. `IndexStore`
works around this with `_purge_stale_fts_schema()` — drop
the schema, checkpoint, close, and reopen.

### Schema versioning

The `meta` table stores `schema_version`. On open,
`_check_schema_version()` compares the stored version
against the current code's `SCHEMA_VERSION`. A mismatch
deletes the database and recreates it — the index is
derived data and can always be rebuilt.

### Embedding versioning

The `meta` table also stores the embedding model ID and a
format version. When either changes, all embeddings are
cleared and recomputed on the next build. This ensures
vectors are always comparable.

---

## Language decomposition

Each file is decomposed into chunks using one of three
strategies, selected by the language plugin:

- **Tree-sitter** — if the plugin provides a grammar and
  query. Extracts functions, classes, methods, and imports
  as structured `Chunk` objects with names, kinds, line
  ranges, and scope.
- **Custom chunker** — if the plugin provides a `chunker`
  function (e.g. markdown heading-hierarchy splitting).
- **Plaintext fallback** — splits the file into fixed-size
  overlapping line chunks. Any file in any language gets
  indexed.

### Tree-sitter extraction

`treesitter.py` runs the plugin's S-expression query against
the parsed tree and maps captures to `Chunk` objects:

- `@function` / `@_fn_name` → `ChunkKind.FUNCTION`
- `@class` / `@_cls_name` → `ChunkKind.CLASS`
- `@method` / `@_method_name` → `ChunkKind.METHOD`
- `@import` → `ChunkKind.IMPORT`

Scope detection uses the plugin's `scope_types` — when a
function node is nested inside a class node, the chunk
gets `scope = "ClassName"` and `kind = METHOD`.

### Markdown chunking

`chunks.py` implements heading-hierarchy chunking for
markdown and RST files. Each heading becomes a chunk whose
content extends to the next heading of equal or higher
level. This preserves document structure better than
fixed-size line splitting.

---

## Dependency graph

`edges.py` infers cross-file relationships from chunk
metadata and content:

- **Import edges** — from tree-sitter import extractors
  (structural) or text-search fallback. Structural
  extraction returns `ImportMeta(module, names, dots)`;
  edges are resolved by matching module paths against
  chunk file paths.
- **Test edges** — `test_foo.py` → `foo.py` by naming
  convention and import analysis.
- **Doc edges** — markdown/RST sections that mention
  function or class names.

The graph powers `find-refs` (find all callers of a
function) and the **importance** signal in search ranking
(chunks with more inbound edges rank higher).

---

## Search fusion

`search.py` fuses three retrieval channels into a single
ranked result list.

### Channels

**Name matching.** Case-insensitive substring and
token-level matching against chunk names. Scores are
binary (1.0 for exact match, scaled for partial). No
external index — runs against in-memory chunk names.

**BM25 keyword search.** DuckDB FTS over `content_tokens`
and `name_tokens`. Code-aware tokenisation means
`camelCase` queries match `camel_case` content.

**Semantic similarity.** Cosine distance between query
embedding and chunk embeddings. Uses a local GGUF model
(bge-m3) — no API calls. Falls back gracefully if the
model is unavailable.

### Fusion

Each channel produces raw scores that are normalised to
[0, 1] via min-max scaling, then combined with a convex
combination. Weights differ by query kind:

- `IDENTIFIER` — heavy on name matching.
- `CONCEPT` — heavy on semantic similarity.
- `PATTERN` — balanced.

`classify_query()` routes each query using heuristics:
dot-separated identifiers, camelCase/snake_case tokens,
natural-language phrases.

### Post-fusion adjustments

After fusion, multipliers adjust the final score:

- **Kind boost** — classes and functions > imports.
- **File category** — source > tests.
- **Importance** — more inbound edges → higher rank.
- **Proximity** — files in a provided changed-file set
  rank higher (used by the review tool for diff-aware
  results).

---

## Tokenisation

`tokenise.py` implements code-aware tokenisation for BM25.
Standard NLP tokenisers break on whitespace and
punctuation — useless for `camelCase`, `snake_case`, and
qualified names like `http.client.HTTPSConnection`.

The tokeniser:

1. Splits on whitespace and punctuation.
2. Splits `camelCase` → `["camel", "Case", "camelCase"]`.
3. Splits `snake_case` → `["snake", "case", "snake_case"]`.
4. Lowercases all tokens.
5. Emits both the compound and its parts.

This means a query for `"camelCase"` matches content
containing `camel_case`, and vice versa.

---

## Embeddings

`embeddings.py` manages a local GGUF embedding model for
semantic search. The model runs on Metal (macOS) or CPU
via `llama-cpp-python`. No API calls.

### Model management

The model is specified in config as a HuggingFace repo ID
with a GGUF filename (e.g.
`gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf`). On first use,
the model is downloaded to the HuggingFace cache directory.
`try_to_load_from_cache` checks for local availability
before downloading.

### Batch processing

`embed_missing()` finds all chunks without embeddings and
processes them in batches of `embedding_batch_size`. The
orchestrator calls this after all chunks are extracted.
Progress is reported via the `on_embed_progress` callback.

### Graceful degradation

If the embedding model fails to load (missing file, GPU
init error), the index works without semantic search — the
weight is redistributed to name and keyword channels.

---

## Language plugins

Language plugins use [pluggy]. Each plugin implements the
`rbtr_register_languages` hook and returns a list of
`LanguageRegistration` instances.

[pluggy]: https://pluggy.readthedocs.io/

### Registration

`LanguageRegistration` is a frozen dataclass with
progressive capability — each field unlocks more analysis:

| Field              | Capability                             |
| ------------------ | -------------------------------------- |
| `id` + `extensions`| File detection + line-based chunks     |
| `chunker`          | Custom chunking (no grammar needed)    |
| `grammar_module`   | Tree-sitter parsing                    |
| `query`            | Structural symbol extraction           |
| `import_extractor` | Structural import metadata for edges   |
| `scope_types`      | Method-in-class scoping                |
| `pygments_lexer`   | Syntax highlighting in CLI output      |

A minimal plugin:

```python
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

class KotlinPlugin:
    @hookimpl
    def rbtr_register_languages(self):
        return [LanguageRegistration(
            id="kotlin",
            extensions=frozenset({".kt", ".kts"}),
        )]
```

### Plugin precedence

The `LanguageManager` registers plugins in order:

1. `DefaultsPlugin` — grammar-only and detection-only
   registrations (lowest priority).
2. Built-in plugins — Python, JS/TS, Go, Rust, Java, C,
   C++, Ruby, Bash (override defaults).
3. External plugins via the `rbtr.languages` entry-point
   group (highest priority).

Later registrations override earlier ones for the same
language ID.

### Shared utilities

`hookspec.py` exports helpers for plugin authors:

- `parse_path_relative(specifier)` — parse `./`/`../`
  prefixes for filesystem-relative imports (JS, TS).
- `collect_scoped_path(node)` — collect `::` or `.`
  separated segments from a tree-sitter node (Rust, Java).

### Adding a language

1. Create `languages/<name>.py` with a plugin class.
2. Implement `@hookimpl rbtr_register_languages()`.
3. Register in `languages/__init__.py`
   (`_register_builtins`).
4. Add the grammar package to `pyproject.toml` optional
   deps under `[project.optional-dependencies] languages`.

Use `languages/bash.py` as a minimal example (functions
only). Use `languages/python.py` for a full example with
import extractor, scope types, and queries.

---

## CLI architecture

The CLI uses pydantic-settings `CliApp` for subcommand
parsing. The root command class (`Rbtr`) inherits from
`RenderedConfig`, so config fields are exposed as CLI
flags, and TOML/env sources are merged automatically.

### Output contract

All CLI output flows through `emit()` in `cli/output.py`.
The function dispatches to JSON (piped or `--json`) or
rich-formatted text (TTY). Both modes present the **same
data** — the pydantic model is the single source of truth.

Output models come from two sources:

- **Index models** (`Chunk`, `Edge`, `ScoredResult`,
  `IndexStats`) — used directly. Subcommands call
  `emit(chunk)`, not `emit(ChunkView(chunk))`.
- **CLI-specific models** (`BuildResult`, `IndexStatus`)
  — composites that don't exist in the index layer.

Internal fields on `Chunk` that are storage-only
(`content_tokens`, `name_tokens`, `embedding`) are marked
`Field(exclude=True)` and never serialised.

### Rich rendering

TTY output uses `rich.Console`, `rich.Syntax`, and
`rich.Text` for coloured, syntax-highlighted output.
Help text uses `rich-argparse` for formatted flag and
subcommand display.

The `compact` parameter on `emit()` controls whether
chunk rendering shows full source (read-symbol) or a
one-line summary (list-symbols, changed-symbols).

---

## Git interface

`git.py` wraps pygit2 for read-only access to the git
object store. All file reads go through blob objects at
exact commit SHAs — the working tree is never read for
repository files.

Key functions:

- `open_repo(path)` — open a pygit2 Repository.
- `walk_tree(repo, ref)` — yield `FileEntry(path,
  blob_sha, content)` for every file in a commit tree.
- `changed_files(repo, base, head)` — set of file paths
  that differ between two refs.
- `resolve_ref(repo, ref)` — resolve a ref string to a
  commit SHA.

The index only needs tree walking and blob reading.
Review-specific git operations (diffs, logs, line
translation) live in the legacy package.

---

## Configuration

`config.py` uses pydantic-settings with three sources,
merged in order (each overrides the previous):

1. **Class defaults** — field defaults on `Config`.
2. **User TOML** — `~/.rbtr/config.toml`.
3. **Workspace TOML** — `.rbtr/config.toml` (nearest to
   CWD, monorepo-friendly).
4. **Environment variables** — `RBTR_` prefix.

`RenderedConfig` extends `Config` with the TOML sources.
The module-level `config` singleton is the merged view.
`config.reload()` re-reads all sources in place.

`workspace.py` discovers the `.rbtr/` directory by walking
from CWD to the git root. `resolve_path()` expands
`${WORKSPACE}` placeholders in config values.

---

## Design decisions

### DuckDB over SQLite

The index needs BM25 full-text search, array columns for
embeddings, and bulk insert performance. DuckDB provides
all three natively. SQLite would require FTS5 (limited
ranking control), blob-serialised embeddings, and
row-by-row inserts.

### pygit2 over git CLI

The index walks entire commit trees and reads blobs by SHA.
Shelling out to `git` would require one subprocess per file.
pygit2 gives direct object-store access with no process
overhead.

### pluggy for language plugins

Language support must be extensible without modifying core
code. pluggy provides hook-based plugin discovery with
precedence ordering and entry-point registration — the same
pattern used by pytest.

### pydantic-settings for CLI + config

The root CLI command inherits from the config class, so
every config field is automatically a CLI flag. TOML, env
vars, and CLI args are merged in one framework. No argparse,
no click, no separate config loading.

### Flat config (no nesting)

The config has no nested sections (`config.chunk_lines`,
not `config.index.chunk_lines`). This keeps env var names
flat (`RBTR_CHUNK_LINES`) and makes the CLI flags
straightforward (`--chunk-lines`).

### Exclude over project

Internal fields on `Chunk` (`content_tokens`,
`name_tokens`, `embedding`) use `Field(exclude=True)`
rather than separate "view" models. The index models are
the public API — consumers get the real objects with
storage internals excluded from serialisation.

### ScoredResult as BaseModel

Search results are `BaseModel`, not dataclasses. This
means they serialise natively through `emit()` with no
wrapper. Score values are full precision in JSON — rounding
happens only in the rich renderer.

### No daemon (yet)

The CLI executes inline — no background process, no IPC.
A daemon that watches for changes and keeps the index
current is planned for a future phase.
