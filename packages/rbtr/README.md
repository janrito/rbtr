# rbtr

A language-agnostic structural code index. rbtr decomposes
source files into functions, classes, methods, variables, and
imports, connects them with a dependency graph, and makes them
searchable through name matching, keyword search, and
semantic similarity — fused into a single ranked result.

## Quick start

```bash
uv tool install rbtr         # install the CLI

cd /path/to/your/repo
rbtr index                   # build the index
rbtr search "retry logic"    # search it
```

A background daemon starts automatically and keeps every
watched ref current (`HEAD` by default; add more with
`rbtr index <ref>`). Subsequent builds are incremental —
unchanged files (by blob SHA) are skipped.

## Walkthrough

Index a repository, then explore it:

```text
rbtr index
Watching: HEAD
Indexing in background; run `rbtr status` to track.

rbtr search "search fusion"
ARCHITECTURE.md:447-468  doc_section  Search fusion
  0.49
    Three channels fused into one ranked list:
    …

src/rbtr/index/search.py:96-127  function  classify_query
  0.33
    def classify_query(query: str) -> QueryKind:
    …
```

Read the top hit's source:

```text
rbtr read-symbol fuse_scores
src/rbtr/index/search.py:298-380  function  fuse_scores
 298 def fuse_scores(
 299     scored: dy.DataFrame[FusionInputRow],
 300     query: str,
 301     ...
```

See what changed between two refs:

```text
rbtr changed-symbols HEAD~3 HEAD
+ doc_section CLI integration  ARCHITECTURE.md
~ function   fuse_scores  src/rbtr/index/search.py
− function   resolveCommand  exec.ts

+1  ~1  −1
```

List symbols in a file:

```text
rbtr list-symbols src/rbtr/index/search.py
    43-43    variable    log
    44-86    function    _name_score_expr
    95-127   function    _kind_boost_expr
   298-380   function    fuse_scores
   381-494   function    search
```

## Commands

### `rbtr index`

Watch refs and keep them indexed. Each positional ref is an
independent watch target the daemon keeps current; with no
arguments it watches `HEAD`.

```bash
rbtr index                    # watch HEAD (the default)
rbtr index main               # watch main, even from another branch
rbtr index main release       # watch several refs independently
rbtr index --remove main      # stop watching main (HEAD can't be removed)
rbtr index --remove            # forget this repo (only when HEAD is all it watches)
rbtr index --remove-stale-refs # stop watching this repo's deleted branches
rbtr index --remove-stale-repos # forget every repo whose checkout is gone
```

A moving ref (branch) tracks its tip; a bare SHA settles
after one build. Removing a ref stops watching it; its index
is reclaimed by `rbtr gc --watched-only` (a plain `rbtr gc`
keeps every branch/tag regardless).

When you're done with a checkout, `rbtr index --remove` (with no
refs) forgets the whole repo — its watch set, indexed commits, and
references. After you've already deleted a worktree or clone, run
`rbtr index --remove-stale-repos` from anywhere to forget every repo
whose path no longer exists. Forgetting is metadata-only and reports
no statistics; run `rbtr gc` to reclaim the freed chunks.

### `rbtr search <query>`

Search the code index.

```bash
rbtr search "IndexStore"          # name match
rbtr search "retry timeout"       # keyword search
rbtr search "how does auth work"  # semantic search
```

Combines name, keyword, and semantic search into a
single ranked result. See
[ARCHITECTURE.md](ARCHITECTURE.md#search-fusion)
for the fusion algorithm.

Supply `--keywords` and `--variants` (both repeatable) to
widen retrieval — keywords extend the lexical query,
variants add semantic rephrasings:

```bash
rbtr search "load config" \
  --keywords settings --keywords env \
  --variants "read configuration from file"
```

Pass `--scope all` to search every indexed repo in the
shared store, not just the current one. Results from all
repos merge into one ranked list, each prefixed with its
repo name:

```bash
$ rbtr search "connection pool" --scope all
ukf/deploy/pgbouncer/settings.env:1-9  config  pgbouncer
  0.84
rbtr/packages/rbtr/src/rbtr/index/store.py:118-140  method  IndexStore.close
  0.23
```

Scope defaults to `workspace` (the current repo only).

### `rbtr read-symbol <name>`

Full source of a symbol by name.

```bash
rbtr read-symbol fuse_scores
```

When a name lives in several files, narrow it with
`--file-path` (repeatable):

```bash
rbtr read-symbol Config --file-path src/auth/config.py
```

### `rbtr list-symbols <file>`

Table of contents for a file — one line per symbol.

```bash
rbtr list-symbols src/rbtr/index/search.py
```

### `rbtr find-refs <symbol>`

Symbols that reference a given symbol via the dependency
graph (imports, docs).

```bash
rbtr find-refs IndexStore
```

Disambiguate a colliding name by restricting resolution to
certain files with `--file-path` (repeatable):

```bash
rbtr find-refs Config --file-path src/auth/config.py
```

### `rbtr changed-symbols <base> <head>`

Symbols in files that changed between two refs.

```bash
rbtr changed-symbols HEAD~5 HEAD
```

Scope the diff to specific files with `--file-path`
(repeatable):

```bash
rbtr changed-symbols HEAD~5 HEAD --file-path src/rbtr/index/store.py
```

### `rbtr status`

Index status: indexed refs, chunk counts, the on-disk index
size, and active builds.

```bash
$ rbtr status
✓  5.8k chunks  1.3 GB · ~/.local/share/rbtr/index.duckdb
   aa2ecc4bbefb (HEAD, main)  5.8k indexed  5.8k embedded ✓
```

Pass `--scope all` to list every indexed repo in the
shared store, grouped by repo:

```bash
$ rbtr status --scope all
✓  indexed repos  1.3 GB · ~/.local/share/rbtr/index.duckdb
  /home/me/projects/ukf
     a4aa7830ad87 (HEAD, main)  68.8k indexed  68.8k embedded ✓
  /home/me/projects/rbtr
     aa2ecc4bbefb (HEAD)  5.8k indexed  5.8k embedded ✓
```

### `rbtr config`

Rendered configuration with all defaults, TOML overrides,
and env vars merged.

```bash
rbtr config
```

### `rbtr daemon`

```bash
rbtr daemon start     # start the daemon
rbtr daemon stop      # stop it
rbtr daemon status    # show state and build progress
```

Starts automatically on first `rbtr index` or `rbtr search`.

### `rbtr gc`

Garbage-collect old index data. **Destructive and not undoable** —
it permanently deletes indexed commits/chunks. Always preview with
`--dry-run` first. It is only ever manual; the daemon never GCs on
its own.

```bash
rbtr gc                       # this repo (default: keep branches/tags + watch set)
rbtr gc --all-repos           # every indexed repo (default reclamation only)
rbtr gc --watched-only        # keep only HEAD and watched refs
rbtr gc --keep-head-only      # keep only HEAD
rbtr gc main release          # keep only HEAD plus these refs
rbtr gc --orphans             # sweep crashed-build residue only
rbtr gc --no-compact          # skip the disk-reclaiming rewrite
rbtr gc --dry-run             # preview what would be dropped
```

`rbtr gc` collects the current repo by default. `--all-repos` reclaims
across **every** indexed repo at once — useful because chunks are shared
between repos — but only with the safe default reclamation; scope an
aggressive mode (`--watched-only`, `--keep-head-only`, or a `keep`
list) to a single repo. (The chunk sweep is global on every gc regardless, so a
plain `rbtr gc` still frees chunks no other repo references.)

By default it keeps HEAD, every local branch and tag, and
every watched ref (plus the current worktree), dropping only
genuinely unreferenced commits — so a routine gc never
discards anything still reachable. `--watched-only` keeps
just HEAD and the watch set, dropping unwatched branches and
tags (the way to reclaim refs you no longer index).

The other modes: `--keep-head-only` keeps only HEAD; `rbtr gc <refs>`
keeps HEAD plus the listed refs; `--orphans` sweeps residue
from crashed builds.

After deleting, gc rewrites the index file to hand the freed disk
space back to the operating system — deleting alone keeps that space
inside the file, so it never shrinks on its own. The rewrite reports
the size change (`index 2.08 GB → 1.28 GB (-800 MB)`). Pass
`--no-compact` to skip it.

## Output modes

- **TTY**: rich-formatted text with syntax highlighting.
- **Piped / `--json`**: a single JSON object — the full response
  model, serialised in one pass (the same shape the daemon returns).

Example from `rbtr search --json`:

```json
{"kind":"search","results":[{"name":"fuse_scores","kind":"function","file_path":"src/rbtr/index/search.py","score":0.49,...}]}
```

See [Daemon protocol](ARCHITECTURE.md#daemon-protocol)
for the full response models.

## Logs

The daemon writes structured logs — one JSON object per line — to
`daemon.log` in the log directory, rotating at 10 MB and keeping five
backups. CLI commands log to stderr: coloured on a terminal, JSON when
piped. stdout is reserved for command output, so logs never pollute a
`--json` result.

Raise verbosity with `--log-level` or the `RBTR_LOG_LEVEL`
environment variable:

```bash
rbtr --log-level debug search "retry logic"   # DEBUG to stderr
RBTR_LOG_LEVEL=debug rbtr status              # same, via env
```

`rbtr config` shows the log directory (`log_dir`); tail the daemon log
with any JSON-aware tool:

```bash
tail -f <log_dir>/daemon.log
```

See [ARCHITECTURE.md](ARCHITECTURE.md#observability) for the logging
pipeline and how requests are correlated.

## Configuration

`{config_dir}/config.toml`. Environment variables with
`RBTR_` prefix override file values. Run `rbtr config` to
see the full rendered config with all defaults.

Notable settings:

- `embedding_model` — HuggingFace GGUF model ID.
- `search_weights` — per-query-kind fusion weights
  `(alpha, beta, gamma)` for the semantic, lexical, and
  name-match channels.
- `reranker_model` — cross-encoder GGUF model ID. Set to
  `""` to disable reranking.
- `reranker_settings` — per-query-kind reranker pool size
  and blend weight.
- `log_level` — root log level (`DEBUG`, `INFO`, …).
- `log_format` — `auto` (console on a TTY, else JSON), `console`,
  or `json`.
- `log_max_bytes` / `log_backup_count` — `daemon.log` rotation size
  and backup count.

Run `rbtr config` to see every field with its current
value.

Query expansion (keywords and variant rephrases) is
client-supplied: the caller passes `keywords` and `variants`
on `SearchRequest`. In pi sessions the LLM generates these
automatically via tool-call parameters.

Directories are resolved via [platformdirs]. Four are
independently overridable (`--data-dir`, `--config-dir`,
`--log-dir`, `--cache-dir`). `runtime_dir` is derived from
`hash(data_dir)` — never overridable.

[platformdirs]: https://platformdirs.readthedocs.io/

## Supported languages

Languages with tree-sitter grammars get structural
extraction (symbol-level chunks, import metadata, scope
detection). Everything else gets line-based chunking.

Built-in: bash, c, cpp, css, go, hcl, html, java,
javascript, json, less, markdown, python, query, rst,
ruby, rust, scss, sql, svelte, toml, tsx, typescript,
vue, yaml.

Code embedded in another file is indexed in its own
language. A fenced code block in Markdown is extracted as
chunks of that language at its real line numbers, so a
Python example in a README is searchable as Python.
HTML and single-file components (Svelte, Vue) extract
inline `<script>` / `<style>` the same way; an SFC's markup
template is indexed too, named after the component file.

See [ARCHITECTURE.md](ARCHITECTURE.md#language-plugins)
for how the plugin system works.

## Writing a language plugin

rbtr's core ships no languages of its own — each is a separate,
installable package, `rbtr-lang-<lang>`, that registers with rbtr
through a `rbtr.languages` entry point. Users add one by installing it
(`pip install rbtr-lang-swift`, or `pip install rbtr[swift]` for the
languages rbtr bundles as extras — see [Installing the
plugin](#installing-the-plugin)). A plugin exposes
`LanguageRegistration` values and provides whatever fields its
language needs — there are no tiers or categories.

### The package

Lay the package out as a standard `src` distribution:

```text
rbtr-lang-swift/
  pyproject.toml
  README.md
  src/rbtr_lang_swift/
    __init__.py
    plugin.py
    swift.scm       # tree-sitter query, shipped as package data
    py.typed
```

```toml
# pyproject.toml
[build-system]
requires = ["uv_build>=0.9,<1"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "rbtr_lang_swift"

[project]
name = "rbtr-lang-swift"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["rbtr", "tree-sitter-swift"]

# How rbtr discovers the plugin: the value points at a module-level
# `LanguageRegistration`, named by its language id.
[project.entry-points."rbtr.languages"]
swift = "rbtr_lang_swift.plugin:swift"

[dependency-groups]
dev = ["rbtr[test]"]   # the test harness (syrupy + pytest-cases)
```

Nothing else registers the plugin — installing the package is enough for
rbtr to find it.

### The plugin — query-based (preferred)

Most languages need only a tree-sitter query. The generic
`extract_symbols` pipeline handles parsing, capture
matching, scope detection, and chunk construction. The query
lives in its own `.scm` file, loaded with `load_query`:

```scm
; src/rbtr_lang_swift/swift.scm
(function_declaration
  name: (identifier) @_fn_name) @function
(class_declaration
  name: (type_identifier) @_cls_name) @class
(import_declaration
  (identifier) @_import_module) @import
(source_file
  (comment) @comment)
```

```python
# src/rbtr_lang_swift/plugin.py
from __future__ import annotations
from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    load_query,
)

swift = LanguageRegistration(
    id="swift",
    extensions=frozenset({".swift"}),
    grammar_module="tree_sitter_swift",
    extraction=QueryExtraction(
        query=load_query(__package__, "swift"),
        scope_types=frozenset({"class_declaration"}),
    ),
)
```

Capture conventions: `@function`/`@_fn_name`,
`@class`/`@_cls_name`, `@method`/`@_method_name`,
`@variable`/`@_var_name`, `@import`/`@_import_module`,
`@doc_section`/`@_section_name`, `@comment`.

Import captures: `@_import_module` populates
`ImportMeta.module` directly from the query, with delimiter
stripping for `<>` (system includes) and `"` (string literals).
Each `@import` match is passed to the language's
`import_extractor`, which reads captures first then walks
the node for what the query can't express (e.g. multi-valued
import names). Languages that don't need custom logic get the
built-in import resolver; those that do attach an override with
`@swift.import_extractor` (see Overrides below).

Cross-language imports: `ImportMeta.language_hint` directs
resolution when the target language differs from the source
(e.g. HTML `<script src>` → `language_hint="javascript"`).

Custom names and scopes: when a query cannot express a
symbol's display name or scope, attach a resolver with the
`@swift.name_extractor` / `@swift.scope_extractor` decorator methods
(see Overrides below) — last-resort callbacks that compute them from
the captured node (e.g. an HCL block named by its type and labels; a
CSS nested rule scoped under its parent selector; a TOML dotted table
split into a name and scope). Each delegates to the default resolver
for the cases it does not special-case.

Comments: capture your grammar's comment nodes as `@comment`, scoped to the
file root (`(source_file (comment) @comment)` above), plus the module
docstring where the language has one. A comment block directly above a
definition folds into its chunk as documentation; a block that stands on its
own — a banner, a licence header, a note between definitions — becomes a
searchable chunk of kind `comment`. A comment trailing code on its line stays
with that statement. The engine handles the grouping and routing identically
for every language; see [ARCHITECTURE](ARCHITECTURE.md) for the rules.

### Chunker-based plugin

When the language's structural units can't be expressed
as query captures (heading hierarchies, composed names,
content-minus-children), write a custom chunker. The
chunker receives the grammar from the manager:

```python
# src/rbtr_lang_example/plugin.py
from __future__ import annotations
from collections.abc import Iterator
from typing import TYPE_CHECKING
from tree_sitter import Parser
from rbtr.index.identity import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind
from rbtr.languages.registration import LanguageRegistration

if TYPE_CHECKING:
    from tree_sitter import Language, Range

example = LanguageRegistration(
    id="example",
    extensions=frozenset({".ex"}),
    grammar_module="tree_sitter_example",
)

@example.chunker
def chunk_example(
    file_path: str,
    blob_sha: str,
    content: str,
    grammar: Language,
    ranges: list[Range] | None = None,
) -> Iterator[Chunk]:
    parser = Parser(grammar)
    if ranges is not None:
        parser.included_ranges = ranges  # serve as an injection target
    tree = parser.parse(content.encode())
    # ... walk tree, yield Chunk objects ...
```

### Overrides — custom resolvers and chunkers

The four extraction overrides are attached as **methods** on the
registration, not constructor arguments. Each works as a decorator on a
fresh function, or as a plain call to reuse a shared/imported one.

The name/scope/import overrides are **wrap-style** (like pydantic's
`WrapValidator`): the first argument is the built-in `resolver` — call it
to delegate, then refine — so you never import the default:

```python
@swift.name_extractor          # fresh, inline
def swift_name(resolver: NameResolver, capture_name: str, node: Node, caps: dict[str, list[Node]]) -> str:
    return resolver(capture_name, node, caps).removesuffix("!")   # delegate, then tweak

swift.import_extractor(extract_swift_imports)   # reuse an existing function
```

The methods are `name_extractor`, `scope_extractor`, `import_extractor`
(all wrap-style, `resolver` first), and `chunker` (no resolver). An
override that doesn't delegate simply ignores its `resolver`. Unset ones
fall back to the engine defaults.

### Testing the plugin

Tests exercise the **real** extraction pipeline: they call
`rbtr.languages.extract.extract_file` — the same per-file entry point
the indexer uses — so there is no test-only code path. The `rbtr[test]`
extra (in your `dev` group) provides `syrupy` and `pytest-cases`. Lay
the tests out beside the code:

```text
src/rbtr_lang_swift/tests/
  __init__.py
  cases_extraction.py
  test_extraction.py
  test_samples.py
  samples/swift/swift.swift
  __snapshots__/
```

The `snapshot_json` fixture — which serialises chunks to canonical JSON for
snapshots — is provided automatically by the `rbtr[test]` pytest plugin, so a
test just takes it as an argument; no `conftest.py` is needed.

Construct tests keep the data (source → expected symbols) in `@case`
functions and run the pipeline in the test body:

```python
# cases_extraction.py
from pytest_cases import case
type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]

@case(tags=["symbol"])
def case_function() -> SymbolCase:
    return "swift", "func greet() {}\n", [("function", "greet", "")]

# test_extraction.py
from pytest_cases import parametrize_with_cases
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file

@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang, source, expected):
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    got = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in got
```

A sample test snapshots a committed example project, guarding extraction
against drift:

```python
# test_samples.py
from pathlib import Path
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager

def test_extraction_matches_snapshot(snapshot_json):
    root = Path(__file__).parent / "samples" / "swift"
    files = [
        (str(p.relative_to(root)), p.read_text())
        for p in sorted(root.rglob("*")) if p.is_file()
    ]
    manager = get_manager()
    chunks = []
    for path, text in files:
        lang = manager.detect_language(path) or "swift"
        chunks.extend(extract_file(FileEntry(path, "sha1", text.encode()), lang))
    assert chunks == snapshot_json
```

Regenerate the golden files after an intended change with
`pytest --snapshot-update`. For edge snapshots,
`rbtr.testing.render_edges(edges, chunks)` turns opaque edge
ids into readable `file::name -> file::name [kind]` lines.

### Installing the plugin

Installing the package is all it takes — the entry point auto-registers.
(Two packages claiming the same language id is a conflict and raises.)

- **Any third-party language:** `pip install rbtr-lang-swift` (it
  depends on `rbtr`).
- **A language rbtr blesses as an extra:** rbtr lists
  `swift = ["rbtr-lang-swift"]` in its own `[project.optional-dependencies]`,
  so `pip install rbtr[swift]` works.
- **A default language:** rbtr lists the package in its own
  `[project.dependencies]`, so plain `pip install rbtr` pulls it.

### Re-indexing after a plugin change

When you change a plugin's extraction logic — its query, chunker, or
anything that shapes the chunks it emits — bump
`extraction_serial` on the registration. Indexed chunks are keyed
by this serial, so a bump triggers re-extraction of every blob stored
at a different serial on the next build; leaving it unchanged keeps the
existing (now stale) chunks. It is independent of the package version —
bump it whenever extraction output changes, including during development
before a release. See
[ARCHITECTURE.md](ARCHITECTURE.md#content-addressed-chunks-and-blob-dedup)
for the dedup mechanism.

## Graceful degradation

- **No grammar** → line-based plaintext chunking.
- **No embedding model** → structural index works, semantic
  search skipped.
- **No client-supplied expansion** → search runs on the
  original query only (no keyword or variant widening).
- **No reranker model** → search returns fusion-ranked
  results without cross-encoder reranking.
- **No FTS index** (first search before any build completes)
  → error with guidance to run `rbtr index`.

## Development

```bash
git clone <repo-url>
cd rbtr
uv sync --extra languages
just check    # lint + typecheck + test
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for internals.

## License

MIT
