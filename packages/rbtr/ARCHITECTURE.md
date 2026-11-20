# Architecture

Technical reference for contributors. For usage, see
[README.md](README.md).

## Overview

rbtr decomposes source files into named chunks (functions,
classes, methods, imports), connects them with a dependency
graph, and makes them searchable through three fused
retrieval channels.

![rbtr in context: you and your agents ask one daemon, which reads your
checkouts, runs local models and owns the index](docs/diagrams/context.png)

The diagrams in this file are draw.io sources in `docs/diagrams/`;
after editing one, `just diagrams` renders the PNGs beside them.

### The path a file takes

A build walks a git tree. Each file is matched to a language
and, unless its bytes have been chunked already, handed to
that language's plugin, which returns chunks — named spans of
source, each with a kind, a scope, and a line range. The
chunks go into the project's DuckDB file; an edge pass infers imports and
doc links between them; a later, lower-priority pass embeds
them.

A query runs three channels over that table — name matching,
BM25, and vector similarity — fuses them into one score,
reranks the top candidates with a cross-encoder, and returns
the winners.

The daemon runs all of it in the background. It watches refs,
rebuilds when they move, and answers queries over ZMQ, so the
CLI and the pi extension read one index rather than each
keeping their own.

### Where things live

Modules only import from layers below them — no upward
dependencies. Top to bottom: `cli` → `daemon` → `index`
→ `languages` → `domain` (the pure kernel: data models,
identity, tokenisation). `config` sits just above the
kernel; `git` and `errors` are foundations below. The
layering is a DAG enforced by import-linter (run in
`just lint`), so boundary drift fails the build instead
of relying on convention. The runtime binds RPC and indexing together;
[import contracts](../../pyproject.toml) require their module independence
and keep RPC below its runtime owner.

![The layers: cli, daemon, index, languages and domain, each importing
only from those below it](docs/diagrams/components.png)

Inside `index` the imports point one way too, and import-linter
checks that as well. `Indexer` (`indexer.py`) uses that API for
[background indexing](#daemon); `watcher.py` derives work from Git and stored
intent. Neither imports daemon code. `Index` (`index.py`) is the index's API: the
reads, and the sessions that write. It owns a `Store` (`store.py`),
which holds the connection, the registry and every change to it, the
attached files, the routing and the per-thread cursors, and which
nothing else imports; `Index` hands registry changes to it.
`WriteSession` writes a repo's rows. `refs.py` decides which
snapshot a read means, and `search` takes an index and ranks what
it returns. An index created with `writable=False` (the default)
rejects `session`, `register`, `forget` and `compact`.

![Indexer and finite operations use Index; Index owns Store and creates
finite write sessions](docs/diagrams/index-code.png)

### Chunk and symbol

The two words name the same thing from either side of the
API boundary. Internally it is a **chunk**: content-addressed,
keyed by a hash of its own bytes, and shared between every
repo that holds those bytes. On the wire it is a **symbol**,
because that is what a caller navigates by — in one of several
shapes, according to how much of it the command returns.
`rbtr.daemon.dto` is where one becomes the other, dropping the
persistence detail on the way out — see
[Storage models vs API DTOs](#storage-models-vs-api-dtos).

### Design priorities

1. **Language-agnostic.** Every file gets indexed. Languages
   with tree-sitter grammars get structural extraction;
   everything else gets line-based chunking.
2. **Local and offline.** Embeddings run on-device (GGUF on
   Metal/CPU). The index is a set of DuckDB files. No API calls.
3. **Incremental.** Builds are keyed by git blob SHA, and
   chunks are content-addressed: only changed files are
   re-extracted, embeddings persist, and byte-identical content
   is shared across worktrees, clones, and repos rather than
   re-indexed.

## Indexing

### Index pipeline

The indexing pipeline flows in one direction: files come
in, chunks come out, and the store makes them searchable.

![How a build indexes a snapshot: three transactions, for extraction,
edges and finishing, and the tables each writes](docs/diagrams/build.png)

![How chunks get vectors: a job after the build reads the chunks still
without one, embeds them in batches and stores each batch](docs/diagrams/embed.png)

`build_index` drives the pipeline. It detects each
file's language, loads the appropriate plugin, and
routes extraction through one of three strategies (see
[Language decomposition](#language-decomposition)).
Chunks and edges are written to the repo's project file via the index
(see [Index files](#index-files)); embedding is a separate step
(`embed_index`), and search reads from the same index.

### Language plugin routing

Plugins register capabilities via `LanguageRegistration`.
`build_index` routes each file to the most specific
extraction path available:

![Which extraction a file gets: a tree-sitter query, a plugin's chunker,
or line-based raw chunks](docs/diagrams/plugin-routing.png)

Plugins provide whatever combination of fields their
language needs. A query-based plugin (most languages)
gets structural extraction automatically. A chunker-based
plugin (markdown, HTML, HCL) handles cases where
query captures can't express the structure. Languages
without a plugin get line-based chunking.

### Content-addressed chunks and blob dedup

![How files become chunks once: two snapshots name the same blob for an
unchanged file, so they share its chunks](docs/diagrams/content-addressing.png)

Because chunks are content-addressed and shared by a project's
checkouts (see [Data model](#data-model) for the keying), a blob
already chunked by any of them need not be parsed again: a main
checkout and its linked worktrees share `.git` object SHAs, so
their blobs and chunk `id`s coincide, and the existing chunks
are reused — no re-parsing, no re-embedding. A clone is a
project of its own, so it parses and embeds its content again.

The build checks `session.blob_is_current(blob_sha, language,
serials)` before extracting a file, where `language` is the
file's detected host language and `serials` is the current
`{language: extraction serial}` registry map (plus `""` for
plaintext). The check spans the repo's project, so a blob
already chunked by any of its checkouts is reused. A blob is
**up to date** — the file skipped and only its `file_snapshots`
row written — iff it has at least
one chunk in the detected `language` *and* every stored chunk
is at its language's current extraction serial. Both are matched
in one query against a registered `_serial_map` view (a
`LEFT JOIN`; a chunk with no matching serial row is stale).

Two things force re-extraction; on a miss the build
deletes the blob's old chunks first (the new extraction may
produce different IDs the upsert can't reconcile):

- **Detected language changed.** A blob indexed as plaintext
  (`language=""`), then a plugin registered for its extension,
  has no chunk in the new language → miss → re-extract. Prose
  (Markdown/RST) is detected by extension or content sniff and
  persisted on the snapshot, so the dedup check sees its real
  language and dedups normally.
- **A plugin's extraction serial bumped.** Chunks stored at the
  old serial no longer match the map → re-extract.

**Every file leaves a host-language chunk.** Extraction
guarantees at least one chunk in the file's own language — the
real definitions for most files, and a content-less
*host-presence chunk* for a file that would otherwise produce
none: an empty file (an empty `__init__.py`), unextractable
content, or a multi-language file whose host contributes no
content (a script-only SFC). This is what makes the
detected-language check and multi-language dedup reliable.

Multi-language files (single-file components: `.svelte`,
`.vue`) hold chunks in several languages — the `<script>`
delegates to TypeScript, the `<style>` to CSS/SCSS, and the
markup template is the host `svelte`/`vue` chunk. Each chunk
carries *its own* language's extraction serial (see
[Versioning](#versioning)). The map lists the host and every
embedded language, so such a file dedups like any other: a
bump to *any* contributor (the host or a delegated language)
re-extracts it, and an unchanged file is skipped.

Language is resolved once per file before the dedup check, in order:
file extension → the `detected_language` recorded on an earlier
`file_snapshots` row for that path (`get_snapshot_language`) →
a prose content-sniff for never-seen files. Persisting the
detected language on the snapshot is what lets prose
(Markdown/RST) identified by content heuristics skip the sniff —
and the needless re-extraction — on every subsequent build.

### Write path

A repo's rows are written through a `WriteSession`, obtained via
`Index.session(repo)`. The session is a context manager:

- `__enter__`: begins a transaction on the cursor the store
  gave it.
- `__exit__`: flushes buffered chunks, commits the
  transaction, and rebuilds the FTS index if chunks
  were modified. Rolls back on exception.

A build's last session runs `ws.cleanup()`, which deletes its
repo's residue from builds that stopped part-way and the chunks no
snapshot claims any more.

![A write session: writes and the reads inside a write go through
one session, which holds one cursor and one repo](docs/diagrams/write-session.png)

A build registers the repo first, in a registry transaction (see
[Repo registration](#repo-registration)), so a crash part-way
through leaves an empty `repos` row rather than indexed rows that
nothing can find. It then opens one session for each of its three
transactions: extraction, edges, and finishing. Embedding is a
separate job.

Embedding is deferred and lower-priority. After
indexing completes, the daemon submits an `EmbedJob`
to the work queue's low-priority deque. The index is
fully queryable (FTS + name search) before embeddings
exist; semantic search degrades gracefully when
embeddings are absent.

A daemon started with `embed` off never embeds: it creates
no embedder, picks no embed job, and its searches leave out
the semantic signal, so no model is loaded. Embeddings
already stored stay, unread; a daemon started later with
`embed` on embeds whatever is missing. The setting is the
daemon's, fixed when it starts, because one daemon serves
every repository and client. A daemon reads it from the
environment and the config file, as every rbtr process does.

### Extraction call flow

The tree-sitter path — the most common case. The build loop calls
`extract_file` (`rbtr.languages.extract`), which dispatches via
`extract_primary` → `extract_query` to `extract_symbols`
(`rbtr.languages.treesitter`); that call chain:

![The tree-sitter extraction call flow, from the build loop through
LanguageManager and the write session to extract_symbols](docs/diagrams/extraction.png)

## Language decomposition

The extraction engine lives in `rbtr.languages`, beside the
`LanguageRegistration` contract and the manager that runs it: `treesitter`
(query execution and doc-span recovery), `extract` (per-file strategy dispatch
and injection), `edges` (import/doc inference), and `chunks` (the
plaintext fallback, and the span arithmetic every chunker shares).
`rbtr.index` depends on `languages` in just two places — the build loop,
which composes extraction with storage, and `classify`, which reuses the
language registry for search — while `languages` reaches down only to the
`rbtr.domain` kernel (`models`, `identity`), so the dependency stays
one-directional.

Each file is routed to one of three extraction strategies
by the language plugin:

- **Tree-sitter** - grammar + query → functions, classes,
  methods, variables, imports as structured `Chunk` objects.
- **Custom chunker** - plugin-provided (e.g. markdown
  heading-hierarchy splitting).
- **Plaintext fallback** - fixed-size overlapping line
  chunks.

The first two are a discriminated union on the registration's
`extraction` field (`QueryExtraction` | `ChunkExtraction` | `None`),
so a file has exactly one primary strategy — "query and chunker both
set" is unrepresentable — and `extract_primary` dispatches by type.
Injection (`injection_query`) is orthogonal and runs in addition.

Scope addressing: each chunk's `scope` is the full path of its
enclosing named scopes, outermost-first, joined by `::` (e.g.
`Outer::Inner`, `make_adder::handler`), composed by walking the
parse tree's scope-bearing nodes (per the plugin's
`QueryExtraction.scope_types`); Markdown/RST compose their heading
hierarchy the same way. A function becomes a `METHOD` only when its
nearest enclosing scope is class-like
(`QueryExtraction.class_scope_types`), so a closure inside a function
or method stays a function.

Variable chunks cover module-level bindings only; class
attributes and function-locals stay within their enclosing
chunk.

Comments: every tree-sitter plugin captures its grammar's comment
nodes as `@comment`, scoped to the file root, plus the module
docstring where the language has one (Python's bare top-level
string). `extract_symbols` groups consecutive captured comments into
blank-line-delimited blocks and routes each one. A block sitting flush
above a definition folds into that definition's chunk, so a symbol
carries its own documentation; a block inside a symbol's body is
already part of that chunk; every other block — a licence or banner
header, a note between definitions, a comment at end of file — becomes
a standalone chunk of kind `COMMENT` at module level (empty scope). A
comment trailing code on its line documents that statement, so it never
folds into what follows. Comment text is therefore indexed and
searchable wherever it sits, not only when it happens to precede a
symbol.

A file can also *embed* another language — a Markdown fenced
block, or an SFC/HTML `<script>`/`<style>`. A host plugin's
`injection_query` marks each embedded block; the engine
delegates the block's byte range to the target's full
extraction (chunker or query) and recurses into that
target's own injection, so nested embeds (an HTML block
inside Markdown) resolve too. Delegated chunks carry the
embedded language. See [Language plugins](#language-plugins)
for the dispatch chain.

## Language plugins

Each plugin is a package (a `plugin.py` plus its tree-sitter
queries as sibling `.scm` files, loaded via `load_query`) that
exposes module-level `LanguageRegistration` values. A plugin
provides whatever combination of fields its language needs -
there are no tiers or categories. Discovery is by entry point
(see below); there is no hook framework — the domain is
single-dispatch by language id, so a plain registry suffices.

For the plugin author's API (field reference, examples,
capture conventions), see
[Writing a language plugin](README.md#writing-a-language-plugin).

### Dispatch chain

`extract_file` (`rbtr.languages.extract`) builds each file's chunks in this
order:

1. **Primary extraction** - a registered `chunker` if the
   language has one (`chunker(path, sha, content, grammar,
   ranges)`; markdown, rst, svelte, vue), else tree-sitter
   `extract_symbols` for a `grammar` + `query` (python,
   rust, go, ...; json, css, html, toml, yaml, hcl; and
   `tree_sitter_query` itself, indexing `.scm` files).
2. **Injection (additive)** - if the registration has an
   `injection_query`, the engine *also* runs
   `extract_injections`: it matches embedded blocks
   (`@injection.content`), determines each block's target
   language, resolves overlaps by highest
   `injection.priority`, and delegates each range to that
   language's `extract_query`. The target is either a
   static `#set! injection.language` on the pattern
   (HTML and Svelte/Vue `<script>` / `<style>`) or a
   dynamic
   `@injection.language` capture whose text is resolved to a
   language id (a Markdown fence's info string — `python`,
   `sh`). Injection runs *alongside* the host chunker
   (step 1), not instead of it.
3. **Plaintext fallback** - only if neither of the above
   applied. If no language was detected (unknown
   extension), `detect_prose_format` sniffs the first 2 KiB:
   RST by underline headings (`Title\n=====`) or directives
   (`.. note::`), Markdown by ATX headings (`# Title`); a
   detected format uses its registered chunker. Otherwise,
   line-based raw chunks (`RAW_CHUNK`, fixed-size
   overlapping windows).
4. **Host-language trace** - if no chunk so far carried the
   file's detected language, a content-less host-presence
   chunk is appended (see
   [Content-addressed chunks and blob dedup](#content-addressed-chunks-and-blob-dedup))
   so the blob-dedup check can skip the file on later builds.

**We author our own injection queries** rather than reuse a
grammar's shipped `injections.scm`, which over-matches — a
bare `(raw_text)` fallback catches every block, including
styles. `injection.priority` disambiguates: a `lang`-tagged
block outranks the bare fallback. The language mapping (step
2) stays out of Python — static `#set!` values, and
free-form hints resolved through the registry's own
id/extension maps — so delegation is language-agnostic, with
no per-language code.

Delegation runs the target's *full* primary extraction
(`extract_primary` — its chunker or query, range-scoped via
`included_ranges`), so the block extracts as if it were its
own file of that language: query targets (python, ts, html,
...) and chunker targets (yaml, toml, ...) alike, imports
included. Delegation also recurses into the target's own
injection, so an HTML block inside Markdown that contains an
inline `<script>` yields its js too (bounded by a depth
cap). Only per-file orchestration — the plaintext fallback
and host-presence trace — is not re-run per block.

### External plugins

Every language registers through the `rbtr.languages`
entry-point group, and every language is a separate
`rbtr-lang-*` distribution — core declares none of its own. Eight
are required dependencies of `rbtr`, so they arrive with it; the
rest are extras. See the README for a step-by-step guide to
writing a plugin.

The entry-point *value* resolves to a module-level
`LanguageRegistration`, named by its language id
(`my_pkg.plugin:swift`, where `swift = LanguageRegistration(...)`).
The manager (`rbtr.languages.LanguageManager`) collects them via
`importlib.metadata.entry_points`; a duplicate language id raises,
and a plugin whose module fails to import is logged and skipped.

### Plugin → core contract

A plugin (built-in or external `rbtr-lang-*` package) depends
on a small, stable surface of core symbols. This is the
contract that survives packaging each language as its own
distribution:

- `rbtr.languages.registration` — `LanguageRegistration`,
  `ModuleStyle`, the wrap-override resolver type aliases
  (`NameResolver` / `ScopeResolver` / `ImportResolver`), and the
  capture/import helpers `collect_scoped_path`,
  `enclosing_nodes_of_type`, `build_quoted_import`,
  `parse_path_relative`, and `load_query` (loads a plugin's co-located
  `.scm` query as package data). (Overrides receive the built-in resolver
  as their wrap `resolver` argument and delegate to it, so plugins never
  import the built-ins; those built-ins live in the private
  `rbtr.languages._resolvers`.)
- `rbtr.domain.models` — `Chunk`, `ChunkKind`, `ImportMeta`. A chunk's
  `id` is a property over its own fields, so a plugin yields chunks and
  never computes an identity.
- `rbtr.languages.chunks` — `chunk_plaintext`, and `last_line`,
  which turns a tree-sitter end point into the 1-based last line a
  node occupies. The query engine and every chunker share it, so a
  span means the same thing in every language.

One dependency crosses *between* plugins rather than to core:
scss and less import `css_nesting_scope` from the css plugin
(CSS-family nesting behaviour). That is a legitimate
plugin-to-plugin dependency, not part of the core contract.

### Sample fixtures

Each plugin owns its samples, beside its code:
`src/rbtr_lang_<id>/tests/samples/` holds a small example project
exercising the constructs that plugin extracts. They are both
worked examples of extraction and golden-tested fixtures.

A plugin's `test_samples.py` runs the real per-file entry point
(`rbtr.languages.extract.extract_file`) over the sample tree and
compares the result against a committed snapshot under
`__snapshots__/`, so any drift in extraction output turns the
test red. The `snapshot_json` fixture that serialises chunks to
canonical JSON ships with core's pytest plugin
(`rbtr.testing`), so a plugin needs no `conftest.py`.

Regenerate every plugin's snapshots after an intended extraction
change with `just snapshots`, and review the diff before
committing.

The samples are exempt from the repo's linters and
type-checker — they are fixtures, validated by their own
tests rather than held to product-code standards.

## Dependency graph

`edges.py` infers cross-file relationships:

- **Import edges** - structural (tree-sitter import
  extractor) or text-search fallback.
- **Doc edges** - markdown sections mentioning symbol names.
- **Data references** - a JSON Schema or OpenAPI `$ref`, a
  tsconfig `extends`, a local Actions `uses`, a Cargo or Poetry
  `path` dependency. A `$ref`'s pointer names a key inside the
  document it reaches, as a Markdown link's fragment names a
  heading, so a data language's keys are in the symbol index.

Import resolution maps a module string to a repo file. A
reference carrying its own depth — `../b`, or a `dots` the
extractor set — is anchored to the importing file and has one
place to look. Anything else is tried from the repository root
first, then, for a `ModuleStyle.PATH` language, beside the
importing file: `@import "b.css"` in `css/a.css` means
`css/b.css`, and the root is tried first so a reference that
already resolves keeps the file it resolved to. Failing both, a
dotted path matches as a *suffix* of a candidate file path (so
`rbtr.index.index` → `…/rbtr/index/index.py`) — which lets a
single index span a monorepo. `module_style` and
`source_roots` tune the mapping per language; ambiguous
matches are dropped rather than guessed, and a file never
resolves to itself.

Some references name a directory rather than a file. Where the
language names a directory's stand-in — `__init__.py`,
`index.js`, `mod.rs`, a crate's `Cargo.toml` — `index_files`
resolves it to that one file. Where every file in the directory
belongs to the unit instead, the language says `package_directory`
and the reference reaches all of them: a Go package is its
directory, a Terraform module is a directory of `.tf` files. A
Markdown link to a directory reaches nothing, which is why this
is declared rather than assumed.

A prefix a language's imports carry may be written down in the
repository rather than in the language: `go.mod` states `module
example.com/m`, which makes `example.com/m/b` the directory `b`.
A registration names such a file as its `manifest`, the
extraction loop reads it while the content is in hand, and
`build_resolution_map` turns it into path substitutions — the
edge pass itself reads no files.

Whether each language's own syntax reaches a file is pinned per
language by `test_import_resolution.py`, which extracts two real
files in nested directories and follows the edge. Nested because
two files at the repository root cannot tell a sibling reference
from a root-relative one.

Powers `find-refs` and the importance signal in search
ranking. `find-refs` first resolves the user's symbol *name*
to chunk IDs via `name_matches` (edges are keyed on chunk
IDs, not names), then a single `inbound_refs` query joins the
inbound edges to their *source* chunk so the response names
the referring symbol (`RefOut`) rather than exposing an opaque
hash.

## Working-tree indexing

Working-tree indexing extends the commit pipeline to staged,
unstaged, and untracked changes, so searches reflect the user's
current edits. The unit of work is no longer only a commit: the
working tree is indexed as its own git tree SHA.

### Identity: tree SHAs

The working tree's identity is a git tree SHA computed by
`worktree_tree_sha(repo_path, max_untracked_files=...)` in
`rbtr.git`:

```python
repo.index.read()  # load on-disk index
for path, flags in changes.items():  # each path git status lists
    if flags & FileStatus.WT_DELETED:
        repo.index.remove(path)
    elif flags != FileStatus.INDEX_DELETED:
        repo.index.add(path)  # stage it in memory
tree_sha = str(repo.index.write_tree())  # write tree object
repo.index.read()  # reset (no side effects on disk)
```

This produces a real git tree object that includes all
committed files plus working-tree modifications (staged,
unstaged, untracked, deleted). When git lists more than
`max_untracked_files` untracked files, only the tracked changes
are staged, and `rbtr status` reports the count. A repository
nested in the working tree, which git lists as one `dir/`
entry, is left out. Properties:

- **Deterministic.** Same file contents → same tree SHA.
- **Content-sensitive.** Any edit changes the SHA.
- **Clean tree = HEAD tree.** `worktree_tree_sha` returns
  `None` when the tree SHA equals HEAD's tree SHA.
- **Blobs are persisted.** Staging writes blob objects
  to `.git/objects`, so `list_files` can walk the tree
  directly — the same code path as commit trees.

Tree SHAs never collide with commit SHAs (different git
object types), so they coexist safely in `indexed_snapshots`.

Describing an indexed SHA back to the user — `status` naming a
worktree snapshot rather than printing a hash — tests the SHAs
already in hand with `is_tree_sha`, which reads. It does not
recompute `worktree_tree_sha`, which would write a tree and a
blob per changed file into the repository being described.

### What a worktree build writes

The working tree is indexed under its tree SHA in the
same schema as commits — no sentinel, no special columns.

When a worktree build runs, the store receives:

- **`file_snapshots`** with `snapshot_sha = <tree_sha>`.
  One row per file in the working tree. Each row's
  `blob_sha` is the git blob hash written by staging
  (dirty files) or the committed blob SHA (clean files).
- **`chunks`** are content-addressed and carry no `repo_id`
  (see [Data model](#data-model)). Because `id` includes the
  `blob_sha`, a dirty file produces different chunk IDs from the
  committed version of the same file, even when symbol name and
  position are identical.
- **`edges`** with `snapshot_sha = <tree_sha>`.
- **`indexed_snapshots`** with `snapshot_sha = <tree_sha>`.

Queries scope results through `file_snapshots` exactly as for
commits ([Snapshot resolution](#snapshot-resolution)) — the join is
on `blob_sha + file_language`, with the repo scope *and the path*
coming from the snapshot, never the chunk:

```sql
FROM chunks c
JOIN file_snapshots fs
  ON c.blob_sha = fs.blob_sha
  AND c.file_language = fs.detected_language
WHERE fs.repo_id = $repo_id AND fs.snapshot_sha = $snapshot_sha
```

When `$snapshot_sha` is a tree SHA, the join resolves to
the working-tree snapshots. When it's a commit SHA, the
join resolves to committed blobs. The same chunk row can
be visible through both paths if the file hasn't changed.

### What is shared, what is duplicated

**Shared (not duplicated):**

- **Chunks for clean files.** A file that hasn't changed
  has the same `blob_sha` in both HEAD and worktree
  snapshots. The `file_snapshots` rows differ (different
  `snapshot_sha`) but point at the same chunks. No chunk
  rows are duplicated.
- **Embeddings.** Since clean files share chunk rows,
  their embeddings are shared too. A dirty file's new
  chunks start with `embedding IS NULL` and are embedded
  in a subsequent pass.

**Duplicated (per tree SHA):**

- **`file_snapshots`** — one full set per tree SHA.
  `replace_snapshots` deletes old snapshots for that SHA
  and inserts the current set.
- **`edges`** — one full set per tree SHA.
- **`indexed_snapshots`** — one row per tree SHA.

**New content-addressed rows (per dirty blob, not per repo):**

- **Chunks for dirty files.** A modified file has a new
  `blob_sha`, so its chunks get new content-addressed IDs
  distinct from the committed version's. Both sets coexist —
  HEAD snapshots point at the committed chunks, worktree
  snapshots at the dirty ones. These rows are still shared:
  any other checkout of the project holding the same dirty
  blob reaches the very same chunks (nothing is duplicated per
  checkout).

### Build lifecycle

`build_index(repo_path, tree_sha, store, ...)` runs the
same pipeline as a commit build, preceded by registering
the repo — and since `repo_path` canonicalises to the
worktree's own directory, a linked worktree registers as a
repo of its own:

1. **Extract** — `list_files(repo_path, tree_sha)` walks
   the tree object directly (same as commit trees). Each
   file gets a `FileEntry` with `blob_sha` from the tree.
   The `blob_is_current` check skips files whose blob SHA already
   has chunks in the store.
2. **Edges** — inferred from the worktree's chunk set.
3. **Finalise** — `mark_indexed(repo_id, tree_sha)`,
   cleanup orphans.
4. **Embed** — deferred, same as commits.

Because `blob_is_current` works on content identity (blob SHA),
a rebuild where nothing changed skips all files.

### Staleness detection

`has_indexed`, asked about the worktree's tree SHA, is the
staleness check — the same mechanism as commits. When the user edits a file,
`worktree_tree_sha` returns a different SHA,
`has_indexed` misses, and a rebuild is triggered. When
the same content is polled again, `has_indexed` hits and
the rebuild is skipped.

### Lifecycle scenarios

The daemon's watcher drives the worktree lifecycle. On
each poll cycle:

- `poll_watched()` resolves each watched ref (HEAD is the
  default) and checks `indexed_snapshots`.
- `pending_builds()` calls both, in that order, and is what
  the watcher and the embed worker use.
- `poll_worktree()` computes `worktree_tree_sha` and
  checks `has_indexed`. It writes no index rows, but computing
  the SHA writes git objects into the repository being polled:
  a tree, plus a blob for each changed file whose content git
  does not already hold. No ref reaches them, so `git gc` may
  prune them at any time — which is why neither cleanup nor
  labelling may ask git to look one up again.

#### Dirty repo (first edit)

```text
State: HEAD=abc123 indexed. User edits foo.py.

poll_worktree → worktree_tree_sha = tree_A (≠ HEAD tree)
             → has_indexed(tree_A) = False
             → returns DirtyWorktree(tree_sha=tree_A)

Indexer.next_job → BuildJob(refs=(tree_A,))
build_index(tree_A):
  list_files(tree_A) → walks tree object
  foo.py: dirty blob_sha → blob_is_current miss → extract
  other files: same blob_sha as HEAD → skip
  replace_snapshots(tree_A), replace_edges(tree_A)
  mark_indexed(tree_A)

Next poll: worktree_tree_sha = tree_A (same edit)
         → has_indexed(tree_A) = True → skip
```

Every later edit repeats this against a new tree SHA, and each
build ends by calling `_drop_stale_worktree_shas`, which drops
any indexed tree SHA that is not the one just built — so at
most one worktree snapshot survives per repo.

The two scenarios below are the ones that do not follow from
that; the [rules](#rules) cover the rest.

#### New commit while still dirty

```text
State: HEAD=abc123, tree_A indexed. User commits → HEAD=def456.

poll_watched → WatchedTarget(ref=HEAD, sha=def456)
Indexer.next_job → BuildJob(refs=(def456,))  # watched refs first

build_index(def456) completes.
_drop_stale_worktree_shas → drops tree_A.

Next poll: worktree_tree_sha = tree_C (against new HEAD)
         → has_indexed(tree_C) = False → rebuild
```

#### Clean tree, worktree lingering

```text
State: tree_A indexed. User runs `git checkout .`.

poll_worktree → worktree_tree_sha = None → skip.
tree_A row lingers in indexed_snapshots but is inert:
  resolve_read_ref(None) computes worktree_tree_sha (None)
  → falls back to HEAD.

tree_A is cleaned up by:
  - The next build (commit or worktree) via _drop_stale_worktree_shas
  - GC (tree SHAs are not reachable from any ref)
```

### Rules

1. **`has_indexed(tree_sha)` is the staleness check.**
   Same mechanism as commits. No fingerprint, no
   in-memory state.
2. **The watcher is read-only.** All writes go through
   `WriteSession`, under `_write_sem`.
3. **Stale worktree SHAs are cleaned eagerly.** After each
   build, `_drop_stale_worktree_shas` scans
   `indexed_snapshots` and drops any indexed SHA that is not a
   commit and is not the one just built. The cleanup asks whether
   a SHA *is not a commit*, rather than whether it is a tree,
   so a row whose object git has already pruned is still
   dropped. Labelling asks the opposite question, because the
   two tolerate different mistakes: dropping a SHA that names
   nothing is harmless, calling it a worktree is a lie.
4. **GC protects the current tree SHA.** `_resolve_drop_set`
   calls `worktree_tree_sha` and excludes it. Stale tree
   SHAs are included in the drop set (not reachable from
   any ref).
5. **Watched-ref builds have priority over worktree
   builds.** `pending_builds()` checks `poll_watched()`
   before `poll_worktree()`.
6. **A build makes an embed job stand aside.** The embed
   worker calls `pending_builds()` between batches and
   returns if one is due; the remainder is re-derived (see
   [Standing aside](#standing-aside)).
7. **A linked worktree is a separate repo.** Its path
   canonicalises to its own directory rather than the main
   checkout's, so it gets its own `repos` row, watch set,
   `file_snapshots`, `edges`, `indexed_snapshots` and HEAD,
   and is built and collected independently. Any path inside
   it — a subdirectory, or the trailing-slash form `pygit2`
   reports — leads back to that same directory, so one
   checkout is always one repo. The two share only `chunks`,
   by content — see [What is shared, what is
   duplicated](#what-is-shared-what-is-duplicated).

## Data model

The index is a registry and one DuckDB file per project (see
[Index files](#index-files)). Its central design point is that
**`chunks` is a content-addressed store**: a chunk row is keyed by `id`
alone (a content hash) and carries **no `repo_id`**. The only table where
a repo meets content is **`file_snapshots`**, which maps
`(repo_id, snapshot_sha, file_path) -> blob_sha`. So:

- A chunk is reached by joining `file_snapshots` on `(blob_sha,
  file_language)`; the repo scope, a result row's `repo_id`, and its
  path all come from that join, never from the chunk. A chunk carries no
  path: identical content at four paths is one row reached four ways,
  and a search result lists every location it was found at.
- Byte-identical content in several repos/worktrees/clones is **one**
  physical `chunks` row, shared. `file_snapshots` rows (one per repo)
  point at it.
- `edges` and `indexed_snapshots` stay per-repo (they carry `repo_id`);
  only `chunks` is shared.

### Index files

The index is a registry and one data file per project, in the data
directory:

```text
<data_dir>/
  index.duckdb          the registry: projects, checkouts, schema version
  index.duckdb.tmp/     where DuckDB spills past its memory limit
  index/<uuid>.duckdb   one project's data, named by the registry
```

A **project** is a git common directory. A main checkout and its linked
worktrees share one, so they share one data file, and a new worktree
finds its content already indexed. A path git does not know is a
project of its own. The registry (`registry.sql`) has two tables:
`projects` names each project's file and the size the file had when it
was last written whole (`compacted_bytes`), and `repos` maps each
checkout to its project. A project file holds the tables of
`schema.sql` and a `meta` of its own, and no `repos` table. File names
carry no data: every file gets a fresh uuid, and a file is live only if
the registry names it.

![Inside one project's file: chunks shared by the project, and each
checkout's rows and snapshots](docs/diagrams/project-file.png)

`Store` connects once, to an in-memory DuckDB instance with the
configured memory limit and threads, and attaches the registry and
every live project file to it. Holding the registry is owning the
index. DuckDB's lock on the file keeps other processes out, and its
check against attaching one file twice keeps out a second index in the
same process; both raise `IndexLockedError`. A writable open deletes
whatever under `index/` the registry does not name: a crash during a
build or a retirement leaves such files.

`Store` also keeps a copy of the registry, which routes each read
and write to its project's file and answers repo lookups. Only the
process holding the registry writes it, and only `Store` changes it:
each change ends by reloading the copy, which also retires any file
the registry stopped naming. The copy is reloaded rather than edited
in step with each transaction, so the two cannot disagree.

One instance holds every file;
[Design decisions](#design-decisions) explains why.

![One DuckDB, many files: the daemon's one instance attaches the registry
and every project's file](docs/diagrams/index-instance.png)

Every read and write is routed by repo. The store looks up the repo's
project and runs on its own thread's cursor that has `USE`d that
project's file. A DuckDB transaction writes one file, so
`index.session(repo)` is bound to one project's file and writes for
that repo alone; registering, forgetting and compacting are the
store's own registry transactions. A
read over several projects runs on each file at once, one pool thread
per file, and merges the results (see
[Cross-repo search](#cross-repo-search)).

**Switching a project's file.** No open file is ever renamed. To
replace a project's file, the store attaches a fresh one, copies the
old file into it with `COPY FROM DATABASE`, checkpoints it, and commits
one registry `UPDATE` naming the new file; that commit is the switch.
Reads already running finish on the old file, which stays attached for
`retire_delay_s` (60 s) and is then detached and deleted. A crash
before the commit leaves the new file unnamed, and a crash after it
leaves the old one unnamed; the next open deletes either.

![How compaction replaces a project's file: copy, switch the registry,
and delete the old file after the retire delay](docs/diagrams/file-switch.png)

### Repo registration

Every per-repo table — `file_snapshots`, `edges`, `indexed_snapshots`,
`watched_refs` — names its repo with a small integer `repo_id` rather than a
path. The registry's `repos` table holds that mapping: one row per
checkout, pairing an id with the checkout's absolute path and its project.
Ids come from one sequence, so they are unique across project files.
**Registering** a repo means adding that row, and the project's row and
file when its project is new.

In code a repo is a `Repo` value, its id and path together, and the
registry hands them out (`register`, `repo`, `find_repo` and
`repos` on the index). Every method about one repo takes
one, and `repo.at(sha)` names one of its snapshots, so an id and a path
cannot be paired wrongly.

Nothing registers a repo on its own. Indexing a path is what makes it known:
`build_index` looks the path up and inserts a row if it has not seen it
before, and adding a watched ref does the same. A repo is in the index
because it was indexed or watched, never because it was declared separately.

That row has to exist, because everything that goes looking for work starts
from the list of repos — garbage collection (`run_gc_all`), the watcher's
reconcile loop, `rbtr forget --stale` and cross-repo search all walk
`repos()`. Rows carrying a `repo_id` with no matching `repos` row are
reachable by none of them: never collected, never rebuilt, and impossible to
forget. They just sit in a project's file.

So `build_index` derives the repo from the path rather than taking one as
an argument, and the two cannot disagree. `run_gc` and search look the path up
the same way but fail when it has no row, since a repo that was never
indexed has nothing to collect or search.

The path is what identifies a checkout, so it is canonicalised before use
(`normalise_repo_path`). The same checkout can be named several ways —
`pygit2` reports a working directory with a trailing slash, a caller may
pass a subdirectory, and a path may reach the repo through a symlink.
Matching on the string as given would register each spelling separately and
split one repo's index between them.

The link between `repos` and the tables that reference it is maintained in
code rather than declared as a foreign key;
[Design decisions](#design-decisions) explains why.

### Record identity

The tables and their keys. `projects` and `repos` are in the registry;
the others are in each project's file:

![The index data model: the registry's meta, projects and repos, and a
project file's tables, with the joins between them](docs/diagrams/data-model.png)

| Table               | Natural key                                                                                                                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `projects`          | `common_dir`, the git common directory, unique; `file` is unique too, so no two projects share a file                                                                                                     |
| `repos`             | `path`, unique and always canonical; `id` is a surrogate the other tables carry                                                                                                                           |
| `file_snapshots`    | `(repo_id, snapshot_sha, file_path)`                                                                                                                                                                      |
| `chunks`            | `(id)` = `blake2b(blob_sha:file_language:kind:name:line_start:line_end, digest_size=8)` — content-addressed, shared across a project's checkouts and paths                                                |
| `edges`             | `(repo_id, snapshot_sha, source_id, target_id, kind, source_path, target_path)` — the whole tuple is the key, the paths included: one import chunk imports a different sibling from each of its locations |
| `indexed_snapshots` | `(repo_id, snapshot_sha)`                                                                                                                                                                                 |
| `watched_refs`      | `(repo_id, ref)`                                                                                                                                                                                          |
| `meta`              | `(key)`                                                                                                                                                                                                   |

### Embedding column

`chunks.embedding` is a variable-length `FLOAT[]`. The dimension
is a property of the configured embedding model, not the schema,
so the column is not fixed-size: the table is created before any
model loads (indexing without embeddings is a supported fast
path), and `search_similar.sql` scores with
`LIST_COSINE_SIMILARITY`, which works on any length. Consistency
is maintained without a stored dimension:

- All stored vectors share one length because they come from one
  model. `_check_embedding_version` clears every embedding when
  `config.embedding_model` changes (see [Versioning](#versioning)),
  so they are re-embedded at the new model's width.
- The embedding staging frame (`EmbeddingStagingRow`) enforces
  uniform length per write batch via the `embedding_dim_is_uniform`
  rule.

The column is also most of the database by size, and a predicate on it
(`embedding IS NULL`) reads all of it. Three paths depend on that
predicate, and each issues it once per request: completeness for a whole
repo comes from one grouped pass (see
[Completion tracking](#completion-tracking)), the worker's job search
reads that same result, and the embed loop resolves its work list in one
pass and then fetches pages by chunk id.

`chunks.embedding_truncated` records whether a chunk's text
exceeded the model's context window and was truncated before
embedding (`embeddings.embed` flags it; `update_embeddings.sql`
persists it). It is a **diagnostic flag only** — written on
embed, never read by search, ranking, or re-embedding.

### Snapshot resolution

Chunks are blob-addressed (shared across snapshots with the
same blob) but visible through the lens of a specific
snapshot's file tree:

```sql
FROM chunks c
JOIN file_snapshots fs
  ON c.blob_sha = fs.blob_sha
  AND c.file_language = fs.detected_language
WHERE fs.repo_id = ? AND fs.snapshot_sha = ?
```

The repo scope, the result row's `repo_id`, and the path all come from
`file_snapshots` (or `_snapshot_refs`), never from the chunk — the chunk
is repo-less and path-less. A chunk shared by several repos, or found at
several paths within one, fans out to one row per location through this
join, which is how a location-sensitive signal like the file-category
penalty still sees each path. Search collapses those rows afterwards,
keeping the best-scoring one and listing every path on the result.

The language is part of the join because it is part of a chunk's
identity: the same bytes extracted as two languages are two chunks, and
each must pair with the files read as that language.

Because the scope is always a repo and a `snapshot_sha` together, the
two travel as one value. `SnapshotRef` carries a `Repo` and a SHA, and
`repo.at(sha)` makes one; `SnapshotRange` carries one `Repo` with a base
and a head SHA for the symbol diff. A query cannot name a snapshot in
one repo and a repo from another, and whatever reads a ref has the
repo's path without a lookup. Neither crosses the RPC boundary, because
clients name repos by path.

Every read and write on the store takes its scope as a keyword, and the
keyword says how many snapshots it holds: `at=` one `SnapshotRef`,
`within=` a list of them, `between=` a `SnapshotRange`. The leading
positional parameter is what the call asks for — the query, the pattern,
the chunk ids, the edges to write — so a call site reads as the subject
first and the scope after it.

### Registered frames as query inputs

Both reads and writes pass polars frames into DuckDB by
**registering them as named views** (`register` → query →
`unregister`), rather than
interpolating values into SQL or issuing one query per
row. The motive is speed: it collapses N round trips into
one set-based operation — one table scan instead of N, one
bulk upsert instead of per-row inserts — and DuckDB reads
the registered frame zero-copy, so one query handles any
number of rows. The multi-vector scan below quantifies the
win (7–32× for 3 vectors).

- **Reads** register a frame and join against it:
  `_snapshot_refs` (the `(repo_id, snapshot_sha)` snapshots a
  search spans — see [Cross-repo search](#cross-repo-search)),
  `_qvecs` (query vectors — see
  [Multi-vector semantic scan](#multi-vector-semantic-scan)),
  and `_chunk_ids` (one page of the embed work list, which the
  fetch matches chunks against by id).
- **Writes** register a staging frame the upsert reads from:
  `_stg` (chunks / snapshots / edges) and `_emb_stg`
  (embeddings), via `WriteSession._bulk_insert`.

The schemas behind these frames are split by direction:
write/staging schemas live in `index/staging.py`, read-result
schemas and the row→model transforms in `index/results.py`,
kept independent by import-linter (a physical CQRS seam).

Registration is cursor-scoped, and the store uses one
thread-local cursor per thread (see
[Concurrency model](#concurrency-model)), so concurrent
calls register identically-named views without colliding —
no locking needed. Registered frames carry a `dataframely`
schema like any other frame, so the view's column types
match the table it joins against. The caller-side
convention lives in the `rbtr-data` skill.

Reads run through `cur.sql(...)` and are read to the end with
`.pl()` or `.fetchall()`; `execute()` runs DDL and DML, and any
rows it returns are read with `.fetchall()`. DuckDB holds a
cursor while an `execute()` result is unread, and refuses
`register`, `unregister` and relation reads on it until then.
The store caches one cursor per thread and file, so one
half-read result would break every later call on that thread. A
relation has no `fetchone()`, so a read through one cannot be
left half-read; it costs about 60 µs more per statement.

A statement that fails inside a transaction aborts it, and an
aborted transaction refuses `unregister`. `registered_views`
therefore lets the statement's exception propagate and leaves
the views to the owner's rollback.

### Completion tracking

`indexed_snapshots` records which `(repo, snapshot)` pairs are
fully indexed. The daemon watcher checks this to trigger
builds. GC uses it to identify orphans: any
`file_snapshots` or `edges` row without a matching
`indexed_snapshots` row is orphan.

`indexed_snapshots` is **completion**; `watched_refs` is
**intent** (which refs to keep indexed). See
[Watched refs](#watched-refs).

### Versioning

`meta` is a key-value table that tracks compatibility, in the
registry and in each project file. Three keys:

- `schema_version`, in the registry — the DDL version of
  both schemas. A mismatch replaces the index with an empty
  one on open: it is derived data, so rebuilding is the only
  migration.
- `embedding_model`, in each project file — the GGUF model
  ID used to compute its vectors.
- `embedding_version`, in each project file — a format
  version for the embedding pipeline.

When the model or version changes, the file's embeddings are
cleared and recomputed on the next build.

**Replacing the index.** Deleting rows does not shrink a DuckDB
file, so a stale index is replaced rather than emptied. Its
registry is unlinked while it is still open, so its lock holds
until a new file exists at the path, and a new registry is then
created there. A process that creates one first wins DuckDB's
lock on it, and this one stops with `IndexLockedError`. The old
project files, which the new registry does not name, are deleted
as residue. An older rbtr finds a newer `schema_version` and
refuses to open the index (`IndexSchemaTooNewError`), changing
nothing.

A fourth invalidation axis lives outside `meta`, per chunk:
`chunks.extraction_serial`, the extraction serial of the chunk's
*own* `language`. For a multi-language file a delegated TypeScript
chunk carries TypeScript's serial, not the host svelte plugin's.
Its only reader is the blob-dedup check, so a plugin can invalidate
only its own stale extractions (see
[Content-addressed chunks and blob dedup](#content-addressed-chunks-and-blob-dedup)),
without the rebuild a `schema_version` bump forces.

**Additive vs breaking changes.** `registry.sql` and `schema.sql`
are applied `CREATE TABLE IF NOT EXISTS` on every writable open,
to the registry and to every project file, so a purely
**additive** table (e.g. `watched_refs`) appears in existing
indexes in place — no `schema_version` bump, no rebuild. Only a
**breaking** change to an existing table or column bumps
`schema_version` (and so replaces the index).

**Constraints reach only new databases.** DuckDB has no
`ALTER TABLE ... ADD CONSTRAINT`, and `CREATE TABLE IF NOT
EXISTS` leaves an existing table untouched. A `CHECK` or
`FOREIGN KEY` added to `schema.sql` therefore applies to
databases created after the change and not to those already
on disk — it will hold on a fresh checkout while being absent
from every index in the field. Applying one everywhere means
bumping `schema_version`, so that each database is rebuilt
with it.

`chunks` carries two: a span ends on or after the line it
starts on, and truncation describes an embedding that exists.
Every index has both, because an index built without them has an
older `schema_version` and is replaced on open.

rbtr declares no foreign keys; see
[Design decisions](#design-decisions).

## Garbage collection

`indexed_snapshots` is the authority. Data without a matching
row is orphan. Both operations live on `WriteSession`:

- `drop_snapshot` — removes all trace of a commit
  (indexed_snapshots row, snapshots, edges, orphaned chunks).
- `cleanup` — removes residue from crashed builds
  (snapshots/edges for uncommitted commits) and stale data
  (chunks not referenced by any snapshot, edges whose
  commit has no snapshots).

Because chunks are content-addressed and **shared by a project's
checkouts**, the orphan sweep is a reference count over the
project's file: a chunk is deleted only when **no** `file_snapshots`
row of **any** of its checkouts references its blob. Dropping a
commit (or a whole checkout) that shared content with another
checkout leaves the shared chunks intact until the last referencing
snapshot is gone. Consequently the chunk count in `GcCounts` is a
per-project figure, not a per-repo one. Content two projects hold
is a copy in each file, collected in each on its own.

This sweep rests on **a chunk entering only for a blob somebody
claims**. `WriteSession._commit` counts the `(blob_sha,
file_language)` pairs the session stored chunks for, and rolls back
when any of them has no `file_snapshots` row in the file. An
unreferenced chunk is therefore one whose claims were removed later,
never a half-written build, so the sweep cannot take chunks another
checkout still needs. The check keys on the blob rather than on
writing a claim in the same session, which is what lets one
re-extraction re-chunk a blob for every checkout sharing it.

**Compaction.** Deleting rows does not shrink a DuckDB file — the
freed space is kept inside it for reuse, so a file never shrinks.
To hand that space back, `gc` finishes by rewriting each project
file it collected: `Index.compact` switches the project to a
fresh copy, search index included (see
[Index files](#index-files)), so search keeps working without a
rebuild.

The switch either commits or leaves the old file live, and the row
deletions are already durable by then, so an interrupted or failed
rewrite leaves the index correct — just not smaller. `gc` logs the
failure and carries on; `--no-compact` skips the rewrite entirely.
The size `gc` reports counts live files only, not the old ones
waiting out `retire_delay_s`.

Both modes rewrite the same way: standalone, `rbtr gc` owns the
database (`handle_gc(..., allow_compact=True)`); in the daemon gc
takes `_write_sem` like every write, and searches keep running
throughout (see [Concurrency model](#concurrency-model)).

The keep-set depends on the `GcMode`. The **default** is
`WATCHED`: keep HEAD, every local branch / tag / note,
**and** every resolved [watched ref](#watched-refs) (the
current worktree tree is always protected). So a routine
`rbtr gc` only drops genuinely unreferenced snapshots — it
never discards anything a branch points at or that you
asked to keep indexed. `WATCHED_ONLY` keeps just HEAD plus
the watch set, dropping unwatched branches/tags — the opt-in
way to reclaim refs you no longer index. `HEAD_ONLY`,
`KEEP`, and `ORPHANS` are the other explicit modes. Each
names a different keep-set, so the CLI refuses two at once
rather than resolving them by precedence. If a mode drops a
still-watched snapshot, the watcher rebuilds it on the next
poll (self-healing); an unwatched ref is rebuilt when
somebody asks to read it. Collection never edits the watch set
itself — that is `rbtr unwatch`'s job — so a thorough tidy trims
first and collects second.

GC is **per-repo by default**; `rbtr gc --scope all` reclaims across
**every** registered repo at once (`GcAllRequest` ⇒ `run_gc_all`
loops `run_gc` over `repos()`), and each repo's sweep reclaims
the chunks no surviving snapshot of its project references.
Globally it takes **`WATCHED` or `WATCHED_ONLY`** — the retentions
stated in a repo's own terms, so every repo keeps at least its HEAD and
its watch set. `HEAD_ONLY` and `KEEP` name one repo's refs and
`ORPHANS` sweeps one repo's residue, so `GcAllRequest` cannot carry
them. A repo is **skipped** when git cannot answer for it: a path
that no longer resolves (never purged — forgetting it is a separate,
explicit action; see below) or an unborn HEAD, which `run_gc` raises on
and which must not end a pass over every other repo. Each repo is
reported through `on_progress` as it is collected: inline, as a bar on
stderr; from the daemon, as a `gc_progress` notification between
`gc_started` and `gc_ended`, which is sent however gc ends. None of
the three names a repo, since one gc covers many, and none touches a
running build's progress.

**Forgetting a repo** removes it entirely from the index, undoing what
[registration](#repo-registration) set up (`Store.forget`).
A project's last checkout takes the project with it: one registry
transaction deletes both rows, and the project's file is retired and
deleted. Another checkout's `watched_refs`, `indexed_snapshots`,
`file_snapshots` and `edges` are deleted from the project's file in one
transaction, then its `repos` row.
That is **metadata-only**: it deliberately does not sweep chunks, so it
reports no statistics; the chunks it orphans are reclaimed by the next
GC (or build cleanup), keeping removal cheap and uniform with ref
removal.

`rbtr forget` forgets the current repo, but only when HEAD is its sole
watched ref; `rbtr forget --stale` forgets every repo whose stored path
no longer resolves. The latter is **daemon-driven enumeration**, not a
per-path request: a removed checkout's path cannot be normalised into a
request, which is why it is a request of its own —
`ForgetStaleRequest` carries no path, and
`rbtr.index.watch.forget_stale_repos` walks `repos()` and forgets
the unresolvable ones.

Its sibling `remove_stale_refs` answers the other
question — which of a *live* repo's watched refs git still resolves, in
one repo or in all of them — and leaves a vanished repo alone, since git
cannot answer for it. The two are separate commands over separate state,
so each runs in its own request and its own sessions.

Outside [maintenance](#maintenance), forgetting is **explicit**:
`poll_watched` skips a vanished path rather than purging it, since the
absence may be transient (an unmounted volume). The wire surface is a dedicated
`ForgetRequest`/`ForgetResponse` (the response carries only the
forgotten paths), kept separate from the reclamation-shaped
`GcResponse`.

### Maintenance

When the daemon's worker has had no job for `maintenance_idle_s`
(10 min), and at most every `maintenance_interval_s` (6 h), the
daemon runs `run_maintenance` under `_write_sem`. In order, it:

1. stops watching refs git no longer resolves (`remove_stale_refs`);
2. forgets checkouts whose path has been gone for `forget_after_s`
   (3 h): `repos.missing_since` records the first pass that found a
   path gone, and is cleared when the path is back;
3. collects every repo in `WATCHED` mode (`run_gc_all`);
4. compacts each project file grown past `compact_growth_ratio` (2)
   times its `compacted_bytes`.

Each step removes only what `rbtr unwatch --stale`, `rbtr forget
--stale`, `rbtr gc --scope all` and `rbtr gc --compact` would. A pass
is published as a gc (`gc_started`, `gc_progress` per repo,
`gc_ended`), which clients already show, and logged as
`maintenance_started` and `maintenance_finished` with what it
removed. The same task deletes the files compactions replaced, every
idle poll, once `retire_delay_s` has passed.

## Search fusion

![How a search is answered: prepare the query, run three searches on
every project file at once, then merge, fuse and rerank](docs/diagrams/search-flow.png)

Three channels fused into one ranked list:

- **Name matching** — case-insensitive substring matching.
- **BM25 keyword search** — DuckDB FTS over code-aware
  tokenised content.
- **Semantic similarity** — exact cosine between query and
  chunk embeddings. Local GGUF model, no API calls.

![How scores are fused: normalise, weigh, multiply, then blend the top few
with the reranker](docs/diagrams/fusion.png)

Each channel is normalised to [0, 1], then combined:

```text
score = alpha × semantic + beta × lexical + gamma × name
```

Four multipliers then scale that score. They carry what a
channel score cannot see: that some kinds of chunk are better
answers than others, that some files matter less than their
contents suggest, and that a symbol many things depend on is
more often the one you meant.

**Kind boost** — what the chunk is. A class scores 1.5, a
function or method 1.3, a variable 1.0, a doc section 0.8, a
test function 0.7, a config key 0.6, a comment or raw
line-chunk 0.5, a migration 0.4, an import 0.3. An import line
matches a name as well as the definition does and is almost
never what was wanted, which is what the bottom of that range
is for.

**File category** — where it lives. Vendored and generated
code 0.3, tests 0.5, config 0.7, documentation 0.8, ordinary
source 1.0. Classified by path pattern, first match wins.

**Importance** — how much depends on it:
`1 + log2(1 + degree) / 4`, capped at 3.0, where `degree` is
the inbound edge count. One inbound edge gives 1.25, three
1.5, fifteen 2.0. Logarithmic because the step from one
dependent to three says far more than the step from thirty to
thirty-two.

**Proximity** — distance from the current diff, when there is
one. 1.5 for a chunk in a changed file, 1.2 for one with an
edge to a changed file, 1.1 for one in a changed directory,
1.0 otherwise.

These magnitudes are hand-set. The fusion weights below are
not — they are measured (see
[`rbtr-eval`](../rbtr-eval/README.md)).

Weights are configured per query kind. Every search
classifies the query via `classify_query` and selects the
matching `(alpha, beta, gamma)` triple from
`config.search_weights`. The classifier is a cheap
heuristic (additive scoring on syntax characters and
language keywords), so it runs on every query regardless
of whether expansion is enabled. Per-query overrides
via `SearchRequest.weights` take precedence over the
config (used by the tuner). Run `rbtr config` for
current defaults.

### Cross-repo search

`SearchRequest.scope` (`workspace` | `all`) controls
breadth. `workspace` searches the single repo at `path`;
`all` searches every indexed repo in the index and
merges the hits into one ranked list.

Both modes run the *same* pipeline — `_retrieve` →
`fuse_scores` → reranker → `materialise_scored`. The only
difference is the list of refs fed in. Rather than branch
into parallel SQL or duplicate the channel methods,
`_retrieve` takes a `list[SnapshotRef]` (see
[Snapshot resolution](#snapshot-resolution)) and each channel query
joins against a cursor-registered temporary view,
`_snapshot_refs(repo_id, snapshot_sha)`. For a workspace search
the view holds one row; for a search across every repo
(`search_all`) it holds one per repo. This mirrors the `_qvecs`
register/unregister pattern used for multi-vector semantic scan,
and needs no scalar `repo_id`/`snapshot_sha` bind params in the
search SQL.

Each channel runs on every project file the refs span, all at
once, with that file's refs in its view (see
[Index files](#index-files)). The store merges what the files
return: the lexical and semantic channels keep the best *k* of
each file's best *k*, the name channel keeps the best match tier
any file found, and a chunk's inbound degree is summed over files.
Each file scores BM25 against its own document lengths, so lexical
scores differ slightly from those of one shared index.

`handle_search` builds the refs list: a one-element list
from `resolve_read_ref` for `workspace`, or
`latest_refs(index)` for `all`. `latest_refs`
applies the same dirty-or-HEAD resolution as
`resolve_read_ref` to every registered repo — preferring
an indexed dirty worktree tree SHA, else the newest indexed
commit — and skips repos with no indexed data.

Result attribution flows through a `repo_id` column carried
on the chunk-result frames (`ChunkResultRow` →
`FusedRow`). When the handler asks for attribution
(`attribute_repos`), `materialise_scored` maps `repo_id` to
the path of the searched ref's repo, populating
`ScoredChunk.repo_path`. It stays `None` for workspace
searches, so single-repo consumers are unaffected. `find_refs`,
`changed_symbols`, `read_symbol`, and `list_symbols` remain
single-repo: their edges and ref comparisons don't span
repos.

### Preview anchor

`materialise_scored` attaches a render hint to each result:
`match_line_offset` (the `"\n"`-line of `content` that best
matches the query) and `matched_terms` (the query tokens
present in it). Both come from `match_preview_exprs` — polars
expressions evaluated over the fused top-k frame, so there is
no extra pass or DB round-trip. The token set is
`tokenise_code(query + keywords)`, the same tokens BM25 saw,
so the preview agrees with why the lexical channel retrieved
the hit; substring containment lets a fragment like `deps`
anchor inside `AgentDeps`. A hit with no literal overlap
(e.g. a purely semantic match) gets `None`/`[]`, and renderers
fall back to the first lines. The CLI emphasises the whole
matched line (rich `Syntax` highlight, which can't overlay
sub-spans); the pi TUI highlights the matched terms within it.
Both renderers window a few lines around the anchor; when that
window scrolls past the chunk's start to reach a deep match,
they keep the first line (the signature) above the elision, so
a result shows what it is as well as where it matched.

### Read-ref resolution and scoping

Read handlers pick their ref through `resolve_read_ref`, in
`index/refs.py`, the one module that decides which snapshot a
read means. It asks git about the worktree and the index which
snapshots are indexed. With no explicit ref, a read takes the
indexed dirty-worktree tree SHA if there is one, else `HEAD`. The
ref has to be indexed to read from, and the two cases differ on
purpose:

- An explicit ref that isn't indexed is an error — you asked
  for that ref, so we don't quietly answer from another.
- An *implicit* ref that isn't indexed yet (a build still
  finalising) falls back to the latest indexed snapshot, so a
  read returns a slightly stale answer rather than nothing. It
  only errors when the repo has no indexed snapshots at all.

`file_paths` is normalised to repo-root-relative POSIX form,
so absolute, `./`-prefixed, and relative inputs all match the
stored (repo-relative) chunk paths.

### Query expansion

Query expansion is client-supplied: the caller passes
`keywords` and `variants` directly on `SearchRequest`,
which strips, de-duplicates, and caps them
(`config.max_expansion_keywords`). In pi sessions the
session LLM generates them inline as tool-call
parameters, guided by `promptGuidelines`. In eval the
`expand` DVC stage pre-generates them via an LLM API.

A heuristic classifier (`classify_query` in
`index/classify.py`) determines the query kind, from
which `search()` selects the per-kind fusion weights.
The caller may force the kind with the
`SearchRequest.query_kind` override; otherwise the
query text is classified.
Classification uses additive scoring: syntax characters
(braces, operators, function-call patterns) and language
keywords each contribute points. A query is classified
as `CODE` when its total reaches a threshold of 2.
Keywords are extracted at runtime from all registered
tree-sitter grammars, so new language plugins
automatically contribute.

`CONCEPT` additionally requires low punctuation density
(`_punct_ratio <= 0.04`) on top of the word-count minimum.
Word count alone routes any structured name long enough to
split — a CSS selector, a dot-joined path — down the concept
path, where it gets variant rephrasings that dilute the exact
lexical signal it needed. The density threshold costs a fraction of true
concept queries and removes most of that misrouting; the
[eval](../rbtr-eval/README.md) has the measured trade.

| Kind           | Example                       | Expansion strategy       |
| -------------- | ----------------------------- | ------------------------ |
| **concept**    | `"how does idle unload work"` | keywords + NL variants   |
| **identifier** | `fuse_scores`                 | keywords only            |
| **code**       | `def fuse_scores(`            | expansion skipped        |

Keywords are synonym identifiers (e.g.
`getName` → `fetchName`, `retrieveName`) appended to
the BM25 query. Variants are natural-language
rephrasings embedded alongside the original query to
widen the semantic candidate pool. Identifier and code
queries skip variant generation — NL rephrasings would
add noise to queries already well-served by BM25 and
name matching.

### Cross-encoder reranking

When a reranker model is configured (on by default), a
cross-encoder re-scores the top fusion candidates before
results are returned. The reranker evaluates each
`(query, chunk)` pair with full cross-attention, improving
ranking precision — particularly promoting correct results
from lower fusion positions to position 1.

The pipeline after fusion becomes:

```text
fuse → rerank → materialise
```

`rerank()` takes the fused frame, scores each row with
the cross-encoder, normalises scores to [0, 1], and
blends them with the fusion score:

```text
score = w × fusion + (1 − w) × reranker
```

where `w` is the per-kind `blend_weight` from
`config.reranker_settings`. The result is re-sorted
and trimmed to `top_k`.

Pool size and blend weight are configured per query kind
via `reranker_settings` (a `dict[QueryKind,
RerankerSettings]`). Every search classifies the query
and selects the matching entry. `SearchRequest` can
override both via `reranker_pool` and
`reranker_blend_weight`. The actual pool is
`max(pool, request.limit)` so the reranker never
discards candidates the caller asked for.

The `Reranker` class follows the same `GpuModelSlot`
lifecycle as the embedder — lazy load, idle
unload, `_gpu_lock` serialisation. On model failure,
`rerank()` returns fusion-order results with `reranker`
scores at `0.0` (graceful fallback).

### Multi-vector semantic scan

When expansion produces variants, the original query
vector and all variant vectors are evaluated in a
single DuckDB scan. The vectors are registered as the
cursor-scoped view `_qvecs` and joined
via `CROSS JOIN` + `MAX(cosine_similarity)` so each
chunk is scored against the closest matching vector.
This avoids N separate table scans — benchmarked at
7–32× faster than sequential scans for 3 vectors.

### FTS index lifecycle

Lexical search uses DuckDB's FTS extension, scoring with BM25F
over `name_tokens` and `content_tokens`. An FTS index is ordinary
tables and SQL macros (`match_bm25`) that the extension writes
into the file. It persists across close and reopen, and does
**not** update after `INSERT` or `DELETE`.

Each project file has its own FTS index, made with the file over
its still empty `chunks` table, so every file has one.

**IDF is neutralised.** After each build every term's document
frequency is set to 1 (`neutralise_fts_idf.sql`), so a score
depends on term frequency and document length only. In code a
term common across chunks is often the one asked for — `config`,
`model` — and IDF would suppress it.

**Write path** (build, GC): a [write session](#write-path) that
modified chunks rebuilds its file's FTS index after it commits
(`overwrite = 1`). The build and the IDF update run in one
transaction, so a failure leaves the previous index whole and no
index is left with its IDF in place. Making a file and this are
the only paths that write FTS DDL; a compaction copies the index
with the rest of the file.

**Read path** (search): queries the persisted index and never
rebuilds. A search running during a rebuild reads the previous
index until the rebuild commits, so a build in progress still
returns results for the chunks indexed before it.

`incremental = 1`, the extension's trigger-maintained index, does
not replace the rebuild. It rejects the chunk upsert's
`ON CONFLICT DO UPDATE` ("not yet supported with REFERENCING NEW
TABLE AS triggers"), leaves rows changed by `UPDATE` unindexed, and
increments `dict.df` on insert, which undoes the neutralised IDF.
Revisit if the extension supports upserts and stops maintaining
`df`.

## Storage models vs API DTOs

The read commands (`search`, `read-symbol`, `list-symbols`,
`find-refs`, `changed-symbols`) do not return the storage
models (`Chunk`, `ScoredChunk`, `Edge`). Those carry
persistence detail — identity hashes (`id`, `blob_sha`), the
embedding, the full ranking-signal breakdown, an
always-present `metadata` bag — that no caller needs and that
adds noise and tokens to an agent's context. Handlers instead
project to output DTOs in `rbtr.daemon.dto` (`SymbolRefOut`,
`SymbolOut`, `SearchHitOut`, `RefOut`): a curated, low-noise
public contract. The boundary is also a vocabulary shift: the
internal unit is a content-addressed `chunk`, but the public
API speaks of `symbol`s — the term agents actually navigate by.

How much source a shape carries follows from what its command
is for. `read-symbol` returns a body, in `SymbolOut`; that is
the command whose whole job is to fetch one. `search` returns
a capped preview, enough to judge a hit's relevance. Anything
that enumerates — `list-symbols`, `changed-symbols` — returns
`SymbolRefOut`: a name, a place, a span, and no `content` field
to put a body in. An outline of a large file would otherwise
cost an agent more context than reading the file, which is the
opposite of what it is asked for. Each shape is its own type
rather than one type with an optional body, so the wrong
payload cannot be assembled by accident.

Empty `metadata` and null `repo_path` are omitted,
as is the preview anchor (`match_line_offset`, `matched_terms`)
when the hit has no literal match; the nine search signals
collapse to a single `score` unless
the request sets `explain`, which attaches a nested
`SearchSignals` (used by the eval tuners). The DTOs are
output-only — never fed to the write/staging path — which is
what lets them omit fields freely; the storage models, which
the same `model_dump` serialises into DuckDB staging frames,
cannot.

## Daemon

The daemon answers requests about code while keeping that code indexed as
repositories change. These duties have different timescales. Reading a
symbol might take milliseconds; rebuilding a repository or computing its
embeddings can take much longer. Making every request wait for background
work would turn an otherwise usable index into an unavailable one.

The design separates responsibility for answering a client from
responsibility for keeping stored data current. RPC owns each accepted
request until its reply has been attempted. Indexer decides what background
work is due. Both use Index, the same synchronous storage API available to
inline CLI and library workflows. Daemon connects these parts and owns their
shared resources and lifetime. The separation lets transport and indexing
policy change without making the database API a service, or making request
handling responsible for background scheduling.

![Separate client service and background policy share storage inside one
application lifetime](docs/diagrams/daemon-tasks.png)

A single event loop coordinates work whose progress depends on other work:
waiting for a request, a resource or a result. Blocking Git, database and
model operations run on executor threads so they leave that loop available.
The distinction matters more than the number of tasks. A request waiting
for the GPU must not consume the thread that could answer an unrelated
status query. Sharing an executor avoids assigning permanent threads to
components, although CPU, database and executor contention can still delay
otherwise independent requests.

### Requests and background work

A client's request and the indexing it may cause have different completion
conditions. Saving a watch succeeds once the requested intent is stored;
the resulting build may take much longer. Holding the request open until
that build finished would couple client availability to the whole indexing
pipeline. Instead, the reply confirms the saved watch, and subsequent
notifications describe background progress.

This also separates responsibility for delivery. A reply belongs to one
requesting client, whereas progress can interest several subscribers,
including clients that did not ask for the build. Request/reply traffic and
published notifications therefore use separate channels. A disconnected
client does not cancel work that has already started, and slow notification
delivery does not determine what a status query reports.

![Replies confirm individual requests; published progress describes shared
background work](docs/diagrams/protocol.png)

Background selection derives work from Git and stored data. A wake means
“look again”, not “perform this queued job”. That keeps an interrupted build,
a branch move and a restart on the same path: compare what should be indexed
with what has completed, then select the outstanding work. Builds take
priority over embedding so new source becomes readable before an older
vector backlog is exhausted.

### State and progress

Stored data and live progress answer different questions. The databases say
which snapshots and vectors are available. Live progress says what the
background worker is currently trying to do. A build can be selected but
still waiting for writer access, so an active job is not proof that its
native computation has begun.

The worker keeps one immutable snapshot of its current run. Run identity,
start time and progress are replaced together, allowing a status reader on
another thread to capture a coherent answer under a short lock. Holding
that lock only for capture and replacement keeps status independent of
long database operations. Updates are associated with their originating
run, so a late report cannot overwrite a newer run's progress.

State changes happen before output is published. A subscriber can be slow,
absent or newly connected without becoming the source of truth for status.
The producer nevertheless waits for its output callback and channel send
to finish. That wait lets shutdown account for both native work and the
output it can still produce before disposing of the transport.

Indexer reports internal typed values rather than public wire messages.
Daemon translates them for the notification channel. This boundary lets
background indexing have a different output consumer while leaving the
public protocol under the runtime's control. The protocol reference below
defines the external messages.

### Watched refs

Watched refs are durable intent; completed snapshots are durable results.
Keeping these separate lets a watch exist before its first build and lets
the system distinguish a moving branch from the commit it last finished.
Symbolic refs are resolved again when polling; a bare SHA settles after its
one snapshot has been built. Several refs pointing at the same repository
snapshot produce one build target.

Every registered repository tracks HEAD. Default watches are seeded during
exclusive startup, and removing HEAD is rejected atomically. Other watches
can be added or removed independently. Removing a watch or forgetting a
missing checkout changes what should be retained; garbage collection later
reclaims data that is no longer needed. Changes across repositories use
separate transactions and can be repeated after interruption.

A failed build is suppressed in memory so the watcher does not retry the
same failure on every poll. A ref moving to another snapshot makes new work
eligible; restarting makes the failed snapshot eligible again. A failure
in selection, watching or maintenance instead stops the application.
Continuing to answer queries while the mechanism responsible for freshness
has silently failed would give clients a misleadingly healthy service.

### Concurrency model

Independent execution does not mean unrestricted execution. Requests are
separated by the resource they need, so a backlog of GPU searches or writes
does not consume the capacity reserved for ordinary reads. The default is
two read workers, one GPU worker and one writer worker. Those counts limit
executing requests, not the number of threads or pending requests.

Extra requests wait in memory. The queues are uncapped, so sustained excess
arrivals increase both waiting time and retained memory. This favours
accepting work over returning capacity errors; it establishes neither a
latency guarantee nor a fixed memory bound.

Foreground requests and background work share one GPU lock and one writer
semaphore. Separate locks for the two sides would preserve apparent
independence while allowing them to use the same resource concurrently.
Waiting for either shared resource happens on the event loop before native
submission, keeping those waits out of the executor.

GPU ownership covers a whole foreground search, including query embedding,
fusion and reranking. Background embedding releases the GPU after each
inference batch, then acquires writer access to persist the result. A
waiting search can therefore run between batches instead of waiting for
the repository's entire embedding job. Status and ordinary symbol reads
need neither resource.

![Shared resource ownership permits reads during GPU work and searches
between embedding batches](docs/diagrams/daemon-locks.png)

The writer semaphore protects complete write workflows rather than
individual database calls. Compaction illustrates why: it copies a project
file and switches to the copy. A batch committing to the old file during
that copy would be missing from the replacement. Locking each call
separately would still leave the gap. Builds retain writer access through
their phases; embedding retains it for one persistence batch. Collection,
watch edits and maintenance use the same writer.

This application-level coordination is distinct from DuckDB's file and
transaction rules. Reads use thread-local cursors and can overlap writes.
A read already using a file finishes against it when compaction switches
to a replacement. Cross-project reads also have their own query pool;
[Index files](#index-files) explains the storage-level ownership. A DuckDB
connection busy with one thread's call refuses another thread's call
rather than waiting, so the store's root connection runs `ATTACH` and
`DETACH` only under the writer semaphore, and other threads only take
cursors from it, which a busy connection still allows.

Model handles need protection beyond scheduling inference. Idle unload
takes the same GPU lock so it cannot dispose of a model still in use.
Per-handle thread locks protect handle access, and a separate shared load
lock serialises native model initialisation. Each lock covers the resource
lifetime it actually protects.

### Standing aside

There is one background job runner. Even with independent foreground
requests, a long embedding job could keep a newly due build waiting for
every vector to finish. Embedding therefore checks for builds after each
committed batch and ends its current run when one is due. Selection gives
the build priority, then returns to the vectors still missing in storage.

The database is the continuation: committed vectors remain available and
unwritten vectors remain outstanding. There is no separate suspended-job
state to reconcile after a crash. The embedding outcome distinguishes
finishing a pass from standing aside or stopping, with stored counts showing
the snapshot's actual completeness.

A build retains writer access instead of yielding between phases. Cleanup
in a competing checkout's build can remove chunks whose snapshot is not yet
marked complete; allowing that cleanup into an unfinished build would break
the completion boundary. Embedding has a safe yield point after each finite
commit, whereas a build needs the wider protected workflow.

### Daemon startup

One daemon serves a data directory. The writable Index takes DuckDB's
process/file lock before the runtime binds sockets, so competing starts
converge on one owner rather than coordinating ownership in their parent
processes. A losing start can encounter the lock before the winner has
published discovery. Clients therefore allow the winner time to become
ready instead of treating that lock failure as immediate startup failure.

Construction performs synchronous writes to seed default HEAD watches.
It is an exclusive startup operation: the caller constructs before sharing
the writable Index, or holds its writer semaphore during construction,
off a UI loop. This keeps initial watch intent established before serving
without introducing asynchronous component initialisation.

Readiness has a narrower meaning than a fully prepared index. The runtime
checks required language plugins, binds its sockets and publishes process
discovery before reporting ready. Models can still need loading and
snapshots can still need building or embedding. Background selection makes
an initial pass over stored intent without waiting for the first watcher
poll.

Discovery records the process identity and connection details, not work in
progress. Process status consequently differs from a query of index state,
and a snapshot-ready notification differs from both: it reports a completed
build's counts, which can still show vectors awaiting computation.

### Stopping

Shutdown is an ownership problem before it is a cancellation problem.
Cancelling an async wait does not stop the native function running behind
it. Releasing a GPU lock, closing a model or switching a database file while
that function continues would let the application dispose of resources
still in use.

Accepted client work therefore has a precise start boundary. Queued calls,
resource waiters and executor submissions that no native thread has claimed
can still be rejected with `shutting_down`. A native thread claims a call
under the same short lock used by shutdown. Whichever side wins determines
whether the call may run; submission to the executor alone is insufficient.
Started calls retain their resources until native execution and the reply
attempt finish.

This produces two completion points. A shutdown reply confirms that
foreground requests have been rejected or finished. Application shutdown
also waits for background work and the output it can still produce.
Notification transport remains available until those producers finish.
Only afterwards can models and sockets be closed.

Background work stops cooperatively at boundaries it can safely leave.
An interrupted build transaction rolls back while earlier committed phases
remain. An embedding run reports its stopped outcome, retaining committed
batches. Maintenance already running finishes with writer access retained.
Cooperative stopping is not an indexing failure.

Every scheduled startup model load also remains owned until completion,
even one still waiting for an executor thread or a companion load after
another fails. Waiting can extend shutdown, but it prevents a load from
creating a model after model cleanup has already run. A caller-supplied
Index remains the caller's responsibility and stays open until serving
has finished.

Client wait budgets and process-exit checks measure different parts of this
sequence. The Python client waits for its shutdown reply before starting
its exit checks and possible SIGTERM fallback. Those checks are not an
overall stop deadline; native work and model loading can delay final exit.

### Crash recovery

Recovery uses the same mismatch between intent and completion that drives
normal indexing. Watched refs and working trees identify desired snapshots;
completion records identify finished builds; missing vectors identify
outstanding embedding. A restart selects work from those inputs rather
than restoring a transient job queue.

Only the active transaction rolls back on failure. Earlier committed build
phases and embedding batches remain available for reuse. A later build or
collection removes snapshot rows never marked complete, while a new
embedding run reads the remaining missing vectors. Standing aside and
crashing therefore resume through the same stored state.

Partially built or retired project files are reconciled against the registry
on the next writable open, as described in [Index files](#index-files).
Live progress, failed-build suppression and accepted client requests
disappear with the process. Recovery of indexing does not imply replay of
client requests; the [protocol](#daemon-protocol) explains why a client
missing a reply must treat execution as uncertain.

### Daemon protocol

Two ZMQ sockets, both bound in `runtime_dir`:

- **ROUTER** (`daemon.rpc`) — independently routed replies to REQ clients.
- **PUB** (`daemon.pub`) — fan-out notifications.

Messages are JSON, discriminated on a `kind` field. Every
request has a `kind`; every response echoes it or returns
`kind: "error"`.

Each message has one JSON body frame. ROUTER keeps the complete
return envelope through its empty delimiter, including any REQ
correlation frame, and sends the reply with that envelope unchanged.
Completion order may differ from receive order. Socket operations
stay on the event loop. Mandatory non-blocking replies make an
unreachable or non-reading peer a local delivery failure; a
disconnect does not cancel its running handler. Other transport
failures stop intake and use the [shutdown procedure](#stopping).
A malformed body with a usable return envelope receives
`invalid_request`; a message with no usable envelope is logged
and discarded.

Requests and responses are pydantic models (`messages.py`).
The daemon writes a status file to `runtime_dir` on startup;
`DaemonClient` reads it to connect. That file is a model too (`DaemonStatus`), read
through pydantic like any other message: unknown fields are
ignored, so an older client still finds a newer daemon's
endpoints, and a missing or malformed file reads as no
daemon.

`StatusResponse` is the most complex response. For one repo (`status`), `repo_path` is
the repository's normalised root, even when nothing in it is indexed, so a client in a
subdirectory or behind a symlink learns what its paths are relative to. `state`
(`IndexState`) says what the index holds for the request: no repository at all, not this
one, this one but not the snapshot a read without a ref would use (`resolve_read_ref`
falls back to the latest indexed commit), or that snapshot. The daemon computes it where
the index is read, so the CLI, the footer and the status tool word the same four cases
alike. Per-ref state is grouped in `IndexedRef` (sha, symbolic names, chunk count, embed
completeness) — a single list rather than parallel maps. `active_build` and
`active_embed` share the same `ActiveJob` type; the field name distinguishes them. Both
are filtered to the requested repo so a status query never leaks activity from unrelated
repos. Text renderers (CLI, TUI, LLM) derive output solely from the response model —
never from external state like config. `watched` lists the repo's watch set (each
`WatchedRef` carries the resolved sha and whether it is indexed yet).

PUB carries build/embedding progress and completion, automatic rebuilds,
index errors and collection start/progress/end notifications. The
[`Notification` union](src/rbtr/daemon/messages.py) defines the payloads.
Snapshot `ready` reports indexed chunks, embedded chunks and edges; it
establishes snapshot readiness for reads, while its counts can still show
outstanding embedding work.

[Requests and background work](#requests-and-background-work) explains the
separate completion and delivery responsibilities of replies and notifications.

A concrete search exchange:

```json
→ {"kind": "search", "repo_path": "/path/to/repo", "query": "retry logic", "limit": 10}
← {"kind": "search", "results": [{"name": "retry_with_backoff", "kind": "function",
    "file_paths": ["src/client.py"], "line_start": 12, "line_end": 30,
    "preview": {"text": "def retry_with_backoff(operation):", "clipped": true,
                "total_lines": 19}, "score": 0.87}],
   "resolved": {"sha": "abc123…", "source": "head"}}
```

A progress notification (PUB):

```json
{"kind": "progress", "repo_path": "/path/to/repo", "phase": "parsing", "current": 42, "total": 177}
```

Error responses carry a `code` field:

- `index_not_built` — what the read needs is not indexed,
  and nothing will index it. Client should trigger
  `rbtr watch`.
- `index_in_progress` — the daemon will index what the read
  needs without being asked; retry after the `ready`
  notification.
- `repo_not_found` — the supplied path does not identify an available Git checkout.
- `index_locked` — another process holds the index's write
  lock; retry once it has let go.
- `shutting_down` — the daemon rejected this call before its native work started.
- `out_of_memory` — DuckDB needed more memory than
  `duckdb_memory_limit` allows; the CLI and pi add how to
  raise it.
- `invalid_request` — malformed or missing fields; the message
  names each offending field and the value it received, so a
  caller can see how an argument was mis-shaped.
- `internal` — unexpected server error.

The Python client's local `DaemonBusyError` uses `daemon_busy` for discovery,
wait-budget or liveness failure. It is distinct from a returned error:
once a request has been sent, a missing reply does not establish whether the
operation ran. The client sends once and keeps receiving on the same REQ
socket, checking process liveness at its configured interval until its reply
budget expires. A budget failure does not cancel the daemon's native work.
A new send requires a fresh REQ socket, but the client does not automatically
resend an ambiguous request. Process checks belong to the Python client;
other clients follow their own wait/abort policy.

A missing-index exception names the snapshot/ref a read needed. After
resource release, RPC calls the runtime's bound error mapper on the native
executor. The mapper reads the Indexer's coherent run and stored watches to
choose `index_in_progress` when that exact target will be built, otherwise
`index_not_built`. An unrelated running build is not evidence that this
request's target will become readable.

**Schema generation.** `messages.py` is the source of
truth. `rbtr schema-dump` produces JSON Schema.
`just schema-check` regenerates `protocol.ts` (the
TypeScript types used by `pi-rbtr`) and fails CI if the
generated file is stale.

## Observability

rbtr's own code logs through `structlog.get_logger(__name__)`,
producing event dictionaries (a short snake_case `event` name plus
keyword fields). `rbtr.logging.configure_logging` is the single setup
point; see its docstring for the call contract.

### Pipeline and sinks

```text
rbtr code ─ structlog.get_logger ─┐
                                  ├─ shared processors ─ ProcessorFormatter ─ handler ─ sink
third-party ─ logging.getLogger ──┘   (foreign_pre_chain)
```

Application events and third-party `logging` records (llama.cpp,
huggingface_hub, duckdb, pyzmq, pygit2) converge on **one** stdlib
handler whose formatter is a `structlog.stdlib.ProcessorFormatter`.
Foreign records are lifted through a `foreign_pre_chain` that mirrors
the structlog processors, so every line carries the same fields
(level, logger, ISO timestamp, callsite) regardless of origin. A
single sink means level, format, and rotation are configured once.

### Renderers

- **CLI** → `StreamHandler` on stderr. `ConsoleRenderer` (coloured)
  on a TTY, JSON otherwise. stdout is reserved for command output
  (`cli.output.emit`), so logs never corrupt a `--json` result.
- **Daemon** → `RotatingFileHandler` on `daemon.log`, always JSON
  lines, sized by `log_max_size` / `log_backup_count`.

`log_format=auto` chooses console-on-TTY-else-JSON for the stream;
`console` and `json` force one. A rendered daemon line:

```json
{"event": "request_complete", "level": "info", "logger": "rbtr.daemon.server",
 "timestamp": "2026-06-15T10:00:44Z", "func_name": "_dispatch", "lineno": 846,
 "request_id": "a1b2c3d4", "kind": "search", "repo": "/p", "elapsed_ms": 12.4}
```

### Correlation

Keys are bound with `structlog.contextvars` so they merge into every
event emitted within their scope, and reset on exit:

- `_dispatch` binds `request_id` (+ `kind`, `repo`) per request.
- the job worker binds `job_id` (+ `repo`, `ref`, `job_kind`) per job.
- `_watcher_loop` binds a `watch_cycle` id per poll.

Bindings ride `asyncio.to_thread` — it copies the context — so a
search handler running in a worker thread, or a build/embed running
in `to_thread`, still tags its lines with the originating request or
job. This is what lets one request or background job be followed
across `server`, `handlers`, `search`, `index`, and the index
pipeline.

### Timing

Latency is logged as structured fields: `request_complete`
(`elapsed_ms`) per dispatch, `embedded_chunks` (`elapsed_ms`) per
embed run, `index_complete` (`elapsed_seconds`) per build, and
`reranked_candidates` (`elapsed_ms`) per rerank.

Watch-set lifecycle events: `watched_ref_stale` (a watched ref
resolved to an un-indexed SHA, naming `repo`/`ref`/`sha`),
`watched_refs_added` / `watched_refs_removed` (an `index`
add/remove request), `head_watch_backfill` (startup seeding), and
`gc_complete` (mode + drop counts). The build job binds `ref` into
its log context, so `index_complete` names the SHA it indexed.

### Testing

Log behaviour is asserted on captured event dicts via
`structlog.testing` (the autouse `log_output` fixture); see the
`rbtr-testing` skill for the fixture and conventions.

## Design decisions

### Git and extraction

**pygit2 over git CLI.** A build reads every blob in a tree.
Through the CLI that is a subprocess per object, and the cost
is process spawn rather than I/O. pygit2 reads the object
store directly, so walking a tree and fetching blobs stays
in-process, and the working-tree tree SHA
([Identity: tree SHAs](#identity-tree-shas)) can be computed by
building an index in memory and writing a tree object — no
staging area is touched, which a CLI-based equivalent could
not promise. The cost is a compiled dependency with its own
libgit2 version skew, paid at install time rather than per
build.

**Staging one path at a time, not `add_all`.** `add_all` compares
each changed file through a memory map, and a file that shrinks
while it is mapped kills the process with SIGBUS, which nothing in
Python can catch: a daemon dies this way mid-snapshot, with
nothing logged. With a writer truncating a changed file, `add_all`
crashed within 60 s; staging each path git status lists ran 12,674
snapshots in 60 s without a crash, and was no slower (290–371 ms
against 324–644 ms for 1,100 changed files). The errors it raises
instead, for a file that changes mid-read, skip that poll's snapshot.

**A cap on untracked files.** Staging a file for the working-tree
snapshot reads, hashes and compresses it, on every watcher poll
while the tree is dirty, and indexing drops oversized and binary
files only after that. Without a cap, untracked data, caches and
nested checkouts are re-read every poll and never indexed: on a
checkout where git listed 6,749 untracked files, among them a
10 GiB database, that was 29 MiB/s of reads while the daemon was
idle. Counting what git's status already lists needs no file
access, and new work stays well below the cap, so it is indexed
before `git add`.

**Entry-point registry for language plugins.** Languages are
discovered from the `rbtr.languages` entry-point group via
`importlib.metadata`, each resolving to a `LanguageRegistration`.
No hook framework (pluggy): the domain is single-dispatch by
language id, where a plain registry is simpler and a
multi-dispatch hook loop buys nothing (YAGNI).

**Suffix-match import resolution.** Absolute/dotted
imports resolve by matching the module path as a suffix
of candidate file paths, not by per-repo module-root
configuration. This supports monorepos transparently;
genuinely ambiguous matches are dropped rather than
guessed.

**Standalone comments as their own `COMMENT` kind.** A comment
that isn't a symbol's documentation — a banner, licence header,
section note, or module docstring — is indexed as its own chunk
rather than dropped, so its text is searchable even before a
reader knows the symbol names. It gets a dedicated kind, not
`DOC_SECTION` (which is structured prose that participates in doc
edges), ranked at 0.5: genuine natural-language signal, but below
structured prose because a comment is rarely the thing you
navigate *to*. Capture is query-driven — a per-grammar `@comment`
capture, like every other construct — rather than a bespoke engine
tree-walk, so one extraction model covers every language. The cost
is that tree-sitter cannot see blank lines, so grouping comment
runs into blocks is one small post-pass in the engine.

### Storage

**DuckDB over SQLite.** Three channels have to run over one
store. SQLite gives BM25 through FTS5, but not the other two:
no array column to hold an embedding, so vectors would live as
blobs and cosine similarity would run in Python over rows
pulled out of the database, and no way to hand a polars frame
to a query without inserting it first. DuckDB has `FLOAT[]`
and `LIST_COSINE_SIMILARITY`, so semantic scoring is a SQL
expression, and it reads registered frames zero-copy, which is
what lets retrieval and bulk upsert both be single set-based
statements (see
[Registered frames as query inputs](#registered-frames-as-query-inputs)).
The cost is a young file format with no in-place schema
migration — hence the wipe-and-rebuild policy in
[Versioning](#versioning) — and one writer at a time. Both are
acceptable for derived data with a single writing daemon.
Revisit if the index ever needs concurrent writers.

**One data file per project, named by a registry.** Deleting rows
never shrinks a DuckDB file, so reclaiming space means writing the
live data into a new file and using that one instead. DuckDB
reuses the freed space for later writes but returns none to the
operating system: a file of 60,000 chunks at about 22 MB was
still 16.0 MB after 90% of them were deleted, and a copy wrote it
as 1.8 MB. Renaming a
compacted copy over the open index loses writes after a crash: a
connection adopted after the rename goes on writing its WAL under
the copy's old name, which the next open does not look for, and a
crash can replay the old WAL's committed inserts onto the compacted
file (2,000 rows for 1,000 primary keys in the reproduction). Switching which file a
registry names never renames an open file.

One data file named by the registry would get the same switch, but
search is slower in it. On a 13-repo, 304,278-chunk index
compacted into one file, the database part of a one-repo search
took 2.74 s against 1.26 s on the repo's own file, and a 13-repo
search 2.85 s against 1.69 s with the 13 files read at once.
Restricting the scan to the repo's chunks first does not recover
it in one file (2.5 s): BM25 still scores over the whole full-text
index, and the vector scan still reads every row. Per-project
files change BM25's document-length statistics; on 1,000 eval
queries, lexical Hit@1, Hit@10 and MRR each moved by about a point,
in both directions.

The files of a cross-repo read are read at once, because a
statement costs 10–30 ms on even a tiny file: in turn, 13 files
took 3.55 s. They cannot be read as one query, because DuckDB's
`match_bm25` is a macro whose body names its tables without a
database, so it reads whichever file the cursor has `USE`d; called
by another file's name, it silently reads the wrong tables. A file
per checkout, rather than per project, would duplicate every chunk
a worktree shares with its main checkout, and a name derived from
the project would be wanted by the old file and the new at once
during a switch. Revisit if DuckDB gains a way to vacuum a file in
place: one file would then need neither the registry nor the
switch.

**One store; scope is a value; a session holds its repo.**
Callers name what they read by value: a `Repo` from the registry,
`repo.at(sha)` for one snapshot, a list of `SnapshotRef`s for a
search. The store routes each to its file. A `WriteSession` is bound
to one file and one repo: a DuckDB transaction writes one file, and a
session that took a repo with every write could file rows under a
repo of another project.

Two other shapes were considered and rejected. A store, or a DuckDB
instance, per repo or per project breaks four things: the memory and
thread caps cover the daemon only when one instance holds every file,
compaction's `COPY FROM DATABASE` needs the old and new file in the
same instance, DuckDB attaches a file once per process, and a
cross-repo search reads every file from one pool of threads. A handle
object per repo is scoped to the wrong thing: the file, the
transaction, blob dedup and compaction belong to the project, so the
handle's `compact` would rewrite its sibling checkouts too, and search
would still need `SnapshotRef`s, giving a snapshot two names. Revisit
if a second storage backend arrives: a port at the store's boundary
would then pay for itself.

**No foreign keys.** The data tables cannot reference `repos`:
they live in other files. `repos.project_id` could reference
`projects`, but DuckDB rejects an `UPDATE` of a referenced row
whose indexed column changes, and switching a project to a new file
updates `projects.file`; it also rejects `ON DELETE CASCADE`
outright. The mappings are inexpensive to keep correct in code: the
store inserts a project in the registry transaction that registers
its first checkout, and deletes it in the one that forgets its last.

**`build_index` canonicalises paths; store lookups do not.**
`find_repo` and `repo` match a path exactly as it
was stored. That is what lets a deleted checkout still be
found: `forget --stale` notices one *because* canonicalising
its path now fails, and then removes the row using the path
recorded when it was registered. Canonicalising inside the
lookups would leave a vanished checkout's rows in the database
with no way left to name them.

**Embedding completeness is counted.** Whether a snapshot is fully
embedded is derived, by counting the chunks it reaches and the
subset of those carrying an `embedding`. The counts are grouped, so
one pass over the embedding column answers for every snapshot in a
repo, and the cost follows the size of the column rather than the
number of snapshots asked about.

A denormalised `embedded` column on `indexed_snapshots` would make
it a lookup. It loses on correctness, not speed: every write path
that sets or clears an embedding would have to maintain it, and
"flag says embedded, column is NULL" becomes a state the schema
admits. Revisit if one grouped pass grows to dominate a status
request.

**A fixed DuckDB memory limit, not DuckDB's default.** DuckDB's
default limit is 80% of RAM, and it treats that limit as a cache:
it keeps every block it reads and does not give memory back. The
exact cosine scan and compaction's `COPY FROM DATABASE` read a
whole file, so on a 10 GiB index a compacting `gc` reached a
5.2 GiB footprint and still held 4.6 GiB afterwards. The store's
one DuckDB instance therefore opens through `Store._connect`
with `duckdb_memory_limit` (1 GiB by default) and
`duckdb_threads`, and every index file is attached to it, so the
limit covers them all. Under 1 GiB the same `gc`
peaked at 1.2 GiB and ran faster (34 s against 43 s), and search
and `gc` began failing only between 256 MiB and 128 MiB. Past the
limit DuckDB spills to `<index>.tmp`, or fails that one query with
`OutOfMemoryException`. The daemon and the CLI reply to that with
the `out_of_memory` error code, and the CLI and pi add how to raise
`duckdb_memory_limit`; a build or embed that fails this way is
logged. The limit covers DuckDB's buffer manager only: query
results are allocated outside it, and ART index memory (every
table has a primary key) counts against it but cannot be evicted,
about 100 MiB at that size. Threads default to half the usable
cores, capped at one per 128 MiB of the limit, just above DuckDB's
documented minimum per thread; the thread count barely moved the
footprint, and half the cores was faster than either a quarter or
all of them. DuckDB also starts an asynchronous I/O pool of
four threads per core, at most 256. rbtr leaves it at its default:
capping it at `duckdb_threads` changed neither the test suite's
time nor memory, only the number of idle threads (49 to 10).
Revisit if rbtr reads files over a network.

### Search

**Exact cosine over approximate NN.** Exact recall
outweighs marginal latency. Revisit when query latency is
measured as a bottleneck. DuckDB's `NEAREST` join returns
the same top 50 more slowly: on a 23,542-chunk file the current
query took 0.68 s, an exact `NEAREST` join 1.50 s and `APPROX
NEAREST` 1.30 s. Revisit if DuckDB gains a vector index the join
uses.

**Client-supplied query expansion over static rules.**
Query expansion (synonym keywords and variant phrasings)
is supplied by the caller rather than a built-in model.
In pi sessions the LLM generates expansion inline; in
eval a dedicated stage pre-generates it via an LLM API.
This stays language-agnostic (no hand-curated synonym
tables) and avoids the latency and GPU memory cost of
a local expansion model.

**Cross-encoder reranking after fusion.** Fusion is
fast but shallow — it combines independent channel
scores without seeing query and document together. A
cross-encoder evaluates each `(query, chunk)` pair
with full cross-attention, catching relevance that
channel scores miss. It is the largest single ranking
improvement measured, in both Hit@1 and MRR, and costs about
two seconds at p50. Current figures are in
[`data/BENCHMARKS.md`](../rbtr-eval/data/BENCHMARKS.md),
regenerated by each eval run — not repeated here, where they
would go stale silently.

### Daemon and interface

The [daemon design](#daemon) explains the separation of client service,
background policy and storage, the shared resource constraints, and why
running work determines resource lifetime. [Index files](#index-files)
covers the separate database ownership rules.

**ZMQ for the daemon.** The transport needs
request/response for queries and fan-out for build progress,
over a local socket, with clients that come and go. HTTP gives
the first and needs polling, SSE or websockets bolted on for
the second, plus a port to allocate and defend. ZMQ has both
patterns natively over IPC — ROUTER for queries, PUB for
progress — and its SUB sockets reconnect on their own, which
is what lets a pi session survive a daemon restart without
re-discovering an endpoint (see
[Daemon protocol](#daemon-protocol)). The cost is a
hand-rolled message contract; `messages.py` and
`just schema-check` exist to keep it honest.

**Watched refs over hardcoded HEAD tracking.** The daemon
derives builds from a durable `watched_refs` table rather
than a hardcoded HEAD poll: this keeps the “derive work from
DB + git” design (no in-memory queue), survives restart, and
lets review run against arbitrary refs, with HEAD as the
default entry. GC follows suit — `rbtr gc` defaults to
keeping HEAD, local branches/tags, and the watch set, so it
never discards anything reachable from a branch
(`--keep watched-only` opts into reclaiming unwatched branches).
See [Watched refs](#watched-refs).

**A request across every repo is a kind of its own.** `search_all`,
`status_all`, `gc_all`, `unwatch_stale_all` and `forget_stale` carry
nothing of one repo. A `scope` field on the single-repo requests
does not work: it needs a path the every-repo handlers never read,
so `--scope all` would fail outside a git repository, and it lets a
`ref` or a single-repo gc mode ride along unread. The CLI
keeps its `--scope` flag and pi its `scope` argument; each picks the
request kind.

**pydantic-settings for CLI + config.** Every setting has to
be reachable three ways — `config.toml`, a `RBTR_` environment
variable, and a command-line flag — with a documented
precedence between them. Written by hand that is three parsers
and a merge; here the `Config` model *is* all three, so a new
field gains a flag, an env var and a TOML key at once, and
`rbtr config` renders the merged result from the same object.
The cost is that the CLI's `--help` is generated from field
names and types, so flag naming is not free to diverge from
the config schema.

**structlog through one stdlib sink.** Application code uses
structlog, but the renderer is a `ProcessorFormatter` on a single
stdlib handler so third-party `logging` output is captured and
formatted identically. A pure `WriteLoggerFactory` was rejected:
third-party libraries still log via stdlib, so capturing them would
need a second handler writing the same file. See [Observability](#observability).

## Testing

For conventions (TDD workflow, fixture design, parametrise
patterns), see the `rbtr-testing` skill. This section
covers the test infrastructure.

**Isolation.** `[tool.pytest_env]` points `RBTR_DATA_DIR`,
`RBTR_CONFIG_DIR`, and `RBTR_LOG_DIR` at the system temp
directory, and the autouse `_test_dirs` fixture asserts that
before creating them — so a misconfigured run fails loudly
instead of writing to real data. `cache_dir` is deliberately not
redirected: it holds the shared model cache and is safe to
reuse. Tests run a tiny CPU embedding model (all-MiniLM-L6-v2,
~19 MiB) rather than the production GGUF; `stub_embedding_model`
is opt-in, for tests that drive the model from several threads.

**Fixture composition.** Three layers: root conftest
(isolation) → domain conftest per subdirectory (data
fixtures — `index/conftest.py` has git repos, ranking
dataset, named chunks; `daemon/conftest.py` has seeded
stores and running servers) → case files (scenarios via
pytest-cases, e.g. `cases_edges.py`, `cases_fuse.py`).

**Real infrastructure, not mocks.** Index tests create
real `pygit2` repositories with deliberate data shapes.
Daemon tests spin up real `Daemon` instances on
IPC sockets in threads (`serving`, `running_daemon` in the
[daemon fixtures](src/rbtr/tests/daemon/conftest.py)). These tests exercise
observable replies, stored effects and notification delivery across the
ZMQ/DuckDB boundary.

Import contracts verify dependency direction and component independence.
Runtime tests verify the facts those checks cannot:
[contention tests](src/rbtr/tests/daemon/test_embed_search_contention.py)
observe model-call overlap and stored writes;
[lifecycle tests](src/rbtr/tests/daemon/test_server.py) cover failure and stop;
[standalone indexing](src/rbtr/tests/index/test_indexer.py) leaves its borrowed
Index usable after joining.
[Progress regressions](src/rbtr/tests/daemon/test_progress.py) hold real
notification delivery while querying actual status, establishing state
correctness separately from publication timing.

**Test location.** In-package: `src/rbtr/tests/`. The
pi-rbtr tests live at `packages/pi-rbtr/tests/` with
a `fake-daemon.ts` test double and case files
(`*.cases.ts`) separating data from assertions.
