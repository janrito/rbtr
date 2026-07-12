---
name: rbtr-languages
description: >-
  Conventions for authoring and extending rbtr language plugins — the
  tree-sitter queries, chunkers, capture conventions, and extraction-engine
  layers. Use when writing or modifying a file under rbtr/languages/, editing
  a `_QUERY`, a `LanguageRegistration`, an `import_extractor`, or a chunker;
  when adding support for a new construct or language; or when reviewing
  extraction behaviour. Also trigger on tree-sitter queries, `extract_symbols`,
  `tags.scm`, or `@definition`/`@function`/`@_scope` captures.
user-invocable: false
---

# rbtr language plugins

How rbtr turns source into chunks, and the conventions for changing it.
Pairs with the **rbtr-testing** skill (the sample/snapshot harness) and the
`rbtr-languages` tags reference in `references/tags-scm-reference.md`.

## Comprehensiveness principle

**rbtr aims to index everything an engineer might search for or navigate to.**
If a developer could plausibly grep a name, or want "jump to definition" on
it, it should be a chunk. Bias toward **over-capture**, not under-capture.

- "Don't capture X" must be justified by **"X is not a definition / has no
  searchable identity"** — never by "X is rare/niche". Rarity is not a reason
  to skip.
- Definition-shaped constructs are in scope even when they need bespoke work:
  C function **prototypes** (the API surface in headers), `namespace`/`mod`
  as their own symbols, Bash `alias`es (a written command definition, like
  `source`), SCSS `@mixin`/`@function`/`$variables`. Capture them.
- **Written constructs only — the written code is the unit, not its runtime
  effects.** rbtr extracts what is in the source, not what a macro/
  metaprogramming call *generates*. Ruby `attr_accessor :name` is written and
  findable as content (it sits in the class chunk), so full-text search
  already locates it — but do **not** synthesise `name`/`name=` method chunks
  for the accessors it generates at runtime; those are not written code.
  (Capturing the written `:name` token as a field is a possible future
  nicety, not a default.)
- Legitimate skips: truly anonymous nodes with no name a human would search
  (a bare `@font-face`, an unnamed `@media` block — captured only as an
  anonymous section), pure metadata (`COMMENT ON`), usage/references (which
  belong to edges, not symbol chunks), and **runtime-generated symbols**
  (per above).
- When unsure, capture it. A spurious chunk is cheaper than an invisible
  symbol.

### Contained definitions & the data-scope exception

A definition that lives *inside* a larger definition we already chunk is
represented **by that parent chunk** — it is full-text-findable there and does
not get its own chunk. Statements inside a function, and **data members of a
class** (class attributes, enum members, dataclass fields), are represented by
the enclosing function/class chunk. **Methods are the exception**: substantial,
separately-navigable units, so they *do* get their own chunk.

**The data-scope exception (language philosophy).** *Variable* capture targets
the language's idiomatic **top-level data scope**, not nested members — and
what counts as "top-level" differs by language:

- Languages with a module / file / package data scope (Python module,
  Go/Rust/C file or package, JS/TS module, Bash script) capture variables
  **there**; class/struct data members stay contained in their parent chunk.
- **Class-only languages with no top-level data scope (Java): the class is the
  only namespace, so fields *are* the top-level data definitions and are
  captured** (scoped to their class). Not capturing them would leave Java with
  zero variable chunks.

This is **not** an inconsistency between languages — it is the *same* rule
(capture top-level data at the language's idiomatic scope) applied to
differing language philosophies. It keeps capture comprehensive without
exploding every class into per-field chunks, while never leaving a language's
data definitions invisible.

## Architecture

A plugin is a file under `rbtr/languages/` returning `LanguageRegistration`s.
The orchestrator's primary path is one of three:

1. **Chunker** — `reg.chunker` set: prose (markdown, rst) and SFCs' markup
   (svelte, vue). The chunker owns extraction. It takes an optional `ranges`
   and must set `parser.included_ranges` when given it, so it can also serve
   as an injection target for an embedded block (see below).
2. **Query** — `reg.grammar_module` + `reg.query`: code, plus config/data
   whose scope a query can express (python, rust, …, json, css, html, toml,
   yaml, hcl, query). Goes through `extract_symbols`.
   HTML captures its semantic elements (`head`, `body`, sectioning content,
   landmarks) as doc sections, named by `id` else tag via a `name_extractor`.
   The `query` plugin indexes `.scm` files themselves: each top-level pattern
   is a `@doc_section`, named by its own outer capture else anonymous.
3. **Plaintext fallback** — no grammar/detection: fixed-size raw chunks.

`extract_symbols` is the query engine: *parse → run query → captures →
`Chunk`s*. It is generic; per-language behaviour is data on the registration.

**Injection (embedded languages)** is an orthogonal capability that runs *in
addition* to the primary path. To extract code embedded in a host file (an
SFC's `<script>`/`<style>`, a Markdown fenced block, an HTML inline
`<script>`/`<style>`), set `reg.injection_query`: a tree-sitter query over the
host grammar that captures each embedded block as `@injection.content` and
names its target language one of two ways:

- **Static** — `(#set! injection.language "<id>")`, for a closed set (SFC/HTML
  `<script>`→js/ts, `<style>`→css), with an optional
  `(#set! injection.priority "<n>")` so a `lang`-tagged rule beats a bare one.
- **Dynamic** — capture the language name as `@injection.language` (a Markdown
  fence's info string). The engine resolves the captured text via the
  registry's own id and extension maps (`python`/`py` both reach python); an
  unknown hint is left unparsed. No per-language mapping table.

The engine delegates each block's range to the target's *full* primary
extraction (`extract_primary` — chunker or query, so a chunker target like
yaml/toml works, not just query targets) and *recurses* into the target's own
injection (an HTML block containing an inline `<script>` yields its js), all
at absolute line numbers. Every file also gets a host-language chunk (a
content-less presence chunk if it would produce none), so dedup works. See
ARCHITECTURE “Dispatch chain” for the mechanism and rationale.

### Where queries live

Every query — `reg.query`, `reg.injection_query`, and any query a chunker
compiles — is a `.scm` file in the plugin's own package
(`languages/<lang>/<name>.scm`), loaded at import via
`load_query(__package__, "<name>")`. Never inline a query as a Python string
literal (the house rule against embedding a foreign language). The `uv`
build backend ships `.scm` as package data with no extra config.

Compose in Python when a query is built from parts — the query language has
no `#include`: js/ts concatenate shared fragment files with `+`
(`load_query(pkg, "javascript") + load_query(pkg, "shared") + …`); SQL groups
its DDL verbs into `[...]` alternations within one `sql.scm`. Prefer these to
generating query text from Python data. Editing a `.scm` is an extraction
change — the `language_plugin_version` bump rule (below) applies.

## Capture conventions

The query's capture names drive the chunk kind (see `_CAPTURE_KIND` in
`index/treesitter.py`):

- `@function` / `@_fn_name` — functions
- `@class` / `@_cls_name` — classes, structs, enums, traits, types
- `@method` / `@_method_name` — methods (a `@function` whose nearest scope is
  class-like is also promoted to a method)
- `@variable` / `@_var_name` — module/top-level variables, constants, fields
- `@import` — import statements (metadata via `import_extractor`)
- `@_scope` — optional: a node whose **text** becomes the symbol's innermost
  scope segment, for scopes lexical nesting can't reach (e.g. a Go method's
  receiver type). Strictly additive: absent → no effect.
- `@_docstring` — interior first-statement docstring (Python)
- `@doc_section` / `@_section_name` — chunker/data section units

Capture names starting with `_` are read but never become chunks.

The display name comes from the paired `@_*_name` capture via the default
`resolve_name`. When a query cannot express the name, set a `name_extractor`
on the registration (signature `(capture_name, node, captures) -> str`) as a
**last resort** — it delegates to `resolve_name` for the cases it does not
special-case, mirroring how `import_extractor` delegates to
`build_import_from_captures`. Bash strips the `=` the grammar fuses onto an
alias; HTML names an element by its `id`, else its tag.

The scope address comes from tree ancestry (`scope_types`, below) plus the
`scope_extractor` — the scope twin of `name_extractor`, defaulting to
`resolve_scope` (which contributes the `@_scope` capture). Set a custom one
(signature `(capture_name, node, captures) -> list[str]`, outermost-first)
as a **last resort**, for a hierarchy neither ancestry nor `@_scope` can
reach; its segments are appended to the ancestry scope. Two real cases:

- CSS/SCSS/Less nested rules — walk the ancestor `rule_set` selectors so
  `.card { .title { … } }` scopes `.title` under `.card`:

  ```python
  def css_nesting_scope(capture_name, node, captures):
      segments: list[str] = []
      for rule_set in enclosing_nodes_of_type(node, frozenset({"rule_set"})):
          for child in rule_set.children:
              if child.type == "selectors" and child.text:
                  segments.append(child.text.decode().strip())
                  break
      return segments
  ```

- TOML dotted tables — the hierarchy is a dotted-key *string*, not tree
  ancestry, so a `name_extractor` returns the last segment and a
  `scope_extractor` the preceding ones (`[tool.ruff]` → name `ruff`, scope
  `tool`).

## Scope, promotion, docs (engine layers — already generic)

Set on `LanguageRegistration`; `extract_symbols` applies them to every
captured node, so they work for any language:

- `scope_types` — node types that open a naming scope; composed into the
  `::` address. Include nesting containers (classes, namespaces, modules,
  functions where nested defs matter).
- `class_scope_types` — the subset that is class-like; a function directly
  inside one is promoted to a method. Defaults to `scope_types`.
- `doc_comment_node_types` — comment node types attached as leading docs
  (walked backwards to a blank line). Empty → chunk is exactly the node span.
- Non-lexical scope comes from `@_scope` (above) or, when even that can't
  reach it, a `scope_extractor` (above), not these.

## Authoring or extending a plugin

1. Read the grammar: its `queries/tags.scm` (the authors' definition list —
   see the tags reference) **and** the real node structure (parse a snippet,
   print the tree). Never guess node types.
2. Edit the language's `.scm` query file (or chunker) — queries live in
   `languages/<lang>/*.scm`, loaded via `load_query`, never inline (see
   *Where queries live*). Verify against a parsed snippet.
3. Add the construct to that language's sample in `tests/languages/samples/`
   and regenerate the snapshot with `just snapshots`; review the diff.
4. Bump `language_plugin_version` — any extraction change triggers
   re-extraction of stored blobs.
5. `just check`. (Samples are exempt from lint/type/format — see below.)

## Gotchas (all learned the hard way; verified)

- **Require a body on type captures.** `(struct_specifier name: …)` matches
  *references* too (e.g. inside `typedef struct G G;`, or a parameter type),
  producing spurious class chunks. Require `body:` so only definitions match
  (C/C++ struct/enum/class).
- **Non-lexical scope → `@_scope`.** Scope is otherwise lexical-ancestry only
  (`_enclosing_scopes` walks parents). A Go method's receiver is a *child*, so
  capture it as `@_scope`.
- **Determinism in test helpers.** `next(iter(reg.extensions))` over a
  `frozenset` varies with `PYTHONHASHSEED`; the chunk id hashes `file_path`,
  so derive paths deterministically (`sorted(...)`, or pass an explicit path).
- **Imports are always bespoke.** `tags.scm` has no imports; the edge system
  depends on them. Keep per-language `import_extractor`s.
- **SQL / multi-dialect: don't gate on parse-clean.** One generic SQL grammar
  serves all `.sql`. Tree-sitter error recovery is *local* — a dialect
  construct it can't parse breaks only its own subtree; surrounding
  statements still extract. Treat `has_error` as informational, not a gate.
- **Known-unsupported constructs → strict xfail.** Record a construct that
  *should* extract but can't (grammar/plugin limit) as an `xfail(strict=True)`
  case, so closing the gap flips the test and prompts an update. Reserve for
  symbol-shaped gaps `(kind, name, scope)`; import-identity gaps don't fit.

## `tags.scm`: a reference, not a runtime source

Every code grammar ships `queries/tags.scm` — the authors' standard
definition/reference query (`@definition.*` / `@name` / `@reference.*`). It is
itself a tree-sitter query, so it is **inspiration for ours, not a drop-in**:

- **Mine it**: take good patterns, modify weak ones, ignore wrong ones.
  Verify every pattern against a real parse — quality is uneven (e.g. C's
  `union` pattern misses standalone `union U {}`).
- It captures **no imports and few variables**, omits some constructs
  (TS `enum`/type alias, Java `enum`/`record`), some grammars inherit others
  (ts ← js), and only the 9 code grammars ship one.
- Running it live would couple extraction to upstream drift. We don't.

Distinct from **rbtr's own** `.scm` files (see *Where queries live*): we load
those at runtime (`reg.query` / `injection_query`) as the source of truth,
whereas `tags.scm` we only mine for ideas. And the `query` plugin now
*indexes* `.scm` files found in a repo — including third-party `tags.scm` /
`highlights.scm` / `injections.scm` — as content, orthogonal to whether we
run them.

Curated per-language verdicts (take / modify / ignore) and the
`@definition.* → ChunkKind` mapping live in
`references/tags-scm-reference.md`.

## Testing

Each language has a sample mini-project under `tests/languages/samples/<lang>/`
(one or more files), golden-snapshotted (full `model_dump_json`),
coverage-checked, and parse-clean-checked. See **rbtr-testing**. Samples are
exempt from the repo's linters/type-checker (they're fixtures) — validated
only by their own tests.
