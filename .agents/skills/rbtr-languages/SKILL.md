---
name: rbtr-languages
description: >-
  Conventions for authoring and extending rbtr language plugins — the
  tree-sitter queries, chunkers, capture conventions, and extraction-engine
  layers. Use when writing or modifying a language plugin (an `rbtr-lang-*`
  package or core's bundled `rbtr/languages/`), editing a `.scm` query, a
  `LanguageRegistration`, an import's captures and directives, or a chunker;
  when adding support for a new construct or language; or when reviewing
  extraction behaviour. Also trigger on tree-sitter queries, `query_captures`,
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
- **Comments are content, not decoration.** Top-level comments — banners,
  licence headers, section notes — and module docstrings are captured as
  `COMMENT` chunks so their text is searchable, not dropped; a comment above
  a definition folds into it as documentation. See *Comment handling*.
- Anonymous constructs remain searchable as unnamed chunks: a
  `tree_sitter_query` pattern with no outer label is an anonymous section.
  Pure metadata (`COMMENT ON`), usage/references and runtime-generated
  symbols have no definition chunk of their own; references belong to edges.
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

Each plugin is an `rbtr-lang-*` package exposing module-level
`LanguageRegistration` values through the `rbtr.languages` entry-point
group. Default and optional languages use the same registration path.
The [plugin guide][plugin-guide] describes the public API; the
[extraction architecture][extraction-architecture] explains how core
uses those declarations.

Prefer `QueryExtraction` with a grammar and query. Use `@reg.chunker`
when queries and resolver overrides cannot describe the units. A chunker
that receives `ranges` must set `parser.included_ranges`, so an embedded
block is read within its original file. Plugins declare language facts
and return tree-derived values or chunks; batch frame processing belongs
to core.

For embedded languages, an `injection_query` captures each block as
`@injection.content`. Name its target with static
`(#set! injection.language "<id>")`, or capture a hint as
`@injection.language` for registry resolution (`python` and `py` both
reach Python). Use `injection.priority` when a specific rule must beat a
fallback. Keep this mapping in queries and the registry rather than
adding per-language tables to core. The [dispatch description][dispatch]
covers target extraction, nested blocks and host-language presence.

[plugin-guide]: ../../../packages/rbtr/README.md#writing-a-language-plugin
[extraction-architecture]: ../../../packages/rbtr/ARCHITECTURE.md#extraction
[dispatch]: ../../../packages/rbtr/ARCHITECTURE.md#dispatch-chain

### Where queries live

Every query — `reg.extraction.query`, `reg.injection_query`, and any query a chunker
compiles — is a `.scm` file co-located in the plugin package
(`rbtr_lang_<lang>/<name>.scm`), loaded at import via `load_query` (import
it: `from rbtr.languages.registration import load_query`; call it:
`load_query(__package__, "<name>")`). Call it directly in the `extraction` or
`injection_query` field — never hoist it into a module-level `_QUERY` constant;
`load_query` is cached on `(package, name)`, so a query shared by two
registrations (svelte and vue's SFC injection query) is read once. Never inline
a query as a Python
string literal (the house rule against embedding a foreign language). The
`uv` build backend ships `.scm` as package data with no extra config.

Compose in Python when a query is built from parts — the query language has
no `#include`: js/ts concatenate shared fragment files with `+`
(`load_query(pkg, "javascript") + load_query(pkg, "shared") + …`); SQL groups
its DDL verbs into `[...]` alternations within one `sql.scm`. Prefer these to
generating query text from Python data. Editing a `.scm` is an extraction
change — the `extraction_serial` bump rule (below) applies.

## Capture conventions

The query's capture names drive the chunk kind (see `CAPTURE_KINDS` in
`languages/treesitter.py`):

- `@function` / `@_fn_name` — functions
- `@class` / `@_cls_name` — classes, structs, enums, traits, types, and
  named collections of declarations (CSS rule sets, `@media`, `@keyframes`)
- `@method` / `@_method_name` — methods (a `@function` whose nearest scope is
  class-like is also promoted to a method)
- `@variable` / `@_var_name` — module/top-level variables, constants, fields
- `@import` — import statements (module, names and bindings via `@_import_module`,
  `@_import_names`, `@_bind_name` / `@_bind_local`, `#strip!` and `#set!`)
- `@_scope` — optional: a node whose **text** becomes the symbol's innermost
  scope segment, for scopes lexical nesting can't reach (e.g. a Go method's
  receiver type). Strictly additive: absent → no effect.
- `@_docstring` — interior first-statement docstring (Python)
- `@doc_section` / `@_section_name` — chunker/data section units
- `@config_key` / `@_section_name` — config/data keys (JSON object keys, TOML
  tables, YAML mapping keys, HCL blocks, CSS `@charset`), reusing the
  `@_section_name` name capture
- `@comment` — top-level comments and module docstrings; grouped into blocks
  and either folded into the definition below or emitted as standalone
  `COMMENT` chunks (see *Comment handling*)
- `@_call` / `@_base` — a call's callee and a class's base, recorded on
  the innermost chunk around them for call and inherits edges; mine
  `@reference.call` and class bases from `tags.scm`
- `@_self` / `@_super` — the names that reach the enclosing class's
  instance and bases (`self`, `this`, `super`), marked with a predicate

Capture names starting with `_` are read but never become chunks.

The display name comes from the paired `@_*_name` capture via the built-in
name resolver. When a query cannot express the name, attach a
`name_extractor` — `@reg.name_extractor` for a single-use local override,
or `reg.name_extractor(fn)` for a shared/imported one — as a **last
resort**. It is wrap-style (pydantic `WrapValidator` shape): signature
`(resolver, capture_name, node, captures) -> str`, where `resolver` is the
built-in resolver handed in; call it to delegate the cases you don't
special-case. Bash strips the `=` the grammar fuses onto an alias; HTML names an
element by its `id`, else its tag.

The scope address comes from tree ancestry (`scope_types`, below) plus the
`scope_extractor` — the scope twin of `name_extractor`, whose built-in
resolver contributes the `@_scope` capture. Attach a custom
one the same way (wrap-style signature
`(resolver, capture_name, node, captures) -> list[str]`, outermost-first)
as a **last resort**, for a hierarchy neither ancestry nor `@_scope` can
reach; its segments are appended to the ancestry scope. Two real cases:

- CSS/SCSS/Less nested rules — walk the ancestor `rule_set` selectors so
  `.card { .title { … } }` scopes `.title` under `.card`:

  ```python
  def css_nesting_scope(_resolver, capture_name, node, captures):
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

## Comment handling

Comments are captured, not configured. Give your grammar's comment node(s) a
**root-scoped `@comment`** capture so only top-level comments match — a comment
inside a function body stays part of that body's chunk:

- most grammars: `(translation_unit (comment) @comment)`, `(program (comment)
  @comment)`, `(source_file (comment) @comment)`, `(stylesheet (comment)
  @comment)` — use the grammar's actual root and comment node type;
- multiple comment node types go in an alternation: Rust/Java `[(line_comment)
  (block_comment)]`, SQL `[(comment) (marginalia)]`, SCSS/Less `[(comment)
  (js_comment)]`;
- Python also captures its module docstring:
  `(module (expression_statement (string) @comment))`.

Core groups and routes these comments as described in the
[extraction architecture][extraction-architecture]. Declare the node types
rather than implementing comment folding in the plugin.

## Fragments: code no pattern names

Searchable code left outside query captures becomes unnamed `FRAGMENT`
chunks, each recording its named syntax-node types (`role.meta.node_types`).
Blank lines and punctuation alone need no chunk. `QueryExtraction.fragments`
controls this behaviour and is enabled by default. See
[Git and extraction][fragment-rationale] for the grouping rationale.

[fragment-rationale]: ../../../packages/rbtr/ARCHITECTURE.md#git-and-extraction

A fragment is a gap report, not a target. Read the gaps two ways:

- **A sample's snapshot.** Each `test_samples.py` snapshots the symbols a
  build of the sample holds, fragments among them, and the lines no chunk
  spans. A fragment holding a definition is a missing capture: a C
  prototype inside an `#ifndef` guard. A fragment holding part of one is a
  capture of the wrong node: Python's decorators sat in fragments until its
  query captured `decorated_definition`.
- **rbtr-eval's `INDEX.md`.** Its fragments table counts, per language, the
  lines left to fragments and the node types holding the most of them, over
  the eval corpus.

To turn a frequent node type into a capture: put the shape in the language's
sample, add the pattern to its `.scm`, regenerate the snapshots (the fragment
becomes a named chunk), and bump `extraction_serial`. A node type with no
searchable identity (a Go `package` clause, `#pragma once`, a parse error)
stays a fragment: that is what fragments are for.

## Shaping captures, and grammar gaps

Syntax a grammar leaves inside one leaf (backticks, `<target>`, a `~`, a label's `_` and
`:`) is stripped in the query with `#strip!`, which the engine applies; never in Python.
`#set!` comes before a pattern's predicates (see Gotchas).

Two engine rules stand in for nodes a grammar does not give. Use one only
where the grammar lacks the node; where it gives the node, an ordinary
pattern is the answer. Say why in a comment beside the pattern.

- **One import per line.** A quantified `@_import_module` names one module
  per line its nodes sit on. An rST `toctree`'s content is word tokens,
  with no node per entry.
- **Flat sections.** A doc section whose pattern captures its adornments
  as `@_level` is levelled by the order each style first appears, scoped
  under the nearest earlier section of a higher level, and holds its text
  up to the next one. tree-sitter-rst's `section` holds its title alone.

Each rule is deleted when its grammar gives the node. A new rule needs the
same case: a node the grammar does not give, and an exit.

## Names a reference uses

A link's fragment or a role reaches a section by its name. Capture
whatever defines such a name (an `id`, a label) as a section of that
name, with a query; declare `section_names=SectionNames.FOLDED` when the
language's renderers turn a heading into an anchor. Test it with links
written as the renderers themselves write them, in an imports sample. How
and why: the `rbtr` README ("How your language's references resolve") and
ARCHITECTURE ("Dependency graph").

## Scope & promotion (engine layers — already generic)

Set on the language's `QueryExtraction` (the `extraction` field);
the adapter (`query_captures`) reads them for every captured node:

- `scope_types` — node types that open a naming scope; composed into the
  `::` address. Include nesting containers (classes, namespaces, modules,
  functions where nested defs matter).
- `class_scope_types` — the subset that is class-like; a function directly
  inside one is promoted to a method. Defaults to `scope_types`.
- Non-lexical scope comes from `@_scope` (above) or, when even that can't
  reach it, a `scope_extractor` (above), not these.

## Authoring or extending a plugin

1. Read the grammar: its `queries/tags.scm` (the authors' definition list —
   see the tags reference) **and** the real node structure (parse a snippet,
   print the tree). Never guess node types.
2. Edit the language's `.scm` query file (or chunker) — queries live in
   the plugin package (`rbtr_lang_<lang>/*.scm`), loaded via `load_query`,
   never inline (see *Where queries live*). Verify against a parsed
   snippet.
3. Add the construct to that language's sample in the package's
   `tests/samples/` and regenerate the snapshot with `just snapshots`;
   review the diff.
4. Bump `extraction_serial` — any extraction change triggers
   re-extraction of stored blobs. It is independent of the package
   version: bump it during development (before any release) when output
   changes, and never bump it for a package-only release or pure move.
5. `just check`. (Samples are exempt from lint/type/format — see below.)

## Gotchas

- **Require a body on type captures.** `(struct_specifier name: …)` matches
  *references* too (e.g. inside `typedef struct G G;`, or a parameter type),
  producing spurious class chunks. Require `body:` so only definitions match
  (C/C++ struct/enum/class).
- **Take a span's last line from `last_line`.** Tree-sitter rows are
  0-based, so `end_point[0] + 1` is right for a node ending mid-line and
  one too many for a node that consumes its trailing newline and ends at
  column 0 of the next row. Import `last_line` from `rbtr.languages.chunks`
  and delete any local trailing-newline correction — a compensation left
  beside it decrements twice.
- **Non-lexical scope → `@_scope`.** Scope otherwise follows tree ancestry.
  A Go method's receiver is a child, so capture it as `@_scope`.
- **Deterministic test paths.** `next(iter(reg.extensions))` over a
  `frozenset` varies with `PYTHONHASHSEED`. Choose test paths
  deterministically with `sorted(...)`, or pass an explicit path.
- **Imports need their own patterns.** `tags.scm` has no imports. An
  import is stated by its query alone: captures for the module, names and
  bindings, `#strip!` to cut written text, and `#set!` for names, relative
  modules and language hints. There is no Python hook for imports.
- **SQL / multi-dialect: extract through parse errors.** One generic SQL
  grammar serves all `.sql`. Tree-sitter recovers locally from a dialect
  construct it cannot parse, so surrounding statements still extract.
  Treat `has_error` as informational rather than refusing the file.
- **Put `#set!` before a pattern's predicates.** In py-tree-sitter 0.26.1,
  putting it after a predicate can double-free that predicate and crash a
  later query. Write `(#set! …)` first, then `#eq?`, `#match?`, `#any-of?`
  or `#strip!`. Remove this restriction when the installed binding fixes
  its predicate reference handling.
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
those at runtime (`reg.extraction.query` / `reg.injection_query`) as the
source of truth, whereas `tags.scm` we only mine for ideas. And the
`tree_sitter_query` plugin now *indexes* `.scm` files found in a repo —
including third-party `tags.scm` / `highlights.scm` / `injections.scm` — as
content, orthogonal to whether we
run them.

Curated per-language verdicts (take / modify / ignore) and the
`@definition.* → ChunkKind` mapping live in
`references/tags-scm-reference.md`.

## Testing

How a plugin is tested (its sample, import scenarios, detection, and the
fixtures `rbtr.testing` provides) is described once, in the `rbtr` README's
"Testing the plugin" section (`packages/rbtr/README.md`). On top of it, in
this repository:

- Regenerate every plugin's snapshots with `just snapshots`, and read the
  diff before committing.
- A language's behaviour is tested in its plugin, never in core's tests;
  see **rbtr-testing**.
- Samples, including `samples/imports/`, are exempt from the repo's
  linters and type-checker; they are fixtures, validated by their own
  tests.
