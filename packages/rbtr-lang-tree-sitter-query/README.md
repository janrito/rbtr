# rbtr-lang-tree-sitter-query

tree-sitter query (`.scm`) support for [rbtr](../rbtr). Optional plugin —
install with `pip install rbtr[tree_sitter_query]`.

## What it ingests

`.scm` files — the tree-sitter query language rbtr's own plugins are written
in, and which third-party grammars ship too (`tags.scm`, `highlights.scm`,
`injections.scm`), so cloned repos are covered. Each top-level pattern becomes
a **doc-section** chunk, named by its own outer capture where the author gave
one. Patterns with no outer label are anonymous sections whose nested captures
stay full-text-searchable.

## Chunks produced

```scm
; doc_section "function"
(function_definition name: (identifier) @_fn_name) @function

; doc_section "keyword"
["if" "else"] @keyword
```

## Embedded / injected chunks

None. Query files do not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-query` grammar (`.scm`). No dependency on other language
plugins.
