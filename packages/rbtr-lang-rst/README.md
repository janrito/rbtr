# rbtr-lang-rst

reStructuredText support for [rbtr](../rbtr). A **default** plugin — installed
with rbtr itself (`pip install rbtr`).

## What it ingests

A custom chunker: it reconstructs the section hierarchy from the *order of
adornment characters* (RST has no fixed heading levels), each section a chunk
scoped by its ancestors. Cross-references are extracted as imports.

- **Sections** — headings (underline and overline adornments) → doc sections,
  with `scope` the `::` chain of enclosing sections.
- **References** — the `:func:` / `:class:` / `:meth:` / `:mod:` / `:doc:`
  roles, `` `text <target>`_ `` references, and `.. toctree::` entries →
  imports. External URLs and same-file anchors are skipped.

## Chunks produced

```rst
Title              doc_section "Title"
=====
Section            doc_section "Section", scope "Title"
-------
See :func:`helper` and :doc:`api/index`.
                   import {names: helper}; import {module: api/index}
```

## Embedded / injected chunks

None. RST does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-rst` grammar. No runtime dependency on other language
plugins; the test suite dev-depends on `rbtr-lang-python` to extract the
sample's `helpers.py` and snapshot the `.rst → .py` reference edges.
