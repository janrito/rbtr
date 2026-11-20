# rbtr-lang-rst

reStructuredText support for [rbtr]. A **default** plugin — installed
with rbtr itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

A tree-sitter query plugin (`rst.scm`). RST has no fixed heading levels, so
a section's level is the order in which its adornment style first appears,
as docutils reads it. Cross-references are extracted as imports.

- **Sections** — headings (underline and overline adornments) → doc sections
  holding their text up to the next heading, with `scope` the `::` chain of
  enclosing sections. A document with no heading at all — a README, a
  changelog — gives each top-level block a fragment of its own.
- **Labels** — `.. _intro:` and other link targets → a doc section named
  `intro`.
- **References** — the `:func:` / `:class:` / `:meth:` / `:mod:` / `:doc:` /
  `:ref:` roles, `` `text <target>`_ `` references, and `.. toctree::` entries
  → imports. A role says where its name lives: `:func:` and its kin look
  in Python, `:ref:` in rST, where it reaches a label or a title; a
  `#fragment` reaches the section whose title it is, as docutils writes it.
  External URLs are skipped.

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
