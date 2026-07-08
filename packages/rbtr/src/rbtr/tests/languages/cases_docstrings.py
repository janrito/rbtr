"""Docstring-extraction test cases for all languages.

Each `@case` returns a 4-tuple `(lang, source, symbol_name,
snippet)` consumed by `test_docstrings.py`:

* `lang`        - language id registered by a plugin.
* `source`      - a small but realistic source fragment.
* `symbol_name` - name of the chunk whose `content` the test
  will inspect.  Cases keep names unique so disambiguation is
  trivial.
* `snippet`     - a substring the test will look for.  For
  `documented` cases the snippet must appear in `content`; for
  `undocumented` cases it must *not* appear.

Tags
----

Every case carries **two** tags.

Primary (drives the assertion direction):

* `documented`   - the chunk's `content` must contain `snippet`.
* `undocumented` - the chunk's `content` must *not* contain
  `snippet`.

Mechanism (partitions `documented` cases by the engine feature
responsible for extraction; lets two tests assert opposite
outcomes over disjoint case slices instead of branching inside
one test):

* `interior_doc` - the docstring is inside the symbol node,
  extracted via the plugin's `@_docstring` query capture.  Python
  is the only language using this mechanism.
* `exterior_doc` - the documentation is a leading sibling
  comment block, attached via the engine's sibling walk.  Every
  non-Python plugin uses this.

Secondary (classifies the scenario for debugging slices):

* `canonical`              - the language's idiomatic,
                              by-the-book documentation form.
* `edge_case`              - a valid but unusual doc form (raw
                              / byte strings, nested quotes,
                              multi-line block comments,
                              language-specific inner doc such
                              as Rust `//!`).
* `unconventional`         - not strictly canonical but clearly
                              intended as docs.  rbtr leans
                              toward extracting these.
* `boundary_attached`      - attachment-logic scenario whose
                              outcome is attachment.
* `no_docs`                - symbol has no documentation.
* `boundary_not_attached`  - attachment-logic scenario whose
                              outcome is no attachment.
* `invalid`                - text that looks like documentation
                              but is not recognised as such.

Policy being tested: "the default rbtr behaviour is that every
documented symbol exposes its documentation in the chunk
content".  Every language has at least one case in each primary
bucket and covers every secondary tag relevant to that language.

Source-template convention
--------------------------

Python source strings use single-quoted triple-string (`'''...'''`)
templates so their body can embed double-quoted triple-string
docstrings (`\"\"\"...\"\"\"`) verbatim.
"""

from __future__ import annotations

type DocstringCase = tuple[str, str, str, str]


# Known limitation: JavaScript/TypeScript plugins do not emit
# `@method` chunks, so method-level JSDoc lives inside the
# class chunk's bytes but is not tied to any `@_docstring`
# capture or leading-comment walk.  No case here.


#
# TypeScript reuses the JS grammar family with extra
# annotations; docstring semantics are identical so cases
# mirror JS while exercising annotation-bearing signatures.
# interface / type declarations are currently not captured by
# the TS plugin — tracked separately; not covered here.


#
# Rust uses `line_comment` for both `//` and `///` (the
# `outer_doc_comment_marker` distinction lives *inside* the
# line_comment node) and `block_comment` for `/* */` and
# `/** */`.  The plugin lists both so canonical `///` runs,
# unconventional `//` runs, block doc comments, and banner
# blocks all attach.


#
# Go convention (enforced by `gofmt` / `go doc`): every
# exported symbol begins its comment with the symbol's name.
# tree-sitter-go emits a single `comment` node type for both
# `//` and `/* */`; the plugin lists `{comment}` so both
# attach.


#
# Java uses `block_comment` for `/** */` Javadoc and
# `line_comment` for `//` — both listed in the plugin.
# Annotations (`@Deprecated` etc.) parse as part of the
# method's `modifiers` child, so the method's
# `prev_named_sibling` is the Javadoc directly — attachment
# works even when annotations sit between.


#
# Ruby convention: `#` comment runs above top-level `def`,
# `class`, and `module` declarations.  The tree-sitter-ruby
# grammar nests methods inside `body_statement` under their
# enclosing class, so comments inside a class body sit at a
# different level than the method node — attachment does not
# cross the `body_statement` boundary.  This mirrors the JS
# limitation noted in D14: method-level documentation inside
# a class is carried inside the class chunk's bytes but does
# not attach to the per-method chunk.
# Cases here focus on top-level `def` / `class` / `module`
# where attachment works cleanly.


#
# tree-sitter-c uses a single `comment` node for both `//`
# and `/* */` (including `/** */`).  Doxygen style is common
# and needs no special handling.


#
# tree-sitter-cpp behaves like tree-sitter-c for comments.
# Cases mirror the C ones but exercise C++-specific constructs
# (classes, methods).
