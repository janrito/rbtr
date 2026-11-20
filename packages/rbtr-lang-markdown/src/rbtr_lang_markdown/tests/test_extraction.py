"""Markdown extraction tests.

The symbol cases (`cases_extraction.py`) drive the shared heading-hierarchy
check; the functions below pin Markdown's chunker behaviour (section content,
headingless text as fragments), fenced-code injection/delegation, and link extraction.
"""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
from rbtr.testing import Indexed
from rbtr.testing.reads import outline, read


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as index:
        reply = outline(index)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_md_subsection_excluded_from_parent_content(indexed: Indexed) -> None:
    """Markdown parent chunk excludes child section content."""
    src = """\
# Parent

Parent body.

## Child

Child body.
"""
    with indexed({"input.md": src}) as index:
        [parent] = read(index, "Parent")
    assert "Child body" not in parent.content
    assert "Parent body" in parent.content


def test_md_headingless_paragraphs(indexed: Indexed) -> None:
    """Markdown without headings is held by fragments of its text."""
    src = """\
First paragraph.

Second paragraph.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    assert [(c.kind, c.line_start, c.line_end) for c in chunks] == [(ChunkKind.FRAGMENT, 1, 3)]


def test_md_chunker_target_extracted(indexed: Indexed) -> None:
    """A chunker-based target (yaml) inside a Markdown fence extracts.

    Query targets (python) already worked; this proves the delegate runs a
    chunker plugin over the block too, at absolute line numbers.
    """
    src = """\
# Doc

```yaml
service: greeter
```
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    yaml_sections = [c for c in chunks if c.language == "yaml"]
    assert [c.name for c in yaml_sections] == ["service"]
    assert yaml_sections[0].line_start == 4


def test_md_nested_injection_extracts_inner_js(indexed: Indexed) -> None:
    """Delegation recurses: markdown -> html -> its inline js.

    An HTML block whose HTML contains an inline `<script>` yields both the
    html chunks and the js function, each at absolute line numbers.
    """
    src = """\
# Doc

```html
<body>
  <main>
    <script>
      function boot() {
        return 1;
      }
    </script>
  </main>
</body>
```
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    assert "html" in {c.language for c in chunks}
    js = [c for c in chunks if c.language == "javascript" and c.kind == ChunkKind.FUNCTION]
    assert [c.name for c in js] == ["boot"]
    assert js[0].line_start == 7


def test_md_unknown_fence_left_unparsed(indexed: Indexed) -> None:
    """A fence naming a language rbtr has no grammar for is not delegated."""
    src = """\
# Doc

```nonexistent
some content here
```
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    assert {c.language for c in chunks} == {"markdown"}


def test_md_local_link_produces_import(indexed: Indexed) -> None:
    """Markdown [text](local.md) produces an IMPORT chunk."""
    src = """\
# Guide

See [other doc](other.md) for details.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "other.md"


def test_md_link_with_fragment_sets_names(indexed: Indexed) -> None:
    """Markdown [text](path.md#section) sets module and names."""
    src = """\
# Guide

See [API section](api.md#my-function) for the API.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "api.md"
    assert imports[0].names == "my-function"


def test_md_link_within_the_document_names_its_own_file(indexed: Indexed) -> None:
    """Markdown [text](#section) imports the anchor from the document it is in."""
    src = """\
# Guide

Jump to [usage](#usage).

## Usage
"""
    with indexed({"docs/guide.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert [(i.module, i.names) for i in imports] == [("guide.md", "usage")]


def test_md_relative_link_produces_import(indexed: Indexed) -> None:
    """Markdown [text](../src/foo.py) with relative path produces import."""
    src = """\
# Docs

See [source](../src/foo.py) for details.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "../src/foo.py"


def test_md_external_link_skipped(indexed: Indexed) -> None:
    """Markdown links to external URLs produce no import chunk."""
    src = """\
# Guide

See [example](https://example.com) and [mail](mailto:a@b.com).
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert imports == []


def test_md_multiple_links_in_one_section(indexed: Indexed) -> None:
    """Multiple links in a single section produce multiple imports."""
    src = """\
# Guide

See [a](one.md) and [b](two.py) here.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    modules = {c.module for c in imports}
    assert modules == {"one.md", "two.py"}


def test_md_bare_mention_no_import(indexed: Indexed) -> None:
    """Prose mentioning a symbol name without a link produces no import."""
    src = """\
# Guide

Call do_stuff to process the data.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert imports == []


def test_md_headings_of_either_style_nest_by_level(indexed: Indexed) -> None:
    """Each heading style states its level: `#` 1, `##` 2, an `=` underline 1, a `-` underline 2.

    So `---` is a sibling of `##` and `===` closes both, whichever style
    each is written in.
    """
    src = """\
# Guide

Intro.

Install
-------

Install text.

## Configure

Configure text.

Reference
=========

Reference text.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    assert [
        (c.name, c.scope, c.line_start, c.line_end)
        for c in chunks
        if c.kind == ChunkKind.DOC_SECTION
    ] == [
        ("Guide", "", 1, 3),
        ("Install", "Guide", 5, 8),
        ("Configure", "Guide", 10, 12),
        ("Reference", "", 14, 17),
    ]


def test_md_a_heading_in_a_block_quote_is_a_section_of_its_own(indexed: Indexed) -> None:
    """A heading inside a block quote or list item is a section within the one around it."""
    src = """\
# Notes

> # Quoted
>
> Text.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    assert [
        (c.name, c.line_start, c.line_end) for c in chunks if c.kind == ChunkKind.DOC_SECTION
    ] == [
        ("Notes", 1, 5),
        ("Quoted", 3, 5),
    ]


@pytest.mark.xfail(
    strict=True,
    reason="a nested heading is scoped by the grammar's sections around it, and a "
    "top-level heading is read flat, naming no section: the scope is empty",
)
def test_md_a_nested_heading_is_scoped_under_the_heading_above_it(indexed: Indexed) -> None:
    """`# Notes` then a heading in a list item: the item's heading is scoped `Notes`."""
    src = """\
# Notes

- # Item

  Text.
"""
    with indexed({"input.md": src}) as index:
        chunks = outline(index)
    assert ("Item", "Notes") in [(c.name, c.scope) for c in chunks]
