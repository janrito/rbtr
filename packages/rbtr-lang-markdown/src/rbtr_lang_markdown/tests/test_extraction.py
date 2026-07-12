"""Markdown extraction tests.

The symbol cases (`cases_extraction.py`) drive the shared heading-hierarchy
check; the functions below pin Markdown's chunker behaviour (section content,
headingless fallback), fenced-code injection/delegation, and link extraction.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_md_subsection_excluded_from_parent_content() -> None:
    """Markdown parent chunk excludes child section content."""
    src = """\
# Parent

Parent body.

## Child

Child body.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    parent = next(c for c in chunks if c.name == "Parent")
    assert "Child body" not in parent.content
    assert "Parent body" in parent.content


def test_md_headingless_paragraphs() -> None:
    """Markdown without headings falls back to plaintext chunking."""
    src = """\
First paragraph.

Second paragraph.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    assert len(chunks) >= 1
    assert all(c.kind == ChunkKind.RAW_CHUNK for c in chunks)


def test_md_chunker_target_extracted() -> None:
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
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    yaml_sections = [c for c in chunks if c.language == "yaml"]
    assert [c.name for c in yaml_sections] == ["service"]
    assert yaml_sections[0].line_start == 4


def test_md_nested_injection_extracts_inner_js() -> None:
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
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    assert "html" in {c.language for c in chunks}
    js = [c for c in chunks if c.language == "javascript" and c.kind == ChunkKind.FUNCTION]
    assert [c.name for c in js] == ["boot"]
    assert js[0].line_start == 7


def test_md_unknown_fence_left_unparsed() -> None:
    """A fence naming a language rbtr has no grammar for is not delegated."""
    src = """\
# Doc

```nonexistent
some content here
```
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    assert {c.language for c in chunks} == {"markdown"}


def test_md_local_link_produces_import() -> None:
    """Markdown [text](local.md) produces an IMPORT chunk."""
    src = """\
# Guide

See [other doc](other.md) for details.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "other.md"


def test_md_link_with_fragment_sets_names() -> None:
    """Markdown [text](path.md#section) sets module and names."""
    src = """\
# Guide

See [API section](api.md#my-function) for the API.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "api.md"
    assert imports[0].metadata.names == "my-function"


def test_md_relative_link_produces_import() -> None:
    """Markdown [text](../src/foo.py) with relative path produces import."""
    src = """\
# Docs

See [source](../src/foo.py) for details.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "../src/foo.py"


def test_md_external_link_skipped() -> None:
    """Markdown links to external URLs produce no import chunk."""
    src = """\
# Guide

See [example](https://example.com) and [mail](mailto:a@b.com).
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


def test_md_fragment_only_link_skipped() -> None:
    """Markdown #-only links (same-file anchors) produce no import."""
    src = """\
# Guide

See [below](#details) for more.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


def test_md_multiple_links_in_one_section() -> None:
    """Multiple links in a single section produce multiple imports."""
    src = """\
# Guide

See [a](one.md) and [b](two.py) here.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    modules = {c.metadata.module for c in imports}
    assert modules == {"one.md", "two.py"}


def test_md_bare_mention_no_import() -> None:
    """Prose mentioning a symbol name without a link produces no import."""
    src = """\
# Guide

Call do_stuff to process the data.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "markdown")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []
