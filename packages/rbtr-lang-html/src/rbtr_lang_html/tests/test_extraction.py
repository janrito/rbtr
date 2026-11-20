"""HTML extraction tests.

The symbol case (`cases_extraction.py`) drives the shared check; the functions
below pin HTML's element naming and its `<script src>` / `<link href>` import
and non-semantic-presence edge behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as build:
        reply = outline(build)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_html_non_semantic_is_a_fragment(indexed: Indexed) -> None:
    """HTML with no head/body/semantic element yields one fragment.

    A non-semantic wrapper (`<div>`) is not elevated to a doc_section, so
    it reaches a chunk only as code no pattern names.
    """
    with indexed({"input.html": "<div>hello</div>"}) as build:
        chunks = outline(build)
    assert [(c.kind, c.line_start, c.line_end) for c in chunks] == [(ChunkKind.FRAGMENT, 1, 1)]


def test_html_script_src_produces_import(indexed: Indexed) -> None:
    """HTML <script src> produces an import chunk."""
    src = """\
<html>
<head><script src="app.js"></script></head>
<body><p>hello</p></body>
</html>
"""
    with indexed({"input.html": src}) as build:
        chunks = outline(build)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "app.js"
    assert imports[0].language_hint == "javascript"


def test_html_link_href_produces_import(indexed: Indexed) -> None:
    """HTML <link href> produces an import chunk."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css"></head>
<body><p>hello</p></body>
</html>
"""
    with indexed({"input.html": src}) as build:
        chunks = outline(build)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "styles.css"
    assert imports[0].language_hint == "css"


def test_html_self_closing_link_produces_import(indexed: Indexed) -> None:
    """An XHTML-style self-closing `<link ... />` produces an import."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css" /></head>
<body><p>hello</p></body>
</html>
"""
    with indexed({"input.html": src}) as build:
        chunks = outline(build)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "styles.css"


def test_html_anchor_href_names_a_page(indexed: Indexed) -> None:
    """`<a href>` to a page imports it, naming the fragment's element; `#top` names its own page.

    A link to an external URL is not recorded.
    """
    src = """\
<html>
<body>
<p>See <a href="guide.html#setup">setup</a>, <a href="about.html">about</a>,
<a href="https://example.com">elsewhere</a> and <a href="#top">the top</a>.</p>
</body>
</html>
"""
    with indexed({"page.html": src}) as build:
        chunks = outline(build)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    # `list-symbols` gives a line's symbols in name order, not source order.
    assert [(i.module, i.names, i.language_hint) for i in imports] == [
        ("about.html", "", ""),
        ("guide.html", "setup", ""),
        ("page.html", "top", ""),
    ]


def test_an_element_a_fragment_can_name_is_a_section_named_by_it(indexed: Indexed) -> None:
    """An element with an `id` or `name` is a section of that name; a section element without one, of its tag.

    A start tag alone, as inline HTML in Markdown gives one, counts too.
    """
    src = """\
<html>
<body>
<section id="setup"><h2 id="install">Install</h2><p><a name="pip"></a>pip.</p></section>
<section><p>Use it.</p></section>
</body>
</html>
"""
    with indexed({"input.html": src}) as build:
        chunks = outline(build)
    # `list-symbols` gives a line's symbols in name order, not source order.
    assert [c.name for c in chunks if c.kind == ChunkKind.DOC_SECTION] == [
        "body",
        "install",
        "pip",
        "setup",
        "section",
    ]
    with indexed({"input.html": '<a id="here">'}) as build:
        lone = outline(build)
    assert [c.name for c in lone if c.kind == ChunkKind.DOC_SECTION] == ["here"]
