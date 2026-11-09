"""HTML extraction tests.

The symbol case (`cases_extraction.py`) drives the shared check; the functions
below pin HTML's element naming and its `<script src>` / `<link href>` import
and non-semantic-presence edge behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind, Import, Section
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_html_non_semantic_is_a_fragment() -> None:
    """HTML with no head/body/semantic element yields one fragment.

    A non-semantic wrapper (`<div>`) is not elevated to a doc_section, so
    it reaches a chunk only as code no pattern names.
    """
    chunks = extract_file(FileEntry("input", "sha1", b"<div>hello</div>"), "html")
    assert [(c.kind, c.content) for c in chunks] == [(ChunkKind.FRAGMENT, "<div>hello</div>")]


def test_html_script_src_produces_import() -> None:
    """HTML <script src> produces an import chunk."""
    src = """\
<html>
<head><script src="app.js"></script></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c.role for c in chunks if isinstance(c.role, Import)]
    assert len(imports) == 1
    assert imports[0].meta.module == "app.js"
    assert imports[0].meta.language_hint == "javascript"


def test_html_link_href_produces_import() -> None:
    """HTML <link href> produces an import chunk."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css"></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c.role for c in chunks if isinstance(c.role, Import)]
    assert len(imports) == 1
    assert imports[0].meta.module == "styles.css"
    assert imports[0].meta.language_hint == "css"


def test_html_self_closing_link_produces_import() -> None:
    """An XHTML-style self-closing `<link ... />` produces an import."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css" /></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c.role for c in chunks if isinstance(c.role, Import)]
    assert len(imports) == 1
    assert imports[0].meta.module == "styles.css"


def test_html_anchor_href_names_another_page() -> None:
    """`<a href>` to another page imports it, naming the fragment's anchor.

    A link to an external URL is not recorded; nor, yet, is a link to an
    anchor on its own page.
    """
    src = """\
<html>
<body>
<p>See <a href="guide.html#setup">setup</a>, <a href="about.html">about</a>,
<a href="https://example.com">elsewhere</a> and <a href="#top">the top</a>.</p>
</body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c.role for c in chunks if isinstance(c.role, Import)]
    assert [(i.meta.module, i.meta.names, i.meta.language_hint) for i in imports] == [
        ("guide.html", "setup", ""),
        ("about.html", "", ""),
    ]


def test_html_section_answers_to_its_id() -> None:
    """A section element answers to its `id`; one without an `id` to none."""
    src = """\
<html>
<body>
<section id="setup"><p>Install it.</p></section>
<section><p>Use it.</p></section>
</body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    sections = [c.role for c in chunks if isinstance(c.role, Section) and c.role.name != "body"]
    assert [(s.name, s.meta.anchors) for s in sections] == [("setup", ("setup",)), ("section", ())]
