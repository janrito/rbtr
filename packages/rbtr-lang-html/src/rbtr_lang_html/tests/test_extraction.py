"""HTML extraction tests.

The symbol case (`cases_extraction.py`) drives the shared check; the functions
below pin HTML's element naming and its `<script src>` / `<link href>` import
and non-semantic-presence edge behaviour.
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


def test_html_non_semantic_only_emits_presence() -> None:
    """HTML with no head/body/semantic element yields only a presence chunk.

    A non-semantic wrapper (`<div>`) is not elevated, so no searchable
    doc_section is produced; the engine appends one content-less host chunk.
    """
    chunks = extract_file(FileEntry("input", "sha1", b"<div>hello</div>"), "html")
    assert [c.kind for c in chunks if c.content] == []
    assert [c.language for c in chunks] == ["html"]


def test_html_script_src_produces_import() -> None:
    """HTML <script src> produces an import chunk."""
    src = """\
<html>
<head><script src="app.js"></script></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "app.js"
    assert imports[0].metadata.language_hint == "javascript"


def test_html_link_href_produces_import() -> None:
    """HTML <link href> produces an import chunk."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css"></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "styles.css"
    assert imports[0].metadata.language_hint == "css"


def test_html_self_closing_link_produces_import() -> None:
    """An XHTML-style self-closing `<link ... />` produces an import."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css" /></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "html")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "styles.css"
