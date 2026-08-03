"""reStructuredText extraction tests.

The symbol cases (`cases_extraction.py`) drive the shared heading check; the
functions below pin RST's adornment-hierarchy chunker and its reference/role
extraction (`:func:`, `:doc:`, `.. toctree::`, ...).
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_rst_headingless_paragraphs_span_their_own_lines() -> None:
    """A file with no headings spans each paragraph over the lines it occupies.

    Taken from django's `docs/README.rst`, where the one-line paragraph
    followed by a blank line and a bullet list reported line 7 to line 6.
    """
    src = """\
The documentation in this tree is in plain text files and can be viewed using
any text file viewer.

It uses `ReST`_ (reStructuredText), and the `Sphinx`_ documentation system.

To create an HTML version of the docs:

* Install Sphinx (using ``python -m pip install Sphinx`` or some other method).
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    spans = [(c.line_start, c.line_end) for c in chunks if c.kind == ChunkKind.DOC_SECTION]
    assert spans == [(1, 2), (4, 4), (6, 6), (8, 8)]


def test_rst_headingless_file_yields_every_top_level_block() -> None:
    """Without headings, each top-level block is its own section.

    A bullet list, a link target and a directive are content someone
    searches for, so none is dropped for want of a heading above it.
    """
    src = """\
An opening paragraph.

* a bullet item
* another item

1. an enumerated item

.. note::
   A directive body.

.. _ReST: https://docutils.sourceforge.io/rst.html
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    sections = [c.content.split("\n")[0] for c in chunks if c.kind == ChunkKind.DOC_SECTION]
    assert sections == [
        "An opening paragraph.",
        "* a bullet item",
        "1. an enumerated item",
        ".. note::",
        ".. _ReST: https://docutils.sourceforge.io/rst.html",
    ]


def test_rst_headingless_file_still_extracts_references() -> None:
    """A document's references are found whether or not it has a heading."""
    src = "See :doc:`guide` and :func:`do_stuff` for details.\n"
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert [c.metadata.module or c.metadata.names for c in imports] == ["guide", "do_stuff"]


def test_rst_hierarchy_from_adornment_order() -> None:
    """RST reconstructs hierarchy from adornment character order."""
    src = """\
Top
===

Mid
---

Deep
^^^

Content.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    deep = next(c for c in chunks if c.name == "Deep")
    # Deep is the final section; scope shows parent chain.
    assert deep.scope == "Top::Mid"


def test_rst_overline_headings() -> None:
    """RST overline headings produce correct scope."""
    src = """\
=====
Title
=====

Intro.

----------
Subsection
----------

Body.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    sub = next(c for c in chunks if c.name == "Subsection")
    assert sub.scope == "Title"


def test_rst_func_role_produces_import() -> None:
    """RST :func:`name` produces IMPORT with names field."""
    src = """\
Title
=====

See :func:`do_stuff` for details.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "do_stuff"
    assert imports[0].metadata.module == ""


def test_rst_class_role_produces_import() -> None:
    """RST :class:`Name` produces IMPORT with names field."""
    src = """\
Title
=====

See :class:`User` for the model.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "User"


def test_rst_meth_role_produces_import() -> None:
    """RST :meth:`Class.method` produces IMPORT with names field."""
    src = """\
Title
=====

See :meth:`User.save` for persistence.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "User.save"


def test_rst_func_tilde_strips_prefix() -> None:
    """RST :func:`~module.name` strips ~ and uses last component."""
    src = """\
Title
=====

See :func:`~mypackage.utils.do_stuff` for details.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "do_stuff"


def test_rst_doc_role_produces_import() -> None:
    """RST :doc:`path` produces IMPORT with module field."""
    src = """\
Title
=====

See :doc:`api/module` for the API.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "api/module"


def test_rst_mod_role_produces_import() -> None:
    """RST :mod:`name` produces IMPORT with module field."""
    src = """\
Title
=====

See :mod:`mypackage.utils` for helpers.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "mypackage.utils"


def test_rst_toctree_produces_imports() -> None:
    """RST toctree directive produces one IMPORT per entry."""
    src = """\
Title
=====

.. toctree::
   :maxdepth: 2

   intro
   api/index
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    modules = {c.metadata.module for c in imports}
    assert modules == {"intro", "api/index"}


def test_rst_reference_local_produces_import() -> None:
    """RST `text <url>`_ with local path produces import."""
    src = """\
Title
=====

See `the guide <other.rst>`_ for details.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "other.rst"


def test_rst_reference_external_skipped() -> None:
    """RST `text <url>`_ with external URL produces no import."""
    src = """\
Title
=====

See `example <https://example.com>`_ for details.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


def test_rst_plain_prose_no_import() -> None:
    """RST prose mentioning a symbol without a role produces no import."""
    src = """\
Title
=====

Call do_stuff to process the data.
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rst")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []
