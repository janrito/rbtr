"""reStructuredText extraction tests.

The symbol cases (`cases_extraction.py`) drive the shared heading check; the
functions below pin RST's adornment-hierarchy chunker and its reference/role
extraction (`:func:`, `:doc:`, `.. toctree::`, ...).
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
