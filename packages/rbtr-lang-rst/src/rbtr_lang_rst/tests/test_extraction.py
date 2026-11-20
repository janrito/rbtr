"""reStructuredText extraction tests.

The symbol cases (`cases_extraction.py`) drive the shared heading check; the
functions below pin RST's adornment-hierarchy chunker and its reference/role
extraction (`:func:`, `:doc:`, `.. toctree::`, ...).
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


def test_rst_headingless_paragraphs_span_their_own_lines(indexed: Indexed) -> None:
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
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    spans = [(c.line_start, c.line_end) for c in chunks if c.kind == ChunkKind.FRAGMENT]
    assert spans == [(1, 2), (4, 4), (6, 6), (8, 8)]


def test_rst_headingless_file_yields_every_top_level_block(indexed: Indexed) -> None:
    """Without headings, each top-level block is a chunk of its own.

    A bullet list, a link target and a directive are content someone
    searches for, so none is dropped for want of a heading above it: each
    block is a fragment, and a link target a section of its name.
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
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    blocks = sorted((c.line_start, c.kind, src.splitlines()[c.line_start - 1]) for c in chunks)
    assert blocks == [
        (1, ChunkKind.FRAGMENT, "An opening paragraph."),
        (3, ChunkKind.FRAGMENT, "* a bullet item"),
        (6, ChunkKind.FRAGMENT, "1. an enumerated item"),
        (8, ChunkKind.FRAGMENT, ".. note::"),
        (11, ChunkKind.DOC_SECTION, ".. _ReST: https://docutils.sourceforge.io/rst.html"),
    ]


def test_rst_headingless_file_still_extracts_references(indexed: Indexed) -> None:
    """A document's references are found whether or not it has a heading."""
    src = "See :doc:`guide` and :func:`do_stuff` for details.\n"
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert [c.module or c.names for c in imports] == ["guide", "do_stuff"]


def test_rst_hierarchy_from_adornment_order(indexed: Indexed) -> None:
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
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    deep = next(c for c in chunks if c.name == "Deep")
    # Deep is the final section; scope shows parent chain.
    assert deep.scope == "Top::Mid"


def test_rst_overline_headings(indexed: Indexed) -> None:
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
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    sub = next(c for c in chunks if c.name == "Subsection")
    assert sub.scope == "Title"


def test_rst_func_role_produces_import(indexed: Indexed) -> None:
    """RST :func:`name` produces IMPORT with names field."""
    src = """\
Title
=====

See :func:`do_stuff` for details.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].names == "do_stuff"
    assert imports[0].module == ""


def test_rst_class_role_produces_import(indexed: Indexed) -> None:
    """RST :class:`Name` produces IMPORT with names field."""
    src = """\
Title
=====

See :class:`User` for the model.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].names == "User"


def test_rst_meth_role_produces_import(indexed: Indexed) -> None:
    """RST :meth:`Class.method` produces IMPORT with names field."""
    src = """\
Title
=====

See :meth:`User.save` for persistence.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].names == "User.save"


def test_rst_func_tilde_strips_prefix(indexed: Indexed) -> None:
    """RST :func:`~module.name` strips ~ and uses last component."""
    src = """\
Title
=====

See :func:`~mypackage.utils.do_stuff` for details.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].names == "do_stuff"


def test_rst_doc_role_produces_import(indexed: Indexed) -> None:
    """RST :doc:`path` produces IMPORT with module field."""
    src = """\
Title
=====

See :doc:`api/module` for the API.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].module == "api/module"


def test_rst_mod_role_produces_import(indexed: Indexed) -> None:
    """RST :mod:`name` produces IMPORT with module field."""
    src = """\
Title
=====

See :mod:`mypackage.utils` for helpers.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].module == "mypackage.utils"


def test_rst_toctree_produces_imports(indexed: Indexed) -> None:
    """RST toctree directive produces one IMPORT per entry."""
    src = """\
Title
=====

.. toctree::
   :maxdepth: 2

   intro
   api/index
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    modules = {c.module for c in imports}
    assert modules == {"intro", "api/index"}


def test_rst_reference_local_produces_import(indexed: Indexed) -> None:
    """RST `text <url>`_ with local path produces import."""
    src = """\
Title
=====

See `the guide <other.rst>`_ for details.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert len(imports) == 1
    assert imports[0].module == "other.rst"


def test_rst_reference_fragment_names_an_anchor(indexed: Indexed) -> None:
    """RST `text <path#id>`_ names an anchor in that file; `text <#id>`_ one in this file."""
    src = """\
Title
=====

See `the setup <guide.rst#setup>`_ and `the title <#title>`_.
"""
    with indexed({"docs/index.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert [(i.module, i.names) for i in imports] == [
        ("guide.rst", "setup"),
        ("index.rst", "title"),
    ]


def test_rst_ref_role_names_a_label(indexed: Indexed) -> None:
    """RST :ref:`label` and :ref:`text <label>` name the label as written, to be found in rST."""
    src = """\
Title
=====

See :ref:`intro-label` and :ref:`the intro <Other Label>`.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert [(i.module, i.names, i.language_hint) for i in imports] == [
        ("", "intro-label", "rst"),
        ("", "Other Label", "rst"),
    ]


def test_rst_label_is_a_section_named_by_it(indexed: Indexed) -> None:
    """A label or hyperlink target is a section of its name, in source order, with headings or none."""
    headed = """\
.. _intro:
.. _Other Label:

Title
=====

.. _docs: https://example.com
"""
    plain = """\
.. _lone:

Text.
"""
    for src, names in ((headed, ["intro", "Other Label", "Title", "docs"]), (plain, ["lone"])):
        with indexed({"input.rst": src}) as build:
            chunks = outline(build)
        assert [c.name for c in chunks if c.kind == ChunkKind.DOC_SECTION] == names


def test_rst_reference_external_skipped(indexed: Indexed) -> None:
    """RST `text <url>`_ with external URL produces no import."""
    src = """\
Title
=====

See `example <https://example.com>`_ for details.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert imports == []


def test_rst_plain_prose_no_import(indexed: Indexed) -> None:
    """RST prose mentioning a symbol without a role produces no import."""
    src = """\
Title
=====

Call do_stuff to process the data.
"""
    with indexed({"input.rst": src}) as build:
        chunks = outline(build)
    imports = [meta for c in chunks for meta in c.imports]
    assert imports == []
