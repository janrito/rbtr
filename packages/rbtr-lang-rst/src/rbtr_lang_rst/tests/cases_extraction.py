"""reStructuredText extraction test cases (heading hierarchy → doc sections)."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_rst_splits_by_heading() -> SymbolCase:
    """RST heading hierarchy."""
    src = """\
Title
=====

Intro text.

Section A
---------

Body A.

Section B
---------

Body B.
"""
    return (
        "rst",
        src,
        [
            ("doc_section", "Title", ""),
            ("doc_section", "Section B", "Title"),
        ],
    )


@case(tags=["symbol"])
def case_rst_scope_chain() -> SymbolCase:
    """RST nested scope from adornment order."""
    src = """\
Top
===

Mid
---

Deep
^^^

Content here.
"""
    return "rst", src, [("doc_section", "Deep", "Top::Mid")]


@case(tags=["symbol"])
def case_rst_same_name_under_different_parents() -> SymbolCase:
    """Same-named RST subsections under different parents stay distinct.

    Both `Overview` subsections are scoped by their parent only — `A`
    and `B` — never themselves, so they stay distinct.
    """
    src = """\
A
=

Alpha intro.

Overview
--------

Alpha overview.

B
=

Beta intro.

Overview
--------

Beta overview.
"""
    return (
        "rst",
        src,
        [("doc_section", "Overview", "A"), ("doc_section", "Overview", "B")],
    )
