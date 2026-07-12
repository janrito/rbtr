"""Markdown extraction test cases (heading hierarchy → doc sections)."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_md_splits_by_heading() -> SymbolCase:
    """Markdown heading hierarchy."""
    src = """\
# Title

Intro text.

## Section A

Body A.

## Section B

Body B.
"""
    return (
        "markdown",
        src,
        [
            ("doc_section", "Title", ""),
            ("doc_section", "Section A", "Title"),
            ("doc_section", "Section B", "Title"),
        ],
    )


@case(tags=["symbol"])
def case_md_scope_chain() -> SymbolCase:
    """Nested heading scope."""
    src = """\
# Top

## Mid

### Deep

Content here.
"""
    return "markdown", src, [("doc_section", "Deep", "Top::Mid")]


@case(tags=["symbol"])
def case_md_same_name_under_different_parents() -> SymbolCase:
    """Same-named sections under different parents get distinct scopes.

    Two `Overview` subsections — one under `A`, one under `B` — are
    `(doc_section, Overview, A)` and `(doc_section, Overview, B)`. Without
    the full heading path they would collide on identity; addressing
    keeps them apart.
    """
    src = """\
# A

Alpha intro.

## Overview

Alpha overview.

# B

Beta intro.

## Overview

Beta overview.
"""
    return (
        "markdown",
        src,
        [("doc_section", "Overview", "A"), ("doc_section", "Overview", "B")],
    )
