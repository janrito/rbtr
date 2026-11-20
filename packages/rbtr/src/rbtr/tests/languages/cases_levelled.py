"""Headings a grammar leaves flat, and where levelling ends and scopes each.

Each case is one file in the installed Markdown (a heading states its
level, `#set! level`) or rST (an adornment style ranked by first use,
`@_level`), and every symbol `list-symbols` gives for it, as
`(kind, name, scope, line_start, line_end)`.
"""

from __future__ import annotations

from dataclasses import dataclass

type Symbol = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class LevelledScenario:
    """A file's path and text, and every symbol it gives."""

    path: str
    text: str
    symbols: list[Symbol]


def case_a_style_is_ranked_by_its_first_use() -> LevelledScenario:
    """`=` used first ranks above `-`, and a later `=` closes the `-` section."""
    return LevelledScenario(
        path="guide.rst",
        text="""\
Guide
=====

Intro.

Install
-------

Run it.

Use
===

Call it.
""",
        symbols=[
            ("doc_section", "Guide", "", 1, 4),
            ("doc_section", "Install", "Guide", 6, 9),
            ("doc_section", "Use", "", 11, 14),
        ],
    )


def case_a_style_first_used_later_ranks_below() -> LevelledScenario:
    """`-` used first ranks above `=`, so the `=` section nests under it."""
    return LevelledScenario(
        path="guide.rst",
        text="""\
Guide
-----

Intro.

Install
=======

Run it.
""",
        symbols=[("doc_section", "Guide", "", 1, 4), ("doc_section", "Install", "Guide", 6, 9)],
    )


def case_stated_levels_compare_as_numbers() -> LevelledScenario:
    """Level 2 first, then level 1, which is not its child."""
    return LevelledScenario(
        path="guide.md",
        text="""\
## Two

text

# One

text
""",
        symbols=[("doc_section", "Two", "", 1, 3), ("doc_section", "One", "", 5, 7)],
    )


def case_a_heading_sits_under_the_whole_chain_above_it() -> LevelledScenario:
    """Each heading sits under the nearest earlier heading of a lower level, the whole chain named."""
    return LevelledScenario(
        path="guide.md",
        text="""\
# A

## B

### C

text
""",
        symbols=[
            ("doc_section", "A", "", 1, 1),
            ("doc_section", "B", "A", 3, 3),
            ("doc_section", "C", "A::B", 5, 7),
        ],
    )


def case_a_heading_in_a_block_ends_with_the_block() -> LevelledScenario:
    """A heading in a fenced block ends with the block, not the file the block sits in."""
    return LevelledScenario(
        path="guide.md",
        text="""\
Intro.

```md
# Inside

text
```

After the fence.
""",
        symbols=[
            ("fragment", "", "", 1, 1),
            ("fragment", "", "", 3, 3),
            ("doc_section", "Inside", "", 4, 6),
            ("fragment", "", "", 7, 7),
            ("fragment", "", "", 9, 9),
        ],
    )


def case_a_heading_holds_the_text_after_it() -> LevelledScenario:
    """The text after a heading is the heading's, not a fragment; text before every heading is a fragment."""
    return LevelledScenario(
        path="guide.md",
        text="""\
Preamble.

# A

Body.
""",
        symbols=[("fragment", "", "", 1, 1), ("doc_section", "A", "", 3, 5)],
    )
