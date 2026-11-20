"""Levelled captures in, each one's end and scope out: rows a query's headings make, with nothing of their language.

Each case gives one file's content and its captures, one per line as
`(kind, name, start_byte, end_byte, line_start, line_end, level,
level_style)`, in order, and the `(order, end_byte, line_end, scope)` the
levelling rule gives each. The captures sit in `source`, which ends at
`source_end`, the content's end unless given.
"""

from __future__ import annotations

from dataclasses import dataclass

type Levelled = tuple[str, str, int, int, int, int, int | None, str | None]


@dataclass(frozen=True)
class LevelledScenario:
    """A file's content, its captures in one source, and each one's `(order, end_byte, line_end, scope)` after levelling."""

    content: bytes
    captures: list[Levelled]
    expected: list[tuple[int, int, int, str]]
    source: int = 0
    source_end: int | None = None


def case_by_first_use() -> LevelledScenario:
    """A style is ranked by its first use: `=` first is above `-`, and a later `=` closes the `-` section."""
    return LevelledScenario(
        content=b"Guide\n=====\n\nIntro.\n\nInstall\n-------\n\nRun it.\n\nUse\n===\n\nCall it.\n",
        captures=[
            ("doc_section", "Guide", 0, 11, 1, 2, None, "="),
            ("doc_section", "Install", 21, 36, 6, 7, None, "-"),
            ("doc_section", "Use", 47, 54, 11, 12, None, "="),
        ],
        expected=[(0, 19, 4, ""), (1, 45, 9, "Guide"), (2, 64, 14, "")],
    )


def case_a_style_first_used_later_ranks_below() -> LevelledScenario:
    """`-` used first ranks above `=`, so the `=` section nests under it."""
    return LevelledScenario(
        content=b"Install\n-------\n\nRun it.\n\nUse\n===\n\nCall it.\n",
        captures=[
            ("doc_section", "Install", 0, 15, 1, 2, None, "-"),
            ("doc_section", "Use", 26, 33, 6, 7, None, "="),
        ],
        expected=[(0, 24, 4, ""), (1, 43, 9, "Install")],
    )


def case_a_stated_level_whatever_the_order() -> LevelledScenario:
    """Stated levels compare as numbers: level 2 first, then level 1, which is not its child."""
    return LevelledScenario(
        content=b"Install\n-------\n\nRun it.\n\nUse\n===\n\nCall it.\n",
        captures=[
            ("doc_section", "Install", 0, 15, 1, 2, 2, None),
            ("doc_section", "Use", 26, 33, 6, 7, 1, None),
        ],
        expected=[(0, 24, 4, ""), (1, 43, 9, "")],
    )


def case_a_chain_three_deep() -> LevelledScenario:
    """Each heading sits under the nearest earlier heading of a lower level, the whole chain named."""
    return LevelledScenario(
        content=b"# A\n\na\n\n## B\n\nb\n\n### C\n\nc\n\n## D\n\nd\n",
        captures=[
            ("doc_section", "A", 0, 3, 1, 1, 1, None),
            ("doc_section", "B", 8, 12, 5, 5, 2, None),
            ("doc_section", "C", 17, 22, 9, 9, 3, None),
            ("doc_section", "D", 27, 31, 13, 13, 2, None),
        ],
        expected=[(0, 6, 3, ""), (1, 15, 7, "A"), (2, 25, 11, "A::B"), (3, 34, 15, "A")],
    )


def case_the_last_ends_at_its_sources_end() -> LevelledScenario:
    """A heading in a block ends with the block, not the file the block sits in."""
    return LevelledScenario(
        content=b"Intro.\n\n```md\n# Inside\n\ntext\n```\n\nAfter the fence.\n",
        captures=[
            ("doc_section", "Inside", 14, 22, 4, 4, 1, None),
        ],
        expected=[(0, 28, 6, "")],
        source=1,
        source_end=29,
    )


def case_a_fragment_inside_is_the_captures_text() -> LevelledScenario:
    """A fragment starting inside a levelled capture's text is that text, and is dropped, its newline past the trimmed end too; one before every levelled capture stays."""
    return LevelledScenario(
        content=b"Preamble.\n\n# A\n\nBody.\n",
        captures=[
            ("fragment", "", 0, 9, 1, 1, None, None),
            ("doc_section", "A", 11, 14, 3, 3, 1, None),
            ("fragment", "", 16, 22, 5, 5, None, None),
        ],
        expected=[(0, 9, 1, ""), (1, 21, 5, "")],
    )
