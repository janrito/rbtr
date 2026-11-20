"""One span, one chunk: a file and every chunk the index holds for it.

Each case is one file, the languages it is built with, and its chunks as
`(kind, names, line_start, line_end, content, imports)`, `imports` the
modules the chunk imports. The expectations are the end state of
`.todo/TODO-one-span-one-chunk.md`; a case strict-xfails until the phase
that makes it true.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
from pytest_cases import case

from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr_lang_html.plugin import html
from rbtr_lang_python.plugin import python

type Chunk = tuple[str, list[str], int, int, str, list[str]]


@dataclass(frozen=True)
class IdentityScenario:
    """A file, the languages that build it, and every chunk it gives."""

    path: str
    text: str
    languages: tuple[LanguageRegistration, ...]
    chunks: list[Chunk]


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="two captures of one kind and name on one line share a line-based id, "
        "so the index keeps one (Phase 3: the id hashes the byte span)",
    )
)
def case_one_name_twice_on_a_line_is_two_chunks() -> IdentityScenario:
    """Two assignments to `kind` on one line are two spans, so two chunks."""
    return IdentityScenario(
        path="a.py",
        text="kind = 1; kind = 2\n",
        languages=(python,),
        chunks=[
            ("variable", ["kind"], 1, 1, "kind = 1", []),
            ("variable", ["kind"], 1, 1, "kind = 2", []),
        ],
    )


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="leftover text is cut at every tree level, and two fragments on one line "
        "share an id (Phase 3: byte-span ids; Phase 4: contiguous leftovers join)",
    )
)
def case_leftover_text_on_a_line_is_one_fragment() -> IdentityScenario:
    """`run` and `(` sit at two tree levels but touch, so they are one fragment; the comment between lines splits the rest.

    The language is Python's grammar capturing comments alone, so what is
    left over is fragments and no statement is declared.
    """
    return IdentityScenario(
        path="a.code",
        text="""\
run(
    # note
    1)
""",
        languages=(
            dataclasses.replace(
                python,
                id="code",
                extensions=frozenset({".code"}),
                extraction=QueryExtraction(query="(comment) @comment\n"),
            ),
        ),
        chunks=[
            ("comment", [], 2, 2, "# note", []),
            ("fragment", [], 1, 1, "run(", []),
            ("fragment", [], 3, 3, "1)", []),
        ],
    )


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="each module of one statement is an import chunk of the same name and "
        "lines, so the index keeps one (Phase 5: a chunk lists its imports)",
    )
)
def case_one_statement_importing_two_modules_is_one_chunk() -> IdentityScenario:
    """`import os, sys` is one chunk that imports both modules."""
    return IdentityScenario(
        path="a.py",
        text="import os, sys\n",
        languages=(python,),
        chunks=[("import", ["import os, sys"], 1, 1, "import os, sys", ["os", "sys"])],
    )


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="each name a declaration binds is a variable chunk over the same span "
        "(Phase 6: a chunk lists its names)",
    )
)
def case_one_declaration_of_two_names_is_one_chunk() -> IdentityScenario:
    """`low, high = 1, 9` is one chunk with both names."""
    return IdentityScenario(
        path="a.py",
        text="low, high = 1, 9\n",
        languages=(python,),
        chunks=[("variable", ["low", "high"], 1, 1, "low, high = 1, 9", [])],
    )


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="an element with an id that also links a file is a section and an import "
        "(Phase 3: one span; Phase 5: the section carries the import)",
    )
)
def case_one_node_in_two_roles_is_one_chunk() -> IdentityScenario:
    """A `<link>` with an `id` is one section, named by the id, that imports the stylesheet."""
    return IdentityScenario(
        path="a.html",
        text='<link id="style" rel="stylesheet" href="a.css">\n',
        languages=(html,),
        chunks=[
            (
                "doc_section",
                ["style"],
                1,
                1,
                '<link id="style" rel="stylesheet" href="a.css">',
                ["a.css"],
            )
        ],
    )
