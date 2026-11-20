"""Engine-level doc-attachment tests, through the engine.

The engine folds symbol-adjacent documentation into chunk content two ways:
an *interior* docstring (Python's `@_docstring` query capture) and an
*exterior* leading comment (a `@comment` block folded into the symbol flush
after it). Per-language doc behaviour is tested in each package; these tests
pin the engine mechanism itself using two default languages — C (exterior)
and Python (interior) — including the `line_start` shift that only exterior
attachment causes, and that a blank-separated comment stays standalone.
"""

from __future__ import annotations

from pathlib import Path

from rbtr.domain.models import ChunkKind
from rbtr.index.refs import resolve_read_ref
from rbtr.languages.manager import get_manager
from rbtr.testing import Indexed


def test_exterior_leading_comment_folds_into_symbol(indexed: Indexed, tmp_path: Path) -> None:
    """A `@comment` block flush before a symbol folds into it, covering L1 (C)."""
    c_language = get_manager().get_registration("c")
    assert c_language is not None
    src = """\
// Compute the answer.
int answer(void) { return 42; }
"""
    with get_manager().limited_to(c_language), indexed({"x.c": src}) as build:
        at = resolve_read_ref(build.index, build.index.repo(str(tmp_path)), None).at
        chunks = build.index.chunks(at=at, file_path="x.c").rows(named=True)
    assert [(c["kind"], c["name"], c["line_start"], c["content"]) for c in chunks] == [
        (ChunkKind.FUNCTION, "answer", 1, src.rstrip("\n"))
    ]


def test_blank_separated_comment_stays_standalone(indexed: Indexed, tmp_path: Path) -> None:
    """A comment separated by a blank line is a standalone COMMENT, not folded (C)."""
    c_language = get_manager().get_registration("c")
    assert c_language is not None
    src = """\
// A banner, attached to nothing.

int answer(void) { return 42; }
"""
    with get_manager().limited_to(c_language), indexed({"x.c": src}) as build:
        at = resolve_read_ref(build.index, build.index.repo(str(tmp_path)), None).at
        chunks = build.index.chunks(at=at, file_path="x.c").rows(named=True)
    assert [(c["kind"], c["name"], c["line_start"], c["content"]) for c in chunks] == [
        (ChunkKind.COMMENT, "", 1, "// A banner, attached to nothing."),
        (ChunkKind.FUNCTION, "answer", 3, "int answer(void) { return 42; }"),
    ]


def test_interior_docstring_is_in_content_without_shifting_line_start(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """A Python interior docstring is part of the function chunk and, unlike an
    exterior leading comment, does not move `line_start` above the def."""
    python = get_manager().get_registration("python")
    assert python is not None
    src = '''\
def answer():
    """Compute the answer."""
    return 42
'''
    with get_manager().limited_to(python), indexed({"x.py": src}) as build:
        at = resolve_read_ref(build.index, build.index.repo(str(tmp_path)), None).at
        chunks = build.index.chunks(at=at, file_path="x.py").rows(named=True)
    assert [(c["kind"], c["name"], c["line_start"], c["content"]) for c in chunks] == [
        (ChunkKind.FUNCTION, "answer", 1, src.rstrip("\n"))
    ]
