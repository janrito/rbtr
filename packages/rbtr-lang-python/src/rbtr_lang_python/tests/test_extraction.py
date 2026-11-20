"""Python extraction tests, through the engine.

Symbol, import, and mixed cases (`cases_extraction.py`) are files the
engine builds (`indexed`) and outlines (`list-symbols`); reference cases
are projects whose calls and bases are read as edges (`find-refs`). The
functions below pin Python's module variables, function locals, class
attributes and import forms.
"""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind, EdgeKind, ImportMeta
from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached
from rbtr.testing.reads import outline, read


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as index:
        reply = outline(index)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "files, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    indexed: Indexed,
    files: dict[str, str],
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    with indexed(files) as index:
        reply = outline(index)
    kinds = {s.kind for s in reply}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(s.name, s.scope) for s in reply if s.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="import")
def test_extracts_import_metadata(indexed: Indexed, files: dict[str, str], expected: dict) -> None:
    """The first import has the expected metadata."""
    with indexed(files) as index:
        reply = outline(index)
    imports = [s.metadata for s in reply if s.kind == ChunkKind.IMPORT]
    assert imports, f"no import extracted from {files!r}"
    assert imports[0] == ImportMeta(**expected)


@parametrize_with_cases(
    "files, count, metadata_list", cases=".cases_extraction", has_tag="multi_import"
)
def test_extracts_multi_import(
    indexed: Indexed,
    files: dict[str, str],
    count: int,
    metadata_list: list[dict],
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    with indexed(files) as index:
        reply = outline(index)
    imports = [s.metadata for s in reply if s.kind == ChunkKind.IMPORT]
    assert len(imports) == count
    assert imports == [ImportMeta(**expected) for expected in metadata_list]


@parametrize_with_cases("files, reached", cases=".cases_extraction", has_tag="references")
def test_a_call_or_base_reaches_what_it_names(
    indexed: Indexed,
    files: dict[str, str],
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` is reached by exactly the callers and subclasses named."""
    with indexed(files) as index:
        assert_reached(index, reached, edges={EdgeKind.CALLS, EdgeKind.INHERITS})


def test_py_module_variable_content_is_whole_statement(indexed: Indexed) -> None:
    """A module-level VARIABLE chunk spans the whole statement, named by LHS."""
    src = """\
MAX_SIZE = 100
"""
    with indexed({"a.py": src}) as index:
        definitions = read(index, "MAX_SIZE")
    assert [(s.kind, s.content) for s in definitions] == [(ChunkKind.VARIABLE, "MAX_SIZE = 100")]


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            """\
def f():
    tmp = 1
    return tmp
""",
            id="function-local",
        ),
        pytest.param(
            """\
class Config:
    DEFAULT = 30
""",
            id="class-attribute",
        ),
    ],
)
def test_py_a_local_or_attribute_is_no_variable(indexed: Indexed, src: str) -> None:
    """Assignments inside a function or a class body stay part of its chunk."""
    with indexed({"a.py": src}) as index:
        reply = outline(index)
    assert [s for s in reply if s.kind == ChunkKind.VARIABLE] == []


def test_py_tuple_unpacking_captured_as_variables(indexed: Indexed) -> None:
    """Flat tuple-unpacking binds each target as its own VARIABLE chunk, spanning the whole statement."""
    src = """\
a, b = compute()
"""
    with indexed({"a.py": src}) as index:
        reply = outline(index)
    assert [(s.kind, s.name, s.line_start, s.line_end) for s in reply] == [
        (ChunkKind.VARIABLE, "a", 1, 1),
        (ChunkKind.VARIABLE, "b", 1, 1),
    ]


def test_py_future_and_aliased_imports_are_imports(indexed: Indexed) -> None:
    """`from __future__ import …` and `import numpy as np` are imports like any other.

    The grammar gives the first its own node type, and nests the second's
    module inside an `aliased_import`, so a query matching only plain
    import statements leaves them in no chunk.
    """
    src = """\
from __future__ import annotations

import os
import numpy as np
"""
    with indexed({"a.py": src}) as index:
        reply = outline(index)
    assert [
        (s.name, s.metadata.module if s.metadata else "")
        for s in reply
        if s.kind == ChunkKind.IMPORT
    ] == [
        ("from __future__ import annotations", ""),
        ("import os", "os"),
        ("import numpy as np", "numpy"),
    ]
