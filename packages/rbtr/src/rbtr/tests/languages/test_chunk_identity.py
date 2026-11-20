"""One span, one chunk, through the engine: a file built as rbtr builds a repository, its chunks read back with `Index.chunks`.

A chunk is a span of a file read as a language: two spans are two
chunks, and one span is one chunk, whatever its captures name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.daemon.handlers import handle_search
from rbtr.daemon.messages import SearchRequest
from rbtr.domain.models import ChunkKind
from rbtr.index.refs import resolve_read_ref
from rbtr.languages.manager import get_manager
from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr.testing.reads import outline, read, repo_path
from rbtr_lang_python.plugin import python

from .cases_chunk_identity import IdentityScenario


@parametrize_with_cases("scenario", cases=".cases_chunk_identity")
def test_each_span_of_a_file_is_one_chunk(
    indexed: Indexed, tmp_path: Path, scenario: IdentityScenario
) -> None:
    with (
        get_manager().limited_to(*scenario.languages),
        indexed({scenario.path: scenario.text}) as build,
    ):
        at = resolve_read_ref(build.index, build.index.repo(str(tmp_path)), None).at
        rows = build.index.chunks(at=at, file_path=scenario.path).rows(named=True)
    chunks = [
        (
            row["kind"],
            row["name"],
            row["line_start"],
            row["line_end"],
            row["content"],
            [imp["module"] for imp in row["metadata"]["imports"]],
        )
        for row in rows
    ]
    assert sorted(chunks) == sorted(scenario.chunks)


def test_every_module_one_statement_imports_is_reached(indexed: Indexed) -> None:
    """`import os, a` imports `a`, so `a.f()` calls `a`'s `f`."""
    files = {
        "a.py": """\
def f():
    pass
""",
        "b.py": """\
import os, a

a.f()
""",
    }
    with get_manager().limited_to(python), indexed(files) as build:
        assert_reached(
            build,
            {("a.py", "f"): {("b.py", "import os, a", "imports"), ("b.py", "", "calls")}},
        )


def test_each_import_of_a_statement_binds_its_own_module(indexed: Indexed) -> None:
    """`import os as o, a as b` makes `b` `a` and `o` `os`: `b.f()` calls `a`'s `f`, `o.f()` does not."""
    files = {
        "a.py": """\
def f():
    pass
""",
        "b.py": """\
import os as o, a as b


def through_b():
    b.f()


def through_o():
    o.f()
""",
    }
    with get_manager().limited_to(python), indexed(files) as build:
        assert_reached(
            build,
            {
                ("a.py", "f"): {
                    ("b.py", "import os as o, a as b", "imports"),
                    ("b.py", "through_b", "calls"),
                }
            },
        )


def test_each_chunk_is_the_text_of_its_byte_span(indexed: Indexed) -> None:
    """Every stored chunk's content is its file's bytes from `start_byte` to `end_byte`, and its lines are the lines those bytes are on.

    A file per way a chunk is made: captures with a folded comment block,
    lines ending with their newline, blocks another language reads, and
    windows of a plaintext file. `é` is two bytes, so a span counted in
    characters would be found out.
    """
    files = {
        "a.py": """\
# Résumé of the module.


# Comment block,
# folded into the function.
def f():
    return "é"


f()
""",
        "b.c": """\
#include <stdio.h>
#include "b.h"

int g(void) { return 1; }
""",
        "c.md": """\
# Café

Text.

```python
x = "é"
```
""",
        "d.html": """\
<p>é</p>
<script>
const y = 1;
</script>
""",
        "notes.txt": """\


Plain é text,
after two blank lines.
""",
    }
    with indexed(files) as build:
        [repo] = build.index.repos()
        chunks = build.index.chunks(at=resolve_read_ref(build.index, repo, None).at)
    assert set(chunks["file_path"]) == set(files)
    disagreeing = [
        chunk
        for chunk in chunks.rows(named=True)
        for text in [files[chunk["file_path"]].encode()]
        if (chunk["content"], chunk["line_start"], chunk["line_end"])
        != (
            text[chunk["start_byte"] : chunk["end_byte"]].decode(),
            text.count(b"\n", 0, chunk["start_byte"]) + 1,
            text.count(b"\n", 0, max(chunk["start_byte"], chunk["end_byte"] - 1)) + 1,
        )
    ]
    assert disagreeing == []


@pytest.fixture
def several_names() -> dict[str, str]:
    """A module declaring two names at once, and a module importing one of them."""
    return {
        "config.py": """\
low, high = 1, 9
""",
        "app.py": """\
from config import high
""",
    }


def test_a_declaration_of_several_names_is_read_by_each(
    indexed: Indexed, several_names: dict[str, str]
) -> None:
    """`read-symbol` finds `low, high = 1, 9` by `low`, by `high`, and by `config.high`.

    `low` matches by prefix; `high` only within the name, as the import
    `from config import high` also does, so both are found.
    """
    with get_manager().limited_to(python), indexed(several_names) as build:
        found = [[s.name for s in read(build, name)] for name in ("low", "high", "config.high")]
    assert found == [["low, high"], ["from config import high", "low, high"], ["low, high"]]


def test_a_declaration_of_several_names_is_listed_once(
    indexed: Indexed, several_names: dict[str, str]
) -> None:
    """`list-symbols` shows the declaration once, named by its names joined."""
    with get_manager().limited_to(python), indexed(several_names) as build:
        variables = [s.name for s in outline(build, ["config.py"]) if s.kind == ChunkKind.VARIABLE]
    assert variables == ["low, high"]


@pytest.mark.xfail(
    strict=True,
    reason="a declaration of several names is one chunk named by them joined, and an \
import names one of them, which matches no chunk's name",
)
def test_an_import_of_one_name_reaches_the_declaration_of_several(
    indexed: Indexed, several_names: dict[str, str]
) -> None:
    """`from config import high` reaches `low, high = 1, 9`."""
    with get_manager().limited_to(python), indexed(several_names) as build:
        assert_reached(
            build, {("config.py", "low, high"): {("app.py", "from config import high", "imports")}}
        )


def test_search_finds_a_declaration_of_several_names_by_each(
    indexed: Indexed, several_names: dict[str, str]
) -> None:
    """A search for `high` finds the declaration of `low, high`."""
    with get_manager().limited_to(python), indexed(several_names) as build:
        results = handle_search(
            SearchRequest(repo_path=repo_path(build), query="high"), build.index
        ).results
    assert "low, high" in [r.name for r in results]


def test_a_declaration_of_several_names_calls_from_its_scope(indexed: Indexed) -> None:
    """`low, high = f(), g()` at module level calls the module's `f`: `find-refs f` lists it."""
    files = {
        "a.py": """\
def f():
    return 1


def g():
    return 9


low, high = f(), g()
"""
    }
    with get_manager().limited_to(python), indexed(files) as build:
        assert_reached(build, {("a.py", "f"): {("a.py", "low, high", "calls")}})
