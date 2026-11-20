"""One span, one chunk, through the engine: a file built as rbtr builds a repository, its chunks read back with `Index.chunks`.

A chunk is a span of a file read as a language: two spans are two
chunks, and one span is one chunk, whatever its captures name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.index.refs import resolve_read_ref
from rbtr.languages.manager import get_manager
from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
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
            [row["name"]] if row["name"] else [],
            row["line_start"],
            row["line_end"],
            row["content"],
            [row["metadata"]["module"]] if row["metadata"] and row["metadata"]["module"] else [],
        )
        for row in rows
    ]
    assert sorted(chunks) == sorted(scenario.chunks)


@pytest.mark.xfail(
    strict=True,
    reason="the statement's two imports share one id, so the index keeps one, and the "
    "import of `a` and the call through it are lost (Phase 5: a chunk lists its imports)",
)
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
