"""How C's calls reach functions, each a small project under `samples/calls/`.

- `prototype`: a call sees the prototype its header declares, and through
  it the definition in another `.c` file, which no include brings in.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    return Path(__file__).parent / "samples" / "calls" / "prototype"


def test_a_call_reaches_the_prototype_and_the_definition(sample_edges: list[str]) -> None:
    """`helper()` in `main.c` calls `util.h`'s prototype and the `helper` `util.c` defines."""
    assert [e for e in sample_edges if e.startswith("main.c::run ->")] == [
        "main.c::run -> util.c::helper [calls]",
        "main.c::run -> util.h::helper [calls]",
    ]
