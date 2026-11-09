"""Python's import forms, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `named-imports`: A named import reaches each name: a function, a class,
  variables, and a relative module's class.
- `bare-import`: A bare import reaches a module's definitions, not its
  imports or the code no query names.
- `suffix-in-a-monorepo`: An absolute import resolves across a
  `packages/*/src` layout by its full path suffix.
- `suffix-in-two-packages`: A full path suffix two packages share is
  ambiguous, so it reaches neither.
- `imports-that-reach-nothing`: Imports whose module the repository does not
  hold reach nothing.
- `import-with-no-module`: An import its extractor records nothing for
  reaches files its words name.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.fixture(
    scope="module",
    params=[
        "named-imports",
        "bare-import",
        "suffix-in-a-monorepo",
        "suffix-in-two-packages",
        "imports-that-reach-nothing",
        "import-with-no-module",
    ],
)
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "imports" / str(request.param)


def test_edges_match_snapshot(sample_edges: list[str], snapshot_json: SnapshotAssertion) -> None:
    assert sample_edges == snapshot_json
