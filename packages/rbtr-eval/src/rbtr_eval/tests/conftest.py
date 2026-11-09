"""Shared data builders and corpus fixtures for the rbtr-eval suite.

`hit` / `outcome_row` assemble `SearchBatch` rows for the measure
pipeline tests. They are called only from fixtures and case
functions, never from test bodies.

`chunk` builds one tokenised chunk for the tests that write index
rows directly.

`mixed_kind_repo` is the suite's one real repo and `mixed_kind_ref` the
same repo built into the reused `index` fixture. Both `extract` and
`paraphrase` measure against it, so a kind it stops producing fails in
both places.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import dataframely as dy
import polars as pl
import pygit2
import pytest

from rbtr.domain.models import (
    SnapshotRef,
)
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit
from rbtr_eval.shared_schemas import RepoHeader


def hit(
    file_path: str,
    scope: str,
    name: str,
    line_start: int = 1,
    line_end: int | None = None,
    symbol_kind: str = "function",
) -> dict[str, str | int | list[str]]:
    """Build one hit-struct dict; keeps `SearchBatch` row literals readable.

    A hit spans one line unless *line_end* says otherwise, which is what
    a case needs when two chunks share a `line_start` and only the span
    or the kind tells them apart.
    """
    return {
        "file_paths": [file_path],
        "scope": scope,
        "name": name,
        "line_start": line_start,
        "line_end": line_start if line_end is None else line_end,
        "symbol_kind": symbol_kind,
    }


def outcome_row(
    *,
    slug: str,
    target: str,
    latency_ms: float,
    hits: list[dict[str, str | int | list[str]]],
    file_path: str = "q.py",
    line_start: int = 1,
    line_end: int | None = None,
    language: str = "python",
    provenance: str = "docstring",
    symbol_kind: str = "function",
    arm: str = "none",
    query_kind: str = "concept",
) -> dict[str, str | int | float | list[dict[str, str | int | list[str]]] | None]:
    """Build one `SearchBatch` row; `target` is the target chunk's name."""
    return {
        "arm": arm,
        "slug": slug,
        "language": language,
        "file_path": file_path,
        "scope": "",
        "name": target,
        "line_start": line_start,
        "line_end": line_start if line_end is None else line_end,
        "provenance": provenance,
        "symbol_kind": symbol_kind,
        "query_kind": query_kind,
        "query_text": f"doc of {target}",
        "latency_ms": latency_ms,
        "hits": hits,
        "expansion_kind": None,
        "expansion_n_keywords": None,
        "expansion_n_variants": None,
    }


@pytest.fixture
def index() -> Generator[Index]:
    """In-memory writable Index."""
    s = Index(writable=True)
    yield s
    s.close()


@pytest.fixture
def mixed_kind_repo(tmp_path: Path) -> pygit2.Repository:
    """A committed repo whose chunks span every measurable kind.

    `guide.md` holds a fenced bash block whose one line is both a
    command and a trailing comment, so the index contains two anonymous
    chunks sharing a location — the shape that collides when a target is
    addressed by location alone.  `lib.py` carries a standalone comment,
    the anonymous kind no named-symbol fixture reaches.
    """
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False, initial_head="main")
    make_commit(
        repo,
        {
            "guide.md": b"""\
# Guide

```bash
uv run dvc repro    # run every stage end-to-end
```
""",
            "lib.py": b"""\
import os

MAX_RETRIES = 3

# tune the backoff multiplier for flaky networks

def connect(host):
    \"\"\"Open a connection to the given host.\"\"\"
    return os.environ.get(host)
""",
            "pyproject.toml": b"""\
[tool.ruff]
line-length = 88
target-version = "py313"
""",
        },
    )
    return repo


@pytest.fixture
def mixed_kind_ref(mixed_kind_repo: pygit2.Repository, index: Index) -> SnapshotRef:
    """`mixed_kind_repo` built into `index`, addressed at its one commit."""
    head = str(mixed_kind_repo.head.target)
    repo = index.register(normalise_repo_path(mixed_kind_repo.workdir))
    build_index(mixed_kind_repo.workdir, head, index)
    return repo.at(head)


@pytest.fixture
def headers() -> dy.DataFrame[RepoHeader]:
    """The header of one repo, `r`, that dropped no language."""
    return pl.DataFrame(
        [
            {
                "slug": "r",
                "sha": "abc123def456",
                "seed": 0,
                "queries_per_cell": 25,
                "n_documented": 2,
                "n_queries": 3,
                "dropped_languages": [],
            }
        ],
        schema_overrides={
            "dropped_languages": pl.List(pl.Struct({"language": pl.String, "n_chunks": pl.UInt32}))
        },
    ).pipe(RepoHeader.validate, cast=True)
