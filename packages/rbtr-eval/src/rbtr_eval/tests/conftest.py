"""Shared data builders and corpus fixtures for the rbtr-eval suite.

`hit` / `outcome_row` assemble `SearchBatch` rows for the measure
pipeline tests. They are called only from fixtures and case
functions, never from test bodies.

`seed_corpus` seeds an `IndexStore` from a `CorpusScenario`;
`chunk` builds one tokenised chunk for the tests that write index
rows directly.

`mixed_kind_repo` is the suite's one real repo and `mixed_kind_ref` the
same repo built into the reused `store` fixture. Both `extract` and
`paraphrase` measure against it, so a kind it stops producing fails in
both places.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.domain.models import ChunkKind, FileSnapshot, SnapshotRef
from rbtr.domain.tokenise import tokenise_code
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.staging import TokenisedChunk
from rbtr.index.store import IndexStore
from rbtr.tests.conftest import make_commit, store
from rbtr_eval.tests.cases_corpus import HEAD

__all__ = ["store"]  # re-declared here so rbtr-eval tests can request it


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


def seed_corpus(store: IndexStore, repo_path: Path, head: str, indexed: list[str]) -> int:
    """Register *repo_path* and mark each snapshot in *indexed* as indexed.

    The `HEAD` placeholder is substituted with *head*, so a scenario
    can name the repo's real HEAD without knowing the SHA. Each
    snapshot gets one file row, because a snapshot with no files is
    a state the indexer never writes.
    """
    with store.session() as ws:
        repo_id = ws.register_repo(str(repo_path))
        for name in indexed:
            sha = head if name == HEAD else name
            ws.insert_snapshots(
                [FileSnapshot(snapshot_sha=sha, file_path="a.py", blob_sha=f"blob_{sha[:8]}")],
                repo_id=repo_id,
            )
            ws.mark_indexed(repo_id, sha)
    return repo_id


def chunk(*, file_path: str, kind: str, name: str) -> TokenisedChunk:
    """Build one python chunk, tokenised as the indexer would leave it.

    `blob_sha` is derived from *file_path* so every chunk in a file
    shares one blob, which `file_snapshots` requires — its key is
    `(repo_id, snapshot_sha, file_path)`, one blob per path.

    The chunk's id follows from its content, so a test names a chunk
    by passing the object around and reading its `.id`.
    """
    content = f"{name}\n"
    return TokenisedChunk(
        blob_sha=f"blob_{file_path}",
        file_path=file_path,
        kind=ChunkKind(kind),
        name=name,
        language="python",
        file_language="python",
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=1,
    )


def snap(snapshot_sha: str, chunk: TokenisedChunk, *, path: str | None = None) -> FileSnapshot:
    """A snapshot row reaching *chunk*, at its own path or a copy's.

    Blob and language come from the chunk, because those two are what
    pair a file to its chunks — a row that names one without the other
    reaches nothing.
    """
    return FileSnapshot(
        snapshot_sha=snapshot_sha,
        file_path=chunk.file_path if path is None else path,
        blob_sha=chunk.blob_sha,
        detected_language=chunk.file_language,
    )


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
def mixed_kind_ref(mixed_kind_repo: pygit2.Repository, store: IndexStore) -> SnapshotRef:
    """`mixed_kind_repo` built into `store`, addressed at its one commit."""
    head = str(mixed_kind_repo.head.target)
    with store.session() as ws:
        repo_id = ws.register_repo(normalise_repo_path(mixed_kind_repo.workdir))
    build_index(mixed_kind_repo.workdir, head, store)
    return SnapshotRef(repo_id=repo_id, snapshot_sha=head)
