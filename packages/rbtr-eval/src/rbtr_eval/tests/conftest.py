"""Shared data builders for the rbtr-eval test suite.

`hit` / `outcome_row` assemble `SearchBatch` rows for the measure
pipeline tests. They are called only from fixtures and case
functions, never from test bodies.

`seed_corpus` seeds an `IndexStore` from a `CorpusScenario`;
`chunk` builds one tokenised chunk for the tests that write index
rows directly.
"""

from __future__ import annotations

from pathlib import Path

from rbtr.domain.models import ChunkKind, FileSnapshot
from rbtr.domain.tokenise import tokenise_code
from rbtr.index.staging import TokenisedChunk
from rbtr.index.store import IndexStore
from rbtr_eval.tests.cases_corpus import HEAD


def hit(
    file_path: str, scope: str, name: str, line_start: int = 1
) -> dict[str, str | int | list[str]]:
    """Build one hit-struct dict; keeps `SearchBatch` row literals readable."""
    return {
        "file_paths": [file_path],
        "scope": scope,
        "name": name,
        "line_start": line_start,
    }


def outcome_row(
    *,
    slug: str,
    target: str,
    latency_ms: float,
    hits: list[dict[str, str | int | list[str]]],
    query_line_start: int = 1,
    language: str = "python",
    provenance: str = "docstring",
    symbol_kind: str = "function",
    arm: str = "none",
    query_kind: str = "concept",
) -> dict[str, str | int | float | list[dict[str, str | int | list[str]]] | None]:
    """Build one `SearchBatch` row; `target` sets query_name."""
    return {
        "arm": arm,
        "slug": slug,
        "language": language,
        "query_file": "q.py",
        "query_scope": "",
        "query_name": target,
        "query_line_start": query_line_start,
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
