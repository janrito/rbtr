"""Shared data builders for the rbtr-eval test suite.

`hit` / `outcome_row` assemble `SearchBatch` rows for the measure
pipeline tests. They are called only from fixtures and case
functions, never from test bodies.

`seed_corpus` seeds an `IndexStore` from a `CorpusScenario`.
"""

from __future__ import annotations

from pathlib import Path

from rbtr.domain.models import FileSnapshot
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
