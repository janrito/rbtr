"""Shared data builders for the rbtr-eval test suite.

`hit` / `outcome_row` assemble `SearchBatch` rows for the measure
pipeline tests. They are called only from fixtures and case
functions, never from test bodies.
"""

from __future__ import annotations


def hit(file_path: str, scope: str, name: str, line_start: int = 1) -> dict[str, str | int]:
    """Build one hit-struct dict; keeps `SearchBatch` row literals readable."""
    return {"file_path": file_path, "scope": scope, "name": name, "line_start": line_start}


def outcome_row(
    *,
    slug: str,
    target: str,
    latency_ms: float,
    hits: list[dict[str, str | int]],
    query_line_start: int = 1,
    language: str = "python",
    provenance: str = "docstring",
    symbol_kind: str = "function",
    arm: str = "none",
    query_kind: str = "concept",
) -> dict[str, str | int | float | list[dict[str, str | int]] | None]:
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
