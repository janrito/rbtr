"""Behaviour tests for extract query generation.

Docstring detection uses tree-sitter's `extract_doc_spans`
on the chunk content. Tested through the observable output:
which provenances are produced for each symbol.
"""

from __future__ import annotations

import polars as pl
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import SnapshotRef
from rbtr.index.store import IndexStore
from rbtr_eval.extract import extract_queries, queries_for_symbol
from rbtr_eval.schemas import IDENTITY_COLUMNS


@parametrize_with_cases(
    "content, language, name, expected_provenances",
    cases=".cases_extract",
    has_tag="yields_queries",
)
def test_generates_expected_provenances(
    content: str, language: str, name: str, expected_provenances: set[str]
) -> None:
    """Symbol produces the expected set of provenances."""
    queries = queries_for_symbol(
        slug="test",
        file_path="test.py",
        scope="",
        name=name,
        symbol_kind="function",
        line_start=1,
        line_end=content.count("\n") + 1,
        language=language,
        content=content,
    )
    actual_provenances = {q["provenance"] for q in queries}
    assert actual_provenances == expected_provenances


def test_extract_covers_measurable_kinds_and_excludes_import(
    store: IndexStore, mixed_kind_ref: SnapshotRef
) -> None:
    """Every measurable target kind yields queries; excluded kinds yield none.

    The regression this whole change is about: a searchable kind
    (`comment`, `config_key`, `variable`) must be measured, and an
    excluded one (`import`) must not — so a dropped kind can never
    again pass unnoticed.
    """
    queries, _, _ = extract_queries(
        store,
        "test",
        mixed_kind_ref.repo_id,
        mixed_kind_ref.snapshot_sha,
        min_per_language=1,
    )
    kinds = set(queries["symbol_kind"].cast(pl.String).to_list())

    assert {"comment", "config_key", "function", "variable"} <= kinds
    assert "import" not in kinds


def test_generated_queries_target_real_chunks(
    store: IndexStore, mixed_kind_ref: SnapshotRef
) -> None:
    """Every query's target identity resolves to an indexed chunk.

    Guarantees the generated queries are scoreable — including the
    anonymous chunks (comment) matched by location, not name.
    """
    queries, _, _ = extract_queries(
        store,
        "test",
        mixed_kind_ref.repo_id,
        mixed_kind_ref.snapshot_sha,
        min_per_language=1,
    )
    chunk_ids = {
        (c.file_path, c.scope, c.name, c.line_start, c.line_end, c.kind.value)
        for c in store.get_chunks(mixed_kind_ref.snapshot_sha, repo_id=mixed_kind_ref.repo_id)
    }
    for row in queries.iter_rows(named=True):
        assert tuple(row[c] for c in IDENTITY_COLUMNS) in chunk_ids


def test_two_chunks_on_one_line_are_two_targets(
    store: IndexStore, mixed_kind_ref: SnapshotRef
) -> None:
    """A shared location yields a query each, not one colliding key.

    A fenced bash line is both an anonymous `doc_section` (the command)
    and an anonymous `comment` (its trailing comment), at one
    `line_start`. Each is separately searchable, so each is its own
    target and the span and kind are what tell them apart.
    """
    queries, _, _ = extract_queries(
        store,
        "test",
        mixed_kind_ref.repo_id,
        mixed_kind_ref.snapshot_sha,
        min_per_language=1,
    )

    fenced = queries.filter(
        (pl.col("file_path") == "guide.md")
        & (pl.col("line_start") == 4)
        & (pl.col("provenance") == "body")
    )
    assert set(fenced["symbol_kind"].cast(pl.String).to_list()) == {"comment", "doc_section"}
