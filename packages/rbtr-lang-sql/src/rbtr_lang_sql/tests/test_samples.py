"""SQL sample extraction tests.

The `sample` fixture runs the real pipeline over `samples/sql/` once and
feeds the shared behavioural checks from `rbtr.languages.testkit`; the
two snapshot comparisons stay inline (syrupy). The `sql_dialect` family
documents how the single generic grammar handles each dialect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pytest_cases import fixture, parametrize_with_cases
from tree_sitter import Parser

from rbtr.languages import LanguageManager
from rbtr.languages.testkit import (
    SampleData,
    assert_sample_blob_sha_propagated,
    assert_sample_chunk_ids_deterministic,
    assert_sample_content_nonempty,
    assert_sample_emits_kinds,
    assert_sample_line_numbers_positive,
    assert_sample_non_import_metadata_empty,
    assert_sample_parses_cleanly,
    assert_sample_survives_syntax_error,
    build_sample_data,
    extract_chunks,
    render_edges,
)

from .conftest import SAMPLES_DIR

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@fixture
@parametrize_with_cases("lang, expected_kinds", cases=".cases_samples", has_tag="sample")
def sample(lang: str, expected_kinds: set) -> SampleData:
    """Language id, expected kinds, project files, and extracted chunks + edges."""
    return build_sample_data(SAMPLES_DIR, lang, expected_kinds)


def test_sample_emits_expected_kinds(sample: SampleData) -> None:
    assert_sample_emits_kinds(sample)


def test_sample_parses_cleanly(sample: SampleData) -> None:
    assert_sample_parses_cleanly(sample)


def test_sample_chunk_ids_deterministic(sample: SampleData) -> None:
    assert_sample_chunk_ids_deterministic(sample)


def test_sample_line_numbers_positive(sample: SampleData) -> None:
    assert_sample_line_numbers_positive(sample)


def test_sample_content_nonempty(sample: SampleData) -> None:
    assert_sample_content_nonempty(sample)


def test_sample_non_import_metadata_empty(sample: SampleData) -> None:
    assert_sample_non_import_metadata_empty(sample)


def test_sample_blob_sha_propagated(sample: SampleData) -> None:
    assert_sample_blob_sha_propagated(sample)


def test_sample_survives_syntax_error(sample: SampleData) -> None:
    assert_sample_survives_syntax_error(sample)


def test_sample_extraction_matches_snapshot(
    sample: SampleData, snapshot_json: SnapshotAssertion
) -> None:
    """Extracted chunks match the committed golden snapshot."""
    _lang, _kinds, _files, chunks, _edges = sample
    assert chunks == snapshot_json


def test_sample_edges_match_snapshot(sample: SampleData, snapshot_json: SnapshotAssertion) -> None:
    """Import edges among the project's files match the golden snapshot."""
    _lang, _kinds, _files, chunks, edges = sample
    assert render_edges(edges, chunks) == snapshot_json


@parametrize_with_cases("source, dialect", cases=".cases_samples", has_tag="sql_dialect")
def test_sql_dialect_extraction_matches_snapshot(
    source: str, dialect: str, snapshot_json: SnapshotAssertion
) -> None:
    """Current extraction for each SQL dialect under the generic grammar."""
    chunks = extract_chunks("sql", source, file_path=f"{dialect}.sql")
    assert chunks == snapshot_json


@pytest.mark.xfail(
    reason="generic SQL grammar does not fully parse dialect-specific syntax",
    strict=True,
)
@parametrize_with_cases("source, dialect", cases=".cases_samples", has_tag="sql_dialect")
def test_sql_dialect_parses_cleanly(
    source: str, dialect: str, language_manager: LanguageManager
) -> None:
    """Sentinel: flips to XPASS (failing) when a dialect parses cleanly."""
    grammar = language_manager.load_grammar("sql")
    assert grammar is not None
    tree = Parser(grammar).parse(source.encode())
    assert not tree.root_node.has_error
