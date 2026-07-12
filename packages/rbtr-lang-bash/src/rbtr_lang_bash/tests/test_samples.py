"""Bash sample extraction tests.

The `sample` fixture runs the real pipeline over `samples/bash/` once and
feeds the shared behavioural checks from `rbtr.languages.testkit`; the
two snapshot comparisons stay inline (syrupy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytest_cases import fixture, parametrize_with_cases

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
