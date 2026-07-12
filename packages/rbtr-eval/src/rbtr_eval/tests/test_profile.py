"""Behaviour tests for the pre-measurement dataset profile.

`_render_report` characterises the query set: which target kinds
are present (coverage), how provenance maps to `classify_query`
(classification), and example queries.  Tested through the
rendered report.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest

from rbtr_eval.profile import _render_report
from rbtr_eval.schemas import QueryRow, RepoHeader


@pytest.fixture
def queries() -> dy.DataFrame[QueryRow]:
    """A small set: a documented function and a prose doc section.

    No `comment`/`config_key` targets, and a `docstring`-provenance
    query whose text is concept-shaped, so the report can be checked
    for coverage and text-based classification.
    """
    base = {"slug": "r", "scope": "", "language": "python"}
    rows = [
        {
            **base,
            "file_path": "a.py",
            "name": "load_config",
            "line_start": 1,
            "symbol_kind": "function",
            "provenance": "name",
            "text": "load_config",
        },
        {
            **base,
            "file_path": "a.py",
            "name": "load_config",
            "line_start": 1,
            "symbol_kind": "function",
            "provenance": "concept",
            "text": "how does configuration loading resolve overrides",
        },
        {
            **base,
            "file_path": "README.md",
            "name": "Getting Started",
            "line_start": 3,
            "symbol_kind": "doc_section",
            "language": "markdown",
            "provenance": "body",
            "text": "walks you through installing and configuring the project cleanly",
        },
    ]
    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True)


@pytest.fixture
def headers() -> dy.DataFrame[RepoHeader]:
    """Header row for the single repo in `queries`."""
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


def test_report_shows_present_kinds_not_absent_ones(
    queries: dy.DataFrame[QueryRow],
    headers: dy.DataFrame[RepoHeader],
) -> None:
    """Coverage: target kinds in the set appear; kinds absent from it do not.

    This is the axis the harness was blind to — a kind that is never
    generated (here `comment`) must be visibly absent, answering "are
    we missing comment/config queries?" from the dataset alone.
    """
    report = _render_report(queries, headers, seed=0)

    assert "function" in report
    assert "doc_section" in report
    assert "comment" not in report
    assert "config_key" not in report


def test_report_classifies_by_text_and_shows_examples(
    queries: dy.DataFrame[QueryRow],
    headers: dy.DataFrame[RepoHeader],
) -> None:
    """The concept-shaped query's text appears and drives classification.

    The example text is surfaced verbatim (so data-quality issues are
    visible pre-run), and `concept` shows a non-zero share because a
    query classifies by its text, not its provenance.
    """
    report = _render_report(queries, headers, seed=0)

    assert "how does configuration loading resolve overrides" in report
    assert "concept" in report


def test_report_lists_exclusions_and_threshold(
    queries: dy.DataFrame[QueryRow],
    headers: dy.DataFrame[RepoHeader],
) -> None:
    """The report names the kinds no queries are generated for and the
    below-threshold drop status, so both are visible pre-measurement."""
    report = _render_report(queries, headers, seed=0)

    assert "Not measured" in report
    assert "`import`" in report
    # The headers fixture drops no language, so the threshold status says so.
    assert "every language met the threshold" in report
