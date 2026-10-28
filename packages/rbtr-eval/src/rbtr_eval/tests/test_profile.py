"""Behaviour tests for the pre-measurement dataset profile.

The profile characterises the query set: which target kinds are
present (coverage), how provenance maps to `classify_query`
(classification), and example queries.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest

from rbtr_eval.profile import _dropped_languages, _examples, _query_counts, _render_report
from rbtr_eval.shared_schemas import QueryRow, RepoHeader


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
            "line_end": 1,
            "symbol_kind": "function",
            "provenance": "name",
            "text": "load_config",
        },
        {
            **base,
            "file_path": "a.py",
            "name": "load_config",
            "line_start": 1,
            "line_end": 1,
            "symbol_kind": "function",
            "provenance": "concept",
            "text": "how does configuration loading resolve overrides",
        },
        {
            **base,
            "file_path": "README.md",
            "name": "Getting Started",
            "line_start": 3,
            "line_end": 3,
            "symbol_kind": "doc_section",
            "language": "markdown",
            "provenance": "body",
            "text": "walks you through installing and configuring the project cleanly",
        },
    ]
    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True)


def test_counts_show_present_kinds_not_absent_ones(
    queries: dy.DataFrame[QueryRow],
) -> None:
    """Coverage: target kinds in the set are counted; kinds absent from it are not.

    A kind that is never generated (here `comment`) must be visibly
    absent, answering "are we missing comment/config queries?" from the
    dataset alone.
    """
    counts = _query_counts(queries)

    assert set(counts["symbol_kind"].cast(pl.String)) == {"function", "doc_section"}


def test_queries_classify_by_text_and_examples_show_it(
    queries: dy.DataFrame[QueryRow],
) -> None:
    """A query's kind follows its text, and the examples carry the text
    verbatim, so data-quality issues are visible pre-run.

    Both `load_config` queries target one function; the name classifies
    as an identifier and the question as a concept.
    """
    examples = _examples(queries, per_provenance=3, seed=0)

    kind_of = dict(zip(examples["text"], examples["query_kind"], strict=True))
    assert kind_of["load_config"] == "identifier"
    assert kind_of["how does configuration loading resolve overrides"] == "concept"


def test_report_lists_exclusions_and_threshold(
    queries: dy.DataFrame[QueryRow],
    headers: dy.DataFrame[RepoHeader],
) -> None:
    """The report names the kinds no queries are generated for and the
    below-threshold drop status, so both are visible pre-measurement."""
    dropped = _dropped_languages(headers)
    report = _render_report(
        _query_counts(queries), dropped, _examples(queries, per_provenance=3, seed=0), headers
    )

    assert dropped.is_empty()
    assert "Not measured" in report
    assert "`import`" in report
    # The headers fixture drops no language, so the threshold status says so.
    assert "every language met the threshold" in report
