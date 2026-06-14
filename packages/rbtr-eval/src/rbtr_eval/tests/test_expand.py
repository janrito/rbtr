"""Tests for the expand module.

Uses pydantic-ai's FunctionModel via per-response fixtures
(`expansion_model`, `identifier_model`, `raising_model`) to mock the
LLM at its boundary for deterministic, no-API-call tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest_cases import parametrize_with_cases

from rbtr_eval.expand import expand_queries
from rbtr_eval.schemas import ExpansionRow, QueryRow

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage


# ── Models ───────────────────────────────────────────────────────────


@pytest.fixture
def expansion_model() -> FunctionModel:
    """LLM returning a fixed keyword/variant expansion (concept/code shape)."""

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=(
                        '{"keywords": ["settings", "configuration", "options"],'
                        ' "variants": ["how is the config file parsed"]}'
                    )
                )
            ]
        )

    return FunctionModel(_respond)


@pytest.fixture
def identifier_model() -> FunctionModel:
    """LLM returning keywords but no variants (identifier shape)."""

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content='{"keywords": ["merge_scores", "combine_results"], "variants": []}'
                )
            ]
        )

    return FunctionModel(_respond)


@pytest.fixture
def raising_model() -> FunctionModel:
    """LLM that raises on every call (simulated failure)."""

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        msg = "simulated LLM failure"
        raise ValueError(msg)

    return FunctionModel(_respond)


# ── Query frames ─────────────────────────────────────────────────────


@pytest.fixture
def concept_query() -> dy.DataFrame[QueryRow]:
    """A single docstring-provenance row (a concept query)."""
    return pl.DataFrame(
        [
            {
                "slug": "test",
                "file_path": "src/config.py",
                "scope": "",
                "name": "load_config",
                "line_start": 1,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "docstring",
                "text": "how does configuration loading work",
            },
        ],
    ).pipe(QueryRow.validate, cast=True)


@pytest.fixture
def mixed_queries() -> dy.DataFrame[QueryRow]:
    """A concept row and a code row, to exercise a multi-kind batch."""
    return pl.DataFrame(
        [
            {
                "slug": "test",
                "file_path": "src/config.py",
                "scope": "",
                "name": "load_config",
                "line_start": 1,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "docstring",
                "text": "how does configuration loading work",
            },
            {
                "slug": "test",
                "file_path": "src/store.py",
                "scope": "",
                "name": "get_chunks",
                "line_start": 20,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "body",
                "text": "def get_chunks(self, sha: str):",
            },
        ],
    ).pipe(QueryRow.validate, cast=True)


# ── Tests ────────────────────────────────────────────────────────────


@parametrize_with_cases(
    "queries, model, expected_kind, expected_keywords, expected_variants",
    cases=".cases_expand",
    has_tag="single_row",
)
def test_expand_single_row(
    queries: dy.DataFrame[QueryRow],
    model: FunctionModel,
    expected_kind: str,
    expected_keywords: list[str],
    expected_variants: list[str],
) -> None:
    """One row expands to a valid `ExpansionRow` with the expected fields."""
    result = expand_queries(queries, model=model, concurrency=1)

    ExpansionRow.validate(result)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["query_kind"] == expected_kind
    assert row["keywords"] == expected_keywords
    assert row["variants"] == expected_variants


def test_expand_drops_query_on_llm_failure(
    concept_query: dy.DataFrame[QueryRow], raising_model: FunctionModel
) -> None:
    """A query whose expansion raises is dropped, leaving an empty frame."""
    result = expand_queries(concept_query, model=raising_model, concurrency=1)

    # The only query failed to expand — result should be empty.
    assert result.height == 0


def test_expand_mixed_queries(
    mixed_queries: dy.DataFrame[QueryRow], expansion_model: FunctionModel
) -> None:
    """Concept + code queries: both kinds are expanded."""
    result = expand_queries(mixed_queries, model=expansion_model, concurrency=1)

    ExpansionRow.validate(result)
    assert result.height == 2

    concept_row = result.filter(pl.col("name") == "load_config").row(0, named=True)
    assert len(concept_row["keywords"]) > 0
    assert len(concept_row["variants"]) > 0

    code_row = result.filter(pl.col("name") == "get_chunks").row(0, named=True)
    assert len(code_row["keywords"]) > 0
    assert len(code_row["variants"]) > 0
