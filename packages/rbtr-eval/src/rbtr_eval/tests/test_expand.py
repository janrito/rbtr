"""Tests for the expand module.

Uses pydantic-ai's FunctionModel for deterministic,
no-API-call tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

from rbtr_eval.expand import expand_queries
from rbtr_eval.schemas import ExpansionRow, QueryRow

# ── Fixtures ─────────────────────────────────────────────────────────


def _concept_query_frame() -> dy.DataFrame[QueryRow]:
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


def _identifier_query_frame() -> dy.DataFrame[QueryRow]:
    return pl.DataFrame(
        [
            {
                "slug": "test",
                "file_path": "src/search.py",
                "scope": "",
                "name": "fuse_scores",
                "line_start": 10,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "name",
                "text": "fuse_scores",
            },
        ],
    ).pipe(QueryRow.validate, cast=True)


def _code_query_frame() -> dy.DataFrame[QueryRow]:
    return pl.DataFrame(
        [
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


def _concept_model() -> FunctionModel:
    def _fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=(
                        '{"keywords": ["settings", "configuration", "options"],'
                        ' "variants": ["how is the config file parsed"]}'
                    ),
                ),
            ],
        )

    return FunctionModel(_fn)


def _identifier_model() -> FunctionModel:
    def _fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=('{"keywords": ["merge_scores", "combine_results"], "variants": []}'),
                ),
            ],
        )

    return FunctionModel(_fn)


def _raising_model() -> FunctionModel:
    def _fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        msg = "simulated LLM failure"
        raise ValueError(msg)

    return FunctionModel(_fn)


# ── Tests ────────────────────────────────────────────────────────────


def test_expand_concept_query_produces_valid_frame() -> None:
    queries = _concept_query_frame()
    result = expand_queries(queries, model=_concept_model(), concurrency=1)

    ExpansionRow.validate(result)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["query_kind"] == "concept"
    assert row["keywords"] == ["settings", "configuration", "options"]
    assert row["variants"] == ["how is the config file parsed"]


def test_expand_identifier_query_passes_variants_through() -> None:
    queries = _identifier_query_frame()
    result = expand_queries(queries, model=_identifier_model(), concurrency=1)

    ExpansionRow.validate(result)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["query_kind"] == "identifier"
    assert row["keywords"] == ["merge_scores", "combine_results"]
    # Variants are no longer stripped for identifier queries;
    # this model returned none, so the row simply has none.
    assert row["variants"] == []


def test_expand_expands_code_queries() -> None:
    queries = _code_query_frame()
    result = expand_queries(queries, model=_concept_model(), concurrency=1)

    ExpansionRow.validate(result)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["query_kind"] == "code"
    # Code queries are now expanded like any other kind.
    assert row["keywords"] == ["settings", "configuration", "options"]
    assert row["variants"] == ["how is the config file parsed"]


def test_expand_drops_query_on_llm_failure() -> None:
    queries = _concept_query_frame()
    result = expand_queries(queries, model=_raising_model(), concurrency=1)

    # The only query failed to expand — result should be empty.
    assert result.height == 0


def test_expand_mixed_queries() -> None:
    """Concept + code queries: both kinds are expanded."""
    queries = pl.concat([_concept_query_frame(), _code_query_frame()]).pipe(
        QueryRow.validate, cast=True
    )
    result = expand_queries(queries, model=_concept_model(), concurrency=1)

    ExpansionRow.validate(result)
    assert result.height == 2

    concept_row = result.filter(pl.col("name") == "load_config").row(0, named=True)
    assert len(concept_row["keywords"]) > 0
    assert len(concept_row["variants"]) > 0

    code_row = result.filter(pl.col("name") == "get_chunks").row(0, named=True)
    assert len(code_row["keywords"]) > 0
    assert len(code_row["variants"]) > 0
