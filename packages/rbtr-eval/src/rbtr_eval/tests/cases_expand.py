"""Cases for single-row query expansion.

Each `single_row` case returns
`(queries, model, expected_kind, expected_keywords, expected_variants)`:
a one-row `QueryRow` frame, the mocked LLM that expands it (taken from
a fixture), and the expansion fields the row should carry afterwards.
The cases differ only in data (provenance → query kind); the assertion
logic is shared by the test that consumes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
from pytest_cases import case

from rbtr_eval.schemas import QueryRow

if TYPE_CHECKING:
    from pydantic_ai.models.function import FunctionModel


@case(tags=["single_row"])
def case_concept_query(
    expansion_model: FunctionModel,
) -> tuple[dy.DataFrame[QueryRow], FunctionModel, str, list[str], list[str]]:
    """A docstring-provenance row expands as a concept query."""
    queries = pl.DataFrame(
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
    return (
        queries,
        expansion_model,
        "concept",
        ["settings", "configuration", "options"],
        ["how is the config file parsed"],
    )


@case(tags=["single_row"])
def case_identifier_query(
    identifier_model: FunctionModel,
) -> tuple[dy.DataFrame[QueryRow], FunctionModel, str, list[str], list[str]]:
    """A name-provenance row expands as an identifier query; variants pass through."""
    queries = pl.DataFrame(
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
    # Variants are no longer stripped for identifier queries; this
    # model returned none, so the row simply has none.
    return queries, identifier_model, "identifier", ["merge_scores", "combine_results"], []


@case(tags=["single_row"])
def case_code_query(
    expansion_model: FunctionModel,
) -> tuple[dy.DataFrame[QueryRow], FunctionModel, str, list[str], list[str]]:
    """A body-provenance row expands as a code query (expanded like any other kind)."""
    queries = pl.DataFrame(
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
    return (
        queries,
        expansion_model,
        "code",
        ["settings", "configuration", "options"],
        ["how is the config file parsed"],
    )
