"""Tests for the paraphrase module.

Uses pydantic-ai's TestModel and FunctionModel for
deterministic, no-API-call tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
import pygit2
import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

from rbtr.index.models import ChunkKind
from rbtr.index.store import IndexStore
from rbtr_eval.paraphrase import (
    SymbolContext,
    _excluded_identifiers,
    paraphrase_agent,
    paraphrase_symbols,
)
from rbtr_eval.schemas import ConceptQuery, QueryRow

# ── _excluded_identifiers ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "scope", "file_path", "expected_subset"),
    [
        (
            "compact_history",
            "Session",
            "src/engine.py",
            {"compact_history", "Session", "engine", "src"},
        ),
        (
            "foo",
            "",
            "packages/rbtr/src/rbtr/index/store.py",
            {"foo", "store", "index", "rbtr", "packages", "src"},
        ),
        ("foo", "Bar.Baz", "x.py", {"foo", "Bar.Baz", "Bar", "Baz"}),
    ],
    ids=["name-and-scope", "path-segments", "dotted-scope"],
)
def test_excluded_includes_expected(
    name: str, scope: str, file_path: str, expected_subset: set[str]
) -> None:
    result = set(_excluded_identifiers(name, scope, file_path))
    assert expected_subset <= result


def test_excluded_skips_short_path_segments() -> None:
    result = _excluded_identifiers("foo", "", "a/b.py")
    assert "a" not in result
    assert "b" not in result


def test_excluded_strips_extensions_via_stem() -> None:
    result = _excluded_identifiers("foo", "", "src/store.py")
    assert "store" in result
    assert "store.py" not in result


# ── Agent: output_validator rejects excluded identifiers ─────────────


@pytest.fixture
def symbol_deps() -> SymbolContext:
    return SymbolContext(
        language="python", symbol_kind=ChunkKind.FUNCTION, excluded_identifiers=["greet"]
    )


def test_agent_produces_structured_output(symbol_deps: SymbolContext) -> None:
    """TestModel produces a ConceptQuery with valid length."""
    result = paraphrase_agent.run_sync(
        "```python\ndef greet(): pass\n```",
        deps=symbol_deps,
        model=TestModel(),
    )
    assert isinstance(result.output, ConceptQuery)
    assert len(result.output.text) >= 15


def test_output_validator_retries_on_excluded(symbol_deps: SymbolContext) -> None:
    """FunctionModel returning an excluded identifier triggers ModelRetry.

    The agent retries and gets the same response, eventually
    exhausting retries.
    """

    def returns_excluded(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"text": "the greet function says hello"}')])

    with pytest.raises(UnexpectedModelBehavior):
        paraphrase_agent.run_sync(
            "```python\ndef greet(): pass\n```",
            deps=symbol_deps,
            model=FunctionModel(returns_excluded),
        )


def test_output_validator_accepts_clean_response(symbol_deps: SymbolContext) -> None:
    """A response without excluded identifiers passes the validator."""

    def returns_clean(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"text": "say hello to a person by their name"}')]
        )

    result = paraphrase_agent.run_sync(
        "```python\ndef greet(): pass\n```",
        deps=symbol_deps,
        model=FunctionModel(returns_clean),
    )
    assert result.output.text == "say hello to a person by their name"


# ── End-to-end: paraphrase_symbols ──────────────────────────────────


@pytest.fixture
def mini_repo(tmp_path: Path) -> tuple[Path, IndexStore, int]:
    """A minimal git repo with one documented Python function, indexed."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = pygit2.init_repository(str(repo_path), bare=False)
    src = repo_path / "lib.py"
    src.write_text('def greet(name):\n    """Return a greeting."""\n    return f"hi {name}"\n')
    index = repo.index
    index.add("lib.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    from rbtr.index.orchestrator import build_index  # deferred: heavy native libs

    resolved = str(repo_path.resolve())
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(resolved)
    head = str(repo.head.target)
    build_index(repo.workdir, head, store, repo_id=repo_id)
    return repo_path, store, repo_id


@pytest.fixture
def mini_queries() -> dy.DataFrame[QueryRow]:
    """A QueryRow frame pointing at the symbol in mini_repo."""
    return pl.DataFrame(
        [
            {
                "slug": "test",
                "file_path": "lib.py",
                "scope": "",
                "name": "greet",
                "line_start": 1,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "docstring",
                "text": "Return a greeting.",
            }
        ]
    ).pipe(QueryRow.validate, cast=True)


def _clean_model() -> FunctionModel:
    def _fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"text": "produce a friendly welcome message for someone"}')]
        )

    return FunctionModel(_fn)


def _excluded_model() -> FunctionModel:
    def _fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"text": "the greet function produces a greeting"}')]
        )

    return FunctionModel(_fn)


def test_paraphrase_symbols_produces_concept_rows(
    mini_repo: tuple[Path, IndexStore, int], mini_queries: dy.DataFrame[QueryRow]
) -> None:
    """End-to-end: dedup → content lookup → LLM → validated output."""
    repo_path, store, repo_id = mini_repo
    result = paraphrase_symbols(
        mini_queries,
        store,
        str(repo_path.resolve()),
        repo_id,
        model=_clean_model(),
        concurrency=1,
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["provenance"] == "concept"
    assert row["name"] == "greet"
    assert row["file_path"] == "lib.py"
    assert "greet" not in str(row["text"]).lower()


def test_paraphrase_symbols_skips_when_excluded_in_response(
    mini_repo: tuple[Path, IndexStore, int], mini_queries: dy.DataFrame[QueryRow]
) -> None:
    """Responses containing excluded identifiers exhaust retries and are dropped."""
    repo_path, store, repo_id = mini_repo
    result = paraphrase_symbols(
        mini_queries,
        store,
        str(repo_path.resolve()),
        repo_id,
        model=_excluded_model(),
        concurrency=1,
    )
    assert result.height == 0


def test_paraphrase_symbols_deduplicates_across_provenances(
    mini_repo: tuple[Path, IndexStore, int], mini_queries: dy.DataFrame[QueryRow]
) -> None:
    """Multiple provenances for the same symbol produce one LLM call."""
    name_row = mini_queries.with_columns(
        pl.lit("name").alias("provenance"),
        pl.lit("greet").alias("text"),
    )
    both = pl.concat([mini_queries, name_row]).pipe(QueryRow.validate, cast=True)
    assert both.height == 2

    call_count = 0

    def counting_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(
            parts=[TextPart(content='{"text": "produce a friendly welcome message for someone"}')]
        )

    repo_path, store, repo_id = mini_repo
    result = paraphrase_symbols(
        both,
        store,
        str(repo_path.resolve()),
        repo_id,
        model=FunctionModel(counting_fn),
        concurrency=1,
    )

    assert call_count == 1
    assert result.height == 1
