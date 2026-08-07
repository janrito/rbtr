"""Tests for the paraphrase module.

Uses the `clean_model` / `excluded_model` fixtures (pydantic-ai
`FunctionModel`) to mock the LLM at its boundary for deterministic,
no-API-call tests.
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
from pytest_cases import fixture, parametrize_with_cases

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

from rbtr.domain.models import ChunkKind, SnapshotRef
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.store import IndexStore
from rbtr.tests.conftest import make_commit
from rbtr_eval.paraphrase import (
    SymbolContext,
    _excluded_identifiers,
    _sampled_content,
    paraphrase_agent,
    paraphrase_symbols,
)
from rbtr_eval.schemas import IDENTITY_COLUMNS, QueryRow

# ── _excluded_identifiers ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "scope", "file_path", "present", "absent"),
    [
        (
            "compact_history",
            "Session",
            "src/engine.py",
            {"compact_history", "Session", "engine", "src"},
            set(),
        ),
        (
            "foo",
            "",
            "packages/rbtr/src/rbtr/index/store.py",
            {"foo", "store", "index", "rbtr", "packages", "src"},
            set(),
        ),
        ("foo", "Bar::Baz", "x.py", {"foo", "Bar::Baz", "Bar", "Baz"}, set()),
        ("foo", "", "a/b.py", {"foo"}, {"a", "b"}),
        ("foo", "", "src/store.py", {"foo", "store"}, {"store.py"}),
        ("", "", "src/lib.py", {"lib", "src"}, {""}),
    ],
    ids=[
        "name-and-scope",
        "path-segments",
        "nested-scope",
        "short-segments-skipped",
        "stem-stripped",
        "anonymous-chunk",
    ],
)
def test_excluded_identifiers_derivation(
    name: str, scope: str, file_path: str, present: set[str], absent: set[str]
) -> None:
    """Identifiers to exclude are derived from name, scope, and path.

    `present` must all appear (so the LLM is told to avoid them);
    `absent` must not (short path segments are skipped, file stems are
    stripped of their extension).
    """
    result = set(_excluded_identifiers(name, scope, file_path))
    assert present <= result
    assert absent.isdisjoint(result)


# ── Agent: output_validator rejects excluded identifiers ─────────────


def test_symbol_context_drops_the_empty_identifier() -> None:
    """An anonymous chunk contributes no name, and a name it has not
    withheld excludes nothing — so the empty string never reaches the
    instructions or the output validator.
    """
    deps = SymbolContext(
        language="python",
        symbol_kind=ChunkKind.COMMENT,
        excluded_identifiers=["", "lib"],
    )
    assert deps.excluded_identifiers == ["lib"]


@pytest.fixture
def symbol_deps() -> SymbolContext:
    return SymbolContext(
        language="python", symbol_kind=ChunkKind.FUNCTION, excluded_identifiers=["connect"]
    )


@pytest.fixture
def clean_model() -> FunctionModel:
    """LLM whose paraphrase contains no excluded identifiers."""

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"text": "say hello to a person by their name"}')]
        )

    return FunctionModel(_respond)


@pytest.fixture
def excluded_model() -> FunctionModel:
    """LLM whose paraphrase leaks the excluded `connect` identifier."""

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='{"text": "the connect function opens a socket"}')]
        )

    return FunctionModel(_respond)


def test_output_validator_retries_on_excluded(
    symbol_deps: SymbolContext, excluded_model: FunctionModel
) -> None:
    """FunctionModel returning an excluded identifier triggers ModelRetry.

    The agent retries and gets the same response, eventually
    exhausting retries.
    """
    with pytest.raises(UnexpectedModelBehavior):
        paraphrase_agent.run_sync(
            "```python\ndef connect(): pass\n```",
            deps=symbol_deps,
            model=excluded_model,
        )


def test_output_validator_accepts_clean_response(
    symbol_deps: SymbolContext, clean_model: FunctionModel
) -> None:
    """A response without excluded identifiers passes the validator."""
    result = paraphrase_agent.run_sync(
        "```python\ndef connect(): pass\n```",
        deps=symbol_deps,
        model=clean_model,
    )
    assert result.output.text == "say hello to a person by their name"


# ── End-to-end: paraphrase_symbols ──────────────────────────────────


@fixture
@parametrize_with_cases("queries", cases=".cases_paraphrase")
def paraphrase_result(
    queries: dy.DataFrame[QueryRow],
    mixed_kind_repo: pygit2.Repository,
    store: IndexStore,
    mixed_kind_ref: SnapshotRef,
    clean_model: FunctionModel,
) -> tuple[dy.DataFrame[QueryRow], dy.DataFrame[QueryRow]]:
    """Paraphrase the targeted chunk of the corpus."""
    result = paraphrase_symbols(
        queries,
        store,
        mixed_kind_repo.workdir,
        mixed_kind_ref.repo_id,
        model=clean_model,
        concurrency=1,
    )
    return queries, result


def test_paraphrase_symbols_produces_concept_rows(
    paraphrase_result: tuple[dy.DataFrame[QueryRow], dy.DataFrame[QueryRow]],
) -> None:
    """End-to-end: dedup → content lookup → LLM → validated output.

    An anonymous chunk yields a row like any other: its empty name
    stays out of the exclusion list rather than matching every
    paraphrase and rejecting them all.
    """
    queries, result = paraphrase_result

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["provenance"] == "concept"
    assert row["name"] == queries["name"][0]
    assert row["file_path"] == "lib.py"


@parametrize_with_cases("queries", cases=".cases_paraphrase", has_tag="named")
def test_paraphrase_symbols_skips_when_excluded_in_response(
    queries: dy.DataFrame[QueryRow],
    mixed_kind_repo: pygit2.Repository,
    store: IndexStore,
    mixed_kind_ref: SnapshotRef,
    excluded_model: FunctionModel,
) -> None:
    """Responses containing excluded identifiers exhaust retries and are dropped."""
    result = paraphrase_symbols(
        queries,
        store,
        mixed_kind_repo.workdir,
        mixed_kind_ref.repo_id,
        model=excluded_model,
        concurrency=1,
    )
    assert result.height == 0


@parametrize_with_cases("queries", cases=".cases_paraphrase", has_tag="named")
def test_paraphrase_symbols_deduplicates_across_provenances(
    queries: dy.DataFrame[QueryRow],
    mixed_kind_repo: pygit2.Repository,
    store: IndexStore,
    mixed_kind_ref: SnapshotRef,
) -> None:
    """Multiple provenances for the same symbol produce one LLM call."""
    both = pl.concat([queries, queries.with_columns(pl.lit("name").alias("provenance"))]).pipe(
        QueryRow.validate, cast=True
    )
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

    result = paraphrase_symbols(
        both,
        store,
        mixed_kind_repo.workdir,
        mixed_kind_ref.repo_id,
        model=FunctionModel(counting_fn),
        concurrency=1,
    )

    assert call_count == 1
    assert result.height == 1


# ── Report examples ──────────────────────────────────────────────────


@pytest.fixture
def two_repos(tmp_path: Path, store: IndexStore) -> IndexStore:
    """Two indexed repos whose `lib.py` differ only in the return value.

    A corpus repeats paths across repos, so `greet` at `lib.py:1`
    exists twice with different source text.
    """
    for slug, greeting in (("alpha", "hi"), ("beta", "yo")):
        repo = pygit2.init_repository(str(tmp_path / slug), bare=False, initial_head="main")
        head = str(
            make_commit(
                repo,
                {"lib.py": f'def greet(name):\n    return f"{greeting} {{name}}"\n'.encode()},
            )
        )
        with store.session() as ws:
            ws.register_repo(normalise_repo_path(repo.workdir))
        build_index(repo.workdir, head, store)
    return store


def test_example_content_comes_from_the_sampled_repo(two_repos: IndexStore) -> None:
    """Repos sharing a path each contribute their own source text."""
    sampled = pl.DataFrame(
        [
            {
                "slug": slug,
                "file_path": "lib.py",
                "scope": "",
                "name": "greet",
                "line_start": 1,
                "line_end": 2,
                "symbol_kind": "function",
                "language": "python",
                "provenance": "name",
                "text": "greet someone",
            }
            for slug in ("alpha", "beta")
        ]
    ).pipe(QueryRow.validate, cast=True)

    content = _sampled_content(two_repos, sampled)
    joined = sampled.join(content, on=["slug", *IDENTITY_COLUMNS], how="left").sort("slug")

    assert joined.height == 2
    assert 'f"hi {name}"' in joined["content"][0]
    assert 'f"yo {name}"' in joined["content"][1]
