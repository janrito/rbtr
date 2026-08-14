"""Behaviour tests for extract query generation.

Docstring detection uses tree-sitter's `extract_doc_spans`
on the chunk content. Tested through the observable output:
which provenances are produced for each symbol.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pygit2
import pytest
from pytest_cases import parametrize_with_cases

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


@pytest.fixture
def mixed_kind_index(tmp_path: Path) -> tuple[IndexStore, int, str]:
    """A repo whose chunks span comment, config_key, function, variable, import.

    `guide.md` holds a fenced bash block whose one line is both a
    command and a trailing comment, so the index contains two anonymous
    chunks sharing a location — the shape that collides when a target is
    addressed by location alone.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "guide.md").write_text(
        """\
# Guide

```bash
uv run dvc repro    # run every stage end-to-end
```
"""
    )
    (repo_path / "lib.py").write_text(
        "import os\n\n"
        "MAX_RETRIES = 3\n\n"
        "# tune the backoff multiplier for flaky networks\n\n"
        "def connect(host):\n"
        '    """Open a connection to the given host."""\n'
        "    return os.environ.get(host)\n"
    )
    (repo_path / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 88\ntarget-version = "py313"\n'
    )
    repo = pygit2.init_repository(str(repo_path), bare=False)
    index = repo.index
    index.add_all()
    index.write()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", index.write_tree(), [])

    from rbtr.index.build import build_index  # deferred: heavy native libs

    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(str(repo_path.resolve()))
    head = str(repo.head.target)
    build_index(repo.workdir, head, store)
    return store, repo_id, head


def test_extract_covers_measurable_kinds_and_excludes_import(
    mixed_kind_index: tuple[IndexStore, int, str],
) -> None:
    """Every measurable target kind yields queries; excluded kinds yield none.

    The regression this whole change is about: a searchable kind
    (`comment`, `config_key`, `variable`) must be measured, and an
    excluded one (`import`) must not — so a dropped kind can never
    again pass unnoticed.
    """
    store, repo_id, sha = mixed_kind_index
    queries, _, _ = extract_queries(store, "test", repo_id, sha, min_per_language=1)
    kinds = set(queries["symbol_kind"].cast(pl.String).to_list())

    assert {"comment", "config_key", "function", "variable"} <= kinds
    assert "import" not in kinds


def test_generated_queries_target_real_chunks(
    mixed_kind_index: tuple[IndexStore, int, str],
) -> None:
    """Every query's target identity resolves to an indexed chunk.

    Guarantees the generated queries are scoreable — including the
    anonymous chunks (comment) matched by location, not name.
    """
    store, repo_id, sha = mixed_kind_index
    queries, _, _ = extract_queries(store, "test", repo_id, sha, min_per_language=1)
    chunk_ids = {
        (c.file_path, c.scope, c.name, c.line_start, c.line_end, c.kind.value)
        for c in store.get_chunks(sha, repo_id=repo_id)
    }
    for row in queries.iter_rows(named=True):
        assert tuple(row[c] for c in IDENTITY_COLUMNS) in chunk_ids


def test_two_chunks_on_one_line_are_two_targets(
    mixed_kind_index: tuple[IndexStore, int, str],
) -> None:
    """A shared location yields a query each, not one colliding key.

    A fenced bash line is both an anonymous `doc_section` (the command)
    and an anonymous `comment` (its trailing comment), at one
    `line_start`. Each is separately searchable, so each is its own
    target and the span and kind are what tell them apart.
    """
    store, repo_id, sha = mixed_kind_index
    queries, _, _ = extract_queries(store, "test", repo_id, sha, min_per_language=1)

    fenced = queries.filter(
        (pl.col("file_path") == "guide.md")
        & (pl.col("line_start") == 4)
        & (pl.col("provenance") == "body")
    )
    assert set(fenced["symbol_kind"].cast(pl.String).to_list()) == {"comment", "doc_section"}
