"""CLI-path smoke test for rbtr-eval.

One happy-path test drives `rbtr-eval extract` end-to-end
through the real argv parser and writes parquet files into
a tmp dir.  Catches the wiring bugs the unit tests can't:
pydantic-settings subcommand shape drift, schema registration
drift, CLI option name drift.

Not covered here: measure / index / tune.  Those need a
running rbtr daemon and the embedding model (roughly 15 s
cold start); they're smoke-tested by running `just eval`
against the committed dev config.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pygit2
import pytest
from pydantic_settings import CliApp

from rbtr.index.store import IndexStore
from rbtr_eval.cli import RbtrEval
from rbtr_eval.schemas import QueryRow, RepoHeader


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A git repo with one python file carrying three documented symbols.

    One function, one class, and one method inside the class
    so the extract exercises the scope / kind axes of the
    `QueryRow` schema.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.py").write_text(
        '''
def greet(name: str) -> str:
    """Say hello to someone nicely."""
    return f"Hello, {name}"


class Greeter:
    """A friendly greeter that welcomes users warmly."""

    def welcome(self) -> str:
        """Return the default welcome message."""
        return "welcome"
'''
    )
    # Trusted args: `git` on PATH is the system git binary.
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)  # noqa: S603
    subprocess.run([*git, "config", "user.email", "t@t"], check=True)  # noqa: S603
    subprocess.run([*git, "config", "user.name", "t"], check=True)  # noqa: S603
    subprocess.run([*git, "add", "."], check=True)  # noqa: S603
    subprocess.run([*git, "commit", "-qm", "init"], check=True)  # noqa: S603
    return repo


def test_extract_writes_validated_parquet_files(tmp_path: Path, tiny_repo: Path) -> None:
    """`rbtr-eval extract` writes per-kind query files and a header.

    The extract reads symbols from the index, generates queries,
    and writes one parquet per provenance plus a header.
    """
    out_dir = tmp_path / "out"
    headers_dir = tmp_path / "headers"

    # Build a file-backed index so ExtractCmd can open it.
    # Register with a path whose last component matches the
    # slug so extract can find it via Path(path).name.
    db_dir = tmp_path / "index"
    db_dir.mkdir()
    file_store = IndexStore(str(db_dir / "index.duckdb"), writable=True)
    with file_store.session() as ws:
        repo_id = ws.register_repo(str(tmp_path / "tiny"))
    repo = pygit2.Repository(str(tiny_repo))
    head = str(repo.head.target)

    from rbtr.index.orchestrator import build_index  # deferred: heavy native libs

    build_index(repo.workdir, head, file_store, repo_id=repo_id)

    argv = [
        "extract",
        "--slug",
        "tiny",
        "--data-dir",
        str(db_dir),
        "--out-dir",
        str(out_dir),
        "--headers-dir",
        str(headers_dir),
        "--seed",
        "0",
        "--queries-per-cell",
        "200",
        "--min-per-language",
        "1",
    ]

    CliApp.run(RbtrEval, cli_args=argv)

    # Single queries parquet.
    queries = pl.read_parquet(out_dir / "tiny.parquet").pipe(QueryRow.validate, cast=True)
    header = pl.read_parquet(headers_dir / "tiny.parquet").pipe(RepoHeader.validate, cast=True)

    # 3 symbols: greet (function), Greeter (class), welcome (method).
    # All documented → each gets name + body + docstring = up to 9 queries.
    assert queries.height >= 3  # at least one query per symbol
    assert "name" in queries["provenance"].unique().to_list()
    assert sorted(queries["symbol_kind"].unique().cast(pl.String).to_list()) == [
        "class",
        "function",
        "method",
    ]
    assert header.height == 1
    assert header["slug"][0] == "tiny"
