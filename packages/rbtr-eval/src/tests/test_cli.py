"""CLI-path smoke test for rbtr-eval.

One happy-path test drives `rbtr-eval extract` end-to-end
through the real argv parser and writes two parquet files
into a tmp dir.  Catches the wiring bugs the unit tests
can't: pydantic-settings subcommand shape drift, schema
registration drift, CLI option name drift.

Not covered here: measure / index / tune.  Those need a
running rbtr daemon and the embedding model (roughly 15 s
cold start); they're smoke-tested by running `just eval`
against the committed dev config.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest
from pydantic_settings import CliApp

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
    """`rbtr-eval extract` writes a queries file and a header file.

    Both files validate against their dataframely schemas; the
    queries frame has one row per documented symbol in the
    tiny repo (three); the header frame is a single row whose
    `n_sampled` matches the queries frame's height.
    """
    out_dir = tmp_path / "out"
    argv = [
        "extract",
        "--slug",
        "tiny",
        "--repo-path",
        str(tiny_repo),
        "--out-dir",
        str(out_dir),
        "--seed",
        "0",
        "--sample-cap",
        "10",
    ]

    CliApp.run(RbtrEval, cli_args=argv)

    queries = pl.read_parquet(out_dir / "tiny.queries.parquet").pipe(QueryRow.validate, cast=True)
    header = pl.read_parquet(out_dir / "tiny.header.parquet").pipe(RepoHeader.validate, cast=True)

    assert queries.height == 3
    assert sorted(queries["symbol_kind"].cast(pl.String).to_list()) == [
        "class",
        "function",
        "method",
    ]
    assert header.height == 1
    assert header["slug"][0] == "tiny"
    assert header["n_sampled"][0] == 3
    assert header["n_documented"][0] == 3
