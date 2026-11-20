"""A CLI command run with no daemon migrates the index when it opens it."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pygit2
from pytest_mock import MockerFixture

from rbtr.cli import Watch
from rbtr.config import config
from rbtr.index.constants import SCHEMA_VERSION
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit


def test_a_command_with_no_daemon_migrates_the_index_then_runs(
    git_repo: pygit2.Repository,
    repo_path: str,
    isolated_db: Path,
    chains: Path,
    mocker: MockerFixture,
) -> None:
    """`rbtr watch --no-daemon` brings an older index forward on its open, then builds.

    The project file is switched for a migrated copy the registry names,
    and the commit made after the index was written is indexed.
    """
    Watch(refs=["HEAD"], repo_path=repo_path, daemon=False, allow_missing_plugins=False).cli_cmd()
    with duckdb.connect(str(config.db_path), read_only=True) as con:
        [(old_file,)] = con.sql("SELECT file FROM projects").fetchall()
    (chains / "project").mkdir()
    (chains / "project" / f"{SCHEMA_VERSION}__2099.1.0.sql").write_text(
        "CREATE TABLE added_note (x INT);"
    )
    mocker.patch("rbtr.index.store.SCHEMA_VERSION", "2099.1.0")
    head = make_commit(
        git_repo,
        {"src/added.py": b"def added():\n    return 1\n"},
        parents=[git_repo.head.peel(pygit2.Commit).id],
    )

    Watch(refs=["HEAD"], repo_path=repo_path, daemon=False, allow_missing_plugins=False).cli_cmd()

    with duckdb.connect(str(config.db_path), read_only=True) as con:
        [(new_file,)] = con.sql("SELECT file FROM projects").fetchall()
    assert new_file != old_file
    with duckdb.connect(str(config.index_dir / new_file), read_only=True) as con:
        assert con.sql("SELECT value FROM meta WHERE key = 'schema_version'").fetchall() == [
            ("2099.1.0",)
        ]
        assert con.sql("SELECT 1 FROM information_schema.tables WHERE table_name = 'added_note'")
    index = Index.from_config()
    try:
        assert index.has_indexed(at=index.repo(repo_path).at(str(head)))
    finally:
        index.close()
