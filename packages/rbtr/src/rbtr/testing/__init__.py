"""Shipped test tools for rbtr, its language plugins and its other packages.

A pytest plugin (the `pytest11` entry point on `rbtr`), so every test suite
gets its fixtures with no conftest: `indexed` builds a test's files into an
index, `snapshot_json` snapshots pydantic values, and `isolated_db` gives a
test its own on-disk index.

A language test reads what it built through `rbtr.testing.reads`
(`outline`, `read`) and checks it with `rbtr.testing.assertions`
(`assert_reached`, `assert_every_line_in_a_chunk`), so every test reads and
checks the same way.

Always shipped (a normal package); it imports `syrupy` and `pytest`, so it
is only *importable* with the `rbtr[test]` extra installed — exactly the
test context that uses it.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2
import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

from rbtr.daemon.status import is_pid_alive, read_status

# A check's asserts report pytest's diff only when rewritten, and pytest
# rewrites a plugin's modules only when it finds them in an installed wheel's
# file list (an editable install lists none), so the module is named here,
# before anything imports it.
pytest.register_assert_rewrite("rbtr.testing.assertions")

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData

    from rbtr.index.index import Index


class PydanticSnapshotExtension(SingleFileSnapshotExtension):
    """FileSnapshot pydantic models via their own JSON serialisation.

    One `.json` file per case; `serialize` delegates to pydantic through
    `TypeAdapter`, so a `list[Chunk]` round-trips through the real
    `model_dump_json` path (enums as values, nested models as objects).
    """

    _write_mode = WriteMode.TEXT
    file_extension = "json"

    def serialize(
        self,
        data: SerializableData,
        *,
        exclude: PropertyFilter | None = None,
        include: PropertyFilter | None = None,
        matcher: PropertyMatcher | None = None,
    ) -> SerializedData:
        return TypeAdapter(type(data)).dump_json(data, indent=2).decode() + "\n"


@pytest.fixture
def snapshot_json(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """syrupy snapshot fixture using `PydanticSnapshotExtension`.

    Auto-provided to every test suite via the `pytest11` entry point, so a
    plugin package needs no conftest to snapshot `Chunk` and edge output.
    """
    return snapshot.use_extension(PydanticSnapshotExtension)


type Indexed = Callable[[Mapping[str, str] | Path], AbstractContextManager[Index]]
"""What the `indexed` fixture gives: `with indexed(files) as index:`, *files* a mapping or a sample directory."""


# ----------------------------------------------------------------------------
# A PURPOSEFUL EXCEPTION TO "NO FACTORY FIXTURES".
#
# The testing rules (`.agents/skills/rbtr-testing/SKILL.md`, "No factory
# fixtures") say a fixture returns a value, never a callable that makes one.
# `indexed` breaks that rule on purpose, and it is the only fixture that may.
#
# It exists for language testing: core's tests and every language plugin's
# build source files and read what rbtr answers about them. Those tests take
# their files from pytest-cases cases, and pytest-cases evaluates a case's
# values only for the test function itself, so a fixture that depends on
# `files` would receive an unevaluated placeholder. The files therefore reach
# the build in the test body, through this factory (pytest's documented
# "Factories as fixtures" pattern), and the index lives only inside the
# `with` block. Shipping it in this pytest plugin gives every plugin package
# the same build without code of its own.
#
# Do not copy this pattern into other fixtures.
# ----------------------------------------------------------------------------
@pytest.fixture
def indexed(tmp_path: Path) -> Indexed:
    """Index files in a repository in `tmp_path`: `with indexed(files) as index:`.

    Writes *files* (`{path: text}`, or every file under a sample directory,
    by its path within it) under `tmp_path`, commits them on top
    of its branch (the engine reads a repository by its path), builds the
    commit into an in-memory index with the languages the manager holds
    (every installed one, or those `limited_to` names), and closes it when
    the block ends. Read it with `rbtr.testing.reads` and check it with
    `rbtr.testing.assertions`.
    """
    # deferred: `rbtr.index` imports `rbtr.config` (see `isolated_db`).
    from rbtr.index.build import build_index
    from rbtr.index.index import Index

    @contextmanager
    def build(files: Mapping[str, str] | Path) -> Iterator[Index]:
        written = (
            {
                str(path.relative_to(files)): path.read_text()
                for path in sorted(files.rglob("*"))
                if path.is_file()
            }
            if isinstance(files, Path)
            else files
        )
        for path, text in written.items():
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / path).write_text(text)
        repo = pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")
        repo.index.add_all()
        repo.index.write()
        author = pygit2.Signature("rbtr", "rbtr@example.com")
        parents = [] if repo.head_is_unborn else [repo.head.target]
        tree = repo.index.write_tree()
        sha = str(repo.create_commit("HEAD", author, author, "files", tree, parents))
        index = Index(writable=True)
        try:
            build_index(str(tmp_path), sha, index)
            yield index
        finally:
            index.close()

    return build


def terminate_daemon_for(runtime_dir: Path, *, timeout: float = 5.0) -> None:
    """Stop a subprocess daemon whose status file is in `runtime_dir`.

    Reads the daemon's own status file for the pid, sends `SIGTERM`,
    and waits for the process to go.  A daemon that is already gone, or
    that never wrote a status file, is a no-op.
    """
    status = read_status(runtime_dir)
    if status is None or not is_pid_alive(status.pid):
        return
    os.kill(status.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_pid_alive(status.pid):
            return
        time.sleep(0.05)


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """Give a test its own on-disk index DB, and outlive no daemon.

    Subprocess CLI smoke tests can't share an in-memory store, so
    they read the on-disk DB at `config.db_path`.  A single shared
    file makes xdist workers collide on DuckDB's write lock and
    lets one test's repos leak into another's cross-repo search.
    Repoint both the subprocess (via the `RBTR_DATA_DIR` env it
    inherits) and the in-process `config` singleton (used by
    `Index.from_config`) at a per-test directory.

    Teardown kills any daemon still serving this data dir.  The
    daemon smoke tests stop their own daemon in a `finally`, but that
    does not run when the run is interrupted part-way through a test,
    and an orphaned daemon then idles indefinitely on a `tmp_path`
    that no later run will reuse.
    """
    # deferred: pytest imports this plugin before pytest-env applies
    # `[tool.pytest_env]`; importing `rbtr.config` at the top would build
    # the `config` singleton from the user's real settings in a
    # single-process run.  pytest-django defers Django's settings the same
    # way.
    from rbtr.config import Config, config

    data_dir = tmp_path / "index"
    data_dir.mkdir()
    monkeypatch.setenv("RBTR_DATA_DIR", str(data_dir))
    monkeypatch.setattr(config, "data_dir", data_dir)
    yield data_dir
    # Fixture finalisers still run when pytest unwinds on Ctrl-C; a
    # SIGKILLed run skips them, and nothing in-process can cover that.
    terminate_daemon_for(Config(data_dir=data_dir).runtime_dir)
