"""Shipped test tools for rbtr, its language plugins and its other packages.

What a test suite (in- or out-of-tree) can't easily re-derive: a
pydantic-aware snapshot serialiser, a sample project built as rbtr
builds a repository, readable renderings of what a build holds, and an
isolated on-disk index.

Also a pytest plugin (the `pytest11` entry point on `rbtr`): it provides the
`snapshot_json`, `sample_index`, `sample_ref` and `isolated_db` fixtures, so
every test suite gets them with no conftest boilerplate.

Always shipped (a normal module); it imports `syrupy` and `pytest`, so it is
only *importable* with the `rbtr[test]` extra installed — exactly the test
context that uses it. `render_edges` and `uncovered_lines` are projections
feeding snapshot assertions; keep them projections — do not fold them into
`assert_*` helpers, which would hide syrupy's snapshot comparison.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
import pygit2
import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

from rbtr.daemon.status import is_pid_alive, read_status
from rbtr.domain.models import Chunk, Chunks

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData

    from rbtr.domain.models import SnapshotRef
    from rbtr.index.index import Index
    from rbtr.index.results import ChunkResultRow, EdgeResultRow
    from rbtr.languages.edges import EdgeChunkRow


def edge_rows(chunks: list[Chunk]) -> dy.DataFrame[EdgeChunkRow]:
    """*chunks*, in order, as `infer_import_edges` reads them.

    The build selects the same columns from what it read back from the
    index; a plugin's tests take them from what extraction yielded.
    """
    # deferred: `rbtr.languages` imports `rbtr.config`, which this plugin
    # must not load before pytest-env applies (see `isolated_db`).
    from rbtr.index.staging import ChunkStagingRow
    from rbtr.languages.edges import EdgeChunkRow

    role = pl.Struct({"kind": pl.String, "name": pl.String, "meta": ChunkStagingRow.metadata.dtype})
    return (
        pl.DataFrame(Chunks.dump_python(chunks, mode="json"), schema_overrides={"role": role})
        .with_columns(pl.col("role").struct.unnest())
        .with_columns(
            pl.col("name").fill_null(""),
            pl.col("meta").struct.unnest(),
            line_start=pl.col("span").struct.field("start"),
        )
        .pipe(EdgeChunkRow.validate, cast=True)
    )


def render_edges(
    edges: dy.DataFrame[EdgeResultRow], chunks: dy.DataFrame[ChunkResultRow]
) -> list[str]:
    """Render edges as readable, sorted `src -> tgt [kind]` lines.

    Joins each edge's endpoint ids back to their chunks so a snapshot shows
    *which* chunks are linked (raw ids are opaque hashes).
    """
    labels = chunks.select("id", label=pl.format("{}::{}", "file_path", "name")).unique(
        "id", keep="last"
    )
    return (
        edges.join(
            labels.rename({"id": "source_id", "label": "source_label"}), on="source_id", how="left"
        )
        .join(
            labels.rename({"id": "target_id", "label": "target_label"}), on="target_id", how="left"
        )
        .select(
            pl.format(
                "{} -> {} [{}]", "source_label", "target_label", pl.col("kind").cast(pl.String)
            )
        )
        .to_series()
        .sort()
        .to_list()
    )


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


def uncovered_lines(sample_dir: Path, chunks: dy.DataFrame[ChunkResultRow]) -> list[str]:
    """Each non-blank line of *sample_dir*'s files that no chunk spans, as `path:line: text`."""
    files = sorted(path for path in sample_dir.rglob("*") if path.is_file())
    lines = pl.DataFrame(
        [
            (str(path.relative_to(sample_dir)), number, text)
            for path in files
            for number, text in enumerate(path.read_text().splitlines(), start=1)
        ],
        schema={"file_path": pl.String, "line": pl.Int32, "text": pl.String},
        orient="row",
    ).filter(pl.col("text").str.strip_chars() != "")
    spanned = chunks.select(
        "file_path", line=pl.int_ranges("line_start", pl.col("line_end") + 1, dtype=pl.Int32)
    ).explode("line")
    return (
        lines.join(spanned, on=["file_path", "line"], how="anti")
        .sort("file_path", "line")
        .select(pl.format("{}:{}: {}", "file_path", "line", "text"))
        .to_series()
        .to_list()
    )


@pytest.fixture(scope="module")
def sample_index(sample_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Generator[Index]:
    """*sample_dir*'s files, committed to a fresh repository and built into an in-memory index.

    A plugin's test module defines `sample_dir`, module-scoped: the sample
    project its tests read. The build is the one `rbtr` runs on a
    repository, so the index holds what a user's index would.
    """
    # deferred: `rbtr.index` imports `rbtr.config`, which this plugin
    # must not load before pytest-env applies (see `isolated_db`).
    from rbtr.git import normalise_repo_path
    from rbtr.index.build import build_index
    from rbtr.index.index import Index

    workdir = tmp_path_factory.mktemp("sample") / sample_dir.name
    shutil.copytree(sample_dir, workdir)
    repo = pygit2.init_repository(str(workdir), bare=False, initial_head="main")
    repo.index.add_all()
    repo.index.write()
    author = pygit2.Signature("rbtr", "rbtr@example.com")
    sha = repo.create_commit("HEAD", author, author, "sample", repo.index.write_tree(), [])
    index = Index(writable=True)
    build_index(normalise_repo_path(str(workdir)), str(sha), index)
    yield index
    index.close()


@pytest.fixture(scope="module")
def sample_ref(sample_index: Index) -> SnapshotRef:
    """The snapshot `sample_index` built."""
    [repo] = sample_index.repos()
    [snapshot] = sample_index.indexed_snapshots(repo)
    return repo.at(snapshot.snapshot_sha)


def _terminate_daemon_for(runtime_dir: Path, *, timeout: float = 5.0) -> None:
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
    _terminate_daemon_for(Config(data_dir=data_dir).runtime_dir)
