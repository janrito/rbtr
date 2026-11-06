"""Shipped test tools for rbtr, its language plugins and its other packages.

What a test suite (in- or out-of-tree) can't easily re-derive: a
pydantic-aware snapshot serialiser, a readable edge renderer, and an
isolated on-disk index. Other setup —
running extraction, loading samples — belongs in each suite's own fixtures,
calling `rbtr.languages.extract.extract_file` directly.

Also a pytest plugin (the `pytest11` entry point on `rbtr`): it provides the
`snapshot_json` and `isolated_db` fixtures, so every test suite
gets them with no conftest boilerplate.

Always shipped (a normal module); it imports `syrupy` and `pytest`, so it is
only *importable* with the `rbtr[test]` extra installed — exactly the test
context that uses it. `render_edges` is a projection feeding a snapshot
assertion; keep it a projection — do not fold it into an `assert_*` helper,
which would hide syrupy's snapshot comparison.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from tree_sitter import Parser

from rbtr.daemon.status import is_pid_alive, read_status
from rbtr.domain.models import Chunk, Chunks

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData
    from tree_sitter import Node

    from rbtr.languages.edges import EdgeChunkRow, InferredEdgeRow


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


def render_edges(edges: dy.DataFrame[InferredEdgeRow], chunks: list[Chunk]) -> list[str]:
    """Render edges as readable, sorted `src -> tgt [kind]` lines.

    Joins each edge's endpoint ids back to their chunks so a snapshot shows
    *which* chunks are linked (raw ids are opaque hashes).
    """
    labels = pl.DataFrame(
        {"id": [c.id for c in chunks], "label": [f"{c.file_path}::{c.name}" for c in chunks]},
        schema={"id": pl.String, "label": pl.String},
    ).unique("id", keep="last")
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


class DroppedBlocks(AssertionError):
    """Syntax in a file that no chunk reaches."""

    def __init__(self, file_path: str, dropped: list[tuple[str, int]]) -> None:
        super().__init__(f"{file_path}: no chunk reaches {dropped}")


def assert_fully_chunked(file_path: str, text: str, chunks: list[Chunk], language: str) -> None:
    """Every named node of a file's syntax, read as *language*, reaches a chunk.

    A node is reached when some chunk overlaps its lines. A node whose
    lines chunks cover only in part is looked into, and each of its
    named children must be reached the same way: an `if` holding an
    import leaves its own header line to be reached. A node holding a
    block the language's injection query delegates (an HTML `<script>`
    element) counts as reached, as extraction treats it.
    """
    # deferred: `rbtr.languages` imports `rbtr.config`, which this plugin
    # must not load before pytest-env applies (see `isolated_db`).
    from rbtr.languages.chunks import last_line
    from rbtr.languages.manager import get_manager
    from rbtr.languages.treesitter import injection_holders

    manager = get_manager()
    registration = manager.get_registration(language)
    grammar = manager.grammar(language)
    if registration is None or grammar is None:
        msg = f"{language} has no grammar to read {file_path} with"
        raise ValueError(msg)
    tree = Parser(grammar).parse(text.encode())
    spans = [(c.line_start, c.line_end) for c in chunks if c.file_path == file_path] + [
        (node.start_point[0] + 1, last_line(node)) for node in injection_holders(registration, tree)
    ]
    covered = {line for start, end in spans for line in range(start, end + 1)}

    def dropped(node: Node) -> Iterator[tuple[str, int]]:
        for child in node.named_children:
            lines = set(range(child.start_point[0] + 1, last_line(child) + 1))
            if not lines & covered:
                yield (child.type, child.start_point[0] + 1)
            elif not lines <= covered:
                yield from dropped(child)

    missing = list(dropped(tree.root_node))
    if missing:
        # Raised rather than asserted: this ships, and `-O` strips an assert.
        raise DroppedBlocks(file_path, missing)


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
