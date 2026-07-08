"""Reusable tools for language-plugin tests.

Two things a plugin test suite (in- or out-of-tree) can't easily
re-derive: a pydantic-aware snapshot serialiser and a readable edge
renderer. Everything else — running extraction, loading samples — is
plain setup that belongs in each suite's own fixtures, calling the
production `rbtr.index.orchestrator.extract_file` directly.

Also a pytest plugin (the `pytest11` entry point on `rbtr`): it provides the
`snapshot_json` fixture, so every plugin test suite gets it with no conftest
boilerplate.

Public API, but shipped only under the `rbtr[test]` extra (it imports `syrupy`
and `pytest`), so a plain `import rbtr` never pulls it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

from rbtr.index.models import Chunk, Edge

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData


def render_edges(edges: list[Edge], chunks: list[Chunk]) -> list[str]:
    """Render edges as readable, sorted `src -> tgt [kind]` lines.

    Joins each edge's endpoint ids back to their chunks so a snapshot shows
    *which* chunks are linked (raw ids are opaque hashes).
    """
    by_id = {c.id: c for c in chunks}

    def label(chunk_id: str) -> str:
        c = by_id.get(chunk_id)
        return f"{c.file_path}::{c.name}" if c is not None else f"<{chunk_id}>"

    return sorted(f"{label(e.source_id)} -> {label(e.target_id)} [{e.kind.value}]" for e in edges)


class PydanticSnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot pydantic models via their own JSON serialisation.

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
    plugin package needs no conftest to snapshot `Chunk`/`Edge` output.
    """
    return snapshot.use_extension(PydanticSnapshotExtension)
