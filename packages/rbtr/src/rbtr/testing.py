"""Shipped test tools for rbtr and its language plugins.

Two things a test suite (in- or out-of-tree) can't easily re-derive: a
pydantic-aware snapshot serialiser and a readable edge renderer, both keyed
off `rbtr.domain.models`. Everything else — running extraction, loading
samples — is plain setup that belongs in each suite's own fixtures, calling
`rbtr.languages.extract.extract_file` directly.

Also a pytest plugin (the `pytest11` entry point on `rbtr`): it provides the
`snapshot_json` fixture, so every plugin test suite gets it with no conftest
boilerplate.

Always shipped (a normal module); it imports `syrupy` and `pytest`, so it is
only *importable* with the `rbtr[test]` extra installed — exactly the test
context that uses it. `render_edges` is a projection feeding a snapshot
assertion; keep it a projection — do not fold it into an `assert_*` helper,
which would hide syrupy's snapshot comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from tree_sitter import Parser

from rbtr.domain.models import Chunk, Edge

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData
    from tree_sitter import Language


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
    plugin package needs no conftest to snapshot `Chunk`/`Edge` output.
    """
    return snapshot.use_extension(PydanticSnapshotExtension)


class DroppedBlocks(AssertionError):
    """Top-level blocks of a chunker-owned document that no chunk reaches."""

    def __init__(self, file_path: str, dropped: list[tuple[str, int]]) -> None:
        super().__init__(f"{file_path}: no chunk reaches {dropped}")


def assert_document_fully_chunked(
    file_path: str,
    text: str,
    chunks: list[Chunk],
    grammar: Language,
) -> None:
    """Every top-level block of a chunker-owned document reaches a chunk.

    A chunker partitions a whole document, unlike a query, which selects
    the definitions out of one — so for a chunker every top-level block is
    content and none may be dropped. A block counts as reached when some
    chunk overlaps its lines, which is what an injected `<script>`'s
    chunks do to the element holding them.

    Call it from a plugin whose extraction is a chunker: prose (markdown,
    rst) and single-file components (svelte, vue).
    """
    spans = [(c.line_start, c.line_end) for c in chunks if c.file_path == file_path]
    root = Parser(grammar).parse(text.encode()).root_node
    dropped = [
        (node.type, node.start_point[0] + 1)
        for node in root.children
        if node.is_named
        and not any(
            start <= node.end_point[0] + 1 and node.start_point[0] + 1 <= end
            for start, end in spans
        )
    ]
    if dropped:
        # Raised rather than asserted: this ships, and `-O` strips an assert.
        raise DroppedBlocks(file_path, dropped)
