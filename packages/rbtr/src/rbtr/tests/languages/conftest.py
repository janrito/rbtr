"""Shared fixtures and grammar helpers for language tests.

Exposes `language_manager` as a session-scoped fixture over
the production `LanguageManager` singleton.  Tests take it as
a parameter wherever they need grammar / query / extraction
lookup.

Also provides the per-language sample-file plumbing: `load_sample`
reads a committed sample from `samples/`, and `snapshot_json`
is a syrupy fixture that serialises chunks through pydantic's
own JSON serialisation (`PydanticSnapshotExtension`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

from rbtr.git import FileEntry
from rbtr.index.models import Chunk, Edge
from rbtr.index.orchestrator import _extract_file, extract_query
from rbtr.languages import LanguageManager, get_manager

if TYPE_CHECKING:
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData

SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.fixture(scope="session")
def language_manager() -> LanguageManager:
    """The production `LanguageManager` singleton."""
    return get_manager()


def render_edges(edges: list[Edge], chunks: list[Chunk]) -> list[str]:
    """Render edges as readable `src -> tgt [kind]` lines, sorted.

    A pure projection: joins each edge's endpoint ids back to their chunks
    so the snapshot shows *which* chunks are linked (raw ids are opaque
    hashes). Both endpoints are always in *chunks* — `infer_import_edges`
    only emits resolved edges. Sorted for deterministic snapshots.
    """
    by_id = {c.id: c for c in chunks}

    def label(chunk_id: str) -> str:
        c = by_id.get(chunk_id)
        return f"{c.file_path}::{c.name}" if c is not None else f"<{chunk_id}>"

    return sorted(f"{label(e.source_id)} -> {label(e.target_id)} [{e.kind.value}]" for e in edges)


def sample_path(lang: str) -> Path:
    """Return the path to a flat committed sample file for *lang*.

    Looks up `samples/<lang>.*`; exactly one file must match. Used by the
    flat dialect fixtures (`sql_postgres` etc.); the per-language `sample`
    family uses `load_project` (a directory) instead.
    """
    matches = sorted(SAMPLES_DIR.glob(f"{lang}.*"))
    assert len(matches) == 1, f"expected exactly one sample for {lang!r}, found {matches}"
    return matches[0]


def load_sample(lang: str) -> str:
    """Return the text of a flat committed sample file for *lang*."""
    return sample_path(lang).read_text()


def load_project(lang: str) -> list[tuple[str, str]]:
    """Return every file of the `samples/<lang>/` mini-project.

    Each language sample is a directory — a small project whose files may
    import one another (and may span languages, e.g. an html page with its
    js/css). Returns `(project-relative path, text)` for every file, sorted
    for determinism (chunk ids hash the path).
    """
    proj = SAMPLES_DIR / lang
    assert proj.is_dir(), f"no sample project dir for {lang!r}"
    files = sorted(p for p in proj.rglob("*") if p.is_file())
    assert files, f"empty sample project for {lang!r}"
    return [(str(p.relative_to(proj)), p.read_text()) for p in files]


class PydanticSnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot pydantic models via their own JSON serialisation.

    One `.json` file per test case. `serialize` delegates to
    pydantic through `TypeAdapter`, so both a bare model and a
    `list[Chunk]` round-trip through the real `model_dump_json`
    path (enums as values, nested models as objects).
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
def snapshot_json(snapshot: object) -> object:
    """syrupy snapshot fixture using `PydanticSnapshotExtension`."""
    return snapshot.use_extension(PydanticSnapshotExtension)  # type: ignore[attr-defined]  # syrupy fixture is untyped


def extract_chunks(
    lang: str,
    source: str,
    file_path: str = "",
    *,
    no_leading_attachment: bool = False,
) -> list[Chunk]:
    """Run the real extraction pipeline for *lang* on *source*.

    Delegates to the orchestrator's per-file dispatch (`_extract_file`)
    so tests exercise the production path, not a copy of it. Calls
    `get_manager()` inline because cases invoked from inside
    `@parametrize_with_cases` cannot consume a fixture.

    `no_leading_attachment` forces an empty `doc_comment_node_types`
    (query path only) to probe leading-comment fall-back behaviour; it
    routes through the same production `extract_query`.
    """
    manager = get_manager()
    reg = manager.get_registration(lang)
    assert reg is not None, f"no registration for {lang}"

    # `sorted` keeps the derived path deterministic across processes:
    # `reg.extensions` is a frozenset, so `next(iter(...))` would vary
    # with PYTHONHASHSEED, and the chunk id hashes the file path.
    ext = next(iter(sorted(reg.extensions)), ".txt").lstrip(".")
    path = file_path or f"test.{ext}"

    if no_leading_attachment:
        return list(
            extract_query(lang, path, "sha1", source.encode(), doc_comment_node_types=frozenset())
        )

    entry = FileEntry(path=path, blob_sha="sha1", content=source.encode())
    return list(_extract_file(entry, lang, reg))
