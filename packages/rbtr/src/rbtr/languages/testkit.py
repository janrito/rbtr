"""Reusable test harness for language plugins.

A language plugin ships as its own distribution (`rbtr-lang-<lang>`)
with its own sample project and golden snapshots. This module is the
shared *driver* those test suites import: fixtures, sample loaders, the
extraction entry point, and the snapshot serialiser. It is public,
supported API — plugin test suites depend on it.

It ships with `rbtr` (it lives under `rbtr/languages/`, outside the
excluded `tests/` tree), but its dependencies — `syrupy`, `pytest`,
`pytest-cases` — are only pulled in by the `rbtr[testkit]` extra. A
plain `import rbtr` never imports this module, so those libraries stay
out of the runtime install.

A plugin's `conftest.py` re-exposes the fixtures::

    from rbtr.languages.testkit import language_manager, snapshot_json

and its test modules call the helpers directly, passing their own
`samples/` directory to the loaders (the loaders are not tied to any
one package's layout).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from tree_sitter import Parser

from rbtr.git import FileEntry
from rbtr.index.edges import build_resolution_map, infer_import_edges
from rbtr.index.models import Chunk, ChunkKind, Edge, ImportMeta
from rbtr.index.orchestrator import _extract_file, extract_query
from rbtr.languages import LanguageManager, get_manager

if TYPE_CHECKING:
    from syrupy.types import PropertyFilter, PropertyMatcher, SerializableData, SerializedData

type ProjectFiles = list[tuple[str, str]]
type SampleData = tuple[str, set[ChunkKind], ProjectFiles, list[Chunk], list[Edge]]


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


def sample_path(samples_dir: Path, lang: str) -> Path:
    """Return the path to a flat committed sample file for *lang*.

    Looks up `<samples_dir>/<lang>.*`; exactly one file must match. Used by
    the flat dialect fixtures (`sql_postgres` etc.); the per-language
    `sample` family uses `load_project` (a directory) instead.
    """
    matches = sorted(samples_dir.glob(f"{lang}.*"))
    if len(matches) != 1:
        msg = f"expected exactly one sample for {lang!r}, found {matches}"
        raise ValueError(msg)
    return matches[0]


def load_sample(samples_dir: Path, lang: str) -> str:
    """Return the text of a flat committed sample file for *lang*."""
    return sample_path(samples_dir, lang).read_text()


def load_project(samples_dir: Path, lang: str) -> ProjectFiles:
    """Return every file of the `<samples_dir>/<lang>/` mini-project.

    Each language sample is a directory — a small project whose files may
    import one another (and may span languages, e.g. an html page with its
    js/css). Returns `(project-relative path, text)` for every file, sorted
    for determinism (chunk ids hash the path).
    """
    proj = samples_dir / lang
    if not proj.is_dir():
        msg = f"no sample project dir for {lang!r}"
        raise NotADirectoryError(msg)
    files = sorted(p for p in proj.rglob("*") if p.is_file())
    if not files:
        msg = f"empty sample project for {lang!r}"
        raise ValueError(msg)
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
    if reg is None:
        msg = f"no registration for {lang}"
        raise ValueError(msg)

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


def build_sample_data(samples_dir: Path, lang: str, expected_kinds: set[ChunkKind]) -> SampleData:
    """Extract a language's `samples/<lang>/` mini-project into `SampleData`.

    The body of each plugin's parametrised `sample` fixture. Extracts every
    project file using *its own* detected language (a project may span
    languages, e.g. an html page plus its js/css), then infers import edges
    across the project. The project-relative path is each file's `file_path`,
    so snapshots and path-derived chunk ids are deterministic.
    """
    manager = get_manager()
    files = load_project(samples_dir, lang)
    chunks: list[Chunk] = []
    for relpath, text in files:
        file_lang = manager.detect_language(relpath) or lang
        chunks.extend(extract_chunks(file_lang, text, file_path=relpath))
    repo_files = {relpath for relpath, _ in files}
    edges = infer_import_edges(chunks, repo_files, build_resolution_map(manager))
    return lang, expected_kinds, files, chunks, edges


# ── Shared sample assertions ─────────────────────────────────────────
#
# Behavioural checks shared by every plugin's `test_samples.py`. They use
# `pytest.fail` rather than bare `assert` because this module ships as
# product source (where `assert` is disallowed); the two snapshot
# comparisons stay inline in each package's test module, where syrupy's
# `==` and the test-tree assert exemption both apply.


def assert_sample_emits_kinds(sample: SampleData) -> None:
    """The sample produces at least one chunk of every expected kind."""
    lang, expected_kinds, _files, chunks, _edges = sample
    kinds = {c.kind for c in chunks}
    missing = expected_kinds - kinds
    if missing:
        pytest.fail(f"{lang}: sample missing expected kinds {missing} (got {kinds})")


def assert_sample_parses_cleanly(sample: SampleData) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    lang, _expected_kinds, files, _chunks, _edges = sample
    manager = get_manager()
    for relpath, text in files:
        file_lang = manager.detect_language(relpath) or lang
        grammar = manager.load_grammar(file_lang)
        if grammar is None:
            pytest.fail(f"grammar for {file_lang} not installed")
        if Parser(grammar).parse(text.encode()).root_node.has_error:
            pytest.fail(f"{lang}: {relpath} does not parse cleanly")


def assert_sample_chunk_ids_deterministic(sample: SampleData) -> None:
    """Re-extracting a sample yields identical chunk ids (no nondeterminism)."""
    lang, _kinds, files, chunks, _edges = sample
    manager = get_manager()
    again: list[Chunk] = []
    for relpath, text in files:
        file_lang = manager.detect_language(relpath) or lang
        again.extend(extract_chunks(file_lang, text, file_path=relpath))
    if [c.id for c in again] != [c.id for c in chunks]:
        pytest.fail(f"{lang}: chunk ids not deterministic across re-extraction")


def assert_sample_line_numbers_positive(sample: SampleData) -> None:
    """Every sample chunk has positive, 1-indexed line numbers."""
    lang, _kinds, _files, chunks, _edges = sample
    bad = [
        (c.name, c.line_start, c.line_end) for c in chunks if not 1 <= c.line_start <= c.line_end
    ]
    if bad:
        pytest.fail(f"{lang}: non-positive line numbers {bad}")


def assert_sample_content_nonempty(sample: SampleData) -> None:
    """Every sample chunk carries source text (a rich sample never produces a
    bare presence chunk)."""
    lang, _kinds, _files, chunks, _edges = sample
    empty = [(c.kind, c.name) for c in chunks if not c.content]
    if empty:
        pytest.fail(f"{lang}: empty-content chunks {empty}")


def assert_sample_non_import_metadata_empty(sample: SampleData) -> None:
    """Only import chunks carry import metadata."""
    lang, _kinds, _files, chunks, _edges = sample
    leaked = [
        (c.kind, c.name)
        for c in chunks
        if c.kind != ChunkKind.IMPORT and c.metadata != ImportMeta()
    ]
    if leaked:
        pytest.fail(f"{lang}: non-import chunks carrying metadata {leaked}")


def assert_sample_blob_sha_propagated(sample: SampleData) -> None:
    """Every chunk carries the extraction blob sha."""
    _lang, _kinds, _files, chunks, _edges = sample
    if not all(c.blob_sha == "sha1" for c in chunks):
        pytest.fail("a chunk lost its blob sha")


def assert_sample_survives_syntax_error(sample: SampleData) -> None:
    """Appending garbage to any sample file still extracts valid parts.

    Tree-sitter error recovery (and each chunker's own tolerance) must keep
    yielding chunks rather than crash or return nothing on malformed input.
    """
    lang, _kinds, files, _chunks, _edges = sample
    manager = get_manager()
    for relpath, text in files:
        file_lang = manager.detect_language(relpath) or lang
        broken = text + "\n\x00\x00INVALID{{{[[\n"
        if not extract_chunks(file_lang, broken, file_path=relpath):
            pytest.fail(f"{lang}: {relpath} yielded nothing under trailing garbage")
