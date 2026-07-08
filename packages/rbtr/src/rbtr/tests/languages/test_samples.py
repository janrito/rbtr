"""Per-language extraction tests over committed sample files.

Each language has one idiomatic sample under `samples/`. A shared,
case-parametrised `sample` fixture runs the real extraction pipeline
once per language and feeds three behavioural checks:

- `test_sample_emits_expected_kinds` — the sample exercises every
  `ChunkKind` its plugin is expected to produce.
- `test_sample_parses_cleanly` — the sample is valid source (no
  tree-sitter ERROR/MISSING nodes), so it is trustworthy as an example.
- `test_sample_extraction_matches_snapshot` — the full extracted
  chunks match the committed golden snapshot.

The samples double as illustrative examples; the snapshots are the
golden record of what extraction produces. Regenerate with
`just snapshots` after an intended extraction change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pytest_cases import fixture, parametrize_with_cases
from tree_sitter import Parser

from rbtr.git import FileEntry
from rbtr.index.edges import build_resolution_map, infer_import_edges
from rbtr.index.models import Chunk, ChunkKind, Edge, ImportMeta
from rbtr.index.orchestrator import extract_file
from rbtr.languages.manager import get_manager
from rbtr.testing import render_edges

# Core's sample suite iterates every in-core language (via cases), so it
# bundles the per-language data into one tuple; a single-language plugin
# package has no such fan-out and just uses small fixtures.
type SampleData = tuple[str, set[ChunkKind], list[tuple[str, str]], list[Chunk], list[Edge]]

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@fixture
@parametrize_with_cases("lang, expected_kinds", cases=".cases_samples", has_tag="sample")
def sample(lang: str, expected_kinds: set[ChunkKind]) -> SampleData:
    """Language id, expected kinds, project files, and extracted chunks.

    Loads the `samples/<lang>/` mini-project and extracts every file using
    *its own* detected language (a project may span languages, e.g. an html
    page plus its js/css). The project-relative path is the `file_path`, so
    snapshots and path-derived chunk ids are deterministic.
    """
    manager = get_manager()
    root = Path(__file__).parent / "samples" / lang
    files = [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]
    chunks: list[Chunk] = []
    for relpath, text in files:
        file_lang = manager.detect_language(relpath) or lang
        chunks.extend(extract_file(FileEntry(relpath, "sha1", text.encode()), file_lang))
    repo_files = {relpath for relpath, _ in files}
    edges = infer_import_edges(chunks, repo_files, build_resolution_map(manager))
    return lang, expected_kinds, files, chunks, edges


def _core_sample_langs() -> list[str]:
    """Language ids whose sample project still lives in core.

    Derived from the sample directories present (not `all_language_ids`):
    once a language is packaged out (e.g. sql → rbtr-lang-sql), its sample
    moves with it and its own package tests cover it.
    """
    return sorted(p.name for p in (Path(__file__).parent / "samples").iterdir() if p.is_dir())


@pytest.mark.parametrize("lang", _core_sample_langs())
def test_every_core_sample_is_registered(lang: str) -> None:
    """Every sample project resident in core maps to a registered language.

    Guards the sample-driven invariant battery and snapshots against silent
    gaps: a core sample whose language stopped registering fails here rather
    than being quietly skipped by the `sample` fixture.
    """
    root = Path(__file__).parent / "samples" / lang
    assert get_manager().get_registration(lang) is not None, f"{lang}: not registered"
    assert any(p.is_file() for p in root.rglob("*")), f"{lang}: empty samples/{lang}/ project"


def test_sample_emits_expected_kinds(sample: SampleData) -> None:
    """The sample produces at least one chunk of every expected kind."""
    lang, expected_kinds, _files, chunks, _edges = sample
    kinds = {c.kind for c in chunks}
    missing = expected_kinds - kinds
    assert not missing, f"{lang}: sample missing expected kinds {missing} (got {kinds})"


def test_sample_parses_cleanly(sample: SampleData) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    lang, _expected_kinds, files, _chunks, _edges = sample
    for relpath, text in files:
        file_lang = get_manager().detect_language(relpath) or lang
        grammar = get_manager().load_grammar(file_lang)
        assert grammar is not None, f"grammar for {file_lang} not installed"
        tree = Parser(grammar).parse(text.encode())
        assert not tree.root_node.has_error, f"{lang}: {relpath} does not parse cleanly"


def test_sample_extraction_matches_snapshot(
    sample: SampleData, snapshot_json: SnapshotAssertion
) -> None:
    """Extracted chunks match the committed golden snapshot."""
    _lang, _expected_kinds, _files, chunks, _edges = sample
    assert chunks == snapshot_json


def test_sample_edges_match_snapshot(sample: SampleData, snapshot_json: SnapshotAssertion) -> None:
    """Import edges among the project's files match the golden snapshot.

    Rendered as readable `src_file::name -> tgt_file::name [kind]` lines so
    the snapshot shows links resolve to the correct chunks. Single-file
    projects (and external imports) produce an empty list.
    """
    _lang, _expected_kinds, _files, chunks, edges = sample
    assert render_edges(edges, chunks) == snapshot_json


def test_sample_chunk_ids_deterministic(sample: SampleData) -> None:
    """Re-extracting a sample yields identical chunk ids (no nondeterminism)."""
    lang, _kinds, files, chunks, _edges = sample
    again: list[Chunk] = []
    for relpath, text in files:
        file_lang = get_manager().detect_language(relpath) or lang
        again.extend(extract_file(FileEntry(relpath, "sha1", text.encode()), file_lang))
    assert [c.id for c in again] == [c.id for c in chunks]


def test_sample_line_numbers_positive(sample: SampleData) -> None:
    """Every sample chunk has positive, 1-indexed line numbers."""
    lang, _kinds, _files, chunks, _edges = sample
    bad = [
        (c.name, c.line_start, c.line_end) for c in chunks if not 1 <= c.line_start <= c.line_end
    ]
    assert bad == [], f"{lang}: non-positive line numbers {bad}"


def test_sample_content_nonempty(sample: SampleData) -> None:
    """Every sample chunk carries source text; a rich sample never produces a
    bare presence chunk."""
    lang, _kinds, _files, chunks, _edges = sample
    empty = [(c.kind, c.name) for c in chunks if not c.content]
    assert empty == [], f"{lang}: empty-content chunks {empty}"


def test_sample_non_import_metadata_empty(sample: SampleData) -> None:
    """Only import chunks carry import metadata."""
    lang, _kinds, _files, chunks, _edges = sample
    leaked = [
        (c.kind, c.name)
        for c in chunks
        if c.kind != ChunkKind.IMPORT and c.metadata != ImportMeta()
    ]
    assert leaked == [], f"{lang}: non-import chunks carrying metadata {leaked}"


def test_sample_blob_sha_propagated(sample: SampleData) -> None:
    """Every chunk carries the extraction blob sha."""
    _lang, _kinds, _files, chunks, _edges = sample
    assert all(c.blob_sha == "sha1" for c in chunks)


def test_sample_survives_syntax_error(sample: SampleData) -> None:
    """Appending garbage to any sample file still extracts valid parts.

    Tree-sitter error recovery (and each chunker's own tolerance) must keep
    yielding chunks rather than crash or return nothing on malformed input.
    """
    lang, _kinds, files, _chunks, _edges = sample
    for relpath, text in files:
        file_lang = get_manager().detect_language(relpath) or lang
        broken = text + "\n\x00\x00INVALID{{{[[\n"
        assert extract_file(FileEntry(relpath, "sha1", broken.encode()), file_lang), (
            f"{lang}: {relpath} yielded nothing under trailing garbage"
        )


@parametrize_with_cases("lang, source, expected", cases=".cases_samples", has_tag="unsupported")
def test_known_unsupported_construct(
    lang: str, source: str, expected: tuple[ChunkKind, str, str]
) -> None:
    """A construct that *should* extract but does not yet (xfail, strict).

    Each case is marked `xfail(strict=True)`, so when a grammar or plugin
    upgrade starts producing the ideal chunk the suite fails, prompting
    removal of the xfail and an update to the language's sample.
    """
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    assert expected in [(c.kind, c.name, c.scope) for c in chunks]
