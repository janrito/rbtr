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

from typing import TYPE_CHECKING

import pytest
from pytest_cases import fixture, parametrize_with_cases
from tree_sitter import Parser

from rbtr.index.edges import build_resolution_map, infer_import_edges
from rbtr.index.models import Chunk, ChunkKind, Edge, ImportMeta
from rbtr.languages import LanguageManager, get_manager

from .conftest import extract_chunks, load_project, render_edges

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

type ProjectFiles = list[tuple[str, str]]
type SampleData = tuple[str, set[ChunkKind], ProjectFiles, list[Chunk], list[Edge]]


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
    files = load_project(lang)
    chunks: list[Chunk] = []
    for relpath, text in files:
        file_lang = manager.detect_language(relpath) or lang
        chunks.extend(extract_chunks(file_lang, text, file_path=relpath))
    repo_files = {relpath for relpath, _ in files}
    edges = infer_import_edges(chunks, repo_files, build_resolution_map(manager))
    return lang, expected_kinds, files, chunks, edges


@pytest.mark.parametrize("lang", sorted(get_manager().all_language_ids()))
def test_every_language_has_a_sample(lang: str) -> None:
    """Every registered language ships a `samples/<lang>/` project.

    Guards the sample-driven invariant battery and snapshots against silent
    gaps: a new language with no sample fails here rather than being quietly
    skipped by the `sample` fixture.
    """
    assert load_project(lang), f"{lang}: no samples/{lang}/ project"


def test_sample_emits_expected_kinds(sample: SampleData) -> None:
    """The sample produces at least one chunk of every expected kind."""
    lang, expected_kinds, _files, chunks, _edges = sample
    kinds = {c.kind for c in chunks}
    missing = expected_kinds - kinds
    assert not missing, f"{lang}: sample missing expected kinds {missing} (got {kinds})"


def test_sample_parses_cleanly(sample: SampleData, language_manager: LanguageManager) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    lang, _expected_kinds, files, _chunks, _edges = sample
    for relpath, text in files:
        file_lang = language_manager.detect_language(relpath) or lang
        grammar = language_manager.load_grammar(file_lang)
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


def test_sample_chunk_ids_deterministic(
    sample: SampleData, language_manager: LanguageManager
) -> None:
    """Re-extracting a sample yields identical chunk ids (no nondeterminism)."""
    lang, _kinds, files, chunks, _edges = sample
    again: list[Chunk] = []
    for relpath, text in files:
        file_lang = language_manager.detect_language(relpath) or lang
        again.extend(extract_chunks(file_lang, text, file_path=relpath))
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


def test_sample_survives_syntax_error(
    sample: SampleData, language_manager: LanguageManager
) -> None:
    """Appending garbage to any sample file still extracts valid parts.

    Tree-sitter error recovery (and each chunker's own tolerance) must keep
    yielding chunks rather than crash or return nothing on malformed input.
    """
    lang, _kinds, files, _chunks, _edges = sample
    for relpath, text in files:
        file_lang = language_manager.detect_language(relpath) or lang
        broken = text + "\n\x00\x00INVALID{{{[[\n"
        assert extract_chunks(file_lang, broken, file_path=relpath), (
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
    chunks = extract_chunks(lang, source)
    assert expected in [(c.kind, c.name, c.scope) for c in chunks]


# ── SQL dialects ───────────────────────────────────────────────
#
# rbtr ships one generic SQL grammar for all `.sql` files. These cases
# document how it handles each major dialect; see TODO-construct-coverage.md
# Appendix O. The snapshot pins current extraction (partial under error
# recovery for unsupported syntax); the parse-clean test is a strict-xfail
# sentinel that fires when the grammar gains full support for a dialect.


@parametrize_with_cases("source, dialect", cases=".cases_samples", has_tag="sql_dialect")
def test_sql_dialect_extraction_matches_snapshot(
    source: str, dialect: str, snapshot_json: SnapshotAssertion
) -> None:
    """Current extraction for each SQL dialect under the generic grammar."""
    chunks = extract_chunks("sql", source, file_path=f"{dialect}.sql")
    assert chunks == snapshot_json


@pytest.mark.xfail(
    reason="generic SQL grammar does not fully parse dialect-specific syntax (Appendix O)",
    strict=True,
)
@parametrize_with_cases("source, dialect", cases=".cases_samples", has_tag="sql_dialect")
def test_sql_dialect_parses_cleanly(
    source: str, dialect: str, language_manager: LanguageManager
) -> None:
    """Sentinel: flips to XPASS (failing) when a dialect parses cleanly."""
    grammar = language_manager.load_grammar("sql")
    assert grammar is not None
    tree = Parser(grammar).parse(source.encode())
    assert not tree.root_node.has_error
