"""Java extraction tests (cases in `cases_extraction.py`)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import (
    ChunkKind,
    EdgeKind,
    ImportMeta,
)
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_batch, file_rows
from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as index:
        reply = outline(index)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "files, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    indexed: Indexed,
    files: dict[str, str],
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    with indexed(files) as index:
        reply = outline(index)
    kinds = {s.kind for s in reply}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(s.name, s.scope) for s in reply if s.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="import")
def test_extracts_import_metadata(indexed: Indexed, files: dict[str, str], expected: dict) -> None:
    """The first import has the expected metadata."""
    with indexed(files) as index:
        reply = outline(index)
    imports = [s.metadata for s in reply if s.kind == ChunkKind.IMPORT]
    assert imports, f"no import extracted from {files!r}"
    assert imports[0] == ImportMeta(**expected)


@parametrize_with_cases(
    "files, count, metadata_list", cases=".cases_extraction", has_tag="multi_import"
)
def test_extracts_multi_import(
    indexed: Indexed,
    files: dict[str, str],
    count: int,
    metadata_list: list[dict],
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    with indexed(files) as index:
        reply = outline(index)
    imports = [s.metadata for s in reply if s.kind == ChunkKind.IMPORT]
    assert len(imports) == count
    assert imports == [ImportMeta(**expected) for expected in metadata_list]


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="references")
def test_records_calls_and_bases(lang: str, source: str, expected: list) -> None:
    """Each chunk records what it calls and a class its bases, written `lookup:a.b`."""
    recorded = []
    rows = extract_batch(file_rows([(FileEntry("input", "sha1", source.encode()), lang)]))
    for row in rows.iter_rows(named=True):
        refs = (row["metadata"] or {}).get("refs") or []
        calls = [r for r in refs if r["kind"] == EdgeKind.CALLS]
        bases = [r for r in refs if r["kind"] == EdgeKind.INHERITS]
        if calls or bases:
            recorded.append(
                (
                    row["kind"],
                    row["name"],
                    [f"{p['lookup']}:{'.'.join(p['path'])}" for p in calls],
                    [f"{p['lookup']}:{'.'.join(p['path'])}" for p in bases],
                )
            )
    assert recorded == expected
