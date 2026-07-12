"""Bash extraction tests.

Construct/mixed cases (`cases_extraction.py`) drive the shared checks;
the function at the end pins bash's source/. import behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
from rbtr.index.orchestrator import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "lang, source, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    lang: str,
    source: str,
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    kinds = {c.kind for c in chunks}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


def test_bash_source_and_dot_extracted_as_imports() -> None:
    """source/. commands are captured as imports."""
    src = """\
source ./env.sh
. /etc/profile
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "bash")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 2
    modules = {c.metadata.module for c in imports}
    assert modules == {"./env.sh", "/etc/profile"}
