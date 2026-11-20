"""Bash extraction tests.

Construct/mixed cases (`cases_extraction.py`) drive the shared checks;
the function at the end pins bash's source/. import behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as build:
        reply = outline(build)
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
    with indexed(files) as build:
        reply = outline(build)
    kinds = {s.kind for s in reply}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(s.name, s.scope) for s in reply if s.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


def test_bash_source_and_dot_extracted_as_imports(indexed: Indexed) -> None:
    """source/. commands are captured as imports."""
    src = """\
source ./env.sh
. /etc/profile
"""
    with indexed({"input.sh": src}) as build:
        chunks = outline(build)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 2
    modules = {c.module for c in imports}
    assert modules == {"./env.sh", "/etc/profile"}


def test_bash_top_level_commands_are_searchable(indexed: Indexed) -> None:
    """A script's commands are what it does, so none is left out.

    A deploy script may define nothing at all, and before this its every
    command sat in no chunk.
    """
    src = """\
set -euo pipefail

DEST=/srv/app

rsync -a ./build/ "$DEST"

if [ -d "$DEST" ]; then
  systemctl restart app
fi
"""
    with indexed({"deploy.sh": src}) as build:
        chunks = outline(build)
    covered = {line for c in chunks for line in range(c.line_start, c.line_end + 1)}
    lines = src.splitlines()
    uncovered = [
        lines[i - 1] for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()
    ]
    assert uncovered == []
