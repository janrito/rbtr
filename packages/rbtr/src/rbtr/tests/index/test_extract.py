"""Tests for extraction routing and prose format detection."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.git import FileEntry
from rbtr.index.chunks import detect_prose_format
from rbtr.index.models import ChunkKind
from rbtr.index.orchestrator import _extract_file, build_index
from rbtr.index.store import IndexStore
from rbtr.languages import get_manager

# ── Prose format detection ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            """\
Title
=====

Body text.
""",
            "rst",
            id="rst-underline",
        ),
        pytest.param(
            """\
Some text.

.. note::

    A note.
""",
            "rst",
            id="rst-directive",
        ),
        pytest.param(
            """\
# Title

Body text.
""",
            "markdown",
            id="md-atx",
        ),
        pytest.param(
            """\
Just some plain text.
Nothing special about it.
""",
            None,
            id="neither",
        ),
    ],
)
def test_detect_prose_format(content: str, expected: str | None) -> None:
    assert detect_prose_format(content) == expected


# ── File routing ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    (
        "file_path",
        "file_content",
        "expected_language",
        "expected_kind",
        "expected_name",
        "min_chunks",
    ),
    [
        pytest.param(
            "pkg.json",
            '{"name": "test", "version": "1.0"}',
            "json",
            ChunkKind.DOC_SECTION,
            None,
            1,
            id="json",
        ),
        pytest.param(
            "doc.md",
            """\
# Title

Body text.
""",
            "markdown",
            None,
            "Title",
            1,
            id="markdown",
        ),
        pytest.param(
            "doc.rst",
            """\
Title
=====

Body text.
""",
            "rst",
            None,
            "Title",
            1,
            id="rst",
        ),
        pytest.param(
            "config.toml",
            """\
[project]
name = 'rbtr'
""",
            "toml",
            None,
            "project",
            1,
            id="toml",
        ),
        pytest.param(
            "ci.yml",
            """\
name: CI
on: [push]
""",
            "yaml",
            None,
            None,
            1,
            id="yaml",
        ),
        pytest.param(
            "CHANGES.txt",
            """\
Title
=====

Body text.
""",
            "rst",
            None,
            None,
            1,
            id="unknown-ext-rst",
        ),
        pytest.param(
            "README.txt",
            """\
# Title

Body text.
""",
            "markdown",
            None,
            None,
            1,
            id="unknown-ext-md",
        ),
        pytest.param(
            "empty.txt",
            "",
            "",
            None,
            None,
            0,
            id="empty-file",
        ),
        pytest.param(
            "blank.txt",
            "   \n\n  \n",
            "",
            None,
            None,
            0,
            id="whitespace-only",
        ),
        pytest.param(
            "data.txt",
            """\
Just some plain text.
Nothing special.
""",
            "",
            ChunkKind.RAW_CHUNK,
            None,
            1,
            id="unknown-ext-plain",
        ),
    ],
)
def test_extract_file_routes_correctly(
    file_path: str,
    file_content: str,
    expected_language: str,
    expected_kind: ChunkKind | None,
    expected_name: str | None,
    min_chunks: int,
) -> None:
    """_extract_file routes each file type to the correct chunker."""
    mgr = get_manager()
    reg = mgr.get_registration(expected_language) if expected_language else None
    entry = FileEntry(path=file_path, blob_sha="sha1", content=file_content.encode())
    chunks = list(_extract_file(entry, expected_language, reg))
    assert len(chunks) >= min_chunks
    if not chunks:
        return
    assert chunks[0].language == expected_language
    if expected_kind is not None:
        assert all(c.kind == expected_kind for c in chunks)
    if expected_name is not None:
        assert chunks[0].name == expected_name


# ── Edge cases ───────────────────────────────────────────────────────


def test_build_index_extraction_error_is_nonfatal(
    tmp_path: Path, store: IndexStore, mocker: MockerFixture
) -> None:
    """A file that triggers an extraction error doesn't crash the build.

    The error is recorded in result.errors, and other files are
    still indexed.
    """
    repo = pygit2.init_repository(str(tmp_path / "err"), bare=False, initial_head="main")

    (tmp_path / "err" / "good.py").write_text("def ok(): pass\n")
    (tmp_path / "err" / "bad.py").write_text("def boom(): pass\n")
    index = repo.index
    index.add("good.py")
    index.add("bad.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])
    sha = str(repo.head.target)

    # Make extraction fail for bad.py only.
    original_extract = _extract_file

    def _patched_extract(entry: FileEntry, language: str, reg: object) -> list:
        if entry.path == "bad.py":
            msg = "parse error"
            raise RuntimeError(msg)
        return list(original_extract(entry, language, reg))  # type: ignore[arg-type]  # test shim

    mocker.patch("rbtr.index.orchestrator._extract_file", side_effect=_patched_extract)

    result = build_index(repo.workdir, sha, store, repo_id=1)

    assert len(result.errors) == 1
    assert "bad.py" in result.errors[0]
    chunks = store.get_chunks(sha, repo_id=1)
    assert any(c.file_path == "good.py" for c in chunks)
