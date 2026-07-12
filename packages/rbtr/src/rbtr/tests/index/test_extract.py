"""Tests for build-index extraction error handling."""

from __future__ import annotations

from pathlib import Path

import pygit2
from pytest_mock import MockerFixture

from rbtr.git import FileEntry
from rbtr.index.orchestrator import build_index
from rbtr.index.store import IndexStore
from rbtr.languages.extract import extract_file

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
    original_extract = extract_file

    def _patched_extract(entry: FileEntry, language: str) -> list:
        if entry.path == "bad.py":
            msg = "parse error"
            raise RuntimeError(msg)
        return list(original_extract(entry, language))

    mocker.patch("rbtr.index.orchestrator.extract_file", side_effect=_patched_extract)

    result = build_index(repo.workdir, sha, store, repo_id=1)

    assert len(result.errors) == 1
    assert "bad.py" in result.errors[0]
    chunks = store.get_chunks(sha, repo_id=1)
    assert any(c.file_path == "good.py" for c in chunks)
