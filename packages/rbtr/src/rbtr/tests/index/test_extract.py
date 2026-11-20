"""Tests for build-index extraction error handling."""

from __future__ import annotations

from pathlib import Path

import dataframely as dy
import pygit2
from pytest_mock import MockerFixture

from rbtr.domain.rules import FileRow
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.languages import extract


def test_build_index_extraction_error_is_nonfatal(
    tmp_path: Path, index: Index, mocker: MockerFixture
) -> None:
    """A file that triggers an extraction error doesn't crash the build.

    The error is recorded in result.errors, and other files are
    still indexed.
    """
    repo = pygit2.init_repository(str(tmp_path / "err"), bare=False, initial_head="main")

    (tmp_path / "err" / "good.py").write_text("def ok(): pass\n")
    (tmp_path / "err" / "bad.py").write_text("def boom(): pass\n")
    git_index = repo.index
    git_index.add("good.py")
    git_index.add("bad.py")
    git_index.write()
    tree_oid = git_index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])
    sha = str(repo.head.target)

    # Make extraction fail for any batch holding bad.py.
    original_extract = extract.extract_batch

    def _patched_extract(files: dy.DataFrame[FileRow]) -> extract.Extraction:
        if "bad.py" in files["path"]:
            msg = "parse error"
            raise RuntimeError(msg)
        return original_extract(files)

    mocker.patch.object(extract, "extract_batch", side_effect=_patched_extract)

    result = build_index(repo.workdir, sha, index)

    assert len(result.errors) == 1
    assert "bad.py" in result.errors[0]
    [registered] = index.repos()
    chunks = index.chunks(at=registered.at(sha))
    assert (chunks["file_path"] == "good.py").any()
