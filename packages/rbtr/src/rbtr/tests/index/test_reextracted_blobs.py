"""A build that re-extracts a blob an older indexed commit lists keeps that commit's edges right.

Commit A is built; a plugin change (a new query and a higher
`extraction_serial`) then makes commit B's build re-extract the files A
shares. A's edges must follow the chunks A reads now.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.handlers import handle_find_refs
from rbtr.daemon.messages import FindRefsRequest
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr_lang_python.plugin import python


@pytest.fixture
def commits_sharing_files(tmp_path: Path) -> tuple[str, str]:
    """Commit A holds `a.py` and `b.py`; commit B adds `c.txt`, sharing A's two files."""
    repo = pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")
    author = pygit2.Signature("rbtr", "rbtr@example.com")
    files = {
        "a.py": """\
# Helper.
def f():
    pass
""",
        "b.py": """\
from a import f


def g():
    f()
""",
    }
    shas: list[str] = []
    for added in (files, {"c.txt": "notes\n"}):
        for path, text in added.items():
            (tmp_path / path).write_text(text)
        repo.index.add_all()
        repo.index.write()
        parents = [] if repo.head_is_unborn else [repo.head.target]
        shas.append(
            str(
                repo.create_commit(
                    "HEAD", author, author, "files", repo.index.write_tree(), parents
                )
            )
        )
    older, newer = shas
    return older, newer


@pytest.fixture
def python_without_comments() -> LanguageRegistration:
    """Python with its comments uncaptured and its serial raised: `f` no longer starts at the comment above it."""
    assert isinstance(python.extraction, QueryExtraction)
    return dataclasses.replace(
        python,
        extraction=dataclasses.replace(
            python.extraction,
            query=python.extraction.query.replace("(comment) @comment\n", "", 1),
        ),
        extraction_serial=python.extraction_serial + 1,
    )


@pytest.mark.xfail(
    strict=True,
    reason="B's build replaces the shared blobs' chunks, and f's id with them; A's stored "
    "edges still name the old ids, so find-refs at A answers nothing (Phase 2)",
)
def test_an_older_commit_keeps_its_edges_when_a_newer_build_re_extracts_its_files(
    tmp_path: Path,
    index: Index,
    commits_sharing_files: tuple[str, str],
    python_without_comments: LanguageRegistration,
) -> None:
    older, newer = commits_sharing_files
    with get_manager().limited_to(python):
        build_index(str(tmp_path), older, index)
    with get_manager().limited_to(python_without_comments):
        build_index(str(tmp_path), newer, index)
    refs = handle_find_refs(
        FindRefsRequest(repo_path=str(tmp_path), symbol="f", ref=older), index
    ).refs
    assert {(ref.file_path, ref.name, ref.edge.value) for ref in refs} == {
        ("b.py", "from a import f", "imports"),
        ("b.py", "g", "calls"),
    }
