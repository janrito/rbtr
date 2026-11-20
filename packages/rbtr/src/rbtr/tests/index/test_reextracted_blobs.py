"""A build that re-extracts a blob an older indexed commit lists keeps that commit's edges right.

Commit A is built; a plugin change (a new query and a higher
`extraction_serial`) then makes commit B's build re-extract the files A
shares. A's edges must follow the chunks A reads now, and a build that
fails after re-extracting them leaves A as it was.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pygit2
import pytest
from structlog.testing import capture_logs

from rbtr.daemon.handlers import handle_find_refs
from rbtr.daemon.messages import FindRefsRequest
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query
from rbtr_lang_python.plugin import python


@pytest.fixture
def commits_sharing_files(tmp_path: Path) -> tuple[str, str]:
    """Commit A holds `a.py`, `b.py` and `project.manifest`; commit B adds `c.txt`, sharing A's files."""
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
        "project.manifest": "name = example\n",
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


@pytest.fixture
def python_without_calls() -> LanguageRegistration:
    """Python without its call and base captures, serial raised: the same chunks, saying no calls."""
    assert isinstance(python.extraction, QueryExtraction)
    return dataclasses.replace(
        python,
        extraction=dataclasses.replace(
            python.extraction, query=load_query("rbtr_lang_python", "python")
        ),
        extraction_serial=python.extraction_serial + 1,
    )


@pytest.fixture
def python_with_an_unreadable_manifest(
    python_without_comments: LanguageRegistration,
) -> LanguageRegistration:
    """Python without comments (so `f`'s id changes), declaring `project.manifest`, whose reader raises."""
    registration = dataclasses.replace(python_without_comments, manifest="project.manifest")

    @registration.manifest_reader
    def unreadable(text: str) -> tuple[tuple[str, str], ...]:
        raise ValueError(text)

    return registration


def test_an_older_commit_follows_what_a_re_extraction_says_when_no_id_changes(
    tmp_path: Path,
    index: Index,
    commits_sharing_files: tuple[str, str],
    python_without_calls: LanguageRegistration,
) -> None:
    """B's re-extraction records no calls; A's call edge goes with it, though every chunk keeps its id."""
    older, newer = commits_sharing_files
    with get_manager().limited_to(python):
        build_index(str(tmp_path), older, index)
    with get_manager().limited_to(python_without_calls):
        build_index(str(tmp_path), newer, index)
    refs = handle_find_refs(
        FindRefsRequest(repo_path=str(tmp_path), symbol="f", ref=older), index
    ).refs
    assert {(ref.file_path, ref.name, ref.edge.value) for ref in refs} == {
        ("b.py", "from a import f", "imports")
    }


def test_a_build_that_re_extracts_nothing_an_older_commit_lists_re_infers_nothing(
    tmp_path: Path, index: Index, commits_sharing_files: tuple[str, str]
) -> None:
    """B's files are A's, current, plus a new text file: no older commit's edges are inferred again."""
    older, newer = commits_sharing_files
    with get_manager().limited_to(python):
        build_index(str(tmp_path), older, index)
        with capture_logs() as logs:
            build_index(str(tmp_path), newer, index)
    assert [log["sha"] for log in logs if log["event"] == "inferred_edges"] == [newer[:12]]
    refs = handle_find_refs(
        FindRefsRequest(repo_path=str(tmp_path), symbol="f", ref=older), index
    ).refs
    assert {(ref.file_path, ref.name, ref.edge.value) for ref in refs} == {
        ("b.py", "from a import f", "imports"),
        ("b.py", "g", "calls"),
    }


def test_a_build_that_fails_after_re_extracting_leaves_an_older_commit_as_it_was(
    tmp_path: Path,
    index: Index,
    commits_sharing_files: tuple[str, str],
    python_with_an_unreadable_manifest: LanguageRegistration,
) -> None:
    """B's re-extraction changes A's chunks, then its manifest cannot be read, so the build rolls back: A's chunks and edges as before, B not indexed."""
    older, newer = commits_sharing_files
    with get_manager().limited_to(python):
        build_index(str(tmp_path), older, index)
    [repo] = index.repos()
    ids_before = set(index.chunks(at=repo.at(older))["id"])
    find_refs_f = FindRefsRequest(repo_path=str(tmp_path), symbol="f", ref=older)
    refs_before = handle_find_refs(find_refs_f, index).refs
    with (
        get_manager().limited_to(python_with_an_unreadable_manifest),
        pytest.raises(ValueError, match="name = example"),
    ):
        build_index(str(tmp_path), newer, index)
    assert index.has_indexed(at=repo.at(older))
    assert not index.has_indexed(at=repo.at(newer))
    assert set(index.chunks(at=repo.at(older))["id"]) == ids_before
    assert handle_find_refs(find_refs_f, index).refs == refs_before
