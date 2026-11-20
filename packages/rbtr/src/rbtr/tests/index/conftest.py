"""Shared fixtures for `tests/index/`.

Every index here is written as production writes one: commits of real
files, built (`build_index`), then collected or embedded through `Index`.
The ranking dataset is a commit of four files whose code makes the edges
the search-ranking tests count.

No module-level test data.  Everything is a fixture so
dependency graphs are explicit.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest

from rbtr.domain.models import (
    SnapshotRef,
)
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.index.refs import latest_ref

from ..conftest import make_commit


@pytest.fixture
def head_ref(index: Index) -> SnapshotRef:
    """Repo `/repo` registered in `index`, at snapshot `head`.

    The repo comes back from the registration, so the ref names a repo
    that exists.  The path is synthetic: these are store-level tests
    that never read git, and registration establishes identity.
    """
    return index.register("/repo").at("head")


# ═════════════════════════════════════════════════════════════════════
# Cross-repo content sharing
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def shared_sha(git_repo: pygit2.Repository) -> str:
    """A commit of `git_repo` holding only `x.py`, which defines `shared_fn`."""
    x = b"""\
def shared_fn():
    pass
"""
    return str(make_commit(git_repo, {"x.py": x}, ref="refs/heads/shared"))


@pytest.fixture
def shared_chunk_index(
    index: Index, shared_sha: str, repo_path: str, linked_worktree: pygit2.Repository
) -> Index:
    """Both checkouts of one project built at `shared_sha`, so they share one chunk.

    The main checkout is repo 1 and the linked worktree repo 2. The
    second build finds the blob already chunked and only records its
    snapshot, as a second checkout does in production.
    """
    build_index(repo_path, shared_sha, index)
    build_index(normalise_repo_path(linked_worktree.workdir), shared_sha, index)
    return index


@pytest.fixture
def shared_chunk_id(shared_chunk_index: Index, shared_sha: str) -> str:
    """The id of `shared_fn`, the one chunk `shared_chunk_index`'s repos share."""
    repo1 = shared_chunk_index.repos()[0]
    return str(shared_chunk_index.chunks(at=repo1.at(shared_sha), name="shared_fn")["id"].item())


@pytest.fixture
def m_py() -> bytes:
    """`m.py`, defining `fn`: byte-identical wherever it is committed."""
    return b"""\
def fn():
    return 1
"""


@pytest.fixture
def importer_a_sha(git_repo: pygit2.Repository, m_py: bytes) -> str:
    """A commit holding `m.py` and `a.py`, which imports `fn` from it."""
    a = b"""\
from m import fn
"""
    return str(make_commit(git_repo, {"m.py": m_py, "a.py": a}, ref="refs/heads/a"))


@pytest.fixture
def importer_b_sha(git_repo: pygit2.Repository, m_py: bytes) -> str:
    """A commit holding `m.py` and `b.py`, which imports `fn` from it."""
    b = b"""\
from m import fn
"""
    return str(make_commit(git_repo, {"m.py": m_py, "b.py": b}, ref="refs/heads/b"))


@pytest.fixture
def shared_target(
    index: Index,
    repo_path: str,
    linked_worktree: pygit2.Repository,
    importer_a_sha: str,
    importer_b_sha: str,
) -> Index:
    """Two checkouts of one project, each importing `fn` from its own copy of `m.py`.

    The main checkout (repo 1) is built at `importer_a_sha` and the
    linked worktree (repo 2) at `importer_b_sha`. `m.py` is the same in
    both, so `fn` is one chunk the two snapshots share.
    """
    build_index(repo_path, importer_a_sha, index)
    build_index(normalise_repo_path(linked_worktree.workdir), importer_b_sha, index)
    return index


# ═════════════════════════════════════════════════════════════════════
# Edges built from source
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def import_graph(module_index: Index, module_repo: pygit2.Repository) -> SnapshotRef:
    """A snapshot, built into `module_index`, where code imports a function and prose links to its file.

    `src/app.py` imports `fn` from `src/m.py`, an `imports` edge;
    `README.md` links to `src/m.py`, a `documents` edge to `fn`.
    """
    m = b"""\
def fn():
    return 1
"""
    app = b"""\
from src.m import fn


def run():
    return fn()
"""
    readme = b"""\
# Guide

See [fn](src/m.py).
"""
    files = {"src/m.py": m, "src/app.py": app, "README.md": readme}
    sha = str(make_commit(module_repo, files, ref="refs/heads/graph"))
    path = normalise_repo_path(module_repo.workdir)
    build_index(path, sha, module_index)
    return module_index.register(path).at(sha)


# ═════════════════════════════════════════════════════════════════════
# Commits over the shared `git_repo` fixture (defined in the root
# conftest, alongside `index` — both are reused beyond `tests/index/`).
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def snapshot_sha(git_repo: pygit2.Repository) -> str:
    """SHA of the initial commit."""
    return str(git_repo.head.target)


@pytest.fixture
def two_commits(git_repo: pygit2.Repository, tmp_path: Path) -> tuple[str, str]:
    """Add a second commit that modifies utils.py and adds a new file."""
    base_sha = str(git_repo.head.target)

    utils_path = tmp_path / "src" / "utils.py"
    utils_path.write_bytes(b"""\
\"\"\"Utility functions.\"\"\"

def helper():
    return 42

def format_name(name):
    return name.strip()

def new_func():
    return "new"
""")

    new_path = tmp_path / "src" / "service.py"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"""\
\"\"\"Service layer.\"\"\"

def serve():
    return True
""")

    git_index = git_repo.index
    git_index.add("src/utils.py")
    git_index.add("src/service.py")
    git_index.write()
    tree_oid = git_index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Add new_func and service", tree_oid, [parent.id])

    head_sha = str(git_repo.head.target)
    return base_sha, head_sha


# ═════════════════════════════════════════════════════════════════════
# Symbol-diff dataset (for test_symbol_diff.py)
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def diff_repo(tmp_path: Path) -> pygit2.Repository:
    """Empty git repo (no commits) for symbol-diff scenarios."""
    return pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")


# ═════════════════════════════════════════════════════════════════════
# Duplicated content (for test_duplicate_content.py)
#
# Copies land in a *second* commit because that is when the bug bites:
# the originals are already committed, so the dedup gate sees their
# blobs and skips the copies.  In one commit the copies survive by
# accident, the chunk buffer not having flushed yet.
#
# `src/caller.py` is a weaker match for the ranking query: min-max
# normalisation scores every candidate 0.0 when they all tie, so a
# penalty needs something to separate.


@pytest.fixture
def dup_index(tmp_path: Path, index: Index) -> Index:
    """Originals, then verbatim copies at new paths, both built."""
    shared = b'''\
"""Shared helpers."""


def parse_manifest(text):
    """Read a manifest and return its entries."""
    return [line for line in text.splitlines() if line]
'''
    api = b"""\
export function describeEndpoint(name) {
  return `endpoint ${name}`;
}
"""
    dup = b'''\
def normalise_widget(widget):
    """Normalise a widget before serialisation."""
    return widget.strip().lower()
'''
    caller = b'''\
from dup import normalise_widget


def render_widget(widget):
    """Render a widget for display."""
    return normalise_widget(widget)
'''

    referrer = b"from .shared import parse_manifest\n"
    repo = pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")
    sig = pygit2.Signature("Test", "test@test.com")
    commits: list[str] = []
    for message, files in (
        (
            "Originals",
            {
                "src/shared.py": shared,
                "types/api.d.ts": api,
                "src/dup.py": dup,
                "src/caller.py": caller,
                "src/referrer.py": referrer,
            },
        ),
        (
            "Copies",
            {
                "lib/shared.py": shared,
                "dist/api.js": api,
                "node_modules/dup.py": dup,
                "lib/referrer.py": referrer,
            },
        ),
    ):
        for path, content in files.items():
            full = tmp_path / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(content)
            repo.index.add(path)
        repo.index.write()
        parents = [repo.head.target] if commits else []
        tree = repo.index.write_tree()
        commits.append(str(repo.create_commit("HEAD", sig, sig, message, tree, parents)))

    build_index(repo.workdir, commits[0], index)
    build_index(repo.workdir, commits[1], index, base_sha=commits[0])
    return index


@pytest.fixture
def dup_ref(dup_index: Index) -> SnapshotRef:
    """The duplicated repo at its head commit."""
    ref = latest_ref(dup_index, dup_index.repos()[0])
    assert ref is not None
    return ref


# Ranking dataset (for test_search_ranking.py, test_search_structural.py)
# ═════════════════════════════════════════════════════════════════════
#
# Small realistic project with deliberate ranking conflicts:
#   * CLASS vs IMPORT of the same name  → kind boost
#   * source vs test mentioning a fn    → file penalty
#   * high-df term ("config") in most   → IDF neutralised
#   * exact name match vs content       → name score
#   * 4 inbound edges vs 2 vs 0         → importance
#   * diff touching src/server.py       → proximity
#
# The edges are the ones its code makes: `AppConfig` is imported by
# `src/server.py` and linked from `docs/setup.md` (2); `load_config` is
# imported by `src/server.py` and the test, and called by `start_server`
# and `test_load_config` (4, two of them calls); `start_server` is reached
# by nothing.


@pytest.fixture
def ranking_commit(fake_repo: str) -> str:
    """A commit of `fake_repo` holding the ranking dataset's files."""
    repo = pygit2.Repository(fake_repo)
    files = {
        "src/config.py": b"""\
class AppConfig:
    database_url: str
    max_retries: int = 3
    timeout: float = 30.0


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        data = json.load(f)
    return data
""",
        "src/server.py": b"""\
from config import AppConfig, load_config


def start_server(path: str) -> None:
    config = load_config(path)
    app = create_app(config)
    app.run(host="0.0.0.0", port=config.port)
""",
        "tests/test_config.py": b"""\
from config import load_config


def test_load_config():
    config = load_config("test.json")
    assert isinstance(config, dict)
    assert config["max_retries"] == 3
    config = load_config("other.json")
    assert config["timeout"] == 30.0
    config = load_config("empty.json")
""",
        "docs/setup.md": b"""\
## Configuration

Use `load_config` to load an [AppConfig](../src/config.py#AppConfig) from a JSON file.
Set `database_url` and `max_retries` as needed.
""",
    }
    return str(make_commit(repo, files, parents=[repo.head.peel(pygit2.Commit).id]))


@pytest.fixture
def ranking_index(fake_repo: str, ranking_commit: str) -> Generator[Index]:
    """An in-memory Index with the ranking dataset built."""
    index = Index(writable=True)
    build_index(fake_repo, ranking_commit, index)
    yield index
    index.close()
