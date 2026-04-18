"""Shared fixtures and helpers for git tests.

Provides realistic multi-commit repositories that tests can query
without building their own from scratch.

`sample_repo` — linear history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    base  (main)        — initial commit with 3 files
      └─ mid            — modifies handler.py, adds config.yaml
          └─ head       — deletes readme.md, adds binary.png
              (feature)

Files at *base*::

    src/handler.py   — "def handle(): ..."
    src/utils.py     — "def helper(): ..."
    readme.md        — "# Project"

Files at *head* (via feature branch)::

    src/handler.py   — modified body
    src/utils.py     — unchanged
    config.yaml      — added at mid
    binary.png       — binary file (added at head)

This gives tests:
- Two-ref diffs (base→head with adds, mods, deletes)
- Single-ref diffs (mid vs its parent, head vs mid)
- Path-scoped diffs
- Commit log walking (3 commits between base and head)
- Blob reads (existing, missing, binary)
- Branch resolution (main, feature, remote-only)

`merge_repo` — non-linear history with merge commit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    A ─── B ─── C  (base)
           \\
            D ─── E ── F  (head, merge of E + C)

The head branch has 3 exclusive commits (D, E, F) and one
merge parent (C) that is reachable from base.  This catches
bugs where commit-graph walks follow all parent chains instead
of excluding base-reachable history.

Files at *base* (C)::

    app.py     — "x = 1"
    shared.py  — "s = 0"

Files at *head* (F)::

    app.py     — "x = 1"  (unchanged — from C via merge)
    shared.py  — "s = 0"  (unchanged)
    feature.py — "f = 1"  (added on side branch)

Changed files base→head: `feature.py` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

# ── File-content fixtures ───────────────────────────────────────────


@pytest.fixture
def handler_v1() -> bytes:
    return b'def handle(request):\n    return "ok"\n'


@pytest.fixture
def handler_v2() -> bytes:
    return b'def handle(request):\n    validate(request)\n    return "ok"\n'


@pytest.fixture
def utils_content() -> bytes:
    return b"def helper():\n    return 42\n"


@pytest.fixture
def readme_content() -> bytes:
    return b"# Project\n\nA sample project.\n"


@pytest.fixture
def config_yaml_content() -> bytes:
    return b"retries: 3\ntimeout: 30\n"


@pytest.fixture
def binary_png_content() -> bytes:
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@pytest.fixture
def app_v1_content() -> bytes:
    return b"x = 1\n"


@pytest.fixture
def shared_py_content() -> bytes:
    return b"s = 0\n"


@pytest.fixture
def feature_py_content() -> bytes:
    return b"f = 1\n"


# ── Tree builder (pure projection over caller-supplied paths) ──────


def build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
    """Build a nested tree from ``{"dir/file.py": b"..."}`` paths.

    Pure projection over the caller-supplied ``files`` mapping; does
    not read from module state.
    """
    subtrees: dict[str, dict[str, bytes]] = {}
    blobs: dict[str, bytes] = {}

    for path, content in files.items():
        if "/" in path:
            top, rest = path.split("/", 1)
            subtrees.setdefault(top, {})[rest] = content
        else:
            blobs[path] = content

    tb = repo.TreeBuilder()
    for name, data in blobs.items():
        tb.insert(name, repo.create_blob(data), pygit2.GIT_FILEMODE_BLOB)
    for name, sub_files in subtrees.items():
        tb.insert(name, build_tree(repo, sub_files), pygit2.GIT_FILEMODE_TREE)
    return tb.write()


def make_commit(
    repo: pygit2.Repository,
    files: dict[str, bytes],
    *,
    message: str = "commit",
    parents: list[pygit2.Oid] | None = None,
    ref: str = "refs/heads/main",
    author: str = "Test Author",
) -> pygit2.Oid:
    """Create a commit with the given file tree and return its OID.

    Pure projection over caller-supplied arguments.
    """
    tree_oid = build_tree(repo, files)
    sig = pygit2.Signature(author, "test@test.com")
    return repo.create_commit(ref, sig, sig, message, tree_oid, parents or [])


# ── Repo datasets ───────────────────────────────────────────────────


@dataclass
class SampleRepo:
    """Holds the repo and key commit OIDs for the linear dataset."""

    repo: pygit2.Repository
    base: pygit2.Oid
    mid: pygit2.Oid
    head: pygit2.Oid


@dataclass
class MergeRepo:
    """Holds the repo and key commit OIDs for the merge dataset.

    Commit graph::

        A ─── B ─── C  (base)
               \\
                D ─── E ── F  (head, merge of E + C)

    `base` is the branch ref pointing at C.
    `head` is the branch ref pointing at F.
    `exclusive` lists commits only reachable from head (D, E, F).
    `shared` lists commits reachable from both (A, B, C).
    """

    repo: pygit2.Repository
    base: pygit2.Oid
    head: pygit2.Oid
    exclusive: list[pygit2.Oid]
    shared: list[pygit2.Oid]


@pytest.fixture
def sample_repo(
    tmp_path: Path,
    handler_v1: bytes,
    handler_v2: bytes,
    utils_content: bytes,
    readme_content: bytes,
    config_yaml_content: bytes,
    binary_png_content: bytes,
) -> SampleRepo:
    """A multi-commit repo with branches, adds, mods, and deletes."""
    repo = pygit2.init_repository(str(tmp_path / "repo"))

    base = make_commit(
        repo,
        {
            "src/handler.py": handler_v1,
            "src/utils.py": utils_content,
            "readme.md": readme_content,
        },
        message="Initial commit",
        author="Alice",
    )

    mid = make_commit(
        repo,
        {
            "src/handler.py": handler_v2,
            "src/utils.py": utils_content,
            "readme.md": readme_content,
            "config.yaml": config_yaml_content,
        },
        message="Add validation and config",
        parents=[base],
        author="Bob",
    )

    head = make_commit(
        repo,
        {
            "src/handler.py": handler_v2,
            "src/utils.py": utils_content,
            "config.yaml": config_yaml_content,
            "binary.png": binary_png_content,
        },
        message="Remove readme, add image",
        parents=[mid],
        author="Alice",
    )

    repo.references.create("refs/heads/feature", head)
    repo.references.create("refs/heads/main", base, force=True)

    return SampleRepo(repo=repo, base=base, mid=mid, head=head)


@pytest.fixture
def merge_repo(
    tmp_path: Path,
    app_v1_content: bytes,
    shared_py_content: bytes,
    feature_py_content: bytes,
) -> MergeRepo:
    """A repo with a merge commit in the head branch.

    The head branch merges C back in, so C (and its ancestors) are
    reachable from *both* branches.  Only D, E, F are exclusive to
    head.
    """
    repo = pygit2.init_repository(str(tmp_path / "merge_repo"))

    a = make_commit(
        repo,
        {"app.py": app_v1_content, "shared.py": shared_py_content},
        message="A",
        author="X",
    )
    b = make_commit(
        repo,
        {"app.py": app_v1_content, "shared.py": shared_py_content},
        message="B",
        parents=[a],
        author="X",
    )
    c = make_commit(
        repo,
        {"app.py": app_v1_content, "shared.py": shared_py_content},
        message="C",
        parents=[b],
        author="X",
    )

    d = make_commit(
        repo,
        {
            "app.py": app_v1_content,
            "shared.py": shared_py_content,
            "feature.py": feature_py_content,
        },
        message="D",
        parents=[b],
        ref="refs/heads/side",
        author="Y",
    )
    e = make_commit(
        repo,
        {
            "app.py": app_v1_content,
            "shared.py": shared_py_content,
            "feature.py": feature_py_content,
        },
        message="E",
        parents=[d],
        ref="refs/heads/side",
        author="Y",
    )

    f = make_commit(
        repo,
        {
            "app.py": app_v1_content,
            "shared.py": shared_py_content,
            "feature.py": feature_py_content,
        },
        message="F",
        parents=[e, c],
        ref="refs/heads/side",
        author="Y",
    )

    repo.references.create("refs/heads/base", c, force=True)
    repo.references.create("refs/heads/head", f, force=True)

    return MergeRepo(
        repo=repo,
        base=c,
        head=f,
        exclusive=[d, e, f],
        shared=[a, b, c],
    )
