"""Shared fixtures and helpers for git tests.

Provides a realistic multi-commit repository (``sample_repo``) that
tests can query without building their own from scratch.  The repo
has the following structure:

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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

# ── Tree builder ─────────────────────────────────────────────────────


def build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
    """Build a nested tree from ``{"dir/file.py": b"..."}`` paths."""
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
    """Create a commit with the given file tree and return its OID."""
    tree_oid = build_tree(repo, files)
    sig = pygit2.Signature(author, "test@test.com")
    return repo.create_commit(ref, sig, sig, message, tree_oid, parents or [])


# ── Shared dataset ───────────────────────────────────────────────────

# File contents — semantically distinct so tests can assert on them.
HANDLER_V1 = b"""\
def handle(request):
    return "ok"
"""

HANDLER_V2 = b"""\
def handle(request):
    validate(request)
    return "ok"
"""

UTILS = b"""\
def helper():
    return 42
"""

README = b"# Project\n\nA sample project.\n"

CONFIG_YAML = b"retries: 3\ntimeout: 30\n"

BINARY_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@dataclass
class SampleRepo:
    """Holds the repo and key commit OIDs for the shared dataset."""

    repo: pygit2.Repository
    base: pygit2.Oid
    mid: pygit2.Oid
    head: pygit2.Oid


@pytest.fixture
def sample_repo(tmp_path: Path) -> SampleRepo:
    """A multi-commit repo with branches, adds, mods, and deletes.

    Commit graph::

        base (main) → mid → head (feature)

    See module docstring for full file layout.
    """
    repo = pygit2.init_repository(str(tmp_path / "repo"))

    # base commit — initial state on main
    base = make_commit(
        repo,
        {
            "src/handler.py": HANDLER_V1,
            "src/utils.py": UTILS,
            "readme.md": README,
        },
        message="Initial commit",
        author="Alice",
    )

    # mid commit — modify handler, add config
    mid = make_commit(
        repo,
        {
            "src/handler.py": HANDLER_V2,
            "src/utils.py": UTILS,
            "readme.md": README,
            "config.yaml": CONFIG_YAML,
        },
        message="Add validation and config",
        parents=[base],
        author="Bob",
    )

    # head commit — delete readme, add binary
    head = make_commit(
        repo,
        {
            "src/handler.py": HANDLER_V2,
            "src/utils.py": UTILS,
            "config.yaml": CONFIG_YAML,
            "binary.png": BINARY_PNG,
        },
        message="Remove readme, add image",
        parents=[mid],
        author="Alice",
    )

    # Create a feature branch pointing at head
    repo.references.create("refs/heads/feature", head)

    # Keep main at base
    repo.references.create("refs/heads/main", base, force=True)

    return SampleRepo(repo=repo, base=base, mid=mid, head=head)
