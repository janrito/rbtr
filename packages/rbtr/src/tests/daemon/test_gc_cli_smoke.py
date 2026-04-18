"""End-to-end smoke test for ``rbtr gc`` via subprocess.

Exercises the full CLI → inline-fallback path (no daemon running).
Seeds an IndexStore directly instead of running ``rbtr index`` to
keep the test fast and to avoid depending on the GGUF embedding
model during unit runs — the indexing path has its own tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.config import config
from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.store import IndexStore


@dataclass(frozen=True)
class TinyRepo:
    path: Path
    c1: str
    c2: str


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    home = Path(tempfile.mkdtemp(prefix="rbtr"))
    monkeypatch.setenv("RBTR_HOME", str(home))
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


@pytest.fixture
def tiny_repo(tmp_path: Path) -> TinyRepo:
    """A real git repo with two commits."""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False)
    sig = pygit2.Signature("t", "t@t.t")

    shas: list[str] = []
    for i, files in enumerate(
        [
            {"a.py": b"def a():\n    return 1\n"},
            {"a.py": b"def a():\n    return 2\n"},
        ]
    ):
        tb = repo.TreeBuilder()
        for name, content in files.items():
            tb.insert(name, repo.create_blob(content), pygit2.GIT_FILEMODE_BLOB)
        parents = [repo.head.target] if not repo.head_is_unborn else []
        shas.append(
            str(repo.create_commit("refs/heads/main", sig, sig, f"c{i}", tb.write(), parents))
        )
    return TinyRepo(path=path, c1=shas[0], c2=shas[1])


@pytest.fixture
def seeded_repo_id_both_commits(isolated_home: Path, tiny_repo: TinyRepo) -> int:
    """Seed both commits and close the store so subprocesses can open it."""
    config.reload()
    store = IndexStore.from_config()
    repo_id = store.register_repo(str(tiny_repo.path))
    for i, sha in enumerate((tiny_repo.c1, tiny_repo.c2)):
        chunk = Chunk(
            id=f"c{i}",
            blob_sha=f"b{i}",
            file_path="a.py",
            kind=ChunkKind.FUNCTION,
            name="f",
            content="",
            line_start=1,
            line_end=1,
        )
        store.insert_chunks([chunk], repo_id=repo_id)
        store.insert_snapshot(sha, "a.py", f"b{i}", repo_id=repo_id)
        store.mark_indexed(repo_id, sha)
    store.close()
    return repo_id


@pytest.fixture
def seeded_repo_id_first_commit(isolated_home: Path, tiny_repo: TinyRepo) -> int:
    """Seed only the first commit and close the store."""
    config.reload()
    store = IndexStore.from_config()
    repo_id = store.register_repo(str(tiny_repo.path))
    chunk = Chunk(
        id="c0",
        blob_sha="b0",
        file_path="a.py",
        kind=ChunkKind.FUNCTION,
        name="f",
        content="",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([chunk], repo_id=repo_id)
    store.insert_snapshot(tiny_repo.c1, "a.py", "b0", repo_id=repo_id)
    store.mark_indexed(repo_id, tiny_repo.c1)
    store.close()
    return repo_id


def _env(home: Path) -> dict[str, str]:
    # Pure projection: takes the caller-supplied path, returns the
    # env dict for subprocess.run.  No I/O, no state besides reading
    # os.environ.
    return {
        **os.environ,
        "RBTR_HOME": str(home),
    }


def test_gc_drop_removes_commit(
    isolated_home: Path,
    tiny_repo: TinyRepo,
    seeded_repo_id_both_commits: int,
) -> None:
    repo_id = seeded_repo_id_both_commits

    r = subprocess.run(  # noqa: S603  # trusted: args built from literals + sys.executable
        [
            sys.executable,
            "-m",
            "rbtr",
            "--json",
            "gc",
            "--repo-path",
            str(tiny_repo.path),
            "--drop",
            tiny_repo.c1,
        ],
        env=_env(isolated_home),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["kind"] == "gc"
    assert payload["commits_dropped"] == 1

    store = IndexStore.from_config()
    assert store.has_indexed(repo_id, tiny_repo.c1) is False
    assert store.has_indexed(repo_id, tiny_repo.c2) is True


def test_gc_dry_run_changes_nothing(
    isolated_home: Path,
    tiny_repo: TinyRepo,
    seeded_repo_id_first_commit: int,
) -> None:
    repo_id = seeded_repo_id_first_commit

    r = subprocess.run(  # noqa: S603  # trusted: args built from literals + sys.executable
        [
            sys.executable,
            "-m",
            "rbtr",
            "--json",
            "gc",
            "--repo-path",
            str(tiny_repo.path),
            "--dry-run",
        ],
        env=_env(isolated_home),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["dry_run"] is True

    store = IndexStore.from_config()
    assert store.has_indexed(repo_id, tiny_repo.c1) is True


def test_gc_keep_and_drop_are_mutually_exclusive(isolated_home: Path, tiny_repo: TinyRepo) -> None:
    r = subprocess.run(  # noqa: S603  # trusted: args built from literals + sys.executable
        [
            sys.executable,
            "-m",
            "rbtr",
            "gc",
            "--repo-path",
            str(tiny_repo.path),
            "--drop",
            "HEAD",
            "main",  # positional → --keep
        ],
        env=_env(isolated_home),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr.lower()
