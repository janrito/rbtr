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
from pathlib import Path

import pygit2
import pytest

from rbtr.config import config
from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.store import IndexStore


@pytest.fixture
def isolated_user_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    user_dir = Path(tempfile.mkdtemp(prefix="rbtr"))
    monkeypatch.setenv("RBTR_USER_DIR", str(user_dir))
    monkeypatch.setenv("RBTR_DB_PATH", str(user_dir / "index.duckdb"))
    try:
        yield user_dir
    finally:
        shutil.rmtree(user_dir, ignore_errors=True)


@pytest.fixture
def tiny_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A real git repo with two commits; returns (path, c1_sha, c2_sha)."""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False)
    sig = pygit2.Signature("t", "t@t.t")

    def commit(files: dict[str, bytes]) -> str:
        tb = repo.TreeBuilder()
        for name, content in files.items():
            tb.insert(name, repo.create_blob(content), pygit2.GIT_FILEMODE_BLOB)
        parents = [repo.head.target] if not repo.head_is_unborn else []
        return str(
            repo.create_commit(
                "refs/heads/main", sig, sig, "c", tb.write(), parents
            )
        )

    c1 = commit({"a.py": b"def a():\n    return 1\n"})
    c2 = commit({"a.py": b"def a():\n    return 2\n"})
    return path, c1, c2


def _run(args: list[str], user_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "RBTR_USER_DIR": str(user_dir),
        "RBTR_DB_PATH": str(user_dir / "index.duckdb"),
    }
    return subprocess.run(
        [sys.executable, "-m", "rbtr", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _seed_store(user_dir: Path, repo_path: str, *shas: str) -> int:
    """Open the central store (from config), register repo, mark shas."""
    # Configure the store to use the isolated user_dir.
    config.reload()
    store = IndexStore.from_config()
    repo_id = store.register_repo(repo_path)
    for i, sha in enumerate(shas):
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
    return repo_id


def test_gc_drop_removes_commit(
    isolated_user_dir: Path, tiny_repo: tuple[Path, str, str]
) -> None:
    path, c1, c2 = tiny_repo
    repo_path = str(path)
    repo_id = _seed_store(isolated_user_dir, repo_path, c1, c2)

    r = _run(
        ["--json", "gc", "--repo-path", repo_path, "--drop", c1],
        isolated_user_dir,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["kind"] == "gc"
    assert payload["commits_dropped"] == 1

    store = IndexStore.from_config()
    assert store.has_indexed(repo_id, c1) is False
    assert store.has_indexed(repo_id, c2) is True


def test_gc_dry_run_changes_nothing(
    isolated_user_dir: Path, tiny_repo: tuple[Path, str, str]
) -> None:
    path, c1, _c2 = tiny_repo
    repo_path = str(path)
    repo_id = _seed_store(isolated_user_dir, repo_path, c1)

    r = _run(
        ["--json", "gc", "--repo-path", repo_path, "--dry-run"],
        isolated_user_dir,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["dry_run"] is True

    store = IndexStore.from_config()
    assert store.has_indexed(repo_id, c1) is True


def test_gc_keep_and_drop_are_mutually_exclusive(
    isolated_user_dir: Path, tiny_repo: tuple[Path, str, str]
) -> None:
    path, _c1, _c2 = tiny_repo
    r = _run(
        [
            "gc",
            "--repo-path",
            str(path),
            "--drop",
            "HEAD",
            "main",  # positional → --keep
        ],
        isolated_user_dir,
    )
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr.lower()
