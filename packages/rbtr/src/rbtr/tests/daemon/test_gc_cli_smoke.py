"""End-to-end smoke test for `rbtr gc` via subprocess.

Exercises the full CLI → inline-fallback path (no daemon running).
Seeds an IndexStore directly instead of running `rbtr index` to
keep the test fast and to avoid depending on the GGUF embedding
model during unit runs — the indexing path has its own tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.index.models import ChunkKind, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore
from rbtr.tests.conftest import run_cli


@dataclass(frozen=True)
class TinyRepo:
    path: Path
    c1: str
    c2: str


@pytest.fixture
def tiny_repo(tmp_path: Path) -> TinyRepo:
    """A real git repo with two commits."""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
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
def seeded_repo_id_both_commits(tiny_repo: TinyRepo, isolated_db: Path) -> int:
    """Seed both commits and close the store so subprocesses can open it."""
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(str(tiny_repo.path))
    for i, sha in enumerate((tiny_repo.c1, tiny_repo.c2)):
        chunk = TokenisedChunk(
            id=f"c{i}",
            blob_sha=f"b{i}",
            file_path="a.py",
            kind=ChunkKind.FUNCTION,
            name="f",
            content="",
            line_start=1,
            line_end=1,
        )
        with store.session() as ws:
            ws.add_chunk(chunk)
            ws.insert_snapshots(
                [Snapshot(commit_sha=sha, file_path="a.py", blob_sha=f"b{i}")], repo_id=repo_id
            )
            ws.mark_indexed(repo_id, sha)
    store.close()
    return repo_id


@pytest.fixture
def seeded_repo_id_first_commit(tiny_repo: TinyRepo, isolated_db: Path) -> int:
    """Seed only the first commit and close the store."""
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(str(tiny_repo.path))
    chunk = TokenisedChunk(
        id="c0",
        blob_sha="b0",
        file_path="a.py",
        kind=ChunkKind.FUNCTION,
        name="f",
        content="",
        line_start=1,
        line_end=1,
    )
    with store.session() as ws:
        ws.add_chunk(chunk)
        ws.insert_snapshots(
            [Snapshot(commit_sha=tiny_repo.c1, file_path="a.py", blob_sha="b0")], repo_id=repo_id
        )
        ws.mark_indexed(repo_id, tiny_repo.c1)
    store.close()
    return repo_id


def test_gc_default_keeps_head_drops_unreferenced(
    tiny_repo: TinyRepo,
    seeded_repo_id_both_commits: int,
) -> None:
    """Default `rbtr gc` keeps ref tips (HEAD/main = c2), drops c1."""
    repo_id = seeded_repo_id_both_commits
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path)])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["kind"] == "gc"
    assert payload["commits_dropped"] == 1  # c1 is not a ref tip

    store = IndexStore.from_config(writable=True)
    assert store.has_indexed(repo_id, tiny_repo.c1) is False
    assert store.has_indexed(repo_id, tiny_repo.c2) is True


def test_gc_dry_run_changes_nothing(
    tiny_repo: TinyRepo,
    seeded_repo_id_first_commit: int,
) -> None:
    repo_id = seeded_repo_id_first_commit
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path), "--dry-run"])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["dry_run"] is True

    store = IndexStore.from_config(writable=True)
    assert store.has_indexed(repo_id, tiny_repo.c1) is True


def test_gc_watched_only_smoke(
    tiny_repo: TinyRepo,
    seeded_repo_id_both_commits: int,
) -> None:
    """`--watched-only` parses and routes; HEAD survives."""
    repo_id = seeded_repo_id_both_commits
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path), "--watched-only"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["kind"] == "gc"
    store = IndexStore.from_config(writable=True)
    assert store.has_indexed(repo_id, tiny_repo.c2) is True  # HEAD kept
