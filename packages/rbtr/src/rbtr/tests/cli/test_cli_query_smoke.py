"""End-to-end smoke tests for query subcommands via subprocess.

Exercises the full CLI → inline-fallback path (no daemon running).
Seeds an IndexStore directly instead of running `rbtr index` to
keep the test fast and avoid depending on the GGUF embedding
model — search is tested through the ZMQ handler path instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.index.models import ChunkKind, Edge, EdgeKind, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from rbtr.tests.conftest import run_cli

# ── Fixtures ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SeededRepo:
    """A real git repo on disk, seeded into an IndexStore."""

    path: Path
    c1: str
    c2: str


@pytest.fixture
def seeded_repo(tmp_path: Path, isolated_db: Path) -> SeededRepo:
    """A real git repo with two commits, seeded into the IndexStore.

    Commit 1: src/config.py with `load_config`, src/app.py with
    `Application` class and an import of `load_config`.

    Commit 2: src/config.py with `load_config` (modified body).
    """
    repo_path = tmp_path / "repo"
    repo = pygit2.init_repository(str(repo_path), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")

    # ── Commit 1 ─────────────────────────────────────────────
    tb = repo.TreeBuilder()

    src_tb = repo.TreeBuilder()
    src_tb.insert(
        "config.py",
        repo.create_blob(b"def load_config(path):\n    return open(path).read()\n"),
        pygit2.GIT_FILEMODE_BLOB,
    )
    src_tb.insert(
        "app.py",
        repo.create_blob(b"from config import load_config\n\nclass Application:\n    pass\n"),
        pygit2.GIT_FILEMODE_BLOB,
    )
    tb.insert("src", src_tb.write(), pygit2.GIT_FILEMODE_TREE)
    c1 = str(repo.create_commit("refs/heads/main", sig, sig, "c1", tb.write(), []))

    # ── Commit 2 — modified load_config ──────────────────────
    tb2 = repo.TreeBuilder()
    src_tb2 = repo.TreeBuilder()
    src_tb2.insert(
        "config.py",
        repo.create_blob(
            b"def load_config(path):\n    with open(path) as f:\n        return f.read()\n"
        ),
        pygit2.GIT_FILEMODE_BLOB,
    )
    src_tb2.insert(
        "app.py",
        repo.create_blob(b"from config import load_config\n\nclass Application:\n    pass\n"),
        pygit2.GIT_FILEMODE_BLOB,
    )
    tb2.insert("src", src_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    parent = repo.get(c1)
    assert parent is not None
    c2 = str(repo.create_commit("refs/heads/main", sig, sig, "c2", tb2.write(), [parent.id]))

    # ── Seed the IndexStore ──────────────────────────────────
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(str(repo_path))

    func_content_v1 = "def load_config(path):\n    return open(path).read()\n"
    func_name = "load_config"
    class_content = "class Application:\n    pass\n"
    class_name = "Application"
    import_content = "from config import load_config"
    import_name = "from config import load_config"
    func_content_v2 = "def load_config(path):\n    with open(path) as f:\n        return f.read()\n"

    chunks_c1 = [
        TokenisedChunk(
            id="fn_config_v1",
            repo_id=repo_id,
            blob_sha="blob_config_v1",
            file_path="src/config.py",
            kind=ChunkKind.FUNCTION,
            name=func_name,
            content=func_content_v1,
            content_tokens=tokenise_code(func_content_v1),
            name_tokens=tokenise_code(func_name),
            line_start=1,
            line_end=2,
        ),
        TokenisedChunk(
            id="cls_app",
            repo_id=repo_id,
            blob_sha="blob_app",
            file_path="src/app.py",
            kind=ChunkKind.CLASS,
            name=class_name,
            content=class_content,
            content_tokens=tokenise_code(class_content),
            name_tokens=tokenise_code(class_name),
            line_start=3,
            line_end=4,
        ),
        TokenisedChunk(
            id="imp_config",
            repo_id=repo_id,
            blob_sha="blob_app",
            file_path="src/app.py",
            kind=ChunkKind.IMPORT,
            name=import_name,
            content=import_content,
            content_tokens=tokenise_code(import_content),
            name_tokens=tokenise_code(import_name),
            line_start=1,
            line_end=1,
        ),
    ]

    chunks_c2 = [
        TokenisedChunk(
            id="fn_config_v2",
            repo_id=repo_id,
            blob_sha="blob_config_v2",
            file_path="src/config.py",
            kind=ChunkKind.FUNCTION,
            name=func_name,
            content=func_content_v2,
            content_tokens=tokenise_code(func_content_v2),
            name_tokens=tokenise_code(func_name),
            line_start=1,
            line_end=3,
        ),
        chunks_c1[1],  # class unchanged
        chunks_c1[2],  # import unchanged
    ]

    edges_c1 = [Edge(source_id="imp_config", target_id="fn_config_v1", kind=EdgeKind.IMPORTS)]
    edges_c2 = [Edge(source_id="imp_config", target_id="fn_config_v2", kind=EdgeKind.IMPORTS)]

    with store.session() as ws:
        for c in chunks_c1:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                Snapshot(commit_sha=c1, file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks_c1
            ],
            repo_id=repo_id,
        )
        ws.insert_edges(edges_c1, c1, repo_id=repo_id)
        ws.mark_indexed(repo_id, c1)

    with store.session() as ws:
        for c in chunks_c2:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                Snapshot(commit_sha=c2, file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks_c2
            ],
            repo_id=repo_id,
        )
        ws.insert_edges(edges_c2, c2, repo_id=repo_id)
        ws.mark_indexed(repo_id, c2)

    store.close()
    return SeededRepo(path=repo_path, c1=c1, c2=c2)


# ── Tests ────────────────────────────────────────────────────────────


def test_status_shows_indexed_refs(seeded_repo: SeededRepo) -> None:
    r = run_cli(["--json", "status", "--repo-path", str(seeded_repo.path)])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["kind"] == "status"
    shas = {ref["sha"] for ref in payload["indexed_refs"]}
    assert seeded_repo.c1 in shas or seeded_repo.c2 in shas


def test_read_symbol_returns_source(seeded_repo: SeededRepo) -> None:
    r = run_cli(["--json", "read-symbol", "load_config", "--repo-path", str(seeded_repo.path)])
    assert r.returncode == 0, r.stderr
    chunks = json.loads(r.stdout)["chunks"]
    assert len(chunks) >= 1
    names = {c["name"] for c in chunks}
    assert "load_config" in names
    assert any("def load_config" in c["content"] for c in chunks)


def test_list_symbols_returns_file_outline(seeded_repo: SeededRepo) -> None:
    r = run_cli(["--json", "list-symbols", "src/config.py", "--repo-path", str(seeded_repo.path)])
    assert r.returncode == 0, r.stderr
    chunks = json.loads(r.stdout)["chunks"]
    assert len(chunks) >= 1
    names = {c["name"] for c in chunks}
    assert "load_config" in names


def test_find_refs_returns_refs(seeded_repo: SeededRepo) -> None:
    r = run_cli(["--json", "find-refs", "load_config", "--repo-path", str(seeded_repo.path)])
    assert r.returncode == 0, r.stderr
    refs = json.loads(r.stdout)["refs"]
    assert len(refs) >= 1
    # The reference is the importing chunk in src/app.py; the response
    # resolves it to a legible referrer rather than an opaque id hash.
    assert any(ref["file_path"] == "src/app.py" and ref["edge"] == "imports" for ref in refs)


def test_changed_symbols_between_commits(seeded_repo: SeededRepo) -> None:
    r = run_cli(
        [
            "--json",
            "changed-symbols",
            seeded_repo.c1,
            seeded_repo.c2,
            "--repo-path",
            str(seeded_repo.path),
        ]
    )
    assert r.returncode == 0, r.stderr
    changes = json.loads(r.stdout)["changes"]
    files = {item["chunk"]["file_path"] for item in changes}
    assert "src/config.py" in files
