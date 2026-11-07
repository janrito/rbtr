"""End-to-end smoke tests for query subcommands via subprocess.

Exercises the full CLI → inline-fallback path (no daemon running).
Seeds an Index directly instead of running `rbtr watch` to
keep the test fast and avoid depending on the GGUF embedding
model — search is tested through the ZMQ handler path instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit, run_cli


@dataclass(frozen=True)
class SeededRepo:
    """A real git repo on disk, seeded into an Index."""

    path: Path
    c1: str
    c2: str


@pytest.fixture
def seeded_repo(tmp_path: Path, isolated_db: Path) -> SeededRepo:
    """A real git repo with two commits, both built into the Index.

    Commit 1: src/config.py with `load_config`, src/app.py with
    `Application` class and an import of `load_config`.

    Commit 2: src/config.py with `load_config` (modified body).
    """
    config_v1 = b"""\
def load_config(path):
    return open(path).read()
"""
    config_v2 = b"""\
def load_config(path):
    with open(path) as f:
        return f.read()
"""
    app = b"""\
from config import load_config


class Application:
    pass
"""
    repo_path = tmp_path / "repo"
    repo = pygit2.init_repository(str(repo_path), bare=False, initial_head="main")
    c1 = make_commit(repo, {"src/config.py": config_v1, "src/app.py": app})
    c2 = make_commit(repo, {"src/config.py": config_v2, "src/app.py": app}, parents=[c1])

    index = Index.from_config(writable=True)
    path = normalise_repo_path(str(repo_path))
    build_index(path, str(c1), index)
    build_index(path, str(c2), index)
    index.close()
    return SeededRepo(path=repo_path, c1=str(c1), c2=str(c2))


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


@pytest.mark.parametrize(
    ("flags", "expected_kinds"),
    [
        (["--exclude-kind", "import"], {"class"}),
        (["--kind", "import"], {"import"}),
    ],
    ids=["exclude-kind", "kind"],
)
def test_list_symbols_filters_by_kind(
    seeded_repo: SeededRepo, flags: list[str], expected_kinds: set[str]
) -> None:
    r = run_cli(
        ["--json", "list-symbols", "src/app.py", "--repo-path", str(seeded_repo.path), *flags]
    )
    assert r.returncode == 0, r.stderr
    assert {c["kind"] for c in json.loads(r.stdout)["chunks"]} == expected_kinds


def test_list_symbols_refuses_to_include_and_exclude_at_once(seeded_repo: SeededRepo) -> None:
    r = run_cli(
        [
            *["--json", "list-symbols", "src/app.py", "--repo-path", str(seeded_repo.path)],
            *["--kind", "class", "--exclude-kind", "import"],
        ]
    )
    assert r.returncode != 0
    error = json.loads(r.stdout)
    assert error["code"] == "invalid_request"
    assert "use one" in error["message"]


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
