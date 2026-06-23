"""Shared test fixtures for rbtr."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest
import structlog
from pytest_mock import MockerFixture

from rbtr.config import config


class StubModel:
    """Deterministic model stub for tests.

    Satisfies the `llama_cpp.Llama` interface the `Reranker` uses
    and the concurrency-detection subclass in
    `test_embed_search_contention`.  Returns `[0.1] * EMBED_DIM` for
    every embedding call.
    """

    EMBED_DIM = 8

    def n_ctx(self) -> int:
        return 2048

    def tokenize(self, b: bytes) -> list[int]:
        return list(range(len(b) // 4))

    def embed(
        self,
        text: str | list[str],
        *,
        normalize: bool = True,
        truncate: bool = False,
    ) -> list[float] | list[list[float]]:
        if isinstance(text, list):
            return [[0.1] * self.EMBED_DIM for _ in text]
        return [0.1] * self.EMBED_DIM

    def close(self) -> None:
        pass


def run_cli(args: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run `python -m rbtr <args>` as a subprocess."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "rbtr", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ── Git commit projections (pure; over caller-supplied paths) ──────


def build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
    """Build a nested tree from `{"dir/file.py": b"..."}` paths.

    Pure projection over the caller-supplied `files` mapping; does
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


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give a test its own on-disk index DB.

    Subprocess CLI smoke tests can't share an in-memory store, so
    they read the on-disk DB at `config.db_path`.  A single shared
    file makes xdist workers collide on DuckDB's write lock and
    lets one test's repos leak into another's cross-repo search.
    Repoint both the subprocess (via the `RBTR_DATA_DIR` env it
    inherits) and the in-process `config` singleton (used by
    `IndexStore.from_config`) at a per-test directory.
    """
    data_dir = tmp_path / "index"
    data_dir.mkdir()
    monkeypatch.setenv("RBTR_DATA_DIR", str(data_dir))
    monkeypatch.setattr(config, "data_dir", data_dir)
    return data_dir


@pytest.fixture(autouse=True)
def _test_dirs() -> None:
    """Validate and create the dirs set by ``[tool.pytest_env]``."""
    sys_tmp = Path(tempfile.gettempdir()).resolve()
    slash_tmp = Path("/tmp").resolve()  # noqa: S108
    for key in ("RBTR_DATA_DIR", "RBTR_CONFIG_DIR", "RBTR_LOG_DIR"):
        d = Path(os.environ[key]).resolve()
        assert d.is_relative_to(sys_tmp) or d.is_relative_to(slash_tmp), f"{key}={d} not under temp"
        assert "test" in d.name, f"{key}={d} missing 'test'"
        d.mkdir(parents=True, exist_ok=True)


@pytest.fixture(name="log_output")
def fixture_log_output() -> structlog.testing.LogCapture:
    """Capture structlog events emitted during a test.

    Assert on `log_output.entries` — a list of event dicts, each with
    `event`, `log_level`, and any bound/contextvar keys.
    """
    return structlog.testing.LogCapture()


@pytest.fixture(autouse=True)
def _configure_structlog(log_output: structlog.testing.LogCapture) -> Generator[None]:
    """Route structlog through `LogCapture` for every test.

    Without this, unconfigured structlog prints to stdout in-process.
    `merge_contextvars` is included so correlation tests (request_id /
    job_id bound via contextvars) see those keys.  Caching stays off so
    per-test reconfiguration is honoured; `reset_defaults` restores the
    real config on teardown.
    """
    structlog.configure(
        processors=[structlog.contextvars.merge_contextvars, log_output],
        cache_logger_on_first_use=False,
    )
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


@pytest.fixture
def stub_embedding_model(mocker: MockerFixture) -> None:
    """Opt-in: replace the embedding model loader with the deterministic stub.

    By default tests load the real (tiny, env-configured) embedding
    model through the normal `_load_model` path.  Request this fixture
    for tests that drive the model from multiple threads — the native
    `llama_cpp` model aborts under concurrent access (it is not
    thread-safe the way this pure-Python stub is), and such tests are
    exercising concurrency, not embedding quality.
    """
    mocker.patch("rbtr.index.embeddings._load_model", return_value=StubModel())


@pytest.fixture(autouse=True)
def _stub_reranker(mocker: MockerFixture) -> None:
    """Never load the real GGUF reranker model during tests."""
    mocker.patch("rbtr.index.reranker._load_model", return_value=StubModel())
