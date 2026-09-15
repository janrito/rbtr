"""Tests for embed_index and embedding helpers."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture, MockType

from rbtr.config import config
from rbtr.domain.models import SnapshotRef
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import EmbedResult
from rbtr.index.store import IndexStore


@pytest.fixture
def stub_embedder(mocker: MockerFixture) -> MockType:
    """MagicMock embedder with deterministic vectors."""
    embedder = mocker.MagicMock()
    embedder.embed = mocker.MagicMock(
        side_effect=lambda texts: [
            EmbedResult(vector=[0.1, 0.2, 0.3], truncated=False) for _ in texts
        ]
    )
    return embedder


def test_embed_batch_failure_skips_batch(
    tmp_path: Path, store: IndexStore, mocker: MockerFixture
) -> None:
    """When a batch persistently fails, the loop terminates and other batches still succeed."""

    def _flaky_embed(texts: list[str]) -> list[EmbedResult]:
        # Fail any batch that contains func_0 (deterministic, persistent).
        if any("func_0\n" in t for t in texts):
            raise RuntimeError("GPU OOM")  # noqa: TRY003 — test simulates a native error
        return [EmbedResult(vector=[0.1, 0.2, 0.3], truncated=False) for _ in texts]

    embedder = mocker.MagicMock()
    embedder.embed = mocker.MagicMock(side_effect=_flaky_embed)

    repo = pygit2.init_repository(str(tmp_path / "batch_fail"), bare=False, initial_head="main")
    # Create enough symbols to span multiple batches (batch_size=32).
    lines = "\n".join(f"def func_{i}(): pass" for i in range(40))
    (tmp_path / "batch_fail" / "funcs.py").write_text(lines)
    idx = repo.index
    idx.add("funcs.py")
    idx.write()
    tree_oid = idx.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    sha = str(repo.head.target)
    result = build_index(repo.workdir, sha, store)
    assert result.stats.total_chunks > 0

    embed_index(store, sha, repo_id=1, embedder=embedder)

    chunks = store.get_chunks(sha, repo_id=1)
    embedded = [c for c in chunks if c.has_embedding]
    unembedded = [c for c in chunks if not c.has_embedding]
    assert len(embedded) > 0, "Successful batch should have embeddings"
    assert len(unembedded) > 0, "Failed batch should lack embeddings"


def test_embed_index_total_failure_is_nonfatal(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, mocker: MockerFixture
) -> None:
    """When every embed batch fails, embed_index returns 0 and the structural index is intact."""
    build_index(git_repo.workdir, snapshot_sha, store)
    embedder = mocker.MagicMock()
    embedder.embed = mocker.MagicMock(side_effect=RuntimeError("GPU OOM"))

    result = embed_index(store, snapshot_sha, repo_id=1, embedder=embedder)

    assert result == 0
    chunks = store.get_chunks(snapshot_sha, repo_id=1)
    assert len(chunks) > 0
    assert all(not c.has_embedding for c in chunks)


def test_build_index_without_embedder_leaves_embeddings_null(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """build_index marks commit indexed with FTS, but all embeddings are NULL."""
    result = build_index(git_repo.workdir, snapshot_sha, store)

    assert store.has_indexed(1, snapshot_sha)
    assert result.stats.total_chunks > 0

    # FTS works.
    fts_results = store.match_fulltext(snapshot_sha, "helper", repo_id=1)
    assert len(fts_results) > 0

    # All embeddings are NULL (no embedder was provided).
    chunks = store.get_chunks(snapshot_sha, repo_id=1)
    assert all(not c.has_embedding for c in chunks)


@pytest.mark.parametrize("page_size", [1000, 2], ids=["one page", "several pages"])
def test_embed_index_populates_embeddings(
    git_repo: pygit2.Repository,
    store: IndexStore,
    snapshot_sha: str,
    stub_embedder: MockType,
    page_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_index fills embedding vectors for an already-indexed commit.

    Run again with a page size below the chunk count, so the second page
    is reached: at the default of 1000 this fixture fits in one.  A page
    boundary that dropped the remainder, or embedded a page twice, shows
    up in the count.
    """
    monkeypatch.setattr(config, "embedding_page_size", page_size)
    build_index(git_repo.workdir, snapshot_sha, store)

    # All NULL before embed.
    chunks_before = store.get_chunks(snapshot_sha, repo_id=1)
    assert all(not c.has_embedding for c in chunks_before)

    embedded = embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)

    assert embedded == len(chunks_before)
    chunks_after = store.get_chunks(snapshot_sha, repo_id=1)
    assert all(c.has_embedding for c in chunks_after)


def test_embed_index_is_incremental(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, stub_embedder: MockType
) -> None:
    """embed_index skips already-embedded chunks."""
    build_index(git_repo.workdir, snapshot_sha, store)

    # First pass embeds all.
    first = embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)
    assert first > 0

    stub_embedder.embed.reset_mock()

    # Second pass is a no-op.
    second = embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)
    assert second == 0
    stub_embedder.embed.assert_not_called()


def test_a_fresh_build_has_everything_left_to_embed(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """A build writes chunks but no vectors, so all of them are outstanding."""
    build_index(git_repo.workdir, snapshot_sha, store)
    counts = store.chunk_counts_for_snapshot(SnapshotRef(repo_id=1, snapshot_sha=snapshot_sha))
    assert counts.total > 0
    assert counts.unembedded == counts.total


def test_embedding_leaves_nothing_outstanding(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, stub_embedder: MockType
) -> None:
    """Once `embed_index` has run, no chunk is left without a vector."""
    build_index(git_repo.workdir, snapshot_sha, store)
    embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)

    counts = store.chunk_counts_for_snapshot(SnapshotRef(repo_id=1, snapshot_sha=snapshot_sha))
    assert counts.is_fully_embedded


def test_the_work_list_names_only_chunks_without_vectors(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """The one pass over the embedding column decides what is outstanding."""
    build_index(git_repo.workdir, snapshot_sha, store)
    ref = SnapshotRef(repo_id=1, snapshot_sha=snapshot_sha)

    chunks = store.get_chunks(snapshot_sha, repo_id=1)
    first_chunk = chunks[0]
    with store.session() as ws:
        ws.update_embeddings([first_chunk.id], [[0.1, 0.2, 0.3]])

    outstanding = store.unembedded_chunk_ids(ref)
    assert len(outstanding) == len(chunks) - 1
    assert first_chunk.id not in outstanding


def test_a_page_of_chunks_is_fetched_by_id(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """Pages come back by id, and an id that no longer resolves is dropped.

    A chunk can be collected between the work list being drawn up and a
    page of it being read, so the page comes back shorter and the loop
    carries on.
    """
    build_index(git_repo.workdir, snapshot_sha, store)
    ref = SnapshotRef(repo_id=1, snapshot_sha=snapshot_sha)
    outstanding = store.unembedded_chunk_ids(ref)
    assert len(outstanding) > 2, "fixture must hold enough chunks to page"

    page = store.get_chunks_by_id(ref, outstanding[:2])
    assert [c.id for c in page] == outstanding[:2]

    assert store.get_chunks_by_id(ref, ["gone", *outstanding[:1]]) == page[:1]


def test_build_then_embed_full_idempotency(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, stub_embedder: MockType
) -> None:
    """Build + embed twice.  Second run is a complete no-op."""
    # First pass: build + embed.
    build_index(git_repo.workdir, snapshot_sha, store)
    embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)

    chunks_1 = store.get_chunks(snapshot_sha, repo_id=1)
    ids_1 = {c.id for c in chunks_1}
    edges_1 = store.get_edges(snapshot_sha, repo_id=1)
    embeddings_1 = {c.id: c.has_embedding for c in chunks_1}

    # Second pass: build + embed.
    r2 = build_index(git_repo.workdir, snapshot_sha, store)
    assert r2.stats.skipped_files == r2.stats.total_files
    assert r2.stats.parsed_files == 0

    stub_embedder.embed.reset_mock()
    second_embed = embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)
    assert second_embed == 0
    stub_embedder.embed.assert_not_called()

    # State is identical.
    chunks_2 = store.get_chunks(snapshot_sha, repo_id=1)
    ids_2 = {c.id for c in chunks_2}
    edges_2 = store.get_edges(snapshot_sha, repo_id=1)
    embeddings_2 = {c.id: c.has_embedding for c in chunks_2}

    assert ids_1 == ids_2
    assert len(edges_1) == len(edges_2)
    assert embeddings_1 == embeddings_2


def test_build_without_embed_then_build_with_embed(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, stub_embedder: MockType
) -> None:
    """Build (no embed) → build (with embed) → third pass is no-op."""
    # First build — no embedding.
    build_index(git_repo.workdir, snapshot_sha, store)
    chunks_no_embed = store.get_chunks(snapshot_sha, repo_id=1)
    assert len(chunks_no_embed) > 0
    assert all(not c.has_embedding for c in chunks_no_embed), "All embeddings should be NULL"
    assert store.get_edges(snapshot_sha, repo_id=1), "Edges should exist"

    # Second build — chunking is idempotent (all files skipped).
    r2 = build_index(git_repo.workdir, snapshot_sha, store)
    assert r2.stats.skipped_files == r2.stats.total_files

    # Now embed.
    embedded = embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)
    assert embedded > 0

    chunks_after_embed = store.get_chunks(snapshot_sha, repo_id=1)
    assert all(c.has_embedding for c in chunks_after_embed), "All should be embedded"

    # Third pass: build + embed are both no-ops.
    r3 = build_index(git_repo.workdir, snapshot_sha, store)
    assert r3.stats.skipped_files == r3.stats.total_files

    stub_embedder.embed.reset_mock()
    third_embed = embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)
    assert third_embed == 0
    stub_embedder.embed.assert_not_called()
