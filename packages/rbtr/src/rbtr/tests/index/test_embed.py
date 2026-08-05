"""Tests for embed_index and embedding helpers."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture, MockType

from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import EmbedResult
from rbtr.index.store import IndexStore


@pytest.fixture
def duplicated_content_sha(tmp_path: Path, store: IndexStore) -> str:
    """Indexed snapshot holding one large module's content at two paths.

    Byte-identical content is a single content-addressed chunk row reached
    once per path, which is the shape every count over the
    `chunks`/`file_snapshots` join has to collapse.

    The module is sized so the two paths together exceed one page of
    `get_unembedded_chunks` (1000 rows).  A duplicate that shares a page
    with its twin is embedded twice and masks the shortfall; only a
    duplicate whose twin falls in a *later* page is dropped from the work
    list, which is what leaves the progress bar short of its total.
    """
    root = tmp_path / "duplicated"
    repo = pygit2.init_repository(str(root), bare=False, initial_head="main")
    source = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(600))
    idx = repo.index
    for path in ("lib.py", "vendored/lib.py"):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        idx.add(path)
    idx.write()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", idx.write_tree(), [])

    sha = str(repo.head.target)
    build_index(repo.workdir, sha, store)
    return sha


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


def test_embed_index_populates_embeddings(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, stub_embedder: MockType
) -> None:
    """embed_index fills embedding vectors for an already-indexed commit."""
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


def test_duplicated_content_counts_and_yields_one_chunk(
    duplicated_content_sha: str, store: IndexStore
) -> None:
    """A chunk reached at two paths is counted once and returned once.

    `get_chunks` deliberately fans out to one row per location, so the
    distinct ids in it are the chunk count that `count_chunks` must report
    and that the embed loop, which writes by id, has to do work for.
    """
    located = store.get_chunks(duplicated_content_sha, repo_id=1)
    distinct_ids = {c.id for c in located}
    # The fixture's two paths must actually share content, or there is
    # nothing to collapse and the test proves nothing.
    assert len(located) > len(distinct_ids)

    assert store.count_chunks(duplicated_content_sha, repo_id=1) == len(distinct_ids)

    unembedded = store.get_unembedded_chunks(repo_id=1, snapshot_sha=duplicated_content_sha)
    returned = [c.id for c in unembedded]
    assert len(returned) == len(set(returned))
    assert set(returned) == distinct_ids


def test_embed_reaches_its_total_when_content_is_duplicated(
    duplicated_content_sha: str, store: IndexStore, stub_embedder: MockType
) -> None:
    """Embedding finishes at 100% of the total it set out to cover.

    The progress total and the work done have to count the same thing, or
    the bar stalls short of the end and a genuine failure to embed becomes
    indistinguishable from duplicated content.
    """
    total = store.count_unembedded(repo_id=1, snapshot_sha=duplicated_content_sha)
    reported: list[tuple[int, int]] = []

    done = embed_index(
        store,
        duplicated_content_sha,
        repo_id=1,
        embedder=stub_embedder,
        on_progress=lambda phase, d, t: reported.append((d, t)) if phase == "embedding" else None,
    )

    assert done == total
    assert reported[-1] == (total, total)
    assert store.count_unembedded(repo_id=1, snapshot_sha=duplicated_content_sha) == 0


def test_count_unembedded_all_null(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """count_unembedded returns the total chunk count when none are embedded."""
    build_index(git_repo.workdir, snapshot_sha, store)
    total = store.count_chunks(snapshot_sha, repo_id=1)
    unembedded = store.count_unembedded(repo_id=1, snapshot_sha=snapshot_sha)
    assert unembedded == total
    assert unembedded > 0


def test_count_unembedded_after_embedding(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str, stub_embedder: MockType
) -> None:
    """count_unembedded returns 0 after all chunks are embedded."""
    build_index(git_repo.workdir, snapshot_sha, store)
    embed_index(store, snapshot_sha, repo_id=1, embedder=stub_embedder)

    assert store.count_unembedded(repo_id=1, snapshot_sha=snapshot_sha) == 0


def test_get_unembedded_chunks_returns_only_nulls(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """get_unembedded_chunks returns only chunks without embeddings."""
    build_index(git_repo.workdir, snapshot_sha, store)

    # Partially embed — embed first batch only.
    chunks = store.get_chunks(snapshot_sha, repo_id=1)
    first_chunk = chunks[0]
    with store.session() as ws:
        ws.update_embeddings([first_chunk.id], [[0.1, 0.2, 0.3]])

    unembedded = store.get_unembedded_chunks(repo_id=1, snapshot_sha=snapshot_sha)
    assert len(unembedded) == len(chunks) - 1
    assert first_chunk.id not in {c.id for c in unembedded}


def test_get_unembedded_chunks_respects_limit(
    git_repo: pygit2.Repository, store: IndexStore, snapshot_sha: str
) -> None:
    """get_unembedded_chunks limits the number of returned rows."""
    build_index(git_repo.workdir, snapshot_sha, store)
    total = store.count_unembedded(repo_id=1, snapshot_sha=snapshot_sha)
    assert total > 1  # need multiple chunks for the test to be meaningful

    limited = store.get_unembedded_chunks(repo_id=1, snapshot_sha=snapshot_sha, limit=1)
    assert len(limited) == 1


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
