"""Tests for `rbtr.index.gc.run_gc`.

Covers every mode against a tiny seeded repo with three commits.
Uses real pygit2 repositories and a real in-memory IndexStore so
we exercise the actual code path, not mocks.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.messages import GcMode
from rbtr.errors import RbtrError
from rbtr.git import worktree_tree_sha
from rbtr.index.gc import run_gc, run_gc_all
from rbtr.index.models import ChunkKind, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore

from .conftest import make_chunk, make_snap


@dataclass(frozen=True)
class GcFixture:
    """A real git repo, an in-memory store, and the three SHAs indexed."""

    repo: pygit2.Repository
    store: IndexStore
    repo_id: int
    c1: str  # first commit on main, tagged as v1
    c2: str  # middle commit, no longer reachable by any named ref
    c3: str  # HEAD on main


@pytest.fixture
def gc_signature() -> pygit2.Signature:
    return pygit2.Signature("t", "t@t.t")


@pytest.fixture
def gc_repo(
    tmp_path: Path, gc_signature: pygit2.Signature
) -> tuple[pygit2.Repository, str, str, str]:
    """Three-commit repo.  Tags `v1` at the first commit so KEEP_REFS has
    a non-HEAD ref to preserve; leaves the middle commit unreachable so
    HEAD_ONLY has something to drop."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False, initial_head="main")
    shas: list[str] = []
    for i in range(3):
        tb = repo.TreeBuilder()
        tb.insert(
            "a.py",
            repo.create_blob(f"x = {i + 1}\n".encode()),
            pygit2.GIT_FILEMODE_BLOB,
        )
        parents = [repo.head.target] if not repo.head_is_unborn else []
        new = repo.create_commit(
            "refs/heads/main",
            gc_signature,
            gc_signature,
            f"c{i}",
            tb.write(),
            parents,
        )
        shas.append(str(new))
    repo.create_reference("refs/tags/v1", shas[0])
    return repo, shas[0], shas[1], shas[2]


@pytest.fixture
def gc(
    gc_repo: tuple[pygit2.Repository, str, str, str],
) -> Generator[GcFixture]:
    """Index all three commits into a fresh in-memory store."""
    repo, c1, c2, c3 = gc_repo
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/repo")
    for i, sha in enumerate((c1, c2, c3)):
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
    yield GcFixture(repo=repo, store=store, repo_id=repo_id, c1=c1, c2=c2, c3=c3)
    store.close()


# ── HEAD_ONLY ────────────────────────────────────────────────────────


def test_head_only_keeps_head_and_drops_rest(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False
    )
    assert counts.commits == 2  # c1 and c2 dropped
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── WATCHED (default) ───────────────────────────────────────


def test_watched_keeps_watched_refs_and_head(gc: GcFixture) -> None:
    """Keeps HEAD plus every watched ref — branch/tag and bare SHA alike."""
    with gc.store.session() as ws:
        ws.add_watched_refs(gc.repo_id, ["v1", gc.c2])  # a tag and a bare SHA
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.WATCHED, refs=[], dry_run=False
    )
    assert counts.commits == 0
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True  # watched tag v1
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True  # watched bare SHA
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD


def test_watched_keeps_all_branches_drops_unreferenced(gc: GcFixture) -> None:
    """With no watched refs, WATCHED still keeps HEAD + branches/tags;
    only the genuinely unreferenced commit is dropped."""
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.WATCHED, refs=[], dry_run=False
    )
    assert counts.commits == 1  # only c2 (unreachable) dropped
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True  # tag v1 kept
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD kept
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


def test_watched_only_drops_unwatched_branches_and_tags(gc: GcFixture) -> None:
    """WATCHED_ONLY keeps only HEAD + watched refs; an unwatched tag/branch
    is dropped (unlike the default which keeps all branches/tags)."""
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.WATCHED_ONLY, refs=[], dry_run=False
    )
    assert counts.commits == 2  # c1 (tag v1) and c2 dropped; only HEAD kept
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── KEEP ──────────────────────────────────────────────────────────────


def test_keep_preserves_listed_refs_and_head(gc: GcFixture) -> None:
    # Keep v1 (c1). HEAD (c3) is kept implicitly.
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.KEEP, refs=["v1"], dry_run=False
    )
    assert counts.commits == 1
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


def test_keep_with_no_refs_is_head_only(gc: GcFixture) -> None:
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.KEEP, refs=[], dry_run=False)
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD kept
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── ORPHANS ──────────────────────────────────────────────────────────


def test_orphans_never_drops_indexed_commits(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False
    )
    assert counts.commits == 0
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True


def test_orphans_sweeps_crashed_residue(gc: GcFixture) -> None:
    # Simulate a crashed build: snapshot without mark_indexed.
    with gc.store.session() as ws:
        ws.insert_snapshots(
            [Snapshot(commit_sha="crashed", file_path="x.py", blob_sha="bx")],
            repo_id=gc.repo_id,
        )
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False)
    # Orphan snapshot is gone (swept on session entry or by GC).
    assert gc.store.count_snapshots_for_commit(gc.repo_id, "crashed") == 0
    # Completed commits untouched.
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True


# ── dry-run ──────────────────────────────────────────────────────────


def test_dry_run_reports_without_writing(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=True
    )
    assert counts.commits == 2
    # Dry-run predicts the drop set's freed chunks (read-only graph query):
    # both dropped commits' chunks are unshared, so both are freed.
    assert counts.chunks == 2
    # Nothing actually dropped.
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True


def test_gc_frees_only_unshared_chunks(gc: GcFixture) -> None:
    """GC frees a dropped commit's chunk only when no other repo shares it.

    A second repo references commit c1's blob, so dropping c1 and c2 in
    repo 1 frees c2's unshared chunk but leaves c1's — still referenced by
    the other repo — in the pool.
    """
    with gc.store.session() as ws:
        other = ws.register_repo("/other")
        ws.insert_snapshots(
            [Snapshot(commit_sha="other_head", file_path="a.py", blob_sha="b0")],
            repo_id=other,
        )
        ws.mark_indexed(other, "other_head")

    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False
    )
    assert counts.chunks == 1  # only c2's unshared chunk freed
    assert gc.store.count_chunks("other_head", repo_id=other) > 0  # shared chunk survives


def test_gc_reports_reclaimed_orphan_chunks(gc: GcFixture) -> None:
    """Pre-existing orphans (e.g. left by a forgotten repo) are reclaimed
    *and counted*, even when gc drops no commits — the reporting gap the
    forget->gc smoke test surfaced.
    """
    with gc.store.session() as ws:
        ws.add_watched_refs(gc.repo_id, [gc.c2])  # keep every commit -> no drops
        ws.add_chunk(make_chunk("orphan", path="z.py", blob="b_orphan"))
    assert gc.store.count_orphan_chunks() == 1

    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.WATCHED, refs=[], dry_run=False
    )

    assert counts.commits == 0  # nothing dropped
    assert counts.chunks == 1  # the orphan was reclaimed AND reported
    assert gc.store.count_orphan_chunks() == 0


# ── error handling ──────────────────────────────────────────────────


def test_unknown_ref_in_keep_raises(gc: GcFixture) -> None:
    with pytest.raises(RbtrError, match="nosuchref"):
        run_gc(
            gc.store,
            gc.repo.workdir,
            gc.repo_id,
            mode=GcMode.KEEP,
            refs=["nosuchref"],
            dry_run=False,
        )


# ── Search survives GC ───────────────────────────────────────────────


def test_search_works_after_gc(gc: GcFixture) -> None:
    """FTS and get_chunks still work after GC drops commits."""
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD (c3) survives — its chunks are still queryable.
    chunks = gc.store.get_chunks(gc.c3, repo_id=gc.repo_id)
    assert len(chunks) > 0

    # Dropped commits return nothing.
    assert gc.store.get_chunks(gc.c1, repo_id=gc.repo_id) == []


# ── Edge cases ───────────────────────────────────────────────────────


def test_gc_preserves_current_worktree_tree_sha(gc: GcFixture) -> None:
    """GC preserves the current worktree tree SHA."""
    # Dirty the working tree to get a real tree SHA.
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("dirty\n")
    tree_sha = worktree_tree_sha(gc.repo.workdir)
    assert tree_sha is not None

    # Re-register with real path and index the tree SHA.
    with gc.store.session() as ws:
        repo_id = ws.register_repo(gc.repo.workdir)
        ws.mark_indexed(repo_id, gc.c3)  # HEAD
        ws.mark_indexed(repo_id, tree_sha)

    run_gc(gc.store, gc.repo.workdir, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD and current tree SHA both preserved.
    assert gc.store.has_indexed(repo_id, gc.c3) is True
    assert gc.store.has_indexed(repo_id, tree_sha) is True


def test_gc_drops_stale_worktree_tree_sha(gc: GcFixture) -> None:
    """GC drops stale worktree tree SHAs (not the current one)."""
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs

    # Create a stale tree SHA.
    (workdir / "a.py").write_text("stale\n")
    stale_sha = worktree_tree_sha(gc.repo.workdir)
    assert stale_sha is not None

    # Now edit again to get a different current tree SHA.
    (workdir / "a.py").write_text("current\n")
    current_sha = worktree_tree_sha(gc.repo.workdir)
    assert current_sha is not None
    assert stale_sha != current_sha

    # Register and index both.
    with gc.store.session() as ws:
        repo_id = ws.register_repo(gc.repo.workdir)
        ws.mark_indexed(repo_id, gc.c3)  # HEAD
        ws.mark_indexed(repo_id, stale_sha)
        ws.mark_indexed(repo_id, current_sha)

    run_gc(gc.store, gc.repo.workdir, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD and current tree SHA preserved; stale dropped.
    assert gc.store.has_indexed(repo_id, gc.c3) is True
    assert gc.store.has_indexed(repo_id, current_sha) is True
    assert gc.store.has_indexed(repo_id, stale_sha) is False


def test_gc_unborn_head_raises(tmp_path: Path) -> None:
    """GC on a repo with no commits (unborn HEAD) raises RbtrError."""
    repo = pygit2.init_repository(str(tmp_path / "empty"), bare=False, initial_head="main")
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/empty")

    with pytest.raises(RbtrError, match="no commits"):
        run_gc(store, repo.workdir, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    store.close()


# ── Global GC (no --repo-path): reclaim across every registered repo ──


@dataclass(frozen=True)
class GlobalGcFixture:
    """Two real repos registered at their *real* workdirs, each with a
    droppable commit `c0` and `HEAD` `c1`. A chunk is shared between repo
    A's droppable commit and repo B's HEAD, so HEAD-only GC must keep it."""

    store: IndexStore
    a_id: int
    b_id: int
    a0: str  # repo A droppable commit
    a1: str  # repo A HEAD
    b0: str  # repo B droppable commit
    b1: str  # repo B HEAD


@pytest.fixture
def global_gc(tmp_path: Path, gc_signature: pygit2.Signature) -> Generator[GlobalGcFixture]:
    def build(name: str) -> tuple[pygit2.Repository, str, str]:
        repo = pygit2.init_repository(str(tmp_path / name), bare=False, initial_head="main")
        shas: list[str] = []
        for i in range(2):
            tb = repo.TreeBuilder()
            tb.insert("a.py", repo.create_blob(f"{name}{i}\n".encode()), pygit2.GIT_FILEMODE_BLOB)
            parents = [repo.head.target] if not repo.head_is_unborn else []
            shas.append(
                str(
                    repo.create_commit(
                        "refs/heads/main", gc_signature, gc_signature, f"c{i}", tb.write(), parents
                    )
                )
            )
        return repo, shas[0], shas[1]

    repo_a, a0, a1 = build("a")
    repo_b, b0, b1 = build("b")
    store = IndexStore(writable=True)

    with store.session() as ws:
        a_id = ws.register_repo(repo_a.workdir)
        b_id = ws.register_repo(repo_b.workdir)
        # Content-addressed chunks (the shared one is a single row). A chunk
        # is referenced by a snapshot only when (blob_sha, file_path) match,
        # so the shared chunk lives on b.py — as do its snapshots.
        for c in (
            make_chunk("k_a_old", path="a.py", blob="a_old"),
            make_chunk("k_a_head", path="a.py", blob="a_head"),
            make_chunk("k_b_old", path="a.py", blob="b_old"),
            make_chunk("k_b_head", path="a.py", blob="b_head"),
            make_chunk("k_shared", path="b.py", blob="shared"),
        ):
            ws.add_chunk(c)
        # Repo A: c0 references a_old + shared; HEAD references a_head.
        ws.insert_snapshots(
            [make_snap(a0, "a.py", "a_old"), make_snap(a0, "b.py", "shared")], repo_id=a_id
        )
        ws.insert_snapshots([make_snap(a1, "a.py", "a_head")], repo_id=a_id)
        ws.mark_indexed(a_id, a0)
        ws.mark_indexed(a_id, a1)
        # Repo B: c0 references b_old; HEAD references shared + b_head.
        ws.insert_snapshots([make_snap(b0, "a.py", "b_old")], repo_id=b_id)
        ws.insert_snapshots(
            [make_snap(b1, "a.py", "b_head"), make_snap(b1, "b.py", "shared")], repo_id=b_id
        )
        ws.mark_indexed(b_id, b0)
        ws.mark_indexed(b_id, b1)
    yield GlobalGcFixture(store=store, a_id=a_id, b_id=b_id, a0=a0, a1=a1, b0=b0, b1=b1)
    store.close()


def test_run_gc_all_drops_unreferenced_in_every_repo(global_gc: GlobalGcFixture) -> None:
    """Global GC drops each repo's unreferenced commits and keeps its ref
    tips — in the default (watched) reclamation that global GC is limited
    to. Here each repo's `c0` is unreachable, so it is dropped everywhere."""
    _counts, repos = run_gc_all(global_gc.store, mode=GcMode.WATCHED, refs=[], dry_run=False)
    s = global_gc.store
    assert repos == 2  # both registered repos collected
    assert s.has_indexed(global_gc.a_id, global_gc.a1) is True
    assert s.has_indexed(global_gc.b_id, global_gc.b1) is True
    assert s.has_indexed(global_gc.a_id, global_gc.a0) is False
    assert s.has_indexed(global_gc.b_id, global_gc.b0) is False


def test_run_gc_all_keeps_chunk_shared_across_repos(global_gc: GlobalGcFixture) -> None:
    """A chunk a dropped commit referenced survives while another repo's
    surviving commit still references it; only the unshared ones are freed."""
    counts, _repos = run_gc_all(global_gc.store, mode=GcMode.WATCHED, refs=[], dry_run=False)
    # a_old and b_old freed; shared kept (repo B HEAD still references it).
    assert counts.chunks == 2
    # Shared chunk still queryable from repo B's HEAD.
    assert global_gc.store.count_chunks(global_gc.b1, repo_id=global_gc.b_id) > 0


def test_run_gc_all_skips_repo_with_vanished_path(
    global_gc: GlobalGcFixture, tmp_path: Path
) -> None:
    """A registered repo whose path no longer resolves is skipped, not
    crashed; live repos are still collected."""
    with global_gc.store.session() as ws:
        ws.register_repo(str(tmp_path / "gone"))  # never created on disk
    _counts, repos = run_gc_all(global_gc.store, mode=GcMode.WATCHED, refs=[], dry_run=False)
    assert repos == 2  # the vanished repo skipped, the two live ones collected
    # Live repos were still GC'd.
    assert global_gc.store.has_indexed(global_gc.a_id, global_gc.a0) is False
    assert global_gc.store.has_indexed(global_gc.b_id, global_gc.b0) is False


def test_run_gc_all_freed_matches_physical_deletion(global_gc: GlobalGcFixture) -> None:
    """Freed count equals the physical `chunks` delta, and no orphan
    survives the pass (the global-aggregation consistency guard)."""
    cur = global_gc.store._cursor
    before = cur.execute("SELECT count(*) FROM chunks").fetchone()
    counts, _repos = run_gc_all(global_gc.store, mode=GcMode.WATCHED, refs=[], dry_run=False)
    after = cur.execute("SELECT count(*) FROM chunks").fetchone()
    assert before is not None
    assert after is not None
    assert counts.chunks == before[0] - after[0]
    assert global_gc.store.count_orphan_chunks() == 0
