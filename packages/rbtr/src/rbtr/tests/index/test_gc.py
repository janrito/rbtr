"""Tests for `rbtr.index.gc.run_gc`.

Covers every mode against a tiny seeded repo with three commits.
Uses real pygit2 repositories and a real in-memory Index so
we exercise the actual code path, not mocks.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.config import config
from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import ChunkKind, FileSnapshot, GcMode, Repo
from rbtr.errors import RbtrError
from rbtr.git import normalise_repo_path, worktree_tree_sha
from rbtr.index.gc import run_gc, run_gc_all
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from .conftest import make_chunk, make_snap


@dataclass(frozen=True)
class GcFixture:
    """A real git repo, an in-memory store, and the three SHAs indexed.

    `repo_path` is the canonical root the repo is *registered* under, so
    it and `registered` always denote the same repo.
    """

    repo: pygit2.Repository
    index: Index
    repo_path: str
    registered: Repo
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
    index = Index(writable=True)
    repo_path = normalise_repo_path(repo.workdir)
    registered = index.register(repo_path)
    for i, sha in enumerate((c1, c2, c3)):
        chunk = TokenisedChunk(
            blob_sha=f"b{i}",
            file_path="a.py",
            kind=ChunkKind.FUNCTION,
            name="f",
            content="",
            line_start=1,
            line_end=1,
        )
        with index.session(registered) as ws:
            ws.add_chunk(chunk)
            ws.insert_snapshots(
                [FileSnapshot(snapshot_sha=sha, file_path="a.py", blob_sha=f"b{i}")],
            )
            ws.mark_indexed(snapshot_sha=sha)
    yield GcFixture(
        repo=repo, index=index, repo_path=repo_path, registered=registered, c1=c1, c2=c2, c3=c3
    )
    index.close()


def test_head_only_keeps_head_and_drops_rest(gc: GcFixture) -> None:
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    assert counts.snapshots == 2  # c1 and c2 dropped
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is False
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is False


def test_watched_keeps_watched_refs_and_head(gc: GcFixture) -> None:
    """Keeps HEAD plus every watched ref — branch/tag and bare SHA alike."""
    with gc.index.session(gc.registered) as ws:
        ws.add_watched_refs(["v1", gc.c2])  # a tag and a bare SHA
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.WATCHED, refs=[], dry_run=False)
    assert counts.snapshots == 0
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is True  # watched tag v1
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is True  # watched bare SHA
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True  # HEAD


def test_watched_keeps_all_branches_drops_unreferenced(gc: GcFixture) -> None:
    """With no watched refs, WATCHED still keeps HEAD + branches/tags;
    only the genuinely unreferenced commit is dropped."""
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.WATCHED, refs=[], dry_run=False)
    assert counts.snapshots == 1  # only c2 (unreachable) dropped
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is True  # tag v1 kept
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True  # HEAD kept
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is False


def test_watched_only_drops_unwatched_branches_and_tags(gc: GcFixture) -> None:
    """WATCHED_ONLY keeps only HEAD + watched refs; an unwatched tag/branch
    is dropped (unlike the default which keeps all branches/tags)."""
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.WATCHED_ONLY, refs=[], dry_run=False)
    assert counts.snapshots == 2  # c1 (tag v1) and c2 dropped; only HEAD kept
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is False
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is False


def test_keep_preserves_listed_refs_and_head(gc: GcFixture) -> None:
    # Keep v1 (c1). HEAD (c3) is kept implicitly.
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.KEEP, refs=["v1"], dry_run=False)
    assert counts.snapshots == 1
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is False


def test_keep_with_no_refs_is_head_only(gc: GcFixture) -> None:
    run_gc(gc.index, gc.repo_path, mode=GcMode.KEEP, refs=[], dry_run=False)
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True  # HEAD kept
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is False
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is False


def test_orphans_never_drops_indexed_snapshots(gc: GcFixture) -> None:
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.ORPHANS, refs=[], dry_run=False)
    assert counts.snapshots == 0
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True


def test_orphans_sweeps_crashed_residue(gc: GcFixture) -> None:
    # Simulate a crashed build: snapshot without mark_indexed.
    with gc.index.session(gc.registered) as ws:
        ws.insert_snapshots(
            [FileSnapshot(snapshot_sha="crashed", file_path="x.py", blob_sha="bx")],
        )
    run_gc(gc.index, gc.repo_path, mode=GcMode.ORPHANS, refs=[], dry_run=False)
    # Orphan snapshot is gone (swept on session entry or by GC).
    assert gc.index.count_file_snapshots(at=gc.registered.at("crashed")) == 0
    # Completed commits untouched.
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is True


def test_dry_run_reports_without_writing(gc: GcFixture) -> None:
    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=True)
    assert counts.snapshots == 2
    # Dry-run predicts the drop set's freed chunks (read-only graph query):
    # both dropped commits' chunks are unshared, so both are freed.
    assert counts.chunks == 2
    # Nothing actually dropped.
    assert gc.index.has_indexed(at=gc.registered.at(gc.c1)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c2)) is True
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True


def test_gc_frees_only_unshared_chunks(gc: GcFixture, tmp_path: Path) -> None:
    """GC frees a dropped commit's chunk only when no other checkout shares it.

    A linked worktree of the repo references commit c1's blob, so dropping
    c1 and c2 in repo 1 frees c2's unshared chunk but leaves c1's — still
    referenced by the other checkout of the project — in the pool.
    """
    worktree = tmp_path / "other"
    gc.repo.add_worktree("other", str(worktree))
    other = gc.index.register(normalise_repo_path(str(worktree)))
    with gc.index.session(other) as ws:
        ws.insert_snapshots(
            [FileSnapshot(snapshot_sha="other_head", file_path="a.py", blob_sha="b0")],
        )
        ws.mark_indexed(snapshot_sha="other_head")

    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    assert counts.chunks == 1  # only c2's unshared chunk freed
    other_head = other.at("other_head")
    assert gc.index.chunk_counts_for_snapshot(at=other_head).total > 0  # shared chunk survives


def test_gc_reports_reclaimed_orphan_chunks(gc: GcFixture) -> None:
    """Pre-existing orphans (e.g. left by a forgotten repo) are reclaimed
    *and counted*, even when gc drops no commits — the reporting gap the
    forget->gc smoke test surfaced.
    """
    # An orphan arises the way production makes one: a chunk stored with
    # its claim, and the claim later removed.
    with gc.index.session(gc.registered) as ws:
        ws.add_watched_refs([gc.c2])  # keep every commit -> no drops
        ws.add_chunk(make_chunk("orphan", path="z.py", blob="b_orphan"))
        ws.insert_snapshots([make_snap("forgotten", "z.py", "b_orphan")])
    with gc.index.session(gc.registered) as ws:
        ws.replace_snapshots([], snapshot_sha="forgotten")
    assert gc.index.count_orphan_chunks() == 1

    counts = run_gc(gc.index, gc.repo_path, mode=GcMode.WATCHED, refs=[], dry_run=False)

    assert counts.snapshots == 0  # nothing dropped
    assert counts.chunks == 1  # the orphan was reclaimed AND reported
    assert gc.index.count_orphan_chunks() == 0


def test_unknown_ref_in_keep_raises(gc: GcFixture) -> None:
    with pytest.raises(RbtrError, match="nosuchref"):
        run_gc(
            gc.index,
            gc.repo_path,
            mode=GcMode.KEEP,
            refs=["nosuchref"],
            dry_run=False,
        )


def test_search_works_after_gc(gc: GcFixture) -> None:
    """FTS and chunks still work after GC drops commits."""
    run_gc(gc.index, gc.repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD (c3) survives — its chunks are still queryable.
    chunks = gc.index.chunks(at=gc.registered.at(gc.c3))
    assert len(chunks) > 0

    # Dropped commits return nothing.
    assert gc.index.chunks(at=gc.registered.at(gc.c1)) == []


def test_gc_preserves_current_worktree_tree_sha(gc: GcFixture) -> None:
    """GC preserves the current worktree tree SHA."""
    # Dirty the working tree to get a real tree SHA.
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("dirty\n")
    tree_sha = worktree_tree_sha(gc.repo.workdir, max_untracked_files=config.max_untracked_files)
    assert tree_sha is not None

    # Index HEAD and the tree SHA.
    with gc.index.session(gc.registered) as ws:
        ws.mark_indexed(snapshot_sha=gc.c3)  # HEAD
        ws.mark_indexed(snapshot_sha=tree_sha)

    run_gc(gc.index, gc.repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD and current tree SHA both preserved.
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True
    assert gc.index.has_indexed(at=gc.registered.at(tree_sha)) is True


def test_gc_drops_stale_worktree_tree_sha(gc: GcFixture) -> None:
    """GC drops stale worktree tree SHAs (not the current one)."""
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs

    # Create a stale tree SHA.
    (workdir / "a.py").write_text("stale\n")
    stale_sha = worktree_tree_sha(gc.repo.workdir, max_untracked_files=config.max_untracked_files)
    assert stale_sha is not None

    # Now edit again to get a different current tree SHA.
    (workdir / "a.py").write_text("current\n")
    current_sha = worktree_tree_sha(gc.repo.workdir, max_untracked_files=config.max_untracked_files)
    assert current_sha is not None
    assert stale_sha != current_sha

    # Index all three.
    with gc.index.session(gc.registered) as ws:
        ws.mark_indexed(snapshot_sha=gc.c3)  # HEAD
        ws.mark_indexed(snapshot_sha=stale_sha)
        ws.mark_indexed(snapshot_sha=current_sha)

    run_gc(gc.index, gc.repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD and current tree SHA preserved; stale dropped.
    assert gc.index.has_indexed(at=gc.registered.at(gc.c3)) is True
    assert gc.index.has_indexed(at=gc.registered.at(current_sha)) is True
    assert gc.index.has_indexed(at=gc.registered.at(stale_sha)) is False


def test_gc_unborn_head_raises(tmp_path: Path) -> None:
    """GC on a repo with no commits (unborn HEAD) raises RbtrError."""
    pygit2.init_repository(str(tmp_path / "empty"), bare=False, initial_head="main")
    repo_path = normalise_repo_path(str(tmp_path / "empty"))
    index = Index(writable=True)
    index.register(repo_path)

    with pytest.raises(RbtrError, match="no commits"):
        run_gc(index, repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    index.close()


@dataclass(frozen=True)
class GlobalGcFixture:
    """Two real repos registered at their *real* workdirs, each with a
    droppable commit `c0` and `HEAD` `c1`. A chunk is shared between repo
    A's droppable commit and repo B's HEAD, so HEAD-only GC must keep it."""

    index: Index
    a: Repo
    b: Repo
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
    index = Index(writable=True)

    a = index.register(repo_a.workdir)
    b = index.register(repo_b.workdir)
    # Two repositories are two projects, so each file holds its own copy of
    # the chunk both reference. A chunk is referenced by a snapshot only
    # when (blob_sha, file_path) match, so the shared chunk lives on b.py —
    # as do its snapshots.
    shared = make_chunk("k_shared", path="b.py", blob="shared")
    with index.session(a) as ws:
        for c in (
            make_chunk("k_a_old", path="a.py", blob="a_old"),
            make_chunk("k_a_head", path="a.py", blob="a_head"),
            shared,
        ):
            ws.add_chunk(c)
        # Repo A: c0 references a_old + shared; HEAD references a_head.
        ws.insert_snapshots([make_snap(a0, "a.py", "a_old"), make_snap(a0, "b.py", "shared")])
        ws.insert_snapshots([make_snap(a1, "a.py", "a_head")])
        ws.mark_indexed(snapshot_sha=a0)
        ws.mark_indexed(snapshot_sha=a1)
    with index.session(b) as ws:
        for c in (
            make_chunk("k_b_old", path="a.py", blob="b_old"),
            make_chunk("k_b_head", path="a.py", blob="b_head"),
            shared,
        ):
            ws.add_chunk(c)
        # Repo B: c0 references b_old; HEAD references shared + b_head.
        ws.insert_snapshots([make_snap(b0, "a.py", "b_old")])
        ws.insert_snapshots([make_snap(b1, "a.py", "b_head"), make_snap(b1, "b.py", "shared")])
        ws.mark_indexed(snapshot_sha=b0)
        ws.mark_indexed(snapshot_sha=b1)
    yield GlobalGcFixture(index=index, a=a, b=b, a0=a0, a1=a1, b0=b0, b1=b1)
    index.close()


def test_run_gc_all_drops_unreferenced_in_every_repo(global_gc: GlobalGcFixture) -> None:
    """Global GC drops each repo's unreferenced commits and keeps its ref
    tips — in the default (watched) reclamation that global GC is limited
    to. Here each repo's `c0` is unreachable, so it is dropped everywhere."""
    _counts, repos = run_gc_all(global_gc.index, mode=GcMode.WATCHED, dry_run=False)
    s = global_gc.index
    assert repos == 2  # both registered repos collected
    assert s.has_indexed(at=global_gc.a.at(global_gc.a1)) is True
    assert s.has_indexed(at=global_gc.b.at(global_gc.b1)) is True
    assert s.has_indexed(at=global_gc.a.at(global_gc.a0)) is False
    assert s.has_indexed(at=global_gc.b.at(global_gc.b0)) is False


def test_run_gc_all_frees_each_projects_own_copy(global_gc: GlobalGcFixture) -> None:
    """Content two projects share is a copy in each file, collected per file.

    Repo A's dropped commit was the last in A to reference the shared
    chunk, so A's copy is freed; repo B's HEAD still references its own.
    """
    counts, _repos = run_gc_all(global_gc.index, mode=GcMode.WATCHED, dry_run=False)
    # a_old, A's shared copy and b_old freed; B's shared copy kept.
    assert counts.chunks == 3
    # Shared chunk still queryable from repo B's HEAD.
    b_head = global_gc.b.at(global_gc.b1)
    assert global_gc.index.chunk_counts_for_snapshot(at=b_head).total > 0


def test_run_gc_all_skips_repo_with_vanished_path(
    global_gc: GlobalGcFixture, tmp_path: Path
) -> None:
    """A registered repo whose path no longer resolves is skipped, not
    crashed; live repos are still collected."""
    global_gc.index.register(str(tmp_path / "gone"))  # never created on disk
    _counts, repos = run_gc_all(global_gc.index, mode=GcMode.WATCHED, dry_run=False)
    assert repos == 2  # the vanished repo skipped, the two live ones collected
    # Live repos were still GC'd.
    assert global_gc.index.has_indexed(at=global_gc.a.at(global_gc.a0)) is False
    assert global_gc.index.has_indexed(at=global_gc.b.at(global_gc.b0)) is False


def test_run_gc_all_freed_matches_physical_deletion(global_gc: GlobalGcFixture) -> None:
    """Freed count equals the physical `chunks` delta, and no orphan
    survives the pass (the global-aggregation consistency guard)."""

    def chunk_rows() -> int:
        total = 0
        for repo in (global_gc.a, global_gc.b):
            with global_gc.index.reader(repo) as cur:
                total += cur.execute("SELECT count(*) FROM chunks").fetchall()[0][0]
        return total

    before = chunk_rows()
    counts, _repos = run_gc_all(global_gc.index, mode=GcMode.WATCHED, dry_run=False)
    assert counts.chunks == before - chunk_rows()
    assert global_gc.index.count_orphan_chunks() == 0


def test_post_build_cleanup_drops_a_stale_worktree_whose_object_is_gone(
    gc: GcFixture,
) -> None:
    """Pruning must not strand a worktree snapshot in the index.

    `worktree_tree_sha` writes the tree into the user's repository and
    nothing references it, so `git gc` there prunes it. The daemon's
    post-build cleanup finds worktree snapshots by asking git which
    indexed shas are tree objects, which is a question git can stop
    being able to answer about rows rbtr still holds.
    """
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs

    (workdir / "a.py").write_text("stale\n")
    stale_sha = worktree_tree_sha(gc.repo.workdir, max_untracked_files=config.max_untracked_files)
    assert stale_sha is not None
    (workdir / "a.py").write_text("current\n")
    current_sha = worktree_tree_sha(gc.repo.workdir, max_untracked_files=config.max_untracked_files)
    assert current_sha is not None

    with gc.index.session(gc.registered) as ws:
        ws.mark_indexed(snapshot_sha=gc.c3)
        ws.mark_indexed(snapshot_sha=stale_sha)
        ws.mark_indexed(snapshot_sha=current_sha)

    # What `git gc` does to an unreferenced loose object.
    loose = Path(gc.repo.path) / "objects" / stale_sha[:2] / stale_sha[2:]
    assert loose.exists(), "the tree should have been written as a loose object"
    loose.unlink()

    DaemonServer._drop_stale_worktree_shas(  # the unit under test
        gc.index, gc.registered, keep=current_sha
    )

    assert gc.index.has_indexed(at=gc.registered.at(stale_sha)) is False
    assert gc.index.has_indexed(at=gc.registered.at(current_sha)) is True


def test_run_gc_all_watched_only_drops_unwatched_refs_everywhere(
    global_gc: GlobalGcFixture,
) -> None:
    """`WATCHED_ONLY` across every repo keeps HEAD and the watch set, and
    drops what only an unwatched branch or tag held."""
    s = global_gc.index
    # Repo A's `c0` is a branch tip nobody watches; repo B watches its own.
    with s.session(global_gc.b) as ws:
        ws.add_watched_refs([global_gc.b0])

    _counts, repos = run_gc_all(s, mode=GcMode.WATCHED_ONLY, dry_run=False)

    assert repos == 2
    assert s.has_indexed(at=global_gc.a.at(global_gc.a1)) is True
    assert s.has_indexed(at=global_gc.b.at(global_gc.b1)) is True
    assert s.has_indexed(at=global_gc.a.at(global_gc.a0)) is False
    assert s.has_indexed(at=global_gc.b.at(global_gc.b0)) is True  # watched, so kept


def test_run_gc_all_watched_only_keeps_head_of_an_unwatched_repo(
    global_gc: GlobalGcFixture,
) -> None:
    """A repo indexed inline watches nothing at all. Keeping only the watch
    set must still keep its HEAD, or the repo loses everything."""
    index = global_gc.index

    run_gc_all(index, mode=GcMode.WATCHED_ONLY, dry_run=False)

    head = global_gc.a.at(global_gc.a1)
    assert index.has_indexed(at=head) is True


def test_run_gc_all_skips_a_repo_with_no_commits(
    global_gc: GlobalGcFixture, tmp_path: Path
) -> None:
    """An unborn HEAD raises from `run_gc`, so a registered empty checkout
    must be skipped rather than ending the whole pass."""
    pygit2.init_repository(str(tmp_path / "empty"), bare=False, initial_head="main")
    global_gc.index.register(normalise_repo_path(str(tmp_path / "empty")))

    _counts, repos = run_gc_all(global_gc.index, mode=GcMode.WATCHED_ONLY, dry_run=False)

    assert repos == 2  # the two real repos; the empty one skipped
    assert global_gc.index.has_indexed(at=global_gc.a.at(global_gc.a0)) is False


def test_run_gc_all_reports_each_repo_as_it_is_collected(
    global_gc: GlobalGcFixture,
) -> None:
    """Progress names the repo under collection and how far through the
    pass it is, so a long run is not silent."""
    seen: list[tuple[str, int, int]] = []

    run_gc_all(
        global_gc.index,
        mode=GcMode.WATCHED_ONLY,
        dry_run=False,
        on_progress=lambda repo_path, done, total: seen.append((repo_path, done, total)),
    )

    assert [(done, total) for _repo, done, total in seen] == [(1, 2), (2, 2)]
    assert len({repo for repo, _done, _total in seen}) == 2
