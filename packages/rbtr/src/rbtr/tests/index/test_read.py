"""Read-side behavioural tests for Index.

Covers: chunks filters, edges filters, blob_is_current
language matching, chunk upsert, delete_chunks_for_blobs,
multi-repo data isolation, and cross-repo content sharing
(content-addressed dedup, shared embeddings, reference-counted
cleanup).
"""

from __future__ import annotations

import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import (
    EdgeKind,
    Import,
    ImportMeta,
    SnapshotRef,
    Span,
)
from rbtr.domain.tokenise import tokenise_code
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from ..conftest import make_commit
from .cases_read import (
    BlobCurrentScenario,
    ChunkQueryScenario,
    EmbedOrderScenario,
    GcCountScenario,
    PathsRead,
)
from .conftest import make_chunk, make_snap, seed_index


@fixture
@parametrize_with_cases("scenario", has_tag="chunks")
def chunk_query(scenario: ChunkQueryScenario) -> ChunkQueryScenario:
    return scenario


@pytest.fixture
def chunk_query_ref(
    chunk_query: ChunkQueryScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, chunk_query.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_chunks_returns_expected(
    chunk_query: ChunkQueryScenario, chunk_query_ref: SnapshotRef, index: Index
) -> None:
    chunks = index.chunks(
        at=chunk_query_ref,
        file_path=chunk_query.file_path,
        kind=chunk_query.kind,
        name=chunk_query.name,
    )
    assert sorted(chunks["name"]) == sorted(chunk_query.expected_names)


@fixture
@parametrize_with_cases("scenario", has_tag="blob_is_current")
def blob_query(scenario: BlobCurrentScenario, index: Index) -> tuple[Index, BlobCurrentScenario]:
    repo = index.register("/repo")
    with index.session(repo) as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
        ws.replace_snapshots(list(scenario.snapshots), snapshot_sha="head")
    return index, scenario


def test_blob_is_current_matches(
    blob_query: tuple[Index, BlobCurrentScenario],
) -> None:
    index, s = blob_query
    with index.session(index.repo("/repo")) as ws:
        current = ws.blob_is_current(s.query_blob, s.query_language, s.serial_map)
    assert current == s.expected


def test_upsert_replaces_content(index: Index) -> None:
    """Adding a chunk with the same ID replaces the content."""
    c1 = make_chunk("fn1", content="def fn1(): return 1")
    c2 = make_chunk("fn1", content="def fn1(): return 2")

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(c1)
        ws.replace_snapshots([make_snap("head", "f.py", c1.blob_sha)], snapshot_sha="head")

    with index.session(repo) as ws:
        ws.add_chunk(c2)

    chunks = index.chunks(at=repo.at("head"))
    assert len(chunks) == 1
    assert "return 2" in chunks["content"][0]


def test_delete_chunks_for_blobs_removes_target(index: Index) -> None:
    """Deleting a blob's chunks removes them, leaves others."""
    c1 = make_chunk("fn1", blob="b1", path="a.py")
    c2 = make_chunk("fn2", blob="b2", path="b.py")

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(c1)
        ws.add_chunk(c2)
        ws.replace_snapshots(
            [make_snap("head", "a.py", "b1"), make_snap("head", "b.py", "b2")], snapshot_sha="head"
        )

    with index.session(repo) as ws:
        ws.delete_chunks_for_blobs({"b1"}, file_language="")

    with index.session(repo) as ws:
        assert ws.blob_is_current("b1", "", {"": 1}) is False
        assert ws.blob_is_current("b2", "", {"": 1}) is True


def test_edges_returns_all(module_index: Index, import_graph: SnapshotRef) -> None:
    """Every edge the build inferred at a snapshot is returned, of every kind."""
    edges = module_index.edges(within=[import_graph])
    assert edges.select("source_path", "target_path", "kind").sort("source_path").to_dicts() == [
        {"source_path": "README.md", "target_path": "src/m.py", "kind": "documents"},
        {"source_path": "src/app.py", "target_path": "src/m.py", "kind": "imports"},
    ]


def test_edges_filter_by_kind(module_index: Index, import_graph: SnapshotRef) -> None:
    """Filtering edges by kind returns only that kind."""
    edges = module_index.edges(within=[import_graph], kind=EdgeKind.IMPORTS)
    assert edges.select("source_path", "kind").to_dicts() == [
        {"source_path": "src/app.py", "kind": "imports"}
    ]


def test_inbound_refs_resolves_source(module_index: Index, import_graph: SnapshotRef) -> None:
    """Each inbound edge resolves to its source chunk plus the edge kind."""
    fn = module_index.chunks(at=import_graph, name="fn")["id"].item()
    frame = module_index.inbound_refs([fn], at=import_graph)
    assert frame.select("file_path", "name", "kind", "edge").sort("file_path").to_dicts() == [
        {"file_path": "README.md", "name": "src/m.py", "kind": "import", "edge": "documents"},
        {
            "file_path": "src/app.py",
            "name": "from src.m import fn",
            "kind": "import",
            "edge": "imports",
        },
    ]


def test_a_read_leaves_the_cursor_free_for_the_next(
    module_index: Index, import_graph: SnapshotRef
) -> None:
    """A read consumes its whole result, so the thread's next read can
    register views on the same cached cursor."""
    fn = module_index.chunks(at=import_graph, name="fn")["id"].item()
    assert module_index.count_edges(at=import_graph) == 2
    assert module_index.chunks_by_id([fn], at=import_graph)["id"].to_list() == [fn]


@pytest.fixture
def script_import() -> TokenisedChunk:
    """An import with every metadata field set, so a field storage drops shows."""
    content = '<script src="greeter.js"></script>'
    return TokenisedChunk(
        blob_sha="blob_page",
        file_path="page.html",
        role=Import(
            name="greeter.js",
            meta=ImportMeta(
                module="greeter.js", names="greet", dots="1", language_hint="javascript"
            ),
        ),
        content=content,
        content_tokens=tokenise_code(content),
        span=Span(start=3, end=3),
    )


def test_an_import_reads_back_as_written(
    script_import: TokenisedChunk, index: Index, head_ref: SnapshotRef
) -> None:
    seed_index(index, [script_import], head_ref)
    assert index.chunks(at=head_ref)["metadata"].to_list() == [
        {
            "module": "greeter.js",
            "names": "greet",
            "dots": "1",
            "language_hint": "javascript",
            "node_types": None,
        }
    ]


def test_inbound_refs_empty_targets(index: Index, head_ref: SnapshotRef) -> None:
    """No target IDs returns an empty frame without touching the store."""
    assert index.inbound_refs([], at=head_ref).is_empty()


@pytest.fixture
def stub_vectors() -> dict[str, list[float]]:
    """Chosen vectors for `checkouts`' functions, so a `[1, 0, 0, 0]` query reaches them."""
    return {"alpha_fn": [1.0, 0.0, 0.0, 0.0], "beta_fn": [1.0, 0.0, 0.0, 0.0]}


@pytest.fixture
def alpha_sha(git_repo: pygit2.Repository) -> str:
    """A commit holding `alpha/fn.py`, defining `alpha_fn`, and `alpha/main.py`, importing it."""
    fn = b"""\
def alpha_fn():
    return "shared_term"
"""
    main = b"""\
from alpha.fn import alpha_fn
"""
    files = {"alpha/fn.py": fn, "alpha/main.py": main}
    return str(make_commit(git_repo, files, ref="refs/heads/alpha"))


@pytest.fixture
def beta_sha(git_repo: pygit2.Repository) -> str:
    """A commit holding `beta/fn.py`, defining `beta_fn`, and `beta/main.py`, importing it."""
    fn = b"""\
def beta_fn():
    return "shared_term"
"""
    main = b"""\
from beta.fn import beta_fn
"""
    files = {"beta/fn.py": fn, "beta/main.py": main}
    return str(make_commit(git_repo, files, ref="refs/heads/beta"))


@pytest.fixture
def checkouts(
    index: Index,
    repo_path: str,
    linked_worktree: pygit2.Repository,
    alpha_sha: str,
    beta_sha: str,
    stubbed_embedder: Embedder,
) -> Index:
    """Two checkouts of one project, built and embedded with different content.

    The main checkout (repo 1) is built at `alpha_sha` and the linked
    worktree (repo 2) at `beta_sha`; each holds a function, an import
    of it (an edge) and embeddings. Both checkouts write one project
    file, so only a read's repo filter keeps them apart.
    """
    main_ref = index.register(repo_path).at(alpha_sha)
    worktree_path = normalise_repo_path(linked_worktree.workdir)
    build_index(repo_path, alpha_sha, index)
    build_index(worktree_path, beta_sha, index)
    embed_index(index, at=main_ref, embedder=stubbed_embedder)
    embed_index(index, at=index.register(worktree_path).at(beta_sha), embedder=stubbed_embedder)
    return index


@parametrize_with_cases("read", cases=".cases_read", has_tag="isolation")
def test_a_read_sees_only_its_own_checkout(
    read: PathsRead, checkouts: Index, alpha_sha: str, beta_sha: str
) -> None:
    """Two checkouts share a project file; a read at one sees none of the other's rows."""
    main, worktree = checkouts.repos()
    assert {path.split("/")[0] for path in read(checkouts, main.at(alpha_sha))} == {"alpha"}
    assert {path.split("/")[0] for path in read(checkouts, worktree.at(beta_sha))} == {"beta"}


def test_shared_content_is_one_row_visible_to_both_repos(
    shared_chunk_index: Index,
    shared_sha: str,
    shared_chunk_id: str,
) -> None:
    """Shared content is a single physical row, visible to every repo."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    assert index.chunks(at=repo1.at(shared_sha))["id"].to_list() == [shared_chunk_id]
    assert index.chunks(at=repo2.at(shared_sha))["id"].to_list() == [shared_chunk_id]
    with index.reader(repo1) as cur:
        rows = cur.sql("SELECT count(*) FROM chunks").fetchall()
    assert rows == [(1,)]


def test_shared_chunk_embedded_once_across_repos(
    shared_chunk_index: Index, shared_sha: str, stubbed_embedder: Embedder
) -> None:
    """Embedding the shared chunk once leaves no repo with work to do."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    in_repo1 = repo1.at(shared_sha)
    in_repo2 = repo2.at(shared_sha)
    assert index.chunk_counts_for_snapshot(at=in_repo1).unembedded == 1
    assert index.chunk_counts_for_snapshot(at=in_repo2).unembedded == 1

    embed_index(index, at=in_repo1, embedder=stubbed_embedder)

    assert index.chunk_counts_for_snapshot(at=in_repo1).is_fully_embedded
    assert index.chunk_counts_for_snapshot(at=in_repo2).is_fully_embedded


def test_cleanup_keeps_chunk_referenced_by_another_repo(
    shared_chunk_index: Index,
    shared_sha: str,
    shared_chunk_id: str,
) -> None:
    """A repo's build-finalise `cleanup()` must not prune a co-tenant's chunk.

    `cleanup()` runs the global `prune_chunks`. While repo 2 still
    references the shared blob, repo 1's cleanup must remove nothing and
    leave the chunk visible to both — the realistic co-tenant case, since
    `cleanup` runs at the end of every build.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    with index.session(repo1) as ws:
        cleaned = ws.cleanup()
    assert cleaned.chunks == 0
    assert index.count_orphan_chunks() == 0
    assert index.chunks(at=repo1.at(shared_sha))["id"].to_list() == [shared_chunk_id]
    assert index.chunks(at=repo2.at(shared_sha))["id"].to_list() == [shared_chunk_id]


def test_shared_chunk_swept_only_after_last_repo_drops_it(
    shared_chunk_index: Index,
    shared_sha: str,
    shared_chunk_id: str,
) -> None:
    """Cross-repo refcount: the chunk survives until the last reference goes."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    # Repo 1 drops its commit: shared chunk survives (repo 2 references it).
    with index.session(repo1) as ws:
        first_drop = ws.drop_snapshot(snapshot_sha=shared_sha)
    assert first_drop.chunks == 0
    assert index.chunks(at=repo2.at(shared_sha))["id"].to_list() == [shared_chunk_id]

    # Repo 2 drops its commit: last reference gone, chunk is swept.
    with index.session(repo2) as ws:
        second_drop = ws.drop_snapshot(snapshot_sha=shared_sha)
    assert second_drop.chunks == 1
    assert index.count_orphan_chunks() == 0
    assert index.chunks(at=repo2.at(shared_sha)).is_empty()


def test_rechunk_of_shared_blob_propagates_to_all_repos(
    shared_chunk_index: Index,
    shared_sha: str,
) -> None:
    """Re-chunking a shared blob updates the content for every repo.

    A plugin upgrade re-chunks a blob globally (delete its chunks, then
    re-insert at the new serial). Because the chunk is content-addressed
    and shared, every repo referencing the blob — not just the one that
    triggered the re-chunk — sees the new chunks, without re-indexing.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    blob = str(index.chunks(at=repo1.at(shared_sha))["blob_sha"].item())
    rechunked = make_chunk("shared_fn_v2", path="x.py", blob=blob, language="python").model_copy(
        update={"extraction_serial": 2}
    )
    # A different name, so a different id: the upgrade replaces the row.
    with index.session(repo1) as ws:
        ws.delete_chunks_for_blobs({blob}, file_language="python")
        ws.add_chunk(rechunked)

    assert index.chunks(at=repo1.at(shared_sha))["id"].to_list() == [rechunked.id]
    assert index.chunks(at=repo2.at(shared_sha))["id"].to_list() == [rechunked.id]


@pytest.fixture
def two_paths_sha(git_repo: pygit2.Repository) -> str:
    """A commit holding the same content at `a.py` and `b.py`."""
    shared = b"""\
def shared():
    pass
"""
    return str(make_commit(git_repo, {"a.py": shared, "b.py": shared}, ref="refs/heads/two"))


@pytest.fixture
def one_path_sha(git_repo: pygit2.Repository) -> str:
    """A commit holding that content at `a.py` only."""
    shared = b"""\
def shared():
    pass
"""
    return str(make_commit(git_repo, {"a.py": shared}, ref="refs/heads/one"))


def test_drop_snapshot_keeps_a_chunk_still_reachable_at_another_path(
    index: Index, repo_path: str, two_paths_sha: str, one_path_sha: str
) -> None:
    """`drop_snapshot` keeps a chunk whose blob is referenced elsewhere.

    Identical content at `a.py` and `b.py` is one chunk reached through
    two paths, so dropping the commit that referenced `b.py` collects
    nothing: the chunk is still reachable at `a.py`, and deleting it
    would empty a file that has not changed.
    """
    build_index(repo_path, two_paths_sha, index)
    build_index(repo_path, one_path_sha, index)
    repo = index.register(repo_path)
    shared = index.chunks(at=repo.at(one_path_sha))["id"].to_list()

    with index.session(repo) as ws:
        dropped = ws.drop_snapshot(snapshot_sha=two_paths_sha)

    assert dropped.chunks == 0
    assert index.count_orphan_chunks() == 0
    assert index.chunks(at=repo.at(one_path_sha))["id"].to_list() == shared


def test_inbound_refs_to_shared_chunk_resolve_per_repo(
    shared_target: Index, importer_a_sha: str, importer_b_sha: str
) -> None:
    """inbound_refs to a shared chunk returns only the querying repo's referrers.

    Edges are per-repo: each repo infers its own inbound edge to the
    shared chunk from its own importer. Querying the shared target must
    resolve the querying repo's referrer, never the other repo's.
    """
    index = shared_target
    repo1, repo2 = index.repos()
    main, other = repo1.at(importer_a_sha), repo2.at(importer_b_sha)
    fn = index.chunks(at=main, name="fn")["id"].item()
    assert index.chunks(at=other, name="fn")["id"].to_list() == [fn], "one shared chunk"
    assert index.inbound_refs([fn], at=main)["file_path"].to_list() == ["a.py"]
    assert index.inbound_refs([fn], at=other)["file_path"].to_list() == ["b.py"]


@fixture
@parametrize_with_cases("scenario", cases=".cases_read", has_tag="gc_counts")
def gc_count(scenario: GcCountScenario) -> GcCountScenario:
    return scenario


@pytest.fixture
def gc_count_shas(
    gc_count: GcCountScenario,
    project_index: Index,
    git_repo: pygit2.Repository,
    repo_path: str,
    linked_worktree: pygit2.Repository,
) -> dict[str, str]:
    """Each of the scenario's commits built in its checkout; returns each label's commit."""
    paths = [repo_path, normalise_repo_path(linked_worktree.workdir)]
    shas: dict[str, str] = {}
    for commit in gc_count.commits:
        ref = f"refs/heads/{commit.label}"
        # The label is the message too, so two commits of one tree stay two commits.
        sha = str(make_commit(git_repo, commit.files, message=commit.label, ref=ref))
        build_index(paths[commit.checkout], sha, project_index)
        shas[commit.label] = sha
    return shas


def test_gc_chunk_split_counts(
    gc_count: GcCountScenario, gc_count_shas: dict[str, str], project_index: Index
) -> None:
    """`count_gc_chunk_split` splits candidate chunks into dropped vs kept.

    A candidate (referenced by the drop set) is kept when a snapshot
    outside the drop set still references its `(blob_sha, file_path)`,
    else it is dropped.
    """
    drop = project_index.repos()[gc_count.drop_checkout]
    split = project_index.count_gc_chunk_split(drop, [gc_count_shas[c] for c in gc_count.drop])
    assert (split.dropped, split.kept) == (gc_count.expected_dropped, gc_count.expected_kept)


def test_gc_chunk_split_agrees_with_real_deletion(
    gc_count: GcCountScenario, gc_count_shas: dict[str, str], project_index: Index
) -> None:
    """The predicted drop count matches a real deletion, leaving no orphans.

    Guards against drift between `count_gc_chunk_split` (the prediction)
    and the `drop_snapshot` sweep (the deletion), and between the sweep and
    `count_orphan_chunks`: dropping the drop set's commits must remove
    exactly the predicted number of chunks and leave the store
    orphan-free. Both sides are read off the same reference graph, so
    they must never diverge.
    """
    index = project_index
    drop = index.repos()[gc_count.drop_checkout]
    drop_shas = [gc_count_shas[c] for c in gc_count.drop]
    predicted_dropped = index.count_gc_chunk_split(drop, drop_shas).dropped

    with index.reader(drop) as cur:
        ((before,),) = cur.sql("SELECT count(*) FROM chunks").fetchall()
    with index.session(drop) as ws:
        for sha in drop_shas:
            ws.drop_snapshot(snapshot_sha=sha)
    with index.reader(drop) as cur:
        ((after,),) = cur.sql("SELECT count(*) FROM chunks").fetchall()

    assert before - after == predicted_dropped
    assert index.count_orphan_chunks() == 0


def test_forget_removes_references_keeps_shared_chunk(
    shared_chunk_index: Index,
    shared_sha: str,
) -> None:
    """`forget` purges a checkout's references and `repos` row but leaves a
    chunk the project's other checkout still references; reclamation is
    gc's job."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    with index.session(repo1) as ws:
        ws.add_watched_refs(["main"])

    index.forget(repo1)

    # Repo 1 is gone entirely, its rows with it.
    assert index.repos() == [repo2]
    gone = {"repo_id": repo1.repo_id}
    with index.reader(repo2) as cur:
        watched = cur.sql("SELECT count(*) FROM watched_refs WHERE repo_id = $repo_id", params=gone)
        assert watched.fetchall() == [(0,)]
        indexed = cur.sql(
            "SELECT count(*) FROM indexed_snapshots WHERE repo_id = $repo_id", params=gone
        )
        assert indexed.fetchall() == [(0,)]
    # Repo 2 is untouched and the shared chunk is still visible to it.
    assert index.has_indexed(at=repo2.at(shared_sha)) is True
    assert len(index.chunks(at=repo2.at(shared_sha))) > 0


@pytest.fixture
def only_in_repo1(shared_chunk_index: Index, git_repo: pygit2.Repository, repo_path: str) -> Index:
    """`shared_chunk_index` with repo 1 also built at a commit holding `y.py`.

    `y.py`'s chunk `only1` is referenced by repo 1 alone.
    """
    y = b"""\
def only1():
    pass
"""
    sha = str(make_commit(git_repo, {"y.py": y}, ref="refs/heads/only1"))
    build_index(repo_path, sha, shared_chunk_index)
    return shared_chunk_index


def test_forget_leaves_orphan_chunks_for_gc(only_in_repo1: Index) -> None:
    """`forget` does not sweep: a chunk only the forgotten repo
    referenced becomes an orphan that a later GC reclaims."""
    index = only_in_repo1
    repo1, _repo2 = index.repos()
    assert index.count_orphan_chunks() == 0

    index.forget(repo1)

    # only1 is now unreferenced (repo 1 gone) but still present — no sweep.
    # (A sweep would have removed it, leaving zero orphans.)
    assert index.count_orphan_chunks() == 1


@pytest.fixture
def nothing_to_index_ref(index: Index, git_repo: pygit2.Repository, repo_path: str) -> SnapshotRef:
    """A commit of an empty tree, built: indexed, with nothing to extract."""
    sha = str(make_commit(git_repo, {}, ref="refs/heads/empty"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_an_indexed_snapshot_with_no_chunks_counts_zero(
    index: Index, nothing_to_index_ref: SnapshotRef
) -> None:
    """A commit that yielded nothing to index is still reported, at zero.

    `mark_indexed` records completion for any tree, including one that
    held nothing extractable, so the query answers for it with a row of
    zeroes and the caller reads a figure rather than supplying one.
    """
    counts = index.chunk_counts_for_snapshot(at=nothing_to_index_ref)
    assert counts.total == 0
    assert counts.is_fully_embedded, "nothing outstanding means no embed work"
    snapshots = index.chunk_counts_by_snapshot(repo=nothing_to_index_ref.repo)
    assert [ref.snapshot_sha for ref in snapshots] == [nothing_to_index_ref.snapshot_sha]


@parametrize_with_cases("scenario", cases=".cases_read", has_tag="embed_order")
def test_snapshots_come_back_in_embed_order(scenario: EmbedOrderScenario, index: Index) -> None:
    """Order decides which snapshot is embedded next, so the queries fix it.

    Asserted through `snapshots_to_embed`, which the job picker reads, and
    `chunk_counts_by_snapshot`, which status reports in the same order.
    Each snapshot holds one chunk without a vector, so each has work.
    """
    for shas in scenario.transactions:
        repo = index.register("/r")
        with index.session(repo) as ws:
            for sha in shas:
                chunk = make_chunk(sha, path=f"{sha}.py")
                ws.add_chunk(chunk)
                ws.replace_snapshots(
                    [make_snap(sha, chunk.file_path, chunk.blob_sha)], snapshot_sha=sha
                )
                ws.mark_indexed(snapshot_sha=sha)

    assert [ref.snapshot_sha for ref in index.snapshots_to_embed()] == scenario.expected
    counted = index.chunk_counts_by_snapshot(repo=repo)
    assert [ref.snapshot_sha for ref in counted] == scenario.expected
