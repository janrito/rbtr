"""Read-side behavioural tests for Index, over what builds wrote.

Covers: chunks filters, edges filters, how an import is stored,
multi-repo data isolation, and cross-repo content sharing
(content-addressed dedup, shared embeddings, reference-counted
gc).
"""

from __future__ import annotations

import dataclasses

import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import ChunkKind, EdgeKind, GcMode, SnapshotRef
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.gc import run_gc
from rbtr.index.index import Index
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import QueryExtraction
from rbtr_lang_python.plugin import python

from ..conftest import make_commit
from .cases_read import ChunkQueryScenario, GcCountScenario, PathsRead


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


def test_chunks_sharing_a_start_line_come_back_in_one_order(
    index: Index, git_repo: pygit2.Repository, repo_path: str
) -> None:
    """Chunks starting on one line are read by end line, kind and name, so every read agrees.

    Each line holds a constant and the comment after it: two chunks on
    one line, twelve times over.
    """
    source = b"""\
JAN = 1  # winter
FEB = 2  # winter
MAR = 3  # spring
APR = 4  # spring
MAY = 5  # spring
JUN = 6  # summer
JUL = 7  # summer
AUG = 8  # summer
SEP = 9  # autumn
OCT = 10  # autumn
NOV = 11  # autumn
DEC = 12  # winter
"""
    sha = str(make_commit(git_repo, {"months.py": source}, ref="refs/heads/months"))
    build_index(repo_path, sha, index)

    chunks = index.chunks(at=index.register(repo_path).at(sha))

    assert chunks.select("line_start", "kind").rows() == [
        (line, kind) for line in range(1, 13) for kind in ("comment", "variable")
    ]


def test_edges_returns_all(module_index: Index, import_graph: SnapshotRef) -> None:
    """Every edge the build inferred at a snapshot is returned, of every kind."""
    edges = module_index.edges(within=[import_graph])
    assert edges.select("source_path", "target_path", "kind").sort(
        "source_path", "kind"
    ).to_dicts() == [
        {"source_path": "README.md", "target_path": "src/m.py", "kind": "documents"},
        {"source_path": "src/app.py", "target_path": "src/m.py", "kind": "calls"},
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
    assert frame.select("file_path", "name", "kind", "edge").sort(
        "file_path", "edge"
    ).to_dicts() == [
        {"file_path": "README.md", "name": "src/m.py", "kind": "import", "edge": "documents"},
        {"file_path": "src/app.py", "name": "run", "kind": "function", "edge": "calls"},
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
    assert module_index.count_edges(at=import_graph) == 3
    assert module_index.chunks_by_id([fn], at=import_graph)["id"].to_list() == [fn]


def test_an_import_reads_back_as_written(
    index: Index, git_repo: pygit2.Repository, repo_path: str
) -> None:
    """Every field of an import is stored and read back: a JavaScript import's module, names and bindings, an HTML script's hint."""
    sha = str(
        make_commit(
            git_repo,
            {
                "app.js": b'import { greet as hello } from "./greeter.js";\n',
                "page.html": b'<script src="greeter.js"></script>\n',
            },
            ref="refs/heads/imports",
        )
    )
    build_index(repo_path, sha, index)
    imports = index.chunks(at=index.register(repo_path).at(sha), kind=ChunkKind.IMPORT)
    assert imports["metadata"].to_list() == [
        {
            "module": "./greeter",
            "names": "greet",
            "language_hint": "",
            "node_types": None,
            "refs": None,
            "bindings": [{"name": "greet", "local": "hello"}],
        },
        {
            "module": "greeter.js",
            "names": "",
            "language_hint": "javascript",
            "node_types": None,
            "refs": None,
            "bindings": [],
        },
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

    assert index.chunk_counts_for_snapshot(at=in_repo1).unembedded == 0
    assert index.chunk_counts_for_snapshot(at=in_repo2).unembedded == 0


def test_shared_chunk_swept_only_after_last_repo_drops_it(
    shared_chunk_index: Index,
    shared_sha: str,
    shared_chunk_id: str,
    repo_path: str,
    linked_worktree: pygit2.Repository,
) -> None:
    """Cross-repo refcount: a gc in one checkout keeps the chunk the other still claims, and the last collects it.

    A gc's clean-up prunes chunks across the project, so the first
    checkout's must not take the chunk its co-tenant still reads.
    """
    index = shared_chunk_index
    _repo1, repo2 = index.repos()
    first = run_gc(index, repo_path, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    assert first.chunks == 0
    assert index.chunks(at=repo2.at(shared_sha))["id"].to_list() == [shared_chunk_id]

    worktree = normalise_repo_path(linked_worktree.workdir)
    second = run_gc(index, worktree, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    assert second.chunks == 1
    assert index.count_orphan_chunks() == 0
    assert index.chunks(at=repo2.at(shared_sha)).is_empty()


def test_rechunk_of_shared_blob_propagates_to_all_repos(
    shared_chunk_index: Index, shared_sha: str, repo_path: str
) -> None:
    """A plugin upgrade's re-extraction in one checkout changes the chunks every checkout reads.

    The chunk is content-addressed and shared, so the second checkout sees
    the new extraction without building again: here a Python query that
    no longer captures functions makes `shared_fn` a fragment.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    upgraded = dataclasses.replace(
        python,
        extraction=QueryExtraction(query="(comment) @comment\n"),
        extraction_serial=python.extraction_serial + 1,
    )
    with get_manager().limited_to(upgraded):
        build_index(repo_path, shared_sha, index)

    assert index.chunks(at=repo1.at(shared_sha))["kind"].to_list() == ["fragment"]
    assert index.chunks(at=repo2.at(shared_sha))["kind"].to_list() == ["fragment"]


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


def test_gc_keeps_a_chunk_still_reachable_at_another_path(
    index: Index, repo_path: str, two_paths_sha: str, one_path_sha: str
) -> None:
    """A gc keeps a chunk whose blob is referenced elsewhere.

    Identical content at `a.py` and `b.py` is one chunk reached through
    two paths, so dropping the commit that referenced `b.py` collects
    nothing: the chunk is still reachable at `a.py`, and deleting it
    would empty a file that has not changed.
    """
    build_index(repo_path, two_paths_sha, index)
    build_index(repo_path, one_path_sha, index)
    repo = index.register(repo_path)
    shared = index.chunks(at=repo.at(one_path_sha))["id"].to_list()

    dropped = run_gc(index, repo_path, mode=GcMode.KEEP, refs=["one"], dry_run=False)

    assert dropped.snapshots == 1
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
    gc_count: GcCountScenario,
    gc_count_shas: dict[str, str],
    project_index: Index,
    repo_path: str,
    linked_worktree: pygit2.Repository,
) -> None:
    """A gc's dry run predicts the chunks its real run deletes, leaving no orphans.

    Guards against drift between `count_gc_chunk_split` (the prediction a
    dry run reports) and the `drop_snapshot` sweep (the deletion), and
    between the sweep and `count_orphan_chunks`. The gc keeps every commit
    the scenario does not drop.
    """
    index = project_index
    path = [repo_path, normalise_repo_path(linked_worktree.workdir)][gc_count.drop_checkout]
    kept = [label for label in gc_count_shas if label not in gc_count.drop]
    predicted = run_gc(index, path, mode=GcMode.KEEP, refs=kept, dry_run=True)
    collected = run_gc(index, path, mode=GcMode.KEEP, refs=kept, dry_run=False)

    assert collected.chunks == predicted.chunks == gc_count.expected_dropped
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
    assert counts.unembedded == 0, "nothing outstanding means no embed work"
    snapshots = index.chunk_counts_by_snapshot(repo=nothing_to_index_ref.repo)
    assert [ref.snapshot_sha for ref in snapshots] == [nothing_to_index_ref.snapshot_sha]


def test_snapshots_come_back_in_embed_order(
    index: Index, git_repo: pygit2.Repository, repo_path: str
) -> None:
    """The newest build is embedded first, so the queries fix the order.

    Asserted through `snapshots_to_embed`, which the job picker reads, and
    `chunk_counts_by_snapshot`, which status reports in the same order.
    Each commit holds one function without a vector, so each has work.
    """
    built = []
    for label in ("older", "newer"):
        sha = str(
            make_commit(
                git_repo,
                {
                    f"{label}.py": f"""\
def {label}():
    pass
""".encode()
                },
                ref=f"refs/heads/{label}",
            )
        )
        build_index(repo_path, sha, index)
        built.append(sha)

    newest_first = built[::-1]
    assert [ref.snapshot_sha for ref in index.snapshots_to_embed()] == newest_first
    counted = index.chunk_counts_by_snapshot(repo=index.register(repo_path))
    assert [ref.snapshot_sha for ref in counted] == newest_first
