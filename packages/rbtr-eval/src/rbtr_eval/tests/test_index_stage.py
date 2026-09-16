"""The eval's reports count the corpus, not the whole database.

Asserts the frames the report is built from, not the rendered
markdown: the figures are the behaviour, the table layout is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.domain.models import Edge, EdgeKind, SnapshotRef
from rbtr.index.store import IndexStore
from rbtr_eval.corpus import Corpus
from rbtr_eval.index_stage import (
    _embedding_counts,
    _kind_counts,
    _language_counts,
    _repo_counts,
    _sentinel_hash,
    _totals,
)
from rbtr_eval.tests.conftest import chunk, snap

# ── Fixtures ─────────────────────────────────────────────────────────

INDEXED = "1" * 40
RESIDUE = "2" * 40


@pytest.fixture
def corpus() -> Corpus:
    """The indexed snapshot of the one repo each store fixture registers.

    In the pipeline `corpus_refs` builds this from the repos' HEADs;
    these stores have no git repo behind them, so the refs are named
    here.
    """
    return Corpus(refs=(SnapshotRef(repo_id=1, snapshot_sha=INDEXED),))


@pytest.fixture
def store_with_residue(tmp_path: Path) -> IndexStore:
    """One repo holding an indexed snapshot and a never-indexed one.

    The indexed snapshot carries `app.py` with an import chunk and a
    function chunk, and one edge between them; the function is
    embedded, the import is not.  The residue snapshot — a build that
    crashed before `mark_indexed` — carries a second file whose chunk
    must not be counted anywhere.
    """
    store = IndexStore(str(tmp_path / "index" / "index.duckdb"), writable=True)
    imp = chunk(file_path="app.py", kind="import", name="import cfg")
    fn = chunk(file_path="app.py", kind="function", name="load")
    orphan = chunk(file_path="gone.py", kind="function", name="dead")

    with store.session() as ws:
        repo_id = ws.register_repo(str(tmp_path / "repo"))
        for sha, chunks in ((INDEXED, [imp, fn]), (RESIDUE, [orphan])):
            for c in chunks:
                ws.add_chunk(c)
            ws.insert_snapshots([snap(sha, chunks[0])], repo_id=repo_id)
        ws.insert_edges(
            [
                Edge(
                    source_id=imp.id,
                    target_id=fn.id,
                    kind=EdgeKind.IMPORTS,
                    source_path=imp.file_path,
                    target_path=fn.file_path,
                )
            ],
            at=SnapshotRef(repo_id=repo_id, snapshot_sha=INDEXED),
        )
        ws.mark_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=INDEXED))
        ws.update_embeddings([fn.id], [[0.5] * 768])

    return store


@pytest.fixture
def store_with_copy(tmp_path: Path) -> IndexStore:
    """One repo whose `lib.py` is vendored verbatim to `vendor/lib.py`.

    Both paths carry the same blob, so both reach the one chunk the
    content was extracted into.
    """
    store = IndexStore(str(tmp_path / "index" / "index.duckdb"), writable=True)
    fn = chunk(file_path="lib.py", kind="function", name="load")

    with store.session() as ws:
        repo_id = ws.register_repo(str(tmp_path / "repo"))
        ws.add_chunk(fn)
        ws.insert_snapshots(
            [snap(INDEXED, fn), snap(INDEXED, fn, path="vendor/lib.py")],
            repo_id=repo_id,
        )
        ws.mark_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=INDEXED))

    return store


# ── Tests ────────────────────────────────────────────────────────────


def test_counts_describe_only_indexed_snapshots(
    store_with_residue: IndexStore, corpus: Corpus
) -> None:
    """A snapshot never marked indexed contributes to no count."""
    assert _repo_counts(store_with_residue, corpus).rows(named=True) == [
        {"repo": "repo", "chunks": 2, "locations": 2, "edges": 1}
    ]
    assert _totals(store_with_residue, corpus) == (2, 2, 1)
    assert _embedding_counts(store_with_residue, corpus).rows(named=True) == [
        {"repo": "repo", "chunks": 2, "embedded": 1, "truncated": 0}
    ]


def test_dimension_tables_attribute_each_edge_to_both_endpoints(
    store_with_residue: IndexStore, corpus: Corpus
) -> None:
    """An edge counts outbound for its source kind, inbound for its target."""
    assert _kind_counts(store_with_residue, corpus).rows(named=True) == [
        {"kind": "function", "n": 1, "outbound_edges": 0, "inbound_edges": 1},
        {"kind": "import", "n": 1, "outbound_edges": 1, "inbound_edges": 0},
    ]
    assert _language_counts(store_with_residue, corpus).rows(named=True) == [
        {"lang": "python", "n": 2, "outbound_edges": 1, "inbound_edges": 1}
    ]


def test_a_vendored_file_counts_once_as_content_and_twice_as_location(
    store_with_copy: IndexStore, corpus: Corpus
) -> None:
    """One chunk reached from two paths is one chunk in two places."""
    assert _repo_counts(store_with_copy, corpus).rows(named=True) == [
        {"repo": "repo", "chunks": 1, "locations": 2, "edges": 0}
    ]
    assert _totals(store_with_copy, corpus) == (1, 2, 0)


# ── DVC sentinel hash ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("embed", "expected"),
    [
        (False, "80d06b379c6ccfbbd9dd327e6c595c3af0e74a01731cee298330f7ef2707b012"),
        (True, "d3e23b5f7fe5de491b8f9ff2eeec02f8d8087cb607f96c6f98633ac4ed6c30ca"),
    ],
    ids=["chunks-ready", "embed-ready"],
)
def test_the_sentinel_hash_is_fixed_for_a_given_index(
    store_with_residue: IndexStore, embed: bool, expected: str
) -> None:
    """The hash is DVC's change detector, so its value is a contract.

    Pinned to a literal, because a hash that moves for an unchanged
    index invalidates every downstream stage and forces a full eval
    re-run.  Recomputing the expected value inside the test would agree
    with any implementation and catch that move too late.
    """
    assert _sentinel_hash(store_with_residue, embed=embed) == expected


def test_writing_an_embedding_moves_only_the_embed_ready_hash(
    store_with_residue: IndexStore,
) -> None:
    """Embedding progress is what the embed-ready sentinel tracks.

    The chunks-ready hash covers which snapshots are indexed, so it
    holds steady while embedding fills vectors in.
    """
    before_chunks = _sentinel_hash(store_with_residue, embed=False)
    before_embed = _sentinel_hash(store_with_residue, embed=True)
    unembedded = store_with_residue.unembedded_chunk_ids(
        at=SnapshotRef(repo_id=1, snapshot_sha=INDEXED)
    )
    assert unembedded, "fixture must leave a chunk to embed"

    with store_with_residue.session() as ws:
        ws.update_embeddings(unembedded, [[0.25] * 768 for _ in unembedded])

    assert _sentinel_hash(store_with_residue, embed=True) != before_embed
    assert _sentinel_hash(store_with_residue, embed=False) == before_chunks
