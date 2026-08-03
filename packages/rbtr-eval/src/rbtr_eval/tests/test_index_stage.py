"""The eval's reports count the corpus, not the whole database.

Asserts the frames the report is built from, not the rendered
markdown: the figures are the behaviour, the table layout is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.domain.models import Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr_eval.index_stage import (
    _embedding_counts,
    _kind_counts,
    _language_counts,
    _repo_counts,
    _totals,
)
from rbtr_eval.tests.conftest import chunk, snap

# ── Fixtures ─────────────────────────────────────────────────────────

INDEXED = "1" * 40
RESIDUE = "2" * 40


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
            INDEXED,
            repo_id=repo_id,
        )
        ws.mark_indexed(repo_id, INDEXED)
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
        ws.mark_indexed(repo_id, INDEXED)

    return store


# ── Tests ────────────────────────────────────────────────────────────


def test_counts_describe_only_indexed_snapshots(store_with_residue: IndexStore) -> None:
    """A snapshot never marked indexed contributes to no count."""
    assert _repo_counts(store_with_residue).rows(named=True) == [
        {"repo": "repo", "chunks": 2, "locations": 2, "edges": 1}
    ]
    assert _totals(store_with_residue) == (2, 2, 1)
    assert _embedding_counts(store_with_residue).rows(named=True) == [
        {"repo": "repo", "chunks": 2, "embedded": 1, "truncated": 0}
    ]


def test_dimension_tables_attribute_each_edge_to_both_endpoints(
    store_with_residue: IndexStore,
) -> None:
    """An edge counts outbound for its source kind, inbound for its target."""
    assert _kind_counts(store_with_residue).rows(named=True) == [
        {"kind": "function", "n": 1, "outbound_edges": 0, "inbound_edges": 1},
        {"kind": "import", "n": 1, "outbound_edges": 1, "inbound_edges": 0},
    ]
    assert _language_counts(store_with_residue).rows(named=True) == [
        {"lang": "python", "n": 2, "outbound_edges": 1, "inbound_edges": 1}
    ]


def test_a_vendored_file_counts_once_as_content_and_twice_as_location(
    store_with_copy: IndexStore,
) -> None:
    """One chunk reached from two paths is one chunk in two places."""
    assert _repo_counts(store_with_copy).rows(named=True) == [
        {"repo": "repo", "chunks": 1, "locations": 2, "edges": 0}
    ]
    assert _totals(store_with_copy) == (1, 2, 0)
