"""The eval's reports count the corpus, not the whole database.

Asserts the frames the report is built from, not the rendered
markdown: the figures are the behaviour, the table layout is not.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.errors import BuildStopped
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.tests.conftest import StubModel, make_commit
from rbtr_eval.corpus import corpus_refs
from rbtr_eval.index_stage import (
    embedding_counts,
    fragment_counts,
    fragment_types,
    kind_counts,
    language_counts,
    repo_counts,
    sentinel_hash,
    totals,
)


@pytest.fixture
def index_with_residue(tmp_path: Path) -> Index:
    """One repo: HEAD built, and a second commit whose build stopped before it was marked indexed.

    HEAD holds `app.py`, importing `load` from `cfg.py`: an import, a
    function and one edge between them, none embedded. The second commit
    adds `gone.py`; its build stops after writing chunks, so no count may
    include them.
    """
    app = b"""\
from cfg import load
"""
    cfg = b"""\
def load():
    return 1
"""
    gone = b"""\
def dead():
    return 0
"""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False, initial_head="main")
    head = make_commit(repo, {"app.py": app, "cfg.py": cfg})
    residue = make_commit(
        repo, {"app.py": app, "cfg.py": cfg, "gone.py": gone}, parents=[head], ref="refs/heads/wip"
    )
    index = Index(str(tmp_path / "index" / "index.duckdb"), writable=True)
    path = normalise_repo_path(repo.workdir)
    build_index(path, str(head), index)
    stop = threading.Event()
    with pytest.raises(BuildStopped):
        build_index(
            path,
            str(residue),
            index,
            stop=stop,
            on_progress=lambda phase, _done, _total: stop.set() if phase == "edges" else None,
        )
    return index


@pytest.fixture
def index_with_copy(tmp_path: Path) -> Index:
    """One repo, built at HEAD, whose `lib.py` is vendored verbatim to `vendor/lib.py`.

    Both paths carry the same blob, so both reach the one chunk the
    content was extracted into.
    """
    lib = b"""\
def load():
    pass
"""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False, initial_head="main")
    sha = str(make_commit(repo, {"lib.py": lib, "vendor/lib.py": lib}))
    index = Index(str(tmp_path / "index" / "index.duckdb"), writable=True)
    build_index(normalise_repo_path(repo.workdir), sha, index)
    return index


@pytest.fixture
def index_with_fragments(tmp_path: Path) -> Index:
    """`app.py` built at HEAD: a class whose method shares its lines, and two fragments.

    Lines 6-8 hold a call and an `if`; line 10 another call. A second
    commit adds `gone.py`, whose `except` is a fragment; its build stops
    after writing chunks, before marking the snapshot indexed, so no
    count may include it.
    """
    app = b"""\
class Loader:
    def load(self):
        return 1


Loader().load()
if __name__ == "__main__":
    Loader().load()

print("done")
"""
    gone = b"""\
try:
    import missing
except ImportError:
    missing = None
"""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False, initial_head="main")
    head = make_commit(repo, {"app.py": app})
    residue = make_commit(
        repo, {"app.py": app, "gone.py": gone}, parents=[head], ref="refs/heads/wip"
    )
    index = Index(str(tmp_path / "index" / "index.duckdb"), writable=True)
    path = normalise_repo_path(repo.workdir)
    build_index(path, str(head), index)
    stop = threading.Event()
    with pytest.raises(BuildStopped):
        build_index(
            path,
            str(residue),
            index,
            stop=stop,
            on_progress=lambda phase, _done, _total: stop.set() if phase == "edges" else None,
        )
    return index


def test_fragments_are_counted_by_language_and_by_node_type(index_with_fragments: Index) -> None:
    """A fragment's lines count toward its language, and toward each node type it holds.

    A line two chunks hold counts once.
    """
    corpus = corpus_refs(index_with_fragments)
    assert fragment_counts(index_with_fragments, corpus).rows(named=True) == [
        {"lang": "python", "fragments": 2, "fragment_lines": 4, "chunk_lines": 7, "share": 4 / 7}
    ]
    assert fragment_types(index_with_fragments, corpus).rows(named=True) == [
        {
            "lang": "python",
            "node_type": "expression_statement",
            "fragments": 2,
            "fragment_lines": 4,
        },
        {"lang": "python", "node_type": "if_statement", "fragments": 1, "fragment_lines": 3},
    ]


def test_counts_describe_only_indexed_snapshots(index_with_residue: Index) -> None:
    """A snapshot never marked indexed contributes to no count."""
    corpus = corpus_refs(index_with_residue)
    assert repo_counts(index_with_residue, corpus).rows(named=True) == [
        {"repo": "repo", "chunks": 2, "locations": 2, "edges": 1}
    ]
    assert totals(index_with_residue, corpus) == (2, 2, 1)
    assert embedding_counts(index_with_residue, corpus).rows(named=True) == [
        {"repo": "repo", "chunks": 2, "embedded": 0, "truncated": 0}
    ]


def test_dimension_tables_attribute_each_edge_to_both_endpoints(index_with_residue: Index) -> None:
    """An edge counts outbound for its source kind, inbound for its target."""
    corpus = corpus_refs(index_with_residue)
    assert kind_counts(index_with_residue, corpus).rows(named=True) == [
        {"kind": "function", "n": 1, "outbound_edges": 0, "inbound_edges": 1},
        {"kind": "import", "n": 1, "outbound_edges": 1, "inbound_edges": 0},
    ]
    assert language_counts(index_with_residue, corpus).rows(named=True) == [
        {"lang": "python", "n": 2, "outbound_edges": 1, "inbound_edges": 1}
    ]


def test_a_vendored_file_counts_once_as_content_and_twice_as_location(
    index_with_copy: Index,
) -> None:
    """One chunk reached from two paths is one chunk in two places."""
    corpus = corpus_refs(index_with_copy)
    assert repo_counts(index_with_copy, corpus).rows(named=True) == [
        {"repo": "repo", "chunks": 1, "locations": 2, "edges": 0}
    ]
    assert totals(index_with_copy, corpus) == (1, 2, 0)


@pytest.mark.parametrize(
    ("embed", "expected"),
    [
        (False, "4fa49d6bc870f587f9d8506a401ebbbb482ca64c1c16f480da9338d2c113b45d"),
        (True, "b84c74874dfc0634c893d7e0bea855cbf4c4ddb4ae201440d7a3b914af9d007f"),
    ],
    ids=["chunks-ready", "embed-ready"],
)
def test_the_sentinel_hash_is_fixed_for_a_given_index(
    index_with_residue: Index, embed: bool, expected: str
) -> None:
    """The hash is DVC's change detector, so its value is a contract.

    Pinned to a literal, because a hash that moves for an unchanged
    index invalidates every downstream stage and forces a full eval
    re-run.  Recomputing the expected value inside the test would agree
    with any implementation and catch that move too late.
    """
    assert sentinel_hash(index_with_residue, embed=embed) == expected


def test_writing_an_embedding_moves_only_the_embed_ready_hash(
    index_with_residue: Index, mocker: MockerFixture
) -> None:
    """Embedding progress is what the embed-ready sentinel tracks.

    The chunks-ready hash covers which snapshots are indexed, so it
    holds steady while embedding fills vectors in.
    """
    before_chunks = sentinel_hash(index_with_residue, embed=False)
    before_embed = sentinel_hash(index_with_residue, embed=True)
    [head] = corpus_refs(index_with_residue).refs
    assert index_with_residue.unembedded_chunk_ids(at=head), "fixture must leave a chunk to embed"

    mocker.patch("rbtr.index.embeddings._load_model", return_value=StubModel())
    embedder = Embedder()
    try:
        embed_index(index_with_residue, at=head, embedder=embedder)
    finally:
        embedder.close()

    assert sentinel_hash(index_with_residue, embed=True) != before_embed
    assert sentinel_hash(index_with_residue, embed=False) == before_chunks
