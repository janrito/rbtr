"""Byte-identical files must each be findable, and stay distinct.

Chunks are keyed by content, so a copy of an already-indexed file needs
no re-extraction and no second row: it is the same row, reached through
`file_snapshots` at a second path. These tests pin that a copy stays
visible, that the same bytes under two languages stay separate, and that
sharing a row does not make two locations rank as one.
"""

from __future__ import annotations

from rbtr.domain.models import SnapshotRef
from rbtr.index.store import IndexStore

# ── Every copy is findable ───────────────────────────────────────────


def test_identical_files_are_indexed_at_every_path(
    dup_store: IndexStore, dup_ref: SnapshotRef
) -> None:
    """Both copies of a duplicated file hold chunks.

    `src/shared.py` and `lib/shared.py` are byte-identical, so they
    share one row; each path must still reach it.
    """
    paths = {
        c.file_path for c in dup_store.get_chunks(dup_ref.snapshot_sha, repo_id=dup_ref.repo_id)
    }

    assert "src/shared.py" in paths
    assert "lib/shared.py" in paths


def test_list_symbols_finds_the_duplicated_copy(
    dup_store: IndexStore, dup_ref: SnapshotRef
) -> None:
    """`list-symbols` on the second copy returns its symbols.

    The outline of a file cannot depend on whether an identical file
    was indexed first.
    """
    names = [
        c.name
        for c in dup_store.get_chunks(
            dup_ref.snapshot_sha, repo_id=dup_ref.repo_id, file_path="lib/shared.py"
        )
    ]

    assert "parse_manifest" in names


# ── Copies stay distinguishable ──────────────────────────────────────


def test_same_bytes_in_two_languages_keep_separate_chunks(
    dup_store: IndexStore,
    dup_ref: SnapshotRef,
) -> None:
    """`api.d.ts` and `api.js` share bytes but not a grammar.

    The language a file was extracted as is part of a chunk's identity,
    so the two parses are separate rows and neither file's outline shows
    the other's. Keyed on content alone, the second copy would re-extract
    and the delete would take the first's chunks with it.
    """
    declarations = {
        c.language
        for c in dup_store.get_chunks(
            dup_ref.snapshot_sha, repo_id=dup_ref.repo_id, file_path="types/api.d.ts"
        )
    }
    bundle = {
        c.language
        for c in dup_store.get_chunks(
            dup_ref.snapshot_sha, repo_id=dup_ref.repo_id, file_path="dist/api.js"
        )
    }

    assert declarations == {"typescript"}
    assert bundle == {"javascript"}


def test_a_copy_adds_a_location_without_demoting_the_source(
    dup_store: IndexStore, dup_ref: SnapshotRef
) -> None:
    """A copy adds a location without dragging the ranking down.

    `src/dup.py` and `node_modules/dup.py` are the same bytes, so they
    are one result carrying both locations. Ranking still sees each
    separately — the join emits the row once per path, and the vendored
    copy scores 0.3 against the source's 1.0 — and the collapse keeps
    the better of the two. Copying a file into `node_modules` therefore
    cannot demote the original.
    """
    results = dup_store.search([dup_ref], "normalise_widget", top_k=10)
    hits = [r for r in results if "src/dup.py" in r.file_paths]

    assert len(hits) == 1, f"the copies did not collapse: {[r.file_paths for r in results]}"
    assert hits[0].file_paths == ["node_modules/dup.py", "src/dup.py"]
    assert hits[0].file_penalty == 1.0, "the vendored copy set the penalty"


# ── Edges are per location ───────────────────────────────────────────


def test_an_import_records_which_copy_it_came_from(
    dup_store: IndexStore, dup_ref: SnapshotRef
) -> None:
    """A referrer's location is the path that referred, not any path.

    `src/caller.py` imports `normalise_widget`, whose definition sits at
    both `src/dup.py` and `node_modules/dup.py` as one chunk. The edge
    records where each end sat when it was inferred, so find-refs can
    name the file that actually did the importing rather than every path
    the referring content happens to exist at.
    """
    edges = dup_store.get_edges(dup_ref.snapshot_sha, repo_id=dup_ref.repo_id)
    from_caller = [e for e in edges if e.source_path == "src/caller.py"]

    assert from_caller, f"no edge from the caller: {sorted({e.source_path for e in edges})}"
    assert all(e.target_path in {"src/dup.py", "node_modules/dup.py"} for e in from_caller)
