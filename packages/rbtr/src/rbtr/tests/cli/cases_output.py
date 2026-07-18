"""Cases for CLI output rendering of cross-repo attribution.

Each case returns a response model to `emit()` plus the
substrings the rendered TTY output must (and must not) contain.
Behaviour, not implementation: cases drive the public `emit`
surface, never the private renderers.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from pytest_cases import case

from rbtr.daemon.dto import SearchHitOut
from rbtr.daemon.messages import (
    GcResponse,
    IndexedRef,
    SearchResponse,
    StatusResponse,
    WatchedRef,
)
from rbtr.index.models import ChunkKind


@dataclass(frozen=True)
class RenderScenario:
    """A model to emit and what its rendered output should show."""

    model: BaseModel
    expected: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


def _hit(*, repo_path: str | None) -> SearchHitOut:
    """A search hit for one file, optionally repo-attributed."""
    return SearchHitOut(
        repo_path=repo_path,
        file_path="src/main.py",
        kind=ChunkKind.FUNCTION,
        name="main",
        content="def main(): ...",
        line_start=1,
        line_end=1,
        score=0.9,
    )


@case(tags=["search"])
def case_search_attributed() -> RenderScenario:
    """A cross-repo hit is prefixed with its repo name."""
    return RenderScenario(
        model=SearchResponse(results=[_hit(repo_path="/projects/widgets")]),
        expected=("widgets/src/main.py",),
    )


@case(tags=["search"])
def case_search_unattributed() -> RenderScenario:
    """A workspace hit shows the bare path, no repo prefix."""
    return RenderScenario(
        model=SearchResponse(results=[_hit(repo_path=None)]),
        expected=("src/main.py",),
        forbidden=("widgets/src/main.py",),
    )


@case(tags=["search"])
def case_search_preview_anchors_on_match() -> RenderScenario:
    """A deep match shows the signature line, then the matched line.

    The match sits at offset 5, past the 4-line window, so the
    preview keeps the chunk's first line (the signature) for
    orientation, drops the lines in between, and anchors on the
    matched line.
    """
    content = """\
def big_function():
a = 1
GAP_HIDDEN = 2
c = 3
d = 4
NEEDLE_HERE = 5
e = 6
f = 7
"""
    hit = SearchHitOut(
        file_path="src/big.py",
        kind=ChunkKind.FUNCTION,
        name="big",
        language="python",
        content=content,
        line_start=1,
        line_end=9,
        score=0.9,
        match_line_offset=5,
        matched_terms=["needle_here"],
    )
    return RenderScenario(
        model=SearchResponse(results=[hit]),
        expected=("big_function", "NEEDLE_HERE"),
        forbidden=("GAP_HIDDEN",),
    )


@case(tags=["search"])
def case_search_preview_without_match_shows_head() -> RenderScenario:
    """With no anchor, the preview keeps the first-N lines."""
    content = """\
def visible_top():
    return 1
"""
    hit = SearchHitOut(
        file_path="src/top.py",
        kind=ChunkKind.FUNCTION,
        name="top",
        language="python",
        content=content,
        line_start=1,
        line_end=3,
        score=0.9,
    )
    return RenderScenario(
        model=SearchResponse(results=[hit]),
        expected=("visible_top",),
    )


@case(tags=["status"])
def case_status_grouped_by_repo() -> RenderScenario:
    """Cross-repo status groups refs under each repo path."""
    return RenderScenario(
        model=StatusResponse(
            db_path="/db",
            db_size_bytes=1_500_000,  # 1.5 MB
            indexed_refs=[
                IndexedRef(sha="a" * 40, total=10, embedded=10, repo_path="/projects/one"),
                IndexedRef(sha="b" * 40, total=20, embedded=20, repo_path="/projects/two"),
            ],
        ),
        # "indexed repos" header is the cross-repo grouping cue,
        # rendered only in TTY mode — proves the rich path, not JSON. The
        # whole-DB size renders on that line too.
        expected=("indexed repos", "1.5 MB", "/projects/one", "/projects/two"),
    )


@case(tags=["gc"])
def case_gc_reports_freed_chunks_and_scope() -> RenderScenario:
    """GC output shows chunks freed and the repo scope, beside per-repo counts."""
    return RenderScenario(
        model=GcResponse(
            repos_collected=3,
            commits_dropped=2,
            snapshots_dropped=5,
            edges_dropped=3,
            chunks_freed=4,
            elapsed_seconds=0.1,
        ),
        expected=("2 commits", "freed 4 chunks", "across 3 repos"),
    )


@case(tags=["gc"])
def case_gc_reports_compaction_shrink() -> RenderScenario:
    """A compacting gc shows before → after and the signed byte change."""
    return RenderScenario(
        model=GcResponse(
            repos_collected=1,
            commits_dropped=1,
            snapshots_dropped=1,
            edges_dropped=1,
            chunks_freed=1,
            size_before_bytes=2_000_000_000,  # 2.0 GB
            size_after_bytes=1_000_000_000,  # 1.0 GB
            elapsed_seconds=0.1,
        ),
        expected=("index 2.0 GB → 1.0 GB", "-1.0 GB"),
    )


@case(tags=["gc"])
def case_gc_reports_compaction_growth() -> RenderScenario:
    """A rewrite that grows the file says so — growth is not hidden."""
    return RenderScenario(
        model=GcResponse(
            repos_collected=1,
            commits_dropped=0,
            snapshots_dropped=0,
            edges_dropped=0,
            chunks_freed=0,
            size_before_bytes=1_000_000,  # 1.0 MB
            size_after_bytes=2_000_000,  # 2.0 MB
            elapsed_seconds=0.1,
        ),
        expected=("index 1.0 MB → 2.0 MB", "+1.0 MB"),
    )


@case(tags=["gc"])
def case_gc_reports_compaction_unchanged() -> RenderScenario:
    """No byte change shows just the size, with no arrow or delta."""
    return RenderScenario(
        model=GcResponse(
            repos_collected=1,
            commits_dropped=0,
            snapshots_dropped=0,
            edges_dropped=0,
            chunks_freed=0,
            size_before_bytes=1_000_000,  # 1.0 MB
            size_after_bytes=1_000_000,
            elapsed_seconds=0.1,
        ),
        expected=("index 1.0 MB",),
        forbidden=("→",),
    )


@case(tags=["status"])
def case_status_shows_watch_set() -> RenderScenario:
    """Watched refs render with indexed / pending / unresolvable markers."""
    return RenderScenario(
        model=StatusResponse(
            db_path="/db",
            indexed_refs=[IndexedRef(sha="a" * 40, total=3, embedded=3)],
            watched=[
                WatchedRef(ref="main", sha="a" * 40, indexed=True),
                WatchedRef(ref="feature", sha="b" * 40, indexed=False),
                WatchedRef(ref="deleted", sha=None, indexed=False),
            ],
        ),
        expected=("watching:", "main", "pending", "unresolvable"),
    )
