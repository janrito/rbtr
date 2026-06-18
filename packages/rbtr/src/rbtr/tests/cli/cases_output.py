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
from rbtr.daemon.messages import IndexedRef, SearchResponse, StatusResponse, WatchedRef
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


@case(tags=["status"])
def case_status_grouped_by_repo() -> RenderScenario:
    """Cross-repo status groups refs under each repo path."""
    return RenderScenario(
        model=StatusResponse(
            db_path="/db",
            indexed_refs=[
                IndexedRef(sha="a" * 40, total=10, embedded=10, repo_path="/projects/one"),
                IndexedRef(sha="b" * 40, total=20, embedded=20, repo_path="/projects/two"),
            ],
        ),
        # "indexed repos" header is the cross-repo grouping cue,
        # rendered only in TTY mode — proves the rich path, not JSON.
        expected=("indexed repos", "/projects/one", "/projects/two"),
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
