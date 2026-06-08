"""Input/output cases for `rbtr.index.search.fuse_scores`.

Each `FuseScenario` describes exactly what the caller supplies to
`fuse_scores` (inputs) and what ranked-result property the
test should assert (expected outputs).  The test file builds a
polars frame from the `chunks` list inside a fixture.

Nothing is implicit: every field the test cares about is named
in the case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import ChunkKind

from .cases_common import ChunkSpec


@dataclass(frozen=True)
class FuseScenario:
    # ── Inputs to fuse_scores ────────────────────────────────────
    chunks: list[ChunkSpec] = field(default_factory=list)
    lexical: dict[str, float] = field(default_factory=dict)
    semantic: dict[str, float] = field(default_factory=dict)
    query: str = ""
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    top_k: int = 10

    # ── Expected output properties ───────────────────────────────
    #
    # Set only the ones the test cares about.  ``None`` means "not
    # asserted by this case".

    # Total number of ``ScoredChunk`` objects returned.
    expected_count: int | None = None
    # Id that must rank first.
    expected_top: str | None = None
    # Full ranked id list (exhaustive order).
    expected_order: list[str] | None = None
    # Ids that must appear in the result, in any order.
    expected_ids: list[str] | None = None
    # Rank 0's score must satisfy approx(expected_first_score).
    expected_first_score_approx: float | None = None


# ── Degenerate ───────────────────────────────────────────────────────


def case_empty_inputs_yields_empty_list() -> FuseScenario:
    return FuseScenario(expected_count=0)


def case_single_result_normalises_to_zero() -> FuseScenario:
    """One chunk → normalised scores are 0.0 on every channel."""
    return FuseScenario(
        chunks=[ChunkSpec(id="a")],
        lexical={"a": 1.0},
        semantic={"a": 1.0},
        expected_count=1,
        expected_top="a",
        expected_first_score_approx=0.0,
    )


# ── Ranking: lexical-only ───────────────────────────────────────────


def case_lexical_only_ranks_higher_score_first() -> FuseScenario:
    """With alpha=0, gamma=0, beta=1, lexical alone decides order."""
    return FuseScenario(
        chunks=[ChunkSpec(id="a"), ChunkSpec(id="b")],
        lexical={"a": 1.0, "b": 5.0},
        alpha=0.0,
        beta=1.0,
        gamma=0.0,
        expected_order=["b", "a"],
    )


# ── Ranking: name match vs higher lexical ──────────────────────────


def case_exact_name_match_beats_higher_lexical() -> FuseScenario:
    """Name score resolves a tie the lexical channel is losing.

    query="IndexStore" gives an exact name match (1.0) for "def"
    and no match (0.0) for "test_search".
    """
    return FuseScenario(
        chunks=[
            ChunkSpec(id="def", name="IndexStore"),
            ChunkSpec(id="test", name="test_search", file_path="tests/test_store.py"),
        ],
        lexical={"def": 1.0, "test": 10.0},
        query="IndexStore",
        expected_top="def",
    )


# ── Ranking: kind boost ──────────────────────────────────────────────


def case_class_outranks_import_via_kind_boost() -> FuseScenario:
    """Identical lexical, no name signal, but CLASS boost > IMPORT boost."""
    return FuseScenario(
        chunks=[
            ChunkSpec(id="cls", kind=ChunkKind.CLASS, name="Engine"),
            ChunkSpec(id="imp", kind=ChunkKind.IMPORT, name="from .core import Engine"),
        ],
        lexical={"cls": 1.0, "imp": 1.0},
        expected_top="cls",
    )


# ── Ranking: file category penalty ──────────────────────────────────


def case_source_outranks_test_via_file_penalty() -> FuseScenario:
    """Source beats test despite 44x more mentions.

    query="build_index" gives exact match (1.0) for src and
    substring match (0.5) for "test_build_index", reinforcing
    the file penalty.
    """
    return FuseScenario(
        chunks=[
            ChunkSpec(id="src", name="build_index", file_path="src/orchestrator.py"),
            ChunkSpec(
                id="tst",
                name="test_build_index",
                file_path="tests/test_orchestrator.py",
            ),
        ],
        lexical={"src": 1.0, "tst": 44.0},
        query="build_index",
        expected_top="src",
    )


# ── Shape: ScoredChunk fields populated ────────────────────────────


def case_score_breakdown_populated_for_every_result() -> FuseScenario:
    """Every `ScoredChunk` has positive kind_boost and file_penalty."""
    return FuseScenario(
        chunks=[
            ChunkSpec(id="a", kind=ChunkKind.CLASS, file_path="src/lib.py"),
            ChunkSpec(id="b", kind=ChunkKind.IMPORT, file_path="tests/test.py"),
        ],
        lexical={"a": 2.0, "b": 1.0},
        semantic={"a": 0.8, "b": 0.2},
        expected_count=2,
    )


# ── Truncation: top_k ───────────────────────────────────────────────


def case_top_k_limits_output() -> FuseScenario:
    return FuseScenario(
        chunks=[ChunkSpec(id=f"c{i}") for i in range(20)],
        lexical={f"c{i}": float(i) for i in range(20)},
        top_k=5,
        expected_count=5,
    )


# ── Partial input: missing channels ─────────────────────────────────


def case_chunks_missing_from_a_channel_get_zero() -> FuseScenario:
    """A chunk only in lexical, another only in semantic — both appear."""
    return FuseScenario(
        chunks=[ChunkSpec(id="a"), ChunkSpec(id="b")],
        lexical={"a": 5.0},
        semantic={"b": 0.9},
        expected_ids=["a", "b"],
        expected_count=2,
    )
