"""Input/output cases for `rbtr.index.reranker.Reranker.rerank`.

Each `RerankScenario` describes candidates (with pre-fused scores),
stub model scores, and the expected-output property to assert.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RerankScenario:
    """One reranking test scenario."""

    # ── Inputs ────────────────────────────────────────────────────
    query: str = "search query"
    # Pre-fused candidates: id → (fusion_score, content).
    candidates: dict[str, tuple[float, str]] = field(default_factory=dict)
    # Stub model scores: id → raw cross-encoder score.
    reranker_scores: dict[str, float] = field(default_factory=dict)
    blend_weight: float = 0.5
    top_k: int = 10
    # When True, stub model raises instead of scoring.
    model_raises: bool = False

    # ── Expected outputs (None = not asserted) ────────────────────
    expected_top: str | None = None
    expected_order: list[str] | None = None
    expected_count: int | None = None


# ── Core reordering ──────────────────────────────────────────────────


def case_reranker_promotes_lower_fusion_candidate() -> RerankScenario:
    """Candidate B has lower fusion but higher reranker score."""
    return RerankScenario(
        candidates={
            "A": (0.9, "class A: pass"),
            "B": (0.3, "class B(Base): ..."),
        },
        reranker_scores={"A": 0.1, "B": 0.95},
        blend_weight=0.5,
        expected_top="B",
    )


# ── Blend weight extremes ───────────────────────────────────────────


def case_blend_weight_zero_ignores_fusion() -> RerankScenario:
    """w=0.0 means pure reranker — fusion scores don't matter."""
    return RerankScenario(
        candidates={
            "X": (0.9, "x content"),
            "Y": (0.1, "y content"),
            "Z": (0.5, "z content"),
        },
        reranker_scores={"X": 0.2, "Y": 0.8, "Z": 0.5},
        blend_weight=0.0,
        expected_order=["Y", "Z", "X"],
    )


def case_blend_weight_one_ignores_reranker() -> RerankScenario:
    """w=1.0 means pure fusion — reranker scores don't matter."""
    return RerankScenario(
        candidates={
            "X": (0.9, "x content"),
            "Y": (0.1, "y content"),
            "Z": (0.5, "z content"),
        },
        reranker_scores={"X": 0.0, "Y": 1.0, "Z": 0.5},
        blend_weight=1.0,
        expected_order=["X", "Z", "Y"],
    )


# ── Edge cases ───────────────────────────────────────────────────────


def case_single_candidate() -> RerankScenario:
    """One candidate normalises to 1.0."""
    return RerankScenario(
        candidates={"solo": (0.7, "only candidate")},
        reranker_scores={"solo": 0.42},
        expected_count=1,
        expected_top="solo",
    )


def case_equal_reranker_scores() -> RerankScenario:
    """All candidates get identical raw scores — fusion order preserved."""
    return RerankScenario(
        candidates={
            "A": (0.9, "a"),
            "B": (0.5, "b"),
            "C": (0.1, "c"),
        },
        reranker_scores={"A": 0.5, "B": 0.5, "C": 0.5},
        blend_weight=0.5,
        expected_order=["A", "B", "C"],
    )


def case_empty_frame() -> RerankScenario:
    """No candidates — returns empty."""
    return RerankScenario(expected_count=0)


# ── Failure ──────────────────────────────────────────────────────────


def case_model_failure() -> RerankScenario:
    """Model raises — falls back to fusion order, reranker stays 0.0."""
    return RerankScenario(
        candidates={
            "A": (0.9, "a"),
            "B": (0.5, "b"),
            "C": (0.1, "c"),
        },
        model_raises=True,
        expected_order=["A", "B", "C"],
    )
