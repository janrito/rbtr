"""Behaviour tests and invariant guards for the tune objective.

The objective `_rescore_and_rank |> _mean_mrr` is what Bayesian
optimisation drives; the tests prove a better weight triple scores
higher than a worse one, and guard the invariants that
keep the search well-formed (a valid probability simplex, a complete
provenance→kind mapping, and a paste-able TOML report).
"""

from __future__ import annotations

import tomllib

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr_eval.queries import with_query_kind
from rbtr_eval.shared_schemas import QueryMeta, QueryRow
from rbtr_eval.tune import (
    DetailedOutcome,
    ImpactComparison,
    ScoredCandidate,
    TrialRow,
    TuneReport,
    Verdict,
    _mean_mrr,
    _paired_delta_se,
    _pool_recall,
    _render_tuning_report,
    _rescore_and_rank,
    _simplex_from_unit_square,
    _toml_snippet,
    _top_trial_spread,
    _verdict,
)

# ── Objective ────────────────────────────────────────────────────────────────


@parametrize_with_cases(
    "candidates, queries, weights, expected_ranks, expected_mrr",
    has_tag="rescore",
)
def test_objective_scores_weight_triples(
    candidates: dy.DataFrame[ScoredCandidate],
    queries: dy.DataFrame[QueryMeta],
    weights: tuple[float, float, float],
    expected_ranks: list[int | None],
    expected_mrr: float,
) -> None:
    """Re-scoring ranks the target and the objective MRR reflects it.

    Asserts both the intermediate ranks (localises a re-ranking bug)
    and the end MRR (the value Optuna optimises). The cases include the
    pair where semantic-heavy vs lexical-heavy weights flip the target
    between rank 1 (MRR 1.0) and rank 2 (MRR 0.5) — the behaviour that
    makes tuning worthwhile — plus the empty-candidates and
    outside-top-10 guards (null rank, zero MRR, no crash).
    """
    ranks = _rescore_and_rank(candidates, queries, weights)
    assert ranks["rank"].to_list() == expected_ranks

    mrr = _mean_mrr(ranks)
    assert mrr == pytest.approx(expected_mrr, abs=1e-6)


@pytest.fixture
def deep_pool() -> tuple[dy.DataFrame[ScoredCandidate], dy.DataFrame[QueryMeta]]:
    """One query whose target sits twelfth in a twelve-candidate pool.

    The decoys score above the target on every channel, so no weight
    triple can lift it into the top 10 — but it *was* retrieved, which
    is the difference the pool ceiling exists to show.
    """
    target = {
        "query_idx": 0,
        "file_paths": ["target.py"],
        "scope": "",
        "name": "fn",
        "line_start": 1,
        "line_end": 1,
        "symbol_kind": "function",
        "semantic": 0.1,
        "lexical": 0.1,
        "name_match": 0.1,
        "kind_boost": 1.0,
        "file_penalty": 1.0,
        "importance": 1.0,
        "proximity": 1.0,
    }
    decoys = [
        {**target, "file_paths": [f"decoy{i}.py"], "name": f"decoy{i}", "semantic": 0.9}
        for i in range(11)
    ]
    candidates = pl.DataFrame([*decoys, target]).pipe(ScoredCandidate.validate, cast=True)
    meta = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "slug": "r",
                "language": "python",
                "provenance": "name",
                "query_kind": "identifier",
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "line_end": 1,
                "symbol_kind": "function",
            }
        ]
    ).pipe(QueryMeta.validate, cast=True)
    return candidates, meta


def test_a_target_below_rank_10_still_counts_toward_the_pool_ceiling(
    deep_pool: tuple[dy.DataFrame[ScoredCandidate], dy.DataFrame[QueryMeta]],
) -> None:
    """Pool recall counts retrieval; the objective counts the top 10.

    The objective scores this query zero however the weights fall, so
    reading the objective alone suggests the target is unreachable.
    Pool recall says it was retrieved and merely ranked badly — the
    distinction between a ranking problem and a retrieval one.
    """
    candidates, meta = deep_pool

    ranks = _rescore_and_rank(candidates, meta, (0.4, 0.3, 0.3))
    recall = _pool_recall(candidates, meta)

    assert ranks["rank"].to_list() == [None]
    assert _mean_mrr(ranks) == pytest.approx(0.0)
    assert recall == pytest.approx(1.0)

    # A target in the top 10 is necessarily in the pool, so the ceiling
    # can never sit below the share of queries the objective scored.
    scored = ranks.filter(pl.col("rank").is_not_null()).height / ranks.height
    assert recall >= scored


def test_a_target_absent_from_the_pool_lowers_the_ceiling(
    deep_pool: tuple[dy.DataFrame[ScoredCandidate], dy.DataFrame[QueryMeta]],
) -> None:
    """A target no weighting could reach is what caps the objective.

    Dropping the target row leaves eleven decoys: the query is
    unscoreable by any triple, and the ceiling records that rather than
    blaming the ranking.
    """
    candidates, meta = deep_pool
    without_target = candidates.filter(pl.col("name") != "fn").pipe(
        ScoredCandidate.validate, cast=True
    )

    assert _pool_recall(without_target, meta) == pytest.approx(0.0)


# ── Invariant guards ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("u", "v", "expected"),
    [
        (0.0, 0.0, (0.0, 0.0, 1.0)),
        (1.0, 0.0, (1.0, 0.0, 0.0)),
        (0.0, 1.0, (0.0, 1.0, 0.0)),
        (0.5, 0.5, (0.5, 0.25, 0.25)),
    ],
)
def test_simplex_maps_unit_square_to_valid_triple(
    u: float, v: float, expected: tuple[float, float, float]
) -> None:
    """Unit-square points map to the specific simplex point, and the
    result is always a valid probability triple (sums to 1, in `[0, 1]`)
    — guarding against an invalid weight triple reaching search.
    """
    result = _simplex_from_unit_square(u, v)
    assert result == pytest.approx(expected)
    assert sum(result) == pytest.approx(1.0)
    assert all(0.0 <= c <= 1.0 for c in result)


@pytest.fixture
def query_kind_rows() -> dy.DataFrame[QueryRow]:
    """Queries whose text shape disagrees with their provenance.

    The same concept-shaped text appears under two provenances, and
    an identifier- and a code-shaped query carry provenances a
    provenance-based mapping would have labelled differently.
    """
    base = {"slug": "r", "scope": "", "symbol_kind": "function", "language": "python"}
    rows = [
        {
            **base,
            "file_path": "a.py",
            "name": "a",
            "line_start": 1,
            "line_end": 1,
            "provenance": "body",
            "text": "how does the config loader resolve paths",
        },
        {
            **base,
            "file_path": "a.py",
            "name": "a",
            "line_start": 1,
            "line_end": 1,
            "provenance": "docstring",
            "text": "how does the config loader resolve paths",
        },
        {
            **base,
            "file_path": "b.py",
            "name": "b",
            "line_start": 1,
            "line_end": 1,
            "provenance": "docstring",
            "text": "fuse_scores",
        },
        {
            **base,
            "file_path": "c.py",
            "name": "c",
            "line_start": 1,
            "line_end": 1,
            "provenance": "name",
            "text": "def fuse_scores(candidates, query, *, alpha):",
        },
    ]
    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True)


def test_query_kind_follows_text_not_provenance(
    query_kind_rows: dy.DataFrame[QueryRow],
) -> None:
    """A query's kind is a function of its request text, not its provenance.

    Tuning must partition queries the way production routes them — by
    `classify_query(text)` — so the same text gets the same kind
    regardless of how the query was generated, and the kind tracks the
    text's shape (concept / identifier / code), not the provenance.
    """
    tagged = with_query_kind(query_kind_rows)
    by_text = dict(zip(tagged["text"], tagged["query_kind"], strict=True))

    # Provenance-independence: same text under body and docstring agrees.
    concept_rows = tagged.filter(pl.col("text") == "how does the config loader resolve paths")
    assert concept_rows["query_kind"].n_unique() == 1

    # Kind tracks the text's shape, not the provenance.
    assert by_text["how does the config loader resolve paths"] == "concept"
    assert by_text["fuse_scores"] == "identifier"
    assert by_text["def fuse_scores(candidates, query, *, alpha):"] == "code"


@pytest.mark.parametrize(
    ("baseline_ranks", "best_ranks", "expected_se"),
    [
        ([2, 1, 1, 1], [2, 1, 1, 1], 0.0),
        ([2, 1, 1, 1], [1, 1, 1, 1], 0.125),
        ([1, 1, 1, 1], [1, 1, 1, None], 0.25),
    ],
)
def test_paired_error_measures_the_spread_of_the_differences(
    baseline_ranks: list[int | None], best_ranks: list[int | None], expected_se: float
) -> None:
    """The error is the standard error of the per-query differences.

    Both arms rank the same queries, so pairing cancels the
    query-to-query spread: identical ranks give exactly zero, which an
    unpaired error would not. Promoting one target of four gives
    differences `[0.5, 0, 0, 0]` — sample sd 0.25 over four queries,
    so 0.125. Dropping one target out of the top 10 gives `[0, 0, 0,
    -1]` — sd 0.5, so 0.25, which also pins the null rank scoring 0.
    """
    frames = [
        pl.DataFrame(
            {
                "query_idx": range(len(ranks)),
                "slug": ["r"] * len(ranks),
                "language": ["python"] * len(ranks),
                "provenance": ["name"] * len(ranks),
                "rank": ranks,
            }
        ).pipe(DetailedOutcome.validate, cast=True)
        for ranks in (baseline_ranks, best_ranks)
    ]

    assert _paired_delta_se(*frames) == pytest.approx(expected_se)


@pytest.mark.parametrize(
    ("delta", "delta_se", "expected"),
    [
        (0.10, 0.01, Verdict.RECOMMEND),
        (0.01, 0.10, Verdict.WITHHOLD),
        (0.03, 0.01, Verdict.RECOMMEND),
        (0.02, 0.01, Verdict.WITHHOLD),
        (-0.05, 0.0, Verdict.WITHHOLD),
    ],
)
def test_verdict_needs_the_delta_to_clear_twice_its_error(
    delta: float, delta_se: float, expected: Verdict
) -> None:
    """A delta inside twice its own error earns no recommendation.

    The boundary case (delta exactly twice the error) withholds: the
    comparison is strict, so a tie is not evidence.
    """
    assert _verdict(delta, delta_se) is expected


@pytest.fixture
def flat_objective_trials() -> list[TrialRow]:
    """Twelve trials spanning the simplex at near-identical MRR.

    alpha walks 0.05 to 0.93 while MRR moves by less than 0.002 — an
    objective that cannot tell these weights apart, which is the
    shape the spread exists to expose.
    """
    return [
        {
            "kind": "concept",
            "trial": i + 1,
            "alpha": 0.05 + 0.08 * i,
            "beta": 0.9 - 0.08 * i,
            "gamma": 0.05,
            "mrr": 0.500 + 0.0001 * i,
            "best_so_far": 0.500 + 0.0001 * i,
        }
        for i in range(12)
    ]


def test_spread_exposes_an_objective_that_cannot_choose(
    flat_objective_trials: list[TrialRow],
) -> None:
    """Wide weights at one MRR are reported as wide weights at one MRR.

    The argmax alone would present one 15-decimal triple as an answer.
    The spread shows the top ten covering most of the alpha axis while
    their scores differ in the fourth decimal.
    """
    spread = _top_trial_spread(flat_objective_trials)

    row = spread.row(0, named=True)
    assert row["kind"] == "concept"
    assert row["n_top"] == 10
    assert row["alpha_max"] - row["alpha_min"] > 0.5
    assert row["mrr_range"] < 0.01


def test_spread_is_reported_per_kind(flat_objective_trials: list[TrialRow]) -> None:
    """Kinds are separate studies, so their trials never pool.

    Without a kind on each trial row the three studies' trials would
    concatenate into one ranking and the spread would describe no
    study at all.
    """
    code_trials: list[TrialRow] = [
        {**trial, "kind": "code", "mrr": 0.8} for trial in flat_objective_trials
    ]

    spread = _top_trial_spread([*flat_objective_trials, *code_trials])

    assert spread["kind"].to_list() == ["concept", "code"]
    assert spread.filter(pl.col("kind") == "code")["mrr_range"].item() == pytest.approx(0.0)


@pytest.fixture
def two_kind_report() -> dy.DataFrame[TuneReport]:
    """A tuning report for two kinds, with every score distinct.

    Each kind is tuned as a separate optimisation problem, so the
    report carries one row per kind. Distinct scores let a test tell
    which row a rendered figure came from.
    """
    return pl.DataFrame(
        {
            "kind": ["concept", "identifier"],
            "best_alpha": [0.5, 0.05],
            "best_beta": [0.3, 0.2],
            "best_gamma": [0.2, 0.75],
            "score_best": [0.37, 0.58],
            "current_alpha": [0.1, 0.1],
            "current_beta": [0.3, 0.3],
            "current_gamma": [0.6, 0.6],
            "score_current": [0.31, 0.42],
            "delta": [0.06, 0.16],
            "delta_se": [0.01, 0.09],
            "verdict": [Verdict.RECOMMEND, Verdict.WITHHOLD],
            "pool_recall": [0.78, 0.91],
            "metric": ["MRR", "MRR"],
            "n_trials": [10, 10],
            "n_queries": [50, 70],
            "elapsed_seconds": [1.0, 1.0],
        }
    ).pipe(TuneReport.validate, cast=True)


@pytest.fixture
def rollup_impact() -> dy.DataFrame[ImpactComparison]:
    """A single all-dimensions rollup row, the shape the report renders."""
    return pl.DataFrame(
        {
            "slug": ["__all__"],
            "language": ["__all__"],
            "provenance": ["__all__"],
            "baseline_mrr": [0.4],
            "best_mrr": [0.45],
            "delta": [0.05],
            "baseline_ndcg_at_10": [0.5],
            "best_ndcg_at_10": [0.55],
            "delta_ndcg_at_10": [0.05],
        }
    ).pipe(ImpactComparison.validate, cast=True)


def test_result_table_reports_every_kind_against_its_error(
    two_kind_report: dy.DataFrame[TuneReport],
    rollup_impact: dy.DataFrame[ImpactComparison],
    flat_objective_trials: list[TrialRow],
) -> None:
    """Every kind reaches the Result table, with the error it must clear.

    Each kind is tuned independently against its own query set, so a
    single current/recommended pair describes one of them and
    misdescribes the rest. A delta printed alone reads as a result;
    printed beside its own uncertainty and a verdict, a delta inside
    the noise is visible as one.
    """
    rendered = _render_tuning_report(
        two_kind_report,
        flat_objective_trials,
        rollup_impact,
        pl.DataFrame({"slug": ["r"], "n_queries": [50]}),
    )

    result_section = rendered.split("## Result")[1].split("##")[0]

    assert "concept" in result_section
    assert "identifier" in result_section
    assert "0.31" in result_section
    assert "0.42" in result_section
    assert "0.01" in result_section
    assert "0.09" in result_section
    assert "recommend" in result_section
    assert "withhold" in result_section


def test_toml_snippet_is_valid_toml() -> None:
    """The report snippet parses as TOML and round-trips the weights.

    Guards against emitting config the operator cannot paste into
    `rbtr`'s settings.
    """
    report = pl.DataFrame(
        {
            "kind": ["concept", "identifier"],
            "best_alpha": [0.5, 0.05],
            "best_beta": [0.3, 0.2],
            "best_gamma": [0.2, 0.75],
            "score_best": [0.4, 0.5],
            "current_alpha": [0.1, 0.1],
            "current_beta": [0.3, 0.3],
            "current_gamma": [0.6, 0.6],
            "score_current": [0.3, 0.4],
            "delta": [0.1, 0.1],
            "delta_se": [0.01, 0.01],
            "verdict": [Verdict.RECOMMEND, Verdict.RECOMMEND],
            "pool_recall": [0.8, 0.8],
            "metric": ["MRR", "MRR"],
            "n_trials": [10, 10],
            "n_queries": [50, 50],
            "elapsed_seconds": [1.0, 1.0],
        }
    ).pipe(TuneReport.validate, cast=True)

    parsed = tomllib.loads(_toml_snippet(report))

    weights = parsed["search_weights"]
    assert weights["concept"] == {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
    assert weights["identifier"] == {"alpha": 0.05, "beta": 0.2, "gamma": 0.75}


def test_toml_snippet_comments_out_a_withheld_kind(
    two_kind_report: dy.DataFrame[TuneReport],
) -> None:
    """Weights the report declined to recommend cannot be pasted by accident.

    `identifier` is withheld in the fixture, so its block is present to
    read but commented out; pasting the snippet ships only the kind the
    report stands behind. The result must still parse as TOML.
    """
    snippet = _toml_snippet(two_kind_report)
    parsed = tomllib.loads(snippet)

    assert parsed["search_weights"] == {"concept": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}}
    assert "# [search_weights.identifier]" in snippet
    assert "# gamma = 0.75" in snippet
