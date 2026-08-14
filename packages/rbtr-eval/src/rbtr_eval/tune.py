"""`rbtr-eval tune` subcommand.

Bayesian-optimise the rbtr search fusion weights `(alpha,
beta, gamma)` against every per-repo query set, using the
index in the shared home.  Reports best vs current weights
in `data/TUNING.md`; never edits source.

Uses Optuna's TPESampler on a unit-square reparameterisation
of the simplex (2 free dimensions).  Each trial evaluates
one weight triple against the full (capped) query set and
returns MRR.  Ask-and-tell interface for progress control.
"""

from __future__ import annotations

import json
import math
import time
from enum import StrEnum
from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import optuna
import polars as pl
import tomli_w
from pydantic import BaseModel, Field, TypeAdapter

from rbtr.cli.output import ProgressCallback, progress_reporter
from rbtr.config import WeightTriple, config as rbtr_config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.domain.models import ChunkKind, QueryKind
from rbtr_eval.agg import search_metric_aggs
from rbtr_eval.charts import render_vl_to_png
from rbtr_eval.formatting import md_table
from rbtr_eval.queries import load_all_queries, sample_distribution, subsample, with_query_kind
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.shared_schemas import (
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
    QueryMeta,
    QueryRow,
)


class Verdict(StrEnum):
    """Whether a tuned weight triple is worth adopting.

    A tuning delta is a difference between two means over a few hundred
    queries, so it carries an error of its own.  `RECOMMEND` says the
    delta is larger than twice that error; `WITHHOLD` says the run
    cannot tell the tuned weights from the current ones.
    """

    RECOMMEND = "recommend"
    WITHHOLD = "withhold"


class ScoredCandidate(dy.Schema):
    """One candidate per query, carrying all component scores.

    Produced by `tune._collect_scored_candidates`; consumed
    by `tune._rescore_and_rank`.

    `file_paths` holds every location the candidate's content sits at,
    as the daemon returns it, so a target is reached when its path is
    one of them.
    """

    query_idx = dy.UInt32()
    file_paths = dy.List(dy.String())
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(k.value for k in ChunkKind)
    semantic = dy.Float64()
    lexical = dy.Float64()
    name_match = dy.Float64()
    kind_boost = dy.Float64()
    file_penalty = dy.Float64()
    importance = dy.Float64()
    proximity = dy.Float64()


class DetailedOutcome(dy.Schema):
    """Per-query rank from a single weight configuration.

    Produced by `tune._rescore_and_rank`; consumed by
    `tune._impact_comparison` and `tune._paired_delta_se`.

    `query_idx` identifies the query within its kind's run, so two
    frames scored under different weights can be compared query by
    query rather than only in aggregate.
    """

    query_idx = dy.UInt32(min=0)
    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    rank = dy.UInt8(nullable=True, min=1, max=10)


class ImpactComparison(dy.Schema):
    """Side-by-side MRR for baseline vs best weights.

    One row per rollup dimension (repo, language, provenance,
    and `__all__` sentinels for rollups).
    """

    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    baseline_mrr = dy.Float64(min=0.0, max=1.0)
    best_mrr = dy.Float64(min=0.0, max=1.0)
    delta = dy.Float64(min=-1.0, max=1.0)
    baseline_ndcg_at_10 = dy.Float64(min=0.0, max=1.0)
    best_ndcg_at_10 = dy.Float64(min=0.0, max=1.0)
    delta_ndcg_at_10 = dy.Float64(min=-1.0, max=1.0)


class TuneReport(dy.Schema):
    """One study's outcome, one row per `QueryKind`.

    Held in memory for the length of the run and rendered into
    `TUNING.md`; nothing persists it.
    """

    kind = dy.String()
    best_alpha = dy.Float64(min=0.0, max=1.0)
    best_beta = dy.Float64(min=0.0, max=1.0)
    best_gamma = dy.Float64(min=0.0, max=1.0)
    score_best = dy.Float64(min=0.0, max=1.0)
    current_alpha = dy.Float64(min=0.0, max=1.0)
    current_beta = dy.Float64(min=0.0, max=1.0)
    current_gamma = dy.Float64(min=0.0, max=1.0)
    score_current = dy.Float64(min=0.0, max=1.0)
    delta = dy.Float64(min=-1.0, max=1.0)
    delta_se = dy.Float64(min=0.0, max=1.0)
    """Standard error of `delta`, from the per-query paired differences."""
    verdict = dy.Enum(tuple(v.value for v in Verdict))
    pool_recall = dy.Float64(min=0.0, max=1.0)
    """Share of queries whose target reached the pool the trials rerank.

    The objective cannot exceed this: a target the default weights did
    not retrieve is unscoreable by every weight triple.
    """
    metric = dy.String()
    n_trials = dy.UInt32(min=1)
    n_queries = dy.UInt32(min=0)
    elapsed_seconds = dy.Float64(min=0.0)


class TrialSpread(dy.Schema):
    """One row of TUNING.md's top-trial spread table.

    One row per `QueryKind`.  The weight ranges are taken over the ten
    best-scoring trials of that kind's study and `mrr_range` over their
    scores, so wide ranges beside a narrow `mrr_range` say the
    objective cannot separate the weights it is choosing between.

    Validated because the trials arrive as hand-built
    `dict[str, float | int | str]` rows whose dtypes polars infers from
    the Python objects — the same untyped boundary the INDEX.md count
    rows cross from a DuckDB cursor.
    """

    kind = dy.String()
    n_top = dy.UInt32(min=1)
    alpha_min = dy.Float64(min=0.0, max=1.0)
    alpha_max = dy.Float64(min=0.0, max=1.0)
    beta_min = dy.Float64(min=0.0, max=1.0)
    beta_max = dy.Float64(min=0.0, max=1.0)
    gamma_min = dy.Float64(min=0.0, max=1.0)
    gamma_max = dy.Float64(min=0.0, max=1.0)
    mrr_range = dy.Float64(min=0.0, max=1.0)


type TrialRow = dict[str, float | int | str]
"""One Optuna trial's record: its kind, index, weights and scores."""

# ── Scored-candidate collection ──────────────────────────────────────────────


def _collect_scored_candidates(
    client: DaemonClient,
    queries: dy.DataFrame[QueryRow],
    repos_dir: Path,
    on_progress: ProgressCallback | None = None,
    offset: int = 0,
    total: int = 0,
) -> tuple[dy.DataFrame[ScoredCandidate], dy.DataFrame[QueryMeta]]:
    """One RPC per query, returning all component scores.

    Sends `SearchRequest(limit=50)` with no weight override so
    that normalisation uses default weights.  Returns a candidate
    frame (one row per result per query) and a query-metadata
    frame (one row per query).
    """
    rows: list[dict[str, object]] = []
    for i, query in enumerate(queries.iter_rows(named=True), 1):
        repo_path = (repos_dir / query["slug"]).resolve()
        resp = client.send_or_raise_as(
            SearchResponse,
            SearchRequest(
                repo_path=str(repo_path),
                query=query["text"],
                limit=50,
                explain=True,
            ),
        )
        idx = i - 1  # 0-based to match with_row_index
        for r in resp.results:
            signals = r.signals
            if signals is None:
                msg = "search must return signals when explain=True"
                raise RuntimeError(msg)
            rows.append(
                {
                    "query_idx": idx,
                    "file_paths": r.file_paths,
                    "scope": r.scope,
                    "name": r.name,
                    "line_start": r.line_start,
                    "line_end": r.line_end,
                    "symbol_kind": r.kind.value,
                    "semantic": signals.semantic,
                    "lexical": signals.lexical,
                    "name_match": signals.name_match,
                    "kind_boost": signals.kind_boost,
                    "file_penalty": signals.file_penalty,
                    "importance": signals.importance,
                    "proximity": signals.proximity,
                }
            )
        if on_progress is not None:
            on_progress(offset + i, total)

    candidates = pl.DataFrame(rows, schema=ScoredCandidate.create_empty().schema).pipe(
        ScoredCandidate.validate, cast=True
    )

    meta = (
        with_query_kind(queries)
        .with_row_index("query_idx")
        .select(
            "query_idx",
            "slug",
            "language",
            "provenance",
            "query_kind",
            *IDENTITY_COLUMNS,
        )
        .pipe(QueryMeta.validate, cast=True)
    )
    return candidates, meta


# ── Re-score and rank ─────────────────────────────────────────────────────────


def _pool_recall(
    candidates: dy.DataFrame[ScoredCandidate],
    queries: dy.DataFrame[QueryMeta],
) -> float:
    """Fraction of queries whose target is anywhere in the retrieved pool.

    The ceiling on the tuning objective.  The pool is fetched once,
    under the *current default* weights, and every trial reranks that
    same frame — so a target the defaults did not retrieve scores zero
    for every triple, and a weight set whose merit would be retrieving
    it earns nothing for that.  Reported so a low objective can be read
    for what it is: either the ranking is poor, or the answer was never
    in the pool to rank.
    """
    in_pool = candidates.join(queries.select("query_idx", "file_path"), on="query_idx").filter(
        pl.col("file_paths").list.contains(pl.col("file_path"))
    )
    hits = queries.join(
        in_pool.select("query_idx", *MATCH_COLUMNS),
        on=["query_idx", *MATCH_COLUMNS],
        how="semi",
    )
    return hits.height / queries.height


def _rescore_and_rank(
    candidates: dy.DataFrame[ScoredCandidate],
    queries: dy.DataFrame[QueryMeta],
    weights: tuple[float, float, float],
) -> dy.DataFrame[DetailedOutcome]:
    """Re-rank candidates with trial weights, no RPC.

    Computes `(a*semantic + b*lexical + g*name_match)
    * kind_boost * file_penalty * importance * proximity`,
    ranks within each query, and joins back to `queries`
    so every query appears (null rank when the target
    is outside the top 10).
    """
    alpha, beta, gamma = weights

    ranked = candidates.with_columns(
        (
            (
                (pl.lit(alpha) * pl.col("semantic"))
                + (pl.lit(beta) * pl.col("lexical"))
                + (pl.lit(gamma) * pl.col("name_match"))
            )
            * pl.col("kind_boost")
            * pl.col("file_penalty")
            * pl.col("importance")
            * pl.col("proximity")
        ).alias("score")
    )
    ranked = ranked.with_columns(
        pl.col("score")
        .rank("ordinal", descending=True)
        .over("query_idx")
        .cast(pl.UInt8)
        .alias("rank")
    )
    top = ranked.filter(pl.col("rank") <= 10)

    target = top.join(queries.select("query_idx", "file_path"), on="query_idx").filter(
        pl.col("file_paths").list.contains(pl.col("file_path"))
    )

    return (
        queries.join(
            target.select("query_idx", *MATCH_COLUMNS, "rank"),
            on=["query_idx", *MATCH_COLUMNS],
            how="left",
        )
        .select("query_idx", "slug", "language", "provenance", "rank")
        .pipe(DetailedOutcome.validate, cast=True)
    )


def _mrr_per_provenance(
    outcomes: dy.DataFrame[DetailedOutcome],
) -> pl.DataFrame:
    """MRR per provenance from per-query ranks.

    Returns a frame with `provenance` and `mrr` columns.
    """
    return (
        outcomes.group_by("provenance")
        .agg(*search_metric_aggs())
        .select("provenance", "mrr")
        .sort("provenance")
    )


def _mean_mrr(ranks: dy.DataFrame[DetailedOutcome]) -> float:
    """Micro-averaged MRR over all queries, reusing `search_metric_aggs`.

    One value per query kind's whole query set — the mean reciprocal
    rank, matching the `mrr` the benchmark reports. Unlike a harmonic
    mean over buckets, a single weak bucket cannot collapse it.
    """
    return ranks.select(*search_metric_aggs()).get_column("mrr").item()


def _paired_delta_se(
    baseline: dy.DataFrame[DetailedOutcome],
    best: dy.DataFrame[DetailedOutcome],
) -> float:
    """Standard error of the MRR difference between two weight triples.

    Both arms rank the same queries over the same candidate pool and
    differ only in the weights, so the comparison is paired: the error
    is taken over the *per-query differences* in reciprocal rank, not
    over each arm's own spread.  That distinction dominates the result
    — most queries rank identically under both arms and contribute a
    difference of exactly zero, which an unpaired error would instead
    count as two large and independent variances.

    A rank outside the top 10 arrives null and scores 0, matching
    `search_metric_aggs`.  Returns 0.0 when fewer than two queries are
    compared, there being no spread to measure.
    """
    reciprocal = pl.col("rank").cast(pl.Float64).pow(-1).fill_null(0.0)
    paired = baseline.select("query_idx", reciprocal.alias("base_rr")).join(
        best.select("query_idx", reciprocal.alias("best_rr")),
        on="query_idx",
    )
    if paired.height < 2:
        return 0.0

    spread = paired.select(
        (pl.col("best_rr") - pl.col("base_rr")).std().alias("sd"),
    ).item()
    return float(spread) / math.sqrt(paired.height)


def _top_trial_spread(trials: list[TrialRow]) -> dy.DataFrame[TrialSpread]:
    """Per-kind weight ranges over the ten best-scoring trials.

    Each kind is its own study, so trials are ranked within a kind and
    never pooled across them.  A study that ran fewer than ten trials
    reports the spread over what it has, which `n_top` states.
    """
    ranked = (
        pl.DataFrame(trials)
        .sort("mrr", descending=True)
        .group_by("kind", maintain_order=True)
        .head(10)
    )
    return (
        ranked.group_by("kind")
        .agg(
            pl.len().cast(pl.UInt32).alias("n_top"),
            pl.col("alpha").min().alias("alpha_min"),
            pl.col("alpha").max().alias("alpha_max"),
            pl.col("beta").min().alias("beta_min"),
            pl.col("beta").max().alias("beta_max"),
            pl.col("gamma").min().alias("gamma_min"),
            pl.col("gamma").max().alias("gamma_max"),
            (pl.col("mrr").max() - pl.col("mrr").min()).alias("mrr_range"),
        )
        # Kind order follows `QueryKind`, so the spread reads against
        # the Result table row for row.
        .sort(pl.col("kind").cast(pl.Enum(tuple(k.value for k in QueryKind))))
        .pipe(TrialSpread.validate, cast=True)
    )


def _verdict(delta: float, delta_se: float) -> Verdict:
    """Whether *delta* clears twice its own standard error.

    Strict, so a delta exactly twice its error withholds: a tie is not
    evidence.  A negative delta can never clear it.
    """
    return Verdict.RECOMMEND if delta > 2 * delta_se else Verdict.WITHHOLD


# ── Simplex parameterisation ─────────────────────────────────────────────────


def _simplex_from_unit_square(u: float, v: float) -> tuple[float, float, float]:
    """Map `(u, v) ∈ [0, 1]²` to `(alpha, beta, gamma)` on the simplex."""
    alpha = u
    beta = v * (1.0 - u)
    gamma = (1.0 - u) * (1.0 - v)
    return (alpha, beta, gamma)


# ── Markdown rendering ──────────────────────────────────────────────────────


def _render_tuning_report(
    report: dy.DataFrame[TuneReport],
    trials_data: list[TrialRow],
    impact: dy.DataFrame[ImpactComparison],
    dist: pl.DataFrame,
    report_dir: Path | None = None,
) -> str:
    """Render a `TuneReport` frame as a human-readable markdown string.

    All formatting is done with polars expressions; the template
    receives pre-rendered markdown table strings.  Vega-Lite specs
    are embedded as fenced JSON via `trials_json`.  The impact
    table shows per-dimension MRR comparison.  Follows the same
    pattern as `measure._render_report`.
    """
    r = report
    weights_df = r.select("kind", "best_alpha", "best_beta", "best_gamma").rename(
        {"best_alpha": "alpha", "best_beta": "beta", "best_gamma": "gamma"}
    )

    # One row per kind: each is a separate optimisation over its own
    # query set, so a single current/recommended pair would describe
    # one kind and misdescribe the others.
    pct = (
        pl.when(pl.col("score_current") != 0)
        .then(pl.col("delta") / pl.col("score_current") * 100)
        .otherwise(0.0)
        .round(1)
    )
    signed_pct = pl.when(pct >= 0).then(pl.lit("+")).otherwise(pl.lit("")) + pct.cast(pl.String)

    result_df = r.select(
        "kind",
        "metric",
        pl.col("score_current").cast(pl.String).alias("current"),
        pl.col("score_best").cast(pl.String).alias("recommended"),
        (pl.col("delta").cast(pl.String) + pl.lit(" (") + signed_pct + pl.lit("%)")).alias("delta"),
        (pl.lit("± ") + pl.col("delta_se").round(4).cast(pl.String)).alias("std error"),
        pl.col("verdict").cast(pl.String).alias("verdict"),
        pl.col("n_queries").alias("n"),
        pl.col("pool_recall").round(4).alias("pool ceiling"),
    )

    template = resources.files("rbtr_eval.templates").joinpath("tuning.md.j2").read_text()
    convergence_spec = json.loads(
        resources.files("rbtr_eval.templates").joinpath("convergence.vl.json").read_text()
    )
    convergence_spec["data"]["values"] = trials_data

    simplex_spec = json.loads(
        resources.files("rbtr_eval.templates").joinpath("simplex.vl.json").read_text()
    )
    simplex_spec["layer"][1]["data"]["values"] = trials_data

    # Impact table: show rows where any key is __all__ (rollups),
    # formatted for readability.
    impact_display = impact.filter(
        (pl.col("slug") == "__all__")
        | (pl.col("language") == "__all__")
        | (pl.col("provenance") == "__all__")
    )

    if report_dir is not None:
        render_vl_to_png(convergence_spec, report_dir / "convergence.png")
        render_vl_to_png(simplex_spec, report_dir / "simplex.png")

    # Per-kind: sum n_queries across all kinds.
    total_queries = r.select(pl.col("n_queries").sum()).item()

    return minijinja.Environment().render_str(
        template,
        weights_table=md_table(weights_df),
        result_table=md_table(result_df),
        spread_table=md_table(_top_trial_spread(trials_data)),
        impact_table=md_table(impact_display),
        sample_table=md_table(dist),
        toml_snippet=_toml_snippet(report),
        n_trials=r["n_trials"][0],
        n_queries=total_queries,
        elapsed_seconds=round(r["elapsed_seconds"][0]),
    )


# ── Impact analysis ────────────────────────────────────────────────────


def _impact_comparison(
    baseline: dy.DataFrame[DetailedOutcome],
    best: dy.DataFrame[DetailedOutcome],
) -> dy.DataFrame[ImpactComparison]:
    """Build a side-by-side MRR comparison across dimensions.

    Takes per-query rank frames for baseline and best weights,
    aggregates at multiple rollup levels, and pivots to produce
    `(dimension, value, baseline_mrr, best_mrr, delta)` rows.
    """
    combined = pl.concat(
        [
            baseline.with_columns(pl.lit("baseline").alias("label")),
            best.with_columns(pl.lit("best").alias("label")),
        ]
    )

    key_cols = ["slug", "language", "provenance"]
    aggs = search_metric_aggs()

    levels = [
        combined.group_by("label", *key_cols).agg(*aggs),
        (
            combined.group_by("label", "slug")
            .agg(*aggs)
            .with_columns(
                pl.lit("__all__").alias("language"),
                pl.lit("__all__").alias("provenance"),
            )
        ),
        (
            combined.group_by("label", "language")
            .agg(*aggs)
            .with_columns(
                pl.lit("__all__").alias("slug"),
                pl.lit("__all__").alias("provenance"),
            )
        ),
        (
            combined.group_by("label", "provenance")
            .agg(*aggs)
            .with_columns(
                pl.lit("__all__").alias("slug"),
                pl.lit("__all__").alias("language"),
            )
        ),
        (
            combined.group_by("label")
            .agg(*aggs)
            .with_columns(
                pl.lit("__all__").alias("slug"),
                pl.lit("__all__").alias("language"),
                pl.lit("__all__").alias("provenance"),
            )
        ),
    ]

    select_cols = ["label", *key_cols, "mrr", "ndcg_at_10"]
    rollup = pl.concat([level.select(select_cols) for level in levels])

    # Pivot: one row per dimension, baseline_mrr + best_mrr columns.
    mrr_pivot = rollup.pivot(on="label", index=key_cols, values="mrr").rename(
        {"baseline": "baseline_mrr", "best": "best_mrr"}
    )
    ndcg_pivot = (
        rollup.pivot(on="label", index=key_cols, values="ndcg_at_10")
        .rename({"baseline": "baseline_ndcg_at_10", "best": "best_ndcg_at_10"})
        .select(*key_cols, "baseline_ndcg_at_10", "best_ndcg_at_10")
    )
    return (
        mrr_pivot.join(ndcg_pivot, on=key_cols)
        .with_columns(
            (pl.col("best_mrr") - pl.col("baseline_mrr")).round(4).alias("delta"),
            (pl.col("best_ndcg_at_10") - pl.col("baseline_ndcg_at_10"))
            .round(4)
            .alias("delta_ndcg_at_10"),
        )
        .sort(key_cols)
        .pipe(ImpactComparison.validate, cast=True)
    )


# ── TOML config snippet ────────────────────────────────────────────────


_TunedWeights = TypeAdapter(dict[QueryKind, WeightTriple])


def _weights_toml(rows: pl.DataFrame) -> str:
    """Serialise *rows*' tuned weights as a `[search_weights]` block.

    Takes a subset of a `TuneReport` — filtering by verdict drops the
    schema — and reads only `kind` and the three `best_*` columns.
    Empty in, empty out: a kind set with no members has no block.
    """
    selected = rows.select(
        "kind",
        pl.col("best_alpha").alias("alpha"),
        pl.col("best_beta").alias("beta"),
        pl.col("best_gamma").alias("gamma"),
    )
    raw = dict(zip(selected["kind"], selected.drop("kind").to_dicts(), strict=True))
    if not raw:
        return ""
    serialised = _TunedWeights.dump_python(_TunedWeights.validate_python(raw), mode="json")
    return tomli_w.dumps({"search_weights": serialised}).strip()


def _toml_snippet(report: dy.DataFrame[TuneReport]) -> str:
    """Build a TOML config snippet from tuning results.

    A kind whose verdict is `WITHHOLD` is emitted commented out: its
    numbers stay readable, but pasting the snippet whole ships only the
    weights this run can tell apart from the current ones.
    """
    recommended = _weights_toml(report.filter(pl.col("verdict") == Verdict.RECOMMEND))
    withheld = _weights_toml(report.filter(pl.col("verdict") == Verdict.WITHHOLD))
    commented = "\n".join(f"# {line}" if line else "#" for line in withheld.splitlines())
    return "\n".join(part for part in (recommended, commented) if part).strip()


# ── Optuna study runner ──────────────────────────────────────────────────

type StudyResult = tuple[
    tuple[float, float, float],
    float,
    dy.DataFrame[DetailedOutcome],
    list[TrialRow],
]


def _run_study(
    candidates: dy.DataFrame[ScoredCandidate],
    meta: dy.DataFrame[QueryMeta],
    baseline_ranks: dy.DataFrame[DetailedOutcome],
    *,
    kind: str,
    n_trials: int,
    seed: int,
) -> StudyResult:
    """Run one Optuna study and return `(best_weights, best_mrr, best_ranks, trials_data)`.

    Every trial row carries *kind*: the three studies' trials are
    concatenated for reporting, and a spread taken across the pooled
    rows would describe none of them.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    best_mrr = float("-inf")
    best_weights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    best_ranks = baseline_ranks
    trials_data: list[TrialRow] = []

    for trial_idx in range(n_trials):
        trial = study.ask()
        u = trial.suggest_float("u", 0.0, 1.0)
        v = trial.suggest_float("v", 0.0, 1.0)
        weights = _simplex_from_unit_square(u, v)

        trial_ranks = _rescore_and_rank(candidates, meta, weights)
        per_prov = trial_ranks.pipe(_mrr_per_provenance)
        mrr = _mean_mrr(trial_ranks)

        study.tell(trial, mrr)

        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = weights
            best_ranks = trial_ranks

        alpha, beta, gamma = weights
        trial_row: TrialRow = {
            "kind": kind,
            "trial": trial_idx + 1,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "mrr": mrr,
            "best_so_far": best_mrr,
        }
        for prov, prov_mrr in per_prov.iter_rows():
            trial_row[f"mrr_{prov}"] = prov_mrr
        trials_data.append(trial_row)

    return best_weights, best_mrr, best_ranks, trials_data


# ── Entry point ──────────────────────────────────────────────────────────────


class TuneCmd(BaseModel):
    """Bayesian-optimise rbtr's fusion weights against the query set."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    concept_dir: Path = Field(description="Directory holding concept parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    data_dir: Path = Field(description="Directory for the DuckDB index.")
    config_dir: Path = Field(description="Directory for config.")
    log_dir: Path = Field(description="Directory for logs.")
    n_trials: int = Field(50, description="Number of Optuna trials.")
    tune_queries_per_cell: int = Field(
        10, description="Queries per (slug, language, provenance) cell for tuning."
    )
    seed: int = Field(0, description="Deterministic RNG seed for subsampling and the sampler.")
    report: Path = Field(description="Output path for TUNING.md.")

    def cli_cmd(self) -> None:
        all_queries = load_all_queries(self.per_repo_dir, self.concept_dir)
        t0 = time.monotonic()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._run_per_kind(all_queries, t0)

    def _run_per_kind(
        self,
        all_queries: dy.DataFrame[QueryRow],
        t0: float,
    ) -> None:
        tagged = with_query_kind(all_queries)
        kinds = [k.value for k in QueryKind]

        # Pre-compute per-kind subsampled query sets.
        kind_queries: dict[str, dy.DataFrame[QueryRow]] = {}
        for kind in kinds:
            kq = (
                tagged.filter(pl.col("query_kind") == kind)
                .drop("query_kind")
                .pipe(QueryRow.validate, cast=True)
            )
            kind_queries[kind] = subsample(
                kq,
                queries_per_cell=self.tune_queries_per_cell,
                seed=self.seed,
                strat_keys=("slug", "language", "provenance"),
            )

        total_queries = sum(kq.height for kq in kind_queries.values())

        report_rows: list[dict[str, float | str | int | None]] = []
        all_trials_data: list[TrialRow] = []
        all_baseline_ranks: list[dy.DataFrame[DetailedOutcome]] = []
        all_best_ranks: list[dy.DataFrame[DetailedOutcome]] = []

        with (
            daemon_session(self.data_dir, self.config_dir, self.log_dir) as client,
            progress_reporter("tune") as (on_progress,),
        ):
            offset = 0
            for kind in kinds:
                queries = kind_queries[kind]
                n_queries = queries.height
                if n_queries == 0:
                    continue

                candidates, meta = _collect_scored_candidates(
                    client,
                    queries,
                    self.repos_dir,
                    on_progress=on_progress,
                    offset=offset,
                    total=total_queries,
                )
                offset += n_queries

                t = rbtr_config.search_weights[QueryKind(kind)]
                baseline_ranks = _rescore_and_rank(candidates, meta, (t.alpha, t.beta, t.gamma))
                baseline_mrr = _mean_mrr(baseline_ranks)

                best_weights, best_mrr, best_ranks, trials_data = _run_study(
                    candidates,
                    meta,
                    baseline_ranks,
                    kind=kind,
                    n_trials=self.n_trials,
                    seed=self.seed,
                )

                all_baseline_ranks.append(baseline_ranks)
                all_best_ranks.append(best_ranks)
                all_trials_data.extend(trials_data)

                best_alpha, best_beta, best_gamma = best_weights
                delta = best_mrr - baseline_mrr
                delta_se = _paired_delta_se(baseline_ranks, best_ranks)
                report_rows.append(
                    {
                        "kind": kind,
                        "best_alpha": best_alpha,
                        "best_beta": best_beta,
                        "best_gamma": best_gamma,
                        "score_best": best_mrr,
                        "current_alpha": rbtr_config.search_weights[QueryKind(kind)].alpha,
                        "current_beta": rbtr_config.search_weights[QueryKind(kind)].beta,
                        "current_gamma": rbtr_config.search_weights[QueryKind(kind)].gamma,
                        "score_current": baseline_mrr,
                        "delta": delta,
                        "delta_se": delta_se,
                        "verdict": _verdict(delta, delta_se),
                        "pool_recall": _pool_recall(candidates, meta),
                        "metric": "MRR",
                        "n_trials": self.n_trials,
                        "n_queries": n_queries,
                        "elapsed_seconds": 0.0,  # filled below
                    }
                )

        elapsed = time.monotonic() - t0
        for row in report_rows:
            row["elapsed_seconds"] = elapsed

        if not report_rows:
            msg = "no queries; dataset empty?"
            raise SystemExit(msg)

        report = pl.DataFrame(report_rows).pipe(TuneReport.validate, cast=True)

        combined_baseline = pl.concat(all_baseline_ranks).pipe(DetailedOutcome.validate, cast=True)
        combined_best = pl.concat(all_best_ranks).pipe(DetailedOutcome.validate, cast=True)
        impact = _impact_comparison(combined_baseline, combined_best)

        all_sampled = pl.concat(list(kind_queries.values())).pipe(QueryRow.validate, cast=True)
        dist = sample_distribution(
            all_sampled,
            strat_keys=("slug", "language", "provenance"),
        )

        self._write_output(report, all_trials_data, impact, dist)

    def _write_output(
        self,
        report: dy.DataFrame[TuneReport],
        trials_data: list[TrialRow],
        impact: dy.DataFrame[ImpactComparison],
        dist: pl.DataFrame,
    ) -> None:
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(
            _render_tuning_report(
                report,
                trials_data,
                impact,
                dist,
                report_dir=self.report.parent,
            ),
            encoding="utf-8",
        )
