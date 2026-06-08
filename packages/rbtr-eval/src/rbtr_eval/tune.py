"""`rbtr-eval tune` subcommand.

Bayesian-optimise the rbtr search fusion weights `(alpha,
beta, gamma)` against every per-repo query set, using the
index in the shared home.  Reports best vs current weights
in `data/tuned-params.json`; never edits source.

Uses Optuna's TPESampler on a unit-square reparameterisation
of the simplex (2 free dimensions).  Each trial evaluates
one weight triple against the full (capped) query set and
returns MRR.  Ask-and-tell interface for progress control.
"""

from __future__ import annotations

import json
import time
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
from rbtr.index.classify import QueryKind
from rbtr_eval.agg import search_metric_aggs
from rbtr_eval.charts import render_vl_to_png
from rbtr_eval.formatting import md_table
from rbtr_eval.queries import PROVENANCE_TO_KIND, load_all_queries, sample_distribution, subsample
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import (
    DetailedOutcome,
    ImpactComparison,
    QueryMeta,
    QueryRow,
    ScoredCandidate,
    TuneReport,
)

# ── Provenance → QueryKind mapping ───────────────────────────────────────────


def _with_query_kind(queries: dy.DataFrame[QueryRow]) -> pl.DataFrame:
    """Add a `query_kind` column mapped from `provenance`."""
    return queries.with_columns(
        pl.col("provenance")
        .replace_strict(PROVENANCE_TO_KIND, default=QueryKind.CONCEPT.value)
        .alias("query_kind"),
    )


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
                path=str(repo_path),
                query=query["text"],
                limit=50,
            ),
        )
        idx = i - 1  # 0-based to match with_row_index
        for r in resp.results:
            rows.append(
                {
                    "query_idx": idx,
                    "file_path": r.file_path,
                    "scope": r.scope,
                    "name": r.name,
                    "line_start": r.line_start,
                    "semantic": r.semantic,
                    "lexical": r.lexical,
                    "name_match": r.name_match,
                    "kind_boost": r.kind_boost,
                    "file_penalty": r.file_penalty,
                    "importance": r.importance,
                    "proximity": r.proximity,
                }
            )
        if on_progress is not None:
            on_progress(offset + i, total)

    candidates = pl.DataFrame(rows, schema=ScoredCandidate.to_polars_schema()).pipe(
        ScoredCandidate.validate, cast=True
    )

    meta = (
        queries.with_row_index("query_idx")
        .select(
            "query_idx",
            "slug",
            "language",
            "provenance",
            "file_path",
            "scope",
            "name",
            "line_start",
        )
        .pipe(QueryMeta.validate, cast=True)
    )
    return candidates, meta


# ── Re-score and rank ─────────────────────────────────────────────────────────


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

    return (
        queries.join(
            top.select("query_idx", "file_path", "scope", "name", "line_start", "rank"),
            on=["query_idx", "file_path", "scope", "name", "line_start"],
            how="left",
        )
        .select("slug", "language", "provenance", "rank")
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


def _hmean_mrr(per_provenance: pl.DataFrame) -> float:
    """Harmonic mean of the `mrr` column, floored at 1e-9."""
    return per_provenance.select(
        pl.len().cast(pl.Float64) / (1.0 / pl.col("mrr").clip(lower_bound=1e-9)).sum()
    ).item()


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
    trials_data: list[dict[str, float | int]],
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

    score_current = r["score_current"][0]
    score_best = r["score_best"][0]
    delta = r["delta"][0]
    pct = (delta / score_current * 100) if score_current != 0 else 0.0

    result_df = pl.DataFrame(
        {
            "metric": [r["metric"][0]],
            "current": [score_current],
            "recommended": [score_best],
            "delta": [delta],
        }
    ).with_columns(
        pl.col("current").cast(pl.String).alias("current"),
        pl.col("recommended").cast(pl.String).alias("recommended"),
        (pl.col("delta").cast(pl.String) + pl.lit(f" ({pct:+.1f}%)")).alias("delta"),
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


def _toml_snippet(report: dy.DataFrame[TuneReport]) -> str:
    """Build a TOML config snippet from tuning results."""
    selected = report.select(
        "kind",
        pl.col("best_alpha").alias("alpha"),
        pl.col("best_beta").alias("beta"),
        pl.col("best_gamma").alias("gamma"),
    )
    raw = dict(zip(selected["kind"], selected.drop("kind").to_dicts(), strict=True))
    serialised = _TunedWeights.dump_python(_TunedWeights.validate_python(raw), mode="json")
    return tomli_w.dumps({"search_weights": serialised}).strip()


# ── Optuna study runner ──────────────────────────────────────────────────

type StudyResult = tuple[
    tuple[float, float, float],
    float,
    dy.DataFrame[DetailedOutcome],
    list[dict[str, float | int]],
]


def _run_study(
    candidates: dy.DataFrame[ScoredCandidate],
    meta: dy.DataFrame[QueryMeta],
    baseline_ranks: dy.DataFrame[DetailedOutcome],
    *,
    n_trials: int,
    seed: int,
) -> StudyResult:
    """Run one Optuna study and return `(best_weights, best_mrr, best_ranks, trials_data)`."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    best_mrr = float("-inf")
    best_weights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    best_ranks = baseline_ranks
    trials_data: list[dict[str, float | int]] = []

    for trial_idx in range(n_trials):
        trial = study.ask()
        u = trial.suggest_float("u", 0.0, 1.0)
        v = trial.suggest_float("v", 0.0, 1.0)
        weights = _simplex_from_unit_square(u, v)

        trial_ranks = _rescore_and_rank(candidates, meta, weights)
        per_prov = trial_ranks.pipe(_mrr_per_provenance)
        mrr = per_prov.pipe(_hmean_mrr)

        study.tell(trial, mrr)

        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = weights
            best_ranks = trial_ranks

        alpha, beta, gamma = weights
        trial_row: dict[str, float | int] = {
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
        tagged = _with_query_kind(all_queries)
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
        all_trials_data: list[dict[str, float | int]] = []
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
                baseline_mrr = baseline_ranks.pipe(_mrr_per_provenance).pipe(_hmean_mrr)

                best_weights, best_mrr, best_ranks, trials_data = _run_study(
                    candidates,
                    meta,
                    baseline_ranks,
                    n_trials=self.n_trials,
                    seed=self.seed,
                )

                all_baseline_ranks.append(baseline_ranks)
                all_best_ranks.append(best_ranks)
                all_trials_data.extend(trials_data)

                best_alpha, best_beta, best_gamma = best_weights
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
                        "delta": best_mrr - baseline_mrr,
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
        trials_data: list[dict[str, float | int]],
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
