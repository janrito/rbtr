"""`rbtr-eval profile` subcommand.

Characterises the query dataset *before* any search runs: how
queries distribute across the three axes (`symbol_kind`,
`provenance`, `query_kind`), how provenance maps onto the
request classification, and concrete examples.  Reads only the
query parquets — no daemon, no index, no LLM — so it runs in
seconds and lets you sanity-check the input before the measure
run.

The three axes are independent: `symbol_kind` is the target
chunk, `provenance` is how the query was generated, and
`query_kind` is `classify_query(text)` — the request shape
production routes on.  See the README "Measurement model".
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.index.models import QueryKind
from rbtr_eval.formatting import heading_label, md_table
from rbtr_eval.kinds import EXCLUDED_KINDS
from rbtr_eval.queries import load_all_queries, with_query_kind
from rbtr_eval.schemas import QueryRow, RepoHeader

_EXAMPLES_PER_PROVENANCE = 3


def _counts(frame: pl.DataFrame, dim: str) -> str:
    """Markdown table of query counts per value of *dim*, descending."""
    return md_table(frame.group_by(dim).len().rename({"len": "n"}).sort("n", descending=True))


def _dropped_languages_table(headers: dy.DataFrame[RepoHeader]) -> str:
    """Per-repo languages skipped for having too few measurable chunks."""
    dropped = (
        headers.select("slug", "dropped_languages")
        .filter(pl.col("dropped_languages").list.len() > 0)
        .explode("dropped_languages")
    )
    if dropped.height == 0:
        return "None — every language met the threshold."
    return md_table(
        dropped.unnest("dropped_languages")
        .select(
            pl.format("`{}`", pl.col("slug")).alias("slug"),
            pl.format("`{}`", pl.col("language")).alias("language"),
            pl.col("n_chunks"),
        )
        .sort("slug", "language")
    )


def _repos_table(headers: dy.DataFrame[RepoHeader]) -> str:
    """Per-repo dataset provenance: the indexed sha and sampled sizes."""
    return md_table(
        headers.sort("slug").select(
            pl.format("`{}`", pl.col("slug")).alias("slug"),
            pl.format("`{}`", pl.col("sha").str.slice(0, 12)).alias("sha"),
            pl.col("n_documented").alias("symbols"),
            pl.col("n_queries").alias("sampled queries"),
        )
    )


def _symbol_kind_by_provenance(frame: pl.DataFrame) -> str:
    """`symbol_kind` x `provenance` count cross-tab.

    Shows which target kinds are present and how their queries
    split across generation strategies.
    """
    provenances = sorted(frame["provenance"].unique().to_list())
    cross = (
        frame.group_by("symbol_kind", "provenance")
        .len()
        .pivot(on="provenance", index="symbol_kind", values="len")
        .fill_null(0)
    )
    return md_table(
        cross.select(
            pl.col("symbol_kind").cast(pl.String),
            *provenances,
            pl.sum_horizontal(provenances).alias("total"),
        ).sort("total", descending=True)
    )


def _symbol_kind_by_query_kind(frame: pl.DataFrame) -> str:
    """`symbol_kind` x `query_kind` count cross-tab.

    For each target kind, which request shapes get generated — the
    target axis crossed with the request axis, both independent of
    how the query was generated.
    """
    kinds = [k.value for k in QueryKind]
    cross = (
        frame.group_by("symbol_kind", "query_kind")
        .len()
        .pivot(on="query_kind", index="symbol_kind", values="len")
        .fill_null(0)
    )
    for kind in kinds:
        if kind not in cross.columns:
            cross = cross.with_columns(pl.lit(0).alias(kind))
    return md_table(
        cross.select(
            pl.col("symbol_kind").cast(pl.String),
            *kinds,
            pl.sum_horizontal(kinds).alias("total"),
        ).sort("total", descending=True)
    )


def _classification_table(frame: pl.DataFrame) -> str:
    """`provenance` x `query_kind` cross-tab, row-normalised to %.

    `query_kind` is `classify_query(text)`.  The scatter is the
    evidence that provenance (how a query was generated) is not
    the same axis as query_kind (how the request reads) — e.g.
    `docstring`-provenance text mostly classifies as identifier.
    """
    cross = (
        frame.group_by("provenance", "query_kind")
        .len()
        .pivot(on="query_kind", index="provenance", values="len")
        .fill_null(0)
    )
    for kind in QueryKind:
        if kind.value not in cross.columns:
            cross = cross.with_columns(pl.lit(0).alias(kind.value))

    cross = cross.with_columns(pl.sum_horizontal(k.value for k in QueryKind).alias("total"))
    return md_table(
        cross.sort("provenance").select(
            pl.format("`{}`", pl.col("provenance")).alias("provenance"),
            *(
                pl.format(
                    "{}%",
                    (pl.col(kind.value) * 100 / pl.col("total")).round(1).cast(pl.String),
                ).alias(kind.value)
                for kind in QueryKind
            ),
            pl.col("total").alias("n"),
        )
    )


def _examples(frame: pl.DataFrame, seed: int) -> list[dict[str, str]]:
    """Sampled queries, up to N per provenance, as example records.

    The template renders each query verbatim in a fenced block, which
    preserves multi-line and punctuation-heavy query text.
    """
    sampled = frame.group_by("provenance").map_groups(
        lambda g: g.sample(min(len(g), _EXAMPLES_PER_PROVENANCE), seed=seed, shuffle=False)
    )
    return [
        {
            "provenance": r["provenance"],
            "query_kind": r["query_kind"],
            "symbol_kind": str(r["symbol_kind"]),
            "language": r["language"],
            "label": heading_label(r["name"]),
            "text": r["text"],
        }
        for r in sampled.sort("provenance", "query_kind").iter_rows(named=True)
    ]


def _render_report(
    queries: dy.DataFrame[QueryRow],
    headers: dy.DataFrame[RepoHeader],
    seed: int,
) -> str:
    """Render DATASET.md from the classified query set."""
    classified = with_query_kind(queries)
    languages = classified["language"].n_unique()
    repos = classified["slug"].n_unique()

    template = resources.files("rbtr_eval.templates").joinpath("dataset.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total=classified.height,
        n_repos=repos,
        n_languages=languages,
        repos_table=_repos_table(headers),
        language_table=_counts(classified, "language"),
        symbol_kind_table=_symbol_kind_by_provenance(classified),
        target_shape_table=_symbol_kind_by_query_kind(classified),
        classification_table=_classification_table(classified),
        excluded_kinds=", ".join(
            f"`{k.value}`" for k in sorted(EXCLUDED_KINDS, key=lambda k: k.value)
        ),
        dropped_languages_table=_dropped_languages_table(headers),
        examples=_examples(classified, seed),
    )


class ProfileCmd(BaseModel):
    """Profile the query dataset before measurement."""

    per_repo_dir: Path = Field(description="Directory with per-repo query parquets.")
    concept_dir: Path = Field(description="Directory with concept parquets.")
    headers_dir: Path = Field(description="Directory with per-repo header parquets.")
    report: Path = Field(description="Output path for DATASET.md.")
    seed: int = Field(0, description="Deterministic-sampling seed for examples.")

    def cli_cmd(self) -> None:
        queries = load_all_queries(self.per_repo_dir, self.concept_dir)
        headers = pl.concat(
            [pl.read_parquet(f) for f in sorted(self.headers_dir.glob("*.parquet"))]
        ).pipe(RepoHeader.validate, cast=True)
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(_render_report(queries, headers, self.seed), encoding="utf-8")
