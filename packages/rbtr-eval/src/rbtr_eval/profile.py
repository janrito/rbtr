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

from rbtr.domain.models import ChunkKind, QueryKind
from rbtr_eval.formatting import heading_label, md_table
from rbtr_eval.kinds import EXCLUDED_KINDS
from rbtr_eval.queries import ClassifiedQuery, load_all_queries, query_kind
from rbtr_eval.shared_schemas import IDENTITY_COLUMNS, QueryRow, RepoHeader


class QueryCount(dy.Schema):
    """The number of queries per repo, language, target kind, provenance and query kind.

    Every count the report shows is a sum of these.
    """

    slug = dy.String(primary_key=True)
    language = dy.String(primary_key=True)
    symbol_kind = dy.Enum((k.value for k in ChunkKind), primary_key=True)
    provenance = dy.String(primary_key=True)
    query_kind = dy.String(primary_key=True)
    n = dy.UInt32(min=1)


class DroppedLanguage(dy.Schema):
    """A language a repo skipped for having too few measurable chunks."""

    slug = dy.String(primary_key=True)
    language = dy.String(primary_key=True)
    n_chunks = dy.UInt32()


def _query_counts(queries: dy.DataFrame[QueryRow]) -> dy.DataFrame[QueryCount]:
    """How many queries share each repo, language, kind, provenance and query kind."""
    return (
        queries.with_columns(query_kind())
        .group_by("slug", "language", "symbol_kind", "provenance", "query_kind")
        .agg(pl.len().alias("n"))
        .sort("slug", "language", "symbol_kind", "provenance", "query_kind")
        .pipe(QueryCount.validate, cast=True)
    )


def _dropped_languages(headers: dy.DataFrame[RepoHeader]) -> dy.DataFrame[DroppedLanguage]:
    """Every language a repo skipped, with its chunk count."""
    return (
        headers.select("slug", "dropped_languages")
        .filter(pl.col("dropped_languages").list.len() > 0)
        # Polars 2.0 changes this default; the filter above means no empty
        # list reaches here either way.
        .explode("dropped_languages", empty_as_null=True)
        .unnest("dropped_languages")
        .sort("slug", "language")
        .pipe(DroppedLanguage.validate, cast=True)
    )


def _examples(
    queries: dy.DataFrame[QueryRow], *, per_provenance: int, seed: int
) -> dy.DataFrame[ClassifiedQuery]:
    """Up to *per_provenance* sampled queries for each provenance."""
    return (
        queries.group_by("provenance")
        .map_groups(lambda g: g.sample(min(len(g), per_provenance), seed=seed, shuffle=False))
        .with_columns(query_kind())
        .sort("provenance", "query_kind", *IDENTITY_COLUMNS)
        .pipe(ClassifiedQuery.validate, cast=True)
    )


def _render_report(
    counts: dy.DataFrame[QueryCount],
    dropped: dy.DataFrame[DroppedLanguage],
    examples: dy.DataFrame[ClassifiedQuery],
    headers: dy.DataFrame[RepoHeader],
) -> str:
    """Render DATASET.md.

    Each table sums *counts* over the columns it does not show.  The
    template renders each example verbatim in a fenced block, which
    keeps multi-line and punctuation-heavy query text intact.
    """
    kinds = [k.value for k in QueryKind]
    provenances = sorted(counts["provenance"].unique().to_list())

    repos_table = headers.sort("slug").select(
        pl.format("`{}`", pl.col("slug")).alias("slug"),
        pl.format("`{}`", pl.col("sha").str.slice(0, 12)).alias("sha"),
        pl.col("n_documented").alias("symbols"),
        pl.col("n_queries").alias("sampled queries"),
    )
    language_table = (
        counts.group_by("language")
        .agg(pl.col("n").sum())
        .sort("n", "language", descending=[True, False])
    )
    # Which target kinds are present, and how their queries split across
    # generation strategies.
    symbol_kind_table = (
        counts.group_by("symbol_kind", "provenance")
        .agg(pl.col("n").sum())
        .pivot(on="provenance", on_columns=provenances, index="symbol_kind", values="n")
        .fill_null(0)
        .select(
            pl.col("symbol_kind").cast(pl.String),
            *provenances,
            pl.sum_horizontal(provenances).alias("total"),
        )
        .sort("total", "symbol_kind", descending=[True, False])
    )
    # For each target kind, which request shapes get generated: the target
    # axis crossed with the request axis.
    target_shape_table = (
        counts.group_by("symbol_kind", "query_kind")
        .agg(pl.col("n").sum())
        .pivot(on="query_kind", on_columns=kinds, index="symbol_kind", values="n")
        .fill_null(0)
        .select(
            pl.col("symbol_kind").cast(pl.String),
            *kinds,
            pl.sum_horizontal(kinds).alias("total"),
        )
        .sort("total", "symbol_kind", descending=[True, False])
    )
    # Row-normalised to %.  The scatter is the evidence that provenance (how
    # a query was generated) is not the same axis as query_kind (how the
    # request reads): `docstring`-provenance text mostly classifies as
    # identifier.
    classification_table = (
        counts.group_by("provenance", "query_kind")
        .agg(pl.col("n").sum())
        .pivot(on="query_kind", on_columns=kinds, index="provenance", values="n")
        .fill_null(0)
        .with_columns(pl.sum_horizontal(kinds).alias("total"))
        .sort("provenance")
        .select(
            pl.format("`{}`", pl.col("provenance")).alias("provenance"),
            *(
                pl.format(
                    "{}%", (pl.col(kind) * 100 / pl.col("total")).round(1).cast(pl.String)
                ).alias(kind)
                for kind in kinds
            ),
            pl.col("total").alias("n"),
        )
    )
    dropped_table = dropped.select(
        pl.format("`{}`", pl.col("slug")).alias("slug"),
        pl.format("`{}`", pl.col("language")).alias("language"),
        pl.col("n_chunks"),
    )
    example_records = examples.select(
        "provenance",
        "query_kind",
        pl.col("symbol_kind").cast(pl.String),
        "language",
        pl.col("name").map_elements(heading_label, return_dtype=pl.String).alias("label"),
        "text",
    ).to_dicts()

    template = resources.files("rbtr_eval.templates").joinpath("dataset.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total=counts["n"].sum(),
        n_repos=counts["slug"].n_unique(),
        n_languages=counts["language"].n_unique(),
        repos_table=md_table(repos_table),
        language_table=md_table(language_table),
        symbol_kind_table=md_table(symbol_kind_table),
        target_shape_table=md_table(target_shape_table),
        classification_table=md_table(classification_table),
        excluded_kinds=", ".join(
            f"`{k.value}`" for k in sorted(EXCLUDED_KINDS, key=lambda k: k.value)
        ),
        dropped_languages_table=md_table(dropped_table)
        if dropped.height
        else "None — every language met the threshold.",
        examples=example_records,
    )


class ProfileCmd(BaseModel):
    """Profile the query dataset before measurement."""

    per_repo_dir: Path = Field(description="Directory with per-repo query parquets.")
    concept_dir: Path = Field(description="Directory with concept parquets.")
    headers_dir: Path = Field(description="Directory with per-repo header parquets.")
    report: Path = Field(description="Output path for DATASET.md.")
    seed: int = Field(0, description="Deterministic-sampling seed for examples.")
    examples_per_provenance: int = Field(3, description="Example queries shown per provenance.")

    def cli_cmd(self) -> None:
        queries = load_all_queries(self.per_repo_dir, self.concept_dir)
        headers = pl.concat(
            [pl.read_parquet(f) for f in sorted(self.headers_dir.glob("*.parquet"))]
        ).pipe(RepoHeader.validate, cast=True)
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(
            _render_report(
                _query_counts(queries),
                _dropped_languages(headers),
                _examples(queries, per_provenance=self.examples_per_provenance, seed=self.seed),
                headers,
            ),
            encoding="utf-8",
        )
