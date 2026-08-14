"""`rbtr-eval index` subcommand.

Sequential indexer: for every repo in the per-repo dir,
build the index for that repo into the
shared isolation root.  ``rbtr index --no-daemon`` blocks
until each build is done; no polling, no daemon here.

The sequential loop is what makes it safe to share one data
dir: only one embedding model ever loads, and DuckDB only
sees one writer at a time.  DVC runs this whole stage as a
single command so the serialisation is visible to the
operator.

*root* holds data / config / logs for the run; cache (embedding
models) stays shared on the user's platformdirs cache.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.cli.output import human_bytes
from rbtr.index.store import IndexStore
from rbtr_eval.corpus import corpus_refs
from rbtr_eval.formatting import md_table
from rbtr_eval.rbtr_cli import run_rbtr
from rbtr_eval.schemas import (
    EmbeddingRepoRow,
    KindCountRow,
    LanguageCountRow,
    RepoCountRow,
)


def _load_sql(name: str) -> str:
    return resources.files("rbtr_eval.sql").joinpath(name).read_text()


_REPOS_SQL = _load_sql("index_repos.sql")
_KINDS_SQL = _load_sql("index_kinds.sql")
_LANGS_SQL = _load_sql("index_languages.sql")
_TOTALS_SQL = _load_sql("index_totals.sql")
_EMB_TOTALS_SQL = _load_sql("embedding_totals.sql")
_EMB_REPOS_SQL = _load_sql("embedding_repos.sql")


def _repo_counts(store: IndexStore) -> dy.DataFrame[RepoCountRow]:
    """Chunks and edges per repo, over the corpus snapshot only."""
    return (
        store._cursor.execute(_REPOS_SQL)
        .pl()
        .with_columns(pl.col("path").str.split("/").list.last().alias("repo"))
        .select("repo", "chunks", "locations", "edges")
        .pipe(RepoCountRow.validate, cast=True)
    )


def _kind_counts(store: IndexStore) -> dy.DataFrame[KindCountRow]:
    """Chunks and incident edges per chunk kind, over the corpus."""
    return store._cursor.execute(_KINDS_SQL).pl().pipe(KindCountRow.validate, cast=True)


def _language_counts(store: IndexStore) -> dy.DataFrame[LanguageCountRow]:
    """Chunks and incident edges per language, over the corpus."""
    return store._cursor.execute(_LANGS_SQL).pl().pipe(LanguageCountRow.validate, cast=True)


def _totals(store: IndexStore) -> tuple[int, int, int]:
    """Corpus-wide `(chunks, locations, edges)`.

    A chunk shared by several files counts once in `chunks` and once
    per file in `locations`, so the pair says how much of the corpus
    is repeated content.
    """
    row = store._cursor.execute(_TOTALS_SQL).fetchone()
    return (int(row[0]), int(row[1]), int(row[2])) if row else (0, 0, 0)


def _embedding_counts(store: IndexStore) -> dy.DataFrame[EmbeddingRepoRow]:
    """Embedding coverage per repo, over the corpus snapshot only."""
    return (
        store._cursor.execute(_EMB_REPOS_SQL)
        .pl()
        .with_columns(pl.col("path").str.split("/").list.last().alias("repo"))
        .select("repo", "chunks", "embedded", "truncated")
        .pipe(EmbeddingRepoRow.validate, cast=True)
    )


def _index_report(store: IndexStore) -> str:
    """Generate INDEX.md from the index store."""
    total_chunks, total_locations, total_edges = _totals(store)

    template = resources.files("rbtr_eval.templates").joinpath("index.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total_chunks=f"{total_chunks:,}",
        total_locations=f"{total_locations:,}",
        total_edges=f"{total_edges:,}",
        repos_table=md_table(_repo_counts(store)),
        kinds_table=md_table(_kind_counts(store)),
        langs_table=md_table(_language_counts(store)),
    )


def _embedding_report(store: IndexStore) -> str:
    """Generate EMBEDDING.md from the index store."""
    repos = _embedding_counts(store).select("repo", "chunks", "embedded")

    totals = store._cursor.execute(_EMB_TOTALS_SQL).fetchone()
    total_chunks = totals[0] if totals else 0
    with_embedding = totals[1] if totals else 0
    truncated = totals[2] if totals else 0
    coverage = f"{with_embedding / total_chunks * 100:.1f}%" if total_chunks else "N/A"

    template = resources.files("rbtr_eval.templates").joinpath("embedding.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total_chunks=f"{total_chunks:,}",
        with_embedding=f"{with_embedding:,}",
        truncated=f"{truncated:,}",
        coverage=coverage,
        index_size_human=human_bytes(store.disk_size_bytes()),
        repos_table=md_table(repos),
    )


def _sentinel_hash(store: IndexStore, *, embed: bool) -> str:
    """Compute a content-hash for DVC sentinel files.

    ``embed=False`` (chunks-ready): hash of ``(repo_id, snapshot_sha)``
    pairs.  ``embed=True`` (embed-ready): also includes the
    unembedded count per commit so the hash changes when embeddings
    are written.
    """
    h = hashlib.sha256()
    for repo in store.list_repos():
        for sha, _ts in store.list_indexed_snapshots(repo.repo_id):
            h.update(f"{repo.repo_id}:{sha}".encode())
            if embed:
                unembedded = store.count_unembedded(repo.repo_id, sha)
                h.update(f":{unembedded}".encode())
    return h.hexdigest()


class IndexCmd(BaseModel):
    """Build indexes for every per-repo query set."""

    data_dir: Path = Field(description="Directory for the DuckDB index.")
    config_dir: Path = Field(description="Directory for config.")
    log_dir: Path = Field(description="Directory for logs.")
    repos_dir: Path = Field(description="Directory of cloned repos.")
    report: Path | None = Field(
        None,
        description="Optional output path for INDEX.md summary.",
    )
    embed: bool = Field(
        True,
        description="Compute embeddings (disable with --no-embed).",
    )
    sentinel: Path = Field(description="Content-hash sentinel file for DVC change detection.")

    def cli_cmd(self) -> None:
        for d in (self.data_dir, self.config_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        for repo_path in sorted(p for p in self.repos_dir.iterdir() if p.is_dir()):
            repo_path = repo_path.resolve()
            dir_flags = [
                "--data-dir",
                str(self.data_dir),
                "--config-dir",
                str(self.config_dir),
                "--log-dir",
                str(self.log_dir),
            ]
            index_cmd = [
                *dir_flags,
                "index",
                "--no-daemon",
                "--repo-path",
                str(repo_path),
            ]
            if not self.embed:
                index_cmd.append("--no-embed")
            run_rbtr(index_cmd)
            run_rbtr(
                [
                    *dir_flags,
                    "gc",
                    "--keep-head-only",
                    "--repo-path",
                    str(repo_path),
                ]
            )

        store = IndexStore(str(self.data_dir / "index.duckdb"))

        # Every count below joins `indexed_snapshots` unscoped, which is
        # only the corpus while each repo holds HEAD and nothing else.
        corpus_refs(store)

        if self.report is not None:
            self.report.parent.mkdir(parents=True, exist_ok=True)
            report_fn = _embedding_report if self.embed else _index_report
            self.report.write_text(report_fn(store), encoding="utf-8")

        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(_sentinel_hash(store, embed=self.embed), encoding="utf-8")
