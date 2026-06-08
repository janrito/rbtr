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
import shutil
from importlib import resources
from pathlib import Path

import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.index.store import IndexStore
from rbtr_eval.formatting import md_table
from rbtr_eval.rbtr_cli import run_rbtr


def _load_sql(name: str) -> str:
    return resources.files("rbtr_eval.sql").joinpath(name).read_text()


_REPOS_SQL = _load_sql("index_repos.sql")
_KINDS_SQL = _load_sql("index_kinds.sql")
_LANGS_SQL = _load_sql("index_languages.sql")
_TOTALS_SQL = _load_sql("index_totals.sql")
_EMB_TOTALS_SQL = _load_sql("embedding_totals.sql")
_EMB_REPOS_SQL = _load_sql("embedding_repos.sql")


def _index_report(store: IndexStore) -> str:
    """Generate INDEX.md from the index store."""
    cur = store._cursor

    repos = (
        cur.execute(_REPOS_SQL)
        .pl()
        .with_columns(
            pl.col("path").str.split("/").list.last().alias("repo"),
        )
        .select("repo", "chunks", "edges")
    )

    kinds = cur.execute(_KINDS_SQL).pl()
    langs = cur.execute(_LANGS_SQL).pl()

    totals = cur.execute(_TOTALS_SQL).fetchone()
    total_chunks = totals[0] if totals else 0
    total_edges = totals[1] if totals else 0

    template = resources.files("rbtr_eval.templates").joinpath("index.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total_chunks=f"{total_chunks:,}",
        total_edges=f"{total_edges:,}",
        repos_table=md_table(repos),
        kinds_table=md_table(kinds),
        langs_table=md_table(langs),
    )


def _embedding_report(store: IndexStore) -> str:
    """Generate EMBEDDING.md from the index store."""
    cur = store._cursor

    repos = (
        cur.execute(_EMB_REPOS_SQL)
        .pl()
        .with_columns(
            pl.col("path").str.split("/").list.last().alias("repo"),
        )
        .select("repo", "chunks", "embedded")
    )

    totals = cur.execute(_EMB_TOTALS_SQL).fetchone()
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
        repos_table=md_table(repos),
    )


def _install_rbtrignore(repo_path: Path) -> None:
    """Copy a per-repo `.rbtrignore` into the cloned repo if one exists.

    Ignore files are stored in `rbtr_eval/rbtrignore/<slug>` and
    copied into the repo root before indexing.
    """
    slug = repo_path.name
    ignore_dir = resources.files("rbtr_eval.rbtrignore")
    src = ignore_dir.joinpath(slug)
    if src.is_file():
        shutil.copy2(str(src), repo_path / ".rbtrignore")


def _sentinel_hash(store: IndexStore, *, embed: bool) -> str:
    """Compute a content-hash for DVC sentinel files.

    ``embed=False`` (chunks-ready): hash of ``(repo_id, commit_sha)``
    pairs.  ``embed=True`` (embed-ready): also includes the
    unembedded count per commit so the hash changes when embeddings
    are written.
    """
    h = hashlib.sha256()
    for repo_id, _repo_path in store.list_repos():
        for sha, _ts in store.list_indexed_commits(repo_id):
            h.update(f"{repo_id}:{sha}".encode())
            if embed:
                unembedded = store.count_unembedded(repo_id, sha)
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
            _install_rbtrignore(repo_path)
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
                "--path",
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
                    "--path",
                    str(repo_path),
                ]
            )

        store = IndexStore(str(self.data_dir / "index.duckdb"))

        if self.report is not None:
            self.report.parent.mkdir(parents=True, exist_ok=True)
            report_fn = _embedding_report if self.embed else _index_report
            self.report.write_text(report_fn(store), encoding="utf-8")

        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(_sentinel_hash(store, embed=self.embed), encoding="utf-8")
