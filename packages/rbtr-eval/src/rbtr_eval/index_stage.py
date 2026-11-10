"""`rbtr-eval index` subcommand.

Sequential indexer: for every repo in the per-repo dir,
build the index for that repo into the
shared isolation root.  ``rbtr watch --no-daemon`` blocks
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
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.cli.output import human_bytes
from rbtr.config import config
from rbtr.domain.models import ChunkKind, EmbeddingState, SnapshotRef
from rbtr.index.index import Index
from rbtr_eval.corpus import Corpus, corpus_refs
from rbtr_eval.formatting import md_table
from rbtr_eval.rbtr_cli import run_rbtr


class RepoCountRow(dy.Schema):
    """One row of INDEX.md's per-repo table.

    Counts are over the repo's corpus snapshot only (see
    `rbtr_eval.corpus`). `chunks` counts content, so a file vendored
    twice contributes one chunk and two `locations`, and two repos
    holding the same file each count it. `locations` is what compares
    against a file count.
    """

    repo = dy.String()
    chunks = dy.UInt64(min=0)
    locations = dy.UInt64(min=0)
    edges = dy.UInt64(min=0)


class KindCountRow(dy.Schema):
    """One row of INDEX.md's per-kind table.

    An edge contributes to `outbound_edges` of its source kind and
    `inbound_edges` of its target kind, so both columns sum to the
    edge total.  Counts are summed over repos, so content two repos
    share counts in each.
    """

    kind = dy.Enum(k.value for k in ChunkKind)
    n = dy.UInt64(min=0)
    outbound_edges = dy.UInt64(min=0)
    inbound_edges = dy.UInt64(min=0)


class LanguageCountRow(dy.Schema):
    """One row of INDEX.md's per-language table.

    `lang` is the plugin's language id, or `(plaintext)` for chunks
    from files no plugin claimed.
    """

    lang = dy.String()
    n = dy.UInt64(min=0)
    outbound_edges = dy.UInt64(min=0)
    inbound_edges = dy.UInt64(min=0)


class EmbeddingRepoRow(dy.Schema):
    """One row of EMBEDDING.md's per-repo table.

    `embedded` counts chunks carrying a vector and `truncated` those
    whose text exceeded the model's context, so
    `embedded <= chunks` holds by construction.
    """

    repo = dy.String()
    chunks = dy.UInt64(min=0)
    embedded = dy.UInt64(min=0)
    truncated = dy.UInt64(min=0)


def _per_ref(corpus: Corpus, frame_of: Callable[[SnapshotRef], pl.DataFrame]) -> pl.DataFrame:
    """*frame_of* each of *corpus*'s refs, stacked; callers sum the stacked rows.

    Each ref's figures are counted within it (a chunk once per ref), as
    the report reads them.
    """
    return pl.concat(
        [frame_of(ref).with_columns(repo_id=pl.lit(ref.repo.repo_id)) for ref in corpus.refs]
    )


def _with_repo_names(index: Index, frame: pl.DataFrame) -> pl.DataFrame:
    """*frame* keyed by `repo_id`, with each repo's directory name, by path."""
    paths = pl.DataFrame(
        [{"repo_id": r.repo_id, "path": r.repo_path} for r in index.repos()],
        schema={"repo_id": frame.schema["repo_id"], "path": pl.String},
    )
    return (
        frame.join(paths, on="repo_id")
        .sort("path")
        .with_columns(pl.col("path").str.split("/").list.last().alias("repo"))
    )


def _incident_edges(index: Index, ref: SnapshotRef, key: str) -> pl.DataFrame:
    """Chunks per *key* at *ref*, with the edges leaving and reaching them."""
    chunks = index.chunks(at=ref).unique("id").select("id", key)
    edges = index.edges(within=[ref])
    ends = [
        edges.select(end)
        .join(chunks, left_on=end, right_on="id")
        .group_by(key)
        .agg(pl.len().alias(name))
        for end, name in (("source_id", "outbound_edges"), ("target_id", "inbound_edges"))
    ]
    return (
        chunks.group_by(key)
        .agg(n=pl.len())
        .join(ends[0], on=key, how="left")
        .join(ends[1], on=key, how="left")
        .fill_null(0)
    )


def _summed_by(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    """*frame*'s counts summed per *key*, largest first."""
    return (
        frame.group_by(key)
        .agg(pl.col("n", "outbound_edges", "inbound_edges").sum())
        .sort(["n", key], descending=[True, False])
    )


def repo_counts(index: Index, corpus: Corpus) -> dy.DataFrame[RepoCountRow]:
    """Chunks, their locations, and edges per repo, over *corpus*."""

    def counts(ref: SnapshotRef) -> pl.DataFrame:
        located = index.chunks(at=ref)
        return pl.DataFrame(
            {
                "chunks": [located["id"].n_unique()],
                "locations": [located.height],
                "edges": [index.edges(within=[ref]).height],
            }
        )

    return (
        _with_repo_names(index, _per_ref(corpus, counts))
        .select("repo", "chunks", "locations", "edges")
        .pipe(RepoCountRow.validate, cast=True)
    )


def kind_counts(index: Index, corpus: Corpus) -> dy.DataFrame[KindCountRow]:
    """Chunks and incident edges per chunk kind, over *corpus*."""
    frame = _per_ref(corpus, lambda ref: _incident_edges(index, ref, "kind"))
    return _summed_by(frame, "kind").pipe(KindCountRow.validate, cast=True)


def language_counts(index: Index, corpus: Corpus) -> dy.DataFrame[LanguageCountRow]:
    """Chunks and incident edges per language, over *corpus*."""
    frame = _per_ref(corpus, lambda ref: _incident_edges(index, ref, "language"))
    return _summed_by(frame.rename({"language": "lang"}), "lang").pipe(
        LanguageCountRow.validate, cast=True
    )


def totals(index: Index, corpus: Corpus) -> tuple[int, int, int]:
    """`(chunks, locations, edges)` over *corpus*.

    A chunk shared by several files counts once in `chunks` and once
    per file in `locations`, so the pair says how much of the corpus
    is repeated content.
    """
    counts = repo_counts(index, corpus)
    return (
        int(counts["chunks"].sum()),
        int(counts["locations"].sum()),
        int(counts["edges"].sum()),
    )


def _embedding_states(index: Index, ref: SnapshotRef) -> pl.DataFrame:
    """At *ref*: its chunks, how many have an embedding, and how many were truncated."""
    state = index.chunks(at=ref).unique("id")["embedding_state"]
    return pl.DataFrame(
        {
            "chunks": [state.len()],
            "embedded": [(state != EmbeddingState.NONE).sum()],
            "truncated": [(state == EmbeddingState.TRUNCATED).sum()],
        }
    )


def embedding_counts(index: Index, corpus: Corpus) -> dy.DataFrame[EmbeddingRepoRow]:
    """Embedding coverage per repo, over *corpus*."""
    return (
        _with_repo_names(index, _per_ref(corpus, lambda ref: _embedding_states(index, ref)))
        .select("repo", "chunks", "embedded", "truncated")
        .pipe(EmbeddingRepoRow.validate, cast=True)
    )


def embedding_totals(index: Index, corpus: Corpus) -> tuple[int, int, int]:
    """`(chunks, embedded, truncated)` over *corpus*."""
    counts = embedding_counts(index, corpus)
    return (
        int(counts["chunks"].sum()),
        int(counts["embedded"].sum()),
        int(counts["truncated"].sum()),
    )


def _index_report(
    totals: tuple[int, int, int],
    repos: dy.DataFrame[RepoCountRow],
    kinds: dy.DataFrame[KindCountRow],
    languages: dy.DataFrame[LanguageCountRow],
) -> str:
    """Render INDEX.md from the counts `totals` and the per-group readers return."""
    total_chunks, total_locations, total_edges = totals

    template = resources.files("rbtr_eval.templates").joinpath("index.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total_chunks=f"{total_chunks:,}",
        total_locations=f"{total_locations:,}",
        total_edges=f"{total_edges:,}",
        repos_table=md_table(repos),
        kinds_table=md_table(kinds),
        langs_table=md_table(languages),
    )


def _embedding_report(
    totals: tuple[int, int, int],
    repos: dy.DataFrame[EmbeddingRepoRow],
    index_bytes: int,
) -> str:
    """Render EMBEDDING.md from `embedding_totals`, `embedding_counts` and the index size."""
    total_chunks, with_embedding, truncated = totals
    coverage = f"{with_embedding / total_chunks * 100:.1f}%" if total_chunks else "N/A"

    template = resources.files("rbtr_eval.templates").joinpath("embedding.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        total_chunks=f"{total_chunks:,}",
        with_embedding=f"{with_embedding:,}",
        truncated=f"{truncated:,}",
        coverage=coverage,
        index_size_human=human_bytes(index_bytes),
        repos_table=md_table(repos.select("repo", "chunks", "embedded")),
    )


def sentinel_hash(index: Index, *, embed: bool) -> str:
    """Compute a content-hash for DVC sentinel files.

    ``embed=False`` (chunks-ready): hash of ``(repo_id, snapshot_sha)``
    pairs.  ``embed=True`` (embed-ready): also includes the
    unembedded count per commit so the hash changes when embeddings
    are written.

    Snapshots are visited in `indexed_snapshots` order and the
    counts are looked up, rather than iterated in the order the counts
    query returns them: DVC compares this value between runs, so a
    reordering would invalidate every downstream stage.
    """
    outstanding = (
        {ref: counts.unembedded for ref, counts in index.chunk_counts_by_snapshot().items()}
        if embed
        else {}
    )
    h = hashlib.sha256()
    for repo in index.repos():
        for indexed in index.indexed_snapshots(repo):
            sha = indexed.snapshot_sha
            h.update(f"{repo.repo_id}:{sha}".encode())
            if embed:
                h.update(f":{outstanding[repo.at(sha)]}".encode())
    return h.hexdigest()


class IndexCmd(BaseModel):
    """Build indexes for every per-repo query set."""

    repos_dir: Path = Field(description="Directory of cloned repos.")
    report: Path | None = Field(
        None,
        description="Optional output path for INDEX.md summary.",
    )
    sentinel: Path = Field(description="Content-hash sentinel file for DVC change detection.")

    def cli_cmd(self) -> None:
        for repo_path in sorted(p for p in self.repos_dir.iterdir() if p.is_dir()):
            repo_path = repo_path.resolve()
            run_rbtr(["watch", "--no-daemon", "--repo-path", str(repo_path)])
            run_rbtr(
                [
                    "gc",
                    "--keep",
                    "head-only",
                    "--repo-path",
                    str(repo_path),
                ]
            )

        index = Index(str(config.db_path))

        corpus = corpus_refs(index)

        if self.report is not None:
            self.report.parent.mkdir(parents=True, exist_ok=True)
            report = (
                _embedding_report(
                    embedding_totals(index, corpus),
                    embedding_counts(index, corpus),
                    index.disk_size_bytes(),
                )
                if config.embed
                else _index_report(
                    totals(index, corpus),
                    repo_counts(index, corpus),
                    kind_counts(index, corpus),
                    language_counts(index, corpus),
                )
            )
            self.report.write_text(report, encoding="utf-8")

        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(sentinel_hash(index, embed=config.embed), encoding="utf-8")
