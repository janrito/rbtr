"""Schemas shared between rbtr-eval stages.

A schema belongs here only when more than one stage reads or writes
it — these are the contracts the stages communicate through, and the
persisted parquet shapes.

A schema a single stage owns lives in that stage's own module. That is
not tidiness: `dvc.yaml` lists this file as a dependency of most
stages, so a column added here re-runs all of them, including the LLM
paraphrase and expansion calls. Keeping a stage's own schemas beside
its code means changing one re-runs one.

Frames are validated at construction via
`pl.DataFrame(rows).pipe(Schema.validate, cast=True)`; the schema is
the one source of truth for the shape.
"""

from __future__ import annotations

import dataframely as dy

from rbtr.domain.models import ChunkKind
from rbtr_eval.kinds import EXCLUDED_KINDS

IDENTITY_COLUMNS: tuple[str, ...] = (
    "file_path",
    "scope",
    "name",
    "line_start",
    "line_end",
    "symbol_kind",
)
"""Identifies one target chunk. Two chunks can start on one line, so
the span needs both ends and the kind."""

MATCH_COLUMNS: tuple[str, ...] = tuple(c for c in IDENTITY_COLUMNS if c != "file_path")
"""`IDENTITY_COLUMNS` without `file_path`, for joining a search result to
the query whose target it is.

A result carries every path its content sits at, so the target's path is
matched by membership against `file_paths` rather than by equality. The
rest of the identity still has to match column for column: derived from
`IDENTITY_COLUMNS` so a column added there cannot be missed here, which
would widen the join and let a sibling chunk score as the target.
"""


class QueryRow(dy.Schema):
    """One row per sampled query, emitted by `extract`.

    The per-repo `<slug>.parquet` file is the
    persisted form of this schema.  `measure` and `tune`
    read those files via `pl.read_parquet` + `QueryRow.validate`.
    `symbol_kind` spans the whole `ChunkKind` domain, so a target frame
    joins one read from the index store; a `check` restricts the values
    to the kinds the eval measures — every kind except `EXCLUDED_KINDS`.

    The key is `IDENTITY_COLUMNS` within a `slug`, plus the
    `provenance` that generated the text: one target yields at most one
    name, one body and one docstring query.
    """

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum(
        (k.value for k in ChunkKind),
        primary_key=True,
        check=lambda kind: ~kind.is_in(tuple(k.value for k in EXCLUDED_KINDS)),
    )
    language = dy.String()
    provenance = dy.String(primary_key=True)
    text = dy.String()


class ExpansionRow(dy.Schema):
    """Pre-generated keywords and variants for a query.

    Keyed like `QueryRow`, so `measure` joins expansions onto queries on
    the target they were generated for.
    """

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum((k.value for k in ChunkKind), primary_key=True)
    provenance = dy.String(primary_key=True)
    query_kind = dy.String()
    keywords = dy.List(dy.String())
    variants = dy.List(dy.String())


class RepoHeader(dy.Schema):
    """One row per indexed repo.  Persisted in the headers directory.

    `sha` is the resolved HEAD SHA at extract time.  `seed` /
    `queries_per_cell` are stage parameters; `n_documented`
    is the total measurable-chunk count; `n_queries` is the
    post-subsample query count.  `dropped_languages` records the
    languages skipped for having fewer than `min_per_language`
    measurable chunks, with their chunk counts.
    """

    slug = dy.String(primary_key=True)
    sha = dy.String()
    seed = dy.UInt32()
    queries_per_cell = dy.UInt32(min=1)
    n_documented = dy.UInt32(min=0)
    n_queries = dy.UInt32(min=0)
    dropped_languages = dy.List(dy.Struct({"language": dy.String(), "n_chunks": dy.UInt32()}))


class QueryMeta(dy.Schema):
    """Query identity columns, indexed by `query_idx`.

    Produced alongside the scored-candidate frame by
    `tune._collect_scored_candidates` and
    `tune_reranker._collect_candidates`.  `query_kind` is the
    request classification from `classify_query(text)` — the same
    axis production routes on.
    """

    query_idx = dy.UInt32()
    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    query_kind = dy.String()
    file_path = dy.String()
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(k.value for k in ChunkKind)
