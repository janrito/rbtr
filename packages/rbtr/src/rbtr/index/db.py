"""Statements on a cursor over a project's file, shared by the store and its sessions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
import polars as pl

from rbtr.index import load_sql

_CREATE_FTS_INDEX_SQL = load_sql("create_fts_index.sql")
_NEUTRALISE_FTS_IDF_SQL = load_sql("neutralise_fts_idf.sql")


def build_fts(cur: duckdb.DuckDBPyConnection) -> None:
    """(Re)create the BM25 index of *cur*'s file and neutralise IDF."""
    cur.execute(_CREATE_FTS_INDEX_SQL)
    cur.execute(_NEUTRALISE_FTS_IDF_SQL)


@contextmanager
def registered_views(
    cur: duckdb.DuckDBPyConnection, **frames: pl.DataFrame
) -> Iterator[duckdb.DuckDBPyConnection]:
    """*cur* with *frames* registered as temp views for the block.

    Registering on the cursor keeps the views private to it, so threads
    reading at once never see each other's.
    """
    for name, frame in frames.items():
        cur.register(name, frame)
    try:
        yield cur
    finally:
        for name in frames:
            cur.unregister(name)
