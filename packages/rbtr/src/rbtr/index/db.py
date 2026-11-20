"""Statements on a cursor over a project's file, shared by the store and its sessions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

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
    reading at once never see each other's.  The views are dropped when
    the block exits.  When the block raises inside a transaction, the
    block's exception propagates and the owner's rollback drops them.
    """
    for name, frame in frames.items():
        cur.register(name, frame)
    try:
        yield cur
    except BaseException:
        # A failed statement aborts the transaction, which then refuses
        # `unregister`; the rollback drops the views instead.
        with suppress(duckdb.TransactionException):
            for name in frames:
                cur.unregister(name)
        raise
    for name in frames:
        cur.unregister(name)
