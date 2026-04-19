"""PyArrow table builders for bulk-loading index data into DuckDB.

DuckDB's `executemany` has ~1 ms/row overhead.  Registering a
PyArrow table as a virtual view and running `INSERT INTO … SELECT`
against it is orders of magnitude faster for large batches.

Each builder converts a list of domain objects (`Chunk`, `Edge`,
or snapshot tuples) into a `pyarrow.Table` whose column names match
the corresponding SQL staging view (`_stg`).

These are pure functions — they never touch DuckDB directly.
"""

from __future__ import annotations

import json

import pyarrow as pa  # type: ignore[import-untyped]  # no stubs available

from rbtr.index.models import Chunk, Edge


def chunks_to_table(chunks: list[Chunk], repo_id: int = 1) -> pa.Table:
    """Convert chunks to a PyArrow table for `upsert_chunks.sql`.

    Builds all columns in a single pass.  The `embedding` column is
    omitted — embeddings are always `NULL` on initial insert and set
    later via `update_embedding`.

    The `content_tokens` and `name_tokens` columns must be
    pre-populated on each `Chunk` via `tokenise_code`
    before calling this function.
    """
    repo_ids: list[int] = []
    ids: list[str] = []
    blob_shas: list[str] = []
    file_paths: list[str] = []
    kinds: list[str] = []
    names: list[str] = []
    scopes: list[str] = []
    contents: list[str] = []
    content_tokens_col: list[str] = []
    name_tokens_col: list[str] = []
    line_starts: list[int] = []
    line_ends: list[int] = []
    metadatas: list[str] = []
    strip_docstrings_col: list[bool] = []

    for c in chunks:
        repo_ids.append(repo_id)
        ids.append(c.id)
        blob_shas.append(c.blob_sha)
        file_paths.append(c.file_path)
        kinds.append(c.kind.value)
        names.append(c.name)
        scopes.append(c.scope)
        contents.append(c.content)
        content_tokens_col.append(c.content_tokens)
        name_tokens_col.append(c.name_tokens)
        line_starts.append(c.line_start)
        line_ends.append(c.line_end)
        metadatas.append(json.dumps(c.metadata))
        strip_docstrings_col.append(c.strip_docstrings)

    return pa.table(
        {
            "repo_id": pa.array(repo_ids, type=pa.int32()),
            "id": pa.array(ids, type=pa.string()),
            "blob_sha": pa.array(blob_shas, type=pa.string()),
            "file_path": pa.array(file_paths, type=pa.string()),
            "kind": pa.array(kinds, type=pa.string()),
            "name": pa.array(names, type=pa.string()),
            "scope": pa.array(scopes, type=pa.string()),
            "content": pa.array(contents, type=pa.string()),
            "content_tokens": pa.array(content_tokens_col, type=pa.string()),
            "name_tokens": pa.array(name_tokens_col, type=pa.string()),
            "line_start": pa.array(line_starts, type=pa.int32()),
            "line_end": pa.array(line_ends, type=pa.int32()),
            "metadata": pa.array(metadatas, type=pa.string()),
            "strip_docstrings": pa.array(strip_docstrings_col, type=pa.bool_()),
        }
    )


def edges_to_table(edges: list[Edge], commit_sha: str, repo_id: int = 1) -> pa.Table:
    """Convert edges to a PyArrow table for `insert_edges.sql`.

    All edges in a batch share the same *commit_sha* and *repo_id*.
    Builds all columns in a single pass.
    """
    repo_ids: list[int] = []
    source_ids: list[str] = []
    target_ids: list[str] = []
    kinds: list[str] = []

    for e in edges:
        repo_ids.append(repo_id)
        source_ids.append(e.source_id)
        target_ids.append(e.target_id)
        kinds.append(e.kind.value)

    return pa.table(
        {
            "repo_id": pa.array(repo_ids, type=pa.int32()),
            "source_id": pa.array(source_ids, type=pa.string()),
            "target_id": pa.array(target_ids, type=pa.string()),
            "kind": pa.array(kinds, type=pa.string()),
            "commit_sha": pa.array([commit_sha] * len(edges), type=pa.string()),
        }
    )


def snapshots_to_table(rows: list[tuple[str, str, str]], repo_id: int = 1) -> pa.Table:
    """Convert snapshot tuples to a PyArrow table for `upsert_snapshots.sql`.

    Each tuple is `(commit_sha, file_path, blob_sha)`.
    Builds all columns in a single pass.
    """
    repo_ids: list[int] = []
    commit_shas: list[str] = []
    file_paths: list[str] = []
    blob_shas: list[str] = []

    for commit_sha, file_path, blob_sha in rows:
        repo_ids.append(repo_id)
        commit_shas.append(commit_sha)
        file_paths.append(file_path)
        blob_shas.append(blob_sha)

    return pa.table(
        {
            "repo_id": pa.array(repo_ids, type=pa.int32()),
            "commit_sha": pa.array(commit_shas, type=pa.string()),
            "file_path": pa.array(file_paths, type=pa.string()),
            "blob_sha": pa.array(blob_shas, type=pa.string()),
        }
    )
