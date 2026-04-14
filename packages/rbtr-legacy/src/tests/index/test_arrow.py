"""Tests for rbtr.index.arrow — PyArrow table builders."""

from __future__ import annotations

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]

from rbtr.index.arrow import chunks_to_table, edges_to_table, snapshots_to_table
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind

# ── Fixtures ─────────────────────────────────────────────────────────

_FUNC_CHUNK = Chunk(
    id="fn_1",
    blob_sha="abc",
    file_path="src/main.py",
    kind=ChunkKind.FUNCTION,
    name="calculate",
    scope="",
    content="def calculate(x): return x * 2",
    line_start=1,
    line_end=1,
    metadata={},
)

_CLASS_CHUNK = Chunk(
    id="cls_1",
    blob_sha="def",
    file_path="src/models.py",
    kind=ChunkKind.CLASS,
    name="User",
    scope="",
    content="class User:\n    pass",
    line_start=5,
    line_end=6,
    metadata={"module": "models"},
)

_IMPORT_CHUNK = Chunk(
    id="imp_1",
    blob_sha="ghi",
    file_path="src/main.py",
    kind=ChunkKind.IMPORT,
    name="import os",
    scope="",
    content="import os",
    line_start=1,
    line_end=1,
    metadata={"module": "os", "names": "os"},
)


# ── chunks_to_table ─────────────────────────────────────────────────


def test_chunks_to_table_schema() -> None:
    """Returned table has the expected column names and types."""
    table = chunks_to_table([_FUNC_CHUNK])
    assert table.num_rows == 1
    assert table.column_names == [
        "id",
        "blob_sha",
        "file_path",
        "kind",
        "name",
        "scope",
        "content",
        "content_tokens",
        "name_tokens",
        "line_start",
        "line_end",
        "metadata",
    ]
    assert table.schema.field("id").type == pa.string()
    assert table.schema.field("line_start").type == pa.int32()
    assert table.schema.field("line_end").type == pa.int32()


def test_chunks_to_table_values() -> None:
    """Column values match the source chunk attributes."""
    table = chunks_to_table([_FUNC_CHUNK])
    assert table.column("id")[0].as_py() == "fn_1"
    assert table.column("blob_sha")[0].as_py() == "abc"
    assert table.column("file_path")[0].as_py() == "src/main.py"
    assert table.column("kind")[0].as_py() == "function"
    assert table.column("name")[0].as_py() == "calculate"
    assert table.column("content")[0].as_py() == "def calculate(x): return x * 2"
    assert table.column("line_start")[0].as_py() == 1
    assert table.column("line_end")[0].as_py() == 1


def test_chunks_to_table_multiple_rows() -> None:
    """Multiple chunks produce one row per chunk, preserving order."""
    table = chunks_to_table([_FUNC_CHUNK, _CLASS_CHUNK, _IMPORT_CHUNK])
    assert table.num_rows == 3
    ids = [v.as_py() for v in table.column("id")]
    assert ids == ["fn_1", "cls_1", "imp_1"]


def test_chunks_to_table_metadata_serialised() -> None:
    """Metadata dict is JSON-serialised to a string column."""
    table = chunks_to_table([_CLASS_CHUNK])
    meta = table.column("metadata")[0].as_py()
    assert isinstance(meta, str)
    assert '"module"' in meta


def test_chunks_to_table_empty_metadata() -> None:
    """Empty metadata produces '{}'."""
    table = chunks_to_table([_FUNC_CHUNK])
    assert table.column("metadata")[0].as_py() == "{}"


def test_chunks_to_table_kind_is_string_value() -> None:
    """ChunkKind enum is stored as its string value, not the enum name."""
    table = chunks_to_table([_CLASS_CHUNK])
    assert table.column("kind")[0].as_py() == "class"


def test_chunks_to_table_empty_list() -> None:
    """Empty input produces a zero-row table with correct schema."""
    table = chunks_to_table([])
    assert table.num_rows == 0
    assert "id" in table.column_names


# ── edges_to_table ───────────────────────────────────────────────────

_CALL_EDGE = Edge(source_id="fn_1", target_id="fn_2", kind=EdgeKind.CALLS)
_IMPORT_EDGE = Edge(source_id="imp_1", target_id="cls_1", kind=EdgeKind.IMPORTS)


def test_edges_to_table_schema() -> None:
    """Returned table has four string columns."""
    table = edges_to_table([_CALL_EDGE], "abc123")
    assert table.num_rows == 1
    assert table.column_names == ["source_id", "target_id", "kind", "commit_sha"]
    for name in table.column_names:
        assert table.schema.field(name).type == pa.string()


def test_edges_to_table_values() -> None:
    """Column values match the source edge plus the commit SHA."""
    table = edges_to_table([_CALL_EDGE], "sha_abc")
    assert table.column("source_id")[0].as_py() == "fn_1"
    assert table.column("target_id")[0].as_py() == "fn_2"
    assert table.column("kind")[0].as_py() == "calls"
    assert table.column("commit_sha")[0].as_py() == "sha_abc"


def test_edges_to_table_commit_sha_broadcast() -> None:
    """Every row gets the same commit_sha."""
    table = edges_to_table([_CALL_EDGE, _IMPORT_EDGE], "head")
    shas = [v.as_py() for v in table.column("commit_sha")]
    assert shas == ["head", "head"]


def test_edges_to_table_multiple_rows() -> None:
    """Multiple edges produce correct row count and order."""
    table = edges_to_table([_CALL_EDGE, _IMPORT_EDGE], "c1")
    assert table.num_rows == 2
    kinds = [v.as_py() for v in table.column("kind")]
    assert kinds == ["calls", "imports"]


def test_edges_to_table_empty_list() -> None:
    """Empty input produces a zero-row table with correct schema."""
    table = edges_to_table([], "c1")
    assert table.num_rows == 0
    assert "source_id" in table.column_names


# ── snapshots_to_table ───────────────────────────────────────────────


def test_snapshots_to_table_schema() -> None:
    """Returned table has three string columns."""
    rows = [("sha1", "file.py", "blob1")]
    table = snapshots_to_table(rows)
    assert table.num_rows == 1
    assert table.column_names == ["commit_sha", "file_path", "blob_sha"]
    for name in table.column_names:
        assert table.schema.field(name).type == pa.string()


def test_snapshots_to_table_values() -> None:
    """Column values match the input tuples."""
    rows = [("sha1", "src/a.py", "blob_a"), ("sha1", "src/b.py", "blob_b")]
    table = snapshots_to_table(rows)
    assert table.num_rows == 2
    paths = [v.as_py() for v in table.column("file_path")]
    assert paths == ["src/a.py", "src/b.py"]


def test_snapshots_to_table_preserves_order() -> None:
    """Row order matches input order."""
    rows = [("c", "z.py", "b3"), ("a", "a.py", "b1"), ("b", "m.py", "b2")]
    table = snapshots_to_table(rows)
    blobs = [v.as_py() for v in table.column("blob_sha")]
    assert blobs == ["b3", "b1", "b2"]


def test_snapshots_to_table_empty_list() -> None:
    """Empty input produces a zero-row table with correct schema."""
    table = snapshots_to_table([])
    assert table.num_rows == 0
    assert "commit_sha" in table.column_names


# ── Roundtrip through DuckDB ─────────────────────────────────────────


def test_chunks_roundtrip_duckdb() -> None:
    """chunks_to_table output can be registered and queried in DuckDB."""

    table = chunks_to_table([_FUNC_CHUNK, _CLASS_CHUNK])
    con = duckdb.connect(":memory:")
    cur = con.cursor()
    cur.register("_stg", table)
    result = cur.execute("SELECT id, kind, line_start FROM _stg ORDER BY id").fetchall()
    cur.unregister("_stg")
    con.close()
    assert result == [("cls_1", "class", 5), ("fn_1", "function", 1)]


def test_edges_roundtrip_duckdb() -> None:
    """edges_to_table output can be registered and queried in DuckDB."""

    table = edges_to_table([_CALL_EDGE, _IMPORT_EDGE], "head_sha")
    con = duckdb.connect(":memory:")
    cur = con.cursor()
    cur.register("_stg", table)
    result = cur.execute("SELECT source_id, kind FROM _stg ORDER BY kind").fetchall()
    cur.unregister("_stg")
    con.close()
    assert result == [("fn_1", "calls"), ("imp_1", "imports")]


def test_snapshots_roundtrip_duckdb() -> None:
    """snapshots_to_table output can be registered and queried in DuckDB."""

    rows = [("sha1", "a.py", "b1"), ("sha1", "b.py", "b2")]
    table = snapshots_to_table(rows)
    con = duckdb.connect(":memory:")
    cur = con.cursor()
    cur.register("_stg", table)
    result = cur.execute("SELECT file_path FROM _stg ORDER BY file_path").fetchall()
    cur.unregister("_stg")
    con.close()
    assert result == [("a.py",), ("b.py",)]
