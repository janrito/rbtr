---
name: rbtr-data
description: >-
  Data handling conventions for rbtr: polars, pydantic, dataframely,
  serialisation, and statistics. Use when writing or modifying code
  that works with data frames, pydantic models, JSON serialisation,
  or statistical aggregations. Also trigger when you see polars,
  dataframely, or pydantic imports in the file being edited.
user-invocable: false
---

# rbtr data conventions

These rules apply on top of the general `dataframely` and
`polars` skills. Where a rule here contradicts those skills,
this file wins.

## Typed data frames

Every function that takes or returns a polars data frame must
declare the shape with a `dataframely` schema and annotate the
parameter / return as `dy.DataFrame[Schema]` (or
`dy.LazyFrame[Schema]`). Validate at boundaries:

```python
frame.pipe(Schema.validate, cast=True)
```

Module-level `_FOO_SCHEMA: dict[str, pl.DataType]` blocks next
to a schema class are banned — the schema is the one source of
shape. `dict[str, object]` / `dict[str, Any]` as a row
annotation is the same mistake.

## Declarative polars aggregations

Never accumulate rows via parallel `dict[str, list]` +
`.append(...)` plumbing. Iterate native Python inputs, collect
`list[dict]`, and build one frame at the end:

```python
pl.DataFrame(rows).pipe(Schema.validate, cast=True)
```

Aggregations go through polars contexts (`select` /
`with_columns` / `group_by().agg()` / `pipe` / `concat`).
Never iterate with `iter_rows` over a frame to produce another
frame — factor that through a transformation context.

## Don't dot-access column names

`pl.col("slug")` and the plain string `"slug"` as a dict key
are fine. `SchemaName.slug.name` to obtain `"slug"` is noise —
the validator catches drift. This overrides the dataframely
skill's `Schema.column.name` recommendation.

## Reuse the library's enum

When a column has a finite domain and the library already ships
a string enum (e.g. `rbtr.index.models.ChunkKind`), use it
directly. In a `dy.Enum`, feed values via genexp:

```python
dy.Enum(k.value for k in ChunkKind)
```

## No scratch pydantic models

A private `_FooRow(BaseModel)` that exists only to shape data
between two functions in the same file is gunk. If the data is
tabular, the polars schema is the shape; return frames. For two
or three values, return a tuple. Pydantic models earn their keep
only at real boundaries — persisted files, CLI config, network
protocols, daemon messages.

## No hand-rolled statistics

`mean`, `median`, `quantile`, `count`, `sum`, `std`,
percentiles, reciprocal-rank aggregates — all native polars or
SQL expressions. Never write `sum(...) / len(...)` or
`sorted(vs)[k]`. Use `.quantile(0.95)` (polars) or
`quantile_cont(col, 0.95)` (SQL).

## No manual JSON serialisation

Use `df.write_json(path)` (polars), `model.model_dump_json()`
(pydantic), or the library's writer. Never build nested
`dict[str, Any]` trees and pass them to `json.dumps`.

## No serialisation loops

Never serialise a list of models with a loop or comprehension:

```python
# WRONG — serialising in a loop
[m.model_dump(mode="json") for m in items]
json.dumps([m.model_dump(mode="json") for m in items])
```

Use a named `TypeAdapter` defined at module level:

```python
# In models.py
foo_list = TypeAdapter(list[Foo])

# At the call site
foo_list.dump_json(items, indent=2)   # → bytes
foo_list.dump_python(items, mode="json")  # → list[dict]
```

If the output needs fields from an existing model plus computed
extras, define a model for that shape — don't `model_dump` with
`include=` and merge dicts.

## Pass polars frames into DuckDB by registering a view

To use a polars frame inside a DuckDB query — as a join input,
a filter set, or insert/update staging — **register it as a
named view**, run the SQL that references it, then unregister.
Never build a giant `IN (...)` literal, interpolate values into
SQL, or loop one query per row. DuckDB reads a registered polars
frame zero-copy; one query handles any number of rows.

### Why — it collapses N operations into one

The reason to reach for this is **speed**, and the win comes
from turning many round trips into a single set-based one:

- **One scan instead of N.** Scoring chunks against 3 query
  vectors by looping `execute` three times scans the chunks
  table three times. Registering the vectors as `_qvecs` and
  `CROSS JOIN`-ing scores all of them in one scan — benchmarked
  at **7–32× faster for 3 vectors** (see ARCHITECTURE,
  "Multi-vector semantic scan"). The gap widens with more rows.
- **One insert instead of per-row round trips.** Buffering
  chunks and registering them as `_stg` for a single bulk upsert
  beats a `INSERT ... VALUES` per chunk: one statement, one
  plan, one commit's worth of work.
- **The planner sees a join, not a literal.** A registered view
  is a real relation DuckDB can hash-join and reorder. A
  `WHERE id IN (... 5000 literals ...)` is a parse-and-build
  cost on every call and a worse plan; string interpolation also
  reopens injection and quoting bugs that bound parameters and
  views avoid.
- **Zero-copy.** DuckDB reads the polars frame's Arrow buffers
  directly — no serialising the data into SQL text and
  re-parsing it.

So the mental model is: *keep the set in a frame, let DuckDB do
the set operation once.* The looping/`IN`-literal alternatives
pay a per-row or per-call tax this avoids.

The lifecycle is always register → execute → unregister, with
the `unregister` in a `finally` so a failed query can't leak the
view onto the cursor:

```python
self._cursor.register("_repo_refs", repo_refs_frame(refs))
try:
    return (
        self._cursor.execute(_SEARCH_SQL, {"top_k": top_k})
        .pl()
        .pipe(ScoredChunkResultRow.validate, cast=True)
    )
finally:
    self._cursor.unregister("_repo_refs")
```

Rules:

- **Name views with a leading underscore** (`_repo_refs`,
  `_qvecs`, `_stg`) so the SQL clearly marks them as transient
  bind inputs, not real tables. The SQL `JOIN _repo_refs` /
  `FROM _stg` against the registered frame.
- **The registered frame still gets a `dataframely` schema.**
  Build it through a schema'd frame-builder
  (`repo_refs_frame`, `chunks_frame`) so its columns and dtypes
  are declared once and match the table it joins against. A
  width mismatch (e.g. `Int64` vs the table's `Int32`) is a
  silent correctness trap DuckDB will paper over.
- **Encode struct columns to JSON text before registering** when
  the target column is TEXT — DuckDB sees a polars `Struct` as a
  nested type, not a string. `_bulk_insert` does this with
  `pl.col(c).struct.json_encode()`.
- **It is thread-safe by cursor.** `register` / `unregister`
  bind to the cursor they are called on, and rbtr uses one
  thread-local cursor per thread, so concurrent calls on
  different threads register identically-named views without
  colliding. Don't add locks for this.
- **Register multiple views for one query** when it needs
  several inputs — register each, execute, unregister each in
  `finally` (e.g. `_qvecs` + `_repo_refs` in the semantic
  search).

This is the polars↔DuckDB bridge: keep data construction and
validation in polars (schema'd frames), hand the SQL-shaped
operation to DuckDB (joins, upserts, set membership), and let
the registered view carry the frame across without copying or
string-building.

### When to reach for it

Reach for a registered view when **a Python-side set of rows
has to meet table-side data**, and you'd otherwise loop or
inline. Concretely:

- You're about to write `for x in xs: cursor.execute(...)` —
  the loop body is one query per element. Register `xs` and do
  it once.
- You're about to build `WHERE col IN (?, ?, …)` or f-string a
  list of ids/values into SQL. Register the values as a view and
  `JOIN` (or `WHERE col IN (SELECT … FROM _view)`).
- You're inserting/updating more than a couple of rows. Stage
  them in a frame and bulk-upsert from the registered view.
- One query needs to be scoped/filtered by a variable-length set
  computed in Python (the `_repo_refs` snapshots, the `_qvecs`
  query vectors).

Don't bother when:

- It's a **fixed, small** set of scalar parameters — bound
  params (`$top_k`, `$commit_sha`) are simpler and clearer than
  a one-row view.
- The data is **already in a table** — join the table directly;
  don't round-trip it out to a frame and back.
- It's a genuine **one-row, one-shot** operation with no set
  involved.

Heuristic: if the row count is dynamic and lives in a frame,
register it; if it's a handful of known scalars, bind them.

## Data handling: use the library

- **JSON → typed records:** `pydantic.TypeAdapter` or
  `Model.model_validate_json`. For a JSON array, use
  `TypeAdapter(list[T]).validate_json` in one call.
- **Tabular data:** `polars` is the default. `duckdb` when the
  operation is SQL-shaped. Never hand-build `pyarrow.Table` in
  product code.
- **Binary payloads:** use the library that produced them.

If you find yourself writing a loop to validate, or using
`dict[str, object]` / `dict[str, Any]` for structured data,
stop — the library has a one-liner.
