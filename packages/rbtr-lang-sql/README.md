# rbtr-lang-sql

SQL support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[sql]`.

## What it ingests

One chunk per top-level SQL statement (plus one per CTE). SQL has no
classes or functions in the object-oriented sense and no native import
mechanism, so statements map onto rbtr's capture conventions by shape:
structural definitions become classes, routines and executable
statements become functions, and standalone named objects become
variables.

- **Structural definitions** — `CREATE TABLE`, `CREATE VIEW` /
  `MATERIALIZED VIEW`, `CREATE TYPE ... AS ENUM` / composite types.
- **Routines & statements** — `CREATE FUNCTION`, `SELECT`, `INSERT`,
  `UPDATE`, `DELETE`, `WITH ... AS` (CTEs), `ALTER` / `DROP`.
- **Named objects** — `CREATE INDEX` / `SEQUENCE` / `SCHEMA` / `ROLE` /
  `TRIGGER`.

`CREATE PROCEDURE` and `PRAGMA` are not extracted — the generic grammar
has no node for them.

## Chunks produced

`name` is the statement's target object; `scope` is always empty (SQL
has no nesting). A statement with no nameable target (`SELECT 1`, a
`UNION`) is named `<anonymous>`.

```sql
CREATE TABLE users (id INT, name TEXT);   -- class    "users"
CREATE VIEW active AS SELECT * FROM users; -- class    "active"
CREATE TYPE mood AS ENUM ('sad','happy');  -- class    "mood"
CREATE FUNCTION add(a INT) ...             -- function "add"
SELECT id, name FROM users;                -- function "users"
INSERT INTO logs (msg) VALUES ('hi');      -- function "logs"
WITH ranked AS (...) SELECT ...            -- function "ranked" (per CTE)
CREATE INDEX idx_name ON users (name);     -- variable "idx_name"
CREATE SEQUENCE order_id START 1;          -- variable "order_id"
```

## Embedded / injected chunks

None. SQL does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-sql` grammar (one generic grammar for all
dialects). No dependency on other language plugins.
