-- DuckDB dialect sample — LIST/STRUCT types, CTAS, CREATE MACRO.

CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    email VARCHAR NOT NULL,
    tags VARCHAR[],
    profile STRUCT(name VARCHAR, age INTEGER)
);

CREATE TABLE recent AS SELECT * FROM users LIMIT 10;

CREATE MACRO add_one(x) AS x + 1;

CREATE VIEW emails AS SELECT email FROM users;
