-- ClickHouse dialect sample — MergeTree engine, ORDER BY, special types.

CREATE TABLE events (
    event_date Date,
    user_id UInt64,
    name LowCardinality(String),
    props Array(String),
    value Nullable(Float64)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (user_id, event_date);

CREATE MATERIALIZED VIEW events_mv
ENGINE = SummingMergeTree
ORDER BY user_id
AS SELECT user_id, count() AS c FROM events GROUP BY user_id;
