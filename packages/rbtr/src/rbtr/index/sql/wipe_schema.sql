-- Drop every object the index schema creates, in place, so a
-- writable open rebuilds a clean schema on the same DB file without
-- ever unlinking it (an unlink would let a second process defeat
-- DuckDB's exclusive lock). Must list every table and sequence in
-- `schema.sql`; the FTS index is dropped separately beforehand via
-- `drop_fts_index.sql`.
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS edges CASCADE;
DROP TABLE IF EXISTS file_snapshots CASCADE;
DROP TABLE IF EXISTS indexed_commits CASCADE;
DROP TABLE IF EXISTS watched_refs CASCADE;
DROP TABLE IF EXISTS repos CASCADE;
DROP TABLE IF EXISTS meta CASCADE;
DROP SEQUENCE IF EXISTS repos_id_seq;
