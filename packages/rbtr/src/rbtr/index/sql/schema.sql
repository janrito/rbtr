CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repos (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS repos_id_seq START 1;

CREATE TABLE IF NOT EXISTS file_snapshots (
  repo_id INTEGER NOT NULL DEFAULT 1,
  commit_sha TEXT NOT NULL,
  file_path TEXT NOT NULL,
  blob_sha TEXT NOT NULL,
  detected_language TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (repo_id, commit_sha, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
  repo_id INTEGER NOT NULL DEFAULT 1,
  id TEXT NOT NULL,
  blob_sha TEXT NOT NULL,
  file_path TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '',
  language TEXT NOT NULL DEFAULT '',
  language_plugin_version INTEGER NOT NULL DEFAULT 1,
  content TEXT NOT NULL,
  content_tokens TEXT NOT NULL DEFAULT '',
  name_tokens TEXT NOT NULL DEFAULT '',
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  -- Variable-length embedding vector; dimension is determined by
  -- the model (see ARCHITECTURE.md "Embedding column").
  embedding FLOAT [] DEFAULT NULL,
  embedding_truncated BOOLEAN NOT NULL DEFAULT FALSE,
  -- Globally-unique surrogate key for DuckDB FTS, which
  -- accepts only a single-column identifier.  Our natural PK
  -- `(repo_id, id)` can collide on `id` alone when multiple
  -- repos share a home, which caused `match_bm25` to raise
  -- "scalar subquery returned more than one row".  VIRTUAL
  -- because DuckDB only supports VIRTUAL generated columns
  -- today; recomputed on every read.  FTS reads it once at
  -- `create_fts_index` time to materialise `docs.name`, so
  -- the per-read cost is irrelevant.
  fts_row_key TEXT GENERATED ALWAYS AS (repo_id || ':' || id),
  PRIMARY KEY (repo_id, id)
);

CREATE TABLE IF NOT EXISTS edges (
  repo_id INTEGER NOT NULL DEFAULT 1,
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  commit_sha TEXT NOT NULL,
  PRIMARY KEY (repo_id, commit_sha, source_id, target_id, kind)
);

CREATE TABLE IF NOT EXISTS indexed_commits (
  repo_id INTEGER NOT NULL,
  commit_sha TEXT NOT NULL,
  indexed_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
  PRIMARY KEY (repo_id, commit_sha)
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo_blob
ON chunks (repo_id, blob_sha);

CREATE INDEX IF NOT EXISTS idx_snapshots_repo_commit
ON file_snapshots (repo_id, commit_sha);
