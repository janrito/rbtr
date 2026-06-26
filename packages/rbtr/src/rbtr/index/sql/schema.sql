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
  -- Content-addressed store: `id` is a blake2b hash of
  -- (file_path, blob_sha, name, line_start) and is therefore
  -- globally unique and repo-independent.  Identical content in
  -- several repos/worktrees is one row here; per-repo attribution
  -- lives entirely in `file_snapshots` (joined on blob_sha +
  -- file_path).  See ARCHITECTURE.md "Content-addressed chunks".
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
  PRIMARY KEY (id)
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

CREATE TABLE IF NOT EXISTS watched_refs (
  repo_id INTEGER NOT NULL,
  ref TEXT NOT NULL,
  PRIMARY KEY (repo_id, ref)
);

CREATE INDEX IF NOT EXISTS idx_chunks_blob
ON chunks (blob_sha, file_path);

CREATE INDEX IF NOT EXISTS idx_snapshots_repo_commit
ON file_snapshots (repo_id, commit_sha);
