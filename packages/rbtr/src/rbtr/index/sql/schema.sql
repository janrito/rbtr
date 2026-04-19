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
  content TEXT NOT NULL,
  content_tokens TEXT NOT NULL DEFAULT '',
  name_tokens TEXT NOT NULL DEFAULT '',
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  strip_docstrings BOOLEAN NOT NULL DEFAULT FALSE,
  embedding FLOAT [] DEFAULT NULL,
  PRIMARY KEY (repo_id, id)
);

CREATE TABLE IF NOT EXISTS edges (
  repo_id INTEGER NOT NULL DEFAULT 1,
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  commit_sha TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_edges_repo_commit_source
ON edges (repo_id, commit_sha, source_id);
