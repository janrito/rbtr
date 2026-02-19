CREATE TABLE IF NOT EXISTS file_snapshots (
  commit_sha TEXT NOT NULL,
  file_path TEXT NOT NULL,
  blob_sha TEXT NOT NULL,
  PRIMARY KEY (commit_sha, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  blob_sha TEXT NOT NULL,
  file_path TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '',
  content TEXT NOT NULL,
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  embedding FLOAT [] DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS edges (
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  commit_sha TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_blob
ON chunks (blob_sha);

CREATE INDEX IF NOT EXISTS idx_snapshots_commit
ON file_snapshots (commit_sha);

CREATE INDEX IF NOT EXISTS idx_edges_commit_source
ON edges (commit_sha, source_id);
