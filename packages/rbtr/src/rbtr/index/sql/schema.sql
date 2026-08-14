CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Maps each registered checkout to the small integer id that every per-repo
-- table below carries.  A `repo_id` with no row here is unreachable: gc, the
-- watcher and forget all find their work by listing repos.  `build_index`
-- inserts the row, which is why no foreign key is needed to keep the two in
-- step.  `path` is always canonical (`normalise_repo_path`), so a linked
-- worktree is a repo of its own.  See ARCHITECTURE.md "Repo registration".
CREATE TABLE IF NOT EXISTS repos (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS repos_id_seq START 1;

-- `repo_id` has no DEFAULT.  Every write sets it explicitly, and a default
-- would let a write that forgot to file its rows under an unrelated repo
-- instead of failing.
CREATE TABLE IF NOT EXISTS file_snapshots (
  repo_id INTEGER NOT NULL,
  snapshot_sha TEXT NOT NULL,
  file_path TEXT NOT NULL,
  blob_sha TEXT NOT NULL,
  detected_language TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (repo_id, snapshot_sha, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
  -- Content-addressed store: `id` is a blake2b hash of
  -- (blob_sha, file_language, kind, name, line_start, line_end) and is
  -- therefore globally unique and location-independent.  Identical
  -- content at several paths, in several repos, or in several
  -- worktrees is one row here; where it lives is entirely
  -- `file_snapshots`, joined on blob_sha + file_language =
  -- detected_language.  See ARCHITECTURE.md "Content-addressed
  -- chunks".
  id TEXT NOT NULL,
  blob_sha TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '',
  -- The chunk's own language; `file_language` is the language the
  -- file was extracted as.  A python fence in a markdown document
  -- has language='python', file_language='markdown'.
  language TEXT NOT NULL DEFAULT '',
  file_language TEXT NOT NULL DEFAULT '',
  extraction_serial INTEGER NOT NULL DEFAULT 1,
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
  -- An edge relates content at a location to content at a location:
  -- chunks are content-addressed, so a duplicated file's import chunk
  -- is one row in `chunks` but imports a different sibling from each
  -- of its paths.  The paths are therefore part of the key.
  repo_id INTEGER NOT NULL,
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  snapshot_sha TEXT NOT NULL,
  source_path TEXT NOT NULL,
  target_path TEXT NOT NULL,
  PRIMARY KEY (
    repo_id, snapshot_sha, source_id, target_id, kind, source_path, target_path
  )
);

CREATE TABLE IF NOT EXISTS indexed_snapshots (
  repo_id INTEGER NOT NULL,
  snapshot_sha TEXT NOT NULL,
  indexed_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
  PRIMARY KEY (repo_id, snapshot_sha)
);

CREATE TABLE IF NOT EXISTS watched_refs (
  repo_id INTEGER NOT NULL,
  ref TEXT NOT NULL,
  PRIMARY KEY (repo_id, ref)
);

CREATE INDEX IF NOT EXISTS idx_chunks_blob
ON chunks (blob_sha, file_language);

CREATE INDEX IF NOT EXISTS idx_snapshots_repo_commit
ON file_snapshots (repo_id, snapshot_sha);
