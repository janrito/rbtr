-- The registry: which projects and checkouts exist, and which file holds
-- each project's data.  It lives in `index.duckdb`; each project's data
-- lives in its own file under `index/`, named by `projects.file`.  See
-- ARCHITECTURE.md "Index files".
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One project per git common directory: a main checkout and its linked
-- worktrees share one, and therefore one data file.
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY,
  common_dir TEXT UNIQUE NOT NULL,
  file TEXT UNIQUE NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS projects_id_seq START 1;

-- One row per registered checkout.  Every data file's rows carry its
-- `id` as `repo_id`; ids come from this sequence, so they are unique
-- across files.  `path` is always canonical (`normalise_repo_path`).
-- `project_id` has no FOREIGN
-- KEY: DuckDB rejects updating a referenced `projects` row's `file`,
-- which switching a project to a new file must do.  The store inserts a
-- project before its first checkout and deletes it with its last.
CREATE TABLE IF NOT EXISTS repos (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  project_id INTEGER NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS repos_id_seq START 1;
