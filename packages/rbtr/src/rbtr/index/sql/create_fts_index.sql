-- Keyed on `fts_row_key` (generated column = `repo_id || ':' ||
-- id`), not `id` alone.  DuckDB FTS accepts only one identifier
-- column and it must be globally unique.  With multiple repos in
-- a shared home, `id` alone collides -- see the comment on
-- chunks.fts_row_key.
PRAGMA create_fts_index(  -- noqa: PRS
  'chunks', 'fts_row_key', 'name_tokens', 'content_tokens',
  stemmer = 'none', stopwords = 'none',
  ignore = '([^a-z0-9_])+',
  overwrite = 1
)
