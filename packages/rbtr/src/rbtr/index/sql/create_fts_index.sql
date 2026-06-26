-- Keyed on `id` — the content-addressed chunk hash, which is
-- globally unique (one row per unique content across all repos),
-- so DuckDB FTS's single-column identifier requirement is met
-- directly.  No surrogate key is needed.
PRAGMA create_fts_index(  -- noqa: PRS
  'chunks', 'id', 'name_tokens', 'content_tokens',
  stemmer = 'none', stopwords = 'none',
  ignore = '([^a-z0-9_])+',
  overwrite = 1
)
