SELECT
  f.id,
  f.scope,
  f.content,
  f.source_session_id,
  f.created_at,
  f.last_confirmed_at,
  f.confirm_count,
  facts_fts.rank AS bm25_score
FROM facts_fts
INNER JOIN facts AS f ON facts_fts.rowid = f.rowid
WHERE
  facts_fts MATCH ?  -- noqa: RF02
  AND f.scope = ?
  AND f.superseded_by IS NULL
ORDER BY facts_fts.rank
LIMIT ?;
