SELECT  -- noqa: ST06
  c.id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding,
  fts.score
FROM (
  SELECT
    *,
    fts_main_chunks.match_bm25(id, ?) AS score
  FROM chunks
) AS fts
INNER JOIN chunks AS c ON fts.id = c.id
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  fs.commit_sha = ?
  AND fts.score IS NOT NULL
ORDER BY fts.score DESC
LIMIT ?
