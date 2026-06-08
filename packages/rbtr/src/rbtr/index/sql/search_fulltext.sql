-- sqlfluff:templater:placeholder:tokenised_query:'q'
SELECT  -- noqa: ST06
  c.id,
  c.repo_id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.language,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding,
  fts.score
FROM (
  SELECT
    chunks.*,
    fts_main_chunks.match_bm25(chunks.fts_row_key, $tokenised_query)
      AS score
  FROM chunks
  INNER JOIN _repo_refs AS rr ON chunks.repo_id = rr.repo_id
) AS fts
INNER JOIN chunks AS c
  ON fts.repo_id = c.repo_id AND fts.id = c.id
INNER JOIN file_snapshots AS fs
  ON
    c.repo_id = fs.repo_id
    AND c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
INNER JOIN _repo_refs AS rr2
  ON c.repo_id = rr2.repo_id AND fs.commit_sha = rr2.commit_sha
WHERE fts.score IS NOT NULL
ORDER BY fts.score DESC
LIMIT $top_k
