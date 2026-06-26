-- sqlfluff:templater:placeholder:tokenised_query:'q'
-- Chunks are content-addressed (one row per unique `id` across all
-- repos), so `match_bm25` scores the global corpus keyed on `id`.
-- Repo/commit scoping is applied by the outer snapshot + `_repo_refs`
-- join, which also sources `repo_id` for the result rows.
SELECT  -- noqa: ST06
  c.id,
  fs.repo_id,
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
    chunks.id,
    fts_main_chunks.match_bm25(chunks.id, $tokenised_query) AS score
  FROM chunks
) AS fts
INNER JOIN chunks AS c
  ON fts.id = c.id
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
INNER JOIN _repo_refs AS rr
  ON fs.repo_id = rr.repo_id AND fs.commit_sha = rr.commit_sha
WHERE fts.score IS NOT NULL
ORDER BY fts.score DESC
LIMIT $top_k
