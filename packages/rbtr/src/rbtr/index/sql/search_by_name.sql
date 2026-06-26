-- sqlfluff:templater:placeholder:name:'main'
-- sqlfluff:templater:placeholder:pattern:'%main%'
WITH ranked AS (
  SELECT
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
    CASE
      WHEN c.name = $name THEN 1
      WHEN c.name ILIKE $name THEN 2
      WHEN c.name ILIKE $name || '%' THEN 3
      ELSE 4
    END AS match_tier
  FROM chunks AS c
  INNER JOIN file_snapshots AS fs
    ON
      c.blob_sha = fs.blob_sha
      AND c.file_path = fs.file_path
  INNER JOIN _repo_refs AS rr
    ON fs.repo_id = rr.repo_id AND fs.commit_sha = rr.commit_sha
  WHERE c.name ILIKE $pattern
)

SELECT
  id,
  repo_id,
  blob_sha,
  file_path,
  kind,
  name,
  scope,
  language,
  content,
  line_start,
  line_end,
  metadata,
  has_embedding
FROM ranked  -- noqa: AL04
WHERE match_tier = (SELECT min(match_tier) FROM ranked)  -- noqa: AL04, RF02
ORDER BY file_path, line_start
