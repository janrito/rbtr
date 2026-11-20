-- sqlfluff:templater:placeholder:literal:'main'
-- sqlfluff:templater:placeholder:name:'main'
-- sqlfluff:templater:placeholder:qualified:false
-- Every chunk the requested name matches, with how well: 1 exact,
-- 2 case-insensitive, 3 prefix, 4 substring.  The request is read as the
-- literal name $literal and, when $qualified, as the final name $name in a
-- place a row of `_qualifiers` admits; the better tier counts.  Only the
-- best tier this file has is returned.
WITH ranked AS (
  SELECT
    c.id,
    fs.repo_id,
    c.blob_sha,
    fs.file_path,
    c.kind,
    c.name,
    c.scope,
    c.language,
    c.file_language,
    c.content,
    c.line_start,
    c.line_end,
    c.metadata,
    c.embedding IS NOT NULL AS has_embedding,
    least(
      CASE
        WHEN c.name = $literal THEN 1
        WHEN c.name ILIKE $literal THEN 2
        WHEN c.name ILIKE $literal || '%' THEN 3
        WHEN c.name ILIKE '%' || $literal || '%' THEN 4
      END,
      CASE
        WHEN NOT $qualified THEN NULL
        WHEN NOT EXISTS (
          SELECT 1
          FROM _qualifiers AS q
          WHERE
            (
              q.file_path IS NULL
              AND (c.scope = q.scope OR ends_with(c.scope, '::' || q.scope))
            )
            OR (
              q.repo_id = fs.repo_id
              AND q.file_path = fs.file_path
              AND c.scope = q.scope
            )
        ) THEN NULL
        WHEN c.name = $name THEN 1
        WHEN c.name ILIKE $name THEN 2
        WHEN c.name ILIKE $name || '%' THEN 3
        WHEN c.name ILIKE '%' || $name || '%' THEN 4
      END
    ) AS match_tier
  FROM chunks AS c
  INNER JOIN file_snapshots AS fs
    ON
      c.blob_sha = fs.blob_sha
      AND c.file_language = fs.detected_language
  INNER JOIN _snapshot_refs AS rr
    ON fs.repo_id = rr.repo_id AND fs.snapshot_sha = rr.snapshot_sha
  WHERE
    c.name ILIKE '%' || $literal || '%'
    OR ($qualified AND c.name ILIKE '%' || $name || '%')
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
  file_language,
  content,
  line_start,
  line_end,
  metadata,
  has_embedding,
  -- Returned so a search over several project files keeps only the best
  -- tier any file found.
  match_tier
FROM ranked  -- noqa: AL04
WHERE match_tier = (SELECT min(match_tier) FROM ranked)  -- noqa: AL04, RF02
ORDER BY file_path, line_start
