SELECT
  source_id,
  target_id,
  kind
FROM edges
WHERE
  repo_id = ?
  AND commit_sha = ?
  AND (? IS NULL OR source_id = ?)
  AND (? IS NULL OR target_id = ?)
  AND (? IS NULL OR kind = ?)
