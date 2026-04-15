SELECT
  target_id AS chunk_id,
  count(*) AS degree
FROM edges
WHERE
  repo_id = ?
  AND commit_sha = ?
  AND target_id IN (SELECT unnest(?::text []))
GROUP BY target_id
