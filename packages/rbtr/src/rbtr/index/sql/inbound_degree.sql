SELECT
  e.target_id AS chunk_id,
  count(*) AS degree
FROM edges AS e
INNER JOIN _snapshot_refs AS rr
  ON e.repo_id = rr.repo_id AND e.snapshot_sha = rr.snapshot_sha
WHERE e.target_id IN (SELECT unnest($chunk_ids::text []))
GROUP BY e.target_id
