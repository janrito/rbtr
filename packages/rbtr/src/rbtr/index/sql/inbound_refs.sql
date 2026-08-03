-- Chunks that reference the given targets, one row per referring
-- location.  The referrer's path comes from the edge, which recorded
-- where each end sat when it was inferred: a chunk reachable at several
-- paths imports from each of them, and this reports the one that did.
SELECT
  c.name,
  c.kind,
  c.line_start,
  e.kind AS edge,
  e.source_path AS file_path
FROM edges AS e
INNER JOIN _snapshot_refs AS rr
  ON e.repo_id = rr.repo_id AND e.snapshot_sha = rr.snapshot_sha
INNER JOIN chunks AS c
  ON e.source_id = c.id
WHERE e.target_id IN (SELECT unnest($target_ids::text []))
ORDER BY e.source_path, c.line_start
