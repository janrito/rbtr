-- Find the most recent compaction summary for a session.
-- Returns the summary_id referenced by the latest compacted_by marks.
-- Params: session_id.
SELECT
  compacted_by AS summary_id,
  MAX(created_at) AS latest_at
FROM fragments
WHERE
  session_id = ?
  AND compacted_by IS NOT NULL
GROUP BY compacted_by
ORDER BY latest_at DESC
LIMIT 1;
