-- History repair incident counts for a session, grouped by strategy.
-- Params: session_id.
SELECT
  JSON_EXTRACT(data_json, '$.strategy') AS strategy,
  COUNT(*) AS total_count,
  JSON_EXTRACT(data_json, '$.reason') AS reason
FROM fragments
WHERE
  session_id = ?
  AND fragment_kind = 'llm-history-repair'
GROUP BY strategy, reason
ORDER BY total_count DESC
