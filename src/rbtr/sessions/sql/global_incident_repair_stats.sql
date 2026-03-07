-- History repair incident counts across all sessions, grouped by strategy.
SELECT
  JSON_EXTRACT(data_json, '$.strategy') AS strategy,
  COUNT(*) AS total_count,
  JSON_EXTRACT(data_json, '$.reason') AS reason
FROM fragments
WHERE fragment_kind = 'llm-history-repair'
GROUP BY strategy, reason
ORDER BY total_count DESC
