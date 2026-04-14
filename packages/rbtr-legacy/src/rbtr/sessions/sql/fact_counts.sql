SELECT
  scope,
  COUNT(*) AS active_count
FROM facts
WHERE superseded_by IS NULL
GROUP BY scope
ORDER BY scope;
