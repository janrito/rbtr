SELECT
  session_id,
  MAX(session_label) AS session_label,
  MAX(created_at) AS last_active,
  COUNT(*) AS message_count,
  SUM(COALESCE(cost, 0)) AS total_cost
FROM messages
WHERE
  (? IS NULL OR repo_owner = ?)
  AND (? IS NULL OR repo_name = ?)
GROUP BY session_id
ORDER BY last_active DESC
LIMIT ?;
