SELECT
  p.session_id,
  MAX(p.session_label) AS session_label,
  MAX(p.created_at) AS last_active,
  COUNT(DISTINCT p.message_id) AS message_count,
  SUM(COALESCE(p.cost, 0)) AS total_cost,
  MAX(p.model_name) AS model_name,
  MAX(p.review_target) AS review_target
FROM fragments AS p
WHERE
  p.fragment_kind IN ('request-message', 'response-message')
  AND (? IS NULL OR p.repo_owner = ?)
  AND (? IS NULL OR p.repo_name = ?)
GROUP BY p.session_id
ORDER BY last_active DESC
LIMIT ?;
