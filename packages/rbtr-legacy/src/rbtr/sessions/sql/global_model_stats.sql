-- Per-model cost and token breakdown across all sessions.
-- No params.
SELECT
  model_name,
  COUNT(DISTINCT session_id) AS session_count,
  SUM(COALESCE(cost, 0)) AS total_cost,
  SUM(COALESCE(input_tokens, 0)) AS total_input_tokens,
  SUM(COALESCE(output_tokens, 0)) AS total_output_tokens
FROM fragments
WHERE
  fragment_kind IN ('request-message', 'response-message')
  AND model_name IS NOT NULL
  AND status = 'complete'
GROUP BY model_name
ORDER BY total_cost DESC
