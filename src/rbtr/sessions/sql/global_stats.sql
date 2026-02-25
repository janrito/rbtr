-- Cross-session aggregate totals.
-- No params.
SELECT
  COUNT(DISTINCT session_id) AS session_count,
  SUM(COALESCE(cost, 0)) AS total_cost,
  SUM(COALESCE(input_tokens, 0)) AS total_input_tokens,
  SUM(COALESCE(output_tokens, 0)) AS total_output_tokens,
  SUM(COALESCE(cache_read_tokens, 0)) AS total_cache_read_tokens,
  SUM(COALESCE(cache_write_tokens, 0)) AS total_cache_write_tokens
FROM fragments
WHERE fragment_kind IN ('request-message', 'response-message')
