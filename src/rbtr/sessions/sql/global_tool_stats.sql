-- Tool call and failure counts across all sessions.
-- No params.
SELECT
  tool_name,
  SUM(
    CASE
      WHEN fragment_kind = 'tool-call' THEN 1
      ELSE 0
    END
  ) AS call_count,
  SUM(
    CASE
      WHEN fragment_kind = 'retry-prompt' THEN 1
      ELSE 0
    END
  ) AS failure_count
FROM fragments
WHERE
  fragment_kind IN ('tool-call', 'retry-prompt')
  AND tool_name IS NOT NULL
  AND status = 'complete'
GROUP BY tool_name
ORDER BY call_count DESC
