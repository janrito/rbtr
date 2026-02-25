-- Tool call and failure counts for a session, grouped by tool name.
-- Params: session_id.
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
  session_id = ?
  AND fragment_kind IN ('tool-call', 'retry-prompt')
  AND tool_name IS NOT NULL
GROUP BY tool_name
ORDER BY call_count DESC
