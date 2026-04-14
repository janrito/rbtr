-- Failure incident counts for a session, grouped by failure kind.
-- Each row has total count plus recovered/failed sub-counts.
-- Params: session_id.
SELECT
  JSON_EXTRACT(data_json, '$.failure_kind') AS failure_kind,
  COUNT(*) AS total_count,
  SUM(
    CASE
      WHEN JSON_EXTRACT(data_json, '$.outcome') = 'recovered' THEN 1
      ELSE 0
    END
  ) AS recovered_count,
  SUM(
    CASE
      WHEN JSON_EXTRACT(data_json, '$.outcome') = 'failed' THEN 1
      ELSE 0
    END
  ) AS failed_count
FROM fragments
WHERE
  session_id = ?
  AND fragment_kind = 'llm-attempt-failed'
GROUP BY failure_kind
ORDER BY total_count DESC
