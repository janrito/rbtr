-- Token statistics for a session, split by compaction status.
-- Returns one row with lifetime totals and active-only totals.
-- Params: session_id.
SELECT
  SUM(COALESCE(input_tokens, 0)) AS total_input_tokens,
  SUM(COALESCE(output_tokens, 0)) AS total_output_tokens,
  SUM(COALESCE(cache_read_tokens, 0)) AS total_cache_read_tokens,
  SUM(COALESCE(cache_write_tokens, 0)) AS total_cache_write_tokens,
  SUM(COALESCE(cost, 0)) AS total_cost,
  SUM(
    CASE
      WHEN compacted_by IS NULL
        THEN COALESCE(input_tokens, 0)
      ELSE 0
    END
  ) AS active_input_tokens,
  SUM(
    CASE
      WHEN compacted_by IS NULL
        THEN COALESCE(output_tokens, 0)
      ELSE 0
    END
  ) AS active_output_tokens,
  SUM(
    CASE
      WHEN compacted_by IS NULL
        THEN COALESCE(cache_read_tokens, 0)
      ELSE 0
    END
  ) AS active_cache_read_tokens,
  SUM(
    CASE
      WHEN compacted_by IS NULL
        THEN COALESCE(cache_write_tokens, 0)
      ELSE 0
    END
  ) AS active_cache_write_tokens,
  SUM(
    CASE
      WHEN compacted_by IS NULL
        THEN COALESCE(cost, 0)
      ELSE 0
    END
  ) AS active_cost,
  COUNT(DISTINCT message_id) AS total_messages,
  COUNT(
    DISTINCT CASE
      WHEN compacted_by IS NULL THEN message_id
    END
  ) AS active_messages,
  COUNT(DISTINCT compacted_by) AS compaction_count
FROM fragments
WHERE
  session_id = ?
  AND fragment_kind IN ('request-message', 'response-message')
