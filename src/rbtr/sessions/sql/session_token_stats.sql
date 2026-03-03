-- Token statistics for a session, split by compaction status.
-- Returns one row with lifetime totals and active-only totals.
-- Params: session_id (x3 — main query + 2 subqueries).
SELECT
  SUM(COALESCE(f.input_tokens, 0)) AS total_input_tokens,
  SUM(COALESCE(f.output_tokens, 0)) AS total_output_tokens,
  SUM(COALESCE(f.cache_read_tokens, 0)) AS total_cache_read_tokens,
  SUM(COALESCE(f.cache_write_tokens, 0)) AS total_cache_write_tokens,
  SUM(COALESCE(f.cost, 0)) AS total_cost,
  SUM(
    CASE
      WHEN f.compacted_by IS NULL
        THEN COALESCE(f.input_tokens, 0)
      ELSE 0
    END
  ) AS active_input_tokens,
  SUM(
    CASE
      WHEN f.compacted_by IS NULL
        THEN COALESCE(f.output_tokens, 0)
      ELSE 0
    END
  ) AS active_output_tokens,
  SUM(
    CASE
      WHEN f.compacted_by IS NULL
        THEN COALESCE(f.cache_read_tokens, 0)
      ELSE 0
    END
  ) AS active_cache_read_tokens,
  SUM(
    CASE
      WHEN f.compacted_by IS NULL
        THEN COALESCE(f.cache_write_tokens, 0)
      ELSE 0
    END
  ) AS active_cache_write_tokens,
  SUM(
    CASE
      WHEN f.compacted_by IS NULL
        THEN COALESCE(f.cost, 0)
      ELSE 0
    END
  ) AS active_cost,
  -- Responses: ModelResponse envelope rows only.
  COUNT(
    DISTINCT CASE
      WHEN f.fragment_kind = 'response-message' THEN f.message_id
    END
  ) AS total_responses,
  COUNT(
    DISTINCT CASE
      WHEN
        f.fragment_kind = 'response-message'
        AND f.compacted_by IS NULL
        THEN f.message_id
    END
  ) AS active_responses,
  -- A "turn" is a message that contains a user-prompt fragment.
  (
    SELECT COUNT(DISTINCT t.message_id)
    FROM fragments AS t
    WHERE
      t.session_id = ?
      AND t.fragment_kind = 'user-prompt'
      AND t.status = 'complete'
  ) AS total_turns,
  (
    SELECT COUNT(DISTINCT a.message_id)
    FROM fragments AS a
    WHERE
      a.session_id = ?
      AND a.fragment_kind = 'user-prompt'
      AND a.status = 'complete'
      AND a.compacted_by IS NULL
  ) AS active_turns,
  COUNT(DISTINCT f.compacted_by) AS compaction_count
FROM fragments AS f
WHERE
  f.session_id = ?
  AND f.fragment_kind IN ('request-message', 'response-message')
  AND f.status = 'complete'
