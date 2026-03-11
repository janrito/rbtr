-- Overhead cost statistics for a session (compaction + extraction).
-- Returns one row with totals for each overhead kind.
-- Params: session_id.
SELECT
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-compaction'
        THEN COALESCE(input_tokens, 0)
      ELSE 0
    END
  ) AS compaction_input_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-compaction'
        THEN COALESCE(output_tokens, 0)
      ELSE 0
    END
  ) AS compaction_output_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-compaction'
        THEN COALESCE(cost, 0)
      ELSE 0
    END
  ) AS compaction_cost,
  COUNT(
    CASE
      WHEN fragment_kind = 'overhead-compaction' THEN 1
    END
  ) AS compaction_count,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-extraction'
        THEN COALESCE(input_tokens, 0)
      ELSE 0
    END
  ) AS extraction_input_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-extraction'
        THEN COALESCE(output_tokens, 0)
      ELSE 0
    END
  ) AS extraction_output_tokens,
  SUM(
    CASE
      WHEN fragment_kind = 'overhead-extraction'
        THEN COALESCE(cost, 0)
      ELSE 0
    END
  ) AS extraction_cost,
  COUNT(
    CASE
      WHEN fragment_kind = 'overhead-extraction' THEN 1
    END
  ) AS extraction_count
FROM fragments
WHERE
  session_id = ?
  AND fragment_kind IN ('overhead-compaction', 'overhead-extraction')
  AND status = 'complete'
